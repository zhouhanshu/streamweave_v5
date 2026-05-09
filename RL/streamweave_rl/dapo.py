"""DAPO-style group filtering helpers for StreamWeave stepwise rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GroupFilterResult:
    score_key: str
    keep_mask: np.ndarray
    total_groups: int
    valid_groups: int
    total_rows: int
    kept_rows: int
    metrics: dict[str, float]


def resolve_group_metric_key(metric: str | None) -> str:
    """Map common DAPO metric names to StreamWeave non_tensor_batch keys."""
    key = str(metric or "trajectory_score")
    aliases = {
        "score": "trajectory_score",
        "seq_reward": "trajectory_score",
        "seq_final_reward": "success_score",
        "traj/score": "trajectory_score",
        "traj/success": "success_score",
    }
    return aliases.get(key, key)


def compute_group_filter_result(
    non_tensor_batch: dict[str, Any],
    *,
    metric: str | None = "trajectory_score",
    min_std: float = 1e-6,
) -> GroupFilterResult | None:
    """Compute group-level score stats and row mask for DAPO-style filtering.

    A group is valid when at least two rollouts in the group have non-identical
    trajectory-level scores. All rows belonging to invalid groups should be
    dropped before log-prob and actor update.
    """
    score_key = resolve_group_metric_key(metric)
    required = ("group_idx", "traj_idx", score_key)
    if any(key not in non_tensor_batch for key in required):
        return None

    groups = np.asarray(non_tensor_batch["group_idx"], dtype=object).reshape(-1)
    trajs = np.asarray(non_tensor_batch["traj_idx"], dtype=object).reshape(-1)
    scores = _float_array(non_tensor_batch[score_key])
    if not (len(groups) == len(trajs) == len(scores)):
        return None

    traj_scores: dict[tuple[str, str], float] = {}
    for group, traj, score in zip(groups, trajs, scores, strict=True):
        key = (str(group), str(traj))
        traj_scores.setdefault(key, float(score))

    group_to_scores: dict[str, list[float]] = {}
    for (group, _traj), score in traj_scores.items():
        group_to_scores.setdefault(group, []).append(score)

    group_means = []
    group_stds = []
    group_ranges = []
    valid_group_labels: set[str] = set()
    min_std = float(min_std)

    for group, values_list in group_to_scores.items():
        values = np.asarray(values_list, dtype=np.float32)
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        value_range = float(values.max() - values.min()) if values.size else 0.0
        group_means.append(mean)
        group_stds.append(std)
        group_ranges.append(value_range)
        if values.size > 1 and std > min_std:
            valid_group_labels.add(group)

    keep_mask = np.asarray([str(group) in valid_group_labels for group in groups], dtype=bool)
    total_groups = len(group_to_scores)
    valid_groups = len(valid_group_labels)
    total_rows = len(groups)
    kept_rows = int(keep_mask.sum())
    valid_ratio = float(valid_groups / total_groups) if total_groups else 0.0

    metrics: dict[str, float] = {
        "traj/score_mean": _mean(group_means),
        "traj/score_std": _mean(group_stds),
        "traj/score_range": _mean(group_ranges),
        "traj/valid_group_ratio": valid_ratio,
        "traj/total_groups": float(total_groups),
        "traj/valid_groups": float(valid_groups),
        "traj/invalid_groups": float(total_groups - valid_groups),
        "traj/dapo_total_rows": float(total_rows),
        "traj/dapo_kept_rows": float(kept_rows),
        "traj/dapo_kept_row_ratio": float(kept_rows / total_rows) if total_rows else 0.0,
    }
    _add_stats(metrics, "traj/group_score_mean", group_means)
    _add_stats(metrics, "traj/group_score_std", group_stds)
    _add_stats(metrics, "traj/group_score_range", group_ranges)
    _add_stats(metrics, "streamweave/group_score_mean", group_means)
    _add_stats(metrics, "streamweave/group_score_std", group_stds)

    return GroupFilterResult(
        score_key=score_key,
        keep_mask=keep_mask,
        total_groups=total_groups,
        valid_groups=valid_groups,
        total_rows=total_rows,
        kept_rows=kept_rows,
        metrics=metrics,
    )


def select_reward_extra_infos(reward_extra_infos: dict[str, Any], keep_mask: np.ndarray) -> dict[str, Any]:
    """Select reward extra info entries that are row-aligned with the rollout batch."""
    selected: dict[str, Any] = {}
    total_rows = len(keep_mask)
    for key, values in reward_extra_infos.items():
        arr = np.asarray(values, dtype=object)
        if arr.shape[:1] == (total_rows,):
            selected[key] = arr[keep_mask]
        else:
            selected[key] = values
    return selected


def _float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=object).reshape(-1)
    out = np.zeros(arr.shape[0], dtype=np.float32)
    for idx, value in enumerate(arr):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        out[idx] = numeric if np.isfinite(numeric) else 0.0
    return out


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _add_stats(metrics: dict[str, float], prefix: str, values: list[float]) -> None:
    if not values:
        return
    arr = np.asarray(values, dtype=np.float32)
    metrics[f"{prefix}/mean"] = float(arr.mean())
    metrics[f"{prefix}/max"] = float(arr.max())
    metrics[f"{prefix}/min"] = float(arr.min())
    metrics[f"{prefix}/std"] = float(arr.std())
