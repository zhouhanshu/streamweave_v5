"""Custom advantage estimators for StreamWeave stepwise rollouts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import warnings

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_adv_est

from .trace import env_flag, trace_group_allowed, trace_print

GRPPO_STEP_REWARD_KEY = "grppo_step_reward"
GRPPO_ANSWER_REWARD_KEY = "grppo_answer_reward"
GRPPO_ANSWER_CREDIT_KEY = "grppo_answer_credit"
GRPPO_REWARD_KEY = "grppo_reward"
GRPPO_STEP_ADVANTAGE_KEY = "grppo_step_advantage"
GRPPO_ANSWER_ADVANTAGE_KEY = "grppo_answer_advantage"
GRPPO_ADVANTAGE_KEY = "grppo_advantage"
GRPPO_STEP_VALID_KEY = "grppo_step_signal_valid"
GRPPO_ANSWER_VALID_KEY = "grppo_answer_signal_valid"
GRPPO_UPDATE_VALID_KEY = "grppo_update_signal_valid"
GRPO_TRAJSUM_TURN_REWARD_KEY = "grpo_trajsum_turn_reward"
GRPO_TRAJSUM_SCORE_KEY = "grpo_trajsum_score"
GRPO_TRAJSUM_ADVANTAGE_KEY = "grpo_trajsum_advantage"
GRPO_TRAJSUM_VALID_KEY = "grpo_trajsum_signal_valid"
GRPO_TRAJSUM_JUDGE_VALID_KEY = "grpo_trajsum_judge_valid"


def _to_numpy_int64(values: Any, *, factorize: bool = False) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    try:
        return arr.astype(np.int64)
    except (TypeError, ValueError):
        if not factorize:
            raise
        _, inverse = np.unique(arr.astype(str), return_inverse=True)
        return inverse.astype(np.int64, copy=False)


def _to_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == object:
        arr = np.array([0.0 if value is None else value for value in arr], dtype=np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _grppo_judge_valid_mask(non_tensor_batch: dict[str, Any], row_count: int) -> np.ndarray:
    """Rows with judge infrastructure failures must not create GRPPO gradients."""
    if "judge_status" not in non_tensor_batch:
        return np.ones(row_count, dtype=bool)
    statuses = np.asarray(non_tensor_batch["judge_status"], dtype=object).reshape(-1)
    if statuses.shape[0] != row_count:
        return np.ones(row_count, dtype=bool)
    invalid_statuses = {"error", "aborted"}
    return np.asarray([str(status).strip().lower() not in invalid_statuses for status in statuses], dtype=bool)


def _unique_turn_rows(data):
    group = _to_numpy_int64(data.non_tensor_batch["group_idx"], factorize=True)
    traj = _to_numpy_int64(data.non_tensor_batch["traj_idx"], factorize=False)
    turn = _to_numpy_int64(data.non_tensor_batch["turn_idx"], factorize=False)
    key = np.stack([group, traj, turn], axis=1)
    uniq_key, first_idx, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return uniq_key, first_idx.astype(np.int64, copy=False), inverse.astype(np.int64, copy=False)


def _prepare_unique_turn_tensors(data):
    token_level_rewards = data.batch["token_level_rewards"]
    values = data.batch.get("values", torch.zeros_like(token_level_rewards))
    response_mask = data.batch["response_mask"]
    device = token_level_rewards.device

    uniq_key, first_idx, inverse = _unique_turn_rows(data)
    first_idx_t = torch.as_tensor(first_idx, dtype=torch.long, device=device)

    rewards_u = token_level_rewards.index_select(0, first_idx_t)
    values_u = values.index_select(0, first_idx_t)
    mask_u = response_mask.index_select(0, first_idx_t).to(dtype=token_level_rewards.dtype)
    mask_b = mask_u.to(dtype=torch.bool)
    inverse_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
    return uniq_key, rewards_u, values_u, mask_u, mask_b, inverse_t


def _last_valid_positions(mask_b: torch.Tensor) -> torch.Tensor:
    rows, length = mask_b.shape
    valid_counts = mask_b.sum(dim=1)
    arange = torch.arange(length, device=mask_b.device).view(1, length).expand(rows, length)
    last_pos = (mask_b.to(torch.long) * arange).max(dim=1).values
    return torch.where(valid_counts > 0, last_pos, torch.full_like(last_pos, -1))


def _trajectory_rows(uniq_key: np.ndarray) -> list[list[int]]:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row, (group_id, traj_id, _turn_id) in enumerate(uniq_key):
        grouped[(int(group_id), int(traj_id))].append(row)
    return [
        sorted(rows, key=lambda row: int(uniq_key[row, 2]))
        for _key, rows in sorted(grouped.items(), key=lambda item: item[0])
    ]


def _masked_whiten_if_possible(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if float(mask.sum().item()) > 1.0:
        return verl_F.masked_whiten(advantages, mask)
    return advantages


@register_adv_est("streamweave_stepwise_ppo_gae")
def compute_streamweave_stepwise_ppo_gae(
    data,
    gamma,
    lam,
    config=None,
    **kwargs,
):
    del config, kwargs

    with torch.no_grad():
        uniq_key, rewards_u, values_u, mask_u, mask_b, inverse_t = _prepare_unique_turn_tensors(data)

        advantages_u = torch.zeros_like(rewards_u)
        returns_u = torch.zeros_like(rewards_u)

        gamma_val = float(gamma.item() if isinstance(gamma, torch.Tensor) else gamma)
        lam_val = float(lam.item() if isinstance(lam, torch.Tensor) else lam)

        for ordered_rows in _trajectory_rows(uniq_key):
            valid_tokens: list[tuple[int, int]] = []
            for row in ordered_rows:
                positions = torch.nonzero(mask_b[row], as_tuple=False).view(-1).tolist()
                valid_tokens.extend((row, int(pos)) for pos in positions)
            if not valid_tokens:
                continue

            next_value = torch.zeros((), dtype=values_u.dtype, device=values_u.device)
            last_gae = torch.zeros((), dtype=rewards_u.dtype, device=rewards_u.device)
            for row, pos in reversed(valid_tokens):
                reward = rewards_u[row, pos]
                value = values_u[row, pos]
                delta = reward + gamma_val * next_value - value
                last_gae = delta + gamma_val * lam_val * last_gae
                advantages_u[row, pos] = last_gae
                returns_u[row, pos] = last_gae + value
                next_value = value

        advantages_u = _masked_whiten_if_possible(advantages_u, mask_u)

        return advantages_u.index_select(0, inverse_t), returns_u.index_select(0, inverse_t)


@register_adv_est("streamweave_stepwise_bilevel_gae")
def compute_streamweave_stepwise_bilevel_gae(
    data,
    gamma,
    lam,
    config=None,
    **kwargs,
):
    del kwargs

    with torch.no_grad():
        uniq_key, rewards_u, values_u, mask_u, mask_b, inverse_t = _prepare_unique_turn_tensors(data)
        last_pos = _last_valid_positions(mask_b)
        gather_pos = last_pos.clamp(min=0).view(rewards_u.size(0), 1)
        turn_values = values_u.gather(1, gather_pos).squeeze(1)
        turn_values = torch.where(last_pos >= 0, turn_values, torch.zeros_like(turn_values))
        turn_rewards = (rewards_u * mask_u).sum(dim=1)

        gamma_val = float(gamma.item() if isinstance(gamma, torch.Tensor) else gamma)
        lam_val = float(lam.item() if isinstance(lam, torch.Tensor) else lam)
        high_gamma_val = _config_get_float(config, "high_level_gamma", gamma_val)

        high_returns = torch.zeros_like(turn_rewards)
        for ordered_rows in _trajectory_rows(uniq_key):
            next_value = torch.zeros((), dtype=values_u.dtype, device=values_u.device)
            last_gae = torch.zeros((), dtype=rewards_u.dtype, device=rewards_u.device)
            for row in reversed(ordered_rows):
                if int(last_pos[row].item()) < 0:
                    continue
                reward = turn_rewards[row]
                value = turn_values[row]
                delta = reward + high_gamma_val * next_value - value
                last_gae = delta + high_gamma_val * lam_val * last_gae
                high_returns[row] = last_gae + value
                next_value = value

        advantages_u = torch.zeros_like(rewards_u)
        returns_u = torch.zeros_like(rewards_u)
        for row in range(rewards_u.size(0)):
            positions = torch.nonzero(mask_b[row], as_tuple=False).view(-1).tolist()
            if not positions:
                continue
            terminal_pos = int(last_pos[row].item())
            next_value = torch.zeros((), dtype=values_u.dtype, device=values_u.device)
            last_gae = torch.zeros((), dtype=rewards_u.dtype, device=rewards_u.device)
            for pos in reversed(positions):
                terminal_reward = high_returns[row] if int(pos) == terminal_pos else torch.zeros_like(high_returns[row])
                value = values_u[row, pos]
                delta = terminal_reward + gamma_val * next_value - value
                last_gae = delta + gamma_val * lam_val * last_gae
                advantages_u[row, pos] = last_gae
                returns_u[row, pos] = last_gae + value
                next_value = value

        advantages_u = _masked_whiten_if_possible(advantages_u, mask_u)
        return advantages_u.index_select(0, inverse_t), returns_u.index_select(0, inverse_t)


@register_adv_est("streamweave_stepwise_gae")
def compute_streamweave_stepwise_gae(*args, **kwargs):
    warnings.warn(
        "streamweave_stepwise_gae is deprecated; use streamweave_stepwise_ppo_gae "
        "for ordinary trajectory-level PPO or streamweave_stepwise_bilevel_gae for bi-level PPO.",
        UserWarning,
        stacklevel=2,
    )
    return compute_streamweave_stepwise_ppo_gae(*args, **kwargs)


@register_adv_est("streamweave_stepwise_traj_grpo")
def compute_streamweave_stepwise_traj_grpo(
    data,
    gamma=1.0,
    lam=1.0,
    config=None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    **kwargs,
):
    del gamma, lam, config, kwargs
    response_mask = data.batch["response_mask"]
    device = response_mask.device
    if "trajectory_score" not in data.non_tensor_batch:
        raise KeyError("streamweave_stepwise_traj_grpo requires non_tensor_batch['trajectory_score']")

    with torch.no_grad():
        uniq_key, first_idx, inverse = _unique_turn_rows(data)
        first_idx_t = torch.as_tensor(first_idx, dtype=torch.long, device=device)
        mask_u = response_mask.index_select(0, first_idx_t).to(dtype=torch.float32)

        group = uniq_key[:, 0]
        traj = uniq_key[:, 1]
        raw_groups = np.asarray(data.non_tensor_batch["group_idx"], dtype=object).reshape(-1)
        group_labels = {int(group[row]): str(raw_groups[int(first_idx[row])]) for row in range(len(first_idx))}
        scores_full = _to_float_array(data.non_tensor_batch["trajectory_score"])
        scores = scores_full[first_idx]

        traj_score: dict[tuple[int, int], float] = {}
        for row, score in enumerate(scores):
            traj_score[(int(group[row]), int(traj[row]))] = float(score)

        group_to_traj_scores: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (group_id, traj_id), score in traj_score.items():
            group_to_traj_scores[group_id].append((traj_id, score))

        adv_by_traj: dict[tuple[int, int], float] = {}
        for group_id, items in group_to_traj_scores.items():
            values = torch.tensor([score for _, score in items], dtype=torch.float32, device=device)
            if len(items) <= 1:
                normed = torch.zeros_like(values)
            else:
                centered = values - values.mean()
                normed = centered / (values.std(unbiased=True) + epsilon) if norm_adv_by_std_in_grpo else centered
            for (traj_id, _), adv in zip(items, normed.tolist(), strict=True):
                adv_by_traj[(group_id, traj_id)] = float(adv)

        if _trace_grpo_enabled():
            _trace_grpo_groups(
                group_to_traj_scores=group_to_traj_scores,
                adv_by_traj=adv_by_traj,
                group_labels=group_labels,
            )

        advantages_u = torch.zeros_like(mask_u)
        for row in range(len(first_idx)):
            adv = adv_by_traj.get((int(group[row]), int(traj[row])), 0.0)
            advantages_u[row] = float(adv) * mask_u[row]

        inverse_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
        advantages = advantages_u.index_select(0, inverse_t)
        score_u = torch.as_tensor(scores, dtype=torch.float32, device=device).unsqueeze(-1) * mask_u
        returns = score_u.index_select(0, inverse_t)
    return advantages, returns


def compute_trajsum_grpo_values(
    non_tensor_batch: dict[str, Any],
    config: Any = None,
    *,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> dict[str, np.ndarray]:
    """Return row-aligned trajectory-sum GRPO scores and advantages.

    This estimator intentionally sums raw per-turn training rewards. It does
    not use ``grppo_answer_credit`` because that value has already propagated a
    future answer reward to earlier turns; summing it over a trajectory would
    double-count the same answer signal.
    """
    reward_values = compute_grppo_step_reward_values(non_tensor_batch, config)
    step_reward_full = reward_values[GRPPO_STEP_REWARD_KEY]
    answer_reward_full = reward_values[GRPPO_ANSWER_REWARD_KEY]
    row_count = int(step_reward_full.shape[0])

    groups = np.asarray(non_tensor_batch["group_idx"], dtype=object).reshape(-1)
    trajs = np.asarray(non_tensor_batch["traj_idx"], dtype=object).reshape(-1)
    turns = _to_numpy_int64(non_tensor_batch["turn_idx"], factorize=False).reshape(-1)
    if not (len(groups) == len(trajs) == len(turns) == row_count):
        raise ValueError("trajectory-sum GRPO group_idx/traj_idx/turn_idx/reward columns must have the same length")

    step_weight = _config_get_float(
        config,
        "trajsum_step_weight",
        _config_get_float(config, "grppo_step_weight", 1.0),
    )
    answer_weight = _config_get_float(
        config,
        "trajsum_answer_weight",
        _config_get_float(config, "grppo_answer_weight", 1.0),
    )
    norm_by_std = _config_get_bool(config, "trajsum_norm_by_std", bool(norm_adv_by_std_in_grpo))
    min_std = max(_config_get_float(config, "trajsum_min_std", _config_get_filter_min_std(config, 0.0)), 0.0)
    exclude_judge_error = _config_get_bool(config, "trajsum_exclude_judge_error", True)

    unique_keys: list[tuple[str, str, int]] = []
    first_indices: list[int] = []
    inverse = np.zeros(row_count, dtype=np.int64)
    key_to_unique: dict[tuple[str, str, int], int] = {}
    for row in range(row_count):
        key = (str(groups[row]), str(trajs[row]), int(turns[row]))
        unique_row = key_to_unique.get(key)
        if unique_row is None:
            unique_row = len(unique_keys)
            key_to_unique[key] = unique_row
            unique_keys.append(key)
            first_indices.append(row)
        inverse[row] = unique_row

    first = np.asarray(first_indices, dtype=np.int64)
    step_reward_u = step_reward_full[first].astype(np.float32, copy=False)
    answer_reward_u = answer_reward_full[first].astype(np.float32, copy=False)
    turn_reward_u = (step_weight * step_reward_u + answer_weight * answer_reward_u).astype(np.float32, copy=False)
    judge_valid_u = _grppo_judge_valid_mask(non_tensor_batch, row_count)[first]

    traj_to_unique_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    for unique_row, (group_id, traj_id, _turn_id) in enumerate(unique_keys):
        traj_to_unique_rows[(group_id, traj_id)].append(unique_row)

    traj_score: dict[tuple[str, str], float] = {}
    traj_judge_valid: dict[tuple[str, str], bool] = {}
    group_to_traj_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for traj_key, rows in traj_to_unique_rows.items():
        ordered_rows = sorted(rows, key=lambda idx: unique_keys[idx][2])
        valid = bool(np.all(judge_valid_u[ordered_rows])) if exclude_judge_error else True
        score = float(turn_reward_u[ordered_rows].sum()) if valid else 0.0
        traj_score[traj_key] = score
        traj_judge_valid[traj_key] = valid
        if valid:
            group_to_traj_scores[traj_key[0]].append((traj_key[1], score))

    adv_by_traj: dict[tuple[str, str], float] = {}
    signal_valid_by_traj: dict[tuple[str, str], float] = {}
    for group_id, items in group_to_traj_scores.items():
        if len(items) <= 1:
            for traj_id, _score in items:
                signal_valid_by_traj[(group_id, traj_id)] = 0.0
            continue
        values = np.asarray([score for _traj_id, score in items], dtype=np.float32)
        std = float(values.std(ddof=1))
        if std <= min_std:
            for traj_id, _score in items:
                signal_valid_by_traj[(group_id, traj_id)] = 0.0
            continue
        centered = values - float(values.mean())
        if norm_by_std:
            centered = centered / (std + float(epsilon))
        for (traj_id, _score), adv in zip(items, centered.tolist(), strict=True):
            traj_key = (group_id, traj_id)
            adv_by_traj[traj_key] = float(adv)
            signal_valid_by_traj[traj_key] = 1.0

    score_u = np.zeros(len(unique_keys), dtype=np.float32)
    adv_u = np.zeros(len(unique_keys), dtype=np.float32)
    signal_valid_u = np.zeros(len(unique_keys), dtype=np.float32)
    judge_valid_out_u = np.zeros(len(unique_keys), dtype=np.float32)
    for unique_row, (group_id, traj_id, _turn_id) in enumerate(unique_keys):
        traj_key = (group_id, traj_id)
        score_u[unique_row] = float(traj_score.get(traj_key, 0.0))
        adv_u[unique_row] = float(adv_by_traj.get(traj_key, 0.0))
        signal_valid_u[unique_row] = float(signal_valid_by_traj.get(traj_key, 0.0))
        judge_valid_out_u[unique_row] = float(bool(traj_judge_valid.get(traj_key, False)))

    return {
        GRPO_TRAJSUM_TURN_REWARD_KEY: turn_reward_u[inverse].astype(np.float32, copy=False),
        GRPO_TRAJSUM_SCORE_KEY: score_u[inverse].astype(np.float32, copy=False),
        GRPO_TRAJSUM_ADVANTAGE_KEY: adv_u[inverse].astype(np.float32, copy=False),
        GRPO_TRAJSUM_VALID_KEY: signal_valid_u[inverse].astype(np.float32, copy=False),
        GRPO_TRAJSUM_JUDGE_VALID_KEY: judge_valid_out_u[inverse].astype(np.float32, copy=False),
    }


@register_adv_est("streamweave_stepwise_trajsum_grpo")
def compute_streamweave_stepwise_trajsum_grpo(
    data,
    gamma=1.0,
    lam=1.0,
    config=None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    **kwargs,
):
    """Trajectory-sum GRPO over StreamWeave stepwise rollouts.

    Per-turn reward is assembled from raw ``grppo_step_reward`` and raw
    ``grppo_answer_reward``. The trajectory sum is normalized inside each
    ``group_idx`` cohort and then broadcast to every response token in every
    turn of the trajectory.
    """
    del gamma, lam, kwargs
    response_mask = data.batch["response_mask"]
    device = response_mask.device

    with torch.no_grad():
        uniq_key, first_idx, inverse = _unique_turn_rows(data)
        first_idx_t = torch.as_tensor(first_idx, dtype=torch.long, device=device)
        mask_u = response_mask.index_select(0, first_idx_t).to(dtype=torch.float32)

        values = compute_trajsum_grpo_values(
            data.non_tensor_batch,
            config,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        for key, value in values.items():
            data.non_tensor_batch[key] = np.asarray(value, dtype=np.float32)

        score_full = _to_float_array(values[GRPO_TRAJSUM_SCORE_KEY])
        adv_full = _to_float_array(values[GRPO_TRAJSUM_ADVANTAGE_KEY])
        score_u = score_full[first_idx]
        adv_u = adv_full[first_idx]

        if _trace_trajsum_grpo_enabled():
            group = uniq_key[:, 0]
            traj = uniq_key[:, 1]
            raw_groups = np.asarray(data.non_tensor_batch["group_idx"], dtype=object).reshape(-1)
            group_labels = {int(group[row]): str(raw_groups[int(first_idx[row])]) for row in range(len(first_idx))}
            group_to_traj_scores: dict[int, list[tuple[int, float]]] = defaultdict(list)
            adv_by_traj: dict[tuple[int, int], float] = {}
            for row, score in enumerate(score_u):
                key = (int(group[row]), int(traj[row]))
                if key not in adv_by_traj:
                    group_to_traj_scores[key[0]].append((key[1], float(score)))
                    adv_by_traj[key] = float(adv_u[row])
            _trace_grpo_groups(
                group_to_traj_scores=group_to_traj_scores,
                adv_by_traj=adv_by_traj,
                group_labels=group_labels,
            )

        advantages_u = torch.zeros_like(mask_u)
        for row, adv in enumerate(adv_u.tolist()):
            advantages_u[row] = float(adv) * mask_u[row]

        inverse_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
        advantages = advantages_u.index_select(0, inverse_t)
        returns_u = torch.as_tensor(score_u, dtype=torch.float32, device=device).unsqueeze(-1) * mask_u
        returns = returns_u.index_select(0, inverse_t)
    return advantages, returns


def compute_grppo_step_reward_values(non_tensor_batch: dict[str, Any], config: Any = None) -> dict[str, np.ndarray]:
    """Return row-aligned GRPPO reward components.

    Answer credit is computed on full trajectories before any row-level DAPO
    filtering. If a batch has duplicate rows for the same turn, the credit is
    computed once for that unique ``(group, traj, turn)`` and copied back.
    """
    required = ("group_idx", "traj_idx", "turn_idx")
    missing = [key for key in required if key not in non_tensor_batch]
    if missing:
        raise KeyError(f"GRPPO requires non_tensor_batch keys {missing}")

    if GRPPO_STEP_REWARD_KEY in non_tensor_batch:
        step_reward_full = _to_float_array(non_tensor_batch[GRPPO_STEP_REWARD_KEY])
    elif all(
        key in non_tensor_batch
        for key in (
            "grppo_delta_groundedness",
            "grppo_anchor_keyframe",
            "grppo_semantic_alignment",
            "grppo_state_groundedness",
        )
    ):
        step_reward_full = (
            _to_float_array(non_tensor_batch["grppo_delta_groundedness"])
            + _to_float_array(non_tensor_batch["grppo_anchor_keyframe"])
            + _to_float_array(non_tensor_batch["grppo_semantic_alignment"])
            + _to_float_array(non_tensor_batch["grppo_state_groundedness"])
        ) / 2.0
    elif all(
        key in non_tensor_batch
        for key in (
            "grppo_delta_groundedness",
            "grppo_note_keyframe",
            "grppo_semantic_alignment",
            "grppo_state_groundedness",
        )
    ):
        step_reward_full = (
            _to_float_array(non_tensor_batch["grppo_delta_groundedness"])
            + _to_float_array(non_tensor_batch["grppo_note_keyframe"])
            + _to_float_array(non_tensor_batch["grppo_semantic_alignment"])
            + _to_float_array(non_tensor_batch["grppo_state_groundedness"])
        ) / 2.0
    elif "step_score" in non_tensor_batch:
        step_reward_full = _to_float_array(non_tensor_batch["step_score"])
    else:
        raise KeyError(
            "GRPPO requires grppo_step_reward, the four GRPPO judge dimensions, "
            "or step_score as a fallback"
        )

    row_count = int(step_reward_full.shape[0])
    answer_reward_full = (
        _to_float_array(non_tensor_batch[GRPPO_ANSWER_REWARD_KEY])
        if GRPPO_ANSWER_REWARD_KEY in non_tensor_batch
        else np.zeros(row_count, dtype=np.float32)
    )
    if answer_reward_full.shape[0] != row_count:
        raise ValueError(
            f"{GRPPO_ANSWER_REWARD_KEY} length {answer_reward_full.shape[0]} "
            f"does not match {GRPPO_STEP_REWARD_KEY} length {row_count}"
        )

    groups = np.asarray(non_tensor_batch["group_idx"], dtype=object).reshape(-1)
    trajs = np.asarray(non_tensor_batch["traj_idx"], dtype=object).reshape(-1)
    turns = _to_numpy_int64(non_tensor_batch["turn_idx"], factorize=False).reshape(-1)
    if not (len(groups) == len(trajs) == len(turns) == row_count):
        raise ValueError("GRPPO group_idx/traj_idx/turn_idx/reward columns must have the same length")

    beta = max(_config_get_float(config, "grppo_answer_decay", 0.7), 0.0)
    step_weight = _config_get_float(config, "grppo_step_weight", 1.0)
    answer_weight = _config_get_float(config, "grppo_answer_weight", 1.0)

    unique_keys: list[tuple[str, str, int]] = []
    first_indices: list[int] = []
    inverse = np.zeros(row_count, dtype=np.int64)
    key_to_unique: dict[tuple[str, str, int], int] = {}
    for row in range(row_count):
        key = (str(groups[row]), str(trajs[row]), int(turns[row]))
        unique_row = key_to_unique.get(key)
        if unique_row is None:
            unique_row = len(unique_keys)
            key_to_unique[key] = unique_row
            unique_keys.append(key)
            first_indices.append(row)
        inverse[row] = unique_row

    first = np.asarray(first_indices, dtype=np.int64)
    step_reward_u = step_reward_full[first].astype(np.float32, copy=False)
    answer_reward_u = answer_reward_full[first].astype(np.float32, copy=False)
    answer_credit_u = np.zeros_like(answer_reward_u, dtype=np.float32)

    traj_to_unique_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    for unique_row, (group_id, traj_id, _turn_id) in enumerate(unique_keys):
        traj_to_unique_rows[(group_id, traj_id)].append(unique_row)

    for rows in traj_to_unique_rows.values():
        running = 0.0
        for unique_row in sorted(rows, key=lambda idx: unique_keys[idx][2], reverse=True):
            running = float(answer_reward_u[unique_row]) + beta * running
            answer_credit_u[unique_row] = running

    grppo_reward_u = (step_weight * step_reward_u + answer_weight * answer_credit_u).astype(np.float32, copy=False)

    return {
        GRPPO_STEP_REWARD_KEY: step_reward_u[inverse].astype(np.float32, copy=False),
        GRPPO_ANSWER_REWARD_KEY: answer_reward_u[inverse].astype(np.float32, copy=False),
        GRPPO_ANSWER_CREDIT_KEY: answer_credit_u[inverse].astype(np.float32, copy=False),
        GRPPO_REWARD_KEY: grppo_reward_u[inverse].astype(np.float32, copy=False),
    }


def compute_grppo_component_advantage_values(
    non_tensor_batch: dict[str, Any],
    config: Any = None,
    *,
    epsilon: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Return row-aligned GRPPO component rewards and component advantages.

    ``step_reward`` and discounted ``answer_credit`` are normalized separately
    inside each ``(group_idx, turn_idx)`` cohort. A low-variance cohort only
    zeroes that signal's component advantage; rows are still trainable when the
    other signal has enough variance.
    """
    reward_values = compute_grppo_step_reward_values(non_tensor_batch, config)
    step_reward_full = reward_values[GRPPO_STEP_REWARD_KEY]
    answer_credit_full = reward_values[GRPPO_ANSWER_CREDIT_KEY]
    row_count = int(step_reward_full.shape[0])

    groups = np.asarray(non_tensor_batch["group_idx"], dtype=object).reshape(-1)
    trajs = np.asarray(non_tensor_batch["traj_idx"], dtype=object).reshape(-1)
    turns = _to_numpy_int64(non_tensor_batch["turn_idx"], factorize=False).reshape(-1)
    if not (len(groups) == len(trajs) == len(turns) == row_count):
        raise ValueError("GRPPO group_idx/traj_idx/turn_idx/reward columns must have the same length")

    unique_keys: list[tuple[str, str, int]] = []
    first_indices: list[int] = []
    inverse = np.zeros(row_count, dtype=np.int64)
    key_to_unique: dict[tuple[str, str, int], int] = {}
    for row in range(row_count):
        key = (str(groups[row]), str(trajs[row]), int(turns[row]))
        unique_row = key_to_unique.get(key)
        if unique_row is None:
            unique_row = len(unique_keys)
            key_to_unique[key] = unique_row
            unique_keys.append(key)
            first_indices.append(row)
        inverse[row] = unique_row

    first = np.asarray(first_indices, dtype=np.int64)
    step_reward_u = step_reward_full[first].astype(np.float32, copy=False)
    answer_reward_u = reward_values[GRPPO_ANSWER_REWARD_KEY][first].astype(np.float32, copy=False)
    answer_credit_u = answer_credit_full[first].astype(np.float32, copy=False)
    judge_valid_u = _grppo_judge_valid_mask(non_tensor_batch, row_count)[first]
    answer_credit_valid_u = judge_valid_u.copy()
    step_adv_u = np.zeros_like(step_reward_u, dtype=np.float32)
    answer_adv_u = np.zeros_like(answer_credit_u, dtype=np.float32)
    step_valid_u = np.zeros_like(step_reward_u, dtype=np.float32)
    answer_valid_u = np.zeros_like(answer_credit_u, dtype=np.float32)

    norm_by_std = _config_get_bool(config, "grppo_norm_by_std", False)
    min_std = max(_config_get_float(config, "grppo_min_std", 0.03), 0.0)
    step_weight = _config_get_float(config, "grppo_step_weight", 1.0)
    answer_weight = _config_get_float(config, "grppo_answer_weight", 1.0)

    cohort_to_unique_rows: dict[tuple[str, int], list[int]] = defaultdict(list)
    for unique_row, (group_id, _traj_id, turn_id) in enumerate(unique_keys):
        cohort_to_unique_rows[(group_id, int(turn_id))].append(unique_row)

    traj_to_unique_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    for unique_row, (group_id, traj_id, _turn_id) in enumerate(unique_keys):
        traj_to_unique_rows[(group_id, traj_id)].append(unique_row)
    for rows in traj_to_unique_rows.values():
        valid_future = True
        for unique_row in sorted(rows, key=lambda idx: unique_keys[idx][2], reverse=True):
            if not bool(judge_valid_u[unique_row]):
                valid_future = False
            answer_credit_valid_u[unique_row] = valid_future

    for rows in cohort_to_unique_rows.values():
        row_idx = np.asarray(rows, dtype=np.int64)
        step_row_idx = row_idx[judge_valid_u[row_idx]]
        answer_row_idx = row_idx[answer_credit_valid_u[row_idx]]
        step_adv, step_valid = _normalize_grppo_component(
            step_reward_u[step_row_idx],
            norm_by_std=norm_by_std,
            min_std=min_std,
            epsilon=epsilon,
        )
        answer_adv, answer_valid = _normalize_grppo_component(
            answer_credit_u[answer_row_idx],
            norm_by_std=norm_by_std,
            min_std=min_std,
            epsilon=epsilon,
        )
        step_adv_u[step_row_idx] = step_adv
        answer_adv_u[answer_row_idx] = answer_adv
        if step_valid:
            step_valid_u[step_row_idx] = 1.0
        if answer_valid:
            answer_valid_u[answer_row_idx] = 1.0

    grppo_adv_u = (step_weight * step_adv_u + answer_weight * answer_adv_u).astype(np.float32, copy=False)
    update_valid_u = np.maximum(step_valid_u, answer_valid_u).astype(np.float32, copy=False)
    step_reward_out_u = step_reward_u.copy()
    answer_reward_out_u = answer_reward_u.copy()
    answer_credit_out_u = answer_credit_u.copy()
    step_reward_out_u[~judge_valid_u] = 0.0
    answer_reward_out_u[~judge_valid_u] = 0.0
    answer_credit_out_u[~answer_credit_valid_u] = 0.0
    grppo_reward_out_u = (step_weight * step_reward_out_u + answer_weight * answer_credit_out_u).astype(
        np.float32,
        copy=False,
    )

    out = dict(reward_values)
    out.update(
        {
            GRPPO_STEP_REWARD_KEY: step_reward_out_u[inverse].astype(np.float32, copy=False),
            GRPPO_ANSWER_REWARD_KEY: answer_reward_out_u[inverse].astype(np.float32, copy=False),
            GRPPO_ANSWER_CREDIT_KEY: answer_credit_out_u[inverse].astype(np.float32, copy=False),
            GRPPO_REWARD_KEY: grppo_reward_out_u[inverse].astype(np.float32, copy=False),
            GRPPO_STEP_ADVANTAGE_KEY: step_adv_u[inverse].astype(np.float32, copy=False),
            GRPPO_ANSWER_ADVANTAGE_KEY: answer_adv_u[inverse].astype(np.float32, copy=False),
            GRPPO_ADVANTAGE_KEY: grppo_adv_u[inverse].astype(np.float32, copy=False),
            GRPPO_STEP_VALID_KEY: step_valid_u[inverse].astype(np.float32, copy=False),
            GRPPO_ANSWER_VALID_KEY: answer_valid_u[inverse].astype(np.float32, copy=False),
            GRPPO_UPDATE_VALID_KEY: update_valid_u[inverse].astype(np.float32, copy=False),
        }
    )
    return out


def _precomputed_grppo_component_values(
    non_tensor_batch: dict[str, Any],
    *,
    row_count: int,
) -> dict[str, np.ndarray] | None:
    """Return precomputed GRPPO values when the trainer already populated them.

    Step-level filtering may remove middle turns from a trajectory. In that
    case answer credit must stay the value computed on the full trajectory
    before filtering; recomputing from the filtered rows changes the discount
    distance. Missing fields mean the estimator is being used without the
    trainer-side GRPPO hook, so the caller should compute them normally.
    """
    keys = (
        GRPPO_STEP_REWARD_KEY,
        GRPPO_ANSWER_REWARD_KEY,
        GRPPO_ANSWER_CREDIT_KEY,
        GRPPO_REWARD_KEY,
        GRPPO_STEP_ADVANTAGE_KEY,
        GRPPO_ANSWER_ADVANTAGE_KEY,
        GRPPO_ADVANTAGE_KEY,
        GRPPO_STEP_VALID_KEY,
        GRPPO_ANSWER_VALID_KEY,
        GRPPO_UPDATE_VALID_KEY,
    )
    if not all(key in non_tensor_batch for key in keys):
        return None

    out: dict[str, np.ndarray] = {}
    for key in keys:
        values = _to_float_array(non_tensor_batch[key]).reshape(-1)
        if values.shape[0] != row_count:
            raise ValueError(
                f"precomputed {key} length {values.shape[0]} does not match batch row count {row_count}"
            )
        out[key] = values.astype(np.float32, copy=False)
    return out


@register_adv_est("streamweave_stepwise_grppo")
def compute_streamweave_stepwise_grppo(
    data,
    gamma=1.0,
    lam=1.0,
    config=None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    **kwargs,
):
    """GRPPO stepwise advantage normalized within ``(group_idx, turn_idx)``."""
    del gamma, lam, norm_adv_by_std_in_grpo, kwargs
    response_mask = data.batch["response_mask"]
    device = response_mask.device
    norm_by_std = _config_get_bool(config, "grppo_norm_by_std", False)

    with torch.no_grad():
        uniq_key, first_idx, inverse = _unique_turn_rows(data)
        first_idx_t = torch.as_tensor(first_idx, dtype=torch.long, device=device)
        mask_u = response_mask.index_select(0, first_idx_t).to(dtype=torch.float32)

        component_values = _precomputed_grppo_component_values(
            data.non_tensor_batch,
            row_count=int(response_mask.shape[0]),
        )
        if component_values is None:
            component_values = compute_grppo_component_advantage_values(data.non_tensor_batch, config, epsilon=epsilon)
        for key, values in component_values.items():
            data.non_tensor_batch[key] = np.asarray(values, dtype=np.float32)

        grppo_reward_full = _to_float_array(component_values[GRPPO_REWARD_KEY])
        grppo_adv_full = _to_float_array(component_values[GRPPO_ADVANTAGE_KEY])
        grppo_reward = grppo_reward_full[first_idx]
        grppo_advantage = grppo_adv_full[first_idx]

        group = uniq_key[:, 0]
        traj = uniq_key[:, 1]
        turn = uniq_key[:, 2]
        raw_groups = np.asarray(data.non_tensor_batch["group_idx"], dtype=object).reshape(-1)
        group_labels = {int(group[row]): str(raw_groups[int(first_idx[row])]) for row in range(len(first_idx))}

        group_turn_to_rewards: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
        for row, reward in enumerate(grppo_reward):
            group_turn_to_rewards[(int(group[row]), int(turn[row]))].append((row, float(reward)))

        adv_by_row = grppo_advantage.astype(np.float32, copy=False)

        if _trace_grppo_enabled():
            _trace_grppo_groups(
                group_turn_to_rewards=group_turn_to_rewards,
                adv_by_row=adv_by_row,
                group_labels=group_labels,
                traj=traj,
                norm_by_std=norm_by_std,
            )

        advantages_u = torch.zeros_like(mask_u)
        for row, adv in enumerate(adv_by_row.tolist()):
            advantages_u[row] = float(adv) * mask_u[row]

        reward_u = torch.as_tensor(grppo_reward, dtype=torch.float32, device=device).unsqueeze(-1) * mask_u
        inverse_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
        advantages = advantages_u.index_select(0, inverse_t)
        returns = reward_u.index_select(0, inverse_t)
    return advantages, returns


def _trace_grpo_enabled() -> bool:
    return env_flag(
        "STREAMWEAVE_TRACE_GRPO_GROUPS",
        default=env_flag("STREAMWEAVE_TRACE_FIRST_ROLLOUT", default=False),
    )


def _trace_trajsum_grpo_enabled() -> bool:
    return env_flag(
        "STREAMWEAVE_TRACE_TRAJSUM_GRPO_GROUPS",
        default=env_flag("STREAMWEAVE_TRACE_GRPO_GROUPS", default=False),
    )


def _trace_grpo_groups(
    *,
    group_to_traj_scores: dict[int, list[tuple[int, float]]],
    adv_by_traj: dict[tuple[int, int], float],
    group_labels: dict[int, str],
) -> None:
    for group_id in sorted(group_to_traj_scores, key=lambda item: group_labels.get(int(item), str(item))):
        if not trace_group_allowed(group_labels.get(int(group_id), str(group_id))):
            continue
        items = sorted(group_to_traj_scores[group_id], key=lambda item: item[0])
        values = np.asarray([score for _, score in items], dtype=np.float32)
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        scores = {int(traj_id): round(float(score), 4) for traj_id, score in items}
        advantages = {
            int(traj_id): round(float(adv_by_traj.get((int(group_id), int(traj_id)), 0.0)), 4)
            for traj_id, _ in items
        }
        trace_print(
            "[SW-TRACE grpo-group] "
            f"group={group_labels.get(int(group_id), str(group_id))} "
            f"traj_scores={scores} mean={mean:.4f} std={std:.4f} advantages={advantages}"
        )


def _trace_grppo_enabled() -> bool:
    return env_flag(
        "STREAMWEAVE_TRACE_GRPPO_GROUPS",
        default=env_flag("STREAMWEAVE_TRACE_FIRST_ROLLOUT", default=False),
    )


def _trace_grppo_groups(
    *,
    group_turn_to_rewards: dict[tuple[int, int], list[tuple[int, float]]],
    adv_by_row: np.ndarray,
    group_labels: dict[int, str],
    traj: np.ndarray,
    norm_by_std: bool,
) -> None:
    for (group_id, turn_id) in sorted(
        group_turn_to_rewards,
        key=lambda item: (group_labels.get(int(item[0]), str(item[0])), int(item[1])),
    ):
        if not trace_group_allowed(group_labels.get(int(group_id), str(group_id))):
            continue
        items = sorted(group_turn_to_rewards[(group_id, turn_id)], key=lambda item: int(traj[item[0]]))
        values = np.asarray([reward for _, reward in items], dtype=np.float32)
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        rewards = {int(traj[row]): round(float(reward), 4) for row, reward in items}
        advantages = {int(traj[row]): round(float(adv_by_row[row]), 4) for row, _ in items}
        trace_print(
            "[SW-TRACE grppo-group] "
            f"group={group_labels.get(int(group_id), str(group_id))} turn={int(turn_id)} "
            f"norm_by_std={norm_by_std} rewards={rewards} mean={mean:.4f} std={std:.4f} "
            f"advantages={advantages}"
        )


def _config_get_float(config: Any, key: str, default: float) -> float:
    if config is None:
        return float(default)
    try:
        return float(config.get(key, default))
    except AttributeError:
        return float(getattr(config, key, default))


def _config_get_filter_min_std(config: Any, default: float) -> float:
    if config is None:
        return float(default)
    try:
        filter_cfg = config.get("filter_groups", None)
    except AttributeError:
        filter_cfg = getattr(config, "filter_groups", None)
    if filter_cfg is None:
        return float(default)
    try:
        return float(filter_cfg.get("min_std", default))
    except AttributeError:
        return float(getattr(filter_cfg, "min_std", default))


def _config_get_bool(config: Any, key: str, default: bool) -> bool:
    if config is None:
        return bool(default)
    try:
        value = config.get(key, default)
    except AttributeError:
        value = getattr(config, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_grppo_component(
    values: np.ndarray,
    *,
    norm_by_std: bool,
    min_std: float,
    epsilon: float,
) -> tuple[np.ndarray, bool]:
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float32), False
    std = float(values.std(ddof=1))
    if std <= float(min_std):
        return np.zeros_like(values, dtype=np.float32), False
    centered = values - float(values.mean())
    if norm_by_std:
        centered = centered / (std + float(epsilon))
    return centered.astype(np.float32, copy=False), True


def _config_get_sequence(config: Any, key: str, default: list[Any]) -> list[Any]:
    if config is None:
        return list(default)
    try:
        value = config.get(key, default)
    except AttributeError:
        value = getattr(config, key, default)
    if value is None:
        return list(default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]
    try:
        return [item for item in value]
    except TypeError:
        return list(default)


@register_adv_est("streamweave_stepwise_rlmlr")
def compute_streamweave_stepwise_rlmlr(
    data,
    gamma=1.0,
    lam=1.0,
    config=None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    **kwargs,
):
    """RLMLR-style stepwise advantage.

    Implements the multi-level reward scheme from "Look Back to Reason Forward"
    (arXiv 2509.23040). Two clean signals are kept separate so they don't double-count:

    - **outcome reward** (default: ``success_score``) is the trajectory-level
      final-answer signal, normalized inside each ``group_idx`` group. ``success_score``
      is back-filled into every turn output at episode end, so taking the first
      occurrence per ``(group, traj)`` is sufficient.
    - **state reward** is a turn-level process signal assembled from one or more
      components (default: ``step_score + format_score``), normalized inside each
      ``(group, turn)`` cohort so step-level baselines only compare same-turn rollouts.

    NOTE: do NOT default the outcome to ``trajectory_score`` — that field already
    bakes ``w_format * format_mean + w_step * step_mean + w_success * success_score``
    together (see ``rewards.compute_trajectory_reward``), so feeding it as outcome
    while also adding state reward would re-credit the process signal twice.
    The legacy behaviour stays reachable via ``rlmlr_outcome_key=trajectory_score``
    for ablation purposes.

    By default we omit the std term, matching the paper's recommendation. The two
    advantages are blended with alpha (default 0.8 favouring outcome):
        A_t = alpha * A_out + (1 - alpha) * A_state,t
    """
    del gamma, lam, norm_adv_by_std_in_grpo, kwargs
    response_mask = data.batch["response_mask"]
    device = response_mask.device

    alpha = _config_get_float(config, "rlmlr_alpha", 0.8)
    alpha = min(max(alpha, 0.0), 1.0)
    norm_by_std = _config_get_bool(config, "rlmlr_norm_by_std", False)
    outcome_key = str(_config_get_str(config, "rlmlr_outcome_key", "success_score"))
    state_components = _config_get_sequence(config, "rlmlr_state_components", ["step_score", "format_score"])
    state_components = [str(item) for item in state_components if str(item).strip()]
    raw_weights = _config_get_sequence(config, "rlmlr_state_weights", [])
    state_weights = _coerce_state_weights(raw_weights, len(state_components))

    if outcome_key not in data.non_tensor_batch:
        raise KeyError(
            f"streamweave_stepwise_rlmlr outcome key {outcome_key!r} not found in non_tensor_batch; "
            f"available keys include {sorted(data.non_tensor_batch.keys())}"
        )
    if not state_components:
        raise ValueError("streamweave_stepwise_rlmlr requires at least one state component")
    for required in state_components:
        if required not in data.non_tensor_batch:
            raise KeyError(
                f"streamweave_stepwise_rlmlr state component {required!r} not found in non_tensor_batch"
            )

    with torch.no_grad():
        uniq_key, first_idx, inverse = _unique_turn_rows(data)
        first_idx_t = torch.as_tensor(first_idx, dtype=torch.long, device=device)
        mask_u = response_mask.index_select(0, first_idx_t).to(dtype=torch.float32)

        group = uniq_key[:, 0]
        traj = uniq_key[:, 1]
        turn = uniq_key[:, 2]
        raw_groups = np.asarray(data.non_tensor_batch["group_idx"], dtype=object).reshape(-1)
        group_labels = {int(group[row]): str(raw_groups[int(first_idx[row])]) for row in range(len(first_idx))}

        outcome_full = _to_float_array(data.non_tensor_batch[outcome_key])
        outcome_scores = outcome_full[first_idx]

        state_scores = np.zeros(len(first_idx), dtype=np.float32)
        for component, weight in zip(state_components, state_weights, strict=True):
            component_full = _to_float_array(data.non_tensor_batch[component])
            state_scores = state_scores + float(weight) * component_full[first_idx]

        outcome_by_traj: dict[tuple[int, int], float] = {}
        for row, score in enumerate(outcome_scores):
            outcome_by_traj[(int(group[row]), int(traj[row]))] = float(score)

        group_to_outcome: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (group_id, traj_id), score in outcome_by_traj.items():
            group_to_outcome[group_id].append((traj_id, score))

        adv_outcome: dict[tuple[int, int], float] = {}
        for group_id, items in group_to_outcome.items():
            values = torch.tensor([score for _, score in items], dtype=torch.float32, device=device)
            normed = _normalize_centered(values, norm_by_std=norm_by_std, epsilon=epsilon)
            for (traj_id, _), adv in zip(items, normed.tolist(), strict=True):
                adv_outcome[(group_id, traj_id)] = float(adv)

        group_turn_to_state: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
        for row, score in enumerate(state_scores):
            key = (int(group[row]), int(turn[row]))
            group_turn_to_state[key].append((int(traj[row]), float(score)))

        adv_state: dict[tuple[int, int, int], float] = {}
        for (group_id, turn_id), items in group_turn_to_state.items():
            values = torch.tensor([score for _, score in items], dtype=torch.float32, device=device)
            normed = _normalize_centered(values, norm_by_std=norm_by_std, epsilon=epsilon)
            for (traj_id, _), adv in zip(items, normed.tolist(), strict=True):
                adv_state[(group_id, traj_id, turn_id)] = float(adv)

        if _trace_rlmlr_enabled():
            _trace_rlmlr_groups(
                group_to_outcome=group_to_outcome,
                group_turn_to_state=group_turn_to_state,
                adv_outcome=adv_outcome,
                adv_state=adv_state,
                group_labels=group_labels,
                alpha=alpha,
                norm_by_std=norm_by_std,
                outcome_key=outcome_key,
                state_components=state_components,
                state_weights=state_weights,
            )

        advantages_u = torch.zeros_like(mask_u)
        for row in range(len(first_idx)):
            g_id = int(group[row])
            t_id = int(traj[row])
            tu_id = int(turn[row])
            adv_out = adv_outcome.get((g_id, t_id), 0.0)
            adv_st = adv_state.get((g_id, t_id, tu_id), 0.0)
            adv_combined = alpha * adv_out + (1.0 - alpha) * adv_st
            advantages_u[row] = float(adv_combined) * mask_u[row]

        inverse_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
        advantages = advantages_u.index_select(0, inverse_t)
        returns_u = torch.as_tensor(outcome_scores, dtype=torch.float32, device=device).unsqueeze(-1) * mask_u
        returns = returns_u.index_select(0, inverse_t)
    return advantages, returns


def _coerce_state_weights(weights: list[Any], num_components: int) -> list[float]:
    if num_components <= 0:
        return []
    if not weights:
        return [1.0] * num_components
    if len(weights) != num_components:
        raise ValueError(
            f"rlmlr_state_weights length {len(weights)} must match rlmlr_state_components length {num_components}"
        )
    out: list[float] = []
    for value in weights:
        try:
            out.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rlmlr_state_weights contains non-numeric value: {value!r}") from exc
    return out


def _config_get_str(config: Any, key: str, default: str) -> str:
    if config is None:
        return str(default)
    try:
        value = config.get(key, default)
    except AttributeError:
        value = getattr(config, key, default)
    if value is None:
        return str(default)
    return str(value)


def _normalize_centered(
    values: torch.Tensor,
    *,
    norm_by_std: bool,
    epsilon: float,
) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    centered = values - values.mean()
    if norm_by_std:
        return centered / (values.std(unbiased=True) + epsilon)
    return centered


def _trace_rlmlr_enabled() -> bool:
    return env_flag(
        "STREAMWEAVE_TRACE_RLMLR_GROUPS",
        default=env_flag("STREAMWEAVE_TRACE_FIRST_ROLLOUT", default=False),
    )


def _trace_rlmlr_groups(
    *,
    group_to_outcome: dict[int, list[tuple[int, float]]],
    group_turn_to_state: dict[tuple[int, int], list[tuple[int, float]]],
    adv_outcome: dict[tuple[int, int], float],
    adv_state: dict[tuple[int, int, int], float],
    group_labels: dict[int, str],
    alpha: float,
    norm_by_std: bool,
    outcome_key: str = "success_score",
    state_components: list[str] | None = None,
    state_weights: list[float] | None = None,
) -> None:
    for group_id in sorted(group_to_outcome, key=lambda item: group_labels.get(int(item), str(item))):
        if not trace_group_allowed(group_labels.get(int(group_id), str(group_id))):
            continue
        items = sorted(group_to_outcome[group_id], key=lambda item: item[0])
        outcome_values = np.asarray([score for _, score in items], dtype=np.float32)
        outcome_mean = float(outcome_values.mean()) if outcome_values.size else 0.0
        outcome_std = float(outcome_values.std(ddof=1)) if outcome_values.size > 1 else 0.0
        outcome_scores = {int(traj_id): round(float(score), 4) for traj_id, score in items}
        outcome_advs = {
            int(traj_id): round(float(adv_outcome.get((int(group_id), int(traj_id)), 0.0)), 4)
            for traj_id, _ in items
        }
        turn_keys = sorted(turn_id for (g_id, turn_id) in group_turn_to_state if g_id == group_id)
        state_summary = {}
        for turn_id in turn_keys:
            state_items = sorted(group_turn_to_state[(group_id, turn_id)], key=lambda item: item[0])
            state_summary[int(turn_id)] = {
                "scores": {int(t_id): round(float(s), 4) for t_id, s in state_items},
                "advs": {
                    int(t_id): round(
                        float(adv_state.get((int(group_id), int(t_id), int(turn_id)), 0.0)),
                        4,
                    )
                    for t_id, _ in state_items
                },
            }
        components_label = ""
        if state_components:
            weights = state_weights or [1.0] * len(state_components)
            components_label = " state_components=" + ",".join(
                f"{name}*{weight:.2f}" for name, weight in zip(state_components, weights, strict=False)
            )
        trace_print(
            "[SW-TRACE rlmlr-group] "
            f"group={group_labels.get(int(group_id), str(group_id))} "
            f"alpha={alpha:.3f} norm_by_std={norm_by_std} outcome_key={outcome_key}{components_label} "
            f"outcome_scores={outcome_scores} outcome_mean={outcome_mean:.4f} outcome_std={outcome_std:.4f} "
            f"outcome_advs={outcome_advs} state={state_summary}"
        )
