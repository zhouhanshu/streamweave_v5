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


def _trace_grpo_enabled() -> bool:
    return env_flag(
        "STREAMWEAVE_TRACE_GRPO_GROUPS",
        default=env_flag("STREAMWEAVE_TRACE_FIRST_ROLLOUT", default=False),
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


def _config_get_float(config: Any, key: str, default: float) -> float:
    if config is None:
        return float(default)
    try:
        return float(config.get(key, default))
    except AttributeError:
        return float(getattr(config, key, default))
