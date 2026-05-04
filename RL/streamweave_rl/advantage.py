"""Custom advantage estimators for StreamWeave stepwise rollouts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_adv_est


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


@register_adv_est("streamweave_stepwise_gae")
def compute_streamweave_stepwise_gae(
    data,
    gamma,
    lam,
    config=None,
    ignore_value: float = -100.0,
    **kwargs,
):
    token_level_rewards = data.batch["token_level_rewards"]
    values = data.batch.get("values", torch.zeros_like(token_level_rewards))
    response_mask = data.batch["response_mask"]
    ignore_value = _config_get_float(config, "ignore_value", ignore_value)

    with torch.no_grad():
        device = token_level_rewards.device
        uniq_key, first_idx, inverse = _unique_turn_rows(data)
        first_idx_t = torch.as_tensor(first_idx, dtype=torch.long, device=device)

        rewards_u = token_level_rewards.index_select(0, first_idx_t)
        values_u = values.index_select(0, first_idx_t)
        mask_u = response_mask.index_select(0, first_idx_t).to(dtype=token_level_rewards.dtype)
        mask_b = mask_u.to(dtype=torch.bool)

        group_ids = torch.as_tensor(uniq_key[:, 0], dtype=torch.long, device="cpu")
        traj_ids = torch.as_tensor(uniq_key[:, 1], dtype=torch.long, device="cpu")
        turn_ids = torch.as_tensor(uniq_key[:, 2], dtype=torch.long, device="cpu")

        bs_u, length = rewards_u.shape
        turn_rewards = (rewards_u * mask_u).sum(dim=1)
        valid_counts = mask_b.sum(dim=1)
        arange = torch.arange(length, device=device).view(1, length).expand(bs_u, length)
        last_pos = (mask_b.to(torch.long) * arange).max(dim=1).values
        last_pos = torch.where(valid_counts > 0, last_pos, torch.full_like(last_pos, -1))
        gather_pos = last_pos.clamp(min=0).view(bs_u, 1)
        turn_values = values_u.gather(1, gather_pos).squeeze(1)
        turn_values = torch.where(last_pos >= 0, turn_values, torch.zeros_like(turn_values))

        advantages_u = torch.zeros_like(rewards_u)
        returns_u = torch.full_like(rewards_u, float(ignore_value))

        gamma_val = float(gamma.item() if isinstance(gamma, torch.Tensor) else gamma)
        lam_val = float(lam.item() if isinstance(lam, torch.Tensor) else lam)

        pair_keys = torch.stack([group_ids, traj_ids], dim=1)
        for group_id, traj_id in torch.unique(pair_keys, dim=0).tolist():
            row_ids = torch.nonzero((group_ids == group_id) & (traj_ids == traj_id), as_tuple=False).view(-1)
            if row_ids.numel() == 0:
                continue
            ordered = row_ids[torch.argsort(turn_ids[row_ids])].tolist()
            next_value = 0.0
            last_gae = 0.0
            for row in reversed(ordered):
                reward = float(turn_rewards[row].item())
                value = float(turn_values[row].item())
                delta = reward + gamma_val * next_value - value
                last_gae = delta + gamma_val * lam_val * last_gae
                ret = last_gae + value
                pos = int(last_pos[row].item())
                if pos >= 0:
                    advantages_u[row, mask_b[row]] = last_gae
                    returns_u[row, pos] = ret
                next_value = value

        if float(mask_u.sum().item()) > 1.0:
            advantages_u = verl_F.masked_whiten(advantages_u, mask_u)

        inverse_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
        return advantages_u.index_select(0, inverse_t), returns_u.index_select(0, inverse_t)


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

        advantages_u = torch.zeros_like(mask_u)
        for row in range(len(first_idx)):
            adv = adv_by_traj.get((int(group[row]), int(traj[row])), 0.0)
            advantages_u[row] = float(adv) * mask_u[row]

        inverse_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
        advantages = advantages_u.index_select(0, inverse_t)
        score_u = torch.as_tensor(scores, dtype=torch.float32, device=device).unsqueeze(-1) * mask_u
        returns = score_u.index_select(0, inverse_t)
    return advantages, returns


def _config_get_float(config: Any, key: str, default: float) -> float:
    if config is None:
        return float(default)
    try:
        return float(config.get(key, default))
    except AttributeError:
        return float(getattr(config, key, default))
