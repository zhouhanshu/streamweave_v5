"""Lightweight checks for StreamWeave RL adapter."""

from __future__ import annotations

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    np = None
    torch = None
    MISSING_NUMERIC_DEP = exc.name
else:
    MISSING_NUMERIC_DEP = None


class FakeBatch:
    def __init__(self) -> None:
        if np is None or torch is None:
            raise RuntimeError(f"missing dependency {MISSING_NUMERIC_DEP!r}")
        response_mask = torch.tensor(
            [
                [1, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 0],
            ],
            dtype=torch.long,
        )
        self.batch = {
            "response_mask": response_mask,
            "token_level_rewards": torch.tensor(
                [
                    [0.0, 0.1, 0.0],
                    [0.4, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.8, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "values": torch.zeros(4, 3),
        }
        self.non_tensor_batch = {
            "group_idx": np.array(["sample-a", "sample-a", "sample-a", "sample-a"], dtype=object),
            "traj_idx": np.array([0, 0, 1, 1], dtype=np.int64),
            "turn_idx": np.array([0, 1, 0, 1], dtype=np.int64),
            "trajectory_score": np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32),
        }


def main() -> None:
    if MISSING_NUMERIC_DEP is not None:
        print(f"StreamWeave RL smoke test skipped: missing dependency {MISSING_NUMERIC_DEP!r}.")
        return
    try:
        from streamweave_rl.advantage import compute_streamweave_stepwise_gae, compute_streamweave_stepwise_traj_grpo
        from streamweave_rl.rewards import StreamWeaveRewardConfig, compute_success_score
    except ModuleNotFoundError as exc:
        print(f"StreamWeave RL smoke test skipped: missing dependency {exc.name!r}.")
        return

    cfg = StreamWeaveRewardConfig()
    assert cfg.w_format == 0.2
    assert compute_success_score("cutting onion", "onion") == 1.0

    data = FakeBatch()
    grpo_adv, grpo_returns = compute_streamweave_stepwise_traj_grpo(data)
    assert grpo_adv.shape == data.batch["response_mask"].shape
    assert grpo_returns.shape == data.batch["response_mask"].shape
    assert torch.isfinite(grpo_adv).all()
    assert torch.isfinite(grpo_returns).all()

    gae_adv, gae_returns = compute_streamweave_stepwise_gae(data, gamma=1.0, lam=1.0)
    assert gae_adv.shape == data.batch["response_mask"].shape
    assert gae_returns.shape == data.batch["response_mask"].shape
    assert torch.isfinite(gae_adv).all()

    _, custom_returns = compute_streamweave_stepwise_gae(data, gamma=1.0, lam=1.0, config={"ignore_value": -1.0})
    ignored = custom_returns[data.batch["response_mask"] == 0]
    assert torch.equal(ignored, torch.full_like(ignored, -1.0))
    print("StreamWeave RL smoke test passed.")


if __name__ == "__main__":
    main()
