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
        from streamweave_rl.agent_loop_stepwise import _finalize_aborted_outputs
        from streamweave_rl.advantage import compute_streamweave_stepwise_gae, compute_streamweave_stepwise_traj_grpo
        from streamweave.schemas import ModelAction, ModelEvent, QualityReport
        from streamweave_rl.judge import JudgeConfig, JudgeResult, _parse_judge_response
        from streamweave_rl.rewards import (
            StreamWeaveRewardConfig,
            compute_note_frequency_score,
            compute_step_reward,
            compute_success_score,
            judge_blocked_by_note_frequency,
        )
    except ModuleNotFoundError as exc:
        print(f"StreamWeave RL smoke test skipped: missing dependency {exc.name!r}.")
        return

    cfg = StreamWeaveRewardConfig()
    assert cfg.w_format == 0.3
    assert cfg.w_step == 0.3
    assert cfg.w_success == 0.4
    assert cfg.score_scale == 2.0
    assert compute_success_score("cutting onion", "onion") == 2.0
    one_note = ModelAction(
        state="state",
        answer="",
        events=[ModelEvent(kind="note", start_time=0.0, end_time=1.0)],
    )
    score, streak, reasons = compute_note_frequency_score(action=one_note, previous_no_note_streak=2, cfg=cfg)
    assert score == 2.0 and streak == 0 and reasons == []

    stale_no_note = ModelAction(
        state="state",
        answer="",
        events=[ModelEvent(kind="bridge", start_time=0.0, end_time=5.0, text="No major change.")],
    )
    score, streak, reasons = compute_note_frequency_score(action=stale_no_note, previous_no_note_streak=2, cfg=cfg)
    assert score == 0.0 and streak == 3 and "note_stale" in reasons

    two_notes = ModelAction(
        state="state",
        answer="",
        events=[
            ModelEvent(kind="note", start_time=0.0, end_time=1.0),
            ModelEvent(kind="note", start_time=1.0, end_time=2.0),
        ],
    )
    score, _, reasons = compute_note_frequency_score(action=two_notes, previous_no_note_streak=0, cfg=cfg)
    assert score == 0.0 and "too_many_notes" in reasons

    reward = compute_step_reward(
        quality=QualityReport(valid=True, parser_ok=True, metrics={"num_notes": 1}),
        cfg=cfg,
        ctx={"action": one_note, "previous_no_note_streak": 0},
        total_steps=5,
    )
    assert reward.format_score == 2.0
    assert reward.note_frequency_score == 2.0
    assert reward.step_score == 2.0

    judge_cfg = StreamWeaveRewardConfig(judge=JudgeConfig(enable=True), judge_weight=1.0)
    judge_reward = compute_step_reward(
        quality=QualityReport(valid=True, parser_ok=True, metrics={"num_notes": 1}),
        cfg=judge_cfg,
        ctx={"action": one_note, "previous_no_note_streak": 0, "judge_result": JudgeResult(score=0.5, status="ok")},
        total_steps=1,
    )
    assert judge_reward.note_frequency_score == 2.0
    assert judge_reward.judge_score == 1.0
    assert judge_reward.step_score == 1.5
    blocked_judge_reward = compute_step_reward(
        quality=QualityReport(valid=True, parser_ok=True, metrics={"num_notes": 2}),
        cfg=judge_cfg,
        ctx={"action": two_notes, "previous_no_note_streak": 0, "judge_result": JudgeResult(score=1.0, status="ok")},
        total_steps=1,
    )
    assert blocked_judge_reward.note_frequency_score == 0.0
    assert blocked_judge_reward.judge_score == 0.0
    assert blocked_judge_reward.step_score == 0.0
    assert blocked_judge_reward.info["judge_blocked_by_note_frequency"] is True
    no_gate_cfg = StreamWeaveRewardConfig(note_frequency_weight=0.0, judge=JudgeConfig(enable=True), judge_weight=1.0)
    assert judge_blocked_by_note_frequency(note_frequency_score=0.0, cfg=no_gate_cfg) is False
    nested_judge = _parse_judge_response(
        """
        {
          "keyframe_selection": {"score": 0.8, "reason": "useful anchor"},
          "bridge_quality": {"score": 0.6, "reason": "mostly concise"},
          "semantic_alignment": {"score": 0.4, "reason": "partial mismatch"},
          "state_factuality": {"score": 0.2, "reason": "speculative"},
          "caps_applied": ["contradiction cap"],
          "overall": 0.5,
          "issues": ["minor mismatch"]
        }
        """
    )
    assert nested_judge.score == 0.5
    assert nested_judge.scores["keyframe_selection"] == 0.8
    assert nested_judge.scores["state_factuality"] == 0.2
    assert "cap: contradiction cap" in nested_judge.issues

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

    class DummyOutput:
        def __init__(self) -> None:
            self.extra_fields = {}
            self.reward_score = None
            self.num_turns = 0

    aborted = DummyOutput()
    _finalize_aborted_outputs([aborted], group_idx="sample-a", traj_idx=0, reason="unit-test")
    assert {
        "format_score",
        "step_score",
        "note_frequency_score",
        "judge_score",
        "success_score",
        "trajectory_score",
        "reward_info",
    } <= set(aborted.extra_fields["reward_extra_info"])
    print("StreamWeave RL smoke test passed.")


if __name__ == "__main__":
    main()
