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
                    [0.0, 0.0, 0.0],
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
            # The following four columns are only consumed by streamweave_stepwise_rlmlr;
            # existing estimators ignore them, so adding the columns is safe.
            # success_score is trajectory-level (back-filled to every turn).
            "success_score": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
            # step_score and format_score are turn-level (each turn distinct).
            "step_score": np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
            "format_score": np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
        }


def main() -> None:
    if MISSING_NUMERIC_DEP is not None:
        print(f"StreamWeave RL smoke test skipped: missing dependency {MISSING_NUMERIC_DEP!r}.")
        return
    try:
        from streamweave_rl.agent_loop_stepwise import _finalize_aborted_outputs
        from streamweave_rl.advantage import (
            compute_streamweave_stepwise_bilevel_gae,
            compute_streamweave_stepwise_ppo_gae,
            compute_streamweave_stepwise_rlmlr,
            compute_streamweave_stepwise_traj_grpo,
        )
        from streamweave_rl.dapo import compute_group_filter_result, select_reward_extra_infos
        from streamweave.schemas import ModelAction, ModelEvent, QualityReport
        from streamweave_rl.judge import (
            JudgeConfig,
            JudgeResult,
            StepJudge,
            _parse_judge_response,
            get_judge_backend,
            reset_judge_backend_cache,
        )
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
    assert cfg.w_format == 0.1
    assert cfg.w_step == 0.1
    assert cfg.w_success == 0.8
    assert cfg.note_frequency_weight == 1.0
    assert cfg.judge_weight == 1.0
    assert cfg.score_scale == 1.0
    assert compute_success_score("cutting onion", "onion") == 1.0
    one_note = ModelAction(
        state="state",
        answer="",
        events=[ModelEvent(kind="note", start_time=0.0, end_time=1.0)],
    )
    score, streak, reasons = compute_note_frequency_score(action=one_note, previous_no_note_streak=2, cfg=cfg)
    assert score == 1.0 and streak == 0 and reasons == []

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
    assert reward.format_score == 1.0
    assert reward.note_frequency_score == 1.0
    assert reward.step_score == 1.0

    judge_cfg = StreamWeaveRewardConfig(judge=JudgeConfig(enable=True))
    judge_reward = compute_step_reward(
        quality=QualityReport(valid=True, parser_ok=True, metrics={"num_notes": 1}),
        cfg=judge_cfg,
        ctx={"action": one_note, "previous_no_note_streak": 0, "judge_result": JudgeResult(score=0.5, status="ok")},
        total_steps=1,
    )
    assert judge_reward.note_frequency_score == 1.0
    assert judge_reward.judge_score == 0.5
    assert judge_reward.step_score == 0.5
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
    no_gate_cfg = StreamWeaveRewardConfig(note_frequency_weight=0.0, judge=JudgeConfig(enable=True), judge_weight=0.7)
    assert judge_blocked_by_note_frequency(note_frequency_score=0.0, cfg=no_gate_cfg) is False
    # Backend reuse: multiple StepJudge instances with the same config must
    # share a single backend instance (one client per process, not one per
    # rollout trajectory).
    reset_judge_backend_cache()
    shared_cfg = JudgeConfig(enable=True, backend="mock")
    backend_a = get_judge_backend(shared_cfg)
    backend_b = get_judge_backend(shared_cfg)
    assert backend_a is backend_b, "get_judge_backend should return cached backend"

    judge_a = StepJudge(shared_cfg)
    judge_b = StepJudge(shared_cfg)
    assert judge_a._ensure_backend() is judge_b._ensure_backend()
    assert judge_a._ensure_backend() is backend_a

    # Distinct config -> distinct backend (cache key must include relevant fields).
    other_cfg = JudgeConfig(enable=True, backend="mock", model="other-model")
    backend_c = get_judge_backend(other_cfg)
    assert backend_c is not backend_a, "different judge config must not share backend"
    reset_judge_backend_cache()

    # Bulk reuse: simulate one training step with many trajectories, each
    # building its own StepJudge. We expect zero extra backend builds beyond
    # the first call per distinct config.
    import streamweave_rl.judge as judge_module
    original_build = judge_module._build_backend
    build_count = {"n": 0}
    def _counting_build(cfg):
        build_count["n"] += 1
        return original_build(cfg)
    judge_module._build_backend = _counting_build
    try:
        bulk_cfg = JudgeConfig(enable=True, backend="mock")
        bulk_judges = [StepJudge(bulk_cfg) for _ in range(256)]
        bulk_backends = [j._ensure_backend() for j in bulk_judges]
        assert len(set(id(b) for b in bulk_backends)) == 1
        assert build_count["n"] == 1, f"expected 1 backend build, got {build_count['n']}"
    finally:
        judge_module._build_backend = original_build
        reset_judge_backend_cache()

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
    assert nested_judge.reasons["keyframe_selection"] == "useful anchor"
    assert nested_judge.reasons["state_factuality"] == "speculative"
    assert "cap: contradiction cap" in nested_judge.issues

    data = FakeBatch()
    grpo_adv, grpo_returns = compute_streamweave_stepwise_traj_grpo(data)
    assert grpo_adv.shape == data.batch["response_mask"].shape
    assert grpo_returns.shape == data.batch["response_mask"].shape
    assert torch.isfinite(grpo_adv).all()
    assert torch.isfinite(grpo_returns).all()

    dapo_result = compute_group_filter_result(
        {
            "group_idx": np.array(["a", "a", "b", "b"], dtype=object),
            "traj_idx": np.array([0, 1, 0, 1], dtype=np.int64),
            "trajectory_score": np.array([1.0, 2.0, 0.5, 0.5], dtype=np.float32),
        },
        metric="trajectory_score",
    )
    assert dapo_result is not None
    assert dapo_result.total_groups == 2
    assert dapo_result.valid_groups == 1
    assert dapo_result.keep_mask.tolist() == [True, True, False, False]
    assert dapo_result.metrics["traj/valid_group_ratio"] == 0.5
    selected_infos = select_reward_extra_infos({"score": np.arange(4), "name": "kept"}, dapo_result.keep_mask)
    assert selected_infos["score"].tolist() == [0, 1]
    assert selected_infos["name"] == "kept"

    ppo_adv, ppo_returns = compute_streamweave_stepwise_ppo_gae(data, gamma=1.0, lam=1.0)
    assert ppo_adv.shape == data.batch["response_mask"].shape
    assert ppo_returns.shape == data.batch["response_mask"].shape
    assert torch.isfinite(ppo_adv).all()
    expected_ppo_returns = torch.tensor(
        [
            [0.4, 0.4, 0.0],
            [0.4, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(ppo_returns, expected_ppo_returns, atol=1e-6)

    # RLMLR (Look Back to Reason Forward) two-level normalize + alpha blend.
    #
    # Default: outcome=success_score, state=step_score+format_score (w=1.0 each).
    # outcome (success_score, trajectory-level, back-filled to all turns):
    #   per-traj: traj0=0.0, traj1=1.0; group "sample-a" mean = 0.5
    #   adv_out: traj0 -> -0.5, traj1 -> +0.5
    # state (step + format) per (traj, turn):
    #   (0,0)=0.2+1.0=1.2; (0,1)=0.4+0.0=0.4; (1,0)=0.6+1.0=1.6; (1,1)=0.8+1.0=1.8
    #   per-turn mean:
    #     turn 0: (1.2+1.6)/2 = 1.4 -> adv: -0.2, +0.2
    #     turn 1: (0.4+1.8)/2 = 1.1 -> adv: -0.7, +0.7
    # alpha=0.8 combined:
    #   (0,0): 0.8*(-0.5) + 0.2*(-0.2) = -0.44
    #   (0,1): 0.8*(-0.5) + 0.2*(-0.7) = -0.54
    #   (1,0): 0.8*(+0.5) + 0.2*(+0.2) = +0.44
    #   (1,1): 0.8*(+0.5) + 0.2*(+0.7) = +0.54
    rlmlr_adv, rlmlr_returns = compute_streamweave_stepwise_rlmlr(
        data,
        config={"rlmlr_alpha": 0.8, "rlmlr_norm_by_std": False},
    )
    assert rlmlr_adv.shape == data.batch["response_mask"].shape
    assert rlmlr_returns.shape == data.batch["response_mask"].shape
    expected_rlmlr_adv = torch.tensor(
        [
            [-0.44, -0.44, 0.0],
            [-0.54, 0.0, 0.0],
            [0.44, 0.44, 0.0],
            [0.54, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rlmlr_adv, expected_rlmlr_adv, atol=1e-6), rlmlr_adv
    # Returns: success_score broadcast over valid response tokens.
    expected_rlmlr_returns = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rlmlr_returns, expected_rlmlr_returns, atol=1e-6), rlmlr_returns

    # alpha=1.0 must reduce to pure outcome (success_score group-centered).
    # success_score per traj: [0.0, 1.0] -> adv: [-0.5, +0.5]
    rlmlr_adv_outcome_only, _ = compute_streamweave_stepwise_rlmlr(
        data,
        config={"rlmlr_alpha": 1.0, "rlmlr_norm_by_std": False},
    )
    expected_outcome_only = torch.tensor(
        [
            [-0.5, -0.5, 0.0],
            [-0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rlmlr_adv_outcome_only, expected_outcome_only, atol=1e-6), rlmlr_adv_outcome_only

    # alpha=0.0 must reduce to pure per-turn state advantage (step + format).
    rlmlr_adv_state_only, _ = compute_streamweave_stepwise_rlmlr(
        data,
        config={"rlmlr_alpha": 0.0, "rlmlr_norm_by_std": False},
    )
    expected_state_only = torch.tensor(
        [
            [-0.2, -0.2, 0.0],
            [-0.7, 0.0, 0.0],
            [0.2, 0.2, 0.0],
            [0.7, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rlmlr_adv_state_only, expected_state_only, atol=1e-6), rlmlr_adv_state_only

    # Ablation: state=step_score only (no format), w=1.0.
    # state per (traj, turn): [0.2, 0.4, 0.6, 0.8]
    # turn 0 mean=0.4 -> adv: -0.2, +0.2 ; turn 1 mean=0.6 -> adv: -0.2, +0.2
    rlmlr_adv_step_only, _ = compute_streamweave_stepwise_rlmlr(
        data,
        config={
            "rlmlr_alpha": 0.0,
            "rlmlr_norm_by_std": False,
            "rlmlr_state_components": ["step_score"],
            "rlmlr_state_weights": [1.0],
        },
    )
    expected_step_only = torch.tensor(
        [
            [-0.2, -0.2, 0.0],
            [-0.2, 0.0, 0.0],
            [0.2, 0.2, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rlmlr_adv_step_only, expected_step_only, atol=1e-6), rlmlr_adv_step_only

    # Legacy compat path: outcome_key=trajectory_score reproduces the buggy
    # double-counting behaviour, kept reachable for ablations. Verifies the
    # config branch wires through to traj_grpo-equivalent semantics when
    # alpha=1.0 and norm_by_std=True.
    rlmlr_adv_legacy, _ = compute_streamweave_stepwise_rlmlr(
        data,
        config={
            "rlmlr_alpha": 1.0,
            "rlmlr_norm_by_std": True,
            "rlmlr_outcome_key": "trajectory_score",
        },
    )
    traj_grpo_adv, _ = compute_streamweave_stepwise_traj_grpo(data)
    assert torch.allclose(rlmlr_adv_legacy, traj_grpo_adv, atol=1e-5), (rlmlr_adv_legacy, traj_grpo_adv)

    bilevel_adv, bilevel_returns = compute_streamweave_stepwise_bilevel_gae(
        data,
        gamma=1.0,
        lam=1.0,
        config={"high_level_gamma": 0.5},
    )
    assert bilevel_adv.shape == data.batch["response_mask"].shape
    assert bilevel_returns.shape == data.batch["response_mask"].shape
    assert torch.isfinite(bilevel_adv).all()
    expected_bilevel_returns = torch.tensor(
        [
            [0.2, 0.2, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.6, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(bilevel_returns, expected_bilevel_returns, atol=1e-6)

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
