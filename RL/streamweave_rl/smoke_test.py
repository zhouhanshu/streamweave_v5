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
            "grppo_step_reward": np.array([0.2, 0.4, 0.8, 0.2], dtype=np.float32),
            "grppo_answer_reward": np.array([0.0, 1.0, 0.0, 0.5], dtype=np.float32),
        }


class FakePrecomputedGRPPOBatch:
    def __init__(self) -> None:
        if np is None or torch is None:
            raise RuntimeError(f"missing dependency {MISSING_NUMERIC_DEP!r}")
        self.batch = {"response_mask": torch.ones(3, 1, dtype=torch.long)}
        self.non_tensor_batch = {
            "group_idx": np.array(["sample-a", "sample-a", "sample-a"], dtype=object),
            "traj_idx": np.array([0, 0, 0], dtype=np.int64),
            "turn_idx": np.array([0, 2, 3], dtype=np.int64),
            "grppo_step_reward": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "grppo_answer_reward": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            # These values come from the full trajectory turns [0, 1, 2, 3]
            # with beta=0.5 and answer rewards [0, 1, 0, 1]. Turn 1 was
            # filtered out, but its credit contribution must remain in turn 0.
            "grppo_answer_credit": np.array([0.625, 0.5, 1.0], dtype=np.float32),
            "grppo_reward": np.array([0.625, 0.5, 1.0], dtype=np.float32),
            "grppo_step_advantage": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "grppo_answer_advantage": np.array([7.0, 8.0, 9.0], dtype=np.float32),
            "grppo_advantage": np.array([7.0, 8.0, 9.0], dtype=np.float32),
            "grppo_step_signal_valid": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "grppo_answer_signal_valid": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "grppo_update_signal_valid": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        }


def main() -> None:
    if MISSING_NUMERIC_DEP is not None:
        print(f"StreamWeave RL smoke test skipped: missing dependency {MISSING_NUMERIC_DEP!r}.")
        return
    try:
        from streamweave_rl.agent_loop_stepwise import (
            _finalize_aborted_outputs,
            _grppo_extra_fields,
            _renormalize_grppo_silence_rewards,
        )
        from streamweave_rl.advantage import (
            compute_streamweave_stepwise_bilevel_gae,
            compute_streamweave_stepwise_grppo,
            compute_streamweave_stepwise_ppo_gae,
            compute_streamweave_stepwise_rlmlr,
            compute_streamweave_stepwise_traj_grpo,
            compute_streamweave_stepwise_trajsum_grpo,
            compute_grppo_component_advantage_values,
            compute_grppo_step_reward_values,
            compute_trajsum_grpo_values,
        )
        from streamweave_rl.env import (
            StreamWeaveRLEnv,
            _answer_reward_scale,
            _final_grppo_answer_reward,
            _grppo_answer_event_enabled,
            _timeline_grppo_answer_supervision,
            _normalize_query_annotations,
            _target_answer_score_from_judge_or_rule,
        )
        from streamweave_rl.rewards import StreamWeaveRewardConfig
        from streamweave_rl.dapo import compute_group_filter_result, select_reward_extra_infos
        from streamweave.schemas import ModelAction, ModelEvent, QualityReport
        from streamweave_rl.judge import (
            JudgeConfig,
            JudgeResult,
            StepJudge,
            _build_grppo_answer_judge_content,
            _build_grppo_judge_content,
            _parse_grppo_judge_response,
            _parse_judge_response,
            get_judge_backend,
            reset_judge_backend_cache,
        )
        from streamweave_rl.rewards import (
            StreamWeaveRewardConfig,
            compute_grppo_step_reward,
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

    score, source = _target_answer_score_from_judge_or_rule(
        rule_score=0.0,
        judge_result=JudgeResult(status="ok", score=1.8, scores={"answer_reward": 1.0}),
        grppo_enabled=True,
        answer_reward_event=True,
        answer_supervision_kind="answer",
    )
    assert score == 1.0
    assert source == "judge"
    score, source = _target_answer_score_from_judge_or_rule(
        rule_score=0.25,
        judge_result=JudgeResult(status="error", score=0.0, scores={"answer_reward": 1.0}),
        grppo_enabled=True,
        answer_reward_event=True,
        answer_supervision_kind="answer",
    )
    assert score == 0.25
    assert source == "rule"
    assert cfg.grppo_process_weight == 1.0
    assert cfg.grppo_format_weight == 0.1
    assert cfg.grppo_note_frequency_weight == 0.0
    assert cfg.score_scale == 1.0
    assert abs(compute_grppo_step_reward(process_score=0.5, format_score=1.0, cfg=cfg) - 0.6) < 1e-6
    assert abs(compute_grppo_step_reward(process_score=2.0, format_score=1.0, cfg=cfg) - 2.1) < 1e-6
    grppo_anchor_cfg = StreamWeaveRewardConfig(
        grppo_process_weight=0.6,
        grppo_format_weight=0.1,
        grppo_note_frequency_weight=0.3,
    )
    assert (
        abs(
            compute_grppo_step_reward(
                process_score=0.9,
                format_score=1.0,
                note_frequency_score=1.0,
                cfg=grppo_anchor_cfg,
            )
            - (0.9 * 0.9 + 0.1)
        )
        < 1e-6
    )
    assert (
        abs(
            compute_grppo_step_reward(
                process_score=0.9,
                format_score=1.0,
                note_frequency_score=0.0,
                cfg=grppo_anchor_cfg,
            )
            - 0.1
        )
        < 1e-6
    )
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

    import asyncio
    class _FakeGenerateResult:
        def __init__(self, text: str) -> None:
            self.text = text

    class _AnswerErrorBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, content, generate_kwargs=None):  # noqa: ANN001
            del content, generate_kwargs
            self.calls += 1
            if self.calls == 1:
                return _FakeGenerateResult(
                    """
                    {
                      "delta_groundedness": {"score": 0.8, "reason": "process ok"},
                      "anchor_keyframe": {"score": 0.6, "reason": "process ok"},
                      "semantic_alignment": {"score": 0.4, "reason": "process ok"},
                      "state_groundedness": {"score": 0.2, "reason": "process ok"},
                      "issues": []
                    }
                    """
                )
            return _FakeGenerateResult("not json")

    original_build = judge_module._build_backend
    original_to_thread = judge_module.asyncio.to_thread
    fake_backend = _AnswerErrorBackend()
    judge_module._build_backend = lambda cfg: fake_backend
    async def _inline_to_thread(func, /, *args, **kwargs):  # noqa: ANN001
        return func(*args, **kwargs)
    judge_module.asyncio.to_thread = _inline_to_thread
    try:
        split_judge = StepJudge(
            JudgeConfig(
                enable=True,
                backend="answer_error_fake",
                prompt_version="streamweave_grppo_judge_v1",
                max_retries=1,
                retry_backoff_seconds=0.0,
            )
        )
        split_result = asyncio.run(
            split_judge.score_step(
                memory_before="",
                qa_history="<qa>q</qa>",
                frames=[],
                raw_action=ModelAction(state="state", answer="", events=[]),
                raw_output="<state>state</state><answer></answer>",
                quality=QualityReport(valid=True, parser_ok=True, metrics={}),
                query_label={"event_type": "answer_target", "question": "Q?", "answer": "A", "timestamp": 1.0},
                answer_reward_event=True,
            )
        )
        assert fake_backend.calls == 3
        assert split_result.status == "error"
        assert "answer_judge judge JSON parse failed" in split_result.error
        assert "answer_judge_attempt_2" in split_result.raw_response
    finally:
        judge_module._build_backend = original_build
        judge_module.asyncio.to_thread = original_to_thread
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

    grppo_judge = _parse_grppo_judge_response(
        """
        {
          "delta_groundedness": {"score": 0.8, "reason": "grounded delta"},
          "anchor_keyframe": {"score": 0.6, "reason": "reasonable anchor"},
          "semantic_alignment": {"score": 0.4, "reason": "partial alignment"},
          "state_groundedness": {"score": 0.2, "reason": "weak state"},
          "answer_reward": {"score": 1.0, "reason": "correct answer"},
          "issues": []
        }
        """
    )
    assert grppo_judge.score == 1.0
    assert grppo_judge.scores["answer_reward"] == 1.0
    assert grppo_judge.reasons["delta_groundedness"] == "grounded delta"
    grppo_checklist_judge = _parse_grppo_judge_response(
        """
        {
          "delta_groundedness": {
            "score": 1.0,
              "checks": {
                "delta_captures_action_progress": 1,
                "delta_captures_state_or_location_changes": 0,
                "delta_preserves_query_relevant_details": 0.5,
                "delta_no_visual_hallucination": 1,
                "delta_not_polluted_by_qa": 1
              },
            "reason": "checklist overrides score"
          },
          "anchor_keyframe": {
            "score": 0.0,
              "checks": {
                "anchor_count_valid": 1,
                "anchor_time_and_body_valid": 0.5,
                "anchor_representative_if_present": 0.5,
                "first_window_rule_followed": 1
              },
            "reason": "one applicable anchor check passes"
          },
          "note_keyframe": {"score": 1.0, "reason": "legacy field should not override explicit anchor"},
          "semantic_alignment": {
            "score": 0.4,
            "checks": {
              "semantic_output_coherent": 1,
              "semantic_text_anchor_express_current_frames": 0,
              "semantic_no_cross_step_contradiction": 1
            },
            "reason": "partial alignment"
          },
          "state_groundedness": {
            "score": 0.2,
            "checks": {
              "state_uses_available_evidence": 1,
              "state_identifies_question_scope": 0,
              "state_decision_is_grounded": 1,
              "state_no_visual_hallucination": 1,
              "state_consistent_with_answer": 0,
              "state_no_unreasonable_hallucination": 1
            },
            "reason": "weak state"
          },
          "answer_reward": {"score": 1.0, "reason": "correct answer"},
          "issues": []
        }
        """
    )
    assert grppo_checklist_judge.scores["delta_groundedness"] == 0.0
    assert grppo_checklist_judge.scores["anchor_keyframe"] == 3 / 4
    assert grppo_checklist_judge.scores["semantic_alignment"] == 0.0
    assert grppo_checklist_judge.scores["state_groundedness"] == 0.0
    assert abs(grppo_checklist_judge.score - (2.0 * 0.75 / 4)) < 1e-8
    grppo_missing_checklists = _parse_grppo_judge_response(
        """
        {
          "delta_groundedness": {
            "checks": {
              "delta_captures_action_progress": 1,
              "delta_captures_state_or_location_changes": 1,
              "delta_preserves_query_relevant_details": 1,
              "delta_no_visual_hallucination": 1,
              "delta_not_polluted_by_qa": 1
            }
          },
          "anchor_keyframe": {"score": 1.0},
          "semantic_alignment": {"score": 1.0},
          "state_groundedness": {"score": 1.0},
          "issues": []
        }
        """
    )
    assert abs(grppo_missing_checklists.score - (2.0 * 1 / 4)) < 1e-8
    assert grppo_missing_checklists.scores["anchor_keyframe"] == 0.0
    assert grppo_missing_checklists.scores["semantic_alignment"] == 0.0
    assert grppo_missing_checklists.scores["state_groundedness"] == 0.0
    grppo_process_content = _build_grppo_judge_content(
        memory_before="",
        qa_history="",
        frames=[],
        raw_output="<state>state</state><answer></answer>",
        quality=QualityReport(valid=True, parser_ok=True, metrics={}),
        query_label=None,
        answer_reward_event=False,
        answer_correctness=None,
    )
    grppo_process_prompt = "\n".join(item.text for item in grppo_process_content if item.type == "text")
    assert "answer_reward" not in grppo_process_prompt
    assert "note_keyframe" not in grppo_process_prompt
    assert "previous_no_note_streak" not in grppo_process_prompt
    assert "anchor_keyframe" in grppo_process_prompt
    assert "previous_no_anchor_streak" not in grppo_process_prompt
    assert "Prompt kind" not in grppo_process_prompt
    assert "Return 4 scores" in grppo_process_prompt
    grppo_answer_content = _build_grppo_judge_content(
        memory_before="",
        qa_history="",
        frames=[],
        raw_output="<state>state</state><answer></answer>",
        quality=QualityReport(valid=True, parser_ok=True, metrics={}),
        query_label={
            "event_type": "answer_target",
            "question": "What color?",
            "timestamp": 3.0,
            "ground_truth": "C",
            "answer": "red",
            "options": ["blue", "green", "red"],
        },
        answer_reward_event=True,
        answer_correctness=1.0,
    )
    grppo_answer_prompt = "\n".join(item.text for item in grppo_answer_content if item.type == "text")
    assert "answer_reward" in grppo_answer_prompt
    assert "Options:\nA. blue\nB. green\nC. red" in grppo_answer_prompt
    assert "Requirement: the model must answer now. Reference answer: C. red" in grppo_answer_prompt
    assert "Reference answer: red" not in grppo_answer_prompt
    assert "rule_answer_correctness" not in grppo_answer_prompt
    assert "Rule Correctness Reference" not in grppo_answer_prompt
    assert "set answer_reward to that scalar exactly" not in grppo_answer_prompt
    grppo_answer_only_content = _build_grppo_answer_judge_content(
        raw_output="<state>no coated pieces yet</state><answer></answer>",
        quality=QualityReport(valid=True, parser_ok=True, metrics={}),
        query_label={
            "event_type": "answer_target",
            "question": "How many?",
            "timestamp": 68.0,
            "gt": "C",
            "answer": "0",
            "content": "0",
            "options": ["6", "8", "0", "3"],
        },
    )
    grppo_answer_only_prompt = "\n".join(item.text for item in grppo_answer_only_content if item.type == "text")
    assert "Memory Before This Step" not in grppo_answer_only_prompt
    assert "Current Video Frames" not in grppo_answer_only_prompt
    assert "Requirement: the model must answer now. Reference answer: C. 0" in grppo_answer_only_prompt
    assert "staying silent is incorrect" in grppo_answer_only_prompt
    grppo_query_only_content = _build_grppo_judge_content(
        memory_before="",
        qa_history="",
        frames=[],
        raw_output="<state>state</state><answer></answer>",
        quality=QualityReport(valid=True, parser_ok=True, metrics={}),
        query_label={"event_type": "query", "content": "What ingredient is being added?", "timestamp": 3.0},
        answer_reward_event=True,
        answer_correctness=1.0,
    )
    grppo_query_only_prompt = "\n".join(item.text for item in grppo_query_only_content if item.type == "text")
    assert "Current question: What ingredient is being added?" in grppo_query_only_prompt
    assert "Requirement: the model should not answer now; <answer> should be empty." in grppo_query_only_prompt
    assert "Reference answer: What ingredient is being added?" not in grppo_query_only_prompt

    annotations = _normalize_query_annotations(
        metadata={
            "query_events": [
                {
                    "qid": "q0",
                    "time": 127.0,
                    "content": "What ingredient is being added?",
                    "answer_type": "mcq",
                    "options": ["bacon", "tomato", "lettuce", "onion", "cheese"],
                    "answer_events": [
                        {"gt": "A", "answer": "bacon", "content": "A. bacon", "time": 180.0},
                        {"gt": "B", "answer": "tomato", "content": "B. tomato", "time": 185.0},
                        {"gt": "C", "answer": "lettuce", "content": "C. lettuce", "time": 192.0},
                    ],
                }
            ],
        },
        question="",
        query_timestamp=0.0,
        ground_truth="",
    )
    assert [item["event_type"] for item in annotations] == ["query", "answer_target", "answer_target", "answer_target"]
    assert [float(item["timestamp"]) for item in annotations] == [127.0, 180.0, 185.0, 192.0]
    assert annotations[1]["ground_truth"] == "A"
    assert annotations[1]["answer"] == "bacon"
    same_step_env = object.__new__(StreamWeaveRLEnv)
    same_step_env.step_idx = 0
    same_step_env.groups = [[type("Frame", (), {"global_index": 0, "end_time": 1.0})()]]
    same_step_env.query_by_frame = {}
    same_step_env.query_annotations_by_frame = {
        0: [
            {"event_type": "query", "question": "What color?", "timestamp": 1.0},
            {"event_type": "answer_target", "question": "What color?", "timestamp": 1.0, "ground_truth": "red"},
        ]
    }
    same_step_env.sample = type("Sample", (), {"sample_id": "same-step"})()
    same_step_env.grppo_enabled = True
    same_step_env.latest_query_event = None
    same_step_env.env = type(
        "Env",
        (),
        {
            "add_question": lambda self, record: None,
            "evict_memory": lambda self, t: None,
            "memory_text": lambda self: "",
            "build_prompt": lambda self, frames: ([], "", [], []),
        },
    )()
    same_step_env._prompt_frames = lambda group: group
    same_step_env.settings = type(
        "Settings",
        (),
        {
            "runtime": type("Runtime", (), {"resolution": 0})(),
            "rl_reward": StreamWeaveRewardConfig(grppo_answer_event_mode="timeline"),
        },
    )()
    obs = StreamWeaveRLEnv._prepare_current_turn(same_step_env)
    assert obs["messages"] == [{"role": "user", "content": []}]
    assert same_step_env.current_query_event["event_type"] == "query"
    assert same_step_env.current_answer_target["event_type"] == "answer_target"
    assert same_step_env.current_answer_label["ground_truth"] == "red"
    assert same_step_env.current_answer_supervision.kind == "answer"
    assert same_step_env.current_step_query_count == 1
    assert same_step_env.current_step_answer_target_count == 1
    assert (
        _final_grppo_answer_reward(
            raw_score=1.0,
            supervision_kind="silence",
            has_answer=True,
            label_status="query_without_target",
            use_llm_for_silence=False,
        )
        == 0.0
    )
    assert (
        _final_grppo_answer_reward(
            raw_score=1.0,
            supervision_kind="silence",
            has_answer=True,
            label_status="query_without_target",
            use_llm_for_silence=True,
            silence_reward_value=0.1,
        )
        == 0.0
    )
    assert (
        _final_grppo_answer_reward(
            raw_score=0.0,
            supervision_kind="silence",
            has_answer=False,
            label_status="query_without_target",
            use_llm_for_silence=False,
        )
        == 1.0
    )
    assert (
        _final_grppo_answer_reward(
            raw_score=1.0,
            supervision_kind="silence",
            has_answer=False,
            label_status="query_without_target",
            use_llm_for_silence=True,
            silence_reward_value=0.1,
        )
        == 0.1
    )
    assert (
        _final_grppo_answer_reward(
            raw_score=0.0,
            supervision_kind="silence",
            has_answer=False,
            label_status="query_without_target",
            use_llm_for_silence=True,
            silence_reward_value=0.1,
        )
        == 0.0
    )
    assert _final_grppo_answer_reward(raw_score=1.0, supervision_kind="none", has_answer=True, label_status="missing_label") == 0.0
    assert _final_grppo_answer_reward(raw_score=1.0, supervision_kind="answer", has_answer=True, label_status="parser_invalid") == 0.0
    assert (
        _final_grppo_answer_reward(
            raw_score=1.0,
            supervision_kind="silence",
            has_answer=False,
            label_status="parser_invalid",
            use_llm_for_silence=True,
            silence_reward_value=0.2,
        )
        == 0.0
    )
    assert (
        _final_grppo_answer_reward(
            raw_score=1.0,
            supervision_kind="silence",
            has_answer=False,
            label_status="query_without_target",
            use_llm_for_silence=True,
            silence_reward=False,
        )
        == 0.0
    )
    assert _final_grppo_answer_reward(raw_score=1.0, supervision_kind="answer", has_answer=False, label_status="scored") == 0.0
    # answer kind with attempt-reward (additive): silence_reward_value plus the fixed answer-attempt
    # offset is added as participation reward.
    # silent -> 0; wrong attempt -> attempt only; correct attempt -> attempt + 1.
    assert abs(
        _final_grppo_answer_reward(
            raw_score=0.0,
            supervision_kind="answer",
            has_answer=True,
            label_status="scored",
            silence_reward=True,
            silence_reward_value=0.1,
        )
        - 0.2
    ) < 1e-8
    assert abs(
        _final_grppo_answer_reward(
            raw_score=1.0,
            supervision_kind="answer",
            has_answer=True,
            label_status="scored",
            silence_reward=True,
            silence_reward_value=0.1,
        )
        - 1.2
    ) < 1e-8
    # silence_reward=False disables the attempt reward (interpretation A).
    assert (
        _final_grppo_answer_reward(
            raw_score=1.0,
            supervision_kind="answer",
            has_answer=True,
            label_status="scored",
            silence_reward=False,
            silence_reward_value=0.1,
        )
        == 1.0
    )
    assert (
        _final_grppo_answer_reward(
            raw_score=0.0,
            supervision_kind="answer",
            has_answer=True,
            label_status="scored",
            silence_reward=False,
            silence_reward_value=0.1,
        )
        == 0.0
    )
    assert abs(_answer_reward_scale("answer", silence_reward_value=0.1) - 1.2) < 1e-8
    assert _answer_reward_scale("answer", silence_reward=False, silence_reward_value=0.1) == 1.0
    assert _answer_reward_scale("silence", silence_reward_value=0.1) == 0.1
    timeline_query = {"event_type": "query", "question": "What color?", "timestamp": 1.0}
    timeline_no_event = _timeline_grppo_answer_supervision(
        grppo_enabled=True,
        current_query_event=None,
        current_answer_target=None,
        latest_query_event=timeline_query,
        has_answer=False,
    )
    assert timeline_no_event.kind == "none"
    timeline_silence = _timeline_grppo_answer_supervision(
        grppo_enabled=True,
        current_query_event=None,
        current_answer_target=None,
        latest_query_event=timeline_query,
        has_answer=True,
    )
    assert timeline_silence.kind == "silence"
    assert timeline_silence.label["event_type"] == "answer_silence"
    assert timeline_silence.label["should_answer"] is False
    timeline_current_query = _timeline_grppo_answer_supervision(
        grppo_enabled=True,
        current_query_event=timeline_query,
        current_answer_target=None,
        latest_query_event=timeline_query,
        has_answer=False,
    )
    assert timeline_current_query.kind == "silence"
    assert timeline_current_query.status == "query_silence"
    timeline_answer = _timeline_grppo_answer_supervision(
        grppo_enabled=True,
        current_query_event=timeline_query,
        current_answer_target={"event_type": "answer_target", "ground_truth": "red"},
        latest_query_event=timeline_query,
        has_answer=False,
    )
    assert timeline_answer.kind == "answer"
    assert not _grppo_answer_event_enabled(
        grppo_enabled=True,
        cfg=StreamWeaveRewardConfig(grppo_answer_event_mode="required_only"),
        label={"event_type": "query", "question": "What color?"},
        has_query=True,
        has_answer_target=False,
        has_answer=True,
    )
    assert _grppo_answer_event_enabled(
        grppo_enabled=True,
        cfg=StreamWeaveRewardConfig(grppo_answer_event_mode="required_only"),
        label={"event_type": "answer_target", "ground_truth": "C"},
        has_query=False,
        has_answer_target=True,
        has_answer=False,
    )

    data = FakeBatch()
    grppo_values = compute_grppo_step_reward_values(data.non_tensor_batch, config={"grppo_answer_decay": 0.5})
    assert np.allclose(grppo_values["grppo_answer_credit"], np.array([0.5, 1.0, 0.25, 0.5], dtype=np.float32))
    assert np.allclose(grppo_values["grppo_reward"], np.array([0.7, 1.4, 1.05, 0.7], dtype=np.float32))
    fallback_non_tensor = dict(data.non_tensor_batch)
    fallback_non_tensor.pop("grppo_step_reward")
    fallback_non_tensor.update(
        {
            "grppo_delta_groundedness": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "grppo_anchor_keyframe": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "grppo_semantic_alignment": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "grppo_state_groundedness": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        }
    )
    fallback_values = compute_grppo_step_reward_values(fallback_non_tensor, config={"grppo_answer_decay": 0.5})
    assert np.allclose(
        fallback_values["grppo_step_reward"],
        np.array([1.5, 1.5, 1.5, 0.5], dtype=np.float32),
    )
    grppo_adv, grppo_returns = compute_streamweave_stepwise_grppo(
        data,
        config={"grppo_answer_decay": 0.5, "grppo_norm_by_std": False},
    )
    expected_grppo_adv = torch.tensor(
        [
            [-0.175, -0.175, 0.0],
            [0.35, 0.0, 0.0],
            [0.175, 0.175, 0.0],
            [-0.35, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_grppo_returns = torch.tensor(
        [
            [0.7, 0.7, 0.0],
            [1.4, 0.0, 0.0],
            [1.05, 1.05, 0.0],
            [0.7, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(grppo_adv, expected_grppo_adv, atol=1e-6), grppo_adv
    assert torch.allclose(grppo_returns, expected_grppo_returns, atol=1e-6), grppo_returns
    component_values = compute_grppo_component_advantage_values(
        data.non_tensor_batch,
        config={"grppo_answer_decay": 0.5, "grppo_norm_by_std": False, "grppo_min_std": 0.2},
    )
    assert np.allclose(
        component_values["grppo_step_advantage"],
        np.array([-0.3, 0.0, 0.3, 0.0], dtype=np.float32),
    )
    assert np.allclose(
        component_values["grppo_answer_advantage"],
        np.array([0.0, 0.25, 0.0, -0.25], dtype=np.float32),
    )
    assert np.allclose(
        component_values["grppo_advantage"],
        np.array([-0.3, 0.25, 0.3, -0.25], dtype=np.float32),
    )
    assert component_values["grppo_update_signal_valid"].tolist() == [1.0, 1.0, 1.0, 1.0]
    judge_error_non_tensor = dict(data.non_tensor_batch)
    judge_error_non_tensor["judge_status"] = np.array(["ok", "error", "ok", "ok"], dtype=object)
    judge_error_values = compute_grppo_component_advantage_values(
        judge_error_non_tensor,
        config={"grppo_answer_decay": 0.5, "grppo_norm_by_std": False, "grppo_min_std": 0.2},
    )
    assert np.allclose(
        judge_error_values["grppo_step_advantage"],
        np.array([-0.3, 0.0, 0.3, 0.0], dtype=np.float32),
    )
    assert np.allclose(
        judge_error_values["grppo_answer_advantage"],
        np.zeros(4, dtype=np.float32),
    )
    assert np.allclose(
        judge_error_values["grppo_answer_credit"],
        np.array([0.0, 0.0, 0.25, 0.5], dtype=np.float32),
    )
    assert judge_error_values["grppo_update_signal_valid"].tolist() == [1.0, 0.0, 1.0, 0.0]
    precomputed_grppo = FakePrecomputedGRPPOBatch()
    precomputed_adv, precomputed_returns = compute_streamweave_stepwise_grppo(
        precomputed_grppo,
        config={"grppo_answer_decay": 0.5, "grppo_norm_by_std": False, "grppo_min_std": 0.0},
    )
    assert torch.allclose(
        precomputed_adv.squeeze(-1),
        torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32),
        atol=1e-6,
    ), precomputed_adv
    assert torch.allclose(
        precomputed_returns.squeeze(-1),
        torch.tensor([0.625, 0.5, 1.0], dtype=torch.float32),
        atol=1e-6,
    ), precomputed_returns
    assert np.allclose(
        precomputed_grppo.non_tensor_batch["grppo_answer_credit"],
        np.array([0.625, 0.5, 1.0], dtype=np.float32),
    )

    grpo_adv, grpo_returns = compute_streamweave_stepwise_traj_grpo(data)
    assert grpo_adv.shape == data.batch["response_mask"].shape
    assert grpo_returns.shape == data.batch["response_mask"].shape
    assert torch.isfinite(grpo_adv).all()
    assert torch.isfinite(grpo_returns).all()

    trajsum_values = compute_trajsum_grpo_values(
        data.non_tensor_batch,
        config={"trajsum_answer_weight": 0.5, "trajsum_norm_by_std": True},
    )
    assert np.allclose(
        trajsum_values["grpo_trajsum_turn_reward"],
        np.array([0.2, 0.9, 0.8, 0.45], dtype=np.float32),
    )
    assert np.allclose(
        trajsum_values["grpo_trajsum_score"],
        np.array([1.1, 1.1, 1.25, 1.25], dtype=np.float32),
    )
    trajsum_filtered_values = compute_trajsum_grpo_values(
        data.non_tensor_batch,
        config={
            "trajsum_answer_weight": 0.5,
            "trajsum_norm_by_std": True,
            "filter_groups": {"min_std": 0.2},
        },
    )
    assert np.allclose(
        trajsum_filtered_values["grpo_trajsum_advantage"],
        np.zeros(4, dtype=np.float32),
    )
    assert trajsum_filtered_values["grpo_trajsum_signal_valid"].tolist() == [0.0, 0.0, 0.0, 0.0]
    trajsum_adv, trajsum_returns = compute_streamweave_stepwise_trajsum_grpo(
        data,
        config={"trajsum_answer_weight": 0.5, "trajsum_norm_by_std": True},
    )
    expected_trajsum_adv = torch.tensor(
        [
            [-0.7071, -0.7071, 0.0],
            [-0.7071, 0.0, 0.0],
            [0.7071, 0.7071, 0.0],
            [0.7071, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_trajsum_returns = torch.tensor(
        [
            [1.1, 1.1, 0.0],
            [1.1, 0.0, 0.0],
            [1.25, 1.25, 0.0],
            [1.25, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(trajsum_adv, expected_trajsum_adv, atol=1e-3), trajsum_adv
    assert torch.allclose(trajsum_returns, expected_trajsum_returns, atol=1e-6), trajsum_returns

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

    def make_grppo_output(supervision: float, has_answer: float, reward: float, scale: float) -> DummyOutput:
        item = DummyOutput()
        grppo_fields = {
            "grppo_answer_supervision": supervision,
            "grppo_has_answer": has_answer,
            "grppo_answer_reward": reward,
            "grppo_answer_reward_scale": scale,
        }
        item.extra_fields.update(grppo_fields)
        item.extra_fields["reward_extra_info"] = dict(grppo_fields)
        return item

    answer_correct = make_grppo_output(2.0, 1.0, 1.2, 1.2)
    answer_wrong = make_grppo_output(2.0, 1.0, 0.2, 1.2)
    silence_correct = make_grppo_output(1.0, 0.0, 0.1, 0.1)
    silence_false_answer = make_grppo_output(1.0, 1.0, 0.0, 0.1)
    _renormalize_grppo_silence_rewards(
        [answer_correct, answer_wrong, silence_correct, silence_false_answer],
        silence_reward_enabled=True,
        silence_reward_value=0.1,
    )
    assert abs(answer_correct.extra_fields["grppo_answer_reward"] - 1.2) < 1e-8
    assert abs(answer_wrong.extra_fields["grppo_answer_reward"] - 0.2) < 1e-8
    assert abs(silence_correct.extra_fields["grppo_answer_reward"] - (0.1 / 3.0)) < 1e-8
    assert silence_false_answer.extra_fields["grppo_answer_reward"] == 0.0
    assert abs(silence_false_answer.extra_fields["grppo_answer_reward_scale"] - (0.1 / 3.0)) < 1e-8
    assert (
        silence_correct.extra_fields["reward_extra_info"]["grppo_answer_reward"]
        == silence_correct.extra_fields["grppo_answer_reward"]
    )

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
    aborted_grppo = DummyOutput()
    _finalize_aborted_outputs(
        [aborted_grppo],
        group_idx="sample-a",
        traj_idx=0,
        reason="unit-test",
        grppo_enabled=True,
    )
    assert {
        "grppo_step_reward",
        "grppo_judge_step_reward",
        "grppo_format_score",
        "grppo_answer_reward",
        "grppo_prompt_kind",
        "grppo_label_status",
    } <= set(aborted_grppo.extra_fields["reward_extra_info"])
    assert _grppo_extra_fields(
        {
            "grppo_enabled": True,
            "grppo_judge_step_reward": 0.5,
            "grppo_format_score": 1.0,
            "grppo_step_reward": 0.55,
        }
    )["grppo_step_reward"] == 0.55
    print("StreamWeave RL smoke test passed.")


if __name__ == "__main__":
    main()
