"""Reward components for StreamWeave RL.

Rewards keep the existing trajectory-level GRPO plumbing while adding denser
process signals.  Per-step reward combines raw XML format compliance with a
configurable step score, and the final turn adds dataset task success.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from streamweave.schemas import ModelAction, QualityReport

from .judge import JudgeConfig, JudgeResult, judge_config_from_mapping
from .scorers import score_answer
from .schemas import StepRewardResult, TrajectoryRewardResult


@dataclass(slots=True)
class StreamWeaveRewardConfig:
    w_format: float = 0.3
    w_success: float = 0.4
    w_step: float = 0.3
    score_scale: float = 2.0
    format_mode: str = "valid"
    success_mode: str = "dataset"
    success_scorer: str = "auto"
    enable_note_frequency_reward: bool = True
    note_frequency_weight: float = 1.0
    judge_weight: float = 0.0
    max_notes_per_step: int = 1
    stale_note_after_steps: int = 3
    note_frequency_penalty_score: float = 0.0
    judge: JudgeConfig = field(default_factory=JudgeConfig)


def reward_config_from_mapping(data: dict[str, Any] | None) -> StreamWeaveRewardConfig:
    if not data:
        return StreamWeaveRewardConfig()
    mapping = dict(data)
    judge_data = mapping.pop("judge", None) or mapping.pop("llm_judge", None)
    allowed = {item.name for item in fields(StreamWeaveRewardConfig)}
    values = {key: value for key, value in mapping.items() if key in allowed and key != "judge"}
    if judge_data:
        values["judge"] = judge_config_from_mapping(judge_data)
    return StreamWeaveRewardConfig(**values)


def compute_format_score(quality: QualityReport, *, mode: str = "valid", score_scale: float = 2.0) -> float:
    if mode == "parser_ok":
        return float(score_scale) if quality.parser_ok else 0.0
    if mode == "format_reward":
        return _clamp_score(float(quality.rewards.format_reward) * float(score_scale), score_scale=score_scale)
    if mode == "valid":
        return float(score_scale) if quality.valid else 0.0
    raise ValueError(f"Unknown format reward mode: {mode}")


def compute_note_frequency_score(
    *,
    action: ModelAction,
    previous_no_note_streak: int,
    cfg: StreamWeaveRewardConfig,
) -> tuple[float, int, list[str]]:
    num_notes = sum(1 for event in action.events if event.kind == "note")
    next_no_note_streak = previous_no_note_streak + 1 if num_notes == 0 else 0
    reasons: list[str] = []
    score = float(cfg.score_scale)
    if cfg.max_notes_per_step >= 0 and num_notes > cfg.max_notes_per_step:
        reasons.append("too_many_notes")
        score = float(cfg.note_frequency_penalty_score)
    if cfg.stale_note_after_steps > 0 and next_no_note_streak >= cfg.stale_note_after_steps:
        reasons.append("note_stale")
        score = float(cfg.note_frequency_penalty_score)
    return _clamp_score(score, cfg=cfg), next_no_note_streak, reasons


def judge_blocked_by_note_frequency(*, note_frequency_score: float, cfg: StreamWeaveRewardConfig) -> bool:
    return bool(
        cfg.enable_note_frequency_reward
        and cfg.note_frequency_weight > 0
        and note_frequency_score < _max_score(cfg)
    )


def compute_step_score(
    *,
    note_frequency_score: float,
    judge_score: float,
    cfg: StreamWeaveRewardConfig,
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    weights: dict[str, float] = {}
    if cfg.enable_note_frequency_reward and cfg.note_frequency_weight > 0:
        components["note_frequency_score"] = _clamp_score(note_frequency_score, cfg=cfg)
        weights["note_frequency_score"] = float(cfg.note_frequency_weight)
    if cfg.judge.enable and cfg.judge_weight > 0:
        components["judge_score"] = _clamp_score(judge_score, cfg=cfg)
        weights["judge_score"] = float(cfg.judge_weight)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0, components
    score = sum(components[key] * weights[key] for key in components) / total_weight
    return _clamp_score(score, cfg=cfg), components


def compute_step_reward(
    *,
    quality: QualityReport,
    cfg: StreamWeaveRewardConfig,
    ctx: dict[str, Any] | None = None,
    total_steps: int = 1,
) -> StepRewardResult:
    ctx = ctx or {}
    denominator = max(int(total_steps), 1)
    format_score = compute_format_score(quality, mode=cfg.format_mode, score_scale=cfg.score_scale)
    action = ctx.get("action")
    if not isinstance(action, ModelAction):
        raise TypeError("compute_step_reward requires ctx['action'] to be a ModelAction")
    previous_no_note_streak = int(ctx.get("previous_no_note_streak", 0) or 0)
    note_frequency_result = ctx.get("note_frequency_result")
    if isinstance(note_frequency_result, tuple) and len(note_frequency_result) == 3:
        note_frequency_score = _clamp_score(float(note_frequency_result[0]), cfg=cfg)
        no_note_streak = int(note_frequency_result[1])
        note_reasons = list(note_frequency_result[2] or [])
    else:
        note_frequency_score, no_note_streak, note_reasons = compute_note_frequency_score(
            action=action,
            previous_no_note_streak=previous_no_note_streak,
            cfg=cfg,
        )
    judge_result = ctx.get("judge_result")
    judge_raw_score = float(judge_result.score) if isinstance(judge_result, JudgeResult) else 0.0
    judge_score = judge_raw_score * float(cfg.score_scale)
    judge_blocked = judge_blocked_by_note_frequency(note_frequency_score=note_frequency_score, cfg=cfg)
    if judge_blocked:
        judge_score = 0.0
    step_score, components = compute_step_score(
        note_frequency_score=note_frequency_score,
        judge_score=judge_score,
        cfg=cfg,
    )
    turn_reward = (cfg.w_format * format_score + cfg.w_step * step_score) / denominator
    return StepRewardResult(
        format_score=format_score,
        step_score=step_score,
        note_frequency_score=note_frequency_score,
        judge_score=judge_score,
        turn_reward=turn_reward,
        info={
            "format_mode": cfg.format_mode,
            "process_reward_denominator": denominator,
            "num_notes": quality.metrics.get("num_notes", 0),
            "previous_no_note_streak": previous_no_note_streak,
            "no_note_streak": no_note_streak,
            "note_frequency_reasons": note_reasons,
            "step_components": components,
            "judge_blocked_by_note_frequency": judge_blocked,
            "judge_raw_score": judge_raw_score,
            "judge_status": judge_result.status if isinstance(judge_result, JudgeResult) else "disabled",
            "judge_scores": judge_result.scores if isinstance(judge_result, JudgeResult) else {},
            "judge_reasons": judge_result.reasons if isinstance(judge_result, JudgeResult) else {},
            "judge_issues": judge_result.issues if isinstance(judge_result, JudgeResult) else [],
            "judge_error": judge_result.error if isinstance(judge_result, JudgeResult) else "",
        },
    )


def compute_success_score(
    answer: str,
    ground_truth: Any,
    *,
    mode: str = "exact_or_contains",
    scorer: str = "auto",
    metadata: dict[str, Any] | None = None,
    score_scale: float = 2.0,
) -> float:
    if mode == "dataset":
        raw_score = score_answer(answer, ground_truth, scorer=scorer, metadata=metadata, fallback_mode="exact_or_contains")
        return _clamp_score(float(raw_score) * float(score_scale), score_scale=score_scale)
    answer_norm = (answer or "").strip().lower()
    gt_norm = str(ground_truth or "").strip().lower()
    if not answer_norm or not gt_norm:
        return 0.0
    if mode == "exact":
        return float(score_scale) if answer_norm == gt_norm else 0.0
    if mode == "contains":
        return float(score_scale) if gt_norm in answer_norm else 0.0
    if mode == "exact_or_contains":
        return float(score_scale) if answer_norm == gt_norm or gt_norm in answer_norm or answer_norm in gt_norm else 0.0
    raise ValueError(f"Unknown success reward mode: {mode}")


def compute_trajectory_reward(
    *,
    step_results: list[StepRewardResult],
    final_answer: str,
    ground_truth: Any,
    cfg: StreamWeaveRewardConfig,
    metadata: dict[str, Any] | None = None,
) -> TrajectoryRewardResult:
    if step_results:
        format_mean = sum(item.format_score for item in step_results) / len(step_results)
        step_mean = sum(item.step_score for item in step_results) / len(step_results)
    else:
        format_mean = 0.0
        step_mean = 0.0
    success_score = compute_success_score(
        final_answer,
        ground_truth,
        mode=cfg.success_mode,
        scorer=cfg.success_scorer,
        metadata=metadata,
        score_scale=cfg.score_scale,
    )
    trajectory_score = cfg.w_format * format_mean + cfg.w_step * step_mean + cfg.w_success * success_score
    return TrajectoryRewardResult(
        trajectory_score=trajectory_score,
        format_mean=format_mean,
        step_mean=step_mean,
        success_score=success_score,
        info={"success_mode": cfg.success_mode, "success_scorer": cfg.success_scorer},
    )


def _max_score(cfg: StreamWeaveRewardConfig | None = None, *, score_scale: float | None = None) -> float:
    if cfg is not None:
        score_scale = cfg.score_scale
    return max(float(score_scale if score_scale is not None else 2.0), 0.0)


def _clamp_score(
    value: float,
    cfg: StreamWeaveRewardConfig | None = None,
    *,
    score_scale: float | None = None,
) -> float:
    return min(max(float(value), 0.0), _max_score(cfg, score_scale=score_scale))
