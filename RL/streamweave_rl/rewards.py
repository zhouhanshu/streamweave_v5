"""Reward components for StreamWeave RL.

The first training version intentionally keeps reward simple:
format compliance plus final task success.  Step-level rewards are exposed as
hooks so richer timing/memory rewards can be added without changing rollout or
trainer plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streamweave.schemas import QualityReport

from .schemas import StepRewardResult, TrajectoryRewardResult


@dataclass(slots=True)
class StreamWeaveRewardConfig:
    w_format: float = 0.2
    w_success: float = 0.8
    w_step: float = 0.0
    format_mode: str = "valid"
    success_mode: str = "exact_or_contains"


def reward_config_from_mapping(data: dict[str, Any] | None) -> StreamWeaveRewardConfig:
    if not data:
        return StreamWeaveRewardConfig()
    allowed = {field for field in StreamWeaveRewardConfig.__dataclass_fields__}
    return StreamWeaveRewardConfig(**{key: value for key, value in data.items() if key in allowed})


def compute_format_score(quality: QualityReport, *, mode: str = "valid") -> float:
    if mode == "parser_ok":
        return 1.0 if quality.parser_ok else 0.0
    if mode == "format_reward":
        return float(quality.rewards.format_reward)
    if mode == "valid":
        return 1.0 if quality.valid else 0.0
    raise ValueError(f"Unknown format reward mode: {mode}")


def compute_step_score(ctx: dict[str, Any]) -> float:
    """Reserved hook for future dense rewards."""
    return 0.0


def compute_step_reward(
    *,
    quality: QualityReport,
    cfg: StreamWeaveRewardConfig,
    ctx: dict[str, Any] | None = None,
) -> StepRewardResult:
    ctx = ctx or {}
    format_score = compute_format_score(quality, mode=cfg.format_mode)
    step_score = compute_step_score({**ctx, "quality": quality})
    turn_reward = cfg.w_format * format_score + cfg.w_step * step_score
    return StepRewardResult(
        format_score=format_score,
        step_score=step_score,
        turn_reward=turn_reward,
        info={"format_mode": cfg.format_mode},
    )


def compute_success_score(answer: str, ground_truth: str, *, mode: str = "exact_or_contains") -> float:
    answer_norm = (answer or "").strip().lower()
    gt_norm = (ground_truth or "").strip().lower()
    if not answer_norm or not gt_norm:
        return 0.0
    if mode == "exact":
        return 1.0 if answer_norm == gt_norm else 0.0
    if mode == "contains":
        return 1.0 if gt_norm in answer_norm else 0.0
    if mode == "exact_or_contains":
        return 1.0 if answer_norm == gt_norm or gt_norm in answer_norm or answer_norm in gt_norm else 0.0
    raise ValueError(f"Unknown success reward mode: {mode}")


def compute_trajectory_reward(
    *,
    step_results: list[StepRewardResult],
    final_answer: str,
    ground_truth: str,
    cfg: StreamWeaveRewardConfig,
) -> TrajectoryRewardResult:
    if step_results:
        format_mean = sum(item.format_score for item in step_results) / len(step_results)
        step_mean = sum(item.step_score for item in step_results) / len(step_results)
    else:
        format_mean = 0.0
        step_mean = 0.0
    success_score = compute_success_score(final_answer, ground_truth, mode=cfg.success_mode)
    trajectory_score = cfg.w_format * format_mean + cfg.w_step * step_mean + cfg.w_success * success_score
    return TrajectoryRewardResult(
        trajectory_score=trajectory_score,
        format_mean=format_mean,
        step_mean=step_mean,
        success_score=success_score,
        info={"success_mode": cfg.success_mode},
    )
