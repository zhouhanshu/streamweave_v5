"""Lightweight schemas for StreamWeave RL rollout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StepRewardResult:
    format_score: float
    step_score: float = 0.0
    note_frequency_score: float = 0.0
    judge_score: float = 0.0
    turn_reward: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryRewardResult:
    trajectory_score: float
    format_mean: float
    step_mean: float
    success_score: float
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepRecord:
    turn_idx: int
    response_text: str
    prompt_text: str
    prompt_images: list[str]
    format_score: float
    step_score: float
    turn_reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
