"""Shared data structures for StreamWeave V5."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ContentType = Literal["text", "image"]
EventType = Literal["note", "bridge"]
QARole = Literal["q", "a"]


@dataclass(slots=True)
class ContentItem:
    type: ContentType
    text: str = ""
    image_path: Path | None = None


@dataclass(slots=True)
class FrameRef:
    video_id: str
    global_index: int
    start_time: float
    end_time: float
    image_path: Path
    step_local_id: int = 0


@dataclass(slots=True)
class QueryEvent:
    text: str
    timestamp: float


@dataclass(slots=True)
class BenchmarkSample:
    sample_id: str
    video_id: str
    video_path: str
    query_events: list[QueryEvent]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QARecord:
    timestamp: float
    text: str
    role: QARole = "q"


@dataclass(slots=True)
class NoteRecord:
    start_time: float
    end_time: float
    image_path: Path
    global_frame_index: int
    image_available: bool = True


@dataclass(slots=True)
class BridgeRecord:
    start_time: float
    end_time: float
    text: str


@dataclass(slots=True)
class ModelEvent:
    kind: EventType
    start_time: float
    end_time: float
    text: str = ""


@dataclass(slots=True)
class ModelAction:
    state: str
    answer: str
    events: list[ModelEvent]
    raw: str = ""
    state_present: bool = False
    answer_present: bool = False


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"


@dataclass(slots=True)
class RewardFeatures:
    format_reward: int = 0
    opentail_bridge_reward: int = 0
    note_bridge_timing_reward: int = 0
    operation_order_reward: int = 0


@dataclass(slots=True)
class QualityReport:
    valid: bool
    parser_ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    rewards: RewardFeatures = field(default_factory=RewardFeatures)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackendResult:
    text: str
    latency_seconds: float
    endpoint_id: str = ""
    attempt_count: int = 1
    retry_errors: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppliedAction:
    action: ModelAction
    notes: list[NoteRecord] = field(default_factory=list)
    bridges: list[BridgeRecord] = field(default_factory=list)
    answer: QARecord | None = None
    replace_open_tail: bool = False
    repair_count: int = 0
    repair_types: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AttemptRecord:
    attempt_index: int
    raw_output: str
    quality: QualityReport
    backend_result: BackendResult | None = None
    feedback: str = ""
    prompt_text: str = ""
    prompt_images: list[str] = field(default_factory=list)
    accepted: bool = False


@dataclass(slots=True)
class Transition:
    sample_id: str
    video_id: str
    step_index: int
    step_start: float
    step_end: float
    prompt_text: str
    prompt_images: list[str]
    current_frames: list[FrameRef]
    raw_action: ModelAction
    quality: QualityReport
    applied: AppliedAction
    backend_result: BackendResult
    memory_before: str
    memory_after: str
    base_prompt_text: str = ""
    base_prompt_images: list[str] = field(default_factory=list)
    attempt_prompt_text: str = ""
    attempt_prompt_images: list[str] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    accepted_attempt_index: int = 1
    target_raw_output: str = ""


@dataclass(slots=True)
class RolloutTrace:
    sample: BenchmarkSample
    transitions: list[Transition]
    task_failed: bool = False
    failure_reason: str = ""

    def final_answer(self) -> str:
        for transition in reversed(self.transitions):
            answer = transition.applied.action.answer.strip()
            if answer:
                return answer
        return ""
