"""Run V4 StreamWeave synthesis rollouts and emit step-level SFT rows."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from backend.base import BaseBackend, MockBackend
from streamweave.config import RewardConfig
from streamweave.env import StreamWeaveEnv
from streamweave.policies import make_policy
from streamweave.postprocess import synthesis_feedback
from streamweave.schemas import (
    AppliedAction,
    BackendResult,
    ContentItem,
    FrameRef as SWFrameRef,
    ModelAction,
    QARecord,
    QualityReport,
    ValidationIssue,
)

from .io_utils import JsonDict, media_path
from .schemas import FrameRef as PlanFrameRef
from .schemas import QueryPlan, SamplePlan

QA_TIME_TOLERANCE = 0.51


class SFTMockBackend(MockBackend):
    """Deterministic local backend for testing the SFT pipeline without model calls."""

    def generate(self, content: list[ContentItem], *, generate_kwargs=None) -> BackendResult:
        text = "".join(item.text for item in content if item.type == "text")
        actual = text.split("[Actual Input]", 1)[-1]
        current_block = actual
        if "=== Current frames ===" in actual:
            current_block = actual.split("=== Current frames ===", 1)[1].split("=== QA History ===", 1)[0]
        intervals = re.findall(r'<frame\s+id="(\d+)"\s+t="([0-9.]+)-([0-9.]+)">', current_block)
        if not intervals:
            output = "<eta></eta>\n<answer></answer>"
        else:
            frame_id, start, end = intervals[0]
            step_start = float(start)
            step_end = float(intervals[-1][2])
            memory_block = actual.split("=== Current frames ===", 1)[0]
            open_tail_start = _mock_open_tail_bridge_start(memory_block)
            has_question = re.search(r'<qa\b[^>]*role="q"[^>]*>', actual) is not None
            eta, answer = _mock_eta_answer_from_retry_feedback(text)
            if eta is None and answer is None:
                eta = ""
                answer = "A" if has_question else ""
            lines = ["<eta></eta>", f"<answer>{answer}</answer>"]
            if eta:
                lines[0] = f"<eta>{eta}</eta>"
            note_intervals = _mock_note_intervals_from_key_frame_context(text, intervals)
            if open_tail_start is not None:
                lines.extend(
                    _mock_observation_lines(
                        note_intervals,
                        bridge_start=open_tail_start,
                        step_end=step_end,
                        bridge_text="Mock observation extends the current open-tail bridge.",
                    )
                )
            else:
                if note_intervals:
                    selected_note_intervals = note_intervals
                elif "Annotated key-frame constraint:" not in text or "No annotated key frame is present" not in text:
                    selected_note_intervals = [(frame_id, start, end)]
                else:
                    selected_note_intervals = []
                lines.extend(
                    _mock_observation_lines(
                        selected_note_intervals,
                        bridge_start=step_start,
                        step_end=step_end,
                        bridge_text="Mock observation for the current streaming window.",
                    )
                )
            output = "\n".join(lines)
        return BackendResult(text=output, latency_seconds=0.0, endpoint_id="sft_mock", attempt_count=1)


def _mock_open_tail_bridge_start(memory_block: str) -> float | None:
    events = list(re.finditer(r'<(?P<kind>note|bridge)\b(?P<attrs>[^>]*)>', memory_block))
    if not events or events[-1].group("kind") != "bridge":
        return None
    match = re.search(r't="([0-9.]+)-([0-9.]+)"', events[-1].group("attrs"))
    if match is None:
        return None
    return float(match.group(1))


def _mock_eta_answer_from_retry_feedback(text: str) -> tuple[str | None, str | None]:
    if "=== Retry Feedback ===" not in text:
        return None, None
    feedback = text.split("=== Retry Feedback ===", 1)[1]
    template = re.search(
        r"Required prefix:\s*<eta>([^<]*)</eta>[^<]*<answer>([^<]*)</answer>",
        feedback,
        flags=re.DOTALL,
    )
    if template is None:
        return None, None
    eta_text = template.group(1).strip()
    answer_template = template.group(2).strip()
    if not eta_text:
        eta = ""
    elif eta_text == "T":
        window = re.search(r"T must be inside ([0-9.]+)-([0-9.]+)", feedback)
        eta = f"{float(window.group(1)) + 0.5:g}" if window else ""
    else:
        eta = eta_text
    if not answer_template:
        answer = ""
    elif answer_template.startswith("..."):
        answer = "A"
    else:
        answer = answer_template
    return eta, answer


def _mock_observation_lines(
    note_intervals: list[tuple[str, str, str]],
    *,
    bridge_start: float,
    step_end: float,
    bridge_text: str,
) -> list[str]:
    lines: list[str] = []
    cursor = bridge_start
    for frame_id, start_text, end_text in sorted(note_intervals, key=lambda item: float(item[1])):
        note_start = float(start_text)
        note_end = float(end_text)
        if note_start > cursor:
            lines.append(f'<bridge t="{cursor:.1f}-{note_start:.1f}">{bridge_text}</bridge>')
        lines.append(f'<note t="{note_start:.1f}-{note_end:.1f}" frame="{frame_id}"></note>')
        cursor = max(cursor, note_end)
    if step_end > cursor:
        lines.append(f'<bridge t="{cursor:.1f}-{step_end:.1f}">{bridge_text}</bridge>')
    if not lines:
        lines.append(f'<bridge t="{bridge_start:.1f}-{step_end:.1f}">{bridge_text}</bridge>')
    return lines


def _mock_note_intervals_from_key_frame_context(
    text: str,
    intervals: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    if "Annotated key-frame constraint:" not in text:
        return []
    constraint = text.split("Annotated key-frame constraint:", 1)[1].split("[Actual Input]", 1)[0]
    matches = re.findall(r'<frame\s+id="(\d+)"\s+t="[^"]+">', constraint)
    if not matches:
        return []
    preferred = set(matches)
    return [interval for interval in intervals if interval[0] in preferred]


@dataclass(slots=True)
class SFTSynthesisConfig:
    prompt_type: str = "teacher_synthesis"
    policy: str = "streamweave"
    frames_per_step: int = 5
    memory_window: float = 180.0
    max_steps: int = 0
    media_dir: Path = Path("dataset")
    keep_invalid: bool = False
    max_attempts: int = 3


@dataclass(slots=True)
class StepContext:
    plan_frames: list[PlanFrameRef]
    sw_frames: list[SWFrameRef]
    extra_context: str
    memory_before: list[JsonDict]
    memory_before_text: str
    qa_before: list[JsonDict]
    base_prompt_text: str
    base_prompt_images: list[str]
    base_local_frames: list[SWFrameRef]


@dataclass(slots=True)
class AcceptedAttempt:
    attempt_index: int
    content: list[ContentItem]
    backend_result: BackendResult
    raw_action: ModelAction
    quality: QualityReport
    applied: AppliedAction


def synthesize_sft_steps(
    samples: Iterable[SamplePlan],
    backend: BaseBackend,
    config: SFTSynthesisConfig,
) -> list[JsonDict]:
    return list(iter_sft_steps(samples, backend, config))


def iter_sft_sample_records(
    samples: Iterable[SamplePlan],
    backend: BaseBackend,
    config: SFTSynthesisConfig,
) -> Iterable[JsonDict]:
    policy = make_policy(config.policy)
    profile = _prompt_profile(config.prompt_type)
    reward_config = RewardConfig()
    for sample in samples:
        yield _run_sft_sample(
            sample=sample,
            backend=backend,
            config=config,
            policy=policy,
            profile=profile,
            reward_config=reward_config,
        )


def iter_sft_steps(
    samples: Iterable[SamplePlan],
    backend: BaseBackend,
    config: SFTSynthesisConfig,
) -> Iterable[JsonDict]:
    for sample_record in iter_sft_sample_records(samples, backend, config):
        if sample_record.get("usable_for_sft") or config.keep_invalid:
            for row in sample_record.get("steps", []):
                if config.keep_invalid or not row.get("task_failed"):
                    yield row


def _run_sft_sample(
    *,
    sample: SamplePlan,
    backend: BaseBackend,
    config: SFTSynthesisConfig,
    policy,
    profile: str,
    reward_config: RewardConfig,
) -> JsonDict:
    env = StreamWeaveEnv(
        prompt_profile=profile,
        policy=policy,
        memory_window=config.memory_window,
        extra_context=str(sample.metadata.get("teacher_context", "")),
    )
    query_by_frame = _query_events_by_frame(sample.frames, sample.query_events)
    groups = list(_group_frames(sample.frames, config.frames_per_step))
    if config.max_steps:
        groups = groups[: config.max_steps]

    steps: list[JsonDict] = []
    failure_step_index = None
    failure_reason = ""
    for step_index, group in enumerate(groups):
        for frame in group:
            for query in query_by_frame.get(frame.frame_index, []):
                env.add_question(QARecord(timestamp=query.timestamp, text=query.text, role="q"))
        env.evict_memory(group[-1].end_time)
        row = _run_sft_step(
            sample=sample,
            env=env,
            backend=backend,
            group=group,
            step_index=step_index,
            config=config,
            reward_config=reward_config,
        )
        steps.append(row)
        if row.get("task_failed"):
            failure_step_index = step_index
            failure_reason = str(row.get("failure_reason") or "synthesis_step_failed")
            break

    all_steps_valid = failure_step_index is None and len(steps) == len(groups) and all(not row.get("task_failed") for row in steps)
    answer_check = _check_sample_answer(sample, steps) if all_steps_valid else _check_sample_answer(sample, steps)
    answer_correct = bool(answer_check.get("answer_correct"))
    usable_for_sft = all_steps_valid and answer_correct
    if usable_for_sft:
        status = "accepted"
    else:
        status = "failed"
        if not failure_reason:
            failure_reason = "answer_incorrect" if not answer_correct else "sample_validation_failed"

    return {
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "qa_id": sample.qa_id,
        "question_type": sample.task,
        "status": status,
        "usable_for_sft": usable_for_sft,
        "answer_correct": answer_correct,
        "failure_reason": failure_reason,
        "failure_step_index": failure_step_index,
        "num_steps": len(steps),
        "num_expected_steps": len(groups),
        "checks": {
            "all_steps_valid": all_steps_valid,
            "no_empty_target": all(bool(str(row.get("target_xml") or "").strip()) for row in steps) if steps else False,
            "answer_correct": answer_correct,
            "answer_check": answer_check,
        },
        "metadata": {
            "answer_time": sample.answer_time,
            "annotation": sample.metadata,
        },
        "steps": steps,
    }


def _run_sft_step(
    *,
    sample: SamplePlan,
    env: StreamWeaveEnv,
    backend: BaseBackend,
    group: list[PlanFrameRef],
    step_index: int,
    config: SFTSynthesisConfig,
    reward_config: RewardConfig,
) -> JsonDict:
    context = _prepare_step_context(sample=sample, env=env, group=group, config=config)
    attempts, accepted = _run_teacher_attempts(
        sample=sample,
        env=env,
        backend=backend,
        context=context,
        config=config,
        reward_config=reward_config,
    )
    if accepted is not None:
        env.commit(accepted.applied)
        return _build_success_row(
            sample=sample,
            env=env,
            step_index=step_index,
            config=config,
            context=context,
            attempts=attempts,
            accepted=accepted,
        )

    return _build_failure_row(
        sample=sample,
        env=env,
        step_index=step_index,
        config=config,
        context=context,
        attempts=attempts,
    )


def _prepare_step_context(
    *,
    sample: SamplePlan,
    env: StreamWeaveEnv,
    group: list[PlanFrameRef],
    config: SFTSynthesisConfig,
) -> StepContext:
    sw_frames = [_to_sw_frame(sample.video_id, frame) for frame in group]
    qa_before = _qa_snapshot(env)
    extra_context = _key_frame_context(sample, group)
    _, base_prompt_text, base_prompt_images, base_local_frames = env.build_prompt(
        sw_frames,
        extra_context=extra_context,
    )
    return StepContext(
        plan_frames=group,
        sw_frames=sw_frames,
        extra_context=extra_context,
        memory_before=_memory_snapshot(env, media_dir=config.media_dir),
        memory_before_text=env.memory_text(),
        qa_before=qa_before,
        base_prompt_text=base_prompt_text,
        base_prompt_images=base_prompt_images,
        base_local_frames=base_local_frames,
    )


def _memory_tail_signals(context: "StepContext") -> tuple[str | None, float | None, float | None]:
    """Return (last memory event kind, its end time, open-tail bridge start if applicable)."""
    if not context.memory_before:
        return None, None, None
    last = context.memory_before[-1]
    kind = str(last.get("type") or "") or None
    interval = last.get("t") or []
    end_time = float(interval[1]) if len(interval) >= 2 else None
    open_tail_start = float(interval[0]) if kind == "bridge" and len(interval) >= 1 else None
    return kind, end_time, open_tail_start


def _retry_generate_kwargs(attempt_index: int) -> dict[str, float] | None:
    """Bump sampling diversity on retries to avoid deterministic re-emission of the same wrong output."""
    if attempt_index <= 1:
        return None
    if attempt_index == 2:
        return {"temperature": 0.4, "top_p": 0.9}
    return {"temperature": 0.7, "top_p": 0.95}


def _run_teacher_attempts(
    *,
    sample: SamplePlan,
    env: StreamWeaveEnv,
    backend: BaseBackend,
    context: StepContext,
    config: SFTSynthesisConfig,
    reward_config: RewardConfig,
) -> tuple[list[JsonDict], AcceptedAttempt | None]:
    attempts: list[JsonDict] = []
    retry_feedback = ""
    max_attempts = max(1, int(config.max_attempts))

    for attempt_index in range(1, max_attempts + 1):
        content, prompt_text, prompt_images, local_frames = env.build_prompt(
            context.sw_frames,
            retry_feedback=retry_feedback,
            extra_context=context.extra_context,
        )
        try:
            backend_result = backend.generate(content, generate_kwargs=_retry_generate_kwargs(attempt_index))
        except Exception as exc:
            quality = QualityReport(
                valid=False,
                parser_ok=False,
                issues=[ValidationIssue("backend_generate_error", _short_error(exc))],
            )
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "accepted": False,
                    "raw_output": "",
                    "quality": _quality_to_json(quality),
                    "feedback": "",
                    "prompt_text": prompt_text,
                    "prompt_images": [media_path(Path(path), config.media_dir) for path in prompt_images],
                    "backend_result": asdict(
                        BackendResult(
                            text="",
                            latency_seconds=0.0,
                            endpoint_id=backend.__class__.__name__,
                            attempt_count=0,
                            retry_errors=[_short_error(exc)],
                        )
                    ),
                }
            )
            if _is_non_retryable_backend_error(exc):
                return attempts, None
            continue
        raw_action, quality, applied = env.evaluate_attempt(
            backend_result.text,
            frames=local_frames,
            reward_config=reward_config,
            repair=False,
        )
        _apply_key_frame_quality_constraints(
            quality=quality,
            raw_action=raw_action,
            context=context,
            sample=sample,
        )
        _apply_qa_eta_answer_constraints(
            quality=quality,
            raw_action=raw_action,
            context=context,
            sample=sample,
            frames_per_step=config.frames_per_step,
        )
        accepted = quality.valid
        if accepted:
            feedback = ""
        else:
            tail_kind, tail_end, open_tail_start = _memory_tail_signals(context)
            step_start = context.sw_frames[0].start_time if context.sw_frames else None
            step_end = context.sw_frames[-1].end_time if context.sw_frames else None
            feedback = synthesis_feedback(
                quality.issues,
                backend_result.text,
                step_start=step_start,
                step_end=step_end,
                memory_tail_kind=tail_kind,
                memory_tail_end=tail_end,
                open_tail_start=open_tail_start,
            )
        attempts.append(
            {
                "attempt_index": attempt_index,
                "accepted": accepted,
                "raw_output": backend_result.text,
                "quality": _quality_to_json(quality),
                "feedback": feedback,
                "prompt_text": prompt_text,
                "prompt_images": [media_path(Path(path), config.media_dir) for path in prompt_images],
                "backend_result": asdict(backend_result),
            }
        )
        if accepted:
            return attempts, AcceptedAttempt(
                attempt_index=attempt_index,
                content=content,
                backend_result=backend_result,
                raw_action=raw_action,
                quality=quality,
                applied=applied,
            )
        retry_feedback = feedback

    return attempts, None


def _build_success_row(
    *,
    sample: SamplePlan,
    env: StreamWeaveEnv,
    step_index: int,
    config: SFTSynthesisConfig,
    context: StepContext,
    attempts: list[JsonDict],
    accepted: AcceptedAttempt,
) -> JsonDict:
    prompt_content, prompt_images = _render_prompt_for_sharegpt(accepted.content, media_dir=config.media_dir)
    base_content, base_images = _text_images_to_sharegpt(
        context.base_prompt_text,
        context.base_prompt_images,
        media_dir=config.media_dir,
    )
    return {
        "sample_id": f"{sample.sample_id}_step_{step_index:04d}",
        "video_id": sample.video_id,
        "qa_id": sample.qa_id,
        "question_type": sample.task,
        "step_index": step_index,
        "step_start": context.plan_frames[0].start_time,
        "step_end": context.plan_frames[-1].end_time,
        "memory_before": context.memory_before,
        "memory_before_text": context.memory_before_text,
        "qa_history": context.qa_before,
        "current_frames": current_frames_snapshot(context.plan_frames, media_dir=config.media_dir),
        "raw_teacher_xml": accepted.backend_result.text,
        "target_xml": accepted.backend_result.text,
        "target_raw_output": accepted.backend_result.text,
        "quality": {
            "raw": _quality_to_json(accepted.quality),
            "target": _quality_to_json(accepted.quality),
            "latency_seconds": accepted.backend_result.latency_seconds,
            "backend_attempt_count": accepted.backend_result.attempt_count,
            "backend_retry_errors": accepted.backend_result.retry_errors,
        },
        "prompt": {
            "prompt_type": profile_label(config.prompt_type),
            "policy": env.policy.name,
            "content": prompt_content,
            "images": prompt_images,
            "base_content": base_content,
            "base_images": base_images,
        },
        "attempts": attempts,
        "accepted_attempt_index": accepted.attempt_index,
        "task_failed": False,
        "metadata": {
            "answer_time": sample.answer_time,
            "annotation": sample.metadata,
            "base_current_frames": current_frames_snapshot_from_sw(context.base_local_frames, media_dir=config.media_dir),
            "raw_action": asdict(accepted.raw_action),
            "applied": _json_safe(asdict(accepted.applied)),
        },
    }


def _build_failure_row(
    *,
    sample: SamplePlan,
    env: StreamWeaveEnv,
    step_index: int,
    config: SFTSynthesisConfig,
    context: StepContext,
    attempts: list[JsonDict],
) -> JsonDict:
    prompt_content, prompt_images = _text_images_to_sharegpt(
        context.base_prompt_text,
        context.base_prompt_images,
        media_dir=config.media_dir,
    )
    return {
        "sample_id": f"{sample.sample_id}_step_{step_index:04d}",
        "video_id": sample.video_id,
        "qa_id": sample.qa_id,
        "question_type": sample.task,
        "step_index": step_index,
        "step_start": context.plan_frames[0].start_time,
        "step_end": context.plan_frames[-1].end_time,
        "memory_before": context.memory_before,
        "qa_history": context.qa_before,
        "current_frames": current_frames_snapshot(context.plan_frames, media_dir=config.media_dir),
        "raw_teacher_xml": attempts[-1]["raw_output"] if attempts else "",
        "target_xml": "",
        "quality": {"attempts": attempts},
        "prompt": {
            "prompt_type": profile_label(config.prompt_type),
            "policy": env.policy.name,
            "content": prompt_content,
            "images": prompt_images,
        },
        "attempts": attempts,
        "task_failed": True,
        "failure_reason": f"synthesis_raw_retry_failed_at_step_{step_index}",
        "metadata": {"answer_time": sample.answer_time, "annotation": sample.metadata},
    }


def current_frames_snapshot(frames: list[PlanFrameRef], *, media_dir: Path) -> list[JsonDict]:
    return [
        {
            "frame_id": idx,
            "global_frame_id": frame.global_frame_id,
            "t": [frame.start_time, frame.end_time],
            "image_path": media_path(frame.image_path, media_dir),
        }
        for idx, frame in enumerate(frames, start=1)
    ]


def current_frames_snapshot_from_sw(frames: list[SWFrameRef], *, media_dir: Path) -> list[JsonDict]:
    return [
        {
            "frame_id": frame.step_local_id,
            "global_frame_id": frame.global_index,
            "t": [frame.start_time, frame.end_time],
            "image_path": media_path(frame.image_path, media_dir),
        }
        for frame in frames
    ]


def _memory_snapshot(env: StreamWeaveEnv, *, media_dir: Path) -> list[JsonDict]:
    events: list[tuple[float, int, JsonDict]] = []
    for bridge in env.memory.bridges:
        events.append((bridge.start_time, 0, {"type": "bridge", "t": [bridge.start_time, bridge.end_time], "text": bridge.text}))
    for note in env.memory.notes:
        events.append(
            (
                note.start_time,
                1,
                {
                    "type": "note",
                    "t": [note.start_time, note.end_time],
                    "global_frame_id": note.global_frame_index,
                    "image_path": media_path(note.image_path, media_dir),
                    "image_available": note.image_available,
                },
            )
        )
    return [item for _, _, item in sorted(events, key=lambda item: (item[0], item[1]))]


def _qa_snapshot(env: StreamWeaveEnv) -> list[JsonDict]:
    return [
        {"t": qa.timestamp, "role": qa.role, "text": qa.text}
        for qa in sorted(env.memory.qa_history, key=lambda item: item.timestamp)
    ]


def _render_prompt_for_sharegpt(content: list[ContentItem], *, media_dir: Path) -> tuple[str, list[str]]:
    rendered: list[str] = []
    images: list[str] = []
    for item in content:
        if item.type == "text":
            rendered.append(item.text)
        elif item.type == "image" and item.image_path is not None:
            rendered.append("<image>")
            images.append(media_path(item.image_path, media_dir))
    return "".join(rendered), images


def _text_images_to_sharegpt(prompt_text: str, prompt_images: list[str], *, media_dir: Path) -> tuple[str, list[str]]:
    text = re.sub(r"<image:[^>]+>", "<image>", prompt_text)
    return text, [media_path(Path(path), media_dir) for path in prompt_images]


def _quality_to_json(quality) -> JsonDict:
    return {
        "valid": quality.valid,
        "parser_ok": quality.parser_ok,
        "issues": [asdict(issue) for issue in quality.issues],
        "rewards": asdict(quality.rewards),
        "metrics": dict(quality.metrics),
    }


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _short_error(exc: Exception, limit: int = 1600) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _is_non_retryable_backend_error(exc: Exception) -> bool:
    text = _short_error(exc, limit=4000)
    markers = (
        "PROHIBITED_CONTENT",
        "prompt is blocked",
        "BlockedReason",
        "FinishReason.SAFETY",
        "Unauthorized",
        "Forbidden",
        "invalid_request",
        "context_length",
        "maximum context",
        "image too large",
    )
    return any(marker in text for marker in markers)


def _to_sw_frame(video_id: str, frame: PlanFrameRef) -> SWFrameRef:
    return SWFrameRef(
        video_id=video_id,
        global_index=frame.global_frame_id,
        start_time=frame.start_time,
        end_time=frame.end_time,
        image_path=frame.image_path,
    )


def _key_frame_context(sample: SamplePlan, frames: list[PlanFrameRef]) -> str:
    key_frame_ids = _annotation_key_frame_ids(sample.metadata)
    if not key_frame_ids:
        return ""
    matches = [
        (local_id, frame)
        for local_id, frame in enumerate(frames, start=1)
        if frame.global_frame_id in key_frame_ids
    ]
    lines = [
        "Annotated key-frame constraint:",
    ]
    if matches:
        current = "; ".join(
            f'<frame id="{local_id}" t="{frame.start_time:.1f}-{frame.end_time:.1f}">'
            for local_id, frame in matches
        )
        required_ids = ", ".join(f'"{local_id}"' for local_id, _ in matches)
        lines.extend(
            [
                "- The annotation key_frame_ids have already been converted to current frame ids and time ranges for this step.",
                f"- Required annotated current frames: {current}.",
                f"- MANDATORY: You must output exactly one <note> tag for each required frame id: {required_ids}.",
                f"- The complete set of <note> frame attributes in this step must be exactly: {required_ids}. No missing annotated frames and no extra frames.",
                "- NEVER use unannotated current frames for <note> tags.",
                "- This key-frame constraint overrides generic examples, mandatory initialization, QA-related anchoring, state-change anchoring, and representative selection.",
            ]
        )
    else:
        lines.extend(
            [
                "- No annotated key frame is present in this current window.",
                "- MANDATORY: Do not output any <note> tag in this step.",
                "- Use bridge-only observation for this step.",
                "- This key-frame constraint overrides generic examples, mandatory initialization, QA-related anchoring, state-change anchoring, and representative selection.",
            ]
        )
    return "\n".join(lines)


def _apply_key_frame_quality_constraints(
    *,
    quality: QualityReport,
    raw_action: ModelAction,
    context: StepContext,
    sample: SamplePlan,
) -> None:
    key_frame_ids = _annotation_key_frame_ids(sample.metadata)
    if not key_frame_ids:
        return

    required_note_ids = {
        local_id
        for local_id, frame in enumerate(context.plan_frames, start=1)
        if frame.global_frame_id in key_frame_ids
    }
    note_frame_ids = [
        event.frame_index + 1
        for event in raw_action.events
        if event.kind == "note" and event.frame_index is not None
    ]
    note_frame_set = set(note_frame_ids)
    issue_count = len(quality.issues)

    if required_note_ids:
        missing = sorted(required_note_ids - note_frame_set)
        extra = sorted(note_frame_set - required_note_ids)
        duplicated = sorted({frame_id for frame_id in note_frame_ids if note_frame_ids.count(frame_id) > 1})
        if missing or extra or duplicated:
            quality.issues.append(
                ValidationIssue(
                    "annotated_key_frame_note_mismatch",
                    (
                        "Annotated key-frame note mismatch. "
                        f"Required note frame ids: {sorted(required_note_ids)}. "
                        f"Output note frame ids: {note_frame_ids}. "
                        f"Missing: {missing}. Extra/unannotated: {extra}. Duplicated: {duplicated}. "
                        "Retry with exactly one <note> for each required id and no other <note> tags."
                    ),
                )
            )
    elif note_frame_ids:
        quality.issues.append(
            ValidationIssue(
                "annotated_key_frame_note_mismatch",
                (
                    "Annotated key-frame note mismatch. "
                    f"No annotated key frame is present in this current window, but output note frame ids are {note_frame_ids}. "
                    "Retry without any <note> tags and use bridge-only observation."
                ),
            )
        )

    if len(quality.issues) != issue_count:
        quality.valid = False
        quality.rewards.note_bridge_timing_reward = 0
    quality.metrics["required_annotated_note_frame_ids"] = sorted(required_note_ids)
    quality.metrics["output_note_frame_ids"] = note_frame_ids


def _apply_qa_eta_answer_constraints(
    *,
    quality: QualityReport,
    raw_action: ModelAction,
    context: StepContext,
    sample: SamplePlan,
    frames_per_step: int,
) -> None:
    expected = _expected_qa_output(sample, context.qa_before, context.plan_frames, frames_per_step)
    if expected is None:
        return

    expected_eta = expected["eta"]
    expected_eta_window = expected.get("eta_window")
    actual_eta = raw_action.eta
    actual_answer = raw_action.answer.strip()
    eta_ok = _eta_matches(actual_eta, expected_eta, expected_eta_window)
    answer_required = bool(expected["answer_required"])
    if answer_required:
        answer_ok = bool(actual_answer)
    else:
        answer_ok = actual_answer == ""

    quality.metrics["expected_eta"] = expected_eta
    quality.metrics["expected_eta_window"] = expected_eta_window
    quality.metrics["expected_answer_required"] = answer_required
    quality.metrics["qa_schedule_reason"] = expected["reason"]

    if eta_ok and answer_ok:
        return

    template = _format_eta_answer_template(expected)
    actual_summary = (
        f"actual: <eta>{'' if actual_eta is None else actual_eta}</eta>, "
        f"answer={_actual_answer_state(actual_answer)}"
    )
    reason = expected.get("reason") or ""
    quality.issues.append(
        ValidationIssue(
            "qa_eta_answer_mismatch",
            (
                "QA eta/answer mismatch. "
                f"Required prefix:\n{template}\n"
                f"Reason: {reason}. {actual_summary}."
            ),
        )
    )
    quality.valid = False
    quality.rewards.format_reward = 0


def _check_sample_answer(sample: SamplePlan, steps: list[JsonDict]) -> JsonDict:
    expected = _expected_sample_answer(sample)
    model_answers = _sample_model_answers(steps)
    if not expected["applicable"]:
        correct = not model_answers
        return {
            **expected,
            "model_answers": model_answers,
            "answer_correct": correct,
            "reason": "no GT answer is available; accepted only if the sample emitted no answer" if correct else "sample emitted an answer but no GT answer is available",
        }

    matches = [_answer_matches(item["answer"], expected) for item in model_answers]
    correct = bool(model_answers) and all(matches)
    return {
        **expected,
        "model_answers": model_answers,
        "answer_correct": correct,
        "reason": "all emitted answers match GT" if correct else "missing answer or emitted answer does not match GT",
    }


def _expected_sample_answer(sample: SamplePlan) -> JsonDict:
    metadata = sample.metadata
    options = metadata.get("options")
    if not isinstance(options, list):
        options = []
    options = [str(option).strip() for option in options]
    answer_text = str(metadata.get("answer") or "").strip()
    gt = metadata.get("gt")
    option_index = _expected_option_index(gt, options, answer_text)
    if option_index is not None:
        return {
            "applicable": True,
            "gt": gt,
            "expected_option_index": option_index,
            "expected_letter": chr(ord("A") + option_index),
            "expected_answer": options[option_index],
            "options": options,
        }
    if answer_text:
        return {
            "applicable": True,
            "gt": gt,
            "expected_option_index": None,
            "expected_letter": "",
            "expected_answer": answer_text,
            "options": options,
        }
    return {
        "applicable": False,
        "gt": gt,
        "expected_option_index": None,
        "expected_letter": "",
        "expected_answer": "",
        "options": options,
    }


def _expected_option_index(gt: object, options: list[str], answer_text: str) -> int | None:
    if not options:
        return None
    if isinstance(gt, str):
        text = gt.strip()
        if len(text) == 1 and text.upper().isalpha():
            index = ord(text.upper()) - ord("A")
            return index if 0 <= index < len(options) else None
        try:
            raw_index = int(text)
        except ValueError:
            normalized = _normalize_answer_text(text)
            for index, option in enumerate(options):
                if _normalize_answer_text(option) == normalized:
                    return index
            return None
    else:
        try:
            raw_index = int(gt)
        except (TypeError, ValueError):
            return None

    answer_norm = _normalize_answer_text(answer_text)
    zero_based_ok = 0 <= raw_index < len(options)
    one_based_ok = 1 <= raw_index <= len(options)
    if answer_norm:
        if zero_based_ok and _normalize_answer_text(options[raw_index]) == answer_norm:
            return raw_index
        if one_based_ok and _normalize_answer_text(options[raw_index - 1]) == answer_norm:
            return raw_index - 1
    if zero_based_ok:
        return raw_index
    if one_based_ok:
        return raw_index - 1
    return None


def _sample_model_answers(steps: list[JsonDict]) -> list[JsonDict]:
    answers: list[JsonDict] = []
    for row in steps:
        metadata = row.get("metadata") or {}
        raw_action = metadata.get("raw_action") if isinstance(metadata, dict) else {}
        answer = ""
        if isinstance(raw_action, dict):
            answer = str(raw_action.get("answer") or "").strip()
        if not answer:
            answer = _extract_answer_from_xml(str(row.get("target_xml") or ""))
        if answer:
            answers.append({"step_index": row.get("step_index"), "answer": answer})
    return answers


def _extract_answer_from_xml(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return (match.group(1).strip() if match else "")


def _answer_matches(answer: str, expected: JsonDict) -> bool:
    answer_norm = _normalize_answer_text(answer)
    expected_letter = str(expected.get("expected_letter") or "").strip().upper()
    if expected_letter:
        letter_match = re.match(r"^(?:option\s*)?([A-Z])(?:[).:]|\s|$)", answer.strip(), flags=re.IGNORECASE)
        if letter_match and letter_match.group(1).upper() == expected_letter:
            return True
    expected_answer = _normalize_answer_text(str(expected.get("expected_answer") or ""))
    return bool(expected_answer and answer_norm == expected_answer)


def _normalize_answer_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"^(?:option\s*)?[a-z][).:]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .。")


def _annotation_key_frame_ids(metadata: JsonDict) -> set[int]:
    raw = metadata.get("selected_key_frame_ids") or metadata.get("key_frame_ids") or []
    if not isinstance(raw, list):
        return set()
    out: set[int] = set()
    for item in raw:
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _expected_qa_output(sample: SamplePlan, qa_before: list[JsonDict], frames: list[PlanFrameRef], frames_per_step: int = 1) -> JsonDict | None:
    task = sample.task.lower().strip()
    has_question = any(str(item.get("role") or "") == "q" for item in qa_before if isinstance(item, dict))
    has_answer = any(str(item.get("role") or "") == "a" for item in qa_before if isinstance(item, dict))
    if not has_question:
        return _qa_expectation(None, False, "no question is present in QA History, so eta and answer must stay empty")
    if has_answer:
        return _qa_expectation(None, False, "the active question has already been answered in QA History, so eta and answer must stay empty")
    if task in {"realtime", "backward"}:
        eta_start, eta_end = _target_window_for_time(_first_query_time(sample, qa_before), sample, frames, frames_per_step)
        return _qa_expectation(
            eta_end,
            True,
            f"{task} question is active and unanswered, so answer now in the target window",
            eta_window=(eta_start, eta_end),
        )
    if task == "forward":
        clue_time = _forward_clue_time(sample)
        if clue_time is None:
            return None
        eta_start, eta_end = _target_window_for_time(clue_time, sample, frames, frames_per_step)
        due = frames[-1].end_time >= eta_end - QA_TIME_TOLERANCE if frames else False
        if due:
            return _qa_expectation(
                eta_end,
                True,
                "forward question has reached the clue window, so output eta inside the target window and answer now",
                eta_window=(eta_start, eta_end),
            )
        return _qa_expectation(
            eta_end,
            False,
            "forward question is active before clue_time, so predict an eta inside the clue window and keep answer empty",
            eta_window=(eta_start, eta_end),
        )
    return None


def _qa_expectation(
    eta: float | None,
    answer_required: bool,
    reason: str,
    eta_window: tuple[float, float] | None = None,
) -> JsonDict:
    return {
        "eta": eta,
        "eta_window": [eta_window[0], eta_window[1]] if eta_window is not None else None,
        "answer_required": answer_required,
        "reason": reason,
    }


def _answer_retry_state(expected: JsonDict) -> str:
    return "non-empty" if expected["answer_required"] else "empty"


def _actual_answer_state(actual_answer: str) -> str:
    return "non-empty" if actual_answer else "empty"


def _first_query_time(sample: SamplePlan, qa_before: list[JsonDict]) -> float:
    if sample.query_time is not None:
        return float(sample.query_time)
    if sample.query_events:
        return float(sample.query_events[0].timestamp)
    for item in qa_before:
        if isinstance(item, dict) and str(item.get("role") or "") == "q":
            return float(item.get("t", item.get("timestamp", 0.0)) or 0.0)
    return 0.0


def _forward_clue_time(sample: SamplePlan) -> float | None:
    for key in ("clue_time", "answer_time"):
        if key in sample.metadata and sample.metadata[key] is not None:
            return float(sample.metadata[key])
    return sample.answer_time


def _target_window_for_time(value: float, sample: SamplePlan, frames: list[PlanFrameRef], frames_per_step: int = 1) -> tuple[float, float]:
    all_frames = sample.frames or frames
    if not all_frames:
        target = float(value)
        return target, target
    target = float(value)
    window_size = max(1, frames_per_step)
    target_index = len(all_frames) - 1
    for index, frame in enumerate(all_frames):
        if target < frame.start_time:
            target_index = index
            break
        if frame.start_time <= target < frame.end_time:
            target_index = index
            break
    window_start = (target_index // window_size) * window_size
    window_end = min(window_start + window_size, len(all_frames))
    return float(all_frames[window_start].start_time), float(all_frames[window_end - 1].end_time)


def _eta_matches(actual: float | None, expected: float | None, expected_window: object | None = None) -> bool:
    if expected is None:
        return actual is None
    if actual is None:
        return False
    if isinstance(expected_window, (list, tuple)) and len(expected_window) >= 2:
        start = float(expected_window[0])
        end = float(expected_window[1])
        return start - QA_TIME_TOLERANCE <= float(actual) <= end + QA_TIME_TOLERANCE
    return abs(float(actual) - float(expected)) <= QA_TIME_TOLERANCE


def _format_expected_eta(value: float | None) -> str:
    return "" if value is None else f"{float(value):.1f}"


def _format_eta_answer_template(expected: JsonDict) -> str:
    eta = expected.get("eta")
    window = expected.get("eta_window")
    answer_required = bool(expected.get("answer_required"))
    if eta is None:
        eta_line = "<eta></eta>"
    elif isinstance(window, (list, tuple)) and len(window) >= 2:
        eta_line = f"<eta>T</eta>   (T must be inside {_format_eta_window(window)})"
    else:
        eta_line = f"<eta>{_format_expected_eta(float(eta))}</eta>"
    if answer_required:
        answer_line = "<answer>...your answer based on QA History, Memory, and Current frames...</answer>"
    else:
        answer_line = "<answer></answer>"
    return f"{eta_line}\n{answer_line}"


def _format_expected_eta_requirement(expected: JsonDict) -> str:
    eta = expected.get("eta")
    window = expected.get("eta_window")
    if eta is None:
        return "leave <eta></eta> empty"
    if isinstance(window, (list, tuple)) and len(window) >= 2:
        return f"set <eta> to any timestamp inside {_format_eta_window(window)}"
    return f"set <eta> to {_format_expected_eta(float(eta))}"


def _format_eta_window(window: object) -> str:
    if isinstance(window, (list, tuple)) and len(window) >= 2:
        return f"{float(window[0]):.1f}-{float(window[1]):.1f}"
    return ""


def _query_events_by_frame(frames: list[PlanFrameRef], query_events: list[QueryPlan]) -> dict[int, list[QueryPlan]]:
    if not frames:
        return {}
    out: dict[int, list] = {}
    for query in sorted(query_events, key=lambda item: item.timestamp):
        frame_index = _query_frame_index(frames, query.timestamp)
        out.setdefault(frame_index, []).append(query)
    return out


def _query_frame_index(frames: list[PlanFrameRef], query_time: float) -> int:
    for frame in frames:
        if frame.start_time <= query_time < frame.end_time:
            return frame.frame_index
    if query_time < frames[0].start_time:
        return frames[0].frame_index
    return frames[-1].frame_index


def _group_frames(frames: list[PlanFrameRef], size: int) -> list[list[PlanFrameRef]]:
    size = max(1, int(size))
    return [frames[idx : idx + size] for idx in range(0, len(frames), size)]


def _prompt_profile(prompt_type: str) -> str:
    if prompt_type in {"teacher", "teacher_synthesis"}:
        return "teacher_synthesis"
    if prompt_type == "teacher_eval":
        return "teacher_eval"
    return prompt_type


def profile_label(prompt_type: str) -> str:
    return _prompt_profile(prompt_type)
