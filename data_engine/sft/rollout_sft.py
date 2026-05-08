"""Run StreamWeave synthesis rollouts and emit step-level SFT rows."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from backend.base import BaseBackend
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

from .constraints import (
    apply_note_count_constraint,
    apply_qa_answer_constraints,
    check_sample_answer,
    note_reminder_context,
)
from .io_utils import JsonDict, media_path
from .schemas import FrameRef as PlanFrameRef
from .schemas import SamplePlan
from .timing import (
    group_frames,
    query_events_by_frame,
    sample_target_timestamp,
    truncate_plan_frames_at_timestamp,
)


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
    max_notes_per_step: int = 1
    bridge_note_reminder_seconds: float = 20.0
    answer_step_rollouts: int = 5
    answer_step_temperature: float = 0.3
    answer_step_top_p: float = 0.95


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
    note_reminder: str
    latest_bridge_seconds: float | None


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
    frames = truncate_plan_frames_at_timestamp(sample.frames, sample_target_timestamp(sample))
    query_by_frame = query_events_by_frame(frames, sample.query_events)
    groups = list(group_frames(frames, config.frames_per_step))
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
    answer_check = check_sample_answer(sample, steps) if all_steps_valid else check_sample_answer(sample, steps)
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
        answer_variants = _sample_answer_variants(
            sample=sample,
            env=env,
            backend=backend,
            context=context,
            config=config,
            reward_config=reward_config,
        ) if accepted.raw_action.answer.strip() else []
        env.commit(accepted.applied)
        return _build_success_row(
            sample=sample,
            env=env,
            step_index=step_index,
            config=config,
            context=context,
            attempts=attempts,
            accepted=accepted,
            answer_variants=answer_variants,
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
    latest_bridge_seconds = _latest_open_tail_bridge_seconds(env)
    note_reminder = note_reminder_context(latest_bridge_seconds, config.bridge_note_reminder_seconds)
    extra_context = note_reminder
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
        note_reminder=note_reminder,
        latest_bridge_seconds=latest_bridge_seconds,
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


def _latest_open_tail_bridge_seconds(env: StreamWeaveEnv) -> float | None:
    bridge = env.memory.open_tail_bridge()
    if bridge is None:
        return None
    return max(0.0, float(bridge.end_time) - float(bridge.start_time))


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
        _apply_sft_constraints(
            quality=quality,
            raw_action=raw_action,
            context=context,
            sample=sample,
            config=config,
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


def _sample_answer_variants(
    *,
    sample: SamplePlan,
    env: StreamWeaveEnv,
    backend: BaseBackend,
    context: StepContext,
    config: SFTSynthesisConfig,
    reward_config: RewardConfig,
) -> list[JsonDict]:
    num_variants = max(0, int(config.answer_step_rollouts))
    if num_variants <= 0:
        return []

    variants: list[JsonDict] = []
    generate_kwargs = {
        "temperature": float(config.answer_step_temperature),
        "top_p": float(config.answer_step_top_p),
    }
    for variant_index in range(1, num_variants + 1):
        content, prompt_text, prompt_images, local_frames = env.build_prompt(
            context.sw_frames,
            extra_context=context.extra_context,
        )
        try:
            backend_result = backend.generate(content, generate_kwargs=generate_kwargs)
        except Exception as exc:
            quality = QualityReport(
                valid=False,
                parser_ok=False,
                issues=[ValidationIssue("backend_generate_error", _short_error(exc))],
            )
            variants.append(
                {
                    "variant_index": variant_index,
                    "accepted": False,
                    "answer": "",
                    "raw_teacher_xml": "",
                    "target_xml": "",
                    "quality": _quality_to_json(quality),
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
            continue

        raw_action, quality, _ = env.evaluate_attempt(
            backend_result.text,
            frames=local_frames,
            reward_config=reward_config,
            repair=False,
        )
        _apply_sft_constraints(
            quality=quality,
            raw_action=raw_action,
            context=context,
            sample=sample,
            config=config,
        )
        answer = raw_action.answer.strip()
        accepted = quality.valid and bool(answer)
        variants.append(
            {
                "variant_index": variant_index,
                "accepted": accepted,
                "answer": answer,
                "raw_teacher_xml": backend_result.text,
                "target_xml": backend_result.text if accepted else "",
                "quality": _quality_to_json(quality),
                "prompt_text": prompt_text,
                "prompt_images": [media_path(Path(path), config.media_dir) for path in prompt_images],
                "backend_result": asdict(backend_result),
                "raw_action": asdict(raw_action),
            }
        )
    return variants


def _apply_sft_constraints(
    *,
    quality: QualityReport,
    raw_action: ModelAction,
    context: StepContext,
    sample: SamplePlan,
    config: SFTSynthesisConfig,
) -> None:
    apply_note_count_constraint(
        quality=quality,
        raw_action=raw_action,
        max_notes_per_step=config.max_notes_per_step,
    )
    apply_qa_answer_constraints(
        quality=quality,
        raw_action=raw_action,
        qa_before=context.qa_before,
        plan_frames=context.plan_frames,
        sample=sample,
        frames_per_step=config.frames_per_step,
    )


def _build_success_row(
    *,
    sample: SamplePlan,
    env: StreamWeaveEnv,
    step_index: int,
    config: SFTSynthesisConfig,
    context: StepContext,
    attempts: list[JsonDict],
    accepted: AcceptedAttempt,
    answer_variants: list[JsonDict],
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
        "answer_variants": answer_variants,
        "accepted_attempt_index": accepted.attempt_index,
        "task_failed": False,
        "metadata": {
            "answer_time": sample.answer_time,
            "annotation": sample.metadata,
            "base_current_frames": current_frames_snapshot_from_sw(context.base_local_frames, media_dir=config.media_dir),
            "note_count": _note_count(accepted.raw_action),
            "note_reminder": context.note_reminder,
            "latest_bridge_seconds": context.latest_bridge_seconds,
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
            "t": [frame.start_time, frame.end_time],
            "image_path": media_path(frame.image_path, media_dir),
        }
        for frame in frames
    ]


def current_frames_snapshot_from_sw(frames: list[SWFrameRef], *, media_dir: Path) -> list[JsonDict]:
    return [
        {
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


def _note_count(action: ModelAction) -> int:
    return sum(1 for event in action.events if event.kind == "note")


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


def _prompt_profile(prompt_type: str) -> str:
    if prompt_type in {"teacher", "teacher_synthesis"}:
        return "teacher_synthesis"
    if prompt_type == "teacher_eval":
        return "teacher_eval"
    return prompt_type


def profile_label(prompt_type: str) -> str:
    return _prompt_profile(prompt_type)
