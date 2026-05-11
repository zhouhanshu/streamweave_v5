"""Single-sample rollout loop."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

from backend.base import BaseBackend

from .config import MemoryConfig, PostprocessConfig, RewardConfig, RuntimeConfig, SynthesisConfig, TraceConfig
from .env import StreamWeaveEnv
from .frame_store import FrameStore
from .policies import make_policy
from .postprocess import synthesis_feedback
from .schemas import AppliedAction, AttemptRecord, BackendResult, BenchmarkSample, FrameRef, ModelAction, QARecord, QualityReport, QueryEvent, RolloutTrace, Transition
from .trace_io import TraceWriter


@dataclass(slots=True)
class StepRunResult:
    prompt_text: str
    prompt_images: list[str]
    current_frames: list[FrameRef]
    backend_result: BackendResult
    raw_action: ModelAction
    quality: QualityReport
    applied: AppliedAction
    base_prompt_text: str
    base_prompt_images: list[str]
    attempt_prompt_text: str
    attempt_prompt_images: list[str]
    attempts: list[AttemptRecord]
    accepted_attempt_index: int
    target_raw_output: str


@dataclass(slots=True)
class MultiQARunResult:
    sample: BenchmarkSample
    prefix_transitions: list[Transition]
    qa_traces: list[RolloutTrace]
    task_failed: bool = False
    failure_reason: str = ""


class RolloutRunner:
    def __init__(
        self,
        *,
        backend: BaseBackend,
        frame_store: FrameStore,
        runtime: RuntimeConfig,
        trace_config: TraceConfig,
        dataset_name: str,
        prompt_profile: str,
        policy: str,
        postprocess_config: PostprocessConfig | None = None,
        reward_config: RewardConfig | None = None,
        synthesis_config: SynthesisConfig | None = None,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        self.backend = backend
        self.frame_store = frame_store
        self.runtime = runtime
        self.trace_config = trace_config
        self.dataset_name = dataset_name
        self.prompt_profile = prompt_profile
        self.policy = make_policy(policy)
        self.postprocess_config = postprocess_config or PostprocessConfig()
        self.reward_config = reward_config or RewardConfig()
        self.synthesis_config = synthesis_config or SynthesisConfig()
        self.memory_config = memory_config or MemoryConfig()

    def run_sample(self, sample: BenchmarkSample) -> RolloutTrace:
        dataset_name = str(sample.metadata.get("frame_dataset_name") or self.dataset_name)
        declared_count = read_declared_frame_count(sample.metadata, sample.sample_id)
        frames = self.frame_store.ensure_frames(
            dataset_name=dataset_name,
            video_id=sample.video_id,
            video_path=sample.video_path,
            sample_fps=self.runtime.sample_fps,
            max_frames=self.runtime.max_frames,
        )
        if declared_count is not None:
            if len(frames) < declared_count:
                raise RuntimeError(
                    f"Sample {sample.sample_id!r} declared frame_count={declared_count} "
                    f"but only {len(frames)} frame(s) are available on disk."
                )
            frames = frames[:declared_count]
        frames = _truncate_frames_at_timestamp(
            frames,
            sample.metadata.get("target_timestamp"),
            sample_fps=self.runtime.sample_fps,
            frame_id_base=self.frame_store.config.frame_id_base,
        )
        env = StreamWeaveEnv(
            prompt_profile=self.prompt_profile,
            policy=self.policy,
            memory_window=self.memory_config.window_seconds,
            extra_context=str(sample.metadata.get("teacher_context", "")),
        )
        trace_dir = self._trace_dir_for_sample(sample)
        writer = TraceWriter(trace_dir, write_jsonl=self.trace_config.write_jsonl)
        transitions: list[Transition] = []
        query_by_frame = _query_events_by_frame(
            frames,
            sample.query_events,
            sample_fps=self.runtime.sample_fps,
            frame_id_base=self.frame_store.config.frame_id_base,
        )
        groups = _group_frames(frames, self.runtime.frames_per_step)
        if self.runtime.max_steps:
            groups = groups[: self.runtime.max_steps]

        for step_index, group in enumerate(groups):
            if not group:
                continue
            for frame in group:
                for query in query_by_frame.get(frame.global_index, []):
                    env.add_question(QARecord(timestamp=query.timestamp, text=query.text, role="q"))

            prompt_frames = self._prompt_frames(frames, group)
            step_end = group[-1].end_time
            env.evict_memory(step_end)
            memory_before = env.memory_text()
            result = self._run_model_step(
                env=env,
                writer=writer,
                step_index=step_index,
                prompt_frames=prompt_frames,
                memory_before=memory_before,
            )
            if result is None:
                return RolloutTrace(
                    sample=sample,
                    transitions=transitions,
                    task_failed=True,
                    failure_reason=f"synthesis_raw_retry_failed_at_step_{step_index}",
                )
            memory_after = env.memory_text()
            writer.write_env_done(step_index=step_index, quality=result.quality, applied=result.applied, memory_after=memory_after)

            transition = Transition(
                sample_id=sample.sample_id,
                video_id=sample.video_id,
                step_index=step_index,
                step_start=group[0].start_time,
                step_end=group[-1].end_time,
                prompt_text=result.prompt_text,
                prompt_images=result.prompt_images,
                current_frames=result.current_frames,
                raw_action=result.raw_action,
                quality=result.quality,
                applied=result.applied,
                backend_result=result.backend_result,
                memory_before=memory_before,
                memory_after=memory_after,
                base_prompt_text=result.base_prompt_text,
                base_prompt_images=result.base_prompt_images,
                attempt_prompt_text=result.attempt_prompt_text,
                attempt_prompt_images=result.attempt_prompt_images,
                attempts=result.attempts,
                accepted_attempt_index=result.accepted_attempt_index,
                target_raw_output=result.target_raw_output,
            )
            writer.write_transition(transition)
            transitions.append(transition)
        return RolloutTrace(sample=sample, transitions=transitions)

    def run_multi_qa_sample(self, sample: BenchmarkSample) -> MultiQARunResult:
        qa_list = [item for item in sample.metadata.get("qa_list") or [] if isinstance(item, dict)]
        if not qa_list:
            return MultiQARunResult(sample=sample, prefix_transitions=[], qa_traces=[], task_failed=True, failure_reason="empty_qa_list")

        dataset_name = str(sample.metadata.get("frame_dataset_name") or self.dataset_name)
        declared_count = read_declared_frame_count(sample.metadata, sample.sample_id)
        frames = self.frame_store.ensure_frames(
            dataset_name=dataset_name,
            video_id=sample.video_id,
            video_path=sample.video_path,
            sample_fps=self.runtime.sample_fps,
            max_frames=self.runtime.max_frames,
        )
        if declared_count is not None:
            if len(frames) < declared_count:
                raise RuntimeError(
                    f"Sample {sample.sample_id!r} declared frame_count={declared_count} "
                    f"but only {len(frames)} frame(s) are available on disk."
                )
            frames = frames[:declared_count]
        frames = _truncate_frames_at_timestamp(
            frames,
            sample.metadata.get("target_timestamp"),
            sample_fps=self.runtime.sample_fps,
            frame_id_base=self.frame_store.config.frame_id_base,
        )
        groups = _group_frames(frames, self.runtime.frames_per_step)
        if self.runtime.max_steps:
            groups = groups[: self.runtime.max_steps]
        if not groups:
            return MultiQARunResult(sample=sample, prefix_transitions=[], qa_traces=[], task_failed=True, failure_reason="empty_frame_groups")

        env = StreamWeaveEnv(
            prompt_profile=self.prompt_profile,
            policy=self.policy,
            memory_window=self.memory_config.window_seconds,
            extra_context=str(sample.metadata.get("teacher_context", "")),
        )
        trace_dir = self._trace_dir_for_sample(sample)
        prefix_writer = TraceWriter(trace_dir / "_prefix", write_jsonl=self.trace_config.write_jsonl)
        prefix_transitions: list[Transition] = []

        for step_index, group in enumerate(groups[:-1]):
            if not group:
                continue
            prompt_frames = self._prompt_frames(frames, group)
            step_end = group[-1].end_time
            env.evict_memory(step_end)
            memory_before = env.memory_text()
            result = self._run_model_step(
                env=env,
                writer=prefix_writer,
                step_index=step_index,
                prompt_frames=prompt_frames,
                memory_before=memory_before,
            )
            if result is None:
                return MultiQARunResult(
                    sample=sample,
                    prefix_transitions=prefix_transitions,
                    qa_traces=[],
                    task_failed=True,
                    failure_reason=f"synthesis_raw_retry_failed_at_prefix_step_{step_index}",
                )
            memory_after = env.memory_text()
            prefix_writer.write_env_done(step_index=step_index, quality=result.quality, applied=result.applied, memory_after=memory_after)
            transition = Transition(
                sample_id=sample.sample_id,
                video_id=sample.video_id,
                step_index=step_index,
                step_start=group[0].start_time,
                step_end=group[-1].end_time,
                prompt_text=result.prompt_text,
                prompt_images=result.prompt_images,
                current_frames=result.current_frames,
                raw_action=result.raw_action,
                quality=result.quality,
                applied=result.applied,
                backend_result=result.backend_result,
                memory_before=memory_before,
                memory_after=memory_after,
                base_prompt_text=result.base_prompt_text,
                base_prompt_images=result.base_prompt_images,
                attempt_prompt_text=result.attempt_prompt_text,
                attempt_prompt_images=result.attempt_prompt_images,
                attempts=result.attempts,
                accepted_attempt_index=result.accepted_attempt_index,
                target_raw_output=result.target_raw_output,
            )
            prefix_writer.write_transition(transition)
            prefix_transitions.append(transition)

        final_group = groups[-1]
        final_step_index = len(groups) - 1
        prompt_frames = self._prompt_frames(frames, final_group)
        qa_traces: list[RolloutTrace] = []
        for qa_index, qa in enumerate(qa_list):
            qa_id = _safe_trace_name(str(qa.get("qa_id") or qa.get("source_annotation_id") or f"qa_{qa_index:04d}"))
            qa_sample = _sample_for_qa_branch(sample, qa, qa_index)
            branch_env = copy.deepcopy(env)
            branch_env.add_question(
                QARecord(
                    timestamp=float(sample.metadata.get("target_timestamp") or final_group[-1].end_time),
                    text=str(qa.get("query_text") or qa.get("question") or ""),
                    role="q",
                )
            )
            branch_writer = TraceWriter(trace_dir / qa_id, write_jsonl=self.trace_config.write_jsonl)
            step_end = final_group[-1].end_time
            branch_env.evict_memory(step_end)
            memory_before = branch_env.memory_text()
            result = self._run_model_step(
                env=branch_env,
                writer=branch_writer,
                step_index=final_step_index,
                prompt_frames=prompt_frames,
                memory_before=memory_before,
            )
            if result is None:
                qa_traces.append(
                    RolloutTrace(
                        sample=qa_sample,
                        transitions=list(prefix_transitions),
                        task_failed=True,
                        failure_reason=f"synthesis_raw_retry_failed_at_qa_step_{final_step_index}",
                    )
                )
                continue
            memory_after = branch_env.memory_text()
            branch_writer.write_env_done(step_index=final_step_index, quality=result.quality, applied=result.applied, memory_after=memory_after)
            transition = Transition(
                sample_id=qa_sample.sample_id,
                video_id=qa_sample.video_id,
                step_index=final_step_index,
                step_start=final_group[0].start_time,
                step_end=final_group[-1].end_time,
                prompt_text=result.prompt_text,
                prompt_images=result.prompt_images,
                current_frames=result.current_frames,
                raw_action=result.raw_action,
                quality=result.quality,
                applied=result.applied,
                backend_result=result.backend_result,
                memory_before=memory_before,
                memory_after=memory_after,
                base_prompt_text=result.base_prompt_text,
                base_prompt_images=result.base_prompt_images,
                attempt_prompt_text=result.attempt_prompt_text,
                attempt_prompt_images=result.attempt_prompt_images,
                attempts=result.attempts,
                accepted_attempt_index=result.accepted_attempt_index,
                target_raw_output=result.target_raw_output,
            )
            branch_writer.write_transition(transition)
            qa_traces.append(RolloutTrace(sample=qa_sample, transitions=[*prefix_transitions, transition]))
        return MultiQARunResult(sample=sample, prefix_transitions=prefix_transitions, qa_traces=qa_traces)

    def _run_model_step(
        self,
        *,
        env: StreamWeaveEnv,
        writer: TraceWriter,
        step_index: int,
        prompt_frames: list,
        memory_before: str,
    ) -> StepRunResult | None:
        mode = self.postprocess_config.mode
        if mode in {"eval_repair", "rollout_repair"}:
            content, prompt_text, prompt_images, local_frames = env.build_prompt(prompt_frames)
            writer.write_step_start(
                step_index=step_index,
                prompt_text=prompt_text,
                prompt_images=prompt_images,
                memory_before=memory_before,
                current_frames=local_frames,
            )
            backend_result = self.backend.generate(content)
            writer.write_backend_done(step_index=step_index, backend_result=backend_result)
            raw_action, quality, applied = env.evaluate_attempt(
                backend_result.text,
                frames=local_frames,
                reward_config=self.reward_config,
                repair=True,
            )
            env.commit(applied)
            attempt = AttemptRecord(
                attempt_index=1,
                raw_output=backend_result.text,
                quality=quality,
                backend_result=backend_result,
                prompt_text=prompt_text,
                prompt_images=prompt_images,
                accepted=True,
            )
            return StepRunResult(
                prompt_text=prompt_text,
                prompt_images=prompt_images,
                current_frames=local_frames,
                backend_result=backend_result,
                raw_action=raw_action,
                quality=quality,
                applied=applied,
                base_prompt_text=prompt_text,
                base_prompt_images=prompt_images,
                attempt_prompt_text=prompt_text,
                attempt_prompt_images=prompt_images,
                attempts=[attempt],
                accepted_attempt_index=1,
                target_raw_output=backend_result.text,
            )

        if mode != "synthesis_raw_retry":
            raise ValueError(f"Unknown postprocess mode: {mode}")

        retry_feedback = ""
        max_attempts = max(1, int(self.synthesis_config.max_attempts))
        _, base_prompt_text, base_prompt_images, base_local_frames = env.build_prompt(prompt_frames)
        attempts: list[AttemptRecord] = []
        for attempt_index in range(1, max_attempts + 1):
            content, prompt_text, prompt_images, local_frames = env.build_prompt(prompt_frames, retry_feedback=retry_feedback)
            writer.write_step_start(
                step_index=step_index,
                prompt_text=prompt_text,
                prompt_images=prompt_images,
                memory_before=memory_before,
                current_frames=local_frames,
            )
            backend_result = self.backend.generate(content)
            writer.write_backend_done(step_index=step_index, backend_result=backend_result)
            raw_action, quality, applied = env.evaluate_attempt(
                backend_result.text,
                frames=local_frames,
                reward_config=self.reward_config,
                repair=False,
            )
            if quality.valid:
                env.commit(applied)
                attempts.append(
                    AttemptRecord(
                        attempt_index=attempt_index,
                        raw_output=backend_result.text,
                        quality=quality,
                        backend_result=backend_result,
                        prompt_text=prompt_text,
                        prompt_images=prompt_images,
                        accepted=True,
                    )
                )
                return StepRunResult(
                    prompt_text=base_prompt_text,
                    prompt_images=base_prompt_images,
                    current_frames=base_local_frames,
                    backend_result=backend_result,
                    raw_action=raw_action,
                    quality=quality,
                    applied=applied,
                    base_prompt_text=base_prompt_text,
                    base_prompt_images=base_prompt_images,
                    attempt_prompt_text=prompt_text,
                    attempt_prompt_images=prompt_images,
                    attempts=attempts,
                    accepted_attempt_index=attempt_index,
                    target_raw_output=backend_result.text,
                )
            retry_feedback = synthesis_feedback(quality.issues, backend_result.text)
            attempts.append(
                AttemptRecord(
                    attempt_index=attempt_index,
                    raw_output=backend_result.text,
                    quality=quality,
                    backend_result=backend_result,
                    feedback=retry_feedback,
                    prompt_text=prompt_text,
                    prompt_images=prompt_images,
                    accepted=False,
                )
            )
            writer.write_attempt_failed(
                step_index=step_index,
                attempt_index=attempt_index,
                quality=quality,
                feedback=retry_feedback,
                memory_after=env.memory_text(),
            )
        return None

    def _prompt_frames(self, frames: list, group: list):
        if self.policy.use_recent_frames:
            end_position = frames.index(group[-1])
            return self.frame_store.recent_frames(frames, end_position, count=self.policy.recent_frame_count)
        return group

    def _trace_dir_for_sample(self, sample: BenchmarkSample) -> Path:
        trace_dir = Path(self.trace_config.output_root) / self.trace_config.experiment_name / sample.video_id
        if sample.sample_id != sample.video_id:
            trace_dir = trace_dir / sample.sample_id
        return trace_dir


def _sample_for_qa_branch(sample: BenchmarkSample, qa: dict, qa_index: int) -> BenchmarkSample:
    qa_id = str(qa.get("qa_id") or qa.get("source_annotation_id") or f"qa_{qa_index:04d}")
    timestamp = float(sample.metadata.get("target_timestamp") or sample.metadata.get("query_timestamp") or 0.0)
    metadata = dict(sample.metadata)
    metadata.update(
        {
            "is_multi_qa_branch": True,
            "qa_id": qa_id,
            "qa_index": qa_index,
            "question": qa.get("query_text") or qa.get("question") or "",
            "options": qa.get("options"),
            "answer": qa.get("answer"),
            "gt": qa.get("gt"),
            "ground_truth": qa.get("ground_truth"),
            "raw_annotation": qa,
            "query_timestamp": timestamp,
            "target_timestamp": timestamp,
        }
    )
    return BenchmarkSample(
        sample_id=f"{sample.sample_id}_{qa_id}",
        video_id=sample.video_id,
        video_path=sample.video_path,
        query_events=[QueryEvent(text=str(qa.get("query_text") or qa.get("question") or ""), timestamp=timestamp)],
        metadata=metadata,
    )


def _safe_trace_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return safe or "qa"


def _group_frames(frames: list, size: int) -> list[list]:
    size = max(1, int(size))
    return [frames[idx : idx + size] for idx in range(0, len(frames), size)]


def read_declared_frame_count(metadata: dict, sample_id: str) -> int | None:
    """Return the data-record-declared frame count, or None if absent.

    Raises ValueError when the field is present but malformed (non-int, <=0),
    so silent garbage in source data fails fast.
    """
    raw = None
    if metadata:
        raw = metadata.get("frame_count")
        if raw is None:
            raw = metadata.get("sampled_frames")
    if raw is None:
        return None
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sample {sample_id!r} has invalid frame_count {raw!r}; expected a positive integer."
        ) from exc
    if count <= 0:
        raise ValueError(
            f"Sample {sample_id!r} has non-positive frame_count {count}; expected a positive integer."
        )
    return count


def required_frame_count_from_metadata(metadata: dict, sample_id: str) -> int:
    count = read_declared_frame_count(metadata, sample_id)
    if count is None:
        raise ValueError(
            f"Sample {sample_id!r} is missing frame_count/sampled_frames in its data record; "
            "step count must be derived from the data record, not the frame folder."
        )
    return count


def _truncate_frames_at_timestamp(
    frames: list[FrameRef],
    timestamp: object,
    *,
    sample_fps: float,
    frame_id_base: int,
) -> list[FrameRef]:
    if timestamp is None or not frames:
        return frames
    try:
        target_timestamp = float(timestamp)
    except (TypeError, ValueError):
        return frames

    min_frame_id = min(frame.global_index for frame in frames)
    max_frame_id = max(frame.global_index for frame in frames)
    target_frame_id = _timestamp_to_frame_id(
        target_timestamp,
        sample_fps=sample_fps,
        frame_id_base=frame_id_base,
        min_frame_id=min_frame_id,
        max_frame_id=max_frame_id,
    )
    return [frame for frame in frames if frame.global_index <= target_frame_id]


def _query_events_by_frame(
    frames: list[FrameRef],
    query_events: list[QueryEvent],
    *,
    sample_fps: float,
    frame_id_base: int,
) -> dict[int, list[QueryEvent]]:
    if not frames:
        return {}
    min_frame_id = min(frame.global_index for frame in frames)
    max_frame_id = max(frame.global_index for frame in frames)
    out: dict[int, list[QueryEvent]] = {}
    for event in sorted(query_events, key=lambda item: item.timestamp):
        frame_id = _timestamp_to_frame_id(
            event.timestamp,
            sample_fps=sample_fps,
            frame_id_base=frame_id_base,
            min_frame_id=min_frame_id,
            max_frame_id=max_frame_id,
        )
        out.setdefault(frame_id, []).append(event)
    return out


def _timestamp_to_frame_id(
    timestamp: float,
    *,
    sample_fps: float,
    frame_id_base: int,
    min_frame_id: int,
    max_frame_id: int,
) -> int:
    seconds_per_frame = 1.0 / max(sample_fps, 1e-6)
    if timestamp <= 0:
        frame_id = frame_id_base
    else:
        frame_offset = max(0, math.ceil((float(timestamp) - 1e-9) / seconds_per_frame) - 1)
        frame_id = frame_id_base + frame_offset
    return min(max(frame_id, min_frame_id), max_frame_id)
