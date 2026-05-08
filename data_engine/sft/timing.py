"""Frame timing helpers for SFT synthesis rollouts."""

from __future__ import annotations

from .schemas import FrameRef as PlanFrameRef
from .schemas import QueryPlan, SamplePlan


def query_events_by_frame(frames: list[PlanFrameRef], query_events: list[QueryPlan]) -> dict[int, list[QueryPlan]]:
    if not frames:
        return {}
    out: dict[int, list[QueryPlan]] = {}
    for query in sorted(query_events, key=lambda item: item.timestamp):
        frame_index = query_frame_index(frames, query.timestamp)
        out.setdefault(frame_index, []).append(query)
    return out


def query_frame_index(frames: list[PlanFrameRef], query_time: float) -> int:
    target_position = target_frame_position_for_time(frames, float(query_time))
    return frames[target_position].frame_index


def target_frame_position_for_time(frames: list[PlanFrameRef], timestamp: float) -> int:
    if not frames:
        return 0
    if timestamp <= frames[0].start_time:
        return 0
    for index, frame in enumerate(frames):
        if frame.start_time < timestamp <= frame.end_time:
            return index
        if timestamp < frame.start_time:
            return index
    return len(frames) - 1


def sample_target_timestamp(sample: SamplePlan) -> float | None:
    for key in ("target_timestamp", "answer_time"):
        value = sample.metadata.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return sample.answer_time


def truncate_plan_frames_at_timestamp(frames: list[PlanFrameRef], timestamp: float | None) -> list[PlanFrameRef]:
    if timestamp is None or not frames:
        return frames
    target_position = target_frame_position_for_time(frames, float(timestamp))
    return frames[: target_position + 1]


def group_frames(frames: list[PlanFrameRef], size: int) -> list[list[PlanFrameRef]]:
    size = max(1, int(size))
    return [frames[idx : idx + size] for idx in range(0, len(frames), size)]


def target_window_for_time(
    value: float,
    sample: SamplePlan,
    frames: list[PlanFrameRef],
    frames_per_step: int = 1,
) -> tuple[float, float]:
    all_frames = sample.frames or frames
    if not all_frames:
        target = float(value)
        return target, target
    target = float(value)
    window_size = max(1, frames_per_step)
    target_index = target_frame_position_for_time(all_frames, target)
    window_start = (target_index // window_size) * window_size
    window_end = min(window_start + window_size, len(all_frames))
    return float(all_frames[window_start].start_time), float(all_frames[window_end - 1].end_time)
