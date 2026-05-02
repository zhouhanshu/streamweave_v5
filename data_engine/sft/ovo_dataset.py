"""OVO-Bench input adapter for StreamWeave SFT synthesis."""

from __future__ import annotations

from pathlib import Path

from evaluation import ovo_adapter
from streamweave.config import DatasetConfig
from streamweave.frame_store import FrameStore

from .io_utils import JsonDict
from .schemas import FrameRef, QueryPlan, SamplePlan


def load_ovo_sample_plans(
    *,
    anno_path: str | Path,
    video_dir: str | Path,
    dataset_root: str | Path = "dataset",
    dataset_name: str = "ovo",
    fps: float = 1.0,
    max_frames: int = 0,
    offset: int = 0,
    limit: int = 0,
    sample_ids: set[str] | None = None,
    task: str = "",
) -> list[SamplePlan]:
    benchmark_args = {
        "anno_path": str(anno_path),
        "video_dir": str(video_dir),
        "sample_ids": sorted(sample_ids or []),
        "task": task,
        "limit": limit,
    }
    samples = ovo_adapter.load_samples(benchmark_args)
    if offset:
        samples = samples[offset:]
    if limit:
        samples = samples[:limit]
    frame_store = FrameStore(DatasetConfig(dataset_root=str(dataset_root), dataset_name=dataset_name))
    return [
        ovo_sample_to_plan(
            sample,
            frame_store=frame_store,
            dataset_name=dataset_name,
            fps=fps,
            max_frames=max_frames,
        )
        for sample in samples
    ]


def ovo_sample_to_plan(
    sample,
    *,
    frame_store: FrameStore,
    dataset_name: str,
    fps: float,
    max_frames: int,
) -> SamplePlan:
    sw_frames = frame_store.ensure_frames(
        dataset_name=dataset_name,
        video_id=sample.video_id,
        video_path=sample.video_path,
        sample_fps=fps,
        max_frames=max_frames,
    )

    frames: list[FrameRef] = []
    for frame_index, sw_frame in enumerate(sw_frames):
        frames.append(
            FrameRef(
                global_frame_id=sw_frame.global_index,
                start_time=sw_frame.start_time,
                end_time=sw_frame.end_time,
                image_path=sw_frame.image_path,
                frame_index=frame_index,
            )
        )

    if not sample.query_events:
        raise ValueError(f"OVO sample {sample.sample_id} has no query event.")
    query_events = [QueryPlan(text=str(query.text), timestamp=float(query.timestamp)) for query in sample.query_events]
    query = query_events[0]
    task = str(sample.metadata.get("task") or "")
    metadata: JsonDict = {
        "benchmark": "ovo",
        "sample_metadata": dict(sample.metadata),
        "video_path": sample.video_path,
        "frame_dataset_root": str(frame_store.dataset_root),
        "frame_dataset_name": dataset_name,
        "frame_dir": str(frame_store.frame_dir(dataset_name, sample.video_id)),
    }
    return SamplePlan(
        sample_id=f"ovo_{sample.sample_id}",
        video_id=sample.video_id,
        qa_id=f"ovo_{sample.sample_id}",
        task=task,
        query_events=query_events,
        question_text=query.text,
        query_time=query.timestamp,
        answer_time=_answer_time(sample.metadata),
        frames=frames,
        metadata=metadata,
    )


def _answer_time(metadata: dict) -> float | None:
    value = metadata.get("target_timestamp")
    if value is not None:
        return float(value)
    value = metadata.get("query_timestamp")
    if value is not None:
        return float(value)
    return None
