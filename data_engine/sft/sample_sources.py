"""Dispatch SFT data sources into the shared SamplePlan protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .schemas import SamplePlan


@dataclass(slots=True)
class SampleSourceConfig:
    source: str
    input: Path
    raw_data_root: Path
    ovo_anno_path: Path
    ovo_video_dir: Path
    ovo_task: str
    frame_dataset_root: Path
    frame_dataset_name: str
    fps: float | None
    max_frames: int
    offset: int
    limit: int
    sample_ids: set[str]


SourceLoader = Callable[[SampleSourceConfig], list[SamplePlan]]


def load_sample_source(config: SampleSourceConfig) -> list[SamplePlan]:
    try:
        loader = SOURCE_LOADERS[config.source]
    except KeyError as exc:
        raise ValueError(f"Unknown SFT source: {config.source}") from exc
    return loader(config)


def source_media_dir(config: SampleSourceConfig) -> Path:
    if config.source == "ovo":
        return config.frame_dataset_root
    return config.raw_data_root


def source_input_path(config: SampleSourceConfig) -> Path:
    if config.source == "ovo":
        return config.ovo_anno_path
    return config.input


def _load_frames(config: SampleSourceConfig) -> list[SamplePlan]:
    from .frame_dataset import load_sample_plans

    return load_sample_plans(
        config.input,
        raw_data_root=config.raw_data_root,
        sample_fps=config.fps,
        offset=config.offset,
        limit=config.limit,
        sample_ids=config.sample_ids,
        max_frames=config.max_frames,
    )


def _load_ovo(config: SampleSourceConfig) -> list[SamplePlan]:
    from .ovo_dataset import load_ovo_sample_plans

    return load_ovo_sample_plans(
        anno_path=config.ovo_anno_path,
        video_dir=config.ovo_video_dir,
        dataset_root=config.frame_dataset_root,
        dataset_name=config.frame_dataset_name,
        fps=config.fps or 1.0,
        max_frames=config.max_frames,
        offset=config.offset,
        limit=config.limit,
        sample_ids=config.sample_ids,
        task=config.ovo_task,
    )


SOURCE_LOADERS: dict[str, SourceLoader] = {
    "frames": _load_frames,
    "ovo": _load_ovo,
}
