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
    return config.raw_data_root


def source_input_path(config: SampleSourceConfig) -> Path:
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


def _load_dataset2(config: SampleSourceConfig) -> list[SamplePlan]:
    from .dataset2_source import load_sample_plans

    return load_sample_plans(
        config.input,
        raw_data_root=config.raw_data_root,
        sample_fps=config.fps,
        offset=config.offset,
        limit=config.limit,
        sample_ids=config.sample_ids,
        max_frames=config.max_frames,
    )


SOURCE_LOADERS: dict[str, SourceLoader] = {
    "frames": _load_frames,
    "dataset2": _load_dataset2,
}
