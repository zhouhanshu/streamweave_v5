"""Shared SFT source schemas.

Every SFT data source is adapted into SamplePlan before rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import JsonDict


@dataclass(slots=True)
class FrameRef:
    """One extracted frame used as one StreamWeave time slice."""

    global_frame_id: int
    start_time: float
    end_time: float
    image_path: Path
    frame_index: int
    _image: Any = field(default=None, repr=False)

    def load_image(self) -> Any:
        if self._image is None:
            from PIL import Image

            self._image = Image.open(self.image_path).convert("RGB")
            self._image.filename = str(self.image_path)
        return self._image


@dataclass(slots=True)
class QueryPlan:
    text: str
    timestamp: float


@dataclass(slots=True)
class SamplePlan:
    sample_id: str
    video_id: str
    qa_id: str
    task: str
    query_events: list[QueryPlan]
    question_text: str
    query_time: float | None
    answer_time: float | None
    frames: list[FrameRef]
    metadata: JsonDict
