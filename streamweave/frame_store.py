"""Access to extracted frame directories."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .schemas import FrameRef
from .video_extract import extract_video_frames


MANIFEST_NAME = "manifest.json"
EXTRACTOR_VERSION = "frame_store_manifest_v1"


class FrameStore:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self._lock_guard = threading.Lock()
        self._locks: dict[tuple[str, str], threading.Lock] = {}

    def frame_dir(self, dataset_name: str, video_id: str) -> Path:
        return self.dataset_root / dataset_name / "video" / video_id

    def ensure_frames(
        self,
        *,
        dataset_name: str,
        video_id: str,
        video_path: str,
        sample_fps: float,
        max_frames: int = 0,
    ) -> list[FrameRef]:
        directory = self.frame_dir(dataset_name, video_id)
        with self._video_lock(dataset_name, video_id):
            if not self._is_complete(directory, sample_fps=sample_fps, video_path=video_path, max_frames=max_frames):
                if not video_path:
                    raise FileNotFoundError(f"No extracted frames and no source video for {video_id}: {directory}")
                self._prepare_for_extraction(directory)
                self._write_manifest(
                    directory,
                    status="extracting",
                    video_path=video_path,
                    sample_fps=sample_fps,
                    max_frames=max_frames,
                    frame_count=0,
                )
                extract_video_frames(
                    video_path,
                    directory,
                    fps=sample_fps,
                    image_ext=self.config.image_ext,
                    jpeg_quality=self.config.jpeg_quality,
                    overwrite=self.config.overwrite_frames,
                    max_frames=max_frames,
                )
                paths = self._frame_paths(directory)
                if not self._frame_names_are_contiguous(paths):
                    raise RuntimeError(f"Extracted frames are not contiguous: {directory}")
                self._write_manifest(
                    directory,
                    status="complete",
                    video_path=video_path,
                    sample_fps=sample_fps,
                    max_frames=max_frames,
                    frame_count=len(paths),
                )
        return self.load_frames(dataset_name=dataset_name, video_id=video_id, sample_fps=sample_fps, max_frames=max_frames)

    def load_frames(
        self,
        *,
        dataset_name: str,
        video_id: str,
        sample_fps: float,
        max_frames: int = 0,
    ) -> list[FrameRef]:
        directory = self.frame_dir(dataset_name, video_id)
        paths = self._frame_paths(directory)
        if max_frames:
            paths = paths[:max_frames]
        if not paths:
            raise FileNotFoundError(f"No extracted frames found: {directory}")
        refs: list[FrameRef] = []
        for offset, path in enumerate(paths):
            frame_id = self._parse_frame_id(path)
            start = (frame_id - self.config.frame_id_base) / max(sample_fps, 1e-6)
            end = start + 1.0 / max(sample_fps, 1e-6)
            refs.append(
                FrameRef(
                    video_id=video_id,
                    global_index=frame_id,
                    start_time=start,
                    end_time=end,
                    image_path=path,
                    step_local_id=offset + 1,
                )
            )
        return refs

    def recent_frames(self, frames: list[FrameRef], end_position: int, *, count: int = 5) -> list[FrameRef]:
        end_position = max(0, min(end_position, len(frames) - 1))
        start = max(0, end_position - count + 1)
        return frames[start : end_position + 1]

    def _is_complete(
        self,
        directory: Path,
        *,
        sample_fps: float | None = None,
        video_path: str | Path | None = None,
        max_frames: int = 0,
    ) -> bool:
        paths = self._frame_paths(directory)
        if not paths:
            return False
        if max_frames and len(paths) < max_frames:
            return False
        if not self._frame_names_are_contiguous(paths):
            return False
        manifest = self._read_manifest(directory)
        if not manifest:
            return False
        if manifest.get("status") != "complete":
            return False
        if manifest.get("extractor_version") != EXTRACTOR_VERSION:
            return False
        if manifest.get("image_ext") != self.config.image_ext:
            return False
        if int(manifest.get("frame_id_base", -1)) != int(self.config.frame_id_base):
            return False
        frame_count = int(manifest.get("frame_count", -1))
        if frame_count != len(paths):
            return False
        if max_frames and frame_count < max_frames:
            return False
        if not max_frames and int(manifest.get("max_frames", 0) or 0) != 0:
            return False
        if sample_fps is not None and abs(float(manifest.get("sample_fps", -1.0)) - float(sample_fps)) > 1e-6:
            return False
        if video_path:
            source = self._source_metadata(video_path)
            if manifest.get("source_path") != source["source_path"]:
                return False
            if int(manifest.get("source_size", -1)) != int(source["source_size"]):
                return False
            if int(manifest.get("source_mtime_ns", -1)) != int(source["source_mtime_ns"]):
                return False
        return True

    def _frame_paths(self, directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        paths = sorted(directory.glob(f"*.{self.config.image_ext}"))
        if not paths and self.config.image_ext.lower() == "jpg":
            paths = sorted(directory.glob("*.jpeg"))
        return paths

    def _frame_names_are_contiguous(self, paths: list[Path]) -> bool:
        expected = list(range(len(paths)))
        actual = [self._parse_frame_id(path) - self.config.frame_id_base for path in paths]
        return actual == expected

    def _prepare_for_extraction(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for pattern in {f"*.{self.config.image_ext}", "*.jpg", "*.jpeg"}:
            for old_path in directory.glob(pattern):
                if old_path.is_file():
                    old_path.unlink()

    def _manifest_path(self, directory: Path) -> Path:
        return directory / MANIFEST_NAME

    def _read_manifest(self, directory: Path) -> dict[str, Any] | None:
        path = self._manifest_path(directory)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _write_manifest(
        self,
        directory: Path,
        *,
        status: str,
        video_path: str | Path,
        sample_fps: float,
        max_frames: int,
        frame_count: int,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "status": status,
            "extractor_version": EXTRACTOR_VERSION,
            "created_at": time.time(),
            "sample_fps": float(sample_fps),
            "max_frames": int(max_frames),
            "frame_count": int(frame_count),
            "frame_id_base": int(self.config.frame_id_base),
            "image_ext": self.config.image_ext,
            "jpeg_quality": int(self.config.jpeg_quality),
        }
        data.update(self._source_metadata(video_path))
        manifest_path = self._manifest_path(directory)
        tmp_path = manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(manifest_path)

    @staticmethod
    def _source_metadata(video_path: str | Path) -> dict[str, Any]:
        source = Path(video_path).resolve()
        stat = source.stat()
        return {
            "source_path": str(source),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "source_size": int(stat.st_size),
        }

    def _video_lock(self, dataset_name: str, video_id: str) -> threading.Lock:
        key = (dataset_name, video_id)
        with self._lock_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    @staticmethod
    def _parse_frame_id(path: Path) -> int:
        try:
            return int(path.stem)
        except ValueError as exc:
            raise ValueError(f"Frame file name must be numeric: {path}") from exc
