#!/usr/bin/env python3
"""Prepare the synthesized streamweave_data dataset under the v4 layout."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


EXTRACTOR_VERSION = "frame_store_manifest_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-annotations", type=Path, required=True)
    parser.add_argument("--source-frame-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="streamweave_data")
    parser.add_argument("--write-manifests", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def normalize_row(row: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    out = dict(row)
    video_id = str(out.get("video_id") or "").strip()
    if not video_id:
        raise ValueError(f"annotation row is missing video_id: {row}")
    frame_rel = f"video/{video_id}"
    out["dataset"] = dataset_name
    out["frames_dir"] = frame_rel
    out["video"] = frame_rel
    out["frame_id_base"] = int(out.get("frame_id_base", 0) or 0)
    out.setdefault("fps", 1.0)
    out.setdefault("sample_fps", float(out.get("fps", 1.0) or 1.0))
    return out


def has_valid_qa(row: dict[str, Any]) -> bool:
    return bool(str(row.get("question") or "").strip())


def frame_paths(video_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def write_manifest(video_dir: Path, source_dir: Path, *, sample_fps: float = 1.0) -> int:
    paths = frame_paths(video_dir)
    frame_count = len(paths)
    data: dict[str, Any] = {
        "status": "complete",
        "extractor_version": EXTRACTOR_VERSION,
        "created_at": time.time(),
        "sample_fps": float(sample_fps),
        "max_frames": 0,
        "frame_count": frame_count,
        "frame_id_base": 0,
        "image_ext": "jpg",
        "jpeg_quality": 95,
        "source_path": str(source_dir.resolve()),
        "source_mtime_ns": int(source_dir.stat().st_mtime_ns) if source_dir.exists() else 0,
        "source_size": 0,
    }
    tmp_path = video_dir / "manifest.json.tmp"
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(video_dir / "manifest.json")
    return frame_count


def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    video_root = dataset_dir / "video"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)

    source_copy = dataset_dir / "annotations_qa_source.jsonl"
    shutil.copyfile(args.source_annotations, source_copy)

    source_rows = iter_jsonl(args.source_annotations)
    rows = [normalize_row(row, args.dataset_name) for row in source_rows]
    valid_rows = [row for row in rows if has_valid_qa(row)]
    write_jsonl(dataset_dir / "annotations_qa.jsonl", rows)
    write_jsonl(dataset_dir / "annotations_qa_valid.jsonl", valid_rows)

    manifest_count = 0
    missing_dirs: list[str] = []
    if args.write_manifests:
        for video_dir in sorted(path for path in video_root.iterdir() if path.is_dir()):
            source_dir = args.source_frame_root / video_dir.name
            write_manifest(video_dir, source_dir)
            manifest_count += 1
        for row in rows:
            video_id = str(row["video_id"])
            if not (video_root / video_id).is_dir():
                missing_dirs.append(video_id)

    summary = {
        "dataset_name": args.dataset_name,
        "source_annotations": str(args.source_annotations),
        "source_frame_root": str(args.source_frame_root),
        "dataset_dir": str(dataset_dir),
        "annotation_rows": len(rows),
        "valid_qa_rows": len(valid_rows),
        "manifest_count": manifest_count,
        "missing_frame_dirs": missing_dirs,
    }
    (dataset_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
