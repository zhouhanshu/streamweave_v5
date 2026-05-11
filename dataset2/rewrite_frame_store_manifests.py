#!/usr/bin/env python3
"""Rewrite dataset2 frame manifests into StreamWeave FrameStore format."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


EXTRACTOR_VERSION = "frame_store_manifest_v1"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    dataset_dirs = [root / name for name in args.datasets] if args.datasets else sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "video_index.jsonl").exists()
    )
    total = 0
    warnings: list[str] = []
    for dataset_dir in dataset_dirs:
        count = rewrite_dataset(dataset_dir, image_ext=args.image_ext, jpeg_quality=args.jpeg_quality, warnings=warnings)
        total += count
        print(f"{dataset_dir.name}: rewrote {count} manifests", flush=True)
    if warnings:
        print("warnings:", flush=True)
        for item in warnings[:50]:
            print(f"- {item}", flush=True)
        if len(warnings) > 50:
            print(f"- ... {len(warnings) - 50} more", flush=True)
    print(f"done: rewrote {total} manifests", flush=True)


def rewrite_dataset(dataset_dir: Path, *, image_ext: str, jpeg_quality: int, warnings: list[str]) -> int:
    index_path = dataset_dir / "video_index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"missing video_index.jsonl: {index_path}")
    count = 0
    for row in iter_jsonl(index_path):
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            warnings.append(f"{dataset_dir.name}: skipped index row without video_id")
            continue
        frames_dir = resolve_frames_dir(dataset_dir, row, video_id)
        if not frames_dir.is_dir():
            warnings.append(f"{dataset_dir.name}/{video_id}: missing frames dir {frames_dir}")
            continue
        frame_count = int(row.get("frame_count") or count_frames(frames_dir, image_ext))
        actual_count = count_frames(frames_dir, image_ext)
        if actual_count != frame_count:
            warnings.append(f"{dataset_dir.name}/{video_id}: frame_count index={frame_count} actual={actual_count}; writing actual")
            frame_count = actual_count
        manifest = {
            "status": "complete",
            "extractor_version": EXTRACTOR_VERSION,
            "created_at": time.time(),
            "sample_fps": float(row.get("sample_fps", row.get("fps", 1.0)) or 1.0),
            "max_frames": 0,
            "frame_count": frame_count,
            "frame_id_base": int(row.get("frame_id_base", 0) or 0),
            "image_ext": image_ext,
            "jpeg_quality": int(jpeg_quality),
        }
        manifest_path = frames_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        count += 1
    return count


def resolve_frames_dir(dataset_dir: Path, row: dict[str, Any], video_id: str) -> Path:
    value = str(row.get("frames_dir") or row.get("video") or f"video/{video_id}").strip()
    path = Path(value)
    return path if path.is_absolute() else dataset_dir / path


def count_frames(frames_dir: Path, image_ext: str) -> int:
    count = len(list(frames_dir.glob(f"*.{image_ext}")))
    if count == 0 and image_ext.lower() == "jpg":
        count = len(list(frames_dir.glob("*.jpeg")))
    return count


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="dataset2")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--image-ext", default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


if __name__ == "__main__":
    main()
