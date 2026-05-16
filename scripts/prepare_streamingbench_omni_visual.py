#!/usr/bin/env python3
"""Prepare visual-only StreamingBench omni videos for the local eval layout."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_ANNO = Path("/mmu_mllm_hdd/zhouhanshu/test/StreamingBench/StreamingBench/StreamingBench/src/data/questions_omni_stream.json")
DEFAULT_EXTRACTED_ROOT = Path("/mmu_mllm_hdd/zhouhanshu/test/StreamingBench/extracted")
DEFAULT_SOURCE_VIDEOS = Path("dataset/streamingbench/source_videos")
DEFAULT_TASK_FILTER = "Misleading Context Understanding,Anomaly Context Understanding"


def main() -> None:
    args = parse_args()
    anno_path = Path(args.anno)
    extracted_root = Path(args.extracted_root)
    source_videos = Path(args.source_videos)
    task_filter = _task_filter(args.task_filter)

    with anno_path.open(encoding="utf-8") as handle:
        entries = json.load(handle)

    selected = _selected_videos(entries, task_filter=task_filter)
    source_videos.mkdir(parents=True, exist_ok=True)

    linked = 0
    existing = 0
    missing: list[str] = []
    conflicts: list[str] = []

    for video_name, task in selected.items():
        sample_id = _sample_id_from_video_name(video_name)
        if not sample_id:
            missing.append(f"{video_name}: cannot parse sample id")
            continue
        source_path = _find_source_video(extracted_root, task=task, sample_id=sample_id)
        if source_path is None:
            missing.append(f"{video_name}: missing extracted video for task={task!r} sample={sample_id!r}")
            continue

        link_path = source_videos / video_name
        if os.path.lexists(link_path):
            if link_path.is_symlink() and link_path.resolve() == source_path.resolve():
                existing += 1
            else:
                conflicts.append(f"{link_path} exists and points somewhere else")
            continue

        os.symlink(source_path, link_path)
        linked += 1

    print(f"selected_videos={len(selected)}")
    print(f"linked={linked}")
    print(f"existing={existing}")
    print(f"missing={len(missing)}")
    print(f"conflicts={len(conflicts)}")
    for item in missing[:20]:
        print(f"missing: {item}")
    for item in conflicts[:20]:
        print(f"conflict: {item}")
    if missing or conflicts:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", default=str(DEFAULT_ANNO))
    parser.add_argument("--extracted-root", default=str(DEFAULT_EXTRACTED_ROOT))
    parser.add_argument("--source-videos", default=str(DEFAULT_SOURCE_VIDEOS))
    parser.add_argument("--task-filter", default=DEFAULT_TASK_FILTER)
    return parser.parse_args()


def _task_filter(value: str) -> set[str]:
    return {item.strip().lower() for item in value.replace(";", ",").split(",") if item.strip()}


def _selected_videos(entries: list[Any], *, task_filter: set[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for raw_entry in entries:
        entry_list = raw_entry if isinstance(raw_entry, list) else [raw_entry]
        for entry in entry_list:
            video_name = os.path.basename(str(entry.get("video_path", "")))
            if not video_name:
                continue
            for question in entry.get("questions", []):
                task = str(question.get("task_type", "")).strip()
                if task.lower() not in task_filter:
                    continue
                previous = selected.setdefault(video_name, task)
                if previous != task:
                    raise ValueError(f"Video {video_name!r} has multiple task types: {previous!r}, {task!r}")
    return selected


def _sample_id_from_video_name(video_name: str) -> str:
    match = re.match(r"^(sample_\d+)", Path(video_name).stem)
    return match.group(1) if match else ""


def _find_source_video(extracted_root: Path, *, task: str, sample_id: str) -> Path | None:
    direct_path = extracted_root / task / sample_id / "video.mp4"
    if direct_path.exists():
        return direct_path
    task_key = task.strip().lower()
    for task_dir in extracted_root.iterdir():
        if not task_dir.is_dir() or task_dir.name.strip().lower() != task_key:
            continue
        candidate = task_dir / sample_id / "video.mp4"
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    main()
