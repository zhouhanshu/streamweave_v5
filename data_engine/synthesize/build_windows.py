#!/usr/bin/env python3
"""Step 1: build VideoXum synthesis windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.synthesize.io_utils import JsonDict, clamp_int, read_json_or_jsonl, write_jsonl


def build_window_record(
    annotation: JsonDict,
    *,
    max_keyframes: int = 8,
    min_selected_gap: int = 10,
    key_radius: int = 2,
    merge_keyframe_gap: int = 8,
    normal_window_size: int = 5,
) -> JsonDict:
    sampled_frames = int(annotation["sampled_frames"])
    selected_keyframes = select_qa_keyframes(
        annotation.get("key_frame_ids", []),
        annotation.get("key_frame_scores", []),
        sampled_frames=sampled_frames,
        max_keyframes=max_keyframes,
        min_gap=min_selected_gap,
    )
    windows = build_windows(
        selected_keyframes,
        sampled_frames=sampled_frames,
        key_radius=key_radius,
        merge_keyframe_gap=merge_keyframe_gap,
        normal_window_size=normal_window_size,
    )
    source_annotation = {
        key: annotation[key]
        for key in (
            "dataset",
            "split",
            "video_id",
            "duration",
            "sampled_frames",
            "frame_count",
            "frames_dir",
            "frame_id_base",
            "key_frame_ids",
            "key_frame_count",
        )
        if key in annotation
    }
    return {
        "source_annotation": source_annotation,
        "dataset": annotation.get("dataset", "streamweave_data"),
        "source_dataset": annotation.get("source_dataset", "VideoXum"),
        "split": annotation.get("split", ""),
        "video_id": annotation["video_id"],
        "video": annotation["frames_dir"],
        "activitynet_video": annotation.get("activitynet_video", ""),
        "frames_dir": annotation["frames_dir"],
        "frame_name_format": annotation.get("frame_name_format", "{frame_id:06d}.jpg"),
        "frame_id_base": annotation.get("frame_id_base", 0),
        "fps": annotation.get("fps", 1.0),
        "duration": annotation.get("duration"),
        "sampled_frames": sampled_frames,
        "frame_count": annotation.get("frame_count", sampled_frames),
        "original_key_frame_ids": list(annotation.get("key_frame_ids", [])),
        "selected_key_frame_ids": selected_keyframes,
        "key_radius": key_radius,
        "normal_window_size": normal_window_size,
        "windows": windows,
    }


def select_qa_keyframes(
    keyframe_ids: list[int],
    keyframe_scores: list[float],
    *,
    sampled_frames: int,
    max_keyframes: int,
    min_gap: int,
) -> list[int]:
    valid = sorted({int(k) for k in keyframe_ids if 0 <= int(k) < sampled_frames})
    if not valid:
        return []
    scored = []
    for keyframe_id in valid:
        score = float(keyframe_scores[keyframe_id]) if keyframe_id < len(keyframe_scores) else 0.0
        scored.append((score, keyframe_id))
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected: list[int] = []
    for _, keyframe_id in scored:
        if all(abs(keyframe_id - chosen) >= min_gap for chosen in selected):
            selected.append(keyframe_id)
        if len(selected) >= max_keyframes:
            break
    if not selected:
        selected.append(scored[0][1])
    return sorted(selected)


def build_windows(
    selected_keyframe_ids: list[int],
    *,
    sampled_frames: int,
    key_radius: int = 2,
    merge_keyframe_gap: int = 8,
    normal_window_size: int = 5,
) -> list[JsonDict]:
    key_windows = build_key_windows(
        selected_keyframe_ids,
        sampled_frames=sampled_frames,
        key_radius=key_radius,
        merge_keyframe_gap=merge_keyframe_gap,
    )
    windows: list[JsonDict] = []
    cursor = 0
    normal_idx = 0
    key_idx = 0
    for key_window in key_windows:
        start = int(key_window["start_frame"])
        end = int(key_window["end_frame"])
        while cursor < start:
            normal_end = min(cursor + normal_window_size - 1, start - 1)
            windows.append(make_normal_window(normal_idx, cursor, normal_end))
            normal_idx += 1
            cursor = normal_end + 1
        key_window = dict(key_window)
        key_window["window_id"] = f"kw_{key_idx:03d}"
        windows.append(key_window)
        key_idx += 1
        cursor = max(cursor, end + 1)

    while cursor < sampled_frames:
        normal_end = min(cursor + normal_window_size - 1, sampled_frames - 1)
        windows.append(make_normal_window(normal_idx, cursor, normal_end))
        normal_idx += 1
        cursor = normal_end + 1
    return windows


def build_key_windows(
    selected_keyframe_ids: list[int],
    *,
    sampled_frames: int,
    key_radius: int,
    merge_keyframe_gap: int,
) -> list[JsonDict]:
    keyframes = sorted({int(k) for k in selected_keyframe_ids if 0 <= int(k) < sampled_frames})
    merged: list[JsonDict] = []
    for keyframe_id in keyframes:
        start = clamp_int(keyframe_id - key_radius, 0, sampled_frames - 1)
        end = clamp_int(keyframe_id + key_radius, 0, sampled_frames - 1)
        if merged:
            previous = merged[-1]
            previous_last_key = int(previous["source_keyframe_ids"][-1])
            should_merge = keyframe_id - previous_last_key < merge_keyframe_gap or start <= int(previous["end_frame"])
            if should_merge:
                previous["end_frame"] = max(int(previous["end_frame"]), end)
                previous["time"] = [int(previous["start_frame"]), int(previous["end_frame"])]
                previous["source_keyframe_ids"].append(keyframe_id)
                previous["frame_ids"] = list(range(int(previous["start_frame"]), int(previous["end_frame"]) + 1))
                previous["duration_frames"] = len(previous["frame_ids"])
                continue
        merged.append(
            {
                "window_id": "",
                "type": "keyframe",
                "time": [start, end],
                "start_frame": start,
                "end_frame": end,
                "frame_ids": list(range(start, end + 1)),
                "duration_frames": end - start + 1,
                "source_keyframe_ids": [keyframe_id],
                "source_keyframe_id": keyframe_id,
            }
        )
    for window in merged:
        if len(window["source_keyframe_ids"]) != 1:
            window["source_keyframe_id"] = None
    return merged


def make_normal_window(index: int, start: int, end: int) -> JsonDict:
    return {
        "window_id": f"nw_{index:03d}",
        "type": "normal",
        "time": [start, end],
        "start_frame": start,
        "end_frame": end,
        "frame_ids": list(range(start, end + 1)),
        "duration_frames": end - start + 1,
        "source_keyframe_ids": [],
        "source_keyframe_id": None,
    }


def run_cli(args: argparse.Namespace) -> None:
    annotations = read_json_or_jsonl(args.input)
    selected = [
        item
        for item in annotations[args.offset :]
        if isinstance(item, dict) and item.get("status", "ok") == "ok"
    ]
    if args.limit:
        selected = selected[: args.limit]
    records = [
        build_window_record(
            item,
            max_keyframes=args.max_keyframes,
            min_selected_gap=args.min_selected_gap,
            key_radius=args.key_radius,
            merge_keyframe_gap=args.merge_keyframe_gap,
            normal_window_size=args.normal_window_size,
        )
        for item in selected
    ]
    write_jsonl(records, args.output)
    print(f"Saved {len(records)} window record(s) to {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("raw_data/anno.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data_engine/synthesize/outputs/windows.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--min-selected-gap", type=int, default=10)
    parser.add_argument("--key-radius", type=int, default=2)
    parser.add_argument("--merge-keyframe-gap", type=int, default=8)
    parser.add_argument("--normal-window-size", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run_cli(parse_args())
