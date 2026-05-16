#!/usr/bin/env python3
"""Pre-extract StreamingBench frames with the same FrameStore used by eval."""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.runner import load_samples
from streamweave.config import load_eval_config
from streamweave.frame_store import FrameStore


def main() -> None:
    args = parse_args()
    cfg = load_eval_config(args.config)
    if args.split:
        cfg.benchmark_args["split"] = args.split
    if args.task_filter:
        cfg.benchmark_args["task_filter"] = args.task_filter
    if args.group_by_video is not None:
        cfg.benchmark_args["group_by_video"] = args.group_by_video
    if args.limit:
        cfg.benchmark_args["limit"] = args.limit

    samples = load_samples(cfg)
    videos = _unique_videos(samples)
    if args.limit_videos:
        videos = videos[: args.limit_videos]
    total = len(videos)
    print(f"videos={total} split={cfg.benchmark_args.get('split')} task_filter={cfg.benchmark_args.get('task_filter', '')}", flush=True)
    print(
        f"frame_root={Path(cfg.dataset.dataset_root) / cfg.dataset.dataset_name / 'video'} "
        f"fps={cfg.runtime.sample_fps} max_frames={cfg.runtime.max_frames}",
        flush=True,
    )
    if not videos:
        return

    tasks = [
        {
            "dataset": cfg.dataset,
            "dataset_name": cfg.dataset.dataset_name or cfg.benchmark,
            "video_id": video_id,
            "video_path": video_path,
            "sample_fps": cfg.runtime.sample_fps,
            "max_frames": cfg.runtime.max_frames,
            "suppress_native_stderr": bool(args.suppress_native_stderr),
        }
        for video_id, video_path in videos
    ]
    workers = max(1, min(int(args.workers), total))
    done = 0
    failed: list[dict[str, Any]] = []
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_extract_one, tasks):
            done += 1
            status = result.get("status", "")
            video_id = result.get("video_id", "")
            frames = result.get("frames", 0)
            if status != "ok":
                failed.append(result)
                print(f"[{done}/{total}] FAIL {video_id}: {result.get('error')}", flush=True)
            else:
                print(f"[{done}/{total}] ok {video_id}: frames={frames}", flush=True)

    print(f"complete={total - len(failed)} failed={len(failed)} total={total}", flush=True)
    for item in failed[:20]:
        print(f"failed: {item.get('video_id')}: {item.get('error')}", flush=True)
    if failed:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="outputs/streamingbench_omni_mislead_anomaly_exp8_step40/run_config.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", default="")
    parser.add_argument("--task-filter", default="")
    parser.add_argument("--group-by-video", dest="group_by_video", action="store_true", default=None)
    parser.add_argument("--no-group-by-video", dest="group_by_video", action="store_false")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--suppress-native-stderr", action="store_true", default=True)
    parser.add_argument("--show-native-stderr", dest="suppress_native_stderr", action="store_false")
    return parser.parse_args()


def _unique_videos(samples: list[Any]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    videos: list[tuple[str, str]] = []
    for sample in samples:
        video_id = str(getattr(sample, "video_id", "") or "")
        video_path = str(getattr(sample, "video_path", "") or "")
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        videos.append((video_id, video_path))
    return videos


def _extract_one(task: dict[str, Any]) -> dict[str, Any]:
    video_id = str(task["video_id"])
    try:
        stderr_cm = _native_stderr_to_devnull() if task.get("suppress_native_stderr") else contextlib.nullcontext()
        with stderr_cm:
            store = FrameStore(task["dataset"])
            frames = store.ensure_frames(
                dataset_name=str(task["dataset_name"]),
                video_id=video_id,
                video_path=str(task["video_path"]),
                sample_fps=float(task["sample_fps"]),
                max_frames=int(task["max_frames"]),
            )
        return {"status": "ok", "video_id": video_id, "frames": len(frames)}
    except Exception as exc:
        return {"status": "error", "video_id": video_id, "error": repr(exc)}


@contextlib.contextmanager
def _native_stderr_to_devnull():
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


if __name__ == "__main__":
    main()
