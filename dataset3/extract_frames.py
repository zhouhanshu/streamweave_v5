"""Extract frames for VideoHallucer or EventHallusion using the framework's FrameStore.

Reads dataset3/<benchmark>/<benchmark>.json, dedups by video_id, and calls
FrameStore.ensure_frames() for each unique video. Frames go to
dataset3/<benchmark>/video/<video_id>/000000.jpg, ... with manifest.json
written in the format the framework's loader expects.

Usage:
    python dataset3/extract_frames.py --benchmark videohallucer
    python dataset3/extract_frames.py --benchmark eventhallusion --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path("/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5")
sys.path.insert(0, str(REPO_ROOT))

from streamweave.config import DatasetConfig  # noqa: E402
from streamweave.frame_store import FrameStore  # noqa: E402

DATASET_ROOT = REPO_ROOT / "dataset3"
SAMPLE_FPS = 1.0
MAX_FRAMES = 0  # 0 = no cap


# benchmark → (json_relpath, raw_video_root)
BENCHMARKS: dict[str, tuple[Path, Path]] = {
    "videohallucer": (
        DATASET_ROOT / "videohallucer" / "videohallucer.json",
        DATASET_ROOT / "raw" / "videohallucer",
    ),
    "eventhallusion": (
        DATASET_ROOT / "eventhallusion" / "eventhallusion.json",
        DATASET_ROOT / "raw" / "eventhallusion" / "videos" / "videos",
    ),
}


def _dedup_videos(entries: list[dict], raw_video_root: Path) -> list[tuple[str, str]]:
    """Return [(video_id, abs_video_path)], one per unique video_id."""
    seen: dict[str, str] = {}
    for entry in entries:
        vid = entry["video_id"]
        if vid in seen:
            continue
        seen[vid] = str(raw_video_root / entry["video"])
    return list(seen.items())


def _extract_one(args: tuple[str, str, str]) -> tuple[str, bool, str]:
    benchmark, video_id, video_path = args
    config = DatasetConfig(
        dataset_root=str(DATASET_ROOT),
        dataset_name=benchmark,
        video_root="",
        image_ext="jpg",
        jpeg_quality=95,
        overwrite_frames=False,
    )
    store = FrameStore(config)
    try:
        frames = store.ensure_frames(
            dataset_name=benchmark,
            video_id=video_id,
            video_path=video_path,
            sample_fps=SAMPLE_FPS,
            max_frames=MAX_FRAMES,
        )
        return video_id, True, f"frames={len(frames)}"
    except Exception as exc:  # noqa: BLE001
        return video_id, False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only extract first N videos (debug)")
    parser.add_argument("--video-id", type=str, default="", help="extract a single video_id (debug)")
    args = parser.parse_args()

    json_path, raw_video_root = BENCHMARKS[args.benchmark]
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run convert_{args.benchmark}.py first.", file=sys.stderr)
        sys.exit(1)

    entries = json.loads(json_path.read_text())
    todo = _dedup_videos(entries, raw_video_root)
    if args.video_id:
        todo = [(vid, path) for vid, path in todo if vid == args.video_id]
        if not todo:
            print(f"video_id {args.video_id!r} not found")
            sys.exit(1)
    if args.limit:
        todo = todo[: args.limit]

    print(f"Benchmark: {args.benchmark}")
    print(f"Unique videos to extract: {len(todo)}")
    print(f"Workers: {args.workers}, sample_fps={SAMPLE_FPS}, max_frames={MAX_FRAMES}")
    print(f"Output root: {DATASET_ROOT / args.benchmark / 'video'}")

    t0 = time.time()
    ok_count = 0
    fail_count = 0
    failures: list[tuple[str, str]] = []
    jobs = [(args.benchmark, vid, path) for vid, path in todo]

    if args.workers <= 1:
        for i, item in enumerate(jobs, 1):
            vid, ok, msg = _extract_one(item)
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                failures.append((vid, msg))
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] ok={ok_count} fail={fail_count} elapsed={time.time() - t0:.1f}s")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_extract_one, item): item[1] for item in jobs}
            done = 0
            for future in as_completed(futures):
                vid, ok, msg = future.result()
                done += 1
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    failures.append((vid, msg))
                if done % 20 == 0 or done == len(jobs):
                    print(f"[{done}/{len(jobs)}] ok={ok_count} fail={fail_count} elapsed={time.time() - t0:.1f}s")

    print()
    print(f"Done in {time.time() - t0:.1f}s. ok={ok_count} fail={fail_count}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for vid, msg in failures[:20]:
            print(f"  {vid}: {msg}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
