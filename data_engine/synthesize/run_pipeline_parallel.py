#!/usr/bin/env python3
"""Run the synthesis pipeline in parallel at video granularity."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.synthesize.build_windows import build_window_record
from data_engine.synthesize.export_annotations import build_qa_annotations
from data_engine.synthesize.filter_qa import filter_qa_record
from data_engine.synthesize.gen_captions import generate_caption_record
from data_engine.synthesize.gen_qa import DEFAULT_QA_MAX_ATTEMPTS, DEFAULT_RAW_RESPONSE_CHARS, generate_qa_record
from data_engine.synthesize.io_utils import JsonDict, append_jsonl, read_json_or_jsonl, write_json
from data_engine.synthesize.run_pipeline import select_annotations
from data_engine.synthesize.vlm_client import VLMClient


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_can_write_outputs(args)
    prepare_output_files(args)

    annotations = select_annotations(
        read_json_or_jsonl(args.input),
        offset=args.offset,
        limit=args.limit,
        sample_ids=args.sample_ids,
    )
    if not annotations:
        raise ValueError("No input annotations selected.")

    indexed_annotations = list(enumerate(annotations))
    resume_summaries = load_resume_summaries(args, indexed_annotations) if args.resume else []
    completed_indices = {int(item["index"]) for item in resume_summaries}
    pending_annotations = [
        (idx, annotation)
        for idx, annotation in indexed_annotations
        if idx not in completed_indices
    ]

    started = time.time()
    print(
        f"[parallel] selected={len(indexed_annotations)} workers={args.workers} "
        f"resume={args.resume} completed={len(resume_summaries)} "
        f"pending={len(pending_annotations)} "
        f"caption={args.caption_backend} facts={args.facts_backend} "
        f"qa={args.qa_backend} filter={args.filter_backend}",
        flush=True,
    )

    per_video = resume_summaries + run_parallel(
        pending_annotations,
        args,
        total_selected=len(indexed_annotations),
        completed_count=len(resume_summaries),
    )
    per_video.sort(key=lambda item: int(item["index"]))
    summary = build_summary(per_video, elapsed_seconds=time.time() - started)
    write_json(summary, args.output_dir / "summary.json")
    print_summary(summary, args.output_dir)


def run_parallel(
    indexed_annotations: list[tuple[int, JsonDict]],
    args: argparse.Namespace,
    *,
    total_selected: int,
    completed_count: int,
) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    workers = max(1, int(args.workers))
    if not indexed_annotations:
        print("[parallel] no pending videos; all selected videos are already completed", flush=True)
        return summaries

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one_video, idx, annotation, args): idx
            for idx, annotation in indexed_annotations
        }
        total = len(futures)
        for done_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            append_result_outputs(result, args)
            summary = summarize_result(result)
            append_jsonl(summary, args.output_dir / "progress.jsonl")
            summaries.append(summary)
            video_id = result["video_id"]
            print(
                f"[parallel] {completed_count + done_count}/{total_selected} "
                f"pending={done_count}/{total} video={video_id} "
                f"candidates={summary['candidates']} "
                f"accepted={summary['accepted']} "
                f"dropped={summary['dropped']} "
                f"errors={summary['errors']} "
                f"seconds={summary['seconds']:.1f}",
                flush=True,
            )
    return summaries


def load_resume_summaries(
    args: argparse.Namespace,
    indexed_annotations: list[tuple[int, JsonDict]],
) -> list[JsonDict]:
    progress_path = args.output_dir / "progress.jsonl"
    if not progress_path.exists():
        return []

    selected_video_ids = {
        idx: annotation_video_id(annotation, idx)
        for idx, annotation in indexed_annotations
    }
    completed_by_index: dict[int, JsonDict] = {}
    invalid_rows = 0
    mismatched_rows = 0
    duplicate_rows = 0

    for row in read_jsonl_lenient(progress_path):
        try:
            idx = int(row["index"])
            video_id = str(row["video_id"])
        except (KeyError, TypeError, ValueError):
            invalid_rows += 1
            continue

        if selected_video_ids.get(idx) != video_id:
            mismatched_rows += 1
            continue
        if idx in completed_by_index:
            duplicate_rows += 1
        completed_by_index[idx] = normalize_progress_summary(row)

    if invalid_rows or mismatched_rows or duplicate_rows:
        print(
            "[resume] "
            f"ignored_invalid={invalid_rows} "
            f"ignored_mismatched={mismatched_rows} "
            f"deduped={duplicate_rows}",
            flush=True,
        )
    print(
        f"[resume] loaded {len(completed_by_index)} completed video(s) "
        f"from {progress_path}",
        flush=True,
    )
    return list(completed_by_index.values())


def read_jsonl_lenient(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                print(f"[resume] ignored malformed progress row {line_no} in {path}", flush=True)
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def normalize_progress_summary(row: JsonDict) -> JsonDict:
    return {
        "index": int(row["index"]),
        "video_id": str(row["video_id"]),
        "candidates": int(row.get("candidates", 0)),
        "accepted": int(row.get("accepted", 0)),
        "verified": int(row.get("verified", 0)),
        "dropped": int(row.get("dropped", 0)),
        "errors": int(row.get("errors", 0)),
        "seconds": round(float(row.get("seconds", 0.0)), 3),
    }


def annotation_video_id(annotation: JsonDict, index: int) -> str:
    return str(annotation.get("video_id", f"index_{index}"))


def process_one_video(index: int, annotation: JsonDict, args: argparse.Namespace) -> JsonDict:
    started = time.time()
    video_id = annotation_video_id(annotation, index)
    window_record: JsonDict | None = None
    caption_record: JsonDict | None = None
    qa_record: JsonDict | None = None
    filtered_record: JsonDict | None = None
    errors: list[JsonDict] = []
    try:
        window_record = build_window_record(
            annotation,
            max_keyframes=args.max_keyframes,
            min_selected_gap=args.min_selected_gap,
            key_radius=args.key_radius,
            merge_keyframe_gap=args.merge_keyframe_gap,
            normal_window_size=args.normal_window_size,
        )

        caption_client = VLMClient.from_backend(args.caption_backend, max_tokens=args.caption_max_tokens)
        facts_client = VLMClient.from_backend(args.facts_backend, max_tokens=args.facts_max_tokens)
        caption_record = generate_caption_record(
            window_record,
            caption_client,
            facts_client=facts_client,
            raw_data_root=args.raw_data_root,
        )

        qa_client = VLMClient.from_backend(args.qa_backend, max_tokens=args.qa_max_tokens)
        qa_record = generate_qa_record(
            caption_record,
            qa_client,
            num_questions=args.num_questions,
            max_attempts=args.qa_max_attempts,
            raw_response_chars=args.qa_raw_response_chars,
            raw_data_root=args.raw_data_root,
        )

        filter_client = VLMClient.from_backend(args.filter_backend, max_tokens=args.filter_max_tokens)
        filtered_record = filter_qa_record(
            caption_record,
            qa_record,
            filter_client,
            keep_per_video=args.keep_per_video,
            max_history_keyframes=args.max_history_keyframes,
            raw_data_root=args.raw_data_root,
        )
    except Exception as exc:
        errors.append({"stage": "worker", "error": repr(exc), "traceback": traceback.format_exc()})
        filtered_record = fallback_filtered_record(annotation, video_id, errors)

    if caption_record is not None:
        errors.extend(caption_record.get("errors", []))
    if qa_record is not None:
        errors.extend(qa_record.get("errors", []))
    if filtered_record is not None:
        errors.extend(filtered_record.get("errors", []))

    return {
        "index": index,
        "video_id": video_id,
        "window_record": window_record,
        "caption_record": caption_record,
        "qa_record": qa_record or fallback_qa_record(annotation, video_id),
        "filtered_record": filtered_record or fallback_filtered_record(annotation, video_id, errors),
        "errors": errors,
        "elapsed_seconds": time.time() - started,
    }


def fallback_qa_record(annotation: JsonDict, video_id: str) -> JsonDict:
    return {
        "video_id": video_id,
        "video": annotation.get("frames_dir", ""),
        "frames_dir": annotation.get("frames_dir", ""),
        "sampled_frames": annotation.get("sampled_frames", 0),
        "selected_key_frame_ids": [],
        "requested_questions": 0,
        "qa_candidates": [],
        "errors": [],
    }


def fallback_filtered_record(annotation: JsonDict, video_id: str, errors: list[JsonDict]) -> JsonDict:
    return {
        "video_id": video_id,
        "source_annotation": source_annotation_from_raw(annotation),
        "video": annotation.get("frames_dir", ""),
        "frames_dir": annotation.get("frames_dir", ""),
        "selected_key_frame_ids": [],
        "accepted_qa": [],
        "accepted_count": 0,
        "verified_qa": [],
        "verified_count": 0,
        "errors": errors,
    }


def source_annotation_from_raw(annotation: JsonDict) -> JsonDict:
    fields = (
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
    return {field: annotation[field] for field in fields if field in annotation}


def append_result_outputs(result: JsonDict, args: argparse.Namespace) -> None:
    if result.get("window_record"):
        append_jsonl(result["window_record"], args.output_dir / "windows.jsonl")
    if result.get("caption_record"):
        append_jsonl(result["caption_record"], args.output_dir / "captions.jsonl")
    append_jsonl(result["qa_record"], args.output_dir / "qa_candidates.jsonl")
    append_jsonl(result["filtered_record"], args.output_dir / "qa_filtered.jsonl")

    annotation_rows = build_qa_annotations([result["filtered_record"]])
    if annotation_rows:
        row = annotation_rows[0]
        if "id" in row:
            row["id"] = int(result["index"])
        append_jsonl(row, annotation_output_path(args))


def summarize_result(result: JsonDict) -> JsonDict:
    qa_record = result.get("qa_record", {})
    filtered_record = result.get("filtered_record", {})
    return {
        "index": int(result["index"]),
        "video_id": result["video_id"],
        "candidates": len(qa_record.get("qa_candidates", [])),
        "accepted": int(filtered_record.get("accepted_count", 0)),
        "verified": int(filtered_record.get("verified_count", 0)),
        "dropped": int(filtered_record.get("dropped_count", 0)),
        "errors": len(result.get("errors", [])),
        "seconds": round(float(result.get("elapsed_seconds", 0.0)), 3),
    }


def build_summary(per_video: list[JsonDict], *, elapsed_seconds: float) -> JsonDict:
    videos = len(per_video)
    candidate_counts = [int(item["candidates"]) for item in per_video]
    accepted_counts = [int(item["accepted"]) for item in per_video]
    dropped_counts = [int(item["dropped"]) for item in per_video]
    return {
        "videos": videos,
        "videos_with_candidates": sum(1 for count in candidate_counts if count > 0),
        "total_candidates": sum(candidate_counts),
        "videos_with_accepted": sum(1 for count in accepted_counts if count > 0),
        "total_accepted": sum(accepted_counts),
        "videos_with_dropped_qa": sum(1 for count in dropped_counts if count > 0),
        "total_dropped_qa": sum(dropped_counts),
        "error_videos": sum(1 for item in per_video if int(item["errors"]) > 0),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "per_video": per_video,
    }


def print_summary(summary: JsonDict, output_dir: Path) -> None:
    print(
        "[summary] "
        f"videos={summary['videos']} "
        f"videos_with_candidates={summary['videos_with_candidates']} "
        f"total_candidates={summary['total_candidates']} "
        f"videos_with_accepted={summary['videos_with_accepted']} "
        f"total_accepted={summary['total_accepted']} "
        f"total_dropped_qa={summary['total_dropped_qa']} "
        f"error_videos={summary['error_videos']} "
        f"elapsed_seconds={summary['elapsed_seconds']}",
        flush=True,
    )
    print(f"[summary] outputs -> {output_dir}", flush=True)
    print("[summary] per video:", flush=True)
    for item in summary["per_video"]:
        print(
            "  "
            f"{item['index']:03d} {item['video_id']} "
            f"candidates={item['candidates']} accepted={item['accepted']} "
            f"verified={item['verified']} dropped={item['dropped']} errors={item['errors']} "
            f"seconds={item['seconds']}",
            flush=True,
        )


def ensure_can_write_outputs(args: argparse.Namespace) -> None:
    paths = [
        args.output_dir / "windows.jsonl",
        args.output_dir / "captions.jsonl",
        args.output_dir / "qa_candidates.jsonl",
        args.output_dir / "qa_filtered.jsonl",
        args.output_dir / "progress.jsonl",
        args.output_dir / "summary.json",
        annotation_output_path(args),
    ]
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    existing = [path for path in paths if path.exists()]
    if args.resume:
        progress_path = args.output_dir / "progress.jsonl"
        if existing and not progress_path.exists():
            raise FileNotFoundError(
                f"Cannot resume because {progress_path} does not exist. "
                "Use --overwrite to restart from scratch."
            )
        required = [path for path in paths if path.name != "summary.json"]
        missing = [path for path in required if progress_path.exists() and not path.exists()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:4])
            raise FileNotFoundError(f"Cannot resume because output file(s) are missing: {preview}")
        return

    if existing and not args.overwrite:
        preview = ", ".join(str(path) for path in existing[:4])
        raise FileExistsError(f"Output file(s) already exist. Use --overwrite or --resume: {preview}")


def prepare_output_files(args: argparse.Namespace) -> None:
    paths = [
        args.output_dir / "windows.jsonl",
        args.output_dir / "captions.jsonl",
        args.output_dir / "qa_candidates.jsonl",
        args.output_dir / "qa_filtered.jsonl",
        args.output_dir / "progress.jsonl",
        args.output_dir / "summary.json",
        annotation_output_path(args),
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite and path.exists():
            path.unlink()


def annotation_output_path(args: argparse.Namespace) -> Path:
    return args.final_output or args.output_dir / "annotations_qa.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("raw_data/anno.jsonl"))
    parser.add_argument("--raw-data-root", type=Path, default=Path("raw_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_engine/synthesize/outputs/parallel_debug_8"))
    parser.add_argument("--final-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos already listed in progress.jsonl and append new outputs.",
    )

    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--min-selected-gap", type=int, default=10)
    parser.add_argument("--key-radius", type=int, default=2)
    parser.add_argument("--merge-keyframe-gap", type=int, default=8)
    parser.add_argument("--normal-window-size", type=int, default=5)

    parser.add_argument("--num-questions", type=int, default=8)
    parser.add_argument("--qa-max-attempts", type=int, default=DEFAULT_QA_MAX_ATTEMPTS)
    parser.add_argument(
        "--qa-raw-response-chars",
        type=int,
        default=DEFAULT_RAW_RESPONSE_CHARS,
        help="Characters of each raw QA model response to store. Use -1 for full text, 0 to omit.",
    )
    parser.add_argument("--keep-per-video", type=int, default=1)
    parser.add_argument(
        "--max-history-keyframes",
        "--max-history-images",
        dest="max_history_keyframes",
        type=int,
        default=8,
        help="Maximum historical keyframe images used by forward/backward filtering.",
    )
    parser.add_argument("--caption-max-tokens", type=int, default=2048)
    parser.add_argument("--facts-max-tokens", type=int, default=2048)
    parser.add_argument("--qa-max-tokens", type=int, default=3072)
    parser.add_argument("--filter-max-tokens", type=int, default=2048)
    parser.add_argument("--caption-backend", default="qwen3vl")
    parser.add_argument("--facts-backend", default="qwen3vl")
    parser.add_argument("--qa-backend", default="gemini")
    parser.add_argument("--filter-backend", default="qwen3vl")
    return parser.parse_args()


if __name__ == "__main__":
    main()
