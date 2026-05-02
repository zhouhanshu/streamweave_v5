#!/usr/bin/env python3
"""Run the VideoXum streaming-QA synthesis pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.synthesize.build_windows import build_window_record
from data_engine.synthesize.export_annotations import export_qa_annotations
from data_engine.synthesize.filter_qa import filter_qa_record, records_by_video_id
from data_engine.synthesize.gen_captions import generate_caption_record
from data_engine.synthesize.gen_qa import generate_qa_record
from data_engine.synthesize.io_utils import JsonDict, append_jsonl, read_json_or_jsonl, read_jsonl, write_jsonl
from data_engine.synthesize.vlm_client import VLMClient


def main() -> None:
    args = parse_args()
    stages = resolve_stages(args.stage)
    paths = output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "windows" in stages:
        run_windows(args, paths["windows"])
    if "captions" in stages:
        run_captions(args, paths["windows"], paths["captions"])
    if "qa" in stages:
        run_qa(args, paths["captions"], paths["qa"])
    if "filter" in stages:
        run_filter(args, paths["captions"], paths["qa"], paths["filtered"])
    if "export" in stages:
        run_export(args, paths["filtered"], args.final_output or paths["annotations"])


def run_windows(args: argparse.Namespace, output_path: Path) -> None:
    if should_skip(output_path, args.overwrite):
        print(f"[windows] skip existing {output_path}", flush=True)
        return
    annotations = select_annotations(
        read_json_or_jsonl(args.input),
        offset=args.offset,
        limit=args.limit,
        sample_ids=args.sample_ids,
    )
    records = [
        build_window_record(
            item,
            max_keyframes=args.max_keyframes,
            min_selected_gap=args.min_selected_gap,
            key_radius=args.key_radius,
            merge_keyframe_gap=args.merge_keyframe_gap,
            normal_window_size=args.normal_window_size,
        )
        for item in annotations
    ]
    write_jsonl(records, output_path)
    print(f"[windows] saved {len(records)} record(s) -> {output_path}", flush=True)


def run_captions(args: argparse.Namespace, windows_path: Path, output_path: Path) -> None:
    if should_skip(output_path, args.overwrite):
        print(f"[captions] skip existing {output_path}", flush=True)
        return
    reset_output(output_path)
    records = read_jsonl(windows_path)
    caption_client = make_client(args.caption_backend, max_tokens=args.caption_max_tokens)
    facts_client = make_client(args.facts_backend, max_tokens=args.facts_max_tokens)
    for idx, record in enumerate(records, start=1):
        result = generate_caption_record(
            record,
            caption_client,
            facts_client=facts_client,
            raw_data_root=args.raw_data_root,
        )
        append_jsonl(result, output_path)
        print(f"[captions] {idx}/{len(records)} video={record['video_id']} errors={len(result['errors'])}", flush=True)


def run_qa(args: argparse.Namespace, captions_path: Path, output_path: Path) -> None:
    if should_skip(output_path, args.overwrite):
        print(f"[qa] skip existing {output_path}", flush=True)
        return
    reset_output(output_path)
    records = read_jsonl(captions_path)
    client = make_client(args.qa_backend, max_tokens=args.qa_max_tokens)
    for idx, record in enumerate(records, start=1):
        result = generate_qa_record(
            record,
            client,
            num_questions=args.num_questions,
            max_attempts=args.qa_max_attempts,
            raw_response_chars=args.qa_raw_response_chars,
            raw_data_root=args.raw_data_root,
        )
        append_jsonl(result, output_path)
        print(
            f"[qa] {idx}/{len(records)} video={record['video_id']} "
            f"candidates={len(result['qa_candidates'])} errors={len(result['errors'])}",
            flush=True,
        )


def run_filter(args: argparse.Namespace, captions_path: Path, qa_path: Path, output_path: Path) -> None:
    if should_skip(output_path, args.overwrite):
        print(f"[filter] skip existing {output_path}", flush=True)
        return
    reset_output(output_path)
    captions = records_by_video_id(read_jsonl(captions_path))
    qa_records = read_jsonl(qa_path)
    filter_client = make_client(args.filter_backend, max_tokens=args.filter_max_tokens)
    for idx, qa_record in enumerate(qa_records, start=1):
        result = filter_qa_record(
            captions[qa_record["video_id"]],
            qa_record,
            filter_client,
            keep_per_video=args.keep_per_video,
            max_history_keyframes=args.max_history_keyframes,
            raw_data_root=args.raw_data_root,
        )
        append_jsonl(result, output_path)
        print(
            f"[filter] {idx}/{len(qa_records)} video={qa_record['video_id']} "
            f"accepted={result['accepted_count']} errors={len(result['errors'])}",
            flush=True,
        )


def run_export(args: argparse.Namespace, filtered_path: Path, output_path: Path) -> None:
    if should_skip(output_path, args.overwrite):
        print(f"[export] skip existing {output_path}", flush=True)
        return
    rows = export_qa_annotations(read_jsonl(filtered_path), output_path)
    print(f"[export] saved {len(rows)} annotation row(s) -> {output_path}", flush=True)


def make_client(backend: str, *, max_tokens: int) -> VLMClient:
    return VLMClient.from_backend(backend, max_tokens=max_tokens)


def select_annotations(data: object, *, offset: int, limit: int, sample_ids: list[str]) -> list[JsonDict]:
    if not isinstance(data, list):
        raise ValueError("Input annotations must be a JSON list or JSONL rows.")
    rows = [item for item in data if isinstance(item, dict) and item.get("status", "ok") == "ok"]
    if sample_ids:
        wanted = set(sample_ids)
        rows = [item for item in rows if str(item.get("video_id", "")) in wanted]
    rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    return rows


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "windows": output_dir / "windows.jsonl",
        "captions": output_dir / "captions.jsonl",
        "qa": output_dir / "qa_candidates.jsonl",
        "filtered": output_dir / "qa_filtered.jsonl",
        "annotations": output_dir / "annotations_qa.jsonl",
    }


def resolve_stages(stage: str) -> list[str]:
    if stage == "all":
        return ["windows", "captions", "qa", "filter", "export"]
    return [stage]


def should_skip(path: Path, overwrite: bool) -> bool:
    return path.exists() and not overwrite


def reset_output(path: Path) -> None:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("raw_data/anno.jsonl"))
    parser.add_argument("--raw-data-root", type=Path, default=Path("raw_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_engine/synthesize/outputs"))
    parser.add_argument("--final-output", type=Path, default=None)
    parser.add_argument("--stage", choices=("all", "windows", "captions", "qa", "filter", "export"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--min-selected-gap", type=int, default=10)
    parser.add_argument("--key-radius", type=int, default=2)
    parser.add_argument("--merge-keyframe-gap", type=int, default=8)
    parser.add_argument("--normal-window-size", type=int, default=5)

    parser.add_argument("--num-questions", type=int, default=8)
    parser.add_argument("--qa-max-attempts", type=int, default=3)
    parser.add_argument(
        "--qa-raw-response-chars",
        type=int,
        default=12000,
        help="Characters of each raw QA model response to store in qa_candidates.jsonl. Use -1 for full text, 0 to omit.",
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
