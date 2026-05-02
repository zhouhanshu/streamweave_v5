#!/usr/bin/env python3
"""Export filtered QA records as flat video annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.synthesize.io_utils import JsonDict, read_jsonl, write_jsonl
from data_engine.synthesize.mcq_utils import choice_to_index, correct_choice, normalize_options, strip_option_prefix


VIDEO_FIELDS = (
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


def export_qa_annotations(filtered_records: list[JsonDict], output_path: Path) -> list[JsonDict]:
    rows = build_qa_annotations(filtered_records)
    write_jsonl(rows, output_path)
    return rows


def build_qa_annotations(filtered_records: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for record in filtered_records:
        row = make_video_row(record)
        accepted = [qa for qa in record.get("accepted_qa", []) if isinstance(qa, dict)]
        if accepted:
            qa_fields = make_qa_fields(accepted[0], len(rows))
            if qa_fields:
                row.update(qa_fields)
        rows.append(row)
    return rows


def make_video_row(record: JsonDict) -> JsonDict:
    source = record.get("source_annotation", {})
    if not isinstance(source, dict) or not source:
        source = record
    return {field: source[field] for field in VIDEO_FIELDS if field in source}


def make_qa_fields(qa: JsonDict, row_id: int) -> JsonDict:
    options = [strip_option_prefix(option) for option in normalize_options(qa.get("options", []))]
    gt = choice_to_index(correct_choice(qa))
    if len(options) != 4 or gt is None:
        return {}

    answer = str(qa.get("answer_text", "")).strip() or options[gt]
    task = str(qa.get("type", "")).strip()
    question = str(qa.get("question", "")).strip()
    if not task or not question or not answer:
        return {}

    fields: JsonDict = {
        "id": row_id,
        "task": task,
        "question": question,
        "answer": answer,
        "options": options,
        "gt": gt,
        "evidence_frame_ids": qa.get("evidence_frame_ids", []),
    }
    query_time = int(qa["query_time"])
    answer_time = int(qa["answer_time"])
    if task == "forward":
        fields["ask_time"] = query_time
        fields["clue_time"] = answer_time
    else:
        fields["realtime"] = query_time
    return fields


def run_cli(args: argparse.Namespace) -> None:
    rows = export_qa_annotations(read_jsonl(args.filtered), args.output)
    print(f"Saved {len(rows)} annotation row(s) to {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered", type=Path, default=Path("data_engine/synthesize/outputs/qa_filtered.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data_engine/synthesize/outputs/annotations_qa.jsonl"))
    return parser.parse_args()


if __name__ == "__main__":
    run_cli(parse_args())
