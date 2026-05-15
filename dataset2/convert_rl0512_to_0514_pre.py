#!/usr/bin/env python3
"""Convert the 0512 MCQ RL file to the 0514 canonical query_events schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "rl_0512.jsonl"
DEFAULT_OUTPUT = ROOT / "rl_0514_pre.jsonl"

TOP_KEYS = {
    "dataset",
    "video_id",
    "video",
    "frames_dir",
    "sample_fps",
    "frame_count",
    "frame_id_base",
    "query_events",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def require_text(value: Any, *, row_index: int, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"row {row_index}: missing {field}")
    return text


def normalize_options(value: Any, *, row_index: int) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"row {row_index}: options must be a list")
    options = [str(item).strip() for item in value if str(item).strip()]
    if len(options) < 2:
        raise ValueError(f"row {row_index}: expected at least 2 options")
    if len(options) > 26:
        raise ValueError(f"row {row_index}: too many options: {len(options)}")
    return options


def option_letter(index: int) -> str:
    return chr(ord("A") + index)


def letter_to_index(letter: str) -> int:
    return ord(letter.upper()) - ord("A")


def normalize_gt(value: Any, options: list[str], answer: str, *, row_index: int) -> str:
    if isinstance(value, bool):
        pass
    elif isinstance(value, int):
        if 0 <= value < len(options):
            return option_letter(value)
    elif isinstance(value, float) and value.is_integer():
        index = int(value)
        if 0 <= index < len(options):
            return option_letter(index)
    else:
        text = str(value or "").strip()
        if text:
            upper = text.upper()
            if len(upper) == 1 and "A" <= upper <= "Z":
                index = letter_to_index(upper)
                if 0 <= index < len(options):
                    return upper
            if upper[0:1].isalpha() and (len(upper) == 1 or upper[1:2] in {".", ")", " "}):
                index = letter_to_index(upper[0])
                if 0 <= index < len(options):
                    return upper[0]
            if text.isdigit():
                index = int(text)
                if 0 <= index < len(options):
                    return option_letter(index)

    normalized_answer = normalize_for_match(answer)
    for index, option in enumerate(options):
        if normalize_for_match(option) == normalized_answer:
            return option_letter(index)
    raise ValueError(f"row {row_index}: cannot map gt={value!r} answer={answer!r} to options")


def normalize_for_match(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def build_mcq_content(question: str, options: list[str]) -> str:
    lines = [question, "Options:"]
    for index, option in enumerate(options):
        lines.append(f"{option_letter(index)}. {option}")
    return "\n".join(lines)


def local_video_path(dataset: str, video_id: str, value: Any) -> str:
    text = str(value or "").strip()
    prefix = f"{dataset}/"
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.startswith("video/"):
        return text
    return f"video/{video_id}"


def row_to_0514(row: dict[str, Any], *, row_index: int) -> dict[str, Any]:
    dataset = require_text(row.get("dataset"), row_index=row_index, field="dataset")
    video_id = require_text(row.get("video_id"), row_index=row_index, field="video_id")
    question = require_text(row.get("question"), row_index=row_index, field="question")
    answer = require_text(row.get("answer"), row_index=row_index, field="answer")
    options = normalize_options(row.get("options"), row_index=row_index)
    gt = normalize_gt(row.get("gt"), options, answer, row_index=row_index)
    answer_text = options[letter_to_index(gt)]
    frame_count = int(row.get("frame_count") or 0)
    if frame_count <= 0:
        raise ValueError(f"row {row_index}: invalid frame_count={row.get('frame_count')!r}")
    sample_fps = float(row.get("sample_fps", row.get("fps", 1.0)) or 1.0)
    frame_id_base = int(row.get("frame_id_base", 0) or 0)
    event_time = float(frame_count)
    video = local_video_path(dataset, video_id, row.get("video"))
    frames_dir = local_video_path(dataset, video_id, row.get("frames_dir") or video)

    return {
        "dataset": dataset,
        "video_id": video_id,
        "video": video,
        "frames_dir": frames_dir,
        "sample_fps": sample_fps,
        "frame_count": frame_count,
        "frame_id_base": frame_id_base,
        "query_events": [
            {
                "qid": "q0",
                "time": event_time,
                "content": build_mcq_content(question, options),
                "answer_type": "mcq",
                "answer_policy": "answer_when_asked",
                "options": options,
                "answer_events": [
                    {
                        "time": event_time,
                        "gt": gt,
                        "answer": answer_text,
                        "content": f"{gt}. {answer_text}",
                    }
                ],
            }
        ],
    }


def validate_output(rows: list[dict[str, Any]]) -> None:
    for row_index, row in enumerate(rows):
        extra_top = set(row) - TOP_KEYS
        if extra_top:
            raise ValueError(f"row {row_index}: extra top-level keys: {sorted(extra_top)}")
        dataset = str(row["dataset"])
        video_id = str(row["video_id"])
        frame_dir = ROOT / dataset / str(row["frames_dir"])
        if not frame_dir.is_dir():
            raise FileNotFoundError(f"row {row_index}: missing frames dir {frame_dir}")
        frame_count = int(row["frame_count"])
        events = row.get("query_events")
        if not isinstance(events, list) or len(events) != 1:
            raise ValueError(f"row {row_index}: expected exactly one query_event")
        event = events[0]
        if event.get("answer_type") != "mcq":
            raise ValueError(f"row {row_index}: expected mcq answer_type")
        if event.get("answer_policy") != "answer_when_asked":
            raise ValueError(f"row {row_index}: expected answer_when_asked")
        if float(event["time"]) != float(frame_count):
            raise ValueError(f"{video_id}: query time must equal frame_count")
        answers = event.get("answer_events")
        if not isinstance(answers, list) or len(answers) != 1:
            raise ValueError(f"row {row_index}: expected exactly one answer_event")
        answer = answers[0]
        if float(answer["time"]) != float(frame_count):
            raise ValueError(f"{video_id}: answer time must equal frame_count")
        gt = str(answer.get("gt") or "").strip()
        if not gt or len(gt) != 1 or not gt.isalpha():
            raise ValueError(f"{video_id}: invalid answer gt={gt!r}")


def summarize(input_rows: list[dict[str, Any]], output_rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    dataset_counts = Counter(str(row.get("dataset") or "") for row in output_rows)
    option_counts = Counter(len(row["query_events"][0].get("options") or []) for row in output_rows)
    difficulty_counts = Counter(str(row.get("difficulty_v2") or "") for row in input_rows)
    student_counts = Counter(str(row.get("student_pass_count") if row.get("student_pass_count") is not None else "") for row in input_rows)
    gemini_counts = Counter(str(row.get("gemini_pass_count") if row.get("gemini_pass_count") is not None else "None") for row in input_rows)
    last4_counts = Counter(str(row.get("last4_pass") if row.get("last4_pass") is not None else "") for row in input_rows)
    frame_counts = [int(row["frame_count"]) for row in output_rows]
    return {
        "output_path": str(output_path),
        "input_rows": len(input_rows),
        "output_rows": len(output_rows),
        "query_events": sum(len(row.get("query_events") or []) for row in output_rows),
        "answer_events": sum(len(event.get("answer_events") or []) for row in output_rows for event in row.get("query_events") or []),
        "schema": "0514_canonical_query_events",
        "answer_type_counts": {"mcq": len(output_rows)},
        "answer_policy_counts": {"answer_when_asked": len(output_rows)},
        "dataset_counts": dict(dataset_counts.most_common()),
        "option_count_distribution": {str(key): option_counts[key] for key in sorted(option_counts)},
        "difficulty_v2_counts": dict(difficulty_counts.most_common()),
        "student_pass_count_counts": dict(student_counts.most_common()),
        "gemini_pass_count_counts": dict(gemini_counts.most_common()),
        "last4_pass_counts": dict(last4_counts.most_common()),
        "frame_count": {
            "min": min(frame_counts) if frame_counts else None,
            "max": max(frame_counts) if frame_counts else None,
        },
        "time_rule": "query_events[0].time == answer_events[0].time == frame_count",
        "path_rule": "video/frames_dir are relative to dataset2/{dataset}",
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    summary_path = (args.summary or output_path.with_suffix(output_path.suffix + ".summary.json")).resolve()
    input_rows = read_jsonl(input_path)
    output_rows = [row_to_0514(row, row_index=index) for index, row in enumerate(input_rows)]
    validate_output(output_rows)
    write_jsonl(output_path, output_rows)
    summary = summarize(input_rows, output_rows, output_path)
    write_json(summary_path, summary)
    print(f"wrote {len(output_rows)} rows -> {output_path}")
    print(f"wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
