#!/usr/bin/env python3
"""Split Streamo 0514 data into unable, one-answer, and multi-answer files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_DATASET2_ROOT = Path("/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset2")
DEFAULT_INPUT = "Streamo-Instruct-465K/query_events_0514_filtered.jsonl"
DEFAULT_OUTPUT_DIR = "Streamo-Instruct-465K"
UNABLE_OPTION = "Unable to answer from the video so far"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset2-root", type=Path, default=DEFAULT_DATASET2_ROOT)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--unable-output", default="rl_0514_unable.jsonl")
    parser.add_argument("--one-output", default="rl_0514_one.jsonl")
    parser.add_argument("--multi-output", default="rl_0514_multi.jsonl")
    parser.add_argument("--summary-output", default="rl_0514_split.summary.json")
    parser.add_argument("--min-query-time-seconds", type=float, default=20.0)
    parser.add_argument("--min-unable-gap-seconds", type=float, default=20.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_index} is not a JSON object")
            yield value


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def stable_hash(parts: Iterable[Any], length: int = 12) -> str:
    text = "\t".join("" if item is None else str(item) for item in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def answer_events(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    query_events = row.get("query_events")
    if not isinstance(query_events, list) or not query_events:
        return []
    event = query_events[0]
    if not isinstance(event, Mapping):
        return []
    answers = event.get("answer_events")
    if not isinstance(answers, list):
        return []
    return [dict(item) for item in answers if isinstance(item, Mapping)]


def query_event(row: Mapping[str, Any]) -> dict[str, Any] | None:
    events = row.get("query_events")
    if not isinstance(events, list) or len(events) != 1 or not isinstance(events[0], Mapping):
        return None
    return dict(events[0])


def query_time(event: Mapping[str, Any]) -> float:
    return coerce_float(event.get("time", event.get("timestamp", event.get("query_timestamp"))), 0.0)


def answer_time(answer: Mapping[str, Any], default: float) -> float:
    return coerce_float(answer.get("time", answer.get("timestamp")), default)


def gt_index_from_letter(letter: Any, option_count: int) -> int | None:
    text = str(letter or "").strip().upper()
    if len(text) != 1 or not ("A" <= text <= "Z"):
        return None
    index = ord(text) - ord("A")
    if index < 0 or index >= option_count:
        return None
    return index


def option_letter(index: int) -> str:
    return chr(ord("A") + index)


def realtime_question(question: str) -> str:
    text = question.strip()
    if not text:
        return "Based on the video so far, answer the question."
    first = text[0].lower() if text[0].isalpha() else text[0]
    return f"Based on the video so far, {first}{text[1:]}"


def mcq_content(question: str, options: list[Any]) -> str:
    lines = [question.strip(), "Options:"]
    for index, option in enumerate(options):
        label = option_letter(index)
        text = str(option).strip()
        if text[:2].upper() == f"{label}.":
            lines.append(text)
        else:
            lines.append(f"{label}. {text}")
    return "\n".join(lines).strip()


def unable_replacement_index(row: Mapping[str, Any], gt_index: int, option_count: int) -> int:
    candidates = [index for index in range(option_count) if index != gt_index]
    if not candidates:
        raise ValueError("Unable conversion requires at least one non-GT option.")
    digest = stable_hash([row.get("sample_id"), row.get("video_id"), row.get("source_video_path")], length=8)
    return candidates[int(digest, 16) % len(candidates)]


def is_unable_candidate(
    row: Mapping[str, Any],
    *,
    min_query_time_seconds: float,
    min_unable_gap_seconds: float,
) -> tuple[bool, str]:
    event = query_event(row)
    if event is None:
        return False, "bad_query_event_count"
    answers = answer_events(row)
    if len(answers) != 1:
        return False, "not_one_answer"
    options = event.get("options")
    if not isinstance(options, list) or len(options) < 2:
        return False, "missing_options"
    gt_index = gt_index_from_letter(answers[0].get("gt"), len(options))
    if gt_index is None:
        return False, "bad_gt"
    qt = query_time(event)
    at = answer_time(answers[0], qt)
    if qt < min_query_time_seconds - 1e-9:
        return False, "query_before_min"
    if at - qt < min_unable_gap_seconds - 1e-9:
        return False, "gap_below_min"
    return True, "eligible"


def make_unable_row(row: Mapping[str, Any], *, min_query_time_seconds: float, min_unable_gap_seconds: float) -> dict[str, Any]:
    event = query_event(row)
    if event is None:
        raise ValueError("Unable row requires exactly one query_event.")
    answers = answer_events(row)
    if len(answers) != 1:
        raise ValueError("Unable row requires exactly one answer_event.")
    options = list(event.get("options") or [])
    gt_index = gt_index_from_letter(answers[0].get("gt"), len(options))
    if gt_index is None:
        raise ValueError("Unable row requires a valid MCQ gt letter.")
    replacement_index = unable_replacement_index(row, gt_index, len(options))
    original_option = options[replacement_index]
    options[replacement_index] = UNABLE_OPTION
    unable_gt = option_letter(replacement_index)
    qt = query_time(event)
    at = answer_time(answers[0], qt)
    frame_count = max(1, int(math.ceil(qt - 1e-9)))
    new_question = realtime_question(str(event.get("question") or event.get("content") or ""))

    new_event = dict(event)
    new_event.update(
        {
            "qid": "q0",
            "time": float(qt),
            "content": mcq_content(new_question, options),
            "question": new_question,
            "answer_policy": "answer_when_asked",
            "query_type": "streamo_unable",
            "options": options,
            "answer_events": [
                {
                    "time": float(qt),
                    "gt": unable_gt,
                    "answer": UNABLE_OPTION,
                    "content": f"{unable_gt}. {UNABLE_OPTION}",
                    "evidence_time": float(qt),
                    "source_answer_time": at,
                    "source_answer_gt": answers[0].get("gt"),
                    "source_answer": answers[0].get("answer"),
                    "source_replaced_option_index": replacement_index,
                    "source_replaced_option": original_option,
                }
            ],
        }
    )

    out = dict(row)
    out["sample_id"] = f"streamo_unable_{stable_hash([row.get('sample_id'), qt, at])}"
    out["task"] = "realtime"
    out["task_family"] = "unable_to_answer"
    out["query_events"] = [new_event]
    out["frame_count"] = frame_count
    out["realtime"] = float(qt)
    if "duration" in out:
        out["duration"] = float(qt)
    if "target_timestamp" in out:
        out["target_timestamp"] = float(qt)
    if "input_window_end" in out:
        out["input_window_end"] = frame_count
    if "input_window_end_seconds" in out:
        out["input_window_end_seconds"] = float(qt)
    out["source_filter_0514"] = row.get("filter_0514")
    out["filter_0514"] = {
        "source_file": DEFAULT_INPUT,
        "derived_for": "unable_to_answer",
        "min_query_time_seconds": min_query_time_seconds,
        "min_unable_gap_seconds": min_unable_gap_seconds,
        "original_sample_id": row.get("sample_id"),
        "original_frame_count": row.get("frame_count"),
        "original_realtime": row.get("realtime"),
        "original_answer_time": at,
        "original_answer_gt": answers[0].get("gt"),
        "truncated_to_query_time": True,
        "final_answer_time": float(qt),
        "final_frame_count": frame_count,
    }
    out["streamo_unable_0514"] = {
        "source_sample_id": row.get("sample_id"),
        "source_query_time": qt,
        "source_answer_time": at,
        "source_gap_seconds": at - qt,
        "unable_option": UNABLE_OPTION,
        "unable_gt": unable_gt,
        "replaced_option_index": replacement_index,
        "replaced_option": original_option,
    }
    return out


def split_rows(
    rows: list[dict[str, Any]],
    *,
    min_query_time_seconds: float,
    min_unable_gap_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    unable_rows: list[dict[str, Any]] = []
    one_rows: list[dict[str, Any]] = []
    multi_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    source_counts = {
        "unable": Counter(),
        "one": Counter(),
        "multi": Counter(),
    }
    answer_count_distribution: Counter[int] = Counter()

    for row in rows:
        answers = answer_events(row)
        answer_count_distribution[len(answers)] += 1
        if len(answers) > 1:
            multi_rows.append(row)
            source_counts["multi"][str(row.get("source_dataset") or "")] += 1
            continue
        eligible, reason = is_unable_candidate(
            row,
            min_query_time_seconds=min_query_time_seconds,
            min_unable_gap_seconds=min_unable_gap_seconds,
        )
        reason_counts[reason] += 1
        if eligible:
            unable = make_unable_row(
                row,
                min_query_time_seconds=min_query_time_seconds,
                min_unable_gap_seconds=min_unable_gap_seconds,
            )
            unable_rows.append(unable)
            source_counts["unable"][str(row.get("source_dataset") or "")] += 1
        else:
            one_rows.append(row)
            source_counts["one"][str(row.get("source_dataset") or "")] += 1

    summary = {
        "input_rows": len(rows),
        "unable_rows": len(unable_rows),
        "one_rows": len(one_rows),
        "multi_rows": len(multi_rows),
        "min_query_time_seconds": min_query_time_seconds,
        "min_unable_gap_seconds": min_unable_gap_seconds,
        "unable_option": UNABLE_OPTION,
        "unable_question_rewrite": "prefix only: Based on the video so far",
        "unable_option_policy": "replace one non-GT option; do not add an explicit unable instruction to the question",
        "answer_count_distribution": dict(sorted(answer_count_distribution.items())),
        "unable_candidate_reason_counts": dict(reason_counts.most_common()),
        "source_dataset_counts": {
            key: dict(counter.most_common()) for key, counter in source_counts.items()
        },
    }
    summary.update(validate_outputs(unable_rows, one_rows, multi_rows))
    return unable_rows, one_rows, multi_rows, summary


def validate_outputs(
    unable_rows: list[dict[str, Any]],
    one_rows: list[dict[str, Any]],
    multi_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    sample_ids: Counter[str] = Counter()
    path_missing = {"unable": 0, "one": 0, "multi": 0}

    for bucket_name, rows in (("unable", unable_rows), ("one", one_rows), ("multi", multi_rows)):
        for row in rows:
            sample_ids[str(row.get("sample_id") or "")] += 1
            dataset_dir = Path(str(row.get("dataset_dir") or ""))
            frames_dir = str(row.get("frames_dir") or row.get("video") or "")
            if dataset_dir and frames_dir and not (dataset_dir / frames_dir).is_dir():
                path_missing[bucket_name] += 1
            events = row.get("query_events")
            if not isinstance(events, list) or len(events) != 1:
                errors[f"{bucket_name}_bad_query_event_count"] += 1
                continue
            answers = events[0].get("answer_events") if isinstance(events[0], Mapping) else None
            if not isinstance(answers, list) or not answers:
                errors[f"{bucket_name}_missing_answers"] += 1
                continue
            if bucket_name == "unable":
                if len(answers) != 1:
                    errors["unable_bad_answer_count"] += 1
                qt = query_time(events[0])
                at = answer_time(answers[0], qt)
                if abs(qt - at) > 1e-9:
                    errors["unable_answer_not_at_query_time"] += 1
                if int(row.get("frame_count") or 0) != max(1, int(math.ceil(qt - 1e-9))):
                    errors["unable_frame_count_not_query_time"] += 1
                if answers[0].get("answer") != UNABLE_OPTION:
                    errors["unable_bad_answer_text"] += 1
                options = events[0].get("options")
                gt_index = gt_index_from_letter(answers[0].get("gt"), len(options) if isinstance(options, list) else 0)
                if not isinstance(options, list) or gt_index is None or options[gt_index] != UNABLE_OPTION:
                    errors["unable_gt_not_unable_option"] += 1
                if "If the answer is not supported" in str(events[0].get("content") or ""):
                    errors["unable_forbidden_prompt_phrase"] += 1
            elif bucket_name == "one" and len(answers) != 1:
                errors["one_bad_answer_count"] += 1
            elif bucket_name == "multi" and len(answers) <= 1:
                errors["multi_bad_answer_count"] += 1

    duplicate_sample_ids = sum(1 for count in sample_ids.values() if count > 1)
    return {
        "validation": {
            "ok": not errors and duplicate_sample_ids == 0 and all(value == 0 for value in path_missing.values()),
            "error_counts": dict(errors.most_common()),
            "duplicate_sample_ids": duplicate_sample_ids,
            "frame_dir_missing_rows": path_missing,
        }
    }


def main() -> None:
    args = parse_args()
    dataset2_root = args.dataset2_root
    input_path = dataset2_root / args.input
    output_dir = dataset2_root / args.output_dir
    rows = list(read_jsonl(input_path))
    unable_rows, one_rows, multi_rows, summary = split_rows(
        rows,
        min_query_time_seconds=args.min_query_time_seconds,
        min_unable_gap_seconds=args.min_unable_gap_seconds,
    )

    unable_path = output_dir / args.unable_output
    one_path = output_dir / args.one_output
    multi_path = output_dir / args.multi_output
    written = {
        "unable": write_jsonl(unable_path, unable_rows),
        "one": write_jsonl(one_path, one_rows),
        "multi": write_jsonl(multi_path, multi_rows),
    }
    summary = {
        "source_path": str(input_path),
        "output_paths": {
            "unable": str(unable_path),
            "one": str(one_path),
            "multi": str(multi_path),
        },
        "written_rows": written,
        **summary,
    }
    summary_path = output_dir / args.summary_output
    write_json(summary_path, summary)
    print(f"Wrote {written['unable']} rows to {unable_path}")
    print(f"Wrote {written['one']} rows to {one_path}")
    print(f"Wrote {written['multi']} rows to {multi_path}")
    print(f"Wrote summary to {summary_path}")
    if not summary.get("validation", {}).get("ok", False):
        raise SystemExit("Streamo 0514 split validation failed; see summary for details.")


if __name__ == "__main__":
    main()
