#!/usr/bin/env python3
"""Build canonical query-event annotations for Streamo and CogStream.

The script does not overwrite the existing dataset2 annotations. It writes new
files inside each dataset folder:

  Streamo-Instruct-465K/query_events.jsonl
  CogStream/query_events.jsonl

These rows keep the existing local frame directories and normalize QA labels to
the GRPPO proposal shape:

  query_events[].answer_events[]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_DATASET2_ROOT = Path("/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset2")
STREAMO_NAME = "Streamo-Instruct-465K"
COGSTREAM_NAME = "CogStream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset2-root", type=Path, default=DEFAULT_DATASET2_ROOT)
    parser.add_argument("--streamo-name", default=STREAMO_NAME)
    parser.add_argument("--cogstream-name", default=COGSTREAM_NAME)
    parser.add_argument("--streamo-output", default="query_events.jsonl")
    parser.add_argument("--cogstream-output", default="query_events.jsonl")
    parser.add_argument("--summary-suffix", default=".summary.json")
    parser.add_argument("--skip-streamo", action="store_true")
    parser.add_argument("--skip-cogstream", action="store_true")
    parser.add_argument("--limit-streamo", type=int, default=0)
    parser.add_argument("--limit-cogstream", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=511)
    parser.add_argument(
        "--cogstream-per-qa",
        action="store_true",
        help="Keep one CogStream row per QA. Default aggregates CogStream by video_id.",
    )
    parser.add_argument(
        "--allow-missing-frames",
        action="store_true",
        help="Keep rows even if their local frame directory is missing. Default is to drop them.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_index + 1} is not a JSON object")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
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


def stable_hash(parts: Iterable[Any], length: int = 12) -> str:
    text = "\t".join("" if item is None else str(item) for item in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def option_letter(index_or_letter: Any) -> str | None:
    if index_or_letter is None:
        return None
    if isinstance(index_or_letter, str):
        text = index_or_letter.strip()
        if len(text) == 1 and text.isalpha():
            return text.upper()
        try:
            index_or_letter = int(text)
        except ValueError:
            return None
    if isinstance(index_or_letter, int):
        if 0 <= index_or_letter < 26:
            return chr(ord("A") + index_or_letter)
        if 1 <= index_or_letter <= 26:
            return chr(ord("A") + index_or_letter - 1)
    return None


def has_frame_dir(dataset_dir: Path, row: Mapping[str, Any]) -> bool:
    frames_dir = str(row.get("frames_dir") or row.get("video") or "").strip()
    if not frames_dir:
        return False
    path = dataset_dir / frames_dir
    if not path.is_dir():
        return False
    if (path / "manifest.json").is_file():
        return True
    return any(path.glob("*.jpg")) or any(path.glob("*.jpeg")) or any(path.glob("*.png"))


def row_base(row: Mapping[str, Any], *, dataset_dir: Path) -> dict[str, Any]:
    keep_keys = (
        "dataset",
        "split",
        "video_id",
        "video",
        "frames_dir",
        "sample_fps",
        "frame_count",
        "frame_id_base",
        "duration",
        "realtime",
        "source_dataset",
        "source_file",
        "source_video_name",
        "source_video_path",
        "source_material_path",
        "source_kind",
        "source_status",
        "source_seq_info",
        "source_segment_path",
        "source_segment_timestamp",
        "input_window_start",
        "input_window_end",
        "input_window_start_seconds",
        "input_window_end_seconds",
    )
    out = {key: row[key] for key in keep_keys if key in row}
    out["dataset_dir"] = str(dataset_dir)
    return out


def build_mcq_content(question: str, options: list[Any]) -> str:
    lines = [question.strip(), "Options:"]
    for index, option in enumerate(options):
        label = chr(ord("A") + index)
        text = str(option).strip()
        if text[:2].upper() == f"{label}.":
            lines.append(text)
        else:
            lines.append(f"{label}. {text}")
    lines.append("")
    lines.append("Update your answer when the video evidence changes.")
    return "\n".join(lines).strip()


def streamo_group_key(row: Mapping[str, Any], qa: Mapping[str, Any]) -> tuple[Any, ...]:
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    return (
        row.get("source_file"),
        row.get("source_dataset"),
        row.get("source_video_name"),
        row.get("source_video_path"),
        coerce_float(row.get("ask_time", row.get("realtime")), 0.0),
        str(qa.get("question") or "").strip(),
        json.dumps(options, ensure_ascii=False, sort_keys=False),
    )


def streamo_answer_event(row: Mapping[str, Any], qa: Mapping[str, Any]) -> dict[str, Any]:
    answer = str(qa.get("answer") or "").strip()
    gt = option_letter(qa.get("gt"))
    event_time = coerce_float(row.get("clue_time", row.get("target_timestamp", row.get("realtime"))), 0.0)
    content = f"{gt}. {answer}" if gt and answer else answer
    out: dict[str, Any] = {
        "time": event_time,
        "answer": answer,
        "content": content,
        "source_qa_id": qa.get("qa_id"),
        "source_sample_id": row.get("sample_id"),
    }
    if gt:
        out["gt"] = gt
    if qa.get("gt") is not None:
        out["source_gt"] = qa.get("gt")
    if row.get("clue_time") is not None:
        out["evidence_time"] = coerce_float(row.get("clue_time"), event_time)
    return out


def choose_cover_row(items: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    def key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, int]:
        row, _qa = item
        end_time = coerce_float(row.get("realtime", row.get("duration")), 0.0)
        frame_count = int(row.get("frame_count") or 0)
        return end_time, frame_count

    return max(items, key=key)[0]


def build_streamo_rows(dataset_dir: Path, *, limit: int, allow_missing_frames: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = dataset_dir / "annotations.jsonl"
    groups: dict[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    raw_rows = 0
    raw_qa = 0
    dropped_missing_frames = 0
    for row in read_jsonl(source_path):
        raw_rows += 1
        if not allow_missing_frames and not has_frame_dir(dataset_dir, row):
            dropped_missing_frames += 1
            continue
        qa_list = row.get("qa_list") if isinstance(row.get("qa_list"), list) else []
        for qa in qa_list:
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            raw_qa += 1
            groups[streamo_group_key(row, qa)].append((dict(row), dict(qa)))

    output_rows: list[dict[str, Any]] = []
    answer_event_counts = Counter()
    source_counts = Counter()
    for index, (key, items) in enumerate(sorted(groups.items(), key=lambda item: stable_hash(item[0]))):
        if limit and len(output_rows) >= limit:
            break
        cover = choose_cover_row(items)
        first_qa = items[0][1]
        question = str(first_qa.get("question") or "").strip()
        options = first_qa.get("options") if isinstance(first_qa.get("options"), list) else []
        answer_events = sorted(
            dedupe_answer_events([streamo_answer_event(row, qa) for row, qa in items]),
            key=lambda item: (coerce_float(item.get("time"), 0.0), str(item.get("content") or "")),
        )
        if not answer_events:
            continue
        qid = "q0"
        query_event: dict[str, Any] = {
            "qid": qid,
            "time": coerce_float(cover.get("ask_time", cover.get("realtime")), 0.0),
            "content": build_mcq_content(question, options) if options else question,
            "question": question,
            "answer_policy": "update_when_changed",
            "query_type": "streamo_update",
            "answer_events": answer_events,
        }
        if options:
            query_event["options"] = options

        row = row_base(cover, dataset_dir=dataset_dir)
        row.update(
            {
                "dataset": STREAMO_NAME,
                "sample_id": f"streamo_qe_{stable_hash(key)}",
                "task": str(cover.get("task") or "forward"),
                "task_family": "semantic_drift",
                "query_events": [query_event],
                "source_row_count": len(items),
                "source_sample_ids": [item[0].get("sample_id") for item in items],
            }
        )
        output_rows.append(row)
        answer_event_counts[len(answer_events)] += 1
        source_counts[str(cover.get("source_dataset") or "")] += 1

    summary = {
        "dataset": STREAMO_NAME,
        "source_path": str(source_path),
        "raw_rows": raw_rows,
        "raw_qa_items": raw_qa,
        "groups": len(groups),
        "output_rows": len(output_rows),
        "dropped_missing_frames": dropped_missing_frames,
        "answer_event_count_distribution": dict(sorted(answer_event_counts.items())),
        "source_dataset_counts": dict(source_counts.most_common()),
    }
    return output_rows, summary


def dedupe_answer_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for event in events:
        key = (
            round(coerce_float(event.get("time"), 0.0), 3),
            event.get("gt"),
            event.get("answer"),
            event.get("content"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out


def normalize_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(loaded, list):
            return [str(item) for item in loaded]
        return [str(loaded)]
    return [str(value)]


def cogstream_query_event(row: Mapping[str, Any]) -> dict[str, Any] | None:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or row.get("gt") or "").strip()
    if not question or not answer:
        return None
    answer_time = coerce_float(row.get("ask_time", row.get("realtime")), 0.0)
    query_type = str(row.get("source_label") or "CogStream")
    qid = f"q{row.get('source_id', 0)}"
    source_depends_on = normalize_id_list(row.get("source_cor"))
    source_relevance = normalize_id_list(row.get("source_relevance"))
    query_event: dict[str, Any] = {
        "qid": qid,
        "time": answer_time,
        "content": question,
        "question": question,
        "answer_policy": "answer_when_asked",
        "query_type": query_type,
        "is_visual": bool(row.get("source_is_visual", True)),
        "source_id": row.get("source_id"),
        "source_sample_id": row.get("sample_id"),
        "answer_events": [
            {
                "time": answer_time,
                "gt": answer,
                "answer": answer,
                "content": answer,
                "evidence_time": coerce_float(
                    row.get("source_event_timestamp", row.get("clue_time", answer_time)),
                    answer_time,
                ),
            }
        ],
    }
    if source_depends_on:
        query_event["source_depends_on_ids"] = source_depends_on
    if source_relevance:
        query_event["source_relevance"] = source_relevance
    return query_event


def build_cogstream_per_qa_row(dataset_dir: Path, row: Mapping[str, Any], query_event: dict[str, Any]) -> dict[str, Any]:
    out = row_base(row, dataset_dir=dataset_dir)
    out.update(
        {
            "dataset": COGSTREAM_NAME,
            "sample_id": f"cogstream_qe_{row.get('sample_id') or stable_hash([row.get('video_id'), query_event.get('qid'), query_event.get('question')])}",
            "task": "realtime",
            "task_family": "streaming_state",
            "query_events": [query_event],
            "source_sample_id": row.get("sample_id"),
            "source_id": row.get("source_id"),
            "source_label": row.get("source_label"),
            "source_is_visual": row.get("source_is_visual"),
            "source_cor": row.get("source_cor"),
            "source_relevance": row.get("source_relevance"),
        }
    )
    return out


def build_cogstream_video_row(dataset_dir: Path, video_id: str, items: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    cover = max(
        (row for row, _event in items),
        key=lambda row: (int(row.get("frame_count") or 0), coerce_float(row.get("realtime"), 0.0)),
    )
    query_events: list[dict[str, Any]] = []
    for index, (_row, event) in enumerate(sorted(items, key=lambda item: (coerce_float(item[1].get("time"), 0.0), str(item[1].get("content") or "")))):
        event = dict(event)
        event["qid"] = f"q{index}"
        query_events.append(event)

    out = row_base(cover, dataset_dir=dataset_dir)
    out.update(
        {
            "dataset": COGSTREAM_NAME,
            "sample_id": f"cogstream_video_qe_{video_id}",
            "task": "realtime",
            "task_family": "streaming_state",
            "query_events": query_events,
            "source_sample_ids": [row.get("sample_id") for row, _event in items],
            "source_query_count": len(query_events),
            "cogstream_aggregation": "by_video_drop_overlapping_time",
        }
    )
    if query_events:
        out["realtime"] = max(coerce_float(event.get("time"), 0.0) for event in query_events)
    return out


def build_cogstream_rows(
    dataset_dir: Path,
    *,
    limit: int,
    allow_missing_frames: bool,
    aggregate_by_video: bool,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = dataset_dir / "annotations.jsonl"
    per_qa_rows: list[dict[str, Any]] = []
    by_video: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    raw_rows = 0
    dropped_missing_frames = 0
    invalid_rows = 0
    candidate_label_counts = Counter()
    candidate_split_counts = Counter()
    for row in read_jsonl(source_path):
        raw_rows += 1
        if not allow_missing_frames and not has_frame_dir(dataset_dir, row):
            dropped_missing_frames += 1
            continue
        query_event = cogstream_query_event(row)
        if query_event is None:
            invalid_rows += 1
            continue
        candidate_label_counts[str(row.get("source_label") or "CogStream")] += 1
        candidate_split_counts[str(row.get("split") or "")] += 1
        if aggregate_by_video:
            video_id = str(row.get("video_id") or "").strip()
            if video_id:
                by_video[video_id].append((dict(row), query_event))
            else:
                invalid_rows += 1
        else:
            per_qa_rows.append(build_cogstream_per_qa_row(dataset_dir, row, query_event))

    output_rows: list[dict[str, Any]]
    kept_label_counts = Counter()
    discarded_label_counts = Counter()
    overlap_groups = 0
    discarded_overlapping_query_events = 0
    if aggregate_by_video:
        rng = random.Random(random_seed)
        output_rows = []
        for video_id in sorted(by_video):
            time_buckets: dict[float, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
            for row, event in by_video[video_id]:
                time_buckets[round(coerce_float(event.get("time"), 0.0), 3)].append((row, event))
            kept_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for _time_key, bucket in sorted(time_buckets.items()):
                if len(bucket) > 1:
                    overlap_groups += 1
                    discarded_overlapping_query_events += len(bucket) - 1
                chosen = rng.choice(bucket)
                kept_items.append(chosen)
                kept_label_counts[str(chosen[0].get("source_label") or "CogStream")] += 1
                for item in bucket:
                    if item is not chosen:
                        discarded_label_counts[str(item[0].get("source_label") or "CogStream")] += 1
            output_rows.append(build_cogstream_video_row(dataset_dir, video_id, kept_items))
    else:
        output_rows = per_qa_rows
        for row in output_rows:
            for event in row.get("query_events") or []:
                kept_label_counts[str(event.get("query_type") or "CogStream")] += 1

    if limit:
        output_rows = output_rows[:limit]

    summary = {
        "dataset": COGSTREAM_NAME,
        "source_path": str(source_path),
        "mode": "by_video_drop_overlapping_time" if aggregate_by_video else "per_qa",
        "random_seed": random_seed if aggregate_by_video else None,
        "raw_rows": raw_rows,
        "candidate_query_events": sum(candidate_label_counts.values()),
        "output_rows": len(output_rows),
        "output_query_events": sum(len(row.get("query_events") or []) for row in output_rows),
        "dropped_missing_frames": dropped_missing_frames,
        "invalid_rows": invalid_rows,
        "overlap_groups": overlap_groups,
        "discarded_overlapping_query_events": discarded_overlapping_query_events,
        "candidate_label_counts": dict(candidate_label_counts.most_common()),
        "kept_label_counts": dict(kept_label_counts.most_common()),
        "discarded_label_counts": dict(discarded_label_counts.most_common()),
        "split_counts": dict(candidate_split_counts.most_common()),
    }
    return output_rows, summary


def write_dataset_output(dataset_dir: Path, filename: str, rows: list[dict[str, Any]], summary: dict[str, Any], suffix: str) -> None:
    output_path = dataset_dir / filename
    count = write_jsonl(output_path, rows)
    summary = dict(summary)
    summary["output_path"] = str(output_path)
    summary["written_rows"] = count
    write_json(output_path.with_suffix(output_path.suffix + suffix), summary)
    print(f"Wrote {count} rows to {output_path}")
    print(f"Wrote summary to {output_path.with_suffix(output_path.suffix + suffix)}")


def main() -> None:
    args = parse_args()
    dataset2_root = args.dataset2_root

    if not args.skip_streamo:
        streamo_dir = dataset2_root / args.streamo_name
        rows, summary = build_streamo_rows(
            streamo_dir,
            limit=args.limit_streamo,
            allow_missing_frames=args.allow_missing_frames,
        )
        write_dataset_output(streamo_dir, args.streamo_output, rows, summary, args.summary_suffix)

    if not args.skip_cogstream:
        cogstream_dir = dataset2_root / args.cogstream_name
        rows, summary = build_cogstream_rows(
            cogstream_dir,
            limit=args.limit_cogstream,
            allow_missing_frames=args.allow_missing_frames,
            aggregate_by_video=not args.cogstream_per_qa,
            random_seed=args.random_seed,
        )
        write_dataset_output(cogstream_dir, args.cogstream_output, rows, summary, args.summary_suffix)


if __name__ == "__main__":
    main()
