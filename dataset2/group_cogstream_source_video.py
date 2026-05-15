#!/usr/bin/env python3
"""Group CogStream query-event rows by original source video.

This script does not modify CogStream's original annotations or filtered
query-event files.  It reads the 0514 filtered CogStream query-event rows and
writes a new file where rows sharing the same ``source_video_name`` are merged
into one RL sample whenever possible.

CogStream windows are cumulative prefixes of the same original video:

  source_video_name=X, video_id=X_000000_000046 -> 0s..46s
  source_video_name=X, video_id=X_000000_000071 -> 0s..71s

The grouped row keeps the longest retained prefix as the video/frame directory
and moves the shorter-prefix QA items into ``query_events`` at their original
times.  Queries are marked as independent because CogStream rows retained by
our preprocessing have no COR dependency and no positive relevance dependency.
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
DEFAULT_INPUT = "CogStream/query_events_0514_filtered.jsonl"
DEFAULT_OUTPUT = "CogStream/rl_0514.jsonl"
RESERVED_COGSTREAM_FILES = {
    "annotations.jsonl",
    "query_events.jsonl",
    "query_events_0514_filtered.jsonl",
}

QUERY_TYPE_PRIORITY = {
    "Streaming/Dynamic Updating": 0,
    "Streaming/Object Tracking": 1,
    "Streaming/Sequence Perception": 2,
    "Streaming/Causal Reasoning": 3,
    "Basic/Actions": 4,
    "Basic/Object": 5,
    "Basic/Attributes": 6,
    "Global/Overall Summary": 7,
    "Global/Global Analysis": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset2-root", type=Path, default=DEFAULT_DATASET2_ROOT)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-suffix", default=".summary.json")
    parser.add_argument("--min-query-gap-seconds", type=float, default=7.0)
    parser.add_argument(
        "--min-queries",
        type=int,
        default=1,
        help="Minimum retained queries required for an output row. Use 2 to write multi-query rows only.",
    )
    parser.add_argument(
        "--same-time-choice",
        choices=("priority", "random"),
        default="priority",
        help="How to choose one query when a source video has multiple candidates at the same timestamp.",
    )
    parser.add_argument("--random-seed", type=int, default=511)
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


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def stable_hash(parts: Iterable[Any], length: int = 12) -> str:
    text = "\t".join("" if item is None else str(item) for item in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def query_time(event: Mapping[str, Any]) -> float:
    return coerce_float(event.get("time", event.get("timestamp", event.get("query_timestamp"))), 0.0)


def frame_count(row: Mapping[str, Any]) -> int:
    return coerce_int(row.get("frame_count"), 0)


def realtime(row: Mapping[str, Any]) -> float:
    return coerce_float(row.get("realtime"), 0.0)


def candidate_priority(candidate: Mapping[str, Any]) -> tuple[int, float, str]:
    event = candidate["event"]
    query_type = str(event.get("query_type") or "")
    return (
        QUERY_TYPE_PRIORITY.get(query_type, 99),
        query_time(event),
        str(event.get("source_sample_id") or candidate["row"].get("sample_id") or ""),
    )


def choose_bucket_candidate(
    bucket: list[dict[str, Any]],
    *,
    rng: random.Random,
    same_time_choice: str,
) -> dict[str, Any]:
    if same_time_choice == "random":
        return rng.choice(bucket)
    return sorted(bucket, key=candidate_priority)[0]


def collect_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        events = row.get("query_events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            answers = event.get("answer_events")
            if not isinstance(answers, list) or not answers:
                continue
            candidates.append({"row": row, "event": event})
    return candidates


def dedupe_same_time(
    candidates: list[dict[str, Any]],
    *,
    rng: random.Random,
    same_time_choice: str,
) -> tuple[list[dict[str, Any]], int]:
    buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        buckets[round(query_time(candidate["event"]), 6)].append(candidate)

    kept: list[dict[str, Any]] = []
    discarded = 0
    for _time, bucket in sorted(buckets.items()):
        chosen = choose_bucket_candidate(bucket, rng=rng, same_time_choice=same_time_choice)
        kept.append(chosen)
        discarded += len(bucket) - 1
    return kept, discarded


def apply_min_gap(candidates: list[dict[str, Any]], *, min_gap: float) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    discarded = 0
    last_time: float | None = None
    for candidate in sorted(candidates, key=lambda item: (query_time(item["event"]), candidate_priority(item))):
        current = query_time(candidate["event"])
        if last_time is not None and current - last_time < min_gap - 1e-9:
            discarded += 1
            continue
        kept.append(candidate)
        last_time = current
    return kept, discarded


def choose_cover_row(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        (candidate["row"] for candidate in candidates),
        key=lambda row: (frame_count(row), realtime(row), str(row.get("video_id") or "")),
    )


def grouped_row(source_video_name: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    cover = dict(choose_cover_row(candidates))
    retained = sorted(candidates, key=lambda item: (query_time(item["event"]), candidate_priority(item)))
    query_events: list[dict[str, Any]] = []
    source_sample_ids: list[Any] = []
    source_video_ids: list[Any] = []
    source_seq_infos: list[Any] = []
    source_query_types: Counter[str] = Counter()

    for index, candidate in enumerate(retained):
        source_row = candidate["row"]
        event = dict(candidate["event"])
        event["qid"] = f"q{index}"
        event["query_dependency"] = "independent_same_video"
        event["depends_on"] = []
        event["source_group_video_id"] = source_row.get("video_id")
        event["source_group_sample_id"] = source_row.get("sample_id")
        event["source_group_source_seq_info"] = source_row.get("source_seq_info")
        query_events.append(event)
        source_sample_ids.extend(source_row.get("source_sample_ids") or [event.get("source_sample_id")])
        source_video_ids.append(source_row.get("video_id"))
        source_seq_infos.append(source_row.get("source_seq_info"))
        source_query_types[str(event.get("query_type") or "")] += 1

    out = cover
    out.update(
        {
            "dataset": "CogStream",
            "sample_id": f"cogstream_source_grouped_{stable_hash([source_video_name, cover.get('video_id'), len(query_events)])}",
            "task": "realtime",
            "task_family": "streaming_state_multi_query",
            "query_events": query_events,
            "source_video_name": source_video_name,
            "source_sample_ids": [item for item in source_sample_ids if item is not None],
            "source_video_ids": [item for item in source_video_ids if item is not None],
            "source_seq_infos": [item for item in source_seq_infos if item is not None],
            "source_query_count": len(query_events),
            "cogstream_aggregation": "by_source_video_name_one_query_per_time",
            "query_dependency": "independent_same_video",
            "query_dependency_note": (
                "Queries share the same original CogStream source video but are not constructed to depend on "
                "previous questions or answers."
            ),
            "grouped_from": {
                "source_file": DEFAULT_INPUT,
                "cover_video_id": cover.get("video_id"),
                "cover_frame_count": cover.get("frame_count"),
                "retained_query_count": len(query_events),
                "query_times": [query_time(event) for event in query_events],
                "query_type_counts": dict(source_query_types.most_common()),
            },
        }
    )
    out["realtime"] = coerce_float(cover.get("realtime"), max(query_time(event) for event in query_events))
    return out


def group_rows(
    rows: list[dict[str, Any]],
    *,
    min_query_gap_seconds: float,
    min_queries: int,
    same_time_choice: str,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(random_seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_source_video_name = 0
    for row in rows:
        source_video_name = str(row.get("source_video_name") or "").strip()
        if not source_video_name:
            missing_source_video_name += 1
            continue
        by_source[source_video_name].append(row)

    output_rows: list[dict[str, Any]] = []
    dropped_same_time = 0
    dropped_min_gap = 0
    dropped_min_queries = 0
    input_query_count = 0
    candidate_group_count_distribution: Counter[int] = Counter()
    output_query_count_distribution: Counter[int] = Counter()

    for source_video_name in sorted(by_source):
        candidates = collect_candidates(by_source[source_video_name])
        input_query_count += len(candidates)
        candidate_group_count_distribution[len(candidates)] += 1
        if not candidates:
            dropped_min_queries += 1
            continue
        deduped, same_time_discarded = dedupe_same_time(
            candidates,
            rng=rng,
            same_time_choice=same_time_choice,
        )
        dropped_same_time += same_time_discarded
        retained, gap_discarded = apply_min_gap(deduped, min_gap=min_query_gap_seconds)
        dropped_min_gap += gap_discarded
        if len(retained) < min_queries:
            dropped_min_queries += 1
            continue
        row = grouped_row(source_video_name, retained)
        output_rows.append(row)
        output_query_count_distribution[len(retained)] += 1

    summary = {
        "input_rows": len(rows),
        "input_source_video_names": len(by_source),
        "input_query_events": input_query_count,
        "missing_source_video_name_rows": missing_source_video_name,
        "output_rows": len(output_rows),
        "output_query_events": sum(len(row.get("query_events") or []) for row in output_rows),
        "min_query_gap_seconds": min_query_gap_seconds,
        "min_queries": min_queries,
        "same_time_choice": same_time_choice,
        "random_seed": random_seed,
        "dropped_same_time_query_events": dropped_same_time,
        "dropped_min_gap_query_events": dropped_min_gap,
        "dropped_groups_below_min_queries": dropped_min_queries,
        "candidate_group_query_count_distribution": {
            str(key): candidate_group_count_distribution[key] for key in sorted(candidate_group_count_distribution)
        },
        "output_query_count_distribution": {
            str(key): output_query_count_distribution[key] for key in sorted(output_query_count_distribution)
        },
    }
    summary.update(validate_grouped_rows(output_rows, min_query_gap_seconds=min_query_gap_seconds))
    return output_rows, summary


def validate_grouped_rows(rows: list[dict[str, Any]], *, min_query_gap_seconds: float) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    query_type_counts: Counter[str] = Counter()
    answer_policy_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    frame_count_values: list[int] = []
    query_gap_values: list[float] = []
    source_video_count: Counter[str] = Counter()
    frame_dir_missing = 0

    for row in rows:
        split_counts[str(row.get("split") or "")] += 1
        source_video_count[str(row.get("source_video_name") or "")] += 1
        frame_count_values.append(frame_count(row))
        dataset_dir = Path(str(row.get("dataset_dir") or ""))
        frames_dir = str(row.get("frames_dir") or row.get("video") or "")
        if dataset_dir and frames_dir and not (dataset_dir / frames_dir).is_dir():
            frame_dir_missing += 1
        events = [event for event in row.get("query_events") or [] if isinstance(event, Mapping)]
        times = [round(query_time(event), 6) for event in events]
        if len(times) != len(set(times)):
            errors["duplicate_query_time"] += 1
        for prev, cur in zip(times, times[1:]):
            gap = cur - prev
            query_gap_values.append(gap)
            if gap < min_query_gap_seconds - 1e-9:
                errors["query_gap_below_min"] += 1
        for event in events:
            if event.get("query_dependency") != "independent_same_video":
                errors["missing_independent_query_dependency"] += 1
            query_type_counts[str(event.get("query_type") or "")] += 1
            answer_policy_counts[str(event.get("answer_policy") or "")] += 1
            answers = event.get("answer_events")
            if not isinstance(answers, list) or not answers:
                errors["missing_answer_events"] += 1

    return {
        "validation": {
            "ok": not errors and frame_dir_missing == 0 and all(count == 1 for count in source_video_count.values()),
            "error_counts": dict(errors.most_common()),
            "frame_dir_missing_rows": frame_dir_missing,
            "duplicate_source_video_output_rows": sum(1 for count in source_video_count.values() if count > 1),
        },
        "split_counts": dict(split_counts.most_common()),
        "query_type_counts": dict(query_type_counts.most_common()),
        "answer_policy_counts": dict(answer_policy_counts.most_common()),
        "frame_count_distribution": numeric_distribution(frame_count_values),
        "query_gap_distribution": numeric_distribution(query_gap_values),
    }


def numeric_distribution(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "p50": None, "p75": None, "max": None}
    sorted_values = sorted(float(value) for value in values)
    length = len(sorted_values)

    def percentile(frac: float) -> float:
        return sorted_values[min(length - 1, int(frac * (length - 1)))]

    return {
        "count": length,
        "min": sorted_values[0],
        "p25": percentile(0.25),
        "p50": percentile(0.5),
        "p75": percentile(0.75),
        "max": sorted_values[-1],
    }


def ensure_safe_output(input_path: Path, output_path: Path, cogstream_dir: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from input path.")
    if output_path.parent.resolve() == cogstream_dir.resolve() and output_path.name in RESERVED_COGSTREAM_FILES:
        raise ValueError(f"Refusing to overwrite reserved CogStream file: {output_path}")


def main() -> None:
    args = parse_args()
    dataset2_root = args.dataset2_root
    input_path = dataset2_root / args.input
    output_path = dataset2_root / args.output
    cogstream_dir = dataset2_root / "CogStream"
    ensure_safe_output(input_path, output_path, cogstream_dir)

    rows = list(read_jsonl(input_path))
    output_rows, summary = group_rows(
        rows,
        min_query_gap_seconds=args.min_query_gap_seconds,
        min_queries=args.min_queries,
        same_time_choice=args.same_time_choice,
        random_seed=args.random_seed,
    )
    written = write_jsonl(output_path, output_rows)
    summary = {
        "source_path": str(input_path),
        "output_path": str(output_path),
        "written_rows": written,
        **summary,
    }
    summary_path = output_path.with_suffix(output_path.suffix + args.summary_suffix)
    write_json(summary_path, summary)
    print(f"Wrote {written} rows to {output_path}")
    print(f"Wrote summary to {summary_path}")
    if not summary.get("validation", {}).get("ok", False):
        raise SystemExit("Grouped CogStream validation failed; see summary for details.")


if __name__ == "__main__":
    main()
