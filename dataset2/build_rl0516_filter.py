#!/usr/bin/env python3
"""Build rl_0516_filter.jsonl from Gemini-cleaned RL candidate pools.

Sampling is done at answer_event level:
- keep all answer_events with eval_pass_rate in {0.25, 0.50, 0.75}
- sample 10% of that middle-count from eval_pass_rate == 0.00
- sample 10% of that middle-count from eval_pass_rate == 1.00
- append a fixed number of full rows from rl_0514_pre.jsonl

The output remains video-row grouped: unselected query/answer events are pruned,
but selected events from the same source video row stay in one JSONL row. Pre
rows are appended as complete rows because they were not part of this Gemini
difficulty-cleaning pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RL_DIR = ROOT / "0516data" / "rl"
DEFAULT_OUTPUT = ROOT / "rl_0516_filter.jsonl"
DEFAULT_SUMMARY = ROOT / "rl_0516_filter.summary.json"
RANDOM_SEED = 516
SIDE_RATIO = 0.10
PRE_SOURCE = "rl_0514_pre.jsonl"
PRE_COUNT = 500
MIDDLE_RATES = {"0.2500", "0.5000", "0.7500"}
LOW_RATE = "0.0000"
HIGH_RATE = "1.0000"

SOURCES = (
    "rl_0514_normalized.jsonl",
    "rl_0514_unable_normalized.jsonl",
    "rl_0514_one_normalized.jsonl",
    "rl_0514_multi_normalized.jsonl",
    "rl_0515_yesno.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-dir", type=Path, default=RL_DIR)
    parser.add_argument("--sources", nargs="+", default=list(SOURCES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--side-ratio", type=float, default=SIDE_RATIO)
    parser.add_argument("--pre-source", default=PRE_SOURCE)
    parser.add_argument("--pre-count", type=int, default=PRE_COUNT)
    parser.add_argument("--no-shuffle-output", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not a JSON object")
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


def rate_key(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def collect_answer_events(
    rl_dir: Path,
    source_names: list[str],
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int, int, int], dict[str, Any]],
    dict[str, list[tuple[str, int, int, int]]],
]:
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    answer_meta: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    keys_by_rate: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)

    for source_name in source_names:
        path = rl_dir / source_name
        rows = read_jsonl(path)
        for row_index, row in enumerate(rows):
            row_key = (source_name, row_index)
            rows_by_key[row_key] = row
            queries = row.get("query_events")
            if not isinstance(queries, list) or not queries:
                raise ValueError(f"{source_name}:{row_index}: missing query_events")
            for query_index, query in enumerate(queries):
                if not isinstance(query, dict):
                    raise ValueError(f"{source_name}:{row_index}: query_events[{query_index}] is not object")
                answers = query.get("answer_events")
                if not isinstance(answers, list):
                    raise ValueError(f"{source_name}:{row_index}: query_events[{query_index}].answer_events is not list")
                for answer_index, answer_event in enumerate(answers):
                    if not isinstance(answer_event, dict):
                        raise ValueError(
                            f"{source_name}:{row_index}: "
                            f"query_events[{query_index}].answer_events[{answer_index}] is not object"
                        )
                    if "eval_pass_rate" not in answer_event:
                        raise ValueError(
                            f"{source_name}:{row_index}: "
                            f"query_events[{query_index}].answer_events[{answer_index}] missing eval_pass_rate"
                        )
                    key = (source_name, row_index, query_index, answer_index)
                    rkey = rate_key(answer_event.get("eval_pass_rate"))
                    keys_by_rate[rkey].append(key)
                    answer_meta[key] = {
                        "rate": rkey,
                        "answer_type": str(query.get("answer_type") or ""),
                        "dataset": str(row.get("dataset") or ""),
                    }
    return rows_by_key, answer_meta, keys_by_rate


def select_answer_events(
    keys_by_rate: dict[str, list[tuple[str, int, int, int]]],
    *,
    rng: random.Random,
    side_ratio: float,
) -> tuple[set[tuple[str, int, int, int]], dict[str, Any]]:
    middle_keys: list[tuple[str, int, int, int]] = []
    for rkey in sorted(MIDDLE_RATES):
        middle_keys.extend(keys_by_rate.get(rkey, []))
    side_count = min(
        half_up(len(middle_keys) * side_ratio),
        len(keys_by_rate.get(LOW_RATE, [])),
        len(keys_by_rate.get(HIGH_RATE, [])),
    )
    low_keys = rng.sample(keys_by_rate.get(LOW_RATE, []), side_count)
    high_keys = rng.sample(keys_by_rate.get(HIGH_RATE, []), side_count)
    selected = set(middle_keys)
    selected.update(low_keys)
    selected.update(high_keys)
    summary = {
        "middle_rates": sorted(MIDDLE_RATES),
        "middle_answer_events": len(middle_keys),
        "side_ratio": side_ratio,
        "side_answer_events_each": side_count,
        "selected_answer_events": len(selected),
        "available_by_rate": {key: len(value) for key, value in sorted(keys_by_rate.items())},
        "selected_by_rate": {
            LOW_RATE: len(low_keys),
            "0.2500": len(keys_by_rate.get("0.2500", [])),
            "0.5000": len(keys_by_rate.get("0.5000", [])),
            "0.7500": len(keys_by_rate.get("0.7500", [])),
            HIGH_RATE: len(high_keys),
        },
    }
    return selected, summary


def build_grouped_rows(
    rows_by_key: dict[tuple[str, int], dict[str, Any]],
    selected_answer_keys: set[tuple[str, int, int, int]],
) -> list[dict[str, Any]]:
    selected_by_row: dict[tuple[str, int], set[tuple[int, int]]] = defaultdict(set)
    for source_name, row_index, query_index, answer_index in selected_answer_keys:
        selected_by_row[(source_name, row_index)].add((query_index, answer_index))

    output_rows: list[dict[str, Any]] = []
    for row_key in sorted(selected_by_row):
        row = rows_by_key[row_key]
        selected_pairs = selected_by_row[row_key]
        out = copy.deepcopy(row)
        query_events = []
        for query_index, query in enumerate(row.get("query_events") or []):
            selected_answers = {
                answer_index
                for selected_query_index, answer_index in selected_pairs
                if selected_query_index == query_index
            }
            if not selected_answers:
                continue
            query_out = copy.deepcopy(query)
            query_out["answer_events"] = [
                copy.deepcopy(answer_event)
                for answer_index, answer_event in enumerate(query.get("answer_events") or [])
                if answer_index in selected_answers
            ]
            if query_out["answer_events"]:
                query_events.append(query_out)
        if query_events:
            out["query_events"] = query_events
            output_rows.append(out)
    return output_rows


def sample_pre_rows(rl_dir: Path, source_name: str, count: int, *, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count <= 0:
        return [], {"source": source_name, "requested_rows": count, "selected_rows": 0}
    rows = read_jsonl(rl_dir / source_name)
    if len(rows) < count:
        raise ValueError(f"{source_name}: available rows {len(rows)} < requested {count}")
    indices = rng.sample(range(len(rows)), count)
    selected = [copy.deepcopy(rows[index]) for index in indices]
    validate_pre_rows(selected, source_name=source_name)
    return selected, {
        "source": source_name,
        "available_rows": len(rows),
        "requested_rows": count,
        "selected_rows": len(selected),
        "selected_source_indices_preview": sorted(indices)[:20],
        "selected_stats": count_grouped_rows(selected),
    }


def validate_pre_rows(rows: list[dict[str, Any]], *, source_name: str) -> None:
    for row_index, row in enumerate(rows):
        if any(str(key).startswith("_source") for key in row):
            raise ValueError(f"{source_name}: sampled row {row_index} contains temporary source key")
        queries = row.get("query_events")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"{source_name}: sampled row {row_index} missing query_events")
        for query_index, query in enumerate(queries):
            answers = query.get("answer_events")
            if not isinstance(answers, list) or not answers:
                raise ValueError(f"{source_name}: sampled row {row_index} query {query_index} has no answer_events")
    assert_no_same_step_conflicts(rows)


def validate_grouped_rows(rows: list[dict[str, Any]]) -> None:
    for row_index, row in enumerate(rows):
        queries = row.get("query_events")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"output row {row_index}: missing query_events")
        if any(str(key).startswith("_source") for key in row):
            raise ValueError(f"output row {row_index}: contains temporary source key")
        for query_index, query in enumerate(queries):
            answers = query.get("answer_events")
            if not isinstance(answers, list) or not answers:
                raise ValueError(f"output row {row_index}: query {query_index} has no answer_events")
    assert_no_same_step_conflicts(rows)


def assert_no_same_step_conflicts(rows: list[dict[str, Any]], *, frames_per_step: int = 5) -> None:
    for row_index, row in enumerate(rows):
        frame_count = int(row.get("frame_count") or 0)
        if frame_count <= 0:
            raise ValueError(f"output row {row_index}: invalid frame_count={row.get('frame_count')!r}")
        query_groups: Counter[int] = Counter()
        answer_groups: Counter[int] = Counter()
        for query in row.get("query_events") or []:
            query_groups[timestamp_to_group(float(query.get("time") or 0.0), frame_count, frames_per_step)] += 1
            for answer_event in query.get("answer_events") or []:
                answer_groups[timestamp_to_group(float(answer_event.get("time") or 0.0), frame_count, frames_per_step)] += 1
        bad_queries = [group for group, count in query_groups.items() if count > 1]
        bad_answers = [group for group, count in answer_groups.items() if count > 1]
        if bad_queries or bad_answers:
            raise ValueError(
                f"output row {row_index}: same-step conflict "
                f"queries={bad_queries[:5]} answers={bad_answers[:5]}"
            )


def timestamp_to_group(timestamp: float, frame_count: int, frames_per_step: int) -> int:
    if timestamp <= 0:
        frame_id = 0
    else:
        frame_id = max(0, math.ceil(float(timestamp) - 1e-9) - 1)
    frame_id = min(max(frame_id, 0), frame_count - 1)
    return frame_id // max(1, frames_per_step)


def count_grouped_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: Counter[str] = Counter()
    query_answer_types: Counter[str] = Counter()
    answer_event_answer_types: Counter[str] = Counter()
    pass_rates: Counter[str] = Counter()
    query_events = 0
    answer_events = 0
    rows_with_multiple_queries = 0
    rows_with_multiple_answers = 0
    for row in rows:
        datasets[str(row.get("dataset") or "")] += 1
        queries = row.get("query_events") or []
        if len(queries) > 1:
            rows_with_multiple_queries += 1
        row_answer_count = 0
        for query in queries:
            query_events += 1
            answer_type = str(query.get("answer_type") or "")
            query_answer_types[answer_type] += 1
            answers = query.get("answer_events") or []
            row_answer_count += len(answers)
            for answer_event in answers:
                answer_events += 1
                answer_event_answer_types[answer_type] += 1
                pass_rates[rate_key(answer_event.get("eval_pass_rate"))] += 1
        if row_answer_count > 1:
            rows_with_multiple_answers += 1
    return {
        "rows": len(rows),
        "query_events": query_events,
        "answer_events": answer_events,
        "rows_with_multiple_queries": rows_with_multiple_queries,
        "rows_with_multiple_answers": rows_with_multiple_answers,
        "dataset_counts": dict(datasets.most_common()),
        "query_answer_type_counts": dict(query_answer_types.most_common()),
        "answer_event_answer_type_counts": dict(answer_event_answer_types.most_common()),
        "eval_pass_rate_counts": dict(sorted(pass_rates.items())),
    }


def count_selected_by_source_and_rate(
    selected_keys: set[tuple[str, int, int, int]],
    answer_meta: dict[tuple[str, int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key in selected_keys:
        source_name = key[0]
        meta = answer_meta[key]
        source_counts[source_name][meta["rate"]] += 1
        type_counts[meta["answer_type"]][meta["rate"]] += 1
    return {
        "by_source_file": {name: dict(sorted(counter.items())) for name, counter in sorted(source_counts.items())},
        "by_answer_type": {name: dict(sorted(counter.items())) for name, counter in sorted(type_counts.items())},
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rl_dir = args.rl_dir.resolve()
    rows_by_key, answer_meta, keys_by_rate = collect_answer_events(rl_dir, list(args.sources))
    selected_keys, selection_summary = select_answer_events(keys_by_rate, rng=rng, side_ratio=float(args.side_ratio))
    output_rows = build_grouped_rows(rows_by_key, selected_keys)
    pre_rows, pre_summary = sample_pre_rows(rl_dir, str(args.pre_source), int(args.pre_count), rng=rng)
    output_rows.extend(pre_rows)
    if not args.no_shuffle_output:
        rng.shuffle(output_rows)
    validate_grouped_rows(output_rows)
    write_jsonl(args.output.resolve(), output_rows)

    summary = {
        "output_path": str(args.output.resolve()),
        "summary_output": str(args.summary_output.resolve()),
        "random_seed": args.seed,
        "input_rl_dir": str(rl_dir),
        "input_sources": list(args.sources),
        "pre_source": str(args.pre_source),
        "sampling_unit": "answer_event",
        "output_unit": "video_row_with_pruned_query_events_plus_full_pre_rows",
        "selection": selection_summary,
        "pre_selection": pre_summary,
        "selected_breakdown": count_selected_by_source_and_rate(selected_keys, answer_meta),
        "output_stats": count_grouped_rows(output_rows),
    }
    write_json(args.summary_output.resolve(), summary)
    print(f"wrote {len(output_rows)} rows -> {args.output.resolve()}")
    print(json.dumps(summary["selection"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(summary["output_stats"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
