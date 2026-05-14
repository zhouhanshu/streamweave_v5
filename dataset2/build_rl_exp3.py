#!/usr/bin/env python3
"""Build the exp3 RL mixture from 0514 query-event annotations.

The default recipe keeps all one-question multi-answer rows and downsamples
one-question one-answer rows to reach the requested total size.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_DATASET2_ROOT = Path("/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset2")
DEFAULT_INPUTS = (
    "Streamo-Instruct-465K/query_events_0514_filtered.jsonl",
    "CogStream/query_events_0514_filtered.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset2-root", type=Path, default=DEFAULT_DATASET2_ROOT)
    parser.add_argument("--inputs", nargs="+", default=list(DEFAULT_INPUTS))
    parser.add_argument("--output", default="rl_exp3.jsonl")
    parser.add_argument("--summary-suffix", default=".summary.json")
    parser.add_argument("--target-size", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=511)
    parser.add_argument(
        "--single-answer-strata",
        default="dataset,source_dataset,query_type,split",
        help="Comma-separated fields used to stratify one-answer downsampling.",
    )
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Shuffle final output after sampling. Default preserves input order.",
    )
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


def query_events(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = row.get("query_events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, Mapping)]


def answer_event_count(row: Mapping[str, Any]) -> int:
    total = 0
    for event in query_events(row):
        answers = event.get("answer_events")
        if isinstance(answers, list):
            total += sum(1 for answer in answers if isinstance(answer, Mapping))
    return total


def is_one_question_one_answer(row: Mapping[str, Any]) -> bool:
    events = query_events(row)
    return len(events) == 1 and answer_event_count(row) == 1


def is_one_question_multi_answer(row: Mapping[str, Any]) -> bool:
    events = query_events(row)
    return len(events) == 1 and answer_event_count(row) > 1


def first_query_type(row: Mapping[str, Any]) -> str:
    events = query_events(row)
    if not events:
        return ""
    return str(events[0].get("query_type") or "")


def stratum_key(row: Mapping[str, Any], fields: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        if field == "query_type":
            values.append(first_query_type(row))
        else:
            values.append(str(row.get(field) or ""))
    return tuple(values)


def allocate_quotas(sizes: Mapping[tuple[str, ...], int], target: int) -> dict[tuple[str, ...], int]:
    total = sum(sizes.values())
    if target < 0:
        raise ValueError(f"Cannot allocate a negative target: {target}")
    if total <= target:
        return dict(sizes)
    raw: list[tuple[tuple[str, ...], int, float, float]] = []
    floor_sum = 0
    for key, size in sizes.items():
        exact = target * size / total
        floor_value = int(exact)
        floor_sum += floor_value
        raw.append((key, size, exact, exact - floor_value))
    quotas = {key: int(exact) for key, _size, exact, _remainder in raw}
    remaining = target - floor_sum
    for key, _size, _exact, _remainder in sorted(raw, key=lambda item: (-item[3], item[0]))[:remaining]:
        quotas[key] += 1
    return quotas


def sample_single_answer_rows(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    target: int,
    fields: list[str],
    rng: random.Random,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    by_stratum: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in rows:
        by_stratum[stratum_key(item[1], fields)].append(item)
    sizes = {key: len(items) for key, items in by_stratum.items()}
    quotas = allocate_quotas(sizes, target)
    sampled: list[tuple[int, dict[str, Any]]] = []
    stratum_summary: dict[str, dict[str, int]] = {}
    for key in sorted(by_stratum):
        items = list(by_stratum[key])
        quota = quotas.get(key, 0)
        if quota >= len(items):
            chosen = items
        else:
            chosen = rng.sample(items, quota)
        sampled.extend(chosen)
        stratum_summary["|".join(key)] = {"available": len(items), "sampled": len(chosen)}
    return sampled, stratum_summary


def count_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    answer_count_distribution: Counter[int] = Counter()
    dataset_counts: Counter[str] = Counter()
    source_dataset_counts: Counter[str] = Counter()
    query_type_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    one_answer = 0
    multi_answer = 0
    other_shape = 0
    for row in rows:
        dataset_counts[str(row.get("dataset") or "")] += 1
        source_dataset_counts[str(row.get("source_dataset") or "")] += 1
        split_counts[str(row.get("split") or "")] += 1
        answer_count = answer_event_count(row)
        answer_count_distribution[answer_count] += 1
        if is_one_question_one_answer(row):
            one_answer += 1
        elif is_one_question_multi_answer(row):
            multi_answer += 1
        else:
            other_shape += 1
        for event in query_events(row):
            query_type_counts[str(event.get("query_type") or "")] += 1
    return {
        "rows": sum(answer_count_distribution.values()),
        "one_question_one_answer": one_answer,
        "one_question_multi_answer": multi_answer,
        "other_shape": other_shape,
        "answer_event_count_distribution": {
            str(key): answer_count_distribution[key] for key in sorted(answer_count_distribution)
        },
        "dataset_counts": dict(dataset_counts.most_common()),
        "source_dataset_counts": dict(source_dataset_counts.most_common()),
        "split_counts": dict(split_counts.most_common()),
        "query_type_counts": dict(query_type_counts.most_common()),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.random_seed)
    fields = [field.strip() for field in args.single_answer_strata.split(",") if field.strip()]

    indexed_rows: list[tuple[int, dict[str, Any]]] = []
    input_summaries: list[dict[str, Any]] = []
    next_index = 0
    for input_name in args.inputs:
        input_path = args.dataset2_root / input_name
        rows = list(read_jsonl(input_path))
        input_summaries.append({"path": str(input_path), **count_rows(rows)})
        for row in rows:
            indexed_rows.append((next_index, row))
            next_index += 1

    multi_rows = [item for item in indexed_rows if is_one_question_multi_answer(item[1])]
    single_rows = [item for item in indexed_rows if is_one_question_one_answer(item[1])]
    other_rows = [
        item for item in indexed_rows if not is_one_question_multi_answer(item[1]) and not is_one_question_one_answer(item[1])
    ]

    always_keep = multi_rows + other_rows
    if len(always_keep) > args.target_size:
        raise ValueError(
            f"Cannot build target_size={args.target_size}: rows kept before single-answer sampling already total "
            f"{len(always_keep)}."
        )

    single_target = args.target_size - len(always_keep)
    sampled_single_rows, stratum_summary = sample_single_answer_rows(
        single_rows,
        target=single_target,
        fields=fields,
        rng=rng,
    )
    output_indexed = always_keep + sampled_single_rows
    if args.shuffle_output:
        rng.shuffle(output_indexed)
    else:
        output_indexed = sorted(output_indexed, key=lambda item: item[0])
    output_rows = [row for _index, row in output_indexed]

    output_path = args.dataset2_root / args.output
    written = write_jsonl(output_path, output_rows)
    summary = {
        "output_path": str(output_path),
        "target_size": args.target_size,
        "written_rows": written,
        "random_seed": args.random_seed,
        "single_answer_strata": fields,
        "shuffle_output": bool(args.shuffle_output),
        "inputs": input_summaries,
        "input_total": count_rows([row for _index, row in indexed_rows]),
        "sampling": {
            "kept_multi_answer_rows": len(multi_rows),
            "kept_other_shape_rows": len(other_rows),
            "available_single_answer_rows": len(single_rows),
            "sampled_single_answer_rows": len(sampled_single_rows),
            "dropped_single_answer_rows": len(single_rows) - len(sampled_single_rows),
            "single_answer_strata": stratum_summary,
        },
        "output": count_rows(output_rows),
    }
    write_json(output_path.with_suffix(output_path.suffix + args.summary_suffix), summary)
    print(f"Wrote {written} rows to {output_path}")


if __name__ == "__main__":
    main()
