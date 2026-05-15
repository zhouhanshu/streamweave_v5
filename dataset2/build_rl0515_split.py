#!/usr/bin/env python3
"""Sample the 0515 train/val RL split from the five 0514 canonical files."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_OUTPUT = ROOT / "rl_0515_train.jsonl"
DEFAULT_VAL_OUTPUT = ROOT / "rl_0515_val.jsonl"
DEFAULT_SUMMARY_OUTPUT = ROOT / "rl_0515_split.summary.json"
RANDOM_SEED = 511

SOURCES = (
    {
        "name": "pre",
        "path": "rl_0514_pre.jsonl",
        "train": 800,
        "val": 20,
    },
    {
        "name": "cogstream",
        "path": "CogStream/rl_0514_normalized.jsonl",
        "train": 1000,
        "val": 30,
    },
    {
        "name": "streamo_unable",
        "path": "Streamo-Instruct-465K/rl_0514_unable_normalized.jsonl",
        "train": 300,
        "val": 15,
    },
    {
        "name": "streamo_one",
        "path": "Streamo-Instruct-465K/rl_0514_one_normalized.jsonl",
        "train": 220,
        "val": 15,
    },
    {
        "name": "streamo_multi",
        "path": "Streamo-Instruct-465K/rl_0514_multi_normalized.jsonl",
        "train": 280,
        "val": 20,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset2-root", type=Path, default=ROOT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--val-output", type=Path, default=DEFAULT_VAL_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-shuffle-output", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
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


def sample_source(
    rows: list[dict[str, Any]],
    *,
    train_count: int,
    val_count: int,
    rng: random.Random,
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    need = train_count + val_count
    if len(rows) < need:
        raise ValueError(f"not enough rows: available={len(rows)}, requested={need}")
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    train_indices = indices[:train_count]
    val_indices = indices[train_count:need]
    train = [(index, rows[index]) for index in train_indices]
    val = [(index, rows[index]) for index in val_indices]
    return train, val


def output_rows(items: list[tuple[str, int, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for _source_path, _source_row_index, row in items]


def row_key(item: tuple[str, int, dict[str, Any]]) -> tuple[str, int]:
    return item[0], item[1]


def validate_split(
    *,
    dataset2_root: Path,
    train_items: list[tuple[str, int, dict[str, Any]]],
    val_items: list[tuple[str, int, dict[str, Any]]],
) -> None:
    train_keys = {row_key(item) for item in train_items}
    val_keys = {row_key(item) for item in val_items}
    overlap = train_keys & val_keys
    if overlap:
        raise ValueError(f"train/val source-row overlap: {sorted(overlap)[:10]}")
    for split_name, items in (("train", train_items), ("val", val_items)):
        for source_path, source_row_index, row in items:
            if not isinstance(row.get("query_events"), list) or not row["query_events"]:
                raise ValueError(f"{split_name}:{source_path}:{source_row_index}: missing query_events")
            dataset = str(row.get("dataset") or "")
            frames_dir = str(row.get("frames_dir") or row.get("video") or "")
            if not dataset or not frames_dir:
                raise ValueError(f"{split_name}:{source_path}:{source_row_index}: missing dataset/frames_dir")
            path = dataset2_root / dataset / frames_dir
            if not path.is_dir():
                raise FileNotFoundError(f"{split_name}:{source_path}:{source_row_index}: missing frames dir {path}")


def count_items(items: list[tuple[str, int, dict[str, Any]]]) -> dict[str, Any]:
    datasets: Counter[str] = Counter()
    source_files: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    policies: Counter[str] = Counter()
    query_events = 0
    answer_events = 0
    for source_path, _index, row in items:
        source_files[source_path] += 1
        datasets[str(row.get("dataset") or "")] += 1
        for event in row.get("query_events") or []:
            query_events += 1
            answer_types[str(event.get("answer_type") or "")] += 1
            policies[str(event.get("answer_policy") or "")] += 1
            answer_events += len(event.get("answer_events") or [])
    return {
        "rows": len(items),
        "query_events": query_events,
        "answer_events": answer_events,
        "dataset_counts": dict(datasets.most_common()),
        "source_file_counts": dict(source_files.most_common()),
        "answer_type_counts": dict(answer_types.most_common()),
        "answer_policy_counts": dict(policies.most_common()),
    }


def main() -> None:
    args = parse_args()
    dataset2_root = args.dataset2_root.resolve()
    rng = random.Random(args.seed)
    train_items: list[tuple[str, int, dict[str, Any]]] = []
    val_items: list[tuple[str, int, dict[str, Any]]] = []
    source_summary: list[dict[str, Any]] = []

    for source in SOURCES:
        rel_path = str(source["path"])
        rows = read_jsonl(dataset2_root / rel_path)
        train, val = sample_source(
            rows,
            train_count=int(source["train"]),
            val_count=int(source["val"]),
            rng=rng,
        )
        train_items.extend((rel_path, index, row) for index, row in train)
        val_items.extend((rel_path, index, row) for index, row in val)
        source_summary.append(
            {
                "name": source["name"],
                "path": rel_path,
                "available": len(rows),
                "train": len(train),
                "val": len(val),
                "unused": len(rows) - len(train) - len(val),
            }
        )

    validate_split(dataset2_root=dataset2_root, train_items=train_items, val_items=val_items)
    if not args.no_shuffle_output:
        rng.shuffle(train_items)
        rng.shuffle(val_items)

    train_rows = output_rows(train_items)
    val_rows = output_rows(val_items)
    write_jsonl(args.train_output.resolve(), train_rows)
    write_jsonl(args.val_output.resolve(), val_rows)
    summary = {
        "random_seed": args.seed,
        "train_output": str(args.train_output.resolve()),
        "val_output": str(args.val_output.resolve()),
        "source_row_overlap": 0,
        "video_overlap_allowed": True,
        "sources": source_summary,
        "train": count_items(train_items),
        "val": count_items(val_items),
    }
    write_json(args.summary_output.resolve(), summary)
    print(f"wrote train {len(train_rows)} -> {args.train_output.resolve()}")
    print(f"wrote val {len(val_rows)} -> {args.val_output.resolve()}")
    print(f"wrote summary -> {args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
