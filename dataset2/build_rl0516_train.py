#!/usr/bin/env python3
"""Build the 0516 RL train file by adding yes/no evidence-sufficiency data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RL_DIR = ROOT / "0516data" / "rl"
DEFAULT_OUTPUT = RL_DIR / "rl_0516_train.jsonl"
DEFAULT_SUMMARY = RL_DIR / "rl_0516_train.summary.json"
RANDOM_SEED = 516

SOURCES = (
    ("rl_0515_train", "0516data/rl/rl_0515_train.jsonl"),
    ("rl_0515_yesno", "0516data/rl/rl_0515_yesno.jsonl"),
)

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
    parser.add_argument("--dataset2-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
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


def validate_rows(dataset2_root: Path, rows: list[dict[str, Any]]) -> None:
    for row_index, row in enumerate(rows):
        extra_top = set(row) - TOP_KEYS
        if extra_top:
            raise ValueError(f"row {row_index}: extra top-level keys {sorted(extra_top)}")
        if not isinstance(row.get("query_events"), list) or not row["query_events"]:
            raise ValueError(f"row {row_index}: missing query_events")
        dataset = str(row.get("dataset") or "")
        frames_dir = str(row.get("frames_dir") or row.get("video") or "")
        if not dataset or not frames_dir:
            raise ValueError(f"row {row_index}: missing dataset/frames_dir")
        path = dataset2_root / dataset / frames_dir
        if not path.is_dir():
            raise FileNotFoundError(f"row {row_index}: missing frames dir {path}")


def count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    policies: Counter[str] = Counter()
    answer_texts: Counter[str] = Counter()
    query_events = 0
    answer_events = 0
    for row in rows:
        datasets[str(row.get("dataset") or "")] += 1
        for query in row.get("query_events") or []:
            query_events += 1
            answer_types[str(query.get("answer_type") or "")] += 1
            policies[str(query.get("answer_policy") or "")] += 1
            for answer in query.get("answer_events") or []:
                answer_events += 1
                answer_text = str(answer.get("answer") or answer.get("content") or "")
                if answer_text in {"Yes", "No"}:
                    answer_texts[answer_text] += 1
    return {
        "rows": len(rows),
        "query_events": query_events,
        "answer_events": answer_events,
        "dataset_counts": dict(datasets.most_common()),
        "answer_type_counts": dict(answer_types.most_common()),
        "answer_policy_counts": dict(policies.most_common()),
        "exact_yes_no_answer_counts": dict(answer_texts.most_common()),
    }


def main() -> None:
    args = parse_args()
    dataset2_root = args.dataset2_root.resolve()
    loaded: list[tuple[str, dict[str, Any]]] = []
    source_summary: list[dict[str, Any]] = []

    for source_name, rel_path in SOURCES:
        rows = read_jsonl(dataset2_root / rel_path)
        validate_rows(dataset2_root, rows)
        source_summary.append({"name": source_name, "path": rel_path, **count_rows(rows)})
        loaded.extend((source_name, row) for row in rows)

    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(loaded)

    output_rows = [row for _source_name, row in loaded]
    validate_rows(dataset2_root, output_rows)
    write_jsonl(args.output.resolve(), output_rows)
    summary = {
        "random_seed": None if args.no_shuffle else args.seed,
        "output": str(args.output.resolve()),
        "summary_output": str(args.summary_output.resolve()),
        "sources": source_summary,
        "merged": count_rows(output_rows),
    }
    write_json(args.summary_output.resolve(), summary)
    print(f"wrote {len(output_rows)} rows -> {args.output.resolve()}")
    print(json.dumps(summary["merged"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
