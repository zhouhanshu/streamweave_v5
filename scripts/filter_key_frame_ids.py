#!/usr/bin/env python3
"""Downsample key_frame_ids so no selected ids are adjacent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


DEFAULT_INPUT = Path("streamweave_v4/dataset/streamweave_data/annotations_qa_filter.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to <input stem>_nonadjacent_keyframes.jsonl.",
    )
    parser.add_argument("--field", default="key_frame_ids")
    parser.add_argument("--count-field", default="key_frame_count")
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_nonadjacent_keyframes{input_path.suffix}")


def downsample_non_adjacent(ids: list[Any]) -> list[int]:
    normalized = sorted({int(item) for item in ids})
    kept: list[int] = []
    for frame_id in normalized:
        if not kept or frame_id - kept[-1] > 1:
            kept.append(frame_id)
    return kept


def has_adjacent(ids: list[int]) -> bool:
    return any(right - left == 1 for left, right in zip(ids, ids[1:]))


def process_row(row: JsonDict, *, field: str, count_field: str) -> tuple[JsonDict, dict[str, int]]:
    out = dict(row)
    raw_ids = out.get(field)
    if not isinstance(raw_ids, list):
        return out, {"before": 0, "after": 0, "changed": 0, "had_adjacent": 0}

    before_ids = sorted({int(item) for item in raw_ids})
    after_ids = downsample_non_adjacent(raw_ids)
    out[field] = after_ids
    if count_field:
        out[count_field] = len(after_ids)

    return out, {
        "before": len(before_ids),
        "after": len(after_ids),
        "changed": int(before_ids != after_ids),
        "had_adjacent": int(has_adjacent(before_ids)),
    }


def filter_jsonl(input_path: Path, output_path: Path, *, field: str, count_field: str) -> dict[str, int]:
    stats = {
        "rows": 0,
        "rows_changed": 0,
        "rows_with_adjacent_before": 0,
        "total_keyframes_before": 0,
        "total_keyframes_after": 0,
        "rows_with_adjacent_after": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with input_path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_no} is not a JSON object")
            out, row_stats = process_row(row, field=field, count_field=count_field)
            ids = out.get(field) or []
            if isinstance(ids, list) and has_adjacent([int(item) for item in ids]):
                stats["rows_with_adjacent_after"] += 1

            stats["rows"] += 1
            stats["rows_changed"] += row_stats["changed"]
            stats["rows_with_adjacent_before"] += row_stats["had_adjacent"]
            stats["total_keyframes_before"] += row_stats["before"]
            stats["total_keyframes_after"] += row_stats["after"]
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")

    tmp_path.replace(output_path)
    return stats


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or default_output_path(input_path)
    stats = filter_jsonl(input_path, output_path, field=args.field, count_field=args.count_field)
    print(json.dumps({"input": str(input_path), "output": str(output_path), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
