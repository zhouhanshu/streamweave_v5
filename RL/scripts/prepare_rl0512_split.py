#!/usr/bin/env python3
"""Create deterministic train/validation split for StreamWeave RL exp2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset2/rl_0512.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("dataset2/rl_0512_train.jsonl"))
    parser.add_argument("--val-output", type=Path, default=Path("dataset2/rl_0512_val.jsonl"))
    parser.add_argument("--val-size", type=int, default=80)
    parser.add_argument("--seed", default="exp2_rlmlr_val")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[video_key(row)].append(row)

    val_keys = choose_val_groups(groups, target_rows=args.val_size, seed=args.seed)
    val_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for row in rows:
        if video_key(row) in val_keys:
            val_rows.append(row)
        else:
            train_rows.append(row)

    if len(val_rows) != args.val_size:
        raise SystemExit(f"Expected {args.val_size} validation rows, got {len(val_rows)}")

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.val_output, val_rows)
    print(f"source: {args.input} rows={len(rows)} videos={len(groups)}")
    print(f"train:  {args.train_output} rows={len(train_rows)} videos={len({video_key(r) for r in train_rows})}")
    print(f"val:    {args.val_output} rows={len(val_rows)} videos={len({video_key(r) for r in val_rows})}")
    print("val dataset:", dict(Counter(str(row.get("dataset", "")) for row in val_rows)))
    print("val difficulty:", dict(Counter(str(row.get("difficulty_v2", "")) for row in val_rows)))
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def video_key(row: dict[str, Any]) -> str:
    for key in ("video_id", "video", "id", "sample_id"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError(f"row has no stable video key: {row}")


def choose_val_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    target_rows: int,
    seed: str,
) -> set[str]:
    ranked = sorted(groups, key=lambda key: stable_hash(seed, key))
    selected: set[str] = set()
    count = 0
    deferred: list[str] = []

    for key in ranked:
        size = len(groups[key])
        if count + size <= target_rows:
            selected.add(key)
            count += size
        else:
            deferred.append(key)
        if count == target_rows:
            return selected

    for key in deferred:
        size = len(groups[key])
        if count + size <= target_rows:
            selected.add(key)
            count += size
        if count == target_rows:
            return selected
    raise ValueError(f"Could not select exactly {target_rows} validation rows; selected {count}")


def stable_hash(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
