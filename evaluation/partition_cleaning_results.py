#!/usr/bin/env python3
"""Partition cleaned StreamWeave samples by duration and difficulty."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DURATION_BINS = "0,30,60,120,180,300,600"
DEFAULT_SPLIT_RATIOS = "train=0.9,val=0.05,test=0.05"


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    rows = [row for row in rows if _decision_allowed(row, args.keep_decisions)]
    rows = [annotate_row(row, duration_bins=parse_duration_bins(args.duration_bins)) for row in rows]
    rows = select_by_target_distribution(
        rows,
        max_samples=args.max_samples,
        duration_ratios=parse_ratios(args.duration_ratios),
        difficulty_ratios=parse_ratios(args.difficulty_ratios),
        seed=args.seed,
    )
    splits = split_rows(rows, split_ratios=parse_ratios(args.split_ratios), seed=args.seed)
    write_outputs(splits, Path(args.output_dir))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def annotate_row(row: dict[str, Any], *, duration_bins: list[float]) -> dict[str, Any]:
    out = dict(row)
    duration = _duration_seconds(out)
    out["duration_seconds"] = duration
    out["duration_bucket"] = duration_bucket(duration, duration_bins)
    out["difficulty"] = out.get("difficulty") or difficulty_from_pass_rate(out.get("pass_rate"))
    return out


def select_by_target_distribution(
    rows: list[dict[str, Any]],
    *,
    max_samples: int,
    duration_ratios: dict[str, float],
    difficulty_ratios: dict[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples <= 0 or max_samples >= len(rows):
        return rows

    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[(str(row.get("duration_bucket", "unknown")), str(row.get("difficulty", "unknown")))].append(row)
    for items in by_stratum.values():
        rng.shuffle(items)

    duration_ratios = duration_ratios or empirical_ratios(str(row.get("duration_bucket", "unknown")) for row in rows)
    difficulty_ratios = difficulty_ratios or empirical_ratios(str(row.get("difficulty", "unknown")) for row in rows)

    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for duration_name, duration_ratio in duration_ratios.items():
        for difficulty_name, difficulty_ratio in difficulty_ratios.items():
            target = int(round(max_samples * duration_ratio * difficulty_ratio))
            if target <= 0:
                continue
            bucket = by_stratum.get((duration_name, difficulty_name), [])
            for row in bucket[:target]:
                selected.append(row)
                used_ids.add(id(row))

    if len(selected) < max_samples:
        leftovers = [row for row in rows if id(row) not in used_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])
    if len(selected) > max_samples:
        rng.shuffle(selected)
        selected = selected[:max_samples]
    selected.sort(key=lambda row: int(row.get("index", 0) or 0))
    return selected


def split_rows(rows: list[dict[str, Any]], *, split_ratios: dict[str, float], seed: int) -> dict[str, list[dict[str, Any]]]:
    if not split_ratios:
        split_ratios = parse_ratios(DEFAULT_SPLIT_RATIOS)
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row.get("duration_bucket", "unknown")), str(row.get("difficulty", "unknown")))].append(row)

    splits = {name: [] for name in split_ratios}
    names = list(split_ratios)
    for items in strata.values():
        rng.shuffle(items)
        total = len(items)
        cumulative = []
        running = 0.0
        for name in names:
            running += split_ratios[name]
            cumulative.append((name, running))
        for idx, row in enumerate(items):
            position = (idx + 0.5) / max(total, 1)
            for name, threshold in cumulative:
                if position <= threshold or name == names[-1]:
                    splits[name].append(row)
                    break

    for items in splits.values():
        items.sort(key=lambda row: int(row.get("index", 0) or 0))
    return splits


def write_outputs(splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for split_name, rows in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                out = dict(row)
                out["split"] = split_name
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                all_rows.append(out)
    summary = summarize_splits(splits)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(format_summary(summary) + "\n", encoding="utf-8")


def summarize_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}, "total": sum(len(rows) for rows in splits.values())}
    for split_name, rows in splits.items():
        pass_rates = [float(row["pass_rate"]) for row in rows if row.get("pass_rate") is not None]
        summary["splits"][split_name] = {
            "count": len(rows),
            "avg_pass_rate": (sum(pass_rates) / len(pass_rates)) if pass_rates else None,
            "duration_bucket_counts": dict(Counter(str(row.get("duration_bucket", "unknown")) for row in rows)),
            "difficulty_counts": dict(Counter(str(row.get("difficulty", "unknown")) for row in rows)),
            "category_counts": dict(Counter(str(row.get("category", "")) for row in rows)),
            "task_counts": dict(Counter(str(row.get("task", "")) for row in rows)),
        }
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = ["# Partition Summary", "", f"- total: {summary.get('total', 0)}"]
    for split_name, split in (summary.get("splits") or {}).items():
        lines.extend(["", f"## {split_name}", f"- count: {split.get('count', 0)}", f"- avg_pass_rate: {_fmt(split.get('avg_pass_rate'))}"])
        for key in ("duration_bucket_counts", "difficulty_counts", "category_counts", "task_counts"):
            lines.append(f"- {key}: {split.get(key, {})}")
    return "\n".join(lines)


def duration_bucket(duration: float | None, bins: list[float]) -> str:
    if duration is None:
        return "unknown"
    previous = 0.0
    for boundary in bins:
        if duration <= boundary:
            return f"{previous:g}-{boundary:g}s"
        previous = boundary
    return f">{bins[-1]:g}s" if bins else "unknown"


def difficulty_from_pass_rate(value: Any) -> str:
    if value is None:
        return "unscored"
    pass_rate = float(value)
    if pass_rate >= 1.0:
        return "easy"
    if pass_rate >= 2.0 / 3.0:
        return "medium"
    if pass_rate >= 1.0 / 3.0:
        return "hard"
    return "unsolved"


def parse_duration_bins(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text or "").split(",") if item.strip()]
    return sorted(value for value in values if value > 0)


def parse_ratios(text: str) -> dict[str, float]:
    if not text:
        return {}
    out: dict[str, float] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        name, value = item.split("=", 1)
        out[name.strip()] = float(value)
    total = sum(out.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in out.items()}


def empirical_ratios(values: Any) -> dict[str, float]:
    counts = Counter(str(value) for value in values)
    total = sum(counts.values())
    return {key: count / total for key, count in counts.items()} if total else {}


def _decision_allowed(row: dict[str, Any], allowed_text: str) -> bool:
    allowed = {item.strip() for item in allowed_text.split(",") if item.strip()}
    return not allowed or str(row.get("filter_decision", "")) in allowed


def _duration_seconds(row: dict[str, Any]) -> float | None:
    for key in ("duration_seconds", "video_duration", "duration", "target_timestamp", "query_timestamp"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Aggregate scores_3x.jsonl from run_data_cleaning_3x.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-bins", default=DEFAULT_DURATION_BINS)
    parser.add_argument("--difficulty-ratios", default="", help='Example: "easy=0.25,medium=0.45,hard=0.30"')
    parser.add_argument("--duration-ratios", default="", help='Example: "0-30s=0.2,30-60s=0.4,60-120s=0.4"')
    parser.add_argument("--split-ratios", default=DEFAULT_SPLIT_RATIOS)
    parser.add_argument("--keep-decisions", default="keep_strong,keep_review")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
