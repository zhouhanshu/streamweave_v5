#!/usr/bin/env python3
"""Allocate cleaned samples into SFT/RL pools with teacher rescue support."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    student_rows = load_jsonl(Path(args.student_scores))
    teacher_rows = load_jsonl(Path(args.teacher_scores)) if args.teacher_scores else []
    teacher_by_sample = {str(row.get("sample_id", "")): row for row in teacher_rows}

    rng = random.Random(args.seed)
    allocator = CurriculumAllocator(args=args, rng=rng, teacher_by_sample=teacher_by_sample)
    outputs = allocator.allocate(student_rows)
    write_outputs(outputs, Path(args.output_dir))


class CurriculumAllocator:
    def __init__(self, *, args: argparse.Namespace, rng: random.Random, teacher_by_sample: dict[str, dict[str, Any]]) -> None:
        self.args = args
        self.rng = rng
        self.teacher_by_sample = teacher_by_sample

    def allocate(self, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        outputs: dict[str, list[dict[str, Any]]] = {
            "sft": [],
            "rl": [],
            "drop": [],
            "unsolved_for_teacher": [],
            "teacher_correct_trajectories": [],
            "all_allocations": [],
        }
        medium_pool: list[dict[str, Any]] = []
        hard_pool: list[dict[str, Any]] = []

        for row in rows:
            difficulty = difficulty_from_row(row)
            if difficulty == "unsolved":
                self._handle_unsolved(row, outputs)
                continue
            if not protocol_ok(row, min_parser=self.args.min_parser_ok_rate, min_quality=self.args.min_quality_valid_rate):
                self._append(outputs, "drop", row, source="student", reason="student_protocol_unstable")
                continue
            duration = duration_seconds(row)
            if difficulty == "easy":
                if duration is not None and duration <= self.args.easy_sft_max_seconds:
                    self._append(outputs, "sft", row, source="student_easy_short", reason="easy_and_short")
                else:
                    self._append(outputs, "rl", row, source="student_easy_long", reason="easy_but_long")
            elif difficulty == "medium":
                medium_pool.append(row)
            elif difficulty == "hard":
                hard_pool.append(row)
            else:
                self._append(outputs, "drop", row, source="student", reason=f"unsupported_difficulty:{difficulty}")

        self._allocate_weighted_pool(
            medium_pool,
            outputs,
            difficulty="medium",
            sft_ratio=self.args.medium_sft_ratio,
            rl_ratio=self.args.medium_rl_ratio,
        )
        self._allocate_weighted_pool(
            hard_pool,
            outputs,
            difficulty="hard",
            sft_ratio=self.args.hard_sft_ratio,
            rl_ratio=self.args.hard_rl_ratio,
        )
        return outputs

    def _handle_unsolved(self, row: dict[str, Any], outputs: dict[str, list[dict[str, Any]]]) -> None:
        sample_id = str(row.get("sample_id", ""))
        teacher_row = self.teacher_by_sample.get(sample_id)
        if teacher_row is None:
            self._append(outputs, "unsolved_for_teacher", row, source="student_unsolved", reason="needs_teacher_3x")
            return
        correct_runs = correct_runs_from_row(teacher_row)
        if not correct_runs:
            dropped = dict(row)
            dropped["teacher_pass_count"] = int(teacher_row.get("pass_count", 0) or 0)
            dropped["teacher_pass_rate"] = teacher_row.get("pass_rate")
            self._append(outputs, "drop", dropped, source="teacher_rescue", reason="teacher_all_failed")
            return
        rescued = dict(row)
        rescued["teacher_pass_count"] = len(correct_runs)
        rescued["teacher_pass_rate"] = teacher_row.get("pass_rate")
        rescued["teacher_answers"] = teacher_row.get("answers", [])
        rescued["teacher_correct_runs"] = correct_runs
        self._append(outputs, "sft", rescued, source="teacher_rescue", reason="unsolved_rescued_by_teacher")
        self._append(outputs, "rl", rescued, source="teacher_rescue", reason="unsolved_rescued_by_teacher")
        for run in correct_runs:
            outputs["teacher_correct_trajectories"].append(
                {
                    "sample_id": sample_id,
                    "video_id": row.get("video_id"),
                    "repeat_index": run.get("repeat_index"),
                    "score": run.get("score"),
                    "response": run.get("response"),
                    "trace_dir": run.get("trace_dir"),
                    "trace_jsonl": run.get("trace_jsonl"),
                    "trace_txt": run.get("trace_txt"),
                    "source": "teacher_rescue",
                }
            )

    def _allocate_weighted_pool(
        self,
        pool: list[dict[str, Any]],
        outputs: dict[str, list[dict[str, Any]]],
        *,
        difficulty: str,
        sft_ratio: float,
        rl_ratio: float,
    ) -> None:
        if not pool:
            return
        sft_count = round_count(len(pool), sft_ratio)
        rl_count = round_count(len(pool), rl_ratio)
        if not self.args.allow_overlap and sft_count + rl_count > len(pool):
            scale = len(pool) / max(sft_count + rl_count, 1)
            sft_count = int(round(sft_count * scale))
            rl_count = len(pool) - sft_count

        sft_rows = weighted_sample(
            pool,
            sft_count,
            rng=self.rng,
            min_weight_seconds=self.args.min_weight_seconds,
            length_weight_power=self.args.length_weight_power,
        )
        sft_ids = {id(row) for row in sft_rows}
        rl_source = pool if self.args.allow_overlap else [row for row in pool if id(row) not in sft_ids]
        rl_rows = weighted_sample(
            rl_source,
            rl_count,
            rng=self.rng,
            min_weight_seconds=self.args.min_weight_seconds,
            length_weight_power=self.args.length_weight_power,
        )
        selected_ids = set()
        for row in sft_rows:
            selected_ids.add(id(row))
            self._append(outputs, "sft", row, source=f"student_{difficulty}", reason=f"{difficulty}_length_weighted_sft")
        for row in rl_rows:
            selected_ids.add(id(row))
            self._append(outputs, "rl", row, source=f"student_{difficulty}", reason=f"{difficulty}_length_weighted_rl")
        for row in pool:
            if id(row) not in selected_ids:
                self._append(outputs, "drop", row, source=f"student_{difficulty}", reason=f"{difficulty}_not_selected_by_ratio")

    def _append(
        self,
        outputs: dict[str, list[dict[str, Any]]],
        target: str,
        row: dict[str, Any],
        *,
        source: str,
        reason: str,
    ) -> None:
        item = dict(row)
        item["allocation_target"] = target
        item["allocation_source"] = source
        item["allocation_reason"] = reason
        outputs[target].append(item)
        outputs["all_allocations"].append(
            {
                "sample_id": item.get("sample_id"),
                "video_id": item.get("video_id"),
                "difficulty": difficulty_from_row(item),
                "duration_seconds": duration_seconds(item),
                "pass_rate": item.get("pass_rate"),
                "target": target,
                "source": source,
                "reason": reason,
            }
        )


def weighted_sample(
    rows: list[dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
    min_weight_seconds: float,
    length_weight_power: float,
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return list(rows)
    keyed = []
    for row in rows:
        duration = duration_seconds(row)
        weight = max(float(duration or 0.0), float(min_weight_seconds))
        weight = max(weight, 1e-6) ** max(float(length_weight_power), 0.0)
        key = rng.random() ** (1.0 / weight)
        keyed.append((key, row))
    keyed.sort(key=lambda item: item[0], reverse=True)
    return [row for _key, row in keyed[:count]]


def correct_runs_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for run in row.get("runs", []) or []:
        try:
            score = int(run.get("score") or 0)
        except (TypeError, ValueError):
            score = 0
        if score == 1 and not run.get("error"):
            out.append(dict(run))
    return out


def protocol_ok(row: dict[str, Any], *, min_parser: float, min_quality: float) -> bool:
    parser_rate = float(row.get("parser_ok_rate", 0.0) or 0.0)
    quality_rate = float(row.get("quality_valid_rate", 0.0) or 0.0)
    return parser_rate >= min_parser and quality_rate >= min_quality


def difficulty_from_row(row: dict[str, Any]) -> str:
    difficulty = str(row.get("difficulty") or "").strip()
    if difficulty:
        return difficulty
    pass_rate = row.get("pass_rate")
    if pass_rate is None:
        return "unscored"
    value = float(pass_rate)
    if value >= 1.0:
        return "easy"
    if value >= 2.0 / 3.0:
        return "medium"
    if value >= 1.0 / 3.0:
        return "hard"
    return "unsolved"


def duration_seconds(row: dict[str, Any]) -> float | None:
    for key in ("duration_seconds", "video_duration", "duration", "target_timestamp", "query_timestamp"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def round_count(total: int, ratio: float) -> int:
    ratio = max(float(ratio), 0.0)
    if total <= 0 or ratio <= 0:
        return 0
    return min(total, int(round(total * ratio)))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_outputs(outputs: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_outputs(outputs)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(format_summary(summary) + "\n", encoding="utf-8")


def summarize_outputs(outputs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {
        "counts": {name: len(rows) for name, rows in outputs.items()},
        "targets": {},
    }
    for name, rows in outputs.items():
        pass_rates = [float(row["pass_rate"]) for row in rows if row.get("pass_rate") is not None]
        summary["targets"][name] = {
            "count": len(rows),
            "avg_pass_rate": (sum(pass_rates) / len(pass_rates)) if pass_rates else None,
            "difficulty_counts": dict(Counter(difficulty_from_row(row) for row in rows)),
            "source_counts": dict(Counter(str(row.get("allocation_source", "")) for row in rows)),
            "reason_counts": dict(Counter(str(row.get("allocation_reason", "")) for row in rows)),
        }
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = ["# Curriculum Allocation Summary", "", "## Counts"]
    for name, count in (summary.get("counts") or {}).items():
        lines.append(f"- {name}: {count}")
    for name, data in (summary.get("targets") or {}).items():
        lines.extend(["", f"## {name}", f"- count: {data.get('count', 0)}", f"- avg_pass_rate: {_fmt(data.get('avg_pass_rate'))}"])
        for key in ("difficulty_counts", "source_counts", "reason_counts"):
            lines.append(f"- {key}: {data.get(key, {})}")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-scores", required=True, help="scores_3x.jsonl from the SFT/student model")
    parser.add_argument("--teacher-scores", default="", help="optional scores_3x.jsonl from a stronger teacher on student-unsolved samples")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--easy-sft-max-seconds", type=float, default=60.0)
    parser.add_argument("--medium-sft-ratio", type=float, default=0.4)
    parser.add_argument("--medium-rl-ratio", type=float, default=0.6)
    parser.add_argument("--hard-sft-ratio", type=float, default=0.5)
    parser.add_argument("--hard-rl-ratio", type=float, default=0.5)
    parser.add_argument("--min-parser-ok-rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--min-quality-valid-rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--min-weight-seconds", type=float, default=5.0)
    parser.add_argument("--length-weight-power", type=float, default=1.0)
    parser.add_argument("--allow-overlap", action="store_true", help="allow medium/hard samples to appear in both SFT and RL pools")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
