#!/usr/bin/env python3
"""Inspect exported LLaMAFactory ShareGPT rows for StreamWeave SFT."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streamweave.parser import strict_validate_raw_output


DEFAULT_INPUT = Path("data_engine/sft/outputs/gemini_answered_full/llamafactory_sharegpt.jsonl")
DELTA_RE = re.compile(r'<delta\s+t="(?P<t>[^"]+)">(?P<text>.*?)</delta>', flags=re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(?P<answer>.*?)</answer>", flags=re.DOTALL)


def main() -> None:
    args = parse_args()
    report = inspect_sharegpt(args.input, bridge_threshold=args.bridge_threshold, max_examples=args.max_examples)
    print_report(report, bridge_threshold=args.bridge_threshold)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inspect_sharegpt(path: Path, *, bridge_threshold: float, max_examples: int) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "input": str(path),
        "rows": 0,
        "answered_steps": 0,
        "silent_steps": 0,
        "missing_answer_tag": 0,
        "format_error_rows": 0,
        "sharegpt_structure_error_rows": 0,
        "delta_over_threshold_rows": 0,
        "delta_over_threshold_count": 0,
        "max_delta_duration": 0.0,
        "target_delta_over_threshold_rows": 0,
        "target_delta_over_threshold_count": 0,
        "max_target_delta_duration": 0.0,
        "memory_delta_over_threshold_rows": 0,
        "memory_delta_over_threshold_count": 0,
        "max_memory_delta_duration": 0.0,
        "format_issue_counts": Counter(),
        "examples": {
            "format_errors": [],
            "long_deltas": [],
            "structure_errors": [],
        },
    }
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            stats["rows"] += 1
            row_errors: list[str] = []
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                row_errors.append(f"json_decode_error: {exc}")
                target_xml = ""
                images: list[str] = []
                user_content = ""
            else:
                target_xml, user_content, images, row_errors = extract_row_content(row)

            if row_errors:
                stats["sharegpt_structure_error_rows"] += 1
                add_example(
                    stats["examples"]["structure_errors"],
                    max_examples,
                    {"line": line_no, "issues": row_errors},
                )

            answer_match = ANSWER_RE.search(target_xml)
            if answer_match is None:
                stats["missing_answer_tag"] += 1
            elif answer_match.group("answer").strip():
                stats["answered_steps"] += 1
            else:
                stats["silent_steps"] += 1

            parse = strict_validate_raw_output(target_xml)
            if not parse.parser_ok:
                stats["format_error_rows"] += 1
                issue_codes = [issue.code for issue in parse.issues]
                stats["format_issue_counts"].update(issue_codes)
                add_example(
                    stats["examples"]["format_errors"],
                    max_examples,
                    {
                        "line": line_no,
                        "issues": issue_codes,
                        "preview": compact_preview(target_xml),
                    },
                )

            target_long_deltas = long_bridge_items(target_xml, threshold=bridge_threshold)
            memory_long_deltas = long_bridge_items(extract_memory_block(user_content), threshold=bridge_threshold)
            long_deltas = target_long_deltas + memory_long_deltas
            if target_long_deltas:
                stats["target_delta_over_threshold_rows"] += 1
                stats["target_delta_over_threshold_count"] += len(target_long_deltas)
                stats["max_target_delta_duration"] = max(
                    float(stats["max_target_delta_duration"]),
                    max(item["duration"] for item in target_long_deltas),
                )
            if memory_long_deltas:
                stats["memory_delta_over_threshold_rows"] += 1
                stats["memory_delta_over_threshold_count"] += len(memory_long_deltas)
                stats["max_memory_delta_duration"] = max(
                    float(stats["max_memory_delta_duration"]),
                    max(item["duration"] for item in memory_long_deltas),
                )
            if long_deltas:
                stats["delta_over_threshold_rows"] += 1
                stats["delta_over_threshold_count"] += len(long_deltas)
                stats["max_delta_duration"] = max(
                    float(stats["max_delta_duration"]),
                    max(item["duration"] for item in long_deltas),
                )
                add_example(
                    stats["examples"]["long_deltas"],
                    max_examples,
                    {
                        "line": line_no,
                        "target_deltas": target_long_deltas[:3],
                        "memory_deltas": memory_long_deltas[:3],
                        "answer": answer_match.group("answer").strip() if answer_match else "",
                        "num_images": len(images),
                        "num_image_placeholders": user_content.count("<image>"),
                    },
                )

    stats["format_issue_counts"] = dict(stats["format_issue_counts"])
    return stats


def extract_row_content(row: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    errors: list[str] = []
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "", "", [], ["missing_or_invalid_messages"]

    user_content = ""
    assistant_content = ""
    for message in messages:
        if not isinstance(message, dict):
            errors.append("invalid_message_item")
            continue
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "user" and not user_content:
            user_content = content
        elif role == "assistant":
            assistant_content = content

    if not user_content:
        errors.append("missing_user_message")
    if not assistant_content:
        errors.append("missing_assistant_message")

    raw_images = row.get("images") or []
    images = [str(item) for item in raw_images] if isinstance(raw_images, list) else []
    if not isinstance(raw_images, list):
        errors.append("images_is_not_list")
    if user_content.count("<image>") != len(images):
        errors.append(
            f"image_placeholder_mismatch:{user_content.count('<image>')}!={len(images)}"
        )
    return assistant_content.strip(), user_content, images, errors


def long_bridge_items(target_xml: str, *, threshold: float) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for match in DELTA_RE.finditer(target_xml):
        interval = parse_interval(match.group("t"))
        if interval is None:
            continue
        start, end = interval
        duration = end - start
        if duration > threshold:
            items.append(
                {
                    "t": match.group("t"),
                    "duration": round(duration, 3),
                    "text": compact_preview(match.group("text"), limit=160),
                }
            )
    return items


MEMORY_BLOCK_RE = re.compile(
    r"=== Memory ===\s*(?P<memory>.*?)\s*=== Current frames ===",
    flags=re.DOTALL,
)


def extract_memory_block(user_content: str) -> str:
    match = MEMORY_BLOCK_RE.search(user_content)
    if match:
        return match.group("memory")
    return ""


def parse_interval(text: str) -> tuple[float, float] | None:
    match = re.match(r"\s*([0-9.]+)\s*[-\u2013]\s*([0-9.]+)\s*$", text)
    if not match:
        return None
    start = float(match.group(1))
    end = float(match.group(2))
    if end <= start:
        return None
    return start, end


def compact_preview(text: str, *, limit: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def add_example(examples: list[dict[str, Any]], max_examples: int, item: dict[str, Any]) -> None:
    if len(examples) < max_examples:
        examples.append(item)


def print_report(report: dict[str, Any], *, bridge_threshold: float) -> None:
    print(f"input: {report['input']}")
    print(f"rows: {report['rows']}")
    print(f"answered_steps: {report['answered_steps']}")
    print(f"silent_steps: {report['silent_steps']}")
    print(f"missing_answer_tag: {report['missing_answer_tag']}")
    print(f"format_error_rows: {report['format_error_rows']}")
    print(f"sharegpt_structure_error_rows: {report['sharegpt_structure_error_rows']}")
    print(
        f"target_or_memory_deltas_longer_than_{bridge_threshold:g}s: "
        f"rows={report['delta_over_threshold_rows']} count={report['delta_over_threshold_count']} "
        f"max_duration={report['max_delta_duration']:.3f}"
    )
    print(
        f"target_deltas_longer_than_{bridge_threshold:g}s: "
        f"rows={report['target_delta_over_threshold_rows']} count={report['target_delta_over_threshold_count']} "
        f"max_duration={report['max_target_delta_duration']:.3f}"
    )
    print(
        f"memory_deltas_longer_than_{bridge_threshold:g}s: "
        f"rows={report['memory_delta_over_threshold_rows']} count={report['memory_delta_over_threshold_count']} "
        f"max_duration={report['max_memory_delta_duration']:.3f}"
    )
    if report["format_issue_counts"]:
        print("format_issue_counts:")
        for code, count in sorted(report["format_issue_counts"].items(), key=lambda item: (-item[1], item[0])):
            print(f"  {code}: {count}")

    for name, examples in report["examples"].items():
        if not examples:
            continue
        print(f"{name}_examples:")
        for example in examples:
            print(json.dumps(example, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--delta-threshold", "--bridge-threshold", dest="bridge_threshold", type=float, default=20.0)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
