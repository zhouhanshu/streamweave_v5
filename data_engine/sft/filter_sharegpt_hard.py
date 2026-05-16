#!/usr/bin/env python3
"""Hard-filter StreamWeave LLaMAFactory ShareGPT exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streamweave.parser import strict_validate_raw_output


ANSWER_RE = re.compile(r"<answer>(?P<answer>.*?)</answer>", flags=re.DOTALL)
DELTA_T_RE = re.compile(r'<delta\s+[^>]*t="(?P<t>[^"]+)"', flags=re.DOTALL)
EVENT_RE = re.compile(r"<(?P<tag>anchor|delta)\b(?P<attrs>[^>]*)>", flags=re.DOTALL)
T_ATTR_RE = re.compile(r't="(?P<t>[^"]+)"')
INTERVAL_RE = re.compile(r"\s*([0-9.]+)\s*[-–]\s*([0-9.]+)\s*$")
MEMORY_BLOCK_RE = re.compile(
    r"=== Memory ===\s*(?P<memory>.*?)\s*=== Current frames ===",
    flags=re.DOTALL,
)


def main() -> None:
    args = parse_args()
    summary = filter_file(
        input_path=args.input,
        output_path=args.output,
        image_root=args.image_root,
        delta_threshold=args.delta_threshold,
        max_examples=args.max_examples,
    )
    summary_path = args.summary_output or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(summary, args.output, summary_path)


def filter_file(
    *,
    input_path: Path,
    output_path: Path,
    image_root: Path,
    delta_threshold: float,
    max_examples: int,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "image_root": str(image_root),
        "delta_threshold": delta_threshold,
        "rows": 0,
        "kept": 0,
        "dropped": 0,
        "answered_kept": 0,
        "silent_kept": 0,
        "answered_dropped": 0,
        "silent_dropped": 0,
        "drop_reasons": Counter(),
        "examples": {},
    }
    seen: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            stats["rows"] += 1
            reasons: list[str] = []
            answered = False
            row: dict[str, Any] | None = None
            user_content = ""
            assistant_content = ""
            images: list[str] = []

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                reasons.append("bad_json")

            if isinstance(row, dict):
                user_content, assistant_content, images, structure_errors = extract_row(row)
                reasons.extend(structure_errors)
                answer_match = ANSWER_RE.search(assistant_content)
                if answer_match is None:
                    reasons.append("missing_answer_tag")
                else:
                    answered = bool(answer_match.group("answer").strip())

                if assistant_content:
                    parse = strict_validate_raw_output(assistant_content)
                    if not parse.parser_ok:
                        reasons.append("format_error")

                target_long_delta_count, _ = count_long_deltas(assistant_content, delta_threshold)
                if target_long_delta_count:
                    reasons.append("target_delta_over_threshold")

                memory_long_delta_count, _ = count_long_deltas(extract_memory_block(user_content), delta_threshold)
                if memory_long_delta_count:
                    reasons.append("memory_delta_over_threshold")

                if user_content and is_first_step_user_prompt(user_content):
                    ok, code = first_step_anchor_ok(assistant_content)
                    if not ok:
                        reasons.append(code)

                missing_images = [path for path in images if not resolve_image(path, image_root).is_file()]
                if missing_images:
                    reasons.append("missing_image")

                dedup_key = row_digest(row)
                if dedup_key in seen:
                    reasons.append("duplicate_row")
                else:
                    seen.add(dedup_key)

            if reasons:
                stats["dropped"] += 1
                if answered:
                    stats["answered_dropped"] += 1
                else:
                    stats["silent_dropped"] += 1
                for reason in sorted(set(reasons)):
                    stats["drop_reasons"][reason] += 1
                    add_example(
                        stats["examples"],
                        reason,
                        max_examples,
                        {
                            "line": line_no,
                            "answered": answered,
                            "preview": compact_preview(assistant_content),
                        },
                    )
                continue

            stats["kept"] += 1
            if answered:
                stats["answered_kept"] += 1
            else:
                stats["silent_kept"] += 1
            fout.write(line if line.endswith("\n") else line + "\n")

    stats["drop_reasons"] = dict(stats["drop_reasons"])
    return stats


def extract_row(row: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
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
    if not isinstance(raw_images, list):
        errors.append("images_is_not_list")
        images: list[str] = []
    else:
        images = [str(item) for item in raw_images]
    if user_content.count("<image>") != len(images):
        errors.append("image_placeholder_mismatch")
    return user_content, assistant_content, images, errors


def count_long_deltas(text: str, threshold: float) -> tuple[int, float]:
    count = 0
    max_delta = 0.0
    for match in DELTA_T_RE.finditer(text):
        interval = parse_interval(match.group("t"))
        if interval is None:
            continue
        start, end = interval
        duration = end - start
        max_delta = max(max_delta, duration)
        if duration > threshold + 1e-9:
            count += 1
    return count, max_delta


def extract_memory_block(user_content: str) -> str:
    match = MEMORY_BLOCK_RE.search(user_content)
    if match:
        return match.group("memory")
    return ""


def is_first_step_user_prompt(user_content: str) -> bool:
    memory_block = extract_memory_block(user_content)
    if not memory_block:
        return False
    memory = " ".join(memory_block.split())
    return memory == "<empty/>"


def first_step_anchor_ok(assistant_content: str) -> tuple[bool, str]:
    first_event = EVENT_RE.search(assistant_content)
    if first_event is None:
        return False, "first_step_missing_observation"
    if first_event.group("tag") != "anchor":
        return False, "first_step_first_event_not_anchor"
    t_match = T_ATTR_RE.search(first_event.group("attrs"))
    if t_match is None:
        return False, "first_step_anchor_missing_t"
    interval = parse_interval(t_match.group("t"))
    if interval is None:
        return False, "first_step_anchor_bad_t"
    start, end = interval
    if abs(start - 0.0) > 1e-6 or abs(end - 1.0) > 1e-6:
        return False, "first_step_anchor_not_0_1"
    return True, ""


def parse_interval(text: str) -> tuple[float, float] | None:
    match = INTERVAL_RE.match(text)
    if not match:
        return None
    start = float(match.group(1))
    end = float(match.group(2))
    if end <= start:
        return None
    return start, end


def resolve_image(path: str, image_root: Path) -> Path:
    image_path = Path(path)
    if image_path.is_absolute():
        return image_path
    return image_root / image_path


def row_digest(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def compact_preview(text: str, *, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


def add_example(examples: dict[str, list[dict[str, Any]]], reason: str, max_examples: int, item: dict[str, Any]) -> None:
    bucket = examples.setdefault(reason, [])
    if len(bucket) < max_examples:
        bucket.append(item)


def print_summary(summary: dict[str, Any], output_path: Path, summary_path: Path) -> None:
    print(f"input: {summary['input']}")
    print(f"output: {output_path}")
    print(f"rows: {summary['rows']}")
    print(f"kept: {summary['kept']}")
    print(f"dropped: {summary['dropped']}")
    print(f"answered_kept: {summary['answered_kept']}")
    print(f"silent_kept: {summary['silent_kept']}")
    print(f"answered_dropped: {summary['answered_dropped']}")
    print(f"silent_dropped: {summary['silent_dropped']}")
    if summary["drop_reasons"]:
        print("drop_reasons:")
        for reason, count in sorted(summary["drop_reasons"].items(), key=lambda item: (-item[1], item[0])):
            print(f"  {reason}: {count}")
    print(f"summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--delta-threshold", type=float, default=20.0)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
