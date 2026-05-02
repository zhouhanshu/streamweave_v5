#!/usr/bin/env python3
"""Inspect StreamWeave SFT intermediate files step by step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


def main() -> None:
    args = parse_args()
    sample, steps = load_intermediate(args.input)
    selected_steps = set(args.step or [])
    lines: list[str] = []

    if sample:
        lines.extend(format_sample_header(sample))
    for row in steps:
        step_index = int(row.get("step_index", -1))
        if selected_steps and step_index not in selected_steps:
            continue
        lines.extend(format_step(row, attempts_mode=args.attempts, max_text_chars=args.max_text_chars))

    text = "\n".join(lines).rstrip() + "\n"
    output = args.output or default_output_path(args.input)
    if args.stdout:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"[inspect] wrote {output}")


def load_intermediate(path: Path) -> tuple[JsonDict, list[JsonDict]]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        sample_rows = [row for row in rows if isinstance(row.get("steps"), list)]
        if sample_rows:
            steps: list[JsonDict] = []
            for row in sample_rows:
                steps.extend(row.get("steps") or [])
            return {}, steps
        return {}, rows

    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if isinstance(value, list):
        return {}, [row for row in value if isinstance(row, dict)]
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported intermediate format: {path}")
    if isinstance(value.get("steps"), list):
        return value, [row for row in value.get("steps") or [] if isinstance(row, dict)]
    return {}, [value]


def read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def format_sample_header(sample: JsonDict) -> list[str]:
    checks = sample.get("checks") if isinstance(sample.get("checks"), dict) else {}
    answer_check = checks.get("answer_check") if isinstance(checks.get("answer_check"), dict) else {}
    return [
        "# Sample",
        f"sample_id: {sample.get('sample_id', '')}",
        f"video_id: {sample.get('video_id', '')}",
        f"qa_id: {sample.get('qa_id', '')}",
        f"question_type: {sample.get('question_type', '')}",
        f"status: {sample.get('status', '')}",
        f"usable_for_sft: {sample.get('usable_for_sft', '')}",
        f"answer_correct: {sample.get('answer_correct', '')}",
        f"failure_reason: {sample.get('failure_reason', '')}",
        f"steps: {sample.get('num_steps', '')}/{sample.get('num_expected_steps', '')}",
        f"expected_answer: {answer_check.get('expected_letter', '')} {answer_check.get('expected_answer', '')}".rstrip(),
        "",
    ]


def format_step(row: JsonDict, *, attempts_mode: str, max_text_chars: int) -> list[str]:
    out = [
        "=" * 100,
        f"Step {row.get('step_index')} | t={row.get('step_start')}-{row.get('step_end')} | failed={row.get('task_failed', False)} | accepted_attempt={row.get('accepted_attempt_index', '')}",
        "",
        "Current Frames:",
    ]
    out.extend(format_current_frames(row.get("current_frames")))
    out.extend(["", "QA History:", format_qa_history(row.get("qa_history")), "", "Memory Before:"])
    out.append(format_memory_before(row))
    out.extend(["", "Target XML:", clip_text(str(row.get("target_xml") or ""), max_text_chars)])

    attempts = select_attempts(row.get("attempts") or [], row.get("accepted_attempt_index"), attempts_mode)
    if attempts:
        out.extend(["", "Attempts:"])
    for attempt in attempts:
        out.extend(format_attempt(attempt, max_text_chars=max_text_chars))
    return out


def format_current_frames(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["<empty/>"]
    lines = []
    for item in value:
        if not isinstance(item, dict):
            continue
        t = item.get("t") or ["", ""]
        start = t[0] if isinstance(t, list) and t else ""
        end = t[1] if isinstance(t, list) and len(t) > 1 else ""
        lines.append(
            f'- frame_id={item.get("frame_id")} global_frame_id={item.get("global_frame_id")} '
            f't={start}-{end} image={item.get("image_path", "")}'
        )
    return lines or ["<empty/>"]


def format_qa_history(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "<empty/>"
    lines = []
    for item in value:
        if not isinstance(item, dict):
            continue
        lines.append(f'<qa t="{item.get("t", item.get("timestamp", ""))}" role="{item.get("role", "")}">{item.get("text", "")}</qa>')
    return "\n".join(lines) if lines else "<empty/>"


def format_memory_before(row: JsonDict) -> str:
    text = str(row.get("memory_before_text") or "").strip()
    if text:
        return text
    memory = row.get("memory_before")
    if not isinstance(memory, list) or not memory:
        return "<empty/>"
    lines = []
    for item in memory:
        if not isinstance(item, dict):
            continue
        t = item.get("t") or ["", ""]
        start = t[0] if isinstance(t, list) and t else ""
        end = t[1] if isinstance(t, list) and len(t) > 1 else ""
        if item.get("type") == "bridge":
            lines.append(f'<bridge t="{start}-{end}">{item.get("text", "")}</bridge>')
        elif item.get("type") == "note":
            lines.append(f'<note t="{start}-{end}" image="{item.get("image_path", "")}"></note>')
    return "\n".join(lines) if lines else "<empty/>"


def select_attempts(attempts: list[Any], accepted_index: Any, mode: str) -> list[JsonDict]:
    valid_attempts = [attempt for attempt in attempts if isinstance(attempt, dict)]
    if mode == "none":
        return []
    if mode == "all":
        return valid_attempts
    if mode == "failed":
        return [attempt for attempt in valid_attempts if not attempt.get("accepted")]
    if mode == "accepted":
        return [attempt for attempt in valid_attempts if attempt.get("accepted")]
    if mode == "last":
        return valid_attempts[-1:] if valid_attempts else []
    raise ValueError(f"Unsupported attempts mode: {mode}")


def format_attempt(attempt: JsonDict, *, max_text_chars: int) -> list[str]:
    quality = attempt.get("quality") if isinstance(attempt.get("quality"), dict) else {}
    backend = attempt.get("backend_result") if isinstance(attempt.get("backend_result"), dict) else {}
    out = [
        "-" * 100,
        f"Attempt {attempt.get('attempt_index')} | accepted={attempt.get('accepted')} | valid={quality.get('valid')} | parser_ok={quality.get('parser_ok')} | backend_attempts={backend.get('attempt_count', '')}",
    ]
    issues = quality.get("issues") if isinstance(quality.get("issues"), list) else []
    if issues:
        out.append("Issues:")
        for issue in issues:
            if isinstance(issue, dict):
                out.append(f"- {issue.get('code')}: {issue.get('message')}")
    retry_errors = backend.get("retry_errors") if isinstance(backend.get("retry_errors"), list) else []
    if retry_errors:
        out.append("Backend Retry Errors:")
        out.extend(f"- {error}" for error in retry_errors)

    prompt_text = str(attempt.get("prompt_text") or "")
    if prompt_text:
        out.extend(["Prompt Input Without System/Few-Shot:", clip_text(clean_prompt_text(prompt_text), max_text_chars)])

    raw = str(attempt.get("raw_output") or "")
    out.extend(["Raw Output:", clip_text(raw, max_text_chars)])
    feedback = str(attempt.get("feedback") or "")
    if feedback:
        out.extend(["Retry Feedback:", clip_text(feedback, max_text_chars)])
    return out


def clean_prompt_text(text: str) -> str:
    if "[Actual Input]" in text:
        actual = text.split("[Actual Input]", 1)[1]
        before_task, after_task = split_once(actual, "Task Instructions:")
        sections = ["[Actual Input]" + before_task.rstrip()]
        teacher_context = extract_section(after_task, "=== Teacher Context ===", "=== Retry Feedback ===")
        retry_feedback = extract_section(after_task, "=== Retry Feedback ===", None)
        if teacher_context:
            sections.append(teacher_context.rstrip())
        if retry_feedback:
            sections.append(retry_feedback.rstrip())
        return "\n\n".join(section for section in sections if section.strip()).strip()

    if "=== Memory ===" in text:
        start = text.find("=== Memory ===")
        end_candidates = [idx for marker in ("Other Rules:", "Output Format:") if (idx := text.find(marker, start)) >= 0]
        end = min(end_candidates) if end_candidates else len(text)
        return text[start:end].strip()
    return text.strip()


def split_once(text: str, marker: str) -> tuple[str, str]:
    if marker not in text:
        return text, ""
    before, after = text.split(marker, 1)
    return before, after


def extract_section(text: str, start_marker: str, end_marker: str | None) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = len(text)
    if end_marker is not None:
        found = text.find(end_marker, start + len(start_marker))
        if found >= 0:
            end = found
    return text[start:end]


def clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n...[truncated {len(text) - max_chars} chars]"


def default_output_path(path: Path) -> Path:
    suffix = "".join(path.suffixes)
    if suffix:
        return path.with_name(path.name[: -len(suffix)] + ".inspect.txt")
    return path.with_name(path.name + ".inspect.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Sample JSON, sample-record JSONL, or sft_steps.jsonl.")
    parser.add_argument("--step", type=int, action="append", help="Only print one step index. Can be repeated.")
    parser.add_argument(
        "--attempts",
        choices=("all", "accepted", "failed", "last", "none"),
        default="all",
        help="Which attempts to print for each step.",
    )
    parser.add_argument("--max-text-chars", type=int, default=0, help="Truncate long text blocks. 0 means no truncation.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output text file.")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of writing a txt file.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
