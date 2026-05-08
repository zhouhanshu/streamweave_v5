"""Deterministic backend used by local SFT pipeline tests."""

from __future__ import annotations

import re

from backend.base import MockBackend
from streamweave.schemas import BackendResult, ContentItem


class SFTMockBackend(MockBackend):
    """Deterministic local backend for testing the SFT pipeline without model calls."""

    def generate(self, content: list[ContentItem], *, generate_kwargs=None) -> BackendResult:
        text = "".join(item.text for item in content if item.type == "text")
        actual = text.split("[Actual Input]", 1)[-1]
        current_block = actual
        if "=== Current frames ===" in actual:
            current_block = actual.split("=== Current frames ===", 1)[1].split("=== QA History ===", 1)[0]
        intervals = re.findall(r'<frame\s+t="([0-9.]+)-([0-9.]+)">', current_block)
        if not intervals:
            output = (
                "<state>No current frames were available, so no visual state can be updated.</state>\n"
                "<answer></answer>"
            )
        else:
            start, end = intervals[0]
            step_start = float(start)
            step_end = float(intervals[-1][1])
            memory_block = actual.split("=== Current frames ===", 1)[0]
            open_tail_start = _mock_open_tail_bridge_start(memory_block)
            has_question = re.search(r'<qa\b[^>]*role="q"[^>]*>', actual) is not None
            answer = _mock_answer_from_retry_feedback(text)
            if answer is None:
                answer = "A" if has_question else ""
            lines = [
                "<state>The mock backend summarizes the current memory and frame window for SFT validation.</state>",
                f"<answer>{answer}</answer>",
            ]
            note_intervals = [(start, end)] if "There has not been a recent <note>" in text else []
            if open_tail_start is not None:
                lines.extend(
                    _mock_observation_lines(
                        note_intervals,
                        bridge_start=open_tail_start,
                        step_end=step_end,
                        bridge_text="Mock observation extends the current open-tail bridge.",
                    )
                )
            else:
                selected_note_intervals = note_intervals or [(start, end)]
                lines.extend(
                    _mock_observation_lines(
                        selected_note_intervals,
                        bridge_start=step_start,
                        step_end=step_end,
                        bridge_text="Mock observation for the current streaming window.",
                    )
                )
            output = "\n".join(lines)
        return BackendResult(text=output, latency_seconds=0.0, endpoint_id="sft_mock", attempt_count=1)


def _mock_open_tail_bridge_start(memory_block: str) -> float | None:
    events = list(re.finditer(r'<(?P<kind>note|bridge)\b(?P<attrs>[^>]*)>', memory_block))
    if not events or events[-1].group("kind") != "bridge":
        return None
    match = re.search(r't="([0-9.]+)-([0-9.]+)"', events[-1].group("attrs"))
    if match is None:
        return None
    return float(match.group(1))


def _mock_answer_from_retry_feedback(text: str) -> str | None:
    if "=== Retry Feedback ===" not in text:
        return None
    feedback = text.split("=== Retry Feedback ===", 1)[1]
    template = re.search(
        r"Required answer tag:\s*<answer>(.*?)</answer>",
        feedback,
        flags=re.DOTALL,
    )
    if template is None:
        return None
    answer_template = template.group(1).strip()
    if not answer_template:
        return ""
    if answer_template.startswith("..."):
        return "A"
    return answer_template


def _mock_observation_lines(
    note_intervals: list[tuple[str, str]],
    *,
    bridge_start: float,
    step_end: float,
    bridge_text: str,
) -> list[str]:
    lines: list[str] = []
    cursor = bridge_start
    for start_text, end_text in sorted(note_intervals, key=lambda item: float(item[0])):
        note_start = float(start_text)
        note_end = float(end_text)
        if note_start > cursor:
            lines.append(f'<bridge t="{cursor:.1f}-{note_start:.1f}">{bridge_text}</bridge>')
        lines.append(f'<note t="{note_start:.1f}-{note_end:.1f}"></note>')
        cursor = max(cursor, note_end)
    if step_end > cursor:
        lines.append(f'<bridge t="{cursor:.1f}-{step_end:.1f}">{bridge_text}</bridge>')
    if not lines:
        lines.append(f'<bridge t="{bridge_start:.1f}-{step_end:.1f}">{bridge_text}</bridge>')
    return lines
