"""Backend base classes."""

from __future__ import annotations

import re
from typing import Any

from streamweave.schemas import BackendResult, ContentItem


FRAME_RE = re.compile(r'<frame\s+t="([0-9.]+)-([0-9.]+)">')
OBS_RE = re.compile(
    r'<anchor\b[^>]*\bt="(?P<anchor_t>[^"]+)"[^>]*>'
    r'|<delta\b[^>]*\bt="(?P<delta_t>[^"]+)"[^>]*>',
    flags=re.DOTALL,
)
ACTIVE_Q_RE = re.compile(r'<qa\s+t="[^"]+"\s+role="q">')


class BaseBackend:
    def generate(
        self,
        content: list[ContentItem],
        *,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> BackendResult:
        raise NotImplementedError


class MockBackend(BaseBackend):
    def generate(
        self,
        content: list[ContentItem],
        *,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> BackendResult:
        text = "".join(item.text for item in content if item.type == "text")
        actual_input = _actual_input_block(text)
        current_frames = _section_between(actual_input, "=== Current frames ===", "=== QA History ===")
        memory = _section_between(actual_input, "=== Memory ===", "=== Current frames ===")
        qa_history = _section_after(actual_input, "=== QA History ===")
        qa_history = qa_history.split("Task instructions:", 1)[0]

        intervals = FRAME_RE.findall(current_frames)
        if intervals:
            start, end = intervals[0]
            bridge_end = intervals[-1][1]
        else:
            start, end, bridge_end = "0.0", "1.0", "1.0"

        open_tail_start = _open_tail_bridge_start(memory)
        active_question = ACTIVE_Q_RE.search(qa_history) is not None
        if active_question:
            bridge_start = open_tail_start or start
            output = (
                "<state>The current frames include an active question, and the mock backend returns a placeholder answer for validation.</state>\n"
                "<answer>A</answer>\n"
                f'<delta t="{bridge_start}-{bridge_end}">Mock delta for the current frames.</delta>'
            )
        else:
            if open_tail_start is not None:
                observation = f'<delta t="{open_tail_start}-{bridge_end}">Mock delta extends the open memory tail.</delta>'
            else:
                observation = f'<anchor t="{start}-{end}"></anchor>'
                if float(bridge_end) > float(end):
                    observation += f'\n<delta t="{end}-{bridge_end}">Mock delta after the current anchor frame.</delta>'
            output = (
                "<state>The current frames are observed without an active question, so the mock backend keeps the answer empty.</state>\n"
                "<answer></answer>\n"
                f"{observation}"
            )
        return BackendResult(text=output, latency_seconds=0.0, endpoint_id="mock", attempt_count=1)


def _actual_input_block(text: str) -> str:
    return text.rsplit("[Actual Input]", 1)[-1]


def _section_between(text: str, start_marker: str, end_marker: str) -> str:
    if start_marker not in text:
        return ""
    after_start = text.split(start_marker, 1)[1]
    if end_marker not in after_start:
        return after_start
    return after_start.split(end_marker, 1)[0]


def _section_after(text: str, marker: str) -> str:
    if marker not in text:
        return ""
    return text.split(marker, 1)[1]


def _open_tail_bridge_start(memory: str) -> str | None:
    last_kind = ""
    last_bridge_start: str | None = None
    for match in OBS_RE.finditer(memory):
        if match.group("anchor_t") is not None:
            last_kind = "note"
            last_bridge_start = None
            continue
        delta_t = match.group("delta_t")
        if delta_t is None:
            continue
        start = delta_t.split("-", 1)[0].strip()
        if not start:
            continue
        last_kind = "bridge"
        last_bridge_start = start
    return last_bridge_start if last_kind == "bridge" else None
