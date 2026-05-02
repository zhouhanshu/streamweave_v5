"""Backend base classes."""

from __future__ import annotations

import re
from typing import Any

from streamweave.schemas import BackendResult, ContentItem


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
        intervals = re.findall(r'<frame id="(\d+)" t="([0-9.]+)-([0-9.]+)">', text)
        if intervals:
            frame_id, start, end = intervals[0]
            bridge_end = intervals[-1][2]
        else:
            frame_id, start, end, bridge_end = "1", "0.0", "1.0", "1.0"
        if 'role="q"' in text:
            output = (
                "<eta></eta>\n"
                "<answer>A</answer>\n"
                f'<bridge t="{start}-{bridge_end}">Mock bridge for the current frames.</bridge>'
            )
        else:
            output = (
                "<eta></eta>\n"
                "<answer></answer>\n"
                f'<note t="{start}-{end}" frame="{frame_id}"/>'
            )
        return BackendResult(text=output, latency_seconds=0.0, endpoint_id="mock", attempt_count=1)
