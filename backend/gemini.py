"""Gemini backend through VertexAI google-genai."""

from __future__ import annotations

import os
import time
from typing import Any

from streamweave.config import BackendConfig, RuntimeConfig
from streamweave.schemas import BackendResult, ContentItem

from .base import BaseBackend
from .image_utils import image_to_jpeg_bytes
from .retry import RetryPolicy, run_with_retry


DEFAULT_VERTEX_PROJECT = "mmu-jichu-shangyehua-gemini"
DEFAULT_VERTEX_LOCATION = "global"
PRO_SAFE_THINKING_BUDGET = 128
DEFAULT_SAFETY_THRESHOLD = "BLOCK_NONE"


class GeminiBackend(BaseBackend):
    def __init__(self, config: BackendConfig, runtime: RuntimeConfig) -> None:
        from google import genai
        from google.genai import types

        self.config = config
        self.runtime = runtime
        self.types = types
        self.client = genai.Client(
            vertexai=True,
            project=os.environ.get("GEMINI_VERTEX_PROJECT", DEFAULT_VERTEX_PROJECT),
            location=os.environ.get("GEMINI_VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION),
            http_options=types.HttpOptions(timeout=int(config.timeout_seconds * 1000)),
        )
        self.retry_policy = RetryPolicy(
            max_retries=config.max_retries,
            backoff_seconds=config.retry_backoff_seconds,
            backoff_multiplier=config.retry_backoff_multiplier,
            retryable_error_patterns=config.retryable_error_patterns,
            non_retryable_error_patterns=config.non_retryable_error_patterns,
        )

    def generate(
        self,
        content: list[ContentItem],
        *,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> BackendResult:
        started = time.time()

        def call() -> str:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=[self.types.Content(role="user", parts=self._to_parts(content))],
                config=self._generate_config(generate_kwargs),
            )
            text = _extract_text(response)
            if not text:
                raise RuntimeError(f"Gemini response has no text content. {_describe_response(response)}")
            return text.strip()

        text, attempt_count, retry_errors = run_with_retry(call, self.retry_policy)
        return BackendResult(
            text=text,
            latency_seconds=time.time() - started,
            endpoint_id="gemini",
            attempt_count=attempt_count,
            retry_errors=retry_errors,
        )

    def _generate_config(self, overrides: dict[str, Any] | None = None) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_output_tokens": self.config.max_tokens,
        }
        if overrides:
            for key in (
                "temperature",
                "top_p",
                "max_output_tokens",
                "response_mime_type",
                "response_schema",
                "response_json_schema",
            ):
                if key in overrides and overrides[key] is not None:
                    kwargs[key] = overrides[key]
        budget = self.config.thinking_budget
        if budget is None:
            budget = resolve_thinking_budget(self.config.model)
        if budget is not None:
            kwargs["thinking_config"] = {"thinking_budget": budget}
        safety_settings = self._safety_settings()
        if safety_settings:
            kwargs["safety_settings"] = safety_settings
        return self.types.GenerateContentConfig(**kwargs)

    def _safety_settings(self) -> list[Any]:
        threshold_name = os.environ.get("GEMINI_SAFETY_THRESHOLD", DEFAULT_SAFETY_THRESHOLD).strip().upper()
        if threshold_name in {"", "DEFAULT"}:
            return []
        threshold = getattr(self.types.HarmBlockThreshold, threshold_name, None)
        if threshold is None:
            raise ValueError(
                f"Unknown GEMINI_SAFETY_THRESHOLD={threshold_name!r}. "
                "Use BLOCK_NONE, OFF, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, or DEFAULT."
            )
        categories = [
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
            "HARM_CATEGORY_IMAGE_HATE",
            "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
            "HARM_CATEGORY_IMAGE_HARASSMENT",
            "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
        ]
        settings = []
        for name in categories:
            category = getattr(self.types.HarmCategory, name, None)
            if category is None:
                continue
            settings.append(self.types.SafetySetting(category=category, threshold=threshold))
        return settings

    def _to_parts(self, content: list[ContentItem]) -> list[Any]:
        parts: list[Any] = []
        for item in content:
            if item.type == "text":
                parts.append(self.types.Part.from_text(text=item.text))
            elif item.type == "image" and item.image_path is not None:
                parts.append(
                    self.types.Part.from_bytes(
                        data=image_to_jpeg_bytes(
                            item.image_path,
                            max_side=self.runtime.resolution,
                            quality=self.config.image_quality,
                        ),
                        mime_type="image/jpeg",
                    )
                )
        return parts


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text is not None and str(text).strip():
        return str(text).strip()
    pieces: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            value = getattr(part, "text", None)
            if value:
                pieces.append(str(value))
    return "\n".join(pieces).strip()


def resolve_thinking_budget(model: str) -> int | None:
    name = model.lower()
    if "gemini-2.5-flash" in name:
        return 0
    if "gemini-2.5-pro" in name:
        return PRO_SAFE_THINKING_BUDGET
    return None


def _describe_response(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    info: dict[str, Any] = {
        "candidate_count": len(candidates),
        "prompt_feedback": _safe_repr(getattr(response, "prompt_feedback", None), limit=800),
    }
    candidate_info = []
    for idx, candidate in enumerate(candidates[:3]):
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        candidate_info.append(
            {
                "index": idx,
                "finish_reason": _safe_repr(getattr(candidate, "finish_reason", None), limit=200),
                "ratings": _safe_repr(getattr(candidate, "safety_ratings", None), limit=800),
                "part_count": len(parts),
                "part_types": [_part_type(part) for part in parts[:8]],
            }
        )
    info["candidates"] = candidate_info
    return f"diagnostics={info}"


def _part_type(part: Any) -> str:
    if getattr(part, "text", None):
        return "text"
    if getattr(part, "inline_data", None):
        return "inline_data"
    if getattr(part, "function_call", None):
        return "function_call"
    if getattr(part, "function_response", None):
        return "function_response"
    if getattr(part, "executable_code", None):
        return "executable_code"
    if getattr(part, "code_execution_result", None):
        return "code_execution_result"
    return type(part).__name__


def _safe_repr(value: Any, *, limit: int) -> str:
    text = repr(value)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text
