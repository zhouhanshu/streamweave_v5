"""OpenAI-compatible backend, including local vLLM services."""

from __future__ import annotations

import time
from typing import Any

from streamweave.config import BackendConfig, RuntimeConfig
from streamweave.schemas import BackendResult, ContentItem

from .base import BaseBackend
from .image_utils import image_to_data_url
from .retry import RetryPolicy, run_with_retry


class OpenAICompatibleBackend(BaseBackend):
    def __init__(self, config: BackendConfig, runtime: RuntimeConfig, *, endpoint: str | None = None) -> None:
        from openai import OpenAI

        self.config = config
        self.runtime = runtime
        self.endpoint = (endpoint or config.base_url).rstrip("/")
        self.client = OpenAI(
            api_key=config.resolved_api_key() or "EMPTY",
            base_url=self.endpoint,
            timeout=config.timeout_seconds,
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
        overrides = generate_kwargs or {}
        temperature = overrides.get("temperature", self.config.temperature)
        max_tokens = overrides.get("max_output_tokens", self.config.max_tokens)
        top_p = overrides.get("top_p", self.config.top_p)

        def call() -> tuple[str, dict[str, Any]]:
            request: dict[str, Any] = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": self._to_api_content(content)}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if top_p is not None:
                request["top_p"] = top_p
            resp = self.client.chat.completions.create(**request)
            if not resp.choices:
                raise RuntimeError("Empty completion response.")
            text = _extract_openai_text(resp.choices[0].message)
            if not text:
                raise RuntimeError("Completion response has no text content.")
            usage = getattr(resp, "usage", None)
            usage_dict = {}
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                }
            return text.strip(), usage_dict

        (text, usage_dict), attempt_count, retry_errors = run_with_retry(call, self.retry_policy)
        return BackendResult(
            text=text,
            latency_seconds=time.time() - started,
            endpoint_id=self.endpoint,
            attempt_count=attempt_count,
            retry_errors=retry_errors,
            usage=usage_dict,
        )

    def _to_api_content(self, content: list[ContentItem]) -> list[dict[str, Any]]:
        api_content: list[dict[str, Any]] = []
        for item in content:
            if item.type == "text":
                api_content.append({"type": "text", "text": item.text})
            elif item.type == "image" and item.image_path is not None:
                api_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(
                                item.image_path,
                                max_side=self.runtime.resolution,
                                quality=self.config.image_quality,
                            )
                        },
                    }
                )
        return api_content


def _extract_openai_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        if content.strip():
            return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
            elif getattr(item, "type", None) == "text":
                texts.append(str(getattr(item, "text", "")))
        text = "\n".join(texts).strip()
        if text:
            return text
    for attr in ("reasoning", "reasoning_content"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""
