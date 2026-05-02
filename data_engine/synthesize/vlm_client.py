"""VLM clients used by data synthesis modules.

The synthesis pipeline uses a small prompt+images interface while allowing
different providers per stage. OpenAI-compatible backends cover local vLLM
services such as Qwen3-VL; Gemini uses the same VertexAI client as the main
StreamWeave rollout code.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.image_utils import image_to_data_url

DEFAULT_QWEN_API_KEY = "EMPTY"
DEFAULT_QWEN_BASE_URL = "http://127.0.0.1:8082/v1"
DEFAULT_QWEN_MODEL = "Qwen3-VL-32B-Instruct"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
DEFAULT_GEMINI_CREDENTIALS = "/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json"
PRO_SAFE_THINKING_BUDGET = 128
OPENAI_COMPATIBLE_BACKENDS = {"openai", "openai_compatible", "vllm", "qwen", "qwen3vl"}


def call_vlm(
    messages: list[dict[str, Any]],
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int = 2048,
    *,
    temperature: float = 0.0,
    timeout_seconds: float = 120.0,
) -> str:
    """Call an OpenAI-compatible chat-completions VLM endpoint."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or "EMPTY",
        base_url=base_url.rstrip("/") if base_url else None,
        timeout=timeout_seconds,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not response.choices:
        raise RuntimeError("Empty completion response.")
    text = extract_text(response.choices[0].message)
    if not text:
        raise RuntimeError("Completion response has no text content.")
    return text.strip()


@dataclass(slots=True)
class VLMClient:
    backend: str
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    max_image_side: int = 768
    image_quality: int = 85
    google_application_credentials: str = ""

    @classmethod
    def from_backend(cls, backend: str, **kwargs: Any) -> "VLMClient":
        name = backend.lower()
        if name == "qwen3vl":
            return cls.qwen3vl32b(**kwargs)
        if name in OPENAI_COMPATIBLE_BACKENDS:
            return cls.openai_compatible(backend=name, **kwargs)
        if name == "gemini":
            return cls.gemini(**kwargs)
        raise ValueError(f"Unknown synthesis VLM backend: {backend}")

    @classmethod
    def openai_compatible(
        cls,
        *,
        backend: str = "openai_compatible",
        api_key: str = DEFAULT_QWEN_API_KEY,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        model: str = DEFAULT_QWEN_MODEL,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        max_image_side: int = 768,
        image_quality: int = 85,
    ) -> "VLMClient":
        api_key = os.environ.get("SYNTHESIS_OPENAI_API_KEY", api_key)
        base_url = os.environ.get("SYNTHESIS_OPENAI_BASE_URL", base_url)
        model = os.environ.get("SYNTHESIS_OPENAI_MODEL", model)
        return cls(
            backend=backend,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_image_side=max_image_side,
            image_quality=image_quality,
        )

    @classmethod
    def qwen3vl32b(
        cls,
        *,
        api_key: str = DEFAULT_QWEN_API_KEY,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        model: str = DEFAULT_QWEN_MODEL,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        max_image_side: int = 768,
        image_quality: int = 85,
    ) -> "VLMClient":
        api_key = os.environ.get("QWEN3VL_API_KEY", api_key)
        base_url = os.environ.get("QWEN3VL_BASE_URL", base_url)
        model = os.environ.get("QWEN3VL_MODEL", model)
        return cls.openai_compatible(
            backend="qwen3vl",
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_image_side=max_image_side,
            image_quality=image_quality,
        )

    @classmethod
    def gemini(
        cls,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        max_image_side: int = 768,
        image_quality: int = 85,
        google_application_credentials: str = DEFAULT_GEMINI_CREDENTIALS,
    ) -> "VLMClient":
        model = os.environ.get("SYNTHESIS_GEMINI_MODEL", os.environ.get("GEMINI_MODEL", model))
        credentials = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            os.environ.get("SYNTHESIS_GEMINI_CREDENTIALS", google_application_credentials),
        )
        return cls(
            backend="gemini",
            api_key="",
            base_url="",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_image_side=max_image_side,
            image_quality=image_quality,
            google_application_credentials=credentials,
        )

    def call(self, messages: list[dict[str, Any]], *, max_tokens: int | None = None) -> str:
        if self.backend.lower() == "gemini":
            return call_gemini(
                messages,
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                timeout_seconds=self.timeout_seconds,
                max_image_side=self.max_image_side,
                image_quality=self.image_quality,
                google_application_credentials=self.google_application_credentials,
            )
        return call_vlm(
            messages,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
        )

    def user_message(self, prompt: str, image_paths: list[Path] | None = None) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths or []:
            content.append(image_content(path, max_side=self.max_image_side, quality=self.image_quality))
        return {"role": "user", "content": content}


def call_gemini(
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
    max_image_side: int,
    image_quality: int,
    google_application_credentials: str,
) -> str:
    """Call Gemini through the same VertexAI google-genai stack as V4."""
    if google_application_credentials:
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", google_application_credentials)

    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GEMINI_VERTEX_PROJECT", "mmu-jichu-shangyehua-gemini"),
        location=os.environ.get("GEMINI_VERTEX_LOCATION", "global"),
        http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
    )
    generate_config: dict[str, Any] = {
        "temperature": temperature,
        "top_p": 0.1,
        "max_output_tokens": max_tokens,
    }
    thinking_config = _thinking_config(model)
    if thinking_config is not None:
        generate_config["thinking_config"] = thinking_config
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=messages_to_gemini_parts(types, messages))],
        config=types.GenerateContentConfig(**generate_config),
    )
    text = _extract_gemini_text(response)
    if not text:
        raise RuntimeError(f"Gemini response has no text content. {_describe_response(response)}")
    return text


def image_content(path: str | Path, *, max_side: int = 768, quality: int = 85) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": image_to_data_url(path, max_side=max_side, quality=quality),
        },
    }


def messages_to_gemini_parts(types: Any, messages: list[dict[str, Any]]) -> list[Any]:
    parts: list[Any] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            if content:
                parts.append(types.Part.from_text(text=content))
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = str(item.get("text", ""))
                if text:
                    parts.append(types.Part.from_text(text=text))
            elif item.get("type") == "image_url":
                parts.append(types.Part.from_bytes(data=image_url_to_bytes(item.get("image_url", {})), mime_type="image/jpeg"))
    if not parts:
        raise ValueError("Gemini content is empty.")
    return parts


def image_url_to_bytes(image_url: object) -> bytes:
    if not isinstance(image_url, dict):
        raise ValueError("image_url content must be a dictionary.")
    url = str(image_url.get("url", ""))
    prefix = "data:image/jpeg;base64,"
    if not url.startswith(prefix):
        raise ValueError("Only JPEG data URLs are supported for Gemini synthesis calls.")
    return base64.b64decode(url[len(prefix) :])


def _extract_gemini_text(response: Any) -> str:
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


def _thinking_config(model: str) -> dict[str, int] | None:
    budget = resolve_thinking_budget(model)
    if budget is None:
        return None
    return {"thinking_budget": budget}


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
                "safety_ratings": _safe_repr(getattr(candidate, "safety_ratings", None), limit=800),
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


def extract_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
            elif getattr(item, "type", None) == "text":
                pieces.append(str(getattr(item, "text", "")))
        return "\n".join(piece for piece in pieces if piece)
    for attr in ("reasoning", "reasoning_content"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""
