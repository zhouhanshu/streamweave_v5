"""LLM-as-judge scoring for StreamWeave RL steps."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass, field, fields
from typing import Any

from backend.base import BaseBackend
from backend.factory import create_backend
from streamweave.config import BackendConfig, RuntimeConfig
from streamweave.schemas import ContentItem, FrameRef, ModelAction, QualityReport

from .judge_prompts import (
    GrppoAnswerJudgePromptContext,
    GrppoJudgePromptContext,
    LegacyJudgePromptContext,
    build_grppo_answer_judge_content,
    build_grppo_judge_content,
    build_legacy_judge_content,
)


JUDGE_PROMPT_VERSION = "streamweave_step_judge_v3"
JUDGE_SCORE_KEYS = ("keyframe_selection", "bridge_quality", "semantic_alignment", "state_factuality")
GRPPO_JUDGE_PROMPT_VERSION = "streamweave_grppo_judge_v1"
GRPPO_STEP_SCORE_KEYS = (
    "delta_groundedness",
    "anchor_keyframe",
    "semantic_alignment",
    "state_groundedness",
)
GRPPO_SCORE_KEYS = (*GRPPO_STEP_SCORE_KEYS, "answer_reward")
GRPPO_PROCESS_SCORE_SCALE = 2.0
GRPPO_CHECK_KEYS = {
    "delta_groundedness": (
        "delta_captures_action_progress",
        "delta_captures_state_or_location_changes",
        "delta_preserves_query_relevant_details",
        "delta_no_visual_hallucination",
        "delta_not_polluted_by_qa",
    ),
    "anchor_keyframe": (
        "anchor_count_valid",
        "anchor_time_and_body_valid",
        "anchor_representative_if_present",
        "first_window_rule_followed",
    ),
    "semantic_alignment": (
        "semantic_output_coherent",
        "semantic_text_anchor_express_current_frames",
        "semantic_no_cross_step_contradiction",
    ),
    "state_groundedness": (
        "state_uses_available_evidence",
        "state_identifies_question_scope",
        "state_decision_is_grounded",
        "state_no_visual_hallucination",
        "state_consistent_with_answer",
        "state_no_unreasonable_hallucination",
    ),
}


# Backends are expensive to instantiate (GeminiBackend builds a google-genai
# client; OpenAICompatibleBackend opens an HTTP session) and StepJudge is
# instantiated once per StreamWeaveRLEnv -> i.e. per rollout trajectory. Without
# a cache we end up paying that init cost batch_size * rollout.n times per
# training step. Cache by JudgeConfig identity so different judge configs (e.g.
# different model / base_url) stay isolated.
_JUDGE_BACKEND_CACHE: dict[tuple, BaseBackend] = {}
_JUDGE_BACKEND_LOCK = threading.Lock()


def _judge_backend_cache_key(config: JudgeConfig) -> tuple:
    return (
        str(config.backend),
        str(config.model),
        str(config.base_url),
        str(config.api_key),
        str(config.api_key_env),
        int(config.max_tokens),
        float(config.timeout_seconds),
        int(config.image_quality),
        int(config.max_image_side),
        int(config.max_retries),
        float(config.retry_backoff_seconds),
        float(config.retry_backoff_multiplier),
    )


def _build_backend(config: JudgeConfig) -> BaseBackend:
    backend_config = BackendConfig(
        backend=config.backend,
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        api_key_env=config.api_key_env,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        timeout_seconds=config.timeout_seconds,
        image_quality=config.image_quality,
        max_retries=config.max_retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
        retry_backoff_multiplier=config.retry_backoff_multiplier,
    )
    runtime = RuntimeConfig(resolution=config.max_image_side)
    return create_backend(backend_config, runtime)


def get_judge_backend(config: JudgeConfig) -> BaseBackend:
    """Return a process-local shared backend for the given judge config.

    Multiple StepJudge instances with the same judge config share one backend
    so we only pay client-init cost once per process (or once per distinct
    config). Thread-safe via _JUDGE_BACKEND_LOCK.
    """
    cache_key = _judge_backend_cache_key(config)
    with _JUDGE_BACKEND_LOCK:
        backend = _JUDGE_BACKEND_CACHE.get(cache_key)
        if backend is None:
            backend = _build_backend(config)
            _JUDGE_BACKEND_CACHE[cache_key] = backend
        return backend


def reset_judge_backend_cache() -> None:
    """Clear the cache. Intended for tests only."""
    with _JUDGE_BACKEND_LOCK:
        _JUDGE_BACKEND_CACHE.clear()


@dataclass(slots=True)
class JudgeConfig:
    enable: bool = False
    backend: str = "openai_compatible"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "STREAMWEAVE_JUDGE_API_KEY"
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.1
    timeout_seconds: float = 120.0
    image_quality: int = 80
    max_image_side: int = 512
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    retry_backoff_multiplier: float = 2.0
    failure_score: float = 0.0
    score_on_invalid: bool = False
    prompt_version: str = JUDGE_PROMPT_VERSION


@dataclass(slots=True)
class JudgeResult:
    score: float
    status: str = "disabled"
    raw_response: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    error: str = ""


class JudgeParseRetryError(RuntimeError):
    """Raised when the judge backend returns text that cannot be parsed."""

    def __init__(self, *, label: str, errors: list[str], raw_responses: list[str]) -> None:
        self.label = label
        self.errors = list(errors)
        self.raw_response = "\n".join(raw_responses)
        joined = "; ".join(errors[-3:])
        super().__init__(f"{label} judge JSON parse failed after {len(errors)} attempt(s): {joined}")


class StepJudge:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    async def score_step(
        self,
        *,
        memory_before: str,
        qa_history: str,
        frames: list[FrameRef],
        raw_action: ModelAction,
        raw_output: str,
        quality: QualityReport,
        query_label: dict[str, Any] | None = None,
        has_query: bool = False,
        has_answer: bool = False,
        answer_reward_event: bool = False,
        answer_correctness: float | None = None,
    ) -> JudgeResult:
        if not self.config.enable:
            return JudgeResult(score=0.0, status="disabled")
        if not self.config.score_on_invalid and not quality.parser_ok:
            return JudgeResult(score=float(self.config.failure_score), status="skipped_invalid")
        grppo_prompt = self.config.prompt_version == GRPPO_JUDGE_PROMPT_VERSION
        if self.config.backend.lower() == "mock":
            if grppo_prompt:
                return _mock_grppo_judge_result(
                    raw_action,
                    quality,
                    has_query=has_query,
                    has_answer=has_answer,
                    answer_reward_event=answer_reward_event,
                    answer_correctness=answer_correctness,
                )
            return _mock_judge_result(raw_action, quality)

        started = time.time()
        if grppo_prompt:
            return await self._score_grppo_step(
                started=started,
                memory_before=memory_before,
                qa_history=qa_history,
                frames=frames,
                raw_output=raw_output,
                quality=quality,
                query_label=query_label,
                answer_reward_event=answer_reward_event,
            )
        else:
            content = build_legacy_judge_content(
                LegacyJudgePromptContext(
                    memory_before=memory_before,
                    qa_history=qa_history,
                    frames=frames,
                    raw_action=raw_action,
                    raw_output=raw_output,
                    quality=quality,
                )
            )
        try:
            backend = self._ensure_backend()
            result = await asyncio.to_thread(
                backend.generate,
                content,
                generate_kwargs={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_output_tokens": self.config.max_tokens,
                    "response_mime_type": "application/json",
                },
            )
            parsed = _parse_grppo_judge_response(result.text) if grppo_prompt else _parse_judge_response(result.text)
            parsed.latency_seconds = time.time() - started
            parsed.raw_response = result.text
            return parsed
        except Exception as exc:
            return JudgeResult(
                score=float(self.config.failure_score),
                status="error",
                latency_seconds=time.time() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _ensure_backend(self):
        return get_judge_backend(self.config)

    async def _generate_and_parse_json(
        self,
        *,
        backend: BaseBackend,
        content: list[Any],
        parser,
        label: str,
    ) -> tuple[JudgeResult, str]:
        attempts = max(0, int(self.config.max_retries)) + 1
        errors: list[str] = []
        raw_responses: list[str] = []
        for attempt in range(1, attempts + 1):
            result = await asyncio.to_thread(
                backend.generate,
                content,
                generate_kwargs={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_output_tokens": self.config.max_tokens,
                    "response_mime_type": "application/json",
                },
            )
            raw = result.text
            raw_responses.append(f"=== {label}_attempt_{attempt} ===\n{raw}")
            try:
                parsed = parser(raw)
                if errors:
                    parsed.issues.append(f"{label} JSON parse succeeded after {attempt} attempts")
                return parsed, raw
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt >= attempts:
                    raise JudgeParseRetryError(label=label, errors=errors, raw_responses=raw_responses) from exc
                sleep_seconds = max(0.0, float(self.config.retry_backoff_seconds)) * (
                    max(1.0, float(self.config.retry_backoff_multiplier)) ** (attempt - 1)
                )
                if sleep_seconds:
                    await asyncio.sleep(sleep_seconds)
        raise RuntimeError("unreachable judge parse retry state")

    async def _score_grppo_step(
        self,
        *,
        started: float,
        memory_before: str,
        qa_history: str,
        frames: list[FrameRef],
        raw_output: str,
        quality: QualityReport,
        query_label: dict[str, Any] | None,
        answer_reward_event: bool,
    ) -> JudgeResult:
        try:
            backend = self._ensure_backend()
            process_content = build_grppo_judge_content(
                GrppoJudgePromptContext(
                    memory_before=memory_before,
                    qa_history=qa_history,
                    frames=frames,
                    raw_output=raw_output,
                    quality=quality,
                    query_label=None,
                    answer_reward_event=False,
                    answer_correctness=None,
                )
            )
            parsed, process_raw = await self._generate_and_parse_json(
                backend=backend,
                content=process_content,
                parser=_parse_grppo_judge_response,
                label="process_judge",
            )
            raw_parts = ["=== process_judge ===", process_raw]

            if answer_reward_event:
                answer_content = build_grppo_answer_judge_content(
                    GrppoAnswerJudgePromptContext(
                        raw_output=raw_output,
                        quality=quality,
                        query_label=query_label,
                    )
                )
                answer_parsed, answer_raw = await self._generate_and_parse_json(
                    backend=backend,
                    content=answer_content,
                    parser=_parse_grppo_answer_judge_response,
                    label="answer_judge",
                )
                parsed.scores["answer_reward"] = answer_parsed.scores.get("answer_reward", 0.0)
                parsed.reasons["answer_reward"] = answer_parsed.reasons.get("answer_reward", "")
                parsed.issues.extend(answer_parsed.issues)
                raw_parts.extend(["=== answer_judge ===", answer_raw])
            else:
                parsed.scores["answer_reward"] = 0.0
                parsed.reasons["answer_reward"] = ""

            parsed.latency_seconds = time.time() - started
            parsed.raw_response = "\n".join(raw_parts)
            return parsed
        except JudgeParseRetryError as exc:
            return JudgeResult(
                score=float(self.config.failure_score),
                status="error",
                raw_response=exc.raw_response,
                issues=list(exc.errors),
                latency_seconds=time.time() - started,
                error=str(exc),
            )
        except Exception as exc:
            return JudgeResult(
                score=float(self.config.failure_score),
                status="error",
                latency_seconds=time.time() - started,
                error=f"{type(exc).__name__}: {exc}",
            )


def judge_config_from_mapping(data: Any) -> JudgeConfig:
    mapping = dict(data or {})
    allowed = {item.name for item in fields(JudgeConfig)}
    return JudgeConfig(**{key: value for key, value in mapping.items() if key in allowed})


def _build_judge_content(
    *,
    memory_before: str,
    qa_history: str,
    frames: list[FrameRef],
    raw_action: ModelAction,
    raw_output: str,
    quality: QualityReport,
) -> list[ContentItem]:
    """Compatibility wrapper for tests and old internal imports."""
    return build_legacy_judge_content(
        LegacyJudgePromptContext(
            memory_before=memory_before,
            qa_history=qa_history,
            frames=frames,
            raw_action=raw_action,
            raw_output=raw_output,
            quality=quality,
        )
    )


def _build_grppo_judge_content(
    *,
    memory_before: str,
    qa_history: str,
    frames: list[FrameRef],
    raw_output: str,
    quality: QualityReport,
    query_label: dict[str, Any] | None,
    answer_reward_event: bool,
    answer_correctness: float | None,
) -> list[ContentItem]:
    """Compatibility wrapper for tests and old internal imports."""
    return build_grppo_judge_content(
        GrppoJudgePromptContext(
            memory_before=memory_before,
            qa_history=qa_history,
            frames=frames,
            raw_output=raw_output,
            quality=quality,
            query_label=query_label,
            answer_reward_event=answer_reward_event,
            answer_correctness=answer_correctness,
        )
    )


def _build_grppo_answer_judge_content(
    *,
    raw_output: str,
    quality: QualityReport,
    query_label: dict[str, Any] | None,
) -> list[ContentItem]:
    """Compatibility wrapper for tests and old internal imports."""
    return build_grppo_answer_judge_content(
        GrppoAnswerJudgePromptContext(
            raw_output=raw_output,
            quality=quality,
            query_label=query_label,
        )
    )


def _parse_judge_response(raw: str) -> JudgeResult:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge response must be a JSON object")

    scores = {key: _extract_score(parsed.get(key, 0.0)) for key in JUDGE_SCORE_KEYS}
    if "overall" in parsed:
        overall = _extract_score(parsed["overall"])
    else:
        overall = sum(scores.values()) / max(len(scores), 1)
    issues = _string_list(parsed.get("issues", []))
    caps = _string_list(parsed.get("caps_applied", []))
    if caps:
        issues.extend(f"cap: {item}" for item in caps)
    reasons = {key: _extract_reason(parsed.get(key, {})) for key in JUDGE_SCORE_KEYS}
    return JudgeResult(score=overall, status="ok", scores=scores, reasons=reasons, issues=issues)


def _parse_grppo_judge_response(raw: str) -> JudgeResult:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("GRPPO judge response must be a JSON object")

    has_process_checklists = _has_any_grppo_process_checklist(parsed)
    scores = {
        key: _extract_grppo_score_from_response(key, parsed.get(key, 0.0), has_process_checklists=has_process_checklists)
        for key in GRPPO_SCORE_KEYS
    }
    if not has_process_checklists and "anchor_keyframe" not in parsed and "note_keyframe" in parsed:
        scores["anchor_keyframe"] = _extract_score(parsed.get("note_keyframe", 0.0))
    overall = _extract_flat_grppo_process_score(parsed)
    if overall is None:
        overall = _clamp_process_total_score(
            GRPPO_PROCESS_SCORE_SCALE
            * sum(scores[key] for key in GRPPO_STEP_SCORE_KEYS)
            / max(len(GRPPO_STEP_SCORE_KEYS), 1)
        )
    issues = _string_list(parsed.get("issues", []))
    caps = _string_list(parsed.get("caps_applied", []))
    if caps:
        issues.extend(f"cap: {item}" for item in caps)
    reasons = {key: _extract_reason(parsed.get(key, {})) for key in GRPPO_SCORE_KEYS}
    if "anchor_keyframe" not in parsed and "note_keyframe" in parsed:
        reasons["anchor_keyframe"] = _extract_reason(parsed.get("note_keyframe", {}))
    return JudgeResult(score=overall, status="ok", scores=scores, reasons=reasons, issues=issues)


def _parse_grppo_answer_judge_response(raw: str) -> JudgeResult:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("GRPPO answer judge response must be a JSON object")
    answer_value = parsed.get("answer_reward", 0.0)
    score = _extract_score(answer_value)
    reason = _extract_reason(answer_value)
    issues = _string_list(parsed.get("issues", []))
    return JudgeResult(
        score=score,
        status="ok",
        scores={"answer_reward": score},
        reasons={"answer_reward": reason},
        issues=issues,
    )


def _mock_judge_result(action: ModelAction, quality: QualityReport) -> JudgeResult:
    num_notes = sum(1 for event in action.events if event.kind == "note")
    num_bridges = sum(1 for event in action.events if event.kind == "bridge")
    keyframe = 1.0 if num_notes == 1 else 0.5 if num_notes == 0 else 0.0
    bridge = 1.0 if num_bridges >= 1 and all(event.text.strip() for event in action.events if event.kind == "bridge") else 0.0
    state = 1.0 if action.state.strip() else 0.0
    semantic = 1.0 if quality.parser_ok else 0.0
    scores = {
        "keyframe_selection": keyframe,
        "bridge_quality": bridge,
        "semantic_alignment": semantic,
        "state_factuality": state,
    }
    return JudgeResult(
        score=sum(scores.values()) / len(scores),
        status="mock",
        scores=scores,
        reasons={
            "keyframe_selection": "Mock score from anchor count.",
            "bridge_quality": "Mock score from delta presence and non-empty text.",
            "semantic_alignment": "Mock score from parser validity.",
            "state_factuality": "Mock score from state presence.",
        },
    )


def _mock_grppo_judge_result(
    action: ModelAction,
    quality: QualityReport,
    *,
    has_query: bool,
    has_answer: bool,
    answer_reward_event: bool,
    answer_correctness: float | None,
) -> JudgeResult:
    num_notes = sum(1 for event in action.events if event.kind == "note")
    num_bridges = sum(1 for event in action.events if event.kind == "bridge")
    delta = 1.0 if num_bridges >= 1 and all(event.text.strip() for event in action.events if event.kind == "bridge") else 0.0
    anchor = 1.0 if num_notes == 1 else 0.5 if num_notes == 0 else 0.0
    semantic = 1.0 if quality.parser_ok else 0.0
    state = 1.0 if action.state.strip() else 0.0
    answer_reward = _clamp_score(answer_correctness) if answer_correctness is not None else 0.0
    if not (answer_reward_event or has_query or has_answer):
        answer_reward = 0.0
    scores = {
        "delta_groundedness": delta,
        "anchor_keyframe": anchor,
        "semantic_alignment": semantic,
        "state_groundedness": state,
        "answer_reward": answer_reward,
    }
    return JudgeResult(
        score=_clamp_process_total_score(
            GRPPO_PROCESS_SCORE_SCALE
            * sum(scores[key] for key in GRPPO_STEP_SCORE_KEYS)
            / max(len(GRPPO_STEP_SCORE_KEYS), 1)
        ),
        status="mock",
        scores=scores,
        reasons={
            "delta_groundedness": "Mock score from delta presence and non-empty text.",
            "anchor_keyframe": "Mock score from anchor count.",
            "semantic_alignment": "Mock score from parser validity.",
            "state_groundedness": "Mock score from state presence.",
            "answer_reward": "Mock score from rule answer correctness.",
        },
    )


def _extract_score(value: Any) -> float:
    if isinstance(value, dict):
        for key in ("score", "value", "rating"):
            if key in value:
                return _clamp_score(value[key])
        return 0.0
    return _clamp_score(value)


def _extract_grppo_score(key: str, value: Any) -> float:
    if key in GRPPO_STEP_SCORE_KEYS:
        checklist_score = _extract_checklist_score(key, value)
        if checklist_score is not None:
            return checklist_score
    return _extract_score(value)


def _extract_grppo_score_from_response(key: str, value: Any, *, has_process_checklists: bool) -> float:
    if key not in GRPPO_STEP_SCORE_KEYS or not has_process_checklists:
        return _extract_grppo_score(key, value)
    checklist_score = _extract_checklist_score(key, value)
    return 0.0 if checklist_score is None else checklist_score


def _extract_checklist_score(key: str, value: Any) -> float | None:
    if not isinstance(value, dict) or not isinstance(value.get("checks"), dict):
        return None
    checks = value["checks"]
    expected_keys = GRPPO_CHECK_KEYS.get(key, ())
    if expected_keys:
        raw_scores = [
            0.0 if check_key not in checks else _extract_check_value(checks[check_key])
            for check_key in expected_keys
        ]
    else:
        raw_scores = [_extract_check_value(item) for item in checks.values()]
    scores = [check_score for check_score in raw_scores if check_score is not None]
    if not scores:
        return None
    return _clamp_score(sum(scores) / len(scores))


def _extract_flat_grppo_process_score(parsed: dict[str, Any]) -> float | None:
    raw_scores: list[float | None] = []
    if not _has_any_grppo_process_checklist(parsed):
        return None
    for key in GRPPO_STEP_SCORE_KEYS:
        value = parsed.get(key)
        expected_keys = GRPPO_CHECK_KEYS.get(key, ())
        if not isinstance(value, dict) or not isinstance(value.get("checks"), dict):
            raw_scores.extend(0.0 for _ in expected_keys)
            continue
        checks = value["checks"]
        if expected_keys:
            raw_scores.extend(
                0.0 if check_key not in checks else _extract_check_value(checks[check_key])
                for check_key in expected_keys
            )
        else:
            raw_scores.extend(_extract_check_value(item) for item in checks.values())
    scores = [score for score in raw_scores if score is not None]
    if not scores:
        return None
    return _clamp_process_total_score(GRPPO_PROCESS_SCORE_SCALE * sum(scores) / len(scores))


def _has_any_grppo_process_checklist(parsed: dict[str, Any]) -> bool:
    return any(
        isinstance(parsed.get(key), dict) and isinstance(parsed.get(key, {}).get("checks"), dict)
        for key in GRPPO_STEP_SCORE_KEYS
    )


def _extract_check_value(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("pass", "passed", "score", "value"):
            if key in value:
                return _extract_check_value(value[key])
        return 0.0
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return _clamp_score(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "null", "none", "na", "n/a", "not_applicable", "not applicable"}:
            return 0.0
        if normalized in {"1", "true", "yes", "y", "pass", "passed"}:
            return 1.0
        if normalized in {"0", "false", "no", "n", "fail", "failed"}:
            return 0.0
        try:
            return _clamp_score(float(normalized))
        except ValueError:
            return 0.0
    return 0.0


def _extract_reason(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("reason", "") or "").strip()
    return ""


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(score, 0.0), 1.0)


def _clamp_process_total_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(score, 0.0), GRPPO_PROCESS_SCORE_SCALE)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
