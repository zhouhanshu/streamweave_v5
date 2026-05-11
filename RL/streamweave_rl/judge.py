"""LLM-as-judge scoring for StreamWeave RL steps."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field, fields
from typing import Any

from backend.factory import create_backend
from streamweave.config import BackendConfig, RuntimeConfig
from streamweave.schemas import ContentItem, FrameRef, ModelAction, QualityReport


JUDGE_PROMPT_VERSION = "streamweave_step_judge_v3"
JUDGE_SCORE_KEYS = ("keyframe_selection", "bridge_quality", "semantic_alignment", "state_factuality")


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


class StepJudge:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self._backend = None

    async def score_step(
        self,
        *,
        memory_before: str,
        qa_history: str,
        frames: list[FrameRef],
        raw_action: ModelAction,
        raw_output: str,
        quality: QualityReport,
    ) -> JudgeResult:
        if not self.config.enable:
            return JudgeResult(score=0.0, status="disabled")
        if not self.config.score_on_invalid and not quality.parser_ok:
            return JudgeResult(score=float(self.config.failure_score), status="skipped_invalid")
        if self.config.backend.lower() == "mock":
            return _mock_judge_result(raw_action, quality)

        started = time.time()
        content = _build_judge_content(
            memory_before=memory_before,
            qa_history=qa_history,
            frames=frames,
            raw_action=raw_action,
            raw_output=raw_output,
            quality=quality,
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
            parsed = _parse_judge_response(result.text)
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
        if self._backend is None:
            backend_config = BackendConfig(
                backend=self.config.backend,
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                api_key_env=self.config.api_key_env,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                timeout_seconds=self.config.timeout_seconds,
                image_quality=self.config.image_quality,
                max_retries=self.config.max_retries,
                retry_backoff_seconds=self.config.retry_backoff_seconds,
                retry_backoff_multiplier=self.config.retry_backoff_multiplier,
            )
            runtime = RuntimeConfig(resolution=self.config.max_image_side)
            self._backend = create_backend(backend_config, runtime)
        return self._backend


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
    if frames:
        frame_window = f't="{frames[0].start_time:.1f}-{frames[-1].end_time:.1f}"'
    else:
        frame_window = "<none>"
    frame_lines = "\n".join(
        f'- frame_{idx}: t="{frame.start_time:.1f}-{frame.end_time:.1f}", global_index={frame.global_index}'
        for idx, frame in enumerate(frames, start=1)
    )
    action_summary = _action_summary(raw_action)
    issue_codes = ", ".join(issue.code for issue in quality.issues) or "<none>"
    prompt = f"""\
You are a strict process-reward judge for one step of a streaming video memory agent.

Your goal is to evaluate whether this step produces a useful, grounded memory update
for future video question answering. Judge the process quality, not the final task score.

The agent output protocol is strict:
- The output must start with exactly one <state>, then exactly one <answer>.
- Only <anchor> and <delta> observation tags may appear after <answer>.
- <state>: transient reasoning about the current evidence and QA decision. It is not saved to memory.
- <answer>: current QA answer, possibly empty.
- <anchor t="..."></anchor>: selected visual anchor frame interval. The anchor saves the frame image, not text.
- <delta t="...">...</delta>: concise text describing observable changes between visual anchors.

Use only the provided memory text, QA history, current frames, and agent output.
Do not use outside knowledge or hidden ground truth.
Do not reward answer correctness except for whether <state>/<answer> are grounded in visible or memory evidence.
Do not infer hidden intent, identity, fine-grained counts, causes, or object attributes unless clearly visible or stated in memory.

=== Memory Before This Step ===
{memory_before or "<empty/>"}

=== Current Frame Window ===
{frame_window}

=== Current Frame Index ===
{frame_lines or "<none>"}

=== QA History ===
{qa_history or "<empty/>"}

=== Parsed Agent Output Summary ===
{action_summary}

=== Raw Agent Output ===
{raw_output}

=== Existing Parser/Timing Issues ===
{issue_codes}

Scoring dimensions, each from 0.0 to 1.0:

1. keyframe_selection
Evaluate whether <anchor> preserves important visual anchor frames in the current window.
Important anchors include query-relevant evidence, major object/person/state changes,
scene/camera/workspace changes, event boundaries, or visual details that a text delta
cannot reliably preserve.
- 1.0: Selects the right important anchor(s), avoids duplicates, and does not save unimportant frames.
- 0.7: Relevant anchor choice, but misses a moderately useful anchor or selects a suboptimal frame.
- 0.4: Weakly relevant, redundant, poorly timed, or only loosely useful as visual memory.
- 0.0: Misses a clearly important anchor, selects irrelevant frames, or emits duplicate/unhelpful anchors.
If no current frame contains meaningful new visual evidence, no anchor can be acceptable; do not punish absence of an anchor solely for frequency, because a separate rule handles anchor frequency.
Special first-window rule: if the current frame window spans exactly t="0.0-5.0" (or "0-5"),
keyframe_selection may be 1.0 only when the output contains exactly one anchor with t="0.0-1.0" (or "0-1").
For that first window, keyframe_selection must be 0.0 if the anchor is absent, duplicated, or uses any timing other than exactly t="0.0-1.0" (or "0-1").

2. bridge_quality
Evaluate whether <delta> uses concise, faithful language to describe frame-to-frame changes.
It should summarize observable action/state/layout changes over its time span, not invent details.
- 1.0: Concise, faithful, covers the main observable changes over the interval, and preserves useful state for future QA.
- 0.7: Mostly faithful, but slightly vague, slightly long, or misses minor changes.
- 0.4: Generic, overlong, under-informative, partially off-interval, or misses important changes.
- 0.0: Hallucinates, contradicts frames/memory, describes the wrong interval, or is unusable.

3. semantic_alignment
Evaluate whether the anchor/delta sequence is semantically aligned with the current frame order and visual content.
The visual anchors and text should form a coherent time-ordered memory update.
- 1.0: Anchor times, delta intervals, frame order, and described content all align.
- 0.7: Minor omissions or mild ambiguity, but no material contradiction.
- 0.4: Partial mismatch between text and images, weak temporal ordering, or unclear grounding.
- 0.0: Clear contradiction, wrong event order, anchor/delta mismatch, or text not grounded in the images.

4. state_factuality
Evaluate whether <state> is correct, useful, and hallucination-free.
It should reason from memory/current frames, state uncertainty when evidence is insufficient,
and make a sound decision about whether <answer> should be empty or non-empty.
- 1.0: Grounded, useful for QA/memory decisions, no hallucination, and handles uncertainty correctly.
- 0.7: Mostly grounded but generic or missing some useful decision context.
- 0.4: Weak, speculative, not useful for QA decisions, or mildly inconsistent with evidence.
- 0.0: Hallucinated, contradicts memory/current frames, claims certainty without evidence, or gives an unsound QA decision.

Hard caps:
- If parser/timing issues make a component unusable, cap that component at 0.4.
- If the output has no usable anchor/delta content, cap overall at 0.2.
- If the output clearly contradicts current frames or memory, cap overall at 0.5.
- If the output fabricates important visual facts, cap overall at 0.3.
- If <state> or <delta> relies on hidden future frames or outside knowledge, cap overall at 0.3.

Overall score:
Use the average of the four dimension scores after applying any hard caps. Keep scores calibrated;
do not give high scores for fluent but ungrounded text.

Return JSON only:
{{
  "keyframe_selection": {{"score": 0.0, "reason": "short reason"}},
  "bridge_quality": {{"score": 0.0, "reason": "short reason"}},
  "semantic_alignment": {{"score": 0.0, "reason": "short reason"}},
  "state_factuality": {{"score": 0.0, "reason": "short reason"}},
  "caps_applied": ["short cap strings, or empty list"],
  "overall": 0.0,
  "issues": ["short issue strings"]
}}
"""
    content: list[ContentItem] = [ContentItem("text", text=prompt)]
    for idx, frame in enumerate(frames, start=1):
        content.append(ContentItem("text", text=f'\n[frame_{idx} t="{frame.start_time:.1f}-{frame.end_time:.1f}"]\n'))
        content.append(ContentItem("image", image_path=frame.image_path))
    return content


def _action_summary(action: ModelAction) -> str:
    lines = [
        f"state: {action.state or '<empty>'}",
        f"answer: {action.answer or '<empty>'}",
    ]
    for event in action.events:
        if event.kind == "note":
            lines.append(f'anchor t="{event.start_time:.1f}-{event.end_time:.1f}"')
        else:
            lines.append(f'delta t="{event.start_time:.1f}-{event.end_time:.1f}": {event.text}')
    return "\n".join(lines)


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


def _extract_score(value: Any) -> float:
    if isinstance(value, dict):
        for key in ("score", "value", "rating"):
            if key in value:
                return _clamp_score(value[key])
        return 0.0
    return _clamp_score(value)


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


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
