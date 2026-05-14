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
            content = _build_grppo_judge_content(
                memory_before=memory_before,
                qa_history=qa_history,
                frames=frames,
                raw_output=raw_output,
                quality=quality,
                query_label=query_label,
                answer_reward_event=answer_reward_event,
                answer_correctness=answer_correctness,
            )
        else:
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
    if frames:
        frame_window = f't="{frames[0].start_time:.1f}-{frames[-1].end_time:.1f}"'
    else:
        frame_window = "<none>"
    frame_lines = "\n".join(
        f'- frame_{idx}: t="{frame.start_time:.1f}-{frame.end_time:.1f}", global_index={frame.global_index}'
        for idx, frame in enumerate(frames, start=1)
    )
    issue_codes = ", ".join(issue.code for issue in quality.issues) or "<none>"
    correctness_text = "null" if answer_correctness is None else f"{_clamp_score(answer_correctness):.4f}"
    answer_label = _grppo_answer_label_section(query_label, answer_reward_event=answer_reward_event)
    answer_score_block = (
        f"""
=== Answer Label ===
{answer_label}

=== Rule Correctness Reference ===
rule_answer_correctness={correctness_text}
This rule score is a reference only. The final answer_reward must be judged from the answer label,
the model's <state>, and the model's actual <answer>.
"""
        if answer_reward_event
        else ""
    )
    output_schema = (
        """Return JSON only. Do not output any other text:
{
  "delta_groundedness": {"score": 0.0, "reason": "brief reason"},
  "anchor_keyframe": {"score": 0.0, "reason": "brief reason"},
  "semantic_alignment": {"score": 0.0, "reason": "brief reason"},
  "state_groundedness": {"score": 0.0, "reason": "brief reason"},
  "answer_reward": {"score": 0.0, "reason": "brief reason"},
  "caps_applied": ["brief cap notes, or empty list"],
  "overall": 0.0,
  "issues": ["brief issue notes"]
}"""
        if answer_reward_event
        else """Return JSON only. Do not output any other text:
{
  "delta_groundedness": {"score": 0.0, "reason": "brief reason"},
  "anchor_keyframe": {"score": 0.0, "reason": "brief reason"},
  "semantic_alignment": {"score": 0.0, "reason": "brief reason"},
  "state_groundedness": {"score": 0.0, "reason": "brief reason"},
  "caps_applied": ["brief cap notes, or empty list"],
  "overall": 0.0,
  "issues": ["brief issue notes"]
}"""
    )
    score_intro = "Return 5 scores in [0.0, 1.0]:" if answer_reward_event else "Return 4 scores in [0.0, 1.0]:"
    answer_dimension = (
        """
5. answer_reward
Binary answer reward. It must be exactly 0.0 or 1.0.
If the answer label says the model should not answer now, <answer> must be empty.
If the answer label requires an answer now, the model must answer and the answer must match the reference answer.
Give 0.0 if the model answers when it should stay silent, stays silent when it must answer,
answers incorrectly, or contradicts its own <state> reasoning.
"""
        if answer_reward_event
        else ""
    )
    answer_caps = "- answer_reward must be exactly 0.0 or 1.0.\n" if answer_reward_event else ""
    overall_rule = (
        "overall is the average of the four process dimensions. Do not include answer_reward in overall."
        if answer_reward_event
        else "overall is the average of the four process dimensions."
    )
    prompt_prefix = f"""\
You are a judge evaluating the current-step output quality of a streaming video understanding agent.

The agent processes video frame windows in temporal order and maintains an interleaved visual-text Memory.
In Memory, <anchor> stores important video frames, and <delta> describes the video changes between anchors
or from the last anchor to the end of the current window.
At every step, the agent also reads QA History and writes an <answer> only when the current evidence is
sufficient and the question should be answered now.

Judge only the model output for the current step. You may use the Memory before this step, the current
video frames, the QA History visible to the model, the current-step answer label when present, and the
model output. Do not use outside knowledge. Do not assume future frames. Do not treat question text,
answer options, or QA History as observed video facts.

## Output Protocol

- <state>: the model's reasoning about the current state. It is not written into long-term Memory,
  but it should reflect how the model uses Memory, current frames, and QA History to make its judgment.
- <answer>: the model's answer to the current question in QA History. If the model should not answer now,
  this field must be empty.
- <anchor t="..."></anchor>: a visual memory anchor used to preserve a key, representative, or hard-to-describe
  frame from the current window. The anchor tag body must be empty.
- <delta t="...">...</delta>: text memory for video changes. It describes what happens from the last anchor
  in previous Memory to the next anchor, or to the end of the current window. If previous Memory ends with
  a delta, the current delta should merge and rewrite that old delta rather than simply repeat it or append
  a disconnected continuation.

## Inputs

=== Memory Before This Step ===
{memory_before or "<empty/>"}

=== Current Video Frames ===
Time window: {frame_window}
Frame index:
{frame_lines or "<none>"}
"""

    prompt_suffix = f"""\

=== QA History Visible To The Model ===
{qa_history or "<empty/>"}

{answer_score_block}

=== Model Output ===
{raw_output}

=== Existing Parser Or Timing Issues ===
{issue_codes}

## Main Penalties

- Treating question text, answer options, or QA History as observed video facts.
- Claiming objects, identities, counts, spatial relations, actions, causes, or intentions that are not
  supported by the current frames or Memory.
- Writing <state> or <delta> that contradicts current frames, existing Memory, or the correct temporal order.
- Forcing an answer when visual or memory evidence is insufficient.
- Letting QA History contaminate Memory, state, delta, or answer with hallucinated content.

## Scoring Dimensions

{score_intro}

1. delta_groundedness
Evaluate whether <delta> accurately and concisely describes video changes, preserves key observable
information, and correctly handles the time span from the previous anchor to the next anchor or to the
end of the current window. If previous Memory ends with a delta, also judge whether the current delta
properly merges and rewrites that old delta. Penalize hallucination caused by answer text, question text,
answer options, or QA History.

2. anchor_keyframe
Evaluate whether <anchor> selects a truly useful keyframe from the current evidence. At most one anchor
should be selected in one step. It is acceptable to select no anchor when the current window has no useful
new visual evidence. Penalize missing an anchor when the current change is important, when the delta span
is long, or when text alone cannot reliably preserve the evidence.

3. semantic_alignment
Evaluate whether the resulting visual-text memory is semantically aligned with the current frames,
keeps the correct temporal order, and remains coherent with existing Memory.

4. state_groundedness
Evaluate whether <state> makes reliable reasoning from Memory, current frames, and QA History without
hallucination, unsupported inference, or QA-history contamination. If <state> reasons about whether to
answer, judge whether that reasoning is grounded and whether the actual <answer> is consistent with it.
{answer_dimension}

## Hard Caps

- If parser or timing issues make a component unusable, cap that component at 0.4.
- If state or delta fabricates important visual facts, cap that component at 0.3.
- If state or delta copies query/options into memory as observed video facts, cap that component at 0.3.
{answer_caps}

## overall

{overall_rule}

{output_schema}
"""
    content: list[ContentItem] = [ContentItem("text", text=prompt_prefix)]
    for idx, frame in enumerate(frames, start=1):
        content.append(ContentItem("text", text=f'\n[frame_{idx} t="{frame.start_time:.1f}-{frame.end_time:.1f}"]\n'))
        content.append(ContentItem("image", image_path=frame.image_path))
    content.append(ContentItem("text", text=prompt_suffix))
    return content


def _grppo_answer_label_section(label: dict[str, Any] | None, *, answer_reward_event: bool) -> str:
    if not answer_reward_event:
        return ""
    if label is None:
        return "No query or answer_target is annotated for this step. The model should not answer; <answer> should be empty."

    event_type = str(label.get("event_type") or "")
    question = str(label.get("question") or label.get("content") or label.get("query") or "").strip()
    timestamp = _label_time_text(label)
    target = _label_answer_text(label)
    should_answer = label.get("should_answer")
    should_stay_silent = should_answer is not None and not _truthy_label_value(should_answer)

    lines: list[str] = []
    if event_type == "query":
        lines.append("Event type: user query.")
    elif event_type == "answer_target":
        lines.append("Event type: answer checkpoint.")
    elif event_type:
        lines.append(f"Event type: {event_type}.")
    if question:
        lines.append(f"Current question: {question}")
    if timestamp:
        lines.append(timestamp)
    if should_stay_silent:
        lines.append("Requirement: the model should not answer now; <answer> should be empty.")
    elif target:
        if event_type == "answer_target":
            lines.append(f"Requirement: the model must answer now. Reference answer: {target}")
        else:
            lines.append(f"Requirement: the current question is answerable now. Reference answer: {target}")
    else:
        lines.append("Requirement: the model should not answer now; <answer> should be empty.")
    return "\n".join(lines)


def _label_time_text(label: dict[str, Any]) -> str:
    time_value = label.get("timestamp", label.get("time"))
    if time_value is None:
        return ""
    try:
        return f"Annotation time: {float(time_value):.1f}s"
    except (TypeError, ValueError):
        return f"Annotation time: {time_value}"


def _label_answer_text(label: dict[str, Any]) -> str:
    for key in ("ground_truth", "gt", "target", "target_answer"):
        value = label.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if label.get("event_type") == "answer_target":
        for key in ("answer", "content"):
            value = label.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    value = label.get("answer")
    if value is not None and str(value).strip():
        return str(value).strip()
    return ""


def _truthy_label_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


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

    scores = {key: _extract_score(parsed.get(key, 0.0)) for key in GRPPO_SCORE_KEYS}
    if "anchor_keyframe" in scores and scores["anchor_keyframe"] == 0.0 and "note_keyframe" in parsed:
        scores["anchor_keyframe"] = _extract_score(parsed.get("note_keyframe", 0.0))
    if "overall" in parsed:
        overall = _extract_score(parsed["overall"])
    else:
        overall = sum(scores[key] for key in GRPPO_STEP_SCORE_KEYS) / max(len(GRPPO_STEP_SCORE_KEYS), 1)
    issues = _string_list(parsed.get("issues", []))
    caps = _string_list(parsed.get("caps_applied", []))
    if caps:
        issues.extend(f"cap: {item}" for item in caps)
    reasons = {key: _extract_reason(parsed.get(key, {})) for key in GRPPO_SCORE_KEYS}
    if "anchor_keyframe" in reasons and not reasons["anchor_keyframe"] and "note_keyframe" in parsed:
        reasons["anchor_keyframe"] = _extract_reason(parsed.get("note_keyframe", {}))
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
        score=sum(scores[key] for key in GRPPO_STEP_SCORE_KEYS) / len(GRPPO_STEP_SCORE_KEYS),
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
