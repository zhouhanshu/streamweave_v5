"""Prompt rendering for StreamWeave LLM judges.

This module is intentionally text-heavy. Keep prompt wording and label rendering
here so ``judge.py`` stays focused on backend calls and response parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streamweave.schemas import ContentItem, FrameRef, ModelAction, QualityReport


@dataclass(slots=True)
class LegacyJudgePromptContext:
    memory_before: str
    qa_history: str
    frames: list[FrameRef]
    raw_action: ModelAction
    raw_output: str
    quality: QualityReport


@dataclass(slots=True)
class GrppoJudgePromptContext:
    memory_before: str
    qa_history: str
    frames: list[FrameRef]
    raw_output: str
    quality: QualityReport
    query_label: dict[str, Any] | None
    answer_reward_event: bool
    answer_correctness: float | None


LEGACY_PROCESS_RUBRIC = """\
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
"""


LEGACY_OUTPUT_SCHEMA = """\
Return JSON only:
{
  "keyframe_selection": {"score": 0.0, "reason": "short reason"},
  "bridge_quality": {"score": 0.0, "reason": "short reason"},
  "semantic_alignment": {"score": 0.0, "reason": "short reason"},
  "state_factuality": {"score": 0.0, "reason": "short reason"},
  "caps_applied": ["short cap strings, or empty list"],
  "overall": 0.0,
  "issues": ["short issue strings"]
}
"""


GRPPO_PROCESS_RUBRIC = """\
1. delta_groundedness
Evaluate whether <delta> is a concise, faithful, and useful text memory of observable video changes.
It should describe the correct time span from the previous anchor to the next anchor, or to the end of
the current window. If previous Memory ends with a delta, the current delta should merge and rewrite
that old delta instead of merely repeating it or appending a disconnected continuation. It should preserve
key observable state needed for future QA, but must not invent details or import facts from the question,
answer, answer options, or QA History.
- 1.0: Concise, faithful, correctly scoped to the interval, captures the main observable changes, preserves
  useful state for future QA, and properly merges any unfinished previous delta.
- 0.7: Mostly faithful and useful, but slightly vague, slightly long, missing minor changes, or only
  partially effective at merging the previous delta.
- 0.4: Generic, overlong, under-informative, partially off-interval, weakly grounded, or misses important
  observable changes.
- 0.0: Hallucinates, contradicts frames or Memory, describes the wrong interval, copies question/options/QA
  content as visual fact, or is unusable as memory.

2. anchor_keyframe
Evaluate whether <anchor> preserves the right visual evidence from the current window. Important anchors
include query-relevant evidence, major object/person/state changes, scene/camera/workspace changes, event
boundaries, or visual details that a text delta cannot reliably preserve. At most one anchor should be
selected in one step. It is acceptable to select no anchor when the current window has no meaningful new
visual evidence. Penalize missing an anchor when the current change is important, when the delta span is
long, or when visual evidence cannot be safely reduced to text.
- 1.0: Selects exactly the right important anchor when needed, avoids duplicates, and does not save
  unimportant frames; or correctly selects no anchor when there is no useful new visual evidence.
- 0.7: Relevant anchor choice, but misses a moderately useful anchor, chooses a suboptimal time, or saves
  a mildly redundant frame.
- 0.4: Weakly relevant, redundant, poorly timed, loosely useful, or omits a useful but not critical anchor.
- 0.0: Misses a clearly important anchor, selects irrelevant frames, emits more than one anchor in this
  step, duplicates existing visual evidence, or uses an invalid/unhelpful anchor.
Special first-window rule: if the current frame window spans exactly t="0.0-5.0" or "0-5",
anchor_keyframe may be 1.0 only when the output contains exactly one anchor with t="0.0-1.0" or "0-1".
For that first window, anchor_keyframe must be 0.0 if the anchor is absent, duplicated, or uses any other
timing.

3. semantic_alignment
Evaluate whether the resulting visual-text memory is aligned with the current frames, frame order, anchor
times, delta intervals, and existing Memory. The anchor/delta sequence should form a coherent, time-ordered
memory update whose text and visual evidence support each other.
- 1.0: Anchor times, delta intervals, frame order, described content, and existing Memory all align.
- 0.7: Minor omissions or mild ambiguity, but no material contradiction and temporal order remains clear.
- 0.4: Partial mismatch between text and images, weak temporal ordering, unclear grounding, or poor
  coherence with existing Memory.
- 0.0: Clear contradiction, wrong event order, anchor/delta mismatch, text not grounded in images, or a
  memory update that breaks temporal coherence.

4. state_groundedness
Evaluate whether <state> makes reliable, useful reasoning from Memory, current frames, and QA History.
It should state uncertainty when evidence is insufficient, avoid hallucination and unsupported inference,
and avoid being contaminated by question wording, answer choices, previous answers, or QA History. If
<state> reasons about whether to answer, judge whether that decision is grounded and whether the actual
<answer> is consistent with the state's reasoning.
- 1.0: Grounded, useful for QA and memory decisions, hallucination-free, handles uncertainty correctly,
  and makes a sound answer-or-silence decision consistent with <answer>.
- 0.7: Mostly grounded, but generic, mildly incomplete, or missing some useful decision context.
- 0.4: Weak, speculative, not useful for QA decisions, mildly inconsistent with evidence, or partially
  contaminated by QA History or answer options.
- 0.0: Hallucinates, contradicts Memory/current frames, claims certainty without evidence, forces an
  unsupported answer, or makes an answer decision inconsistent with the actual <answer>.
"""


GRPPO_ANSWER_RUBRIC = """\
5. answer_reward
Binary answer reward. It must be exactly 0.0 or 1.0.
Judge answer_reward primarily from the Answer Label. The label's current requirement and reference
answer are authoritative for the current step.
- If the label says the model should not answer now, <answer> must be empty.
- If the label requires an answer now, the model must answer and the answer must match the reference
  answer or be a clearly equivalent paraphrase.
- The model's <state> is used only to check whether its reasoning is grounded and consistent with
  its <answer>. The state must not override, reinterpret, or reject the reference answer.
- Give 0.0 if the model answers when it should stay silent, stays silent when it must answer, answers
  incorrectly, contradicts its own <state>, or uses ungrounded/contradictory state reasoning to support
  the answer.
"""


GRPPO_OUTPUT_SCHEMA_4 = """\
Return JSON only. Do not output any other text:
{
  "delta_groundedness": {"score": 0.0, "reason": "brief reason"},
  "anchor_keyframe": {"score": 0.0, "reason": "brief reason"},
  "semantic_alignment": {"score": 0.0, "reason": "brief reason"},
  "state_groundedness": {"score": 0.0, "reason": "brief reason"},
  "caps_applied": ["brief cap notes, or empty list"],
  "overall": 0.0,
  "issues": ["brief issue notes"]
}
"""


GRPPO_OUTPUT_SCHEMA_5 = """\
Return JSON only. Do not output any other text:
{
  "delta_groundedness": {"score": 0.0, "reason": "brief reason"},
  "anchor_keyframe": {"score": 0.0, "reason": "brief reason"},
  "semantic_alignment": {"score": 0.0, "reason": "brief reason"},
  "state_groundedness": {"score": 0.0, "reason": "brief reason"},
  "answer_reward": {"score": 0.0, "reason": "brief reason"},
  "caps_applied": ["brief cap notes, or empty list"],
  "overall": 0.0,
  "issues": ["brief issue notes"]
}
"""


def build_legacy_judge_content(ctx: LegacyJudgePromptContext) -> list[ContentItem]:
    frame_window = _frame_window_text(ctx.frames)
    frame_lines = _frame_index_text(ctx.frames)
    issue_codes = _issue_codes_text(ctx.quality)
    action_summary = _action_summary(ctx.raw_action)

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
{ctx.memory_before or "<empty/>"}

=== Current Frame Window ===
{frame_window}

=== Current Frame Index ===
{frame_lines or "<none>"}

=== QA History ===
{ctx.qa_history or "<empty/>"}

=== Parsed Agent Output Summary ===
{action_summary}

=== Raw Agent Output ===
{ctx.raw_output}

=== Existing Parser/Timing Issues ===
{issue_codes}

{LEGACY_PROCESS_RUBRIC}

Hard caps:
- If parser/timing issues make a component unusable, cap that component at 0.4.
- If the output has no usable anchor/delta content, cap overall at 0.2.
- If the output clearly contradicts current frames or memory, cap overall at 0.5.
- If the output fabricates important visual facts, cap overall at 0.3.
- If <state> or <delta> relies on hidden future frames or outside knowledge, cap overall at 0.3.

Overall score:
Use the average of the four dimension scores after applying any hard caps. Keep scores calibrated;
do not give high scores for fluent but ungrounded text.

{LEGACY_OUTPUT_SCHEMA}
"""
    return _content_with_frames(prompt, ctx.frames)


def build_grppo_judge_content(ctx: GrppoJudgePromptContext) -> list[ContentItem]:
    frame_window = _frame_window_text(ctx.frames)
    frame_lines = _frame_index_text(ctx.frames)
    issue_codes = _issue_codes_text(ctx.quality)
    answer_label_block = _answer_label_block(ctx)
    answer_reward_event = bool(ctx.answer_reward_event)
    score_intro = "Return 5 scores in [0.0, 1.0]:" if answer_reward_event else "Return 4 scores in [0.0, 1.0]:"
    answer_rubric = f"\n{GRPPO_ANSWER_RUBRIC}" if answer_reward_event else ""
    answer_caps = "- answer_reward must be exactly 0.0 or 1.0.\n" if answer_reward_event else ""
    overall_rule = (
        "overall is the average of the four process dimensions. Do not include answer_reward in overall."
        if answer_reward_event
        else "overall is the average of the four process dimensions."
    )
    output_schema = GRPPO_OUTPUT_SCHEMA_5 if answer_reward_event else GRPPO_OUTPUT_SCHEMA_4

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
{ctx.memory_before or "<empty/>"}

=== Current Video Frames ===
Time window: {frame_window}
Frame index:
{frame_lines or "<none>"}
"""

    prompt_suffix = f"""\

=== QA History Visible To The Model ===
{ctx.qa_history or "<empty/>"}

{answer_label_block}

=== Model Output ===
{ctx.raw_output}

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

{GRPPO_PROCESS_RUBRIC}{answer_rubric}

## Hard Caps

- If parser or timing issues make a component unusable, cap that component at 0.4.
- If state or delta fabricates important visual facts, cap that component at 0.3.
- If state or delta copies query/options into memory as observed video facts, cap that component at 0.3.
- If the output has no usable anchor/delta content, cap overall at 0.2.
- If the output clearly contradicts current frames or Memory, cap overall at 0.5.
- If state or delta relies on hidden future frames or outside knowledge, cap overall at 0.3.
{answer_caps}
## overall

{overall_rule}

{output_schema}
"""
    return _content_with_frames(prompt_prefix, ctx.frames, suffix=prompt_suffix)


def build_grppo_answer_label_section(label: dict[str, Any] | None, *, answer_reward_event: bool) -> str:
    if not answer_reward_event:
        return ""
    if label is None:
        return "Current requirement: the model should not answer in this step; <answer> should be empty."

    event_type = str(label.get("event_type") or "")
    question = str(label.get("question") or label.get("content") or label.get("query") or "").strip()
    timestamp = _label_time_text(label)
    target = _label_answer_text(label)
    options = _label_options_text(label)
    should_answer = label.get("should_answer")
    should_stay_silent = should_answer is not None and not _truthy_label_value(should_answer)

    lines: list[str] = []
    if event_type == "query":
        lines.append("Event type: user query.")
    elif event_type == "answer_silence":
        lines.append("Event type: silence checkpoint; no answer is required now.")
    elif event_type == "answer_target":
        lines.append("Event type: answer checkpoint.")
    elif event_type:
        lines.append(f"Event type: {event_type}.")
    if question:
        lines.append(f"Current question: {question}")
    if options:
        lines.append(f"Options:\n{options}")
    if timestamp:
        lines.append(timestamp)
    if should_stay_silent:
        lines.append("Requirement: the model should not answer now; <answer> should be empty.")
    elif target:
        if event_type == "answer_target":
            lines.append(f"Requirement: the model must answer now. Reference answer: {target}")
        else:
            lines.append(f"Requirement: the model should answer now. Reference answer: {target}")
    else:
        lines.append("Requirement: the model should not answer now; <answer> should be empty.")
    return "\n".join(lines)


def _answer_label_block(ctx: GrppoJudgePromptContext) -> str:
    if not ctx.answer_reward_event:
        return ""
    correctness_text = "null" if ctx.answer_correctness is None else f"{_clamp_score(ctx.answer_correctness):.4f}"
    answer_label = build_grppo_answer_label_section(
        ctx.query_label,
        answer_reward_event=ctx.answer_reward_event,
    )
    return f"""\
=== Answer Label ===
{answer_label}

=== Rule Correctness Reference ===
rule_answer_correctness={correctness_text}
This rule score is an auxiliary reference. Judge answer_reward primarily from the Answer Label:
the model's <answer> must satisfy the current requirement and match the reference answer when one is required.
Also check whether the model's <state> is grounded and consistent with its <answer>.
If the answer is wrong, missing when required, present when silence is required, or inconsistent with the
state reasoning, answer_reward must be 0.0.
"""


def _content_with_frames(prefix: str, frames: list[FrameRef], *, suffix: str = "") -> list[ContentItem]:
    content: list[ContentItem] = [ContentItem("text", text=prefix)]
    for idx, frame in enumerate(frames, start=1):
        content.append(ContentItem("text", text=f'\n[frame_{idx} t="{frame.start_time:.1f}-{frame.end_time:.1f}"]\n'))
        content.append(ContentItem("image", image_path=frame.image_path))
    if suffix:
        content.append(ContentItem("text", text=suffix))
    return content


def _frame_window_text(frames: list[FrameRef]) -> str:
    if not frames:
        return "<none>"
    return f't="{frames[0].start_time:.1f}-{frames[-1].end_time:.1f}"'


def _frame_index_text(frames: list[FrameRef]) -> str:
    return "\n".join(
        f'- frame_{idx}: t="{frame.start_time:.1f}-{frame.end_time:.1f}", global_index={frame.global_index}'
        for idx, frame in enumerate(frames, start=1)
    )


def _issue_codes_text(quality: QualityReport) -> str:
    return ", ".join(issue.code for issue in quality.issues) or "<none>"


def _label_options_text(label: dict[str, Any]) -> str:
    options = label.get("options")
    if isinstance(options, dict):
        lines = []
        for key, value in options.items():
            letter = str(key).strip()
            text = str(value).strip()
            if letter and text:
                lines.append(f"{letter}. {text}" if not text.upper().startswith(f"{letter.upper()}.") else text)
        return "\n".join(lines)
    if not isinstance(options, list):
        return ""
    lines = []
    for index, value in enumerate(options):
        letter = chr(ord("A") + index)
        text = str(value).strip()
        if not text:
            continue
        lines.append(text if text[:2].upper() == f"{letter}." else f"{letter}. {text}")
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
    if label.get("event_type") == "answer_target":
        for key in ("answer", "content", "target_answer", "target"):
            value = label.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    for key in ("ground_truth", "gt", "target", "target_answer"):
        value = label.get(key)
        if value is not None and str(value).strip():
            option_text = _option_text_for_label(value, label)
            return option_text or str(value).strip()
    value = label.get("answer")
    if value is not None and str(value).strip():
        return str(value).strip()
    return ""


def _option_text_for_label(value: Any, label: dict[str, Any]) -> str:
    raw = str(value).strip()
    if len(raw) != 1 or not raw.isalpha():
        return ""
    options = label.get("options")
    option_value: Any | None = None
    if isinstance(options, dict):
        option_value = options.get(raw) or options.get(raw.upper()) or options.get(raw.lower())
    elif isinstance(options, list):
        index = ord(raw.upper()) - ord("A")
        if 0 <= index < len(options):
            option_value = options[index]
    if option_value is None:
        return ""
    text = str(option_value).strip()
    if not text:
        return ""
    return text if text[:2].upper() == f"{raw.upper()}." else f"{raw.upper()}. {text}"


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


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(score, 0.0), 1.0)
