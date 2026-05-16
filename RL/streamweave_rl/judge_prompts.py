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


@dataclass(slots=True)
class GrppoAnswerJudgePromptContext:
    raw_output: str
    quality: QualityReport
    query_label: dict[str, Any] | None


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
For each process dimension, fill every listed check with 1, 0.5, or 0:
- 1 means the check passes.
- 0.5 means the check is neutral or partially applicable only where explicitly allowed below.
- 0 means the check fails.
- Do not use null.
Set the dimension score to the mean of all listed checks. Reflect severe failures by failing the
relevant checks; do not hide them only in caps_applied. Include every listed check for each dimension;
omitting a listed check is treated as a failed check.

1. delta_groundedness checks
- delta_captures_action_progress: <delta> records the main observable actions, progress, starts, stops,
  completions, or phase changes in the frames between the previous visual anchor and the next visual
  anchor, or the current window end if there is no next anchor.
- delta_captures_state_or_location_changes: when visible, <delta> preserves important object/person
  state, location, layout, or cumulative-progress changes that are not already preserved by the
  surrounding visual anchors.
- delta_preserves_query_relevant_details: <delta> keeps textual details that may matter for current or
  future QA and can be safely represented in text.
- delta_no_visual_hallucination: <delta> does not invent unsupported objects, attributes, actions, counts,
  relations, causes, or outcomes.
- delta_not_polluted_by_qa: <delta> does not treat question text, answer choices, prior answers, or QA
  History as observed video facts.

2. anchor_keyframe checks
- anchor_count_valid: the step emits at most one <anchor>.
- anchor_time_and_body_valid: if an <anchor> is emitted, its time range must come from the current frame
  window and the <anchor> tag body must be empty; if no <anchor> is emitted, use 0.5.
- anchor_representative_if_present: when an anchor is used, it is the best representative frame for the
  current state/evidence rather than a poorly timed frame; if no <anchor> is emitted, use 0.5.
- first_window_rule_followed: if the current window is exactly t="0.0-5.0" or "0-5", the output contains
  exactly one anchor at t="0.0-1.0" or "0-1"; if this is not the first window, use 1.

3. semantic_alignment checks
- semantic_output_coherent: the current model output is semantically coherent as a whole; its <anchor>,
  <delta>, <state>, and <answer> do not conflict with each other.
- semantic_text_anchor_express_current_frames: compared with the current frames, the model's text and
  anchored frame, if any, effectively express the visually important content of this step.
- semantic_no_cross_step_contradiction: the output does not move events to the wrong time, reverse
  cause/order, or create a contradiction between Memory before this step and the current output.

4. state_groundedness checks
- state_uses_available_evidence: <state> reasons from historical Memory and current frames, captures
  the important evidence, and stays focused rather than verbose.
- state_identifies_question_scope: when QA History contains a question, <state> recognizes the question
  and reasons toward the correct answer from the visible content.
- state_decision_is_grounded: <state> gives an evidence-based reason for answering or staying silent when
  a QA decision is relevant.
- state_no_visual_hallucination: <state> does not invent unsupported visual facts.
- state_consistent_with_answer: <state>'s reasoning is consistent with the actual <answer>.
- state_no_unreasonable_hallucination: <state> does not contain unreasonable hallucinations, including
  unsupported causes, intentions, conclusions, answer inferences, or claims imported from QA History
  without visual/memory evidence.
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


GRPPO_ANSWER_ONLY_SCHEMA = """\
Return JSON only. Do not output any other text:
{
  "answer_reward": {"score": 0.0, "reason": "brief reason"}
}
"""


GRPPO_OUTPUT_SCHEMA_4 = """\
Return JSON only. Do not output any other text:
{
  "delta_groundedness": {
    "score": 0.0,
    "checks": {
      "delta_captures_action_progress": 0.0,
      "delta_captures_state_or_location_changes": 0.0,
      "delta_preserves_query_relevant_details": 0.0,
      "delta_no_visual_hallucination": 0.0,
      "delta_not_polluted_by_qa": 0.0
    },
    "reason": "brief reason"
  },
  "anchor_keyframe": {
    "score": 0.0,
    "checks": {
      "anchor_count_valid": 0.0,
      "anchor_time_and_body_valid": 0.0,
      "anchor_representative_if_present": 0.0,
      "first_window_rule_followed": 0.0
    },
    "reason": "brief reason"
  },
  "semantic_alignment": {
    "score": 0.0,
    "checks": {
      "semantic_output_coherent": 0.0,
      "semantic_text_anchor_express_current_frames": 0.0,
      "semantic_no_cross_step_contradiction": 0.0
    },
    "reason": "brief reason"
  },
  "state_groundedness": {
    "score": 0.0,
    "checks": {
      "state_uses_available_evidence": 0.0,
      "state_identifies_question_scope": 0.0,
      "state_decision_is_grounded": 0.0,
      "state_no_visual_hallucination": 0.0,
      "state_consistent_with_answer": 0.0,
      "state_no_unreasonable_hallucination": 0.0
    },
    "reason": "brief reason"
  },
  "caps_applied": ["brief cap notes, or empty list"],
  "overall": 0.0,
  "issues": ["brief issue notes"]
}
"""


GRPPO_OUTPUT_SCHEMA_5 = """\
Return JSON only. Do not output any other text:
{
  "delta_groundedness": {
    "score": 0.0,
    "checks": {
      "delta_captures_action_progress": 0.0,
      "delta_captures_state_or_location_changes": 0.0,
      "delta_preserves_query_relevant_details": 0.0,
      "delta_no_visual_hallucination": 0.0,
      "delta_not_polluted_by_qa": 0.0
    },
    "reason": "brief reason"
  },
  "anchor_keyframe": {
    "score": 0.0,
    "checks": {
      "anchor_count_valid": 0.0,
      "anchor_time_and_body_valid": 0.0,
      "anchor_representative_if_present": 0.0,
      "first_window_rule_followed": 0.0
    },
    "reason": "brief reason"
  },
  "semantic_alignment": {
    "score": 0.0,
    "checks": {
      "semantic_output_coherent": 0.0,
      "semantic_text_anchor_express_current_frames": 0.0,
      "semantic_no_cross_step_contradiction": 0.0
    },
    "reason": "brief reason"
  },
  "state_groundedness": {
    "score": 0.0,
    "checks": {
      "state_uses_available_evidence": 0.0,
      "state_identifies_question_scope": 0.0,
      "state_decision_is_grounded": 0.0,
      "state_no_visual_hallucination": 0.0,
      "state_consistent_with_answer": 0.0,
      "state_no_unreasonable_hallucination": 0.0
    },
    "reason": "brief reason"
  },
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
        "overall is a process-only diagnostic score. Compute it as 2.0 times the mean of all "
        "process checks across the four process dimensions. Do not include answer_reward in overall."
        if answer_reward_event
        else "overall is a process-only diagnostic score. Compute it as 2.0 times the mean of all "
        "process checks across the four process dimensions."
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

## Hard Failure Guidance

- If parser or timing issues make a component unusable, fail the affected checks.
- If state or delta fabricates important visual facts, fail the hallucination and grounding checks.
- If state or delta copies query/options into memory as observed video facts, fail the QA-pollution checks.
- If the output has no usable anchor/delta content, fail the relevant delta/anchor/semantic checks.
- If the output clearly contradicts current frames or Memory, fail the relevant grounding/alignment checks.
- If state or delta relies on hidden future frames or outside knowledge, fail the relevant state/delta checks.
{answer_caps}
## overall

{overall_rule}

{output_schema}
"""
    return _content_with_frames(prompt_prefix, ctx.frames, suffix=prompt_suffix)


def build_grppo_answer_judge_content(ctx: GrppoAnswerJudgePromptContext) -> list[ContentItem]:
    answer_label = build_grppo_answer_label_section(ctx.query_label, answer_reward_event=True)
    issue_codes = _issue_codes_text(ctx.quality)
    prompt = f"""\
You are a strict answer-reward judge for one step of a streaming video QA agent.

Judge only whether the model's <answer> satisfies the current Answer Label.
Use only the Answer Label and the Model Output below. Do not use video frames, Memory, future frames,
outside knowledge, or your own interpretation of whether the visual evidence changed.
The Answer Label is authoritative for this step.

Important rule for answer checkpoints:
- If the label says "the model must answer now", the model must output a non-empty <answer> now.
- This is true even if the original question says "update your answer when the video evidence changes".
- At an answer checkpoint, staying silent is incorrect.

For multiple-choice questions, treat the full option form and its content as equivalent.
For example, if the reference answer is "C. 0", then "C", "0", and "C. 0" are equivalent.

=== Answer Label ===
{answer_label}

=== Model Output ===
{ctx.raw_output}

=== Existing Parser Or Timing Issues ===
{issue_codes}

Scoring:
- Return answer_reward = 1.0 only if the model follows the current requirement.
- If the label requires an answer now, <answer> must be non-empty and match the reference answer or a
  clearly equivalent paraphrase.
- If the label requires silence now, <answer> must be empty.
- The model's <state> can only be used as a consistency check. It must not override or reinterpret the
  Answer Label.
- Give answer_reward = 0.0 if the model stays silent when it must answer, answers when it must stay
  silent, gives the wrong answer, or gives an answer contradicted by its own <state>.
- If parser issues make <answer> unusable, give answer_reward = 0.0.

{GRPPO_ANSWER_ONLY_SCHEMA}
"""
    return [ContentItem("text", text=prompt)]


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
    answer_label = build_grppo_answer_label_section(
        ctx.query_label,
        answer_reward_event=ctx.answer_reward_event,
    )
    return f"""\
=== Answer Label ===
{answer_label}

Judge answer_reward from the Answer Label:
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
        combined = _combined_option_answer_text(label)
        if combined:
            return combined
        content = label.get("content")
        if content is not None and str(content).strip():
            return str(content).strip()
        for key in ("answer", "target_answer", "target"):
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


def _combined_option_answer_text(label: dict[str, Any]) -> str:
    option = None
    for key in ("ground_truth", "gt"):
        value = label.get(key)
        if value is not None and str(value).strip():
            option = str(value).strip()
            break
    if not option or len(option) != 1 or not option.isalpha():
        return ""
    answer = label.get("answer") or label.get("target_answer") or label.get("target")
    if answer is None or not str(answer).strip():
        return ""
    text = str(answer).strip()
    return text if text[:2].upper() == f"{option.upper()}." else f"{option.upper()}. {text}"


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
