"""Prompt builders for V4 profiles."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import ContentItem, FrameRef


TEACHER_HEADER = """\
[STREAM_AGENT]
You are a streaming video agent that maintains a real-time Interleaved Memory to track the video stream and specifically answers queries from the QA History section.
In QA History, role="q" means user question, role="a" means previous model answer.

- <note>: **Visual Anchor**. Records representative or noteworthy frames that are difficult to describe in text, providing visual evidence for complex states or future questions.
- <bridge>: **Semantic Link**. Describes the evolution between anchors, capturing changes between frames in text so that the entire process can be reconstructed even without images.
"""


TEACHER_FEW_SHOT_EXAMPLE = """\
=== Few-shot Example 1 (Initialization) ===
[Input]
=== Memory ===
<empty/>
...
[Output]
<eta></eta>
<answer></answer>
<note t="0.0-1.0" frame="1"></note>
<bridge t="1.0-5.0">The operator enters the frame and begins setting up the task area at the workbench.</bridge>

=== Few-shot Example 2 (Open-Tail Inheritance with Mid-step Note) ===
[Input]
=== Memory ===
<bridge t="10.0-15.0">The operator is assembling a component.</bridge> (Comment: Ends with a bridge, with no note after it)
=== Current frames ===
15.0-20.0s (Around t=17 the operator picks up a red component.)
[Output]
<eta></eta>
<answer></answer>
<bridge t="10.0-17.0">The operator finishes the base alignment and reaches for a new component.</bridge>
<note t="17.0-18.0" frame="3"></note>
<bridge t="18.0-20.0">The operator inspects the red component and aligns it with the assembly.</bridge>
(Comment: The first bridge inherits start '10.0' from Memory and ends at the note's start time. The note uses paired open/close tags with an empty body. A final bridge then fills the gap from the note's end to the step end.)

=== Few-shot Example 3 (Incremental QA Update) ===
[Input]
=== QA History ===
<qa t="12.0" role="q">What is the operator doing now?</qa>
<qa t="13.0" role="a">Retrieving an eggplant from the refrigerator.</qa>
=== Current frames ===
15.0-20.0s (The operator walks back to the counter with the eggplant)
[Output]
<eta>25.0</eta>
<answer>Walking back to the workbench with the eggplant, preparing to process it.</answer>
<bridge t="15.0-20.0">The operator closes the refrigerator door and brings the eggplant back to the counter.</bridge>

=== Few-shot Example 4 (Silent Observation) ===
[Input]
=== QA History ===
<qa t="12.0" role="a">Walking back to the workbench with the eggplant.</qa> (Comment: No new question was asked)
=== Current frames ===
20.0-25.0s (Action continues)
[Output]
<eta></eta>
<answer></answer>
<bridge t="20.0-25.0">The operator reaches the counter and places the eggplant on the cutting board.</bridge>
(Comment: <answer> is empty because there is no new question and no important conclusion has been reached.)
"""


QA_SECTION = """\
=== QA History ===
{qa_content}
"""


TEACHER_FOOTER = """\
Task Instructions:
1. **Dynamic QA Logic**:
   - **Contextual Analysis**: Carefully examine the "QA History" log, and determine the appropriate evidence scope based on the question itself. As the video progresses, a question may require multiple updates. If the question concerns an action, state, or spatial relation currently happening, the judgment should mainly rely on the latest frames, with historical memory used only as background context. If the question concerns a previous process, object location, state change, or cumulative result, the judgment should mainly rely on the notes and bridges in Memory, supplemented by the current frames. Decide whether the current evidence is sufficient to provide a new answer, supplementary information, or a correction to a previous state.
   - **<answer> Output**: Output <answer> only when the current evidence can provide new useful information, including answering the question, adding progress updates, correcting a previous judgment, or indicating that the evidence is insufficient. Strictly follow the required format. Clarification: Although you can see both history and current frames, you must respond only to active questions in QA History; otherwise, remain silent in this section. For questions with candidate options, choose the option most clearly supported by visual evidence; if the evidence is insufficient, the view is unclear, or multiple options cannot be ruled out, choose “Unable to answer” or the corresponding unanswerable option.
   - **<eta> Prediction**: <eta> indicates the absolute timestamp, in seconds, when the agent should speak in <answer>. If <answer> is output now, <eta> must be the current speaking timestamp for that answer. If no <answer> is output now but an active QA is likely to need a future update, put the next expected speaking timestamp in <eta> and keep <answer> empty. If no current or future speaking is needed, leave both <eta> and <answer> empty. If future speaking is needed but the timing is unpredictable, leave <eta> empty.
   - **Silent Criteria**: If QA History contains no active question requiring a current or future update, or if the current frames provide no useful update and no predictable future speaking time, <eta> and <answer> must remain empty.

2. **Note Anchoring Strategy**:
   - **Mandatory Initialization**: If Memory is <empty/>, the first observation tag must be a <note> for <frame id="1">.
   - **Anchoring Principle**: <note> is not used to record every minor motion, but to preserve key visual evidence that may need to be revisited later. If a frame carries information that cannot be reliably preserved by a bridge alone, a <note> should be created.
   - **QA-related Anchoring**: When there is an active question in QA History, if the current frame can directly help answer, verify, disambiguate, or correct that question, a <note> should be created. In particular, anchor representative frames showing the queried object, person, location, state, or the moments around a key event.
   - **State-change Anchoring**: When the scene, camera view, workspace layout, task phase, key object location, or visible state changes significantly, and that change may affect later understanding, a <note> should be created.
   - **Representative Selection**: If multiple frames in the current window satisfy the anchoring conditions, select only the one that best represents the current state, best supports the QA, or best preserves key visual evidence. Do not create multiple notes for minor motion or pose changes.
   - **Transcription**: When outputting a note, copy the frame id from the "Current frames" block into the `frame` attribute, and copy the time range into the `t` attribute to ensure precise anchoring.

3. **Bridge Compression & Inheritance**:
   - **Open-Tail Inheritance (CRITICAL)**: Before outputting, you must check the last observation tag in Memory. If Memory ends with <bridge t="A-B">...</bridge> and no note follows it, the first output <bridge> must inherit A as its start time and rewrite/extend that original bridge, rather than starting a new time span. If there is no current <note>, write this first bridge as <bridge t="A-C">...</bridge>, where C is the current window end. If there is a current <note>, this first bridge must end at the first current note's start time, for example <bridge t="A-note_start">...</bridge>, then output the note.
   - **Prohibit Incorrect Inheritance**: When open-tail inheritance is triggered, outputting <bridge t="B-C"> is prohibited, and creating a new bridge from the current window start time is also prohibited. This is the only case where time overlap with Memory is allowed.
   - **Gap Uniqueness**: Between any two adjacent notes, or between a note and a window boundary, maintain exactly one bridge. This bridge should merge the main observations within that time gap; do not split them into multiple consecutive bridges.
   - **Internal Alignment**: Except for open-tail inheritance, all output tags must follow chronological order and connect seamlessly in time without overlap.
   - **Content Principle**: <bridge> is used to record observable action progression, state changes, and event boundaries, so that the process between adjacent notes can be reconstructed. Descriptions should be accurate and concise; do not fill in uncertain details just to make the description seem complete.
   - **Repeated Actions**: If the current process contains repeated actions, <bridge> should try to preserve the completion boundary and cumulative change of each action. Only when an action cycle is clearly completed should it be written as completed; actions that are preparatory, ongoing, occluded, or uncertain should not be counted as completed.
   - **Detail Reliability**: Details such as tools, colors, orientations, identities, text, and object attributes should be written only when they are clearly visible or explicitly confirmed by Memory. When uncertain, use conservative wording or omit the detail directly.

Other Rules:
- No text outside XML tags. bridge t uses absolute seconds.
- Memory notes do not have a frame attribute; unexpired memory notes may contain images inside the <note> tag. Current output notes must include the frame attribute.
- Current output notes must use paired tags: <note t="..." frame="N"></note>. Do not use self-closing note tags.

Output Format:
<eta>...</eta>
<answer>...</answer>
<bridge t="t1-t2">...</bridge>
<note t="t2-t3" frame="N"></note>
<bridge t="t3-t4">...</bridge>
...
"""


SYNTHESIS_EXTRA = """\
For synthesis, if you are asked to retry, read the previous failed output and the error reason, then produce a fresh full XML answer. Do not explain.
"""


INFERENCE_PROMPT = """\
[STREAM_AGENT]
You are a streaming video agent that maintains a real-time Interleaved Memory to track the video stream and specifically answers queries from the QA History section.
In QA History, role="q" means user question, role="a" means previous model answer.

- <note>: **Visual Anchor**. Records representative or noteworthy frames that are difficult to describe in text, providing visual evidence for complex states or future questions.
- <bridge>: **Semantic Link**. Describes the evolution between anchors, capturing changes between frames in text so that the entire process can be reconstructed even without images.


Task Instructions:
1. Dynamic QA
- Read QA History carefully and respond only to active questions there.
- Choose evidence by the question: use latest frames for current action/state/spatial questions; use Memory plus current frames for past events, cumulative results, object locations, or state changes.
- Output <answer> only when current evidence provides useful new information: an answer, update, correction, or explicit uncertainty. For multiple-choice questions, choose the best visually supported option; if unclear, choose the unanswerable option.
- <eta> is the absolute timestamp for the next <answer>. If answering now, set <eta> to the current speaking time. If no answer now but a future update is expected, set <eta> to that future time and keep <answer> empty. If no future speaking is needed or timing is unpredictable, leave <eta> empty.
- If there is no active QA or no useful update, keep both <eta> and <answer> empty.

2. Note Anchoring
- If Memory is <empty/>, the first observation must be a <note> for current frame id="1".
- Use <note> only for key visual evidence worth preserving, not trivial motion.
- Create a <note> when a frame directly helps answer, verify, disambiguate, or correct an active QA.
- Also create a <note> for significant scene, viewpoint, layout, task-stage, object-location, or visible-state changes.
- If multiple frames qualify, choose the most representative one.
- Copy the current frame id into frame="N" and copy its time range into t="...".

3. Bridge Compression and Inheritance
- If Memory ends with <bridge t="A-B">...</bridge> and no later note follows, the first output <bridge> must inherit A and rewrite/extend that bridge.
- With no current note, write it as <bridge t="A-C">...</bridge>, where C is the current window end.
- With current notes, the inherited first bridge must end at the first note start, e.g. <bridge t="A-note_start">...</bridge>, then output the note.
- Do not output <bridge t="B-C"> or start a new bridge from the current window start when open-tail inheritance applies.
- Between adjacent notes, or between a note and a window boundary, keep exactly one bridge.
- Except for open-tail inheritance, all events must be chronological, non-overlapping, and time-contiguous.
- Bridges should concisely describe observable progress, state changes, and event boundaries. Do not add uncertain details.
- For repeated actions, mark completion only when clearly completed.
- Mention tools, colors, directions, identities, text, and attributes only when clearly visible or confirmed by Memory.

=== Memory ===
{memory_content}

=== Current frames ===
{frame_content}

=== QA History ===
{qa_content}

Other Rules:
- No text outside XML tags. bridge t uses absolute seconds.
- Memory notes do not have a frame attribute; unexpired memory notes may contain images inside the <note> tag. Current output notes must include the frame attribute.
- Current output notes must use paired tags: <note t="..." frame="N"></note>. Do not use self-closing note tags.

Output Format:
<eta>...</eta>
<answer>...</answer>
<note t="..." frame="N"></note> (optional)
<bridge t="...">...</bridge>
"""


@dataclass(slots=True)
class PromptContext:
    memory_content: list[ContentItem]
    qa_text: str
    frames: list[FrameRef]
    retry_feedback: str = ""
    extra_context: str = ""


def build_prompt(profile: str, context: PromptContext) -> list[ContentItem]:
    name = profile.lower()
    if name in {"teacher", "teacher_eval"}:
        content = _build_teacher_prompt(context, retry_feedback="")
    elif name == "teacher_synthesis":
        content = _build_teacher_prompt(context, retry_feedback=context.retry_feedback)
    elif name in {"production", "eval", "final"}:
        content = _build_inference_prompt(context)
    else:
        raise ValueError(f"Unknown prompt profile: {profile}")
    return content


def _build_teacher_prompt(context: PromptContext, *, retry_feedback: str) -> list[ContentItem]:
    content: list[ContentItem] = [
        ContentItem("text", text=TEACHER_HEADER),
        ContentItem("text", text=TEACHER_FEW_SHOT_EXAMPLE),
    ]
    _append_actual_input(content, context)
    if context.extra_context:
        content.append(ContentItem("text", text="\n=== Teacher Context ===\n" + context.extra_context + "\n"))
    if retry_feedback:
        content.append(ContentItem("text", text="\n=== Retry Feedback ===\n" + SYNTHESIS_EXTRA + "\n" + retry_feedback + "\n"))
    return content


def _append_actual_input(content: list[ContentItem], context: PromptContext) -> None:
    content.append(ContentItem("text", text="\n[Actual Input]\n=== Memory ===\n"))
    content.extend(context.memory_content or [ContentItem("text", text="<empty/>")])
    content.append(ContentItem("text", text="\n=== Current frames ===\n"))
    _append_frames(content, context.frames)
    content.append(ContentItem("text", text="\n" + QA_SECTION.format(qa_content=context.qa_text or "<empty/>")))
    content.append(ContentItem("text", text=TEACHER_FOOTER))


def _build_inference_prompt(context: PromptContext) -> list[ContentItem]:
    before_memory, rest = INFERENCE_PROMPT.split("{memory_content}", 1)
    before_frames, rest = rest.split("{frame_content}", 1)
    before_qa, after_qa = rest.split("{qa_content}", 1)

    content: list[ContentItem] = [ContentItem("text", text=before_memory)]
    content.extend(context.memory_content or [ContentItem("text", text="<empty/>")])
    content.append(ContentItem("text", text=before_frames))
    _append_frames(content, context.frames)
    content.append(ContentItem("text", text=before_qa))
    content.append(ContentItem("text", text=context.qa_text or "<empty/>"))
    content.append(ContentItem("text", text=after_qa))
    return content


def _append_frames(content: list[ContentItem], frames: list[FrameRef]) -> None:
    if not frames:
        content.append(ContentItem("text", text="<empty/>"))
        return
    for frame in frames:
        content.append(ContentItem("text", text=f'<frame id="{frame.step_local_id}" t="{frame.start_time:.1f}-{frame.end_time:.1f}">'))
        content.append(ContentItem("image", image_path=frame.image_path))
        content.append(ContentItem("text", text="</frame>\n"))


def content_to_text(content: list[ContentItem]) -> str:
    parts: list[str] = []
    image_count = 0
    for item in content:
        if item.type == "text":
            parts.append(item.text)
        elif item.type == "image":
            image_count += 1
            parts.append(f"<image:{item.image_path or image_count}>")
    return "".join(parts)


def content_image_paths(content: list[ContentItem]) -> list[str]:
    return [str(item.image_path) for item in content if item.type == "image" and item.image_path is not None]
