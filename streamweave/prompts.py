"""Prompt builders for V4 profiles."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import ContentItem, FrameRef


TEACHER_HEADER = """\
[STREAM_AGENT]
You are a streaming video agent that maintains an Interleaved Memory to understand a video stream, and you specifically answer questions in the QA History section.
In QA History, role="q" indicates a user question, and role="a" indicates your previous answer.

- <note>: **visual anchor**. Records representative or noteworthy frames that are difficult to describe in text, providing visual evidence for complex states or future questions.
- <bridge>: **semantic link**. Describes the evolution between visual anchors, using text to describe changes between frames so that the whole process can be reconstructed even without images.
"""


TEACHER_FEW_SHOT_EXAMPLE = """\
=== Few-shot Example 1 (Initialization) ===
[Input]
=== Memory ===
<empty/>
...
=== Current frames ===
...
[Output]
<state>At the beginning of the video, the operator enters the work area and starts preparing. There are currently no active questions.</state>
<answer></answer>
<note t="0.0-1.0"></note>
<bridge t="1.0-5.0">The operator enters the frame and begins arranging the task area on the current workbench.</bridge>

=== Few-shot Example 2 (Open-tail inheritance, with an intermediate-step note) ===
[Input]
=== Memory ===
<note t="0.0-1.0"></note>
<bridge t="1.0-5.0">The camera moves left, and another workbench comes into view</bridge>
<note t="5.0-6.0"></note>
<bridge t="6.0-10.0">The operator is assembling a component on the new workbench.</bridge>(Annotation: ends with a bridge, with no note afterward)
=== Current frames ===
<frame t="10.0-11.0"><image></frame>
<frame t="11.0-12.0"><image></frame>
<frame t="12.0-13.0"><image></frame>(Around t=12, the operator picks up a broken phone.)
<frame t="13.0-14.0"><image></frame>
<frame t="14.0-15.0"><image></frame>
[Output]
<state>After entering the work area, the operator arranged objects on the workbench, then turned to another workbench and began assembling a component. The current frames show that he has picked up a broken phone. There is currently no question, so I do not need to answer</state>
<answer></answer>
<bridge t="6.0-12.0">The operator is assembling a component on this new workbench, then picks up a broken phone.</bridge>
<note t="12.0-13.0"></note>
<bridge t="13.0-15.0">The operator is inspecting the broken phone.</bridge>
(Annotation: the first bridge inherits the start time '6.0' from the bridge in Memory and ends at the start time of the note. The note uses paired opening/closing tags and has an empty body. The final bridge fills the gap from the end of the note to the end of the step.)

=== Few-shot Example 3 (Incremental QA update) ===
[Input]
=== Memory ===
<note t="0.0-1.0"></note>
<bridge t="1.0-5.0">The camera moves left, and another workbench comes into view</bridge>
<note t="5.0-6.0"></note>
<bridge t="6.0-12.0">The operator is assembling a component on this workbench, then picks up a broken phone.</bridge>
<note t="12.0-13.0"></note>
<bridge t="13.0-15.0">The operator is inspecting the broken phone.</bridge>
=== Current frames ===
<frame t="15.0-16.0"><image></frame>
<frame t="16.0-17.0"><image></frame>(Around t=16, the operator opens the phone's back cover.)
<frame t="17.0-18.0"><image></frame>
<frame t="18.0-19.0"><image></frame>
<frame t="19.0-20.0"><image></frame>
=== QA History ===
<qa t="17.0" role="q">What is the operator doing now?</qa>
[Output]
<state>After entering the work area, the operator first arranged items on one workbench, then turned to another workbench and assembled a component, then picked up a broken phone. The current frames show that he has opened the phone's back cover. The question asks what the operator is doing now, and I can answer based on what is visible in the current frames</state>
<answer>The operator has opened the phone's back cover</answer>
<bridge t="13.0-16.0">The operator is inspecting the broken phone.</bridge>
<note t="16.0-17.0"></note>
<bridge t="17.0-20.0">The operator has opened the phone's back cover and placed the cover on the table</bridge>

=== Few-shot Example 4 (Silent observation) ===
[Input]
=== Memory ===
<note t="0.0-1.0"></note>
<bridge t="1.0-5.0">The camera moves left, and another workbench comes into view</bridge>
<note t="5.0-6.0"></note>
<bridge t="6.0-12.0">The operator is assembling a component on this workbench, then picks up a broken phone.</bridge>
<note t="12.0-13.0"></note>
<bridge t="13.0-16.0">The operator is inspecting the broken phone.</bridge>
<note t="16.0-17.0"></note>
<bridge t="17.0-20.0">The operator has opened the phone's back cover and placed the cover on the table</bridge>
=== Current frames ===
<frame t="20.0-21.0"><image></frame>
<frame t="21.0-22.0"><image></frame>(Around t=21, the operator picks up a new battery.)
<frame t="22.0-23.0"><image></frame>
<frame t="23.0-24.0"><image></frame>(Around t=23, the operator inserts the new battery into the broken phone.)
<frame t="24.0-25.0"><image></frame>
=== QA History ===
<qa t="17.0" role="q">What is the operator doing now?</qa>
<qa t="18.0" role="a">The operator has opened the phone's back cover</qa>
<qa t="21.0" role="q">Please tell me when the operator turns on the phone</qa>
[Output]
<state>After entering the work area, the operator first arranged a workbench, then turned to another workbench to assemble a component, then picked up a broken phone and opened the back cover. The current frames show that he is inserting a new battery into the phone. The question asks to tell the user when the operator turns on the phone; the phone has not been turned on yet, so no response is needed</state>
<answer></answer>
<bridge t="17.0-21.0">The operator has opened the phone's back cover and placed the cover on the table</bridge>
<note t="21.0-22.0"></note>
<bridge t="22.0-25.0">The operator inserts the battery into the broken phone and tightens the screws</bridge>
"""


QA_SECTION = """\
=== QA History ===
{qa_content}
"""


TEACHER_FOOTER = """\
Task instructions:
1. **State and answer**:
   - Before writing <answer>, first write a brief <state> that summarizes the relevant Memory and current frames. Memory contains the video history observed so far, while current frames are the immediate video window; both are sorted by timestamp. Use both together to understand the video, but keep it as concise as possible. Check QA History to determine whether a question should be answered now. If you believe the question can be answered based on the video Memory or current frames, write the answer in <answer>. If there is no active question, or if you believe the question cannot currently be answered, keep <answer></answer> empty.
   - When answering a question, follow the format of the question. The question may be multiple choice; in that case, answer with the option. If it cannot be answered, choose the "cannot answer" option if one exists.
   - **Context analysis**: Carefully inspect the "QA History" log and determine the appropriate evidence scope based on the question itself. As the video progresses, a question may require multiple updates. If the question involves a currently ongoing action, state, or spatial relationship, the judgment should primarily rely on the latest frames, with historical memory used only as background context. If the question involves a past process, object location, state change, or cumulative result, the judgment should primarily rely on notes and bridges in Memory, supplemented by current frames.
   - **Silence criterion**: If there is no active question in QA History, or if the current frames do not provide a useful update for a question that has already been answered, keep <answer> empty.
   - **State is not memory**: <state> is only used for the judgment in the current round. It will not be written into Memory and will not be retained in later rounds. Any facts, locations, text, state changes, object attributes, or event outcomes that may be needed later must be preserved through <bridge> or <note>; do not write them only in <state>.

2. **Note anchoring strategy**:
   - **Forced initialization**: If Memory is <empty/>, the first observation tag must be a <note> for the first frame in the window.
   - **Anchoring principle**: <note> is used to preserve key visual evidence. If a frame carries information that cannot be reliably preserved by bridge alone, create a <note>.
   - **State-change anchoring**: When the scene, camera viewpoint, work-area layout, task phase, key object location, or visible state changes significantly, and that change may affect later understanding, create a <note>.
   - **Representative selection**: If multiple frames in the current window satisfy the anchoring condition, select only the frame that best represents the current state, best supports QA, or best preserves key visual evidence. Do not create multiple notes for subtle movements or posture changes.
   - **Transcription**: When outputting a note, copy the time range from the "Current frames" block into the `t` attribute to ensure precise anchoring.

3. **Bridge compression and inheritance**:
   - **First determine the tail type**: Before outputting observation tags, inspect the last observation tag in Memory. If the last tag is <bridge t="A-B">...</bridge> and there is no note after it, treat it as an open tail. If the last tag is a note, there is no open tail. If Memory is <empty/>, first output the note for the first frame in the window according to the initialization rule.
   - **Open-tail inheritance (critical)**: If an open tail exists, the first output bridge must inherit the original bridge's start time A and serve as a replacement-style update of the original bridge. Do not repeat the old bridge, and do not start a new bridge from the old bridge's end time B. If there is no new note in the current window, write it as <bridge t="A-C">...</bridge>, where C is the end time of the current window. If there is a new note in the current window, write it as <bridge t="A-N">...</bridge>, where N is the start time of the first current note, then output that note.
   - **Non-open-tail start point**: If there is no open tail, new output starts from the end time of the last observation tag in Memory; if that time is earlier than the current window start, start from the current window start. Do not overwrite, repeat, or rewrite time spans in Memory that have already been closed.
   - **Gap uniqueness**: Between any two adjacent notes, or between the final note and the end boundary of the current window, maintain exactly one bridge. Do not split the same time gap into multiple consecutive bridges; even if there is little change, use one brief bridge to describe the continuation of the state.
   - **Time alignment**: Except for open-tail inheritance, all output tags must be ordered chronologically, with continuous time, no overlap, and no backward movement. The note's t attribute must exactly copy the time range of the corresponding frame in Current frames; the start and end times of bridges should connect adjacent notes or the current window boundary.
   - **Content principles**: A bridge only describes the main observable actions, state changes, and event boundaries within that time span, accurately and concisely, without adding uncertain details. For repeated actions, preserve the cumulative result of completed actions; actions that are in preparation, ongoing, occluded, or uncertain must not be written as completed. Details such as tools, colors, orientation, identity, text, and object attributes should only be written when clearly visible or explicitly confirmed by Memory; if uncertain, describe conservatively or omit.

The final output must contain only the specified XML tags. Do not output Markdown, explanations, annotations, code blocks, or extra text.
The body of the <note> tag must be empty; it should reference the corresponding frame only through the t attribute. Do not write descriptive text inside <note>.
Output format:
<state>...</state>
<answer>...</answer>
<bridge t="t1-t2">...</bridge>
<note t="t2-t3"></note>
<bridge t="t3-t4">...</bridge>
...
"""


SYNTHESIS_EXTRA = """\
When used for the synthesis stage, if retry feedback is provided, apply the listed corrections and generate a completely new full XML answer. Do not explain.
"""


INFERENCE_PROMPT = """\
[STREAM_AGENT]
You are a streaming video agent that maintains an Interleaved Memory to understand a video stream, and you specifically answer questions in the QA History section.
In QA History, role="q" indicates a user question, and role="a" indicates your previous answer.

- <note>: **visual anchor**. Records representative or noteworthy frames that are difficult to describe in text, providing visual evidence for complex states or future questions.
- <bridge>: **semantic link**. Describes the evolution between visual anchors, using text to describe changes between frames so that the whole process can be reconstructed even without images.

Task instructions:
1. **State and answer**:
   - Before writing <answer>, first write a brief <state> that summarizes the relevant Memory and current frames. Memory contains the video history observed so far, while current frames are the immediate video window; both are sorted by timestamp. Use both together to understand the video, but keep it as concise as possible. Check QA History to determine whether a question should be answered now. If you believe the question can be answered based on the video Memory or current frames, write the answer in <answer>. If there is no active question, or if you believe the question cannot currently be answered, keep <answer></answer> empty.
   - When answering a question, follow the format of the question. The question may be multiple choice; in that case, answer with the option. If it cannot be answered, choose the "cannot answer" option if one exists.
   - **Context analysis**: Carefully inspect the "QA History" log and determine the appropriate evidence scope based on the question itself. As the video progresses, a question may require multiple updates. If the question involves a currently ongoing action, state, or spatial relationship, the judgment should primarily rely on the latest frames, with historical memory used only as background context. If the question involves a past process, object location, state change, or cumulative result, the judgment should primarily rely on notes and bridges in Memory, supplemented by current frames.
   - **Silence criterion**: If there is no active question in QA History, or if the current frames do not provide a useful update for a question that has already been answered, keep <answer> empty.
   - **State is not memory**: <state> is only used for the judgment in the current round. It will not be written into Memory and will not be retained in later rounds. Any facts, locations, text, state changes, object attributes, or event outcomes that may be needed later must be preserved through <bridge> or <note>; do not write them only in <state>.

2. **Note anchoring strategy**:
   - **Forced initialization**: If Memory is <empty/>, the first observation tag must be a <note> for the first frame in the window.
   - **Anchoring principle**: <note> is used to preserve key visual evidence. If a frame carries information that cannot be reliably preserved by bridge alone, create a <note>.
   - **State-change anchoring**: When the scene, camera viewpoint, work-area layout, task phase, key object location, or visible state changes significantly, and that change may affect later understanding, create a <note>.
   - **Representative selection**: If multiple frames in the current window satisfy the anchoring condition, select only the frame that best represents the current state, best supports QA, or best preserves key visual evidence. Do not create multiple notes for subtle movements or posture changes.
   - **Transcription**: When outputting a note, copy the time range from the "Current frames" block into the `t` attribute to ensure precise anchoring.

3. **Bridge compression and inheritance**:
   - **First determine the tail type**: Before outputting observation tags, inspect the last observation tag in Memory. If the last tag is <bridge t="A-B">...</bridge> and there is no note after it, treat it as an open tail. If the last tag is a note, there is no open tail. If Memory is <empty/>, first output the note for the first frame in the window according to the initialization rule.
   - **Open-tail inheritance (critical)**: If an open tail exists, the first output bridge must inherit the original bridge's start time A and serve as a replacement-style update of the original bridge. Do not repeat the old bridge, and do not start a new bridge from the old bridge's end time B. If there is no new note in the current window, write it as <bridge t="A-C">...</bridge>, where C is the end time of the current window. If there is a new note in the current window, write it as <bridge t="A-N">...</bridge>, where N is the start time of the first current note, then output that note.
   - **Non-open-tail start point**: If there is no open tail, new output starts from the end time of the last observation tag in Memory; if that time is earlier than the current window start, start from the current window start. Do not overwrite, repeat, or rewrite time spans in Memory that have already been closed.
   - **Gap uniqueness**: Between any two adjacent notes, or between the final note and the end boundary of the current window, maintain exactly one bridge. Do not split the same time gap into multiple consecutive bridges; even if there is little change, use one brief bridge to describe the continuation of the state.
   - **Time alignment**: Except for open-tail inheritance, all output tags must be ordered chronologically, with continuous time, no overlap, and no backward movement. The note's t attribute must exactly copy the time range of the corresponding frame in Current frames; the start and end times of bridges should connect adjacent notes or the current window boundary.
   - **Content principles**: A bridge only describes the main observable actions, state changes, and event boundaries within that time span, accurately and concisely, without adding uncertain details. For repeated actions, preserve the cumulative result of completed actions; actions that are in preparation, ongoing, occluded, or uncertain must not be written as completed. Details such as tools, colors, orientation, identity, text, and object attributes should only be written when clearly visible or explicitly confirmed by Memory; if uncertain, describe conservatively or omit.

=== Memory ===
{memory_content}

=== Current frames ===
{frame_content}

=== QA History ===
{qa_content}

The final output must contain only the specified XML tags. Do not output Markdown, explanations, annotations, code blocks, or extra text.
The body of the <note> tag must be empty; it should reference the corresponding frame only through the t attribute. Do not write descriptive text inside <note>.
Output format:
<state>...</state>
<answer>...</answer>
<bridge t="t1-t2">...</bridge>
<note t="t2-t3"></note>
<bridge t="t3-t4">...</bridge>
...
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
        content.append(ContentItem("text", text=f'<frame t="{frame.start_time:.1f}-{frame.end_time:.1f}">'))
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
