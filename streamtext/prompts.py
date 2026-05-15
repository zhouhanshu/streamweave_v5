"""Prompt builders for the StreamText text-memory sidecar."""

from __future__ import annotations

from dataclasses import dataclass

from streamweave.schemas import ContentItem, FrameRef


TEXT_MEMORY_PROMPT = """\
[STREAM_TEXT_AGENT]
You are a streaming video agent. You observe the video in fixed frame windows.
Your long-term Memory is text only: it contains closed <delta> descriptions of
previous frame windows. You do not store images in Memory.

At each step:
- Read Text Memory, Current frames, and QA History.
- Write a concise <state> for the current decision. <state> is not saved.
- Write <answer> only if QA History contains a question that can be answered now.
- When answering, follow the question format. For multiple-choice questions,
  answer with the option letter only unless the question explicitly asks otherwise.
- Always write exactly one <delta> for the current frame window.
- The <delta> must describe the visible content, actions, state changes, objects,
  scene layout, text, and other useful evidence in the current frames.
- Create only the current-window <delta>. Leave previous Text Memory unchanged.

=== Text Memory ===
{memory_content}

=== Current frames ===
{frame_content}

=== QA History ===
{qa_content}

Output XML only. No Markdown, no explanations, no extra text.
Required format:
<state>...</state>
<answer>...</answer>
<delta t="{step_start}-{step_end}">...</delta>
"""


@dataclass(slots=True)
class PromptContext:
    memory_text: str
    qa_text: str
    frames: list[FrameRef]
    extra_context: str = ""


def build_prompt(profile: str, context: PromptContext) -> list[ContentItem]:
    name = profile.lower()
    if name not in {"text_memory", "text_memory_eval", "streamtext", "streamtext_eval"}:
        raise ValueError(f"Unknown StreamText prompt profile: {profile}")

    frame_content = _frame_text(context.frames)
    if context.frames:
        step_start = f"{context.frames[0].start_time:.1f}"
        step_end = f"{context.frames[-1].end_time:.1f}"
    else:
        step_start = "0.0"
        step_end = "0.0"
    prompt = TEXT_MEMORY_PROMPT.format(
        memory_content=context.memory_text or "<empty/>",
        frame_content=frame_content,
        qa_content=context.qa_text or "<empty/>",
        step_start=step_start,
        step_end=step_end,
    )
    if context.extra_context:
        prompt += "\n=== Extra Context ===\n" + context.extra_context.strip() + "\n"
    content: list[ContentItem] = [ContentItem("text", text=prompt)]
    for frame in context.frames:
        content.append(ContentItem("text", text=f'\n[frame t="{frame.start_time:.1f}-{frame.end_time:.1f}"]\n'))
        content.append(ContentItem("image", image_path=frame.image_path))
    return content


def _frame_text(frames: list[FrameRef]) -> str:
    if not frames:
        return "<empty/>"
    return "\n".join(
        f'<frame t="{frame.start_time:.1f}-{frame.end_time:.1f}"><image></frame>'
        for frame in frames
    )


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
