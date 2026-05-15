"""Stateful text-memory environment for StreamText."""

from __future__ import annotations

from dataclasses import replace

from streamweave.memory import MemoryStore
from streamweave.schemas import AppliedAction, ContentItem, FrameRef, ModelAction, QARecord, QualityReport

from .postprocess import apply_raw_action, repair_for_execution
from .prompts import PromptContext, build_prompt, content_image_paths, content_to_text
from .quality import TextQualityContext, score_raw_output


class StreamTextEnv:
    def __init__(
        self,
        *,
        prompt_profile: str,
        memory_window: float = 120.0,
        extra_context: str = "",
    ) -> None:
        self.prompt_profile = prompt_profile
        self.memory = MemoryStore(memory_window=memory_window)
        self.extra_context = extra_context

    def add_question(self, qa: QARecord) -> None:
        self.memory.add_qa(qa)

    def memory_text(self) -> str:
        return self.memory.dump_text()

    def evict_memory(self, current_time: float) -> None:
        del current_time
        return None

    def build_prompt(
        self,
        frames: list[FrameRef],
        *,
        extra_context: str = "",
    ) -> tuple[list[ContentItem], str, list[str], list[FrameRef]]:
        local_frames = _with_local_ids(frames)
        context = PromptContext(
            memory_text=self._build_delta_memory_text(),
            qa_text=self.memory.build_qa_text(),
            frames=local_frames,
            extra_context=_join_context(self.extra_context, extra_context),
        )
        content = build_prompt(self.prompt_profile, context)
        return content, content_to_text(content), content_image_paths(content), local_frames

    def evaluate_attempt(
        self,
        raw_output: str,
        *,
        frames: list[FrameRef],
        reward_config: object | None = None,
        repair: bool = True,
    ) -> tuple[ModelAction, QualityReport, AppliedAction]:
        local_frames = _with_local_ids(frames)
        context = self.quality_context(local_frames)
        raw_action, quality = score_raw_output(raw_output, context, reward_config=reward_config)
        raw_deltas = [event for event in raw_action.events if event.kind == "bridge"]
        use_raw = (not repair) and quality.valid and len(raw_deltas) == 1
        applied = apply_raw_action(raw_action, context) if use_raw else repair_for_execution(raw_output, context)
        if applied.answer is not None and not self.memory.has_question():
            applied = replace(
                applied,
                action=replace(applied.action, answer=""),
                answer=None,
                repair_count=applied.repair_count + 1,
                repair_types=[*applied.repair_types, "drop_answer_without_question"],
            )
        return raw_action, quality, applied

    def quality_context(self, frames: list[FrameRef]) -> TextQualityContext:
        return TextQualityContext(
            frames=frames,
            step_start=frames[0].start_time if frames else 0.0,
            step_end=frames[-1].end_time if frames else 0.0,
        )

    def commit(self, applied: AppliedAction) -> None:
        for bridge in applied.bridges:
            self.memory.add_bridge(bridge)
        if applied.answer is not None:
            self.memory.add_qa(applied.answer)

    def _build_delta_memory_text(self) -> str:
        if not self.memory.bridges:
            return "<empty/>"
        lines = []
        for bridge in sorted(self.memory.bridges, key=lambda item: (item.start_time, item.end_time)):
            lines.append(f'<delta t="{bridge.start_time:.1f}-{bridge.end_time:.1f}">{bridge.text}</delta>')
        return "\n".join(lines)


def _with_local_ids(frames: list[FrameRef]) -> list[FrameRef]:
    return [
        FrameRef(
            video_id=frame.video_id,
            global_index=frame.global_index,
            start_time=frame.start_time,
            end_time=frame.end_time,
            image_path=frame.image_path,
            step_local_id=index,
        )
        for index, frame in enumerate(frames, start=1)
    ]


def _join_context(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())
