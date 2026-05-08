"""Stateful StreamWeave environment."""

from __future__ import annotations

from dataclasses import replace

from .memory import MemoryStore
from .policies import MemoryPolicy
from .postprocess import apply_raw_action, repair_for_execution
from .prompts import PromptContext, build_prompt, content_image_paths, content_to_text
from .quality import QualityContext, score_raw_output
from .schemas import AppliedAction, ContentItem, FrameRef, ModelAction, QualityReport, QARecord


class StreamWeaveEnv:
    def __init__(
        self,
        *,
        prompt_profile: str,
        policy: MemoryPolicy,
        memory_window: float = 120.0,
        extra_context: str = "",
    ) -> None:
        self.prompt_profile = prompt_profile
        self.policy = policy
        self.memory = MemoryStore(memory_window=memory_window)
        self.extra_context = extra_context

    def add_question(self, qa: QARecord) -> None:
        self.memory.add_qa(qa)

    def memory_text(self) -> str:
        return self.memory.dump_text()

    def evict_memory(self, current_time: float) -> None:
        self.memory.evict(current_time)

    def build_prompt(
        self,
        frames: list[FrameRef],
        *,
        retry_feedback: str = "",
        extra_context: str = "",
    ) -> tuple[list[ContentItem], str, list[str], list[FrameRef]]:
        local_frames = _with_local_ids(frames)
        context = PromptContext(
            memory_content=self.memory.build_memory_content(self.policy),
            qa_text=self.memory.build_qa_text(),
            frames=local_frames,
            retry_feedback=retry_feedback,
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
        applied = repair_for_execution(raw_output, context) if repair else apply_raw_action(raw_action, context)
        if applied.answer is not None and not self.memory.has_unanswered_question():
            applied = replace(
                applied,
                action=replace(applied.action, answer=""),
                answer=None,
                repair_count=applied.repair_count + 1,
                repair_types=[*applied.repair_types, "drop_answer_without_active_question"],
            )
        return raw_action, quality, applied

    def quality_context(self, frames: list[FrameRef]) -> QualityContext:
        local_frames = _with_local_ids(frames)
        context = QualityContext(
            frames=local_frames,
            step_start=local_frames[0].start_time if local_frames else 0.0,
            step_end=local_frames[-1].end_time if local_frames else 0.0,
            open_tail_bridge=self.memory.open_tail_bridge() if self.policy.use_open_tail else None,
        )
        return context

    def commit(self, applied: AppliedAction) -> None:
        bridge_start = 0
        if self.policy.commit_bridges and applied.replace_open_tail and applied.bridges:
            open_tail = self.memory.open_tail_bridge()
            if open_tail is not None:
                self.memory.replace_bridge(open_tail, applied.bridges[0])
                bridge_start = 1
        if self.policy.commit_bridges:
            for bridge in applied.bridges[bridge_start:]:
                self.memory.add_bridge(bridge)
        if self.policy.commit_notes:
            for note in applied.notes:
                self.memory.add_note(note)
        if applied.answer is not None:
            self.memory.add_qa(applied.answer)


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
