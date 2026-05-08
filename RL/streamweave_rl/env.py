"""Stateful StreamWeave environment used by verl agent rollout."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from streamweave.config import DatasetConfig, MemoryConfig, RewardConfig, RuntimeConfig
from streamweave.env import StreamWeaveEnv
from streamweave.frame_store import FrameStore
from streamweave.policies import make_policy
from streamweave.rollout import _group_frames, _query_events_by_frame, _truncate_frames_at_timestamp
from streamweave.schemas import BenchmarkSample, ContentItem, FrameRef, QARecord, QueryEvent

from .judge import JudgeResult, StepJudge
from .rewards import (
    StreamWeaveRewardConfig,
    compute_note_frequency_score,
    compute_step_reward,
    compute_trajectory_reward,
    judge_blocked_by_note_frequency,
    reward_config_from_mapping,
)
from .schemas import StepRecord, StepRewardResult, TrajectoryRewardResult


@dataclass(slots=True)
class StreamWeaveRLSettings:
    dataset_name: str = "default"
    prompt_profile: str = "eval"
    policy: str = "streamweave"
    extra_context: str = ""
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    rl_reward: StreamWeaveRewardConfig = field(default_factory=StreamWeaveRewardConfig)


class StreamWeaveRLEnv:
    def __init__(
        self,
        *,
        sample_id: str,
        video_id: str,
        video_path: str,
        question: str,
        query_timestamp: float,
        ground_truth: Any,
        sample_metadata: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        metadata = dict(sample_metadata or {})
        metadata.setdefault("dataset", (config or {}).get("dataset_name", ""))
        metadata.setdefault("ground_truth", ground_truth)
        self.sample = BenchmarkSample(
            sample_id=sample_id,
            video_id=video_id,
            video_path=video_path,
            query_events=[QueryEvent(text=question, timestamp=float(query_timestamp))] if question else [],
            metadata=metadata,
        )
        self.ground_truth = ground_truth
        self.settings = _settings_from_mapping(config or {})
        self.frame_store = FrameStore(self.settings.dataset)
        self.policy = make_policy(self.settings.policy)
        self.env: StreamWeaveEnv | None = None
        self.frames: list[FrameRef] = []
        self.groups: list[list[FrameRef]] = []
        self.query_by_frame: dict[int, list[QueryEvent]] = {}
        self.step_idx = 0
        self.current_frames: list[FrameRef] = []
        self.current_prompt_text = ""
        self.current_prompt_images: list[str] = []
        self.current_memory_before = ""
        self.step_rewards: list[StepRewardResult] = []
        self.records: list[StepRecord] = []
        self.trajectory_reward: TrajectoryRewardResult | None = None
        self.no_note_streak = 0
        self.judge = StepJudge(self.settings.rl_reward.judge)

    async def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed
        runtime = self.settings.runtime
        try:
            self.frames = self.frame_store.load_frames(
                dataset_name=self.settings.dataset_name,
                video_id=self.sample.video_id,
                sample_fps=runtime.sample_fps,
                max_frames=runtime.max_frames,
            )
        except FileNotFoundError as exc:
            frame_dir = self.frame_store.frame_dir(self.settings.dataset_name, self.sample.video_id)
            raise FileNotFoundError(
                "StreamWeave RL expects pre-extracted frames. "
                f"Missing frames for video_id={self.sample.video_id!r} under {frame_dir}."
            ) from exc
        self.frames = _truncate_frames_at_timestamp(
            self.frames,
            self.sample.metadata.get("target_timestamp"),
            sample_fps=runtime.sample_fps,
            frame_id_base=self.settings.dataset.frame_id_base,
        )
        self.groups = _group_frames(self.frames, runtime.frames_per_step)
        if runtime.max_steps:
            self.groups = self.groups[: runtime.max_steps]
        if not self.groups:
            raise RuntimeError(f"No frame groups for sample {self.sample.sample_id}")

        self.query_by_frame = _query_events_by_frame(
            self.frames,
            self.sample.query_events,
            sample_fps=runtime.sample_fps,
            frame_id_base=self.settings.dataset.frame_id_base,
        )
        if self.sample.query_events:
            retained_frame_ids = {frame.global_index for group in self.groups for frame in group}
            if not any(frame_id in retained_frame_ids for frame_id in self.query_by_frame):
                retained_end = self.groups[-1][-1].end_time if self.groups and self.groups[-1] else 0.0
                raise RuntimeError(
                    "No query event remains after StreamWeave RL frame truncation. "
                    f"sample_id={self.sample.sample_id!r}, max_steps={runtime.max_steps}, "
                    f"retained_end={retained_end:.1f}s"
                )
        self.env = StreamWeaveEnv(
            prompt_profile=self.settings.prompt_profile,
            policy=self.policy,
            memory_window=self.settings.memory.window_seconds,
            extra_context=self.settings.extra_context,
        )
        self.step_idx = 0
        self.step_rewards = []
        self.records = []
        self.trajectory_reward = None
        self.no_note_streak = 0
        obs = self._prepare_current_turn()
        return obs, {"sample_id": self.sample.sample_id, "video_id": self.sample.video_id}

    async def step(self, action_str: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("StreamWeaveRLEnv.step() called before reset().")

        raw_action, quality, applied = self.env.evaluate_attempt(
            action_str,
            frames=self.current_frames,
            reward_config=self.settings.reward_config,
            repair=True,
        )
        note_frequency_result = compute_note_frequency_score(
            action=raw_action,
            previous_no_note_streak=self.no_note_streak,
            cfg=self.settings.rl_reward,
        )
        note_frequency_score, _next_no_note_streak, note_frequency_reasons = note_frequency_result
        judge_allowed = not judge_blocked_by_note_frequency(
            note_frequency_score=note_frequency_score,
            cfg=self.settings.rl_reward,
        )
        if judge_allowed:
            qa_history_before = self.env.memory.build_qa_text()
            judge_result = await self.judge.score_step(
                memory_before=self.current_memory_before,
                qa_history=qa_history_before,
                frames=self.current_frames,
                raw_action=raw_action,
                raw_output=action_str,
                quality=quality,
            )
        else:
            judge_result = JudgeResult(
                score=0.0,
                status="blocked_note_frequency",
                issues=list(note_frequency_reasons),
            )
        self.env.commit(applied)
        reward_result = compute_step_reward(
            quality=quality,
            cfg=self.settings.rl_reward,
            ctx={
                "sample": self.sample,
                "frames": self.current_frames,
                "action": raw_action,
                "previous_no_note_streak": self.no_note_streak,
                "note_frequency_result": note_frequency_result,
                "judge_result": judge_result,
            },
            total_steps=len(self.groups),
        )
        self.no_note_streak = int(reward_result.info.get("no_note_streak", self.no_note_streak) or 0)
        self.step_rewards.append(reward_result)

        self.step_idx += 1
        done = self.step_idx >= len(self.groups)
        final_answer = self._final_answer()
        trajectory_info: dict[str, Any] = {}
        turn_reward = reward_result.turn_reward
        if done:
            self.trajectory_reward = compute_trajectory_reward(
                step_results=self.step_rewards,
                final_answer=final_answer,
                ground_truth=self.ground_truth,
                cfg=self.settings.rl_reward,
                metadata=self.sample.metadata,
            )
            turn_reward += self.settings.rl_reward.w_success * self.trajectory_reward.success_score
            reward_result.turn_reward = turn_reward
            trajectory_info = {
                "trajectory_score": self.trajectory_reward.trajectory_score,
                "format_mean": self.trajectory_reward.format_mean,
                "step_mean": self.trajectory_reward.step_mean,
                "success_score": self.trajectory_reward.success_score,
                "final_answer": final_answer,
            }

        info = {
            "turn_idx": self.step_idx,
            "last_turn": done,
            "format_score": reward_result.format_score,
            "step_score": reward_result.step_score,
            "note_frequency_score": reward_result.note_frequency_score,
            "judge_score": reward_result.judge_score,
            "turn_reward": turn_reward,
            "quality_valid": quality.valid,
            "parser_ok": quality.parser_ok,
            "issue_codes": [issue.code for issue in quality.issues],
            "reward_info": reward_result.info,
            **trajectory_info,
        }
        self.records.append(
            StepRecord(
                turn_idx=self.step_idx,
                response_text=action_str,
                prompt_text=self.current_prompt_text,
                prompt_images=list(self.current_prompt_images),
                format_score=reward_result.format_score,
                step_score=reward_result.step_score,
                turn_reward=turn_reward,
                done=done,
                info=info,
            )
        )

        if done:
            return {"messages": [], "images": [], "prompt_text": ""}, turn_reward, True, info
        return self._prepare_current_turn(), turn_reward, False, info

    async def close(self) -> None:
        return None

    def _prepare_current_turn(self) -> dict[str, Any]:
        assert self.env is not None
        group = self.groups[self.step_idx]
        for frame in group:
            for query in self.query_by_frame.get(frame.global_index, []):
                self.env.add_question(QARecord(timestamp=query.timestamp, text=query.text, role="q"))

        prompt_frames = self._prompt_frames(group)
        self.env.evict_memory(group[-1].end_time)
        self.current_memory_before = self.env.memory_text()
        content, prompt_text, prompt_images, local_frames = self.env.build_prompt(prompt_frames)
        self.current_frames = local_frames
        self.current_prompt_text = prompt_text
        self.current_prompt_images = prompt_images
        image_resolution = int(getattr(self.settings.runtime, "resolution", 768) or 0)
        messages, images = _content_to_messages_and_images(content, max_side=image_resolution)
        return {"messages": messages, "images": images, "prompt_text": prompt_text, "prompt_images": prompt_images}

    def _prompt_frames(self, group: list[FrameRef]) -> list[FrameRef]:
        if self.policy.use_recent_frames:
            end_position = self.frames.index(group[-1])
            return self.frame_store.recent_frames(self.frames, end_position, count=self.policy.recent_frame_count)
        return group

    def _final_answer(self) -> str:
        if self.env is None:
            return ""
        for qa in reversed(self.env.memory.qa_history):
            if qa.role == "a" and qa.text.strip():
                return qa.text.strip()
        return ""


def _content_to_messages_and_images(
    content: list[ContentItem], *, max_side: int = 768
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    blocks: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    for item in content:
        if item.type == "text":
            if item.text:
                blocks.append({"type": "text", "text": item.text})
        elif item.type == "image":
            if item.image_path is None:
                continue
            image = _load_image(item.image_path, max_side=max_side)
            if image is None:
                continue
            blocks.append({"type": "image"})
            images.append(image)
    return [{"role": "user", "content": blocks}], images


def _load_image(path: str | Path, *, max_side: int = 768) -> Image.Image | None:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if max_side and max(image.size) > max_side:
                image.thumbnail((max_side, max_side))
            return image.copy()
    except (FileNotFoundError, PermissionError, OSError, UnidentifiedImageError) as exc:
        message = f"StreamWeave RL failed to read frame image {path}: {type(exc).__name__}: {exc}"
        print(message, file=sys.stderr, flush=True)
        raise RuntimeError(message) from exc


def _settings_from_mapping(data: dict[str, Any]) -> StreamWeaveRLSettings:
    runtime = _dataclass_from_mapping(RuntimeConfig, data.get("runtime"))
    dataset = _dataclass_from_mapping(DatasetConfig, data.get("dataset"))
    memory = _dataclass_from_mapping(MemoryConfig, data.get("memory"))
    reward_config = _reward_config_from_mapping(data.get("streamweave_reward") or data.get("streamweave_reward_config"))
    rl_reward = reward_config_from_mapping(dict(data.get("reward", {}) or {}))
    return StreamWeaveRLSettings(
        dataset_name=str(data.get("dataset_name", dataset.dataset_name)),
        prompt_profile=str(data.get("prompt_profile", data.get("prompt", "eval"))),
        policy=str(data.get("policy", "streamweave")),
        extra_context=str(data.get("extra_context", "")),
        runtime=runtime,
        dataset=dataset,
        memory=memory,
        reward_config=reward_config,
        rl_reward=rl_reward,
    )


def _dataclass_from_mapping(cls, data: Any):
    mapping = dict(data or {})
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in mapping.items() if key in allowed})


def _reward_config_from_mapping(data: Any) -> RewardConfig:
    mapping = dict(data or {})
    values = {
        "enable_format_reward": True,
        "enable_timing_reward": False,
        "enable_open_tail_reward": False,
    }
    values.update({key: value for key, value in mapping.items() if key in values})
    return RewardConfig(**values)
