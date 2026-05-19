"""Stateful StreamWeave environment used by verl agent rollout."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from streamweave.config import DatasetConfig, MemoryConfig, RewardConfig, RuntimeConfig
from streamweave.env import StreamWeaveEnv
from streamweave.frame_store import FrameStore
from streamweave.policies import make_policy
from streamweave.rollout import (
    _group_frames,
    _query_events_by_frame,
    _timestamp_to_frame_id,
    _truncate_frames_at_timestamp,
    required_frame_count_from_metadata,
)
from streamweave.schemas import BenchmarkSample, ContentItem, FrameRef, QARecord, QueryEvent

from .judge import GRPPO_JUDGE_PROMPT_VERSION, GRPPO_STEP_SCORE_KEYS, JudgeResult, StepJudge
from .rewards import (
    StreamWeaveRewardConfig,
    compute_grppo_step_reward,
    compute_grppo_target_trajectory_reward,
    compute_note_frequency_score,
    compute_step_reward,
    compute_trajectory_reward,
    judge_blocked_by_note_frequency,
    reward_config_from_mapping,
)
from .scorers import score_answer
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


@dataclass(slots=True)
class GrppoAnswerSupervision:
    kind: str = "none"
    label: dict[str, Any] | None = None
    status: str = "no_answer_supervision"

    @property
    def enabled(self) -> bool:
        return self.kind != "none"


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
        self.query_annotations = _normalize_query_annotations(
            metadata=metadata,
            question=question,
            query_timestamp=query_timestamp,
            ground_truth=ground_truth,
        )
        query_events = [
            QueryEvent(text=str(annotation.get("content") or ""), timestamp=float(annotation.get("timestamp", 0.0) or 0.0))
            for annotation in self.query_annotations
            if annotation.get("event_type") == "query" and str(annotation.get("content") or "").strip()
        ]
        self.sample = BenchmarkSample(
            sample_id=sample_id,
            video_id=video_id,
            video_path=video_path,
            query_events=query_events,
            metadata=metadata,
        )
        self.ground_truth = ground_truth
        self.settings = _settings_from_mapping(config or {})
        self.grppo_enabled = self.settings.rl_reward.judge.prompt_version == GRPPO_JUDGE_PROMPT_VERSION
        self.frame_store = FrameStore(self.settings.dataset)
        self.policy = make_policy(self.settings.policy)
        self.env: StreamWeaveEnv | None = None
        self.frames: list[FrameRef] = []
        self.groups: list[list[FrameRef]] = []
        self.query_by_frame: dict[int, list[QueryEvent]] = {}
        self.query_annotations_by_frame: dict[int, list[dict[str, Any]]] = {}
        self.step_idx = 0
        self.current_frames: list[FrameRef] = []
        self.current_prompt_text = ""
        self.current_prompt_images: list[str] = []
        self.current_memory_before = ""
        self.current_query_event: dict[str, Any] | None = None
        self.current_answer_target: dict[str, Any] | None = None
        self.current_answer_label: dict[str, Any] | None = None
        self.current_answer_supervision = GrppoAnswerSupervision()
        self.latest_query_event: dict[str, Any] | None = None
        self.current_step_query_count = 0
        self.current_step_answer_target_count = 0
        self.step_rewards: list[StepRewardResult] = []
        self.records: list[StepRecord] = []
        self.trajectory_reward: TrajectoryRewardResult | None = None
        self.target_answer_scores: list[float] = []
        self.no_note_streak = 0
        self.judge = StepJudge(self.settings.rl_reward.judge)

    async def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed
        runtime = self.settings.runtime
        required_count = required_frame_count_from_metadata(self.sample.metadata, self.sample.sample_id)
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
        self.frames = self.frames[:required_count]
        if len(self.frames) < required_count:
            raise RuntimeError(
                f"Sample {self.sample.sample_id!r} declared frame_count={required_count} "
                f"but only {len(self.frames)} frame(s) are available on disk."
            )
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
        self.query_annotations_by_frame = _query_annotations_by_frame(
            self.frames,
            self.query_annotations,
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
        self.target_answer_scores = []
        self.no_note_streak = 0
        self.current_query_event = None
        self.current_answer_target = None
        self.current_answer_label = None
        self.current_answer_supervision = GrppoAnswerSupervision()
        self.latest_query_event = None
        self.current_step_query_count = 0
        self.current_step_answer_target_count = 0
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
        grppo_has_answer = bool(raw_action.answer.strip())
        grppo_has_query = bool(self.current_query_event is not None)
        grppo_has_answer_target = bool(self.current_answer_target is not None)
        if _uses_timeline_answer_supervision(self.settings.rl_reward):
            grppo_supervision = _timeline_grppo_answer_supervision(
                grppo_enabled=self.grppo_enabled,
                current_query_event=self.current_query_event,
                current_answer_target=self.current_answer_target,
                latest_query_event=self.latest_query_event,
                has_answer=grppo_has_answer,
            )
            self.current_answer_supervision = grppo_supervision
        else:
            grppo_supervision = _legacy_grppo_answer_supervision(
                grppo_enabled=self.grppo_enabled,
                cfg=self.settings.rl_reward,
                label=self.current_answer_label,
                has_query=grppo_has_query,
                has_answer_target=grppo_has_answer_target,
                has_answer=grppo_has_answer,
            )
        grppo_label = grppo_supervision.label
        grppo_answer_event = grppo_supervision.enabled
        if grppo_answer_event and not quality.parser_ok:
            grppo_answer_correctness = 0.0
            grppo_label_status = "parser_invalid"
        elif grppo_answer_event:
            grppo_answer_correctness, grppo_label_status = _score_current_step_answer(
                answer=raw_action.answer,
                label=grppo_label,
                metadata=self.sample.metadata,
                cfg=self.settings.rl_reward,
            )
        else:
            grppo_answer_correctness = 0.0
            grppo_label_status = grppo_supervision.status
        target_step_answer_required = (
            self.current_answer_target is not None and _grppo_label_requires_answer(self.current_answer_target)
        )
        target_step_answer_rule_score = 0.0
        if target_step_answer_required:
            if not quality.parser_ok:
                target_step_answer_rule_score = 0.0
            else:
                target_step_answer_rule_score, _ = _score_current_step_answer(
                    answer=raw_action.answer,
                    label=self.current_answer_target,
                    metadata=self.sample.metadata,
                    cfg=self.settings.rl_reward,
                )
        note_frequency_result = compute_note_frequency_score(
            action=raw_action,
            previous_no_note_streak=self.no_note_streak,
            cfg=self.settings.rl_reward,
        )
        note_frequency_score, _next_no_note_streak, note_frequency_reasons = note_frequency_result
        judge_blocked = judge_blocked_by_note_frequency(
            note_frequency_score=note_frequency_score,
            cfg=self.settings.rl_reward,
        )
        judge_allowed = self.grppo_enabled or not judge_blocked
        if judge_allowed:
            qa_history_before = self.env.memory.build_qa_text()
            judge_result = await self.judge.score_step(
                memory_before=self.current_memory_before,
                qa_history=qa_history_before,
                frames=self.current_frames,
                raw_action=raw_action,
                raw_output=action_str,
                quality=quality,
                query_label=grppo_label if self.grppo_enabled else None,
                has_query=grppo_has_query,
                has_answer=grppo_has_answer,
                answer_reward_event=grppo_answer_event,
                answer_correctness=grppo_answer_correctness if grppo_answer_event else None,
            )
        else:
            judge_result = JudgeResult(
                score=0.0,
                status="blocked_note_frequency",
                issues=list(note_frequency_reasons),
            )
        if target_step_answer_required:
            target_step_answer_score, _target_step_answer_source = _target_answer_score_from_judge_or_rule(
                rule_score=target_step_answer_rule_score,
                judge_result=judge_result,
                grppo_enabled=self.grppo_enabled,
                answer_reward_event=grppo_answer_event,
                answer_supervision_kind=grppo_supervision.kind,
            )
            self.target_answer_scores.append(float(target_step_answer_score))
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
            target_label = _trajectory_answer_label(annotations=self.query_annotations)
            target_ground_truth = _label_ground_truth(target_label) if target_label is not None else self.ground_truth
            target_answer_count = len(self.target_answer_scores)
            target_answer_reward = (
                sum(self.target_answer_scores) / target_answer_count
                if target_answer_count > 0
                else 0.0
            )
            target_trajectory_score = compute_grppo_target_trajectory_reward(
                answer_score=target_answer_reward,
                format_score=self.trajectory_reward.format_mean,
                cfg=self.settings.rl_reward,
            )
            if not self.grppo_enabled:
                turn_reward += self.settings.rl_reward.w_success * self.trajectory_reward.success_score
            reward_result.turn_reward = turn_reward
            trajectory_info = {
                "trajectory_score": self.trajectory_reward.trajectory_score,
                "format_mean": self.trajectory_reward.format_mean,
                "step_mean": self.trajectory_reward.step_mean,
                "success_score": self.trajectory_reward.success_score,
                "final_answer": final_answer,
                "grppo_target_trajectory_score": target_trajectory_score,
                "grppo_target_answer_reward": target_answer_reward,
                "grppo_target_format_reward": self.trajectory_reward.format_mean,
                "grppo_target_answer_count": float(target_answer_count),
                "grppo_target_ground_truth": "" if target_ground_truth is None else str(target_ground_truth),
            }

        grppo_info: dict[str, Any] = {}
        if self.grppo_enabled:
            judge_scores = judge_result.scores if isinstance(judge_result, JudgeResult) else {}
            grppo_dims = {key: float(judge_scores.get(key, 0.0) or 0.0) for key in GRPPO_STEP_SCORE_KEYS}
            grppo_judge_step_reward = (
                float(judge_result.score)
                if isinstance(judge_result, JudgeResult)
                else sum(grppo_dims.values()) / max(len(grppo_dims), 1)
            )
            grppo_step_reward = compute_grppo_step_reward(
                process_score=grppo_judge_step_reward,
                format_score=reward_result.format_score,
                note_frequency_score=reward_result.note_frequency_score,
                cfg=self.settings.rl_reward,
            )
            grppo_answer_reward_raw = float(judge_scores.get("answer_reward", 0.0) or 0.0) if grppo_answer_event else 0.0
            grppo_answer_reward_raw = min(max(grppo_answer_reward_raw, 0.0), 1.0)
            grppo_answer_reward = _final_grppo_answer_reward(
                raw_score=grppo_answer_reward_raw,
                supervision_kind=grppo_supervision.kind,
                has_answer=grppo_has_answer,
                label_status=grppo_label_status,
                use_llm_for_silence=_uses_timeline_answer_supervision(self.settings.rl_reward),
                silence_reward=self.settings.rl_reward.grppo_silence_reward,
                silence_reward_value=self.settings.rl_reward.grppo_silence_reward_value,
            )
            grppo_info = {
                "grppo_enabled": True,
                "grppo_delta_groundedness": grppo_dims["delta_groundedness"],
                "grppo_anchor_keyframe": grppo_dims["anchor_keyframe"],
                "grppo_semantic_alignment": grppo_dims["semantic_alignment"],
                "grppo_state_groundedness": grppo_dims["state_groundedness"],
                "grppo_judge_step_reward": grppo_judge_step_reward,
                "grppo_format_score": reward_result.format_score,
                "grppo_note_frequency_score": reward_result.note_frequency_score,
                "grppo_step_reward": grppo_step_reward,
                "grppo_answer_reward": grppo_answer_reward,
                "grppo_answer_reward_raw": grppo_answer_reward_raw,
                "grppo_answer_event": float(grppo_answer_event),
                "grppo_answer_supervision": float(_answer_supervision_code(grppo_supervision.kind)),
                "grppo_answer_reward_scale": _answer_reward_scale(
                    grppo_supervision.kind,
                    silence_reward=self.settings.rl_reward.grppo_silence_reward,
                    silence_reward_value=self.settings.rl_reward.grppo_silence_reward_value,
                ),
                "grppo_has_query": float(grppo_has_query),
                "grppo_has_answer_target": float(grppo_has_answer_target),
                "grppo_has_answer": float(grppo_has_answer),
                "grppo_answer_correctness": float(grppo_answer_correctness),
                "grppo_query_count": float(self.current_step_query_count),
                "grppo_answer_target_count": float(self.current_step_answer_target_count),
                "grppo_prompt_kind": "answer_aware" if grppo_answer_event else "process",
                "grppo_label_status": grppo_label_status,
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
            **grppo_info,
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
        current_labels: list[dict[str, Any]] = []
        for frame in group:
            for query in self.query_by_frame.get(frame.global_index, []):
                self.env.add_question(QARecord(timestamp=query.timestamp, text=query.text, role="q"))
            current_labels.extend(self.query_annotations_by_frame.get(frame.global_index, []))
        query_events = [item for item in current_labels if item.get("event_type") == "query"]
        answer_targets = [item for item in current_labels if item.get("event_type") == "answer_target"]
        self.current_step_query_count = len(query_events)
        self.current_step_answer_target_count = len(answer_targets)
        if len(query_events) > 1:
            raise RuntimeError(
                f"Sample {self.sample.sample_id!r} has {len(query_events)} query events in one step; "
                "GRPPO expects at most one query per step."
            )
        if len(answer_targets) > 1:
            raise RuntimeError(
                f"Sample {self.sample.sample_id!r} has {len(answer_targets)} answer targets in one step; "
                "filter the data so at most one answer target lands in each step."
            )
        self.current_query_event = query_events[0] if query_events else None
        self.current_answer_target = answer_targets[0] if answer_targets else None
        if self.current_query_event is not None:
            self.latest_query_event = self.current_query_event
        self.current_answer_supervision = _timeline_grppo_answer_supervision(
            grppo_enabled=self.grppo_enabled,
            current_query_event=self.current_query_event,
            current_answer_target=self.current_answer_target,
            latest_query_event=self.latest_query_event,
            has_answer=False,
        )
        self.current_answer_label = (
            self.current_answer_supervision.label
            if _uses_timeline_answer_supervision(self.settings.rl_reward)
            else self.current_answer_target
            if self.current_answer_target is not None
            else self.current_query_event
        )

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


def _normalize_query_annotations(
    *,
    metadata: dict[str, Any],
    question: Any,
    query_timestamp: Any,
    ground_truth: Any,
) -> list[dict[str, Any]]:
    del question, query_timestamp, ground_truth
    query_events = metadata.get("query_events")
    if not isinstance(query_events, list) or not query_events:
        raise ValueError("StreamWeave RL requires canonical query_events; legacy query/question fields are not accepted.")
    annotations: list[dict[str, Any]] = []
    for query_index, item in enumerate(query_events):
        if not isinstance(item, Mapping):
            raise ValueError(f"query_events[{query_index}] must be an object")
        qid = str(item.get("qid") or "").strip()
        content_text = str(item.get("content") or "").strip()
        question_text = str(item.get("question") or content_text).strip()
        if not qid:
            raise ValueError(f"query_events[{query_index}].qid is required")
        if not content_text:
            raise ValueError(f"query_events[{query_index}].content is required")
        timestamp = _coerce_float(item.get("time"), default=0.0)
        answer_type = _canonical_answer_type(item)
        query_annotation = {
            "event_type": "query",
            "qid": qid,
            "timestamp": timestamp,
            "time": timestamp,
            "question": question_text,
            "content": content_text,
            "answer_type": answer_type,
            "answer_policy": item.get("answer_policy", ""),
            "dataset": metadata.get("dataset", ""),
            "benchmark": metadata.get("benchmark", ""),
            "scorer": "mcq" if answer_type == "mcq" else "text",
        }
        if answer_type == "mcq":
            query_annotation["options"] = _canonical_options(item.get("options"))
        annotations.append(
            query_annotation
        )
        answer_events = item.get("answer_events")
        if not isinstance(answer_events, list):
            raise ValueError(f"query_events[{query_index}].answer_events must be a list")
        for answer_index, answer_item in enumerate(answer_events):
            if not isinstance(answer_item, Mapping):
                raise ValueError(f"query_events[{query_index}].answer_events[{answer_index}] must be an object")
            annotations.append(
                _answer_target_annotation(
                    answer_item,
                    qid=qid,
                    question=question_text,
                    answer_type=answer_type,
                    options=query_annotation.get("options", []),
                    default_id=f"{qid}:a{answer_index}",
                )
            )
    return sorted(annotations, key=_annotation_sort_key)


def _answer_target_annotation(
    item: Mapping[str, Any],
    *,
    qid: str,
    question: str,
    answer_type: str,
    options: Any,
    default_id: str,
) -> dict[str, Any]:
    timestamp = _coerce_float(item.get("time"), default=0.0)
    answer_text = str(item.get("answer") or item.get("content") or "").strip()
    gt = str(item.get("gt") or answer_text).strip()
    annotation = {
        "event_type": "answer_target",
        "qid": qid,
        "timestamp": timestamp,
        "time": timestamp,
        "question": question,
        "content": str(item.get("content") or answer_text).strip(),
        "answer": answer_text,
        "ground_truth": gt,
        "gt": gt,
        "answer_type": answer_type,
        "answer_event_id": str(item.get("id") or default_id),
        "should_answer": item.get("should_answer", True),
        "scorer": "mcq" if answer_type == "mcq" else "text",
    }
    if answer_type == "mcq":
        annotation["options"] = _canonical_options(options)
    return annotation


def _canonical_answer_type(query: Mapping[str, Any]) -> str:
    explicit = str(query.get("answer_type") or "").strip().lower()
    if explicit in {"mcq", "multiple_choice", "multiple-choice"}:
        return "mcq"
    if explicit in {"text", "freeform", "natural_language", "natural-language"}:
        return "text"
    raise ValueError("query_events[].answer_type is required and must be 'mcq' or 'text'")


def _canonical_options(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _query_annotations_by_frame(
    frames: list[FrameRef],
    annotations: list[dict[str, Any]],
    *,
    sample_fps: float,
    frame_id_base: int,
) -> dict[int, list[dict[str, Any]]]:
    if not frames:
        return {}
    min_frame_id = min(frame.global_index for frame in frames)
    max_frame_id = max(frame.global_index for frame in frames)
    out: dict[int, list[dict[str, Any]]] = {}
    for annotation in sorted(annotations, key=_annotation_sort_key):
        frame_id = _timestamp_to_frame_id(
            float(annotation.get("timestamp", 0.0) or 0.0),
            sample_fps=sample_fps,
            frame_id_base=frame_id_base,
            min_frame_id=min_frame_id,
            max_frame_id=max_frame_id,
        )
        out.setdefault(frame_id, []).append(annotation)
    return out


def _timeline_grppo_answer_supervision(
    *,
    grppo_enabled: bool,
    current_query_event: dict[str, Any] | None,
    current_answer_target: dict[str, Any] | None,
    latest_query_event: dict[str, Any] | None,
    has_answer: bool = False,
) -> GrppoAnswerSupervision:
    if not grppo_enabled:
        return GrppoAnswerSupervision()
    if current_answer_target is not None:
        if _grppo_label_requires_answer(current_answer_target):
            return GrppoAnswerSupervision(kind="answer", label=current_answer_target, status="answer_target")
        return GrppoAnswerSupervision(
            kind="silence",
            label=_silence_supervision_label(current_answer_target),
            status="answer_target_without_target",
        )
    if current_query_event is not None:
        if _grppo_label_requires_answer(current_query_event):
            return GrppoAnswerSupervision(kind="answer", label=current_query_event, status="query_answer")
        return GrppoAnswerSupervision(
            kind="silence",
            label=_silence_supervision_label(current_query_event),
            status="query_silence",
        )
    if has_answer:
        return GrppoAnswerSupervision(
            kind="silence",
            label=_unprompted_answer_silence_label(latest_query_event),
            status="unprompted_answer",
        )
    return GrppoAnswerSupervision()


def _legacy_grppo_answer_supervision(
    *,
    grppo_enabled: bool,
    cfg: StreamWeaveRewardConfig,
    label: dict[str, Any] | None,
    has_query: bool,
    has_answer_target: bool,
    has_answer: bool,
) -> GrppoAnswerSupervision:
    if not grppo_enabled:
        return GrppoAnswerSupervision()
    mode = str(cfg.grppo_answer_event_mode or "legacy").strip().lower()
    if mode in {"timeline", "schedule", "scheduled"}:
        raise ValueError("timeline GRPPO answer supervision must be prepared before model output is scored")
    if mode in {"required_only", "required", "target_only"}:
        if _grppo_label_requires_answer(label):
            return GrppoAnswerSupervision(kind="answer", label=label, status="answer_target")
        return GrppoAnswerSupervision()
    if mode in {"legacy", "answer_or_label", "any"}:
        if not bool(has_query or has_answer_target or has_answer):
            return GrppoAnswerSupervision()
        if _grppo_label_requires_answer(label):
            return GrppoAnswerSupervision(kind="answer", label=label, status="answer_label")
        status = "missing_label" if label is None else "legacy_silence"
        return GrppoAnswerSupervision(kind="silence", label=label, status=status)
    raise ValueError(f"Unknown GRPPO answer event mode: {cfg.grppo_answer_event_mode}")


def _uses_timeline_answer_supervision(cfg: StreamWeaveRewardConfig) -> bool:
    return str(cfg.grppo_answer_event_mode or "").strip().lower() in {"timeline", "schedule", "scheduled"}


def _silence_supervision_label(label: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(label)
    out["event_type"] = "answer_silence"
    out["should_answer"] = False
    return out


def _unprompted_answer_silence_label(latest_query_event: Mapping[str, Any] | None) -> dict[str, Any]:
    if latest_query_event is not None:
        return _silence_supervision_label(latest_query_event)
    return {"event_type": "answer_silence", "should_answer": False}


def _score_current_step_answer(
    *,
    answer: str,
    label: dict[str, Any] | None,
    metadata: dict[str, Any],
    cfg: StreamWeaveRewardConfig,
) -> tuple[float, str]:
    answer_text = str(answer or "").strip()
    if label is None:
        return 0.0, "missing_label"

    should_answer = label.get("should_answer")
    if should_answer is not None and not _bool_from_value(should_answer):
        return _binary_silence_score(answer_text), "expected_silence"

    ground_truth = _label_ground_truth(label)
    if ground_truth is None:
        if label.get("event_type") == "query":
            return _binary_silence_score(answer_text), "query_without_target"
        return 0.0, "missing_ground_truth"

    merged_metadata = {**metadata, **label, "ground_truth": ground_truth}
    scorer = str(label.get("scorer") or cfg.success_scorer or "auto")
    try:
        score = score_answer(answer_text, ground_truth, scorer=scorer, metadata=merged_metadata)
    except Exception as exc:
        return 0.0, f"scorer_error:{type(exc).__name__}"
    return min(max(float(score), 0.0), 1.0), "scored"


def _target_answer_score_from_judge_or_rule(
    *,
    rule_score: float,
    judge_result: JudgeResult,
    grppo_enabled: bool,
    answer_reward_event: bool,
    answer_supervision_kind: str,
) -> tuple[float, str]:
    """Use answer judge for target answer metrics when it is available.

    The rule scorer is kept only as a fallback for disabled/failed judge calls.
    """

    if grppo_enabled and answer_reward_event and answer_supervision_kind == "answer":
        scores = judge_result.scores if isinstance(judge_result, JudgeResult) else {}
        if judge_result.status == "ok" and "answer_reward" in scores:
            return _clamp_unit(scores.get("answer_reward", 0.0)), "judge"
    return _clamp_unit(rule_score), "rule"


def _clamp_unit(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return min(max(score, 0.0), 1.0)


def _binary_silence_score(answer_text: str) -> float:
    return 1.0 if not str(answer_text or "").strip() else 0.0


GRPPO_ANSWER_ATTEMPT_REWARD_OFFSET = 0.1


def _final_grppo_answer_reward(
    *,
    raw_score: float,
    supervision_kind: str,
    has_answer: bool,
    label_status: str,
    use_llm_for_silence: bool = False,
    silence_reward: bool = True,
    silence_reward_value: float = 1.0,
) -> float:
    """Convert the judge answer score into the scalar consumed by GRPPO."""
    kind = str(supervision_kind or "none")
    if kind == "none":
        return 0.0
    if str(label_status or "") == "parser_invalid":
        return 0.0
    binary_score = 1.0 if float(raw_score) >= 0.5 else 0.0
    if kind == "silence":
        if not _bool_from_value(silence_reward):
            return 0.0
        if has_answer:
            return 0.0
        if use_llm_for_silence:
            return float(silence_reward_value) * binary_score
        return float(silence_reward_value)
    if kind == "answer":
        if not has_answer:
            return 0.0
        attempt = (
            float(silence_reward_value) + GRPPO_ANSWER_ATTEMPT_REWARD_OFFSET
            if _bool_from_value(silence_reward)
            else 0.0
        )
        return attempt + binary_score
    raise ValueError(f"Unknown GRPPO answer supervision kind: {supervision_kind}")


def _answer_reward_scale(
    kind: str,
    *,
    silence_reward: bool = True,
    silence_reward_value: float = 1.0,
) -> float:
    if kind == "answer":
        attempt = (
            float(silence_reward_value) + GRPPO_ANSWER_ATTEMPT_REWARD_OFFSET
            if _bool_from_value(silence_reward)
            else 0.0
        )
        return 1.0 + attempt
    if kind == "silence" and _bool_from_value(silence_reward):
        return float(silence_reward_value)
    return 0.0


def _answer_supervision_code(kind: str) -> int:
    return {"none": 0, "silence": 1, "answer": 2}.get(str(kind or "none"), 0)


def _grppo_answer_event_enabled(
    *,
    grppo_enabled: bool,
    cfg: StreamWeaveRewardConfig,
    label: dict[str, Any] | None,
    has_query: bool,
    has_answer_target: bool,
    has_answer: bool,
) -> bool:
    return _legacy_grppo_answer_supervision(
        grppo_enabled=grppo_enabled,
        cfg=cfg,
        label=label,
        has_query=has_query,
        has_answer_target=has_answer_target,
        has_answer=has_answer,
    ).enabled


def _grppo_label_requires_answer(label: Mapping[str, Any] | None) -> bool:
    if label is None:
        return False
    should_answer = label.get("should_answer")
    if should_answer is not None and not _bool_from_value(should_answer):
        return False
    return _label_ground_truth(label) is not None


def _trajectory_answer_label(*, annotations: list[dict[str, Any]]) -> dict[str, Any] | None:
    answer_targets = [item for item in annotations if item.get("event_type") == "answer_target"]
    for item in sorted(answer_targets, key=_annotation_sort_key, reverse=True):
        if _label_ground_truth(item) is not None:
            return item
    for item in sorted(annotations, key=_annotation_sort_key, reverse=True):
        if _label_ground_truth(item) is not None:
            return item
    return None


def _label_ground_truth(label: Mapping[str, Any]) -> Any:
    for key in ("ground_truth", "gt", "target", "target_answer"):
        value = label.get(key)
        if value is not None and str(value).strip():
            return value
    if label.get("event_type") == "answer_target":
        for key in ("answer", "content"):
            value = label.get(key)
            if value is not None and str(value).strip():
                return value
    value = label.get("answer")
    if value is not None and str(value).strip():
        return value
    return None


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _annotation_sort_key(annotation: Mapping[str, Any]) -> tuple[float, int]:
    event_type = str(annotation.get("event_type") or "")
    order = 0 if event_type == "query" else 1 if event_type == "answer_target" else 2
    return float(annotation.get("timestamp", 0.0) or 0.0), order


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
