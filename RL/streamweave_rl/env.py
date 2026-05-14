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
            QueryEvent(text=str(annotation.get("question") or ""), timestamp=float(annotation.get("timestamp", 0.0) or 0.0))
            for annotation in self.query_annotations
            if annotation.get("event_type") == "query" and str(annotation.get("question") or "").strip()
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
        self.current_step_query_count = 0
        self.current_step_answer_target_count = 0
        self.step_rewards: list[StepRewardResult] = []
        self.records: list[StepRecord] = []
        self.trajectory_reward: TrajectoryRewardResult | None = None
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
        self.no_note_streak = 0
        self.current_query_event = None
        self.current_answer_target = None
        self.current_answer_label = None
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
        grppo_answer_event = bool(
            self.grppo_enabled and (grppo_has_query or grppo_has_answer_target or grppo_has_answer)
        )
        grppo_label = self.current_answer_label
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
            grppo_label_status = "no_answer_event"
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
            if not self.grppo_enabled:
                turn_reward += self.settings.rl_reward.w_success * self.trajectory_reward.success_score
            reward_result.turn_reward = turn_reward
            trajectory_info = {
                "trajectory_score": self.trajectory_reward.trajectory_score,
                "format_mean": self.trajectory_reward.format_mean,
                "step_mean": self.trajectory_reward.step_mean,
                "success_score": self.trajectory_reward.success_score,
                "final_answer": final_answer,
            }

        grppo_info: dict[str, Any] = {}
        if self.grppo_enabled:
            judge_scores = judge_result.scores if isinstance(judge_result, JudgeResult) else {}
            grppo_dims = {key: float(judge_scores.get(key, 0.0) or 0.0) for key in GRPPO_STEP_SCORE_KEYS}
            grppo_judge_step_reward = sum(grppo_dims.values()) / max(len(grppo_dims), 1)
            grppo_step_reward = compute_grppo_step_reward(
                process_score=grppo_judge_step_reward,
                format_score=reward_result.format_score,
                cfg=self.settings.rl_reward,
            )
            grppo_answer_reward_raw = float(judge_scores.get("answer_reward", 0.0) or 0.0) if grppo_answer_event else 0.0
            grppo_answer_reward_raw = min(max(grppo_answer_reward_raw, 0.0), 1.0)
            grppo_answer_reward = _final_grppo_answer_reward(
                raw_score=grppo_answer_reward_raw,
                has_answer=grppo_has_answer,
                label_status=grppo_label_status,
            )
            grppo_info = {
                "grppo_enabled": True,
                "grppo_delta_groundedness": grppo_dims["delta_groundedness"],
                "grppo_anchor_keyframe": grppo_dims["anchor_keyframe"],
                "grppo_semantic_alignment": grppo_dims["semantic_alignment"],
                "grppo_state_groundedness": grppo_dims["state_groundedness"],
                "grppo_judge_step_reward": grppo_judge_step_reward,
                "grppo_format_score": reward_result.format_score,
                "grppo_step_reward": grppo_step_reward,
                "grppo_answer_reward": grppo_answer_reward,
                "grppo_answer_reward_raw": grppo_answer_reward_raw,
                "grppo_answer_event": float(grppo_answer_event),
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
        self.current_answer_label = (
            self.current_answer_target
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
    if isinstance(metadata.get("query_events"), list):
        return _normalize_structured_query_events(metadata["query_events"], metadata=metadata, ground_truth=ground_truth)
    if isinstance(metadata.get("queries"), list):
        return _normalize_structured_query_events(metadata["queries"], metadata=metadata, ground_truth=ground_truth)

    question_value = metadata.get("raw_question", metadata.get("question", question))
    response_value = metadata.get("response")
    if isinstance(question_value, list):
        annotations = _normalize_question_response_lists(
            question_value,
            response_value if isinstance(response_value, list) else [],
            metadata=metadata,
            ground_truth=ground_truth,
        )
        if annotations:
            return annotations

    text = str(question or question_value or "").strip()
    if not text:
        return []
    annotation = _base_annotation(
        event_type="query",
        qid=str(metadata.get("qid") or "q0"),
        timestamp=_coerce_float(metadata.get("query_timestamp", query_timestamp), default=0.0),
        question=text,
        source={**metadata, "ground_truth": ground_truth},
    )
    return [annotation]


def _normalize_structured_query_events(
    query_events: list[Any],
    *,
    metadata: dict[str, Any],
    ground_truth: Any,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for index, item in enumerate(query_events):
        if not isinstance(item, Mapping):
            continue
        qid = str(item.get("qid") or item.get("id") or f"q{index}")
        question_text = str(item.get("question") or item.get("query") or item.get("content") or item.get("text") or "").strip()
        timestamp = _coerce_float(item.get("timestamp", item.get("time", item.get("query_timestamp", 0.0))), default=0.0)
        merged = {**metadata, **dict(item), "ground_truth": item.get("ground_truth", item.get("gt", ground_truth))}
        if question_text:
            annotations.append(
                _base_annotation(
                    event_type="query",
                    qid=qid,
                    timestamp=timestamp,
                    question=question_text,
                    source=merged,
                )
            )
        answer_events = item.get("answer_events", item.get("target_answers", []))
        if isinstance(answer_events, list):
            for answer_index, answer_item in enumerate(answer_events):
                if not isinstance(answer_item, Mapping):
                    continue
                annotations.append(
                    _answer_target_annotation(
                        answer_item,
                        qid=qid,
                        question=question_text,
                        default_timestamp=timestamp,
                        default_id=f"{qid}:a{answer_index}",
                        parent=merged,
                    )
                )
    return sorted(annotations, key=_annotation_sort_key)


def _normalize_question_response_lists(
    questions: list[Any],
    responses: list[Any],
    *,
    metadata: dict[str, Any],
    ground_truth: Any,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for index, item in enumerate(questions):
        if isinstance(item, Mapping):
            qid = str(item.get("qid") or item.get("id") or f"q{index}")
            question_text = str(item.get("question") or item.get("query") or item.get("content") or item.get("text") or "").strip()
            timestamp = _coerce_float(item.get("timestamp", item.get("time", 0.0)), default=0.0)
            source = {**metadata, **dict(item), "ground_truth": item.get("ground_truth", item.get("gt", ground_truth))}
        else:
            qid = f"q{index}"
            question_text = str(item or "").strip()
            timestamp = _coerce_float(metadata.get("query_timestamp", 0.0), default=0.0)
            source = {**metadata, "ground_truth": ground_truth}
        if not question_text:
            continue
        annotations.append(
            _base_annotation(
                event_type="query",
                qid=qid,
                timestamp=timestamp,
                question=question_text,
                source=source,
            )
        )

    # Compatibility for the draft shape:
    # {"question": [{"content": "..."}], "response": [{"content": "C. ..."}]}
    parent = annotations[0] if annotations else {"qid": "q0", "question": ""}
    for index, item in enumerate(responses):
        if not isinstance(item, Mapping):
            continue
        annotations.append(
            _answer_target_annotation(
                item,
                qid=str(parent.get("qid") or "q0"),
                question=str(parent.get("question") or ""),
                default_timestamp=float(parent.get("timestamp", 0.0) or 0.0),
                default_id=f"{parent.get('qid', 'q0')}:a{index}",
                parent=metadata,
            )
        )
    return sorted(annotations, key=_annotation_sort_key)


def _base_annotation(
    *,
    event_type: str,
    qid: str,
    timestamp: float,
    question: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    annotation = {
        "event_type": event_type,
        "qid": qid,
        "timestamp": float(timestamp),
        "question": question,
        "content": question,
    }
    for key in (
        "ground_truth",
        "gt",
        "answer",
        "options",
        "task",
        "scorer",
        "should_answer",
        "answer_policy",
        "dataset",
        "benchmark",
    ):
        if key in source and source[key] is not None:
            annotation[key] = source[key]
    return annotation


def _answer_target_annotation(
    item: Mapping[str, Any],
    *,
    qid: str,
    question: str,
    default_timestamp: float,
    default_id: str,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    timestamp = _coerce_float(item.get("timestamp", item.get("time", default_timestamp)), default=default_timestamp)
    content = str(item.get("content") or item.get("text") or item.get("answer") or "").strip()
    annotation = _base_annotation(
        event_type="answer_target",
        qid=qid,
        timestamp=timestamp,
        question=question,
        source={**dict(parent), **dict(item)},
    )
    annotation["answer_event_id"] = str(item.get("id") or default_id)
    annotation["content"] = content
    annotation["should_answer"] = item.get("should_answer", parent.get("should_answer", True))
    if "ground_truth" not in annotation and "gt" not in annotation:
        if item.get("gt") is not None:
            annotation["gt"] = item.get("gt")
        elif item.get("answer") is not None:
            annotation["ground_truth"] = item.get("answer")
        elif content:
            annotation["ground_truth"] = content
    return annotation


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
        return (1.0 if not answer_text else 0.0), "expected_silence"

    ground_truth = _label_ground_truth(label)
    if ground_truth is None:
        if label.get("event_type") == "query":
            return (1.0 if not answer_text else 0.0), "query_without_target"
        return 0.0, "missing_ground_truth"

    merged_metadata = {**metadata, **label, "ground_truth": ground_truth}
    scorer = str(label.get("scorer") or cfg.success_scorer or "auto")
    try:
        score = score_answer(answer_text, ground_truth, scorer=scorer, metadata=merged_metadata)
    except Exception as exc:
        return 0.0, f"scorer_error:{type(exc).__name__}"
    return min(max(float(score), 0.0), 1.0), "scored"


def _final_grppo_answer_reward(*, raw_score: float, has_answer: bool, label_status: str) -> float:
    """Use deterministic timing labels before trusting the LLM answer score."""
    if str(label_status) in {"expected_silence", "query_without_target", "missing_label"}:
        return 0.0 if has_answer else 1.0
    return 1.0 if float(raw_score) >= 0.5 else 0.0


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
