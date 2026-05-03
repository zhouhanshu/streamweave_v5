"""Stepwise StreamWeave agent loop for verl."""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from .env import StreamWeaveRLEnv

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("streamweave_agent")
class StreamWeaveAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        request_id = uuid4().hex
        group_idx = kwargs.get("group_idx", kwargs.get("uid", request_id))
        traj_idx = int(kwargs.get("traj_idx", kwargs.get("rollout_n", 0)) or 0)
        env = StreamWeaveRLEnv(
            sample_id=str(kwargs["sample_id"]),
            video_id=str(kwargs["video_id"]),
            video_path=str(kwargs.get("video_path", "")),
            question=str(kwargs.get("question", "")),
            query_timestamp=float(kwargs.get("query_timestamp", 0.0) or 0.0),
            ground_truth=str(kwargs.get("ground_truth", "")),
            config=dict(kwargs.get("streamweave_config", {}) or {}),
        )

        outputs: list[AgentLoopOutput] = []
        try:
            obs, _ = await env.reset(seed=int(kwargs.get("seed", 0) or 0))
            while True:
                output, response_text = await self._generate_turn(
                    obs=obs,
                    sampling_params=sampling_params,
                    request_id=request_id,
                )
                next_obs, reward, done, info = await env.step(response_text)
                turn_idx = int(info.get("turn_idx", len(outputs)))

                output.reward_score = float(reward)
                output.num_turns = 1
                output.extra_fields.update(
                    {
                        "group_idx": group_idx,
                        "traj_idx": traj_idx,
                        "turn_idx": turn_idx,
                        "last_turn": bool(info.get("last_turn", done)),
                        "format_score": float(info.get("format_score", 0.0)),
                        "step_score": float(info.get("step_score", 0.0)),
                        "success_score": float(info.get("success_score", 0.0)),
                        "trajectory_score": float(info.get("trajectory_score", 0.0)),
                        "turn_reward": float(info.get("turn_reward", reward)),
                        "final_answer": info.get("final_answer", ""),
                        "reward_extra_info": {
                            "format_score": float(info.get("format_score", 0.0)),
                            "step_score": float(info.get("step_score", 0.0)),
                            "success_score": float(info.get("success_score", 0.0)),
                            "trajectory_score": float(info.get("trajectory_score", 0.0)),
                        },
                    }
                )
                outputs.append(output)

                if done:
                    trajectory_score = float(info.get("trajectory_score", 0.0))
                    success_score = float(info.get("success_score", 0.0))
                    final_answer = info.get("final_answer", "")
                    for item in outputs:
                        item.extra_fields["trajectory_score"] = trajectory_score
                        item.extra_fields["success_score"] = success_score
                        item.extra_fields["final_answer"] = final_answer
                        item.extra_fields["reward_extra_info"]["trajectory_score"] = trajectory_score
                        item.extra_fields["reward_extra_info"]["success_score"] = success_score
                    break
                obs = next_obs
        except _TrajectoryAbort as exc:
            logger.warning("Aborting StreamWeave trajectory: %s", exc.reason)
            outputs.append(self._build_abort_output(exc.reason, prompt_ids=exc.prompt_ids))
            _finalize_aborted_outputs(outputs, group_idx=group_idx, traj_idx=traj_idx, reason=exc.reason)
        except Exception as exc:
            logger.exception("StreamWeave trajectory failed; returning a zero-reward fallback.")
            outputs.append(self._build_abort_output(f"{type(exc).__name__}: {exc}"))
            _finalize_aborted_outputs(outputs, group_idx=group_idx, traj_idx=traj_idx, reason=str(exc))
        finally:
            await env.close()
        return outputs

    async def _generate_turn(
        self,
        *,
        obs: dict[str, Any],
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> tuple[AgentLoopOutput, str]:
        messages = obs.get("messages", [])
        images = obs.get("images", []) or None
        if images and self.processor is None:
            raise ValueError("StreamWeave returned images but the model processor is not available.")

        prompt_ids = await self.apply_chat_template(messages, images=images)
        max_prompt_tokens = self._max_prompt_tokens()
        if max_prompt_tokens is not None and len(prompt_ids) > max_prompt_tokens:
            raise _TrajectoryAbort(
                f"prompt too long: {len(prompt_ids)} tokens > {max_prompt_tokens}",
                prompt_ids=prompt_ids,
            )
        turn_sampling_params = dict(sampling_params)
        max_new_tokens = int(turn_sampling_params.get("max_new_tokens") or self.response_length)
        turn_sampling_params["max_new_tokens"] = min(max_new_tokens, self.response_length)

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            token_output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=turn_sampling_params,
                image_data=images,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1

        truncated_prompt_ids = prompt_ids[-self.prompt_length :]
        response_ids = token_output.token_ids[: self.response_length]
        response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True),
        )
        output = AgentLoopOutput(
            prompt_ids=truncated_prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=token_output.log_probs[: self.response_length] if token_output.log_probs else None,
            routed_experts=_align_routed_experts(
                token_output.routed_experts,
                original_prompt_len=len(prompt_ids),
                kept_prompt_len=len(truncated_prompt_ids),
                response_len=len(response_ids),
            ),
            multi_modal_data={"images": images} if images else {},
            reward_score=None,
            num_turns=1,
            metrics=metrics,
            extra_fields=dict(token_output.extra_fields or {}),
        )
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output, response_text

    def _max_prompt_tokens(self) -> int | None:
        max_model_len = getattr(self.rollout_config, "max_model_len", None)
        if max_model_len is None:
            model_max_length = getattr(self.tokenizer, "model_max_length", None)
            if isinstance(model_max_length, int) and 0 < model_max_length < 10**8:
                max_model_len = model_max_length
        if max_model_len is None:
            return None
        return max(1, int(max_model_len) - int(self.response_length))

    def _build_abort_output(self, reason: str, prompt_ids: list[int] | None = None) -> AgentLoopOutput:
        prompt = (prompt_ids or _encode_text(self.tokenizer, "[StreamWeave rollout aborted]"))[-self.prompt_length :]
        response = _encode_text(self.tokenizer, "<eta></eta><answer></answer>")[: max(1, self.response_length)]
        if not response:
            response = [self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0]
        return AgentLoopOutput(
            prompt_ids=prompt,
            response_ids=response,
            response_mask=[0] * len(response),
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data={},
            reward_score=0.0,
            num_turns=1,
            metrics={"generate_sequences": 0.0, "tool_calls": 0.0, "num_preempted": -1},
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "rollout_error": reason,
            },
        )


def _align_routed_experts(
    routed_experts: Any,
    *,
    original_prompt_len: int,
    kept_prompt_len: int,
    response_len: int,
) -> Any:
    if routed_experts is None:
        return None
    target_len = kept_prompt_len + response_len
    total_len = _first_dim_len(routed_experts)
    if total_len is None:
        return routed_experts
    if total_len <= response_len:
        response_part = routed_experts[:response_len]
        return _prepend_routed_padding(response_part, kept_prompt_len)
    if total_len >= original_prompt_len + response_len:
        start = max(0, original_prompt_len - kept_prompt_len)
        return routed_experts[start : start + target_len]
    return routed_experts[: min(total_len, target_len)]


def _first_dim_len(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    try:
        return len(value)
    except TypeError:
        return None


class _TrajectoryAbort(Exception):
    def __init__(self, reason: str, prompt_ids: list[int] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.prompt_ids = prompt_ids


def _prepend_routed_padding(routed_experts: Any, prefix_len: int) -> Any:
    if prefix_len <= 0:
        return routed_experts
    if hasattr(routed_experts, "new_zeros"):
        prefix = routed_experts.new_zeros((prefix_len, *routed_experts.shape[1:]))
        import torch

        return torch.cat([prefix, routed_experts], dim=0)
    shape = getattr(routed_experts, "shape", None)
    if shape is not None and len(shape) > 0:
        import numpy as np

        prefix = np.zeros((prefix_len, *shape[1:]), dtype=routed_experts.dtype)
        return np.concatenate([prefix, routed_experts], axis=0)
    return None


def _encode_text(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    return list(tokenizer(text, add_special_tokens=False).get("input_ids", []))


def _finalize_aborted_outputs(
    outputs: list[AgentLoopOutput],
    *,
    group_idx: Any,
    traj_idx: int,
    reason: str,
) -> None:
    for turn_idx, item in enumerate(outputs, start=1):
        item.reward_score = 0.0
        item.num_turns = 1
        reward_info = {
            "format_score": 0.0,
            "step_score": 0.0,
            "success_score": 0.0,
            "trajectory_score": 0.0,
        }
        item.extra_fields.update(
            {
                "group_idx": group_idx,
                "traj_idx": traj_idx,
                "turn_idx": int(item.extra_fields.get("turn_idx", turn_idx) or turn_idx),
                "last_turn": item is outputs[-1],
                "format_score": 0.0,
                "step_score": 0.0,
                "success_score": 0.0,
                "trajectory_score": 0.0,
                "turn_reward": 0.0,
                "final_answer": "",
                "rollout_error": reason,
                "reward_extra_info": reward_info,
            }
        )
