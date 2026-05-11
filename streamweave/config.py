"""Configuration loading for StreamWeave V5."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RuntimeConfig:
    sample_fps: float = 1.0
    frames_per_step: int = 5
    max_frames: int = 0
    max_steps: int = 0
    resolution: int = 768


@dataclass(slots=True)
class PromptConfig:
    profile: str = "teacher_eval"


@dataclass(slots=True)
class PostprocessConfig:
    mode: str = "eval_repair"


@dataclass(slots=True)
class RewardConfig:
    enable_format_reward: bool = True
    enable_timing_reward: bool = True
    enable_open_tail_reward: bool = True


@dataclass(slots=True)
class SynthesisConfig:
    max_attempts: int = 3


@dataclass(slots=True)
class MemoryConfig:
    window_seconds: float = 120.0


@dataclass(slots=True)
class BatchConfig:
    workers: int = 0
    endpoints: list[str] = field(default_factory=list)
    limit: int = 0
    output: str = ""
    worker_log_dir: str = ""


@dataclass(slots=True)
class DatasetConfig:
    dataset_root: str = "dataset"
    dataset_name: str = "default"
    video_root: str = ""
    frame_id_base: int = 0
    image_ext: str = "jpg"
    jpeg_quality: int = 95
    overwrite_frames: bool = False


@dataclass(slots=True)
class BackendConfig:
    backend: str = "mock"
    model: str = "mock"
    base_url: str = ""
    endpoints: list[str] = field(default_factory=list)
    api_key: str = ""
    api_key_env: str = ""
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.1
    thinking_budget: int | None = None
    timeout_seconds: float = 120.0
    image_quality: int = 85
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    retryable_error_patterns: list[str] = field(
        default_factory=lambda: [
            "Timeout",
            "ReadTimeout",
            "ConnectionError",
            "temporarily unavailable",
            "429",
            "500",
            "502",
            "503",
            "504",
            "DEADLINE_EXCEEDED",
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
        ]
    )
    non_retryable_error_patterns: list[str] = field(
        default_factory=lambda: [
            "400",
            "BadRequest",
            "bad request",
            "401",
            "403",
            "Unauthorized",
            "Forbidden",
            "invalid_request",
            "context_length",
            "maximum context",
            "image too large",
            "FinishReason.SAFETY",
        ]
    )

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""


@dataclass(slots=True)
class TraceConfig:
    output_root: str = "output"
    experiment_name: str = "debug"
    write_jsonl: bool = True


@dataclass(slots=True)
class EvalConfig:
    benchmark: str = "ovo"
    policy: str = "streamweave"
    result_output: str = ""
    prompt: PromptConfig = field(default_factory=PromptConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    benchmark_args: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    cfg_path = Path(path)
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install pyyaml or use JSON config files.") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def eval_config_from_dict(data: dict[str, Any]) -> EvalConfig:
    runtime = RuntimeConfig(**(data.get("runtime") or {}))
    dataset = DatasetConfig(**(data.get("dataset") or {}))
    backend_data = dict(data.get("backend") or {})
    if isinstance(backend_data.get("endpoints"), str):
        backend_data["endpoints"] = [
            item.strip() for item in backend_data["endpoints"].split(",") if item.strip()
        ]
    backend = BackendConfig(**backend_data)
    trace = TraceConfig(**(data.get("trace") or {}))
    prompt_data = dict(data.get("prompt") or {})
    if not prompt_data and (data.get("prompt_profile") or data.get("prompt_type")):
        prompt_data["profile"] = data.get("prompt_profile", data.get("prompt_type"))
    batch_data = dict(data.get("batch") or {})
    if isinstance(batch_data.get("endpoints"), str):
        batch_data["endpoints"] = [
            item.strip() for item in batch_data["endpoints"].split(",") if item.strip()
        ]
    return EvalConfig(
        benchmark=str(data.get("benchmark", "ovo")),
        policy=str(data.get("policy", "streamweave")),
        result_output=str(data.get("result_output", "")),
        prompt=PromptConfig(**prompt_data),
        postprocess=PostprocessConfig(**(data.get("postprocess") or {})),
        reward=RewardConfig(**(data.get("reward") or {})),
        synthesis=SynthesisConfig(**(data.get("synthesis") or {})),
        memory=MemoryConfig(**(data.get("memory") or {})),
        batch=BatchConfig(**batch_data),
        runtime=runtime,
        dataset=dataset,
        backend=backend,
        trace=trace,
        benchmark_args=dict(data.get("benchmark_args") or {}),
    )


def load_eval_config(path: str | os.PathLike[str]) -> EvalConfig:
    return eval_config_from_dict(load_config(path))
