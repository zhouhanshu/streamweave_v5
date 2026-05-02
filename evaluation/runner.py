"""Common evaluation runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.factory import create_backend
from streamweave.config import EvalConfig, load_eval_config
from streamweave.frame_store import FrameStore
from streamweave.rollout import RolloutRunner

from evaluation import ovo_adapter, streamingbench_adapter


def run_eval(cfg: EvalConfig, *, output_path: Path, limit: int = 0) -> list[dict[str, Any]]:
    samples = load_samples(cfg)
    if limit:
        samples = samples[:limit]
    backend = create_backend(cfg.backend, cfg.runtime)
    frame_store = FrameStore(cfg.dataset)
    runner = RolloutRunner(
        backend=backend,
        frame_store=frame_store,
        runtime=cfg.runtime,
        trace_config=cfg.trace,
        dataset_name=cfg.dataset.dataset_name or cfg.benchmark,
        prompt_profile=cfg.prompt.profile,
        policy=cfg.policy,
        postprocess_config=cfg.postprocess,
        reward_config=cfg.reward,
        synthesis_config=cfg.synthesis,
        memory_config=cfg.memory,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as f:
        for index, sample in enumerate(samples):
            result = _run_one(cfg.benchmark, runner, index, sample)
            result.update({"policy": cfg.policy, "prompt_type": cfg.prompt.profile})
            results.append(result)
            _write_result(f, result, index + 1, len(samples))
    write_summary(cfg.benchmark, results, output_path)
    return results


def _run_one(benchmark: str, runner: RolloutRunner, index: int, sample) -> dict[str, Any]:
    try:
        trace = runner.run_sample(sample)
        result = result_from_trace(benchmark, trace)
    except Exception as exc:
        result = error_result(benchmark, sample, repr(exc))
    result["index"] = index
    return result


def result_from_trace(benchmark: str, trace) -> dict[str, Any]:
    result = score_trace(benchmark, trace)
    if trace.task_failed:
        result["error"] = trace.failure_reason or "task_failed"
        result["error_type"] = "task_failed"
    else:
        result["error"] = ""
        result["error_type"] = ""
    return result


def _write_result(f, result: dict[str, Any], completed: int, total: int) -> None:
    f.write(json.dumps(result, ensure_ascii=False) + "\n")
    f.flush()
    print(
        f"[{completed}/{total}] sample={result.get('sample_id')} score={result.get('score')} error={bool(result.get('error'))}",
        flush=True,
    )


def load_samples(cfg: EvalConfig):
    if cfg.benchmark == "ovo":
        return ovo_adapter.load_samples(cfg.benchmark_args)
    if cfg.benchmark == "streamingbench":
        return streamingbench_adapter.load_samples(cfg.benchmark_args)
    raise ValueError(f"Unknown benchmark: {cfg.benchmark}")


def score_trace(benchmark: str, trace):
    if benchmark == "ovo":
        return ovo_adapter.score_trace(trace)
    if benchmark == "streamingbench":
        return streamingbench_adapter.score_trace(trace)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def write_summary(benchmark: str, results: list[dict[str, Any]], output_path: Path) -> None:
    if benchmark == "ovo":
        _, table, summary_path, table_path = ovo_adapter.write_summary_files(results, output_path)
        print(table, flush=True)
        print(f"Saved OVO summary to {summary_path} and {table_path}", flush=True)
        return
    elif benchmark == "streamingbench":
        summary = streamingbench_adapter.summarize_results(results)
        path = output_path.with_name(f"{output_path.stem}_summary.json")
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        return
    print(f"Saved summary to {path}", flush=True)


def error_result(benchmark: str, sample, error: str) -> dict[str, Any]:
    meta = sample.metadata
    result = {
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "response": "",
        "ground_truth": meta.get("ground_truth"),
        "score": 0,
        "num_steps": 0,
        "task_failed": True,
        "failure_reason": error,
        "error": error,
        "error_type": classify_error(error),
    }
    if benchmark == "ovo":
        result.update(
            {
                "annotation_id": meta.get("annotation_id", sample.sample_id),
                "test_index": meta.get("test_index"),
                "category": meta.get("category", ""),
                "task": meta.get("task", ""),
            }
        )
    elif benchmark == "streamingbench":
        question = meta.get("question", {})
        result.update({"task": question.get("task_type", "") if isinstance(question, dict) else ""})
    return result


def classify_error(error: str) -> str:
    text = str(error)
    non_retryable_markers = (
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
        "safety",
        "Safety",
    )
    if any(marker in text for marker in non_retryable_markers):
        return "non_retryable_backend_error"
    retryable_markers = (
        "Timeout",
        "ReadTimeout",
        "ConnectionError",
        "429",
        "500",
        "502",
        "503",
        "504",
        "DEADLINE_EXCEEDED",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
    )
    if any(marker in text for marker in retryable_markers):
        return "retryable_backend_error"
    return "sample_error"


def run_cli(default_benchmark: str | None = None) -> None:
    args = parse_args()
    cfg = load_eval_config(args.config)
    if args.benchmark:
        cfg.benchmark = args.benchmark
    elif default_benchmark:
        cfg.benchmark = default_benchmark
    if args.backend:
        cfg.backend.backend = args.backend
    if args.model:
        cfg.backend.model = args.model
    if args.endpoint:
        cfg.backend.base_url = args.endpoint
        cfg.backend.endpoints = []
    if args.output:
        output_path = Path(args.output)
    elif cfg.result_output:
        output_path = Path(cfg.result_output)
    else:
        output_path = Path(cfg.trace.output_root) / cfg.trace.experiment_name / f"{cfg.benchmark}_results.jsonl"
    run_eval(cfg, output_path=output_path, limit=args.limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=("ovo", "streamingbench"), default="")
    parser.add_argument("--backend", choices=("mock", "openai", "vllm", "openai_compatible", "local", "gemini"), default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run_cli()
