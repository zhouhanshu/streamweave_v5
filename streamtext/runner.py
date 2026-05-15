"""Common StreamText evaluation runner."""

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
from evaluation import ovo_adapter, streamingbench_adapter
from evaluation.runner import classify_error
from streamweave.config import EvalConfig, load_eval_config
from streamweave.frame_store import FrameStore

from streamtext.rollout import StreamTextRunner


STREAMTEXT_PROMPT_PROFILES = {
    "text_memory",
    "text_memory_eval",
    "streamtext",
    "streamtext_eval",
}


def run_eval(
    cfg: EvalConfig,
    *,
    output_path: Path,
    limit: int = 0,
    print_steps: bool = False,
) -> list[dict[str, Any]]:
    ensure_streamtext_config(cfg)
    samples = load_samples(cfg)
    if limit:
        samples = samples[:limit]
    backend = create_backend(cfg.backend, cfg.runtime)
    frame_store = FrameStore(cfg.dataset)
    runner = StreamTextRunner(
        backend=backend,
        frame_store=frame_store,
        runtime=cfg.runtime,
        trace_config=cfg.trace,
        dataset_name=cfg.dataset.dataset_name or cfg.benchmark,
        prompt_profile=cfg.prompt.profile,
        postprocess_config=cfg.postprocess,
        reward_config=cfg.reward,
        memory_config=cfg.memory,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            result = _run_one(cfg.benchmark, runner, index, sample, print_steps=print_steps)
            result.update({"policy": "streamtext", "prompt_type": cfg.prompt.profile})
            results.append(result)
            _write_result(handle, result, index + 1, len(samples))
    write_summary(cfg.benchmark, results, output_path)
    return results


def _run_one(
    benchmark: str,
    runner: StreamTextRunner,
    index: int,
    sample: Any,
    *,
    print_steps: bool = False,
) -> dict[str, Any]:
    try:
        trace = runner.run_sample(
            sample,
            step_start_callback=_print_step_start if print_steps else None,
            step_done_callback=_print_transition_step if print_steps else None,
        )
        result = result_from_trace(benchmark, trace)
    except Exception as exc:
        result = error_result(benchmark, sample, repr(exc))
    result["index"] = index
    return result


def _write_result(handle: Any, result: dict[str, Any], completed: int, total: int) -> None:
    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    handle.flush()
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


def score_trace(benchmark: str, trace: Any) -> dict[str, Any]:
    if benchmark == "ovo":
        return ovo_adapter.score_trace(trace)
    if benchmark == "streamingbench":
        return streamingbench_adapter.score_trace(trace)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def result_from_trace(benchmark: str, trace: Any) -> dict[str, Any]:
    result = score_trace(benchmark, trace)
    if trace.task_failed:
        result["error"] = trace.failure_reason or "task_failed"
        result["error_type"] = "task_failed"
    else:
        result["error"] = ""
        result["error_type"] = ""
    return result


def _print_step_start(step_index: int, frames: list[Any]) -> None:
    if frames:
        start = frames[0].start_time
        end = frames[-1].end_time
        print(f"--- StreamText step index={step_index} t={start:.1f}-{end:.1f} start ---", flush=True)
    else:
        print(f"--- StreamText step index={step_index} start ---", flush=True)


def _print_transition_step(transition: Any) -> None:
    print(
        (
            f"--- StreamText step sample={transition.sample_id} index={transition.step_index} "
            f"t={transition.step_start:.1f}-{transition.step_end:.1f} raw ---"
        ),
        flush=True,
    )
    print(transition.backend_result.text, flush=True)
    print(
        (
            f"--- StreamText step sample={transition.sample_id} index={transition.step_index} "
            "applied ---"
        ),
        flush=True,
    )
    print(_action_to_text(transition.applied.action), flush=True)


def _action_to_text(action: Any) -> str:
    lines = [
        f"<state>{getattr(action, 'state', '')}</state>",
        f"<answer>{getattr(action, 'answer', '')}</answer>",
    ]
    for event in getattr(action, "events", []) or []:
        if event.kind == "note":
            lines.append(f'<anchor t="{event.start_time:.1f}-{event.end_time:.1f}"></anchor>')
        elif event.kind == "bridge":
            lines.append(f'<delta t="{event.start_time:.1f}-{event.end_time:.1f}">{event.text}</delta>')
    return "\n".join(lines)


def write_summary(benchmark: str, results: list[dict[str, Any]], output_path: Path) -> None:
    if benchmark == "ovo":
        _, table, summary_path, table_path = ovo_adapter.write_summary_files(results, output_path)
        print(table, flush=True)
        print(f"Saved OVO summary to {summary_path} and {table_path}", flush=True)
        return
    if benchmark == "streamingbench":
        summary = streamingbench_adapter.summarize_results(results)
        path = output_path.with_name(f"{output_path.stem}_summary.json")
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved summary to {path}", flush=True)


def error_result(benchmark: str, sample: Any, error: str) -> dict[str, Any]:
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
    if args.max_steps:
        cfg.runtime.max_steps = args.max_steps
    ensure_streamtext_config(cfg)
    if args.output:
        output_path = Path(args.output)
    elif cfg.result_output:
        output_path = Path(cfg.result_output)
    else:
        output_path = Path(cfg.trace.output_root) / cfg.trace.experiment_name / f"{cfg.benchmark}_results.jsonl"
    run_eval(cfg, output_path=output_path, limit=args.limit, print_steps=args.print_steps)


def ensure_streamtext_config(cfg: EvalConfig) -> None:
    was_streamtext = cfg.policy.lower() == "streamtext"
    cfg.policy = "streamtext"
    _ensure_streamtext_prompt(cfg)
    if not was_streamtext:
        _redirect_default_outputs(cfg)


def _ensure_streamtext_prompt(cfg: EvalConfig) -> None:
    if cfg.prompt.profile.lower() in STREAMTEXT_PROMPT_PROFILES:
        return
    print(
        f"StreamText overrides prompt profile {cfg.prompt.profile!r} -> 'text_memory_eval'",
        flush=True,
    )
    cfg.prompt.profile = "text_memory_eval"


def _redirect_default_outputs(cfg: EvalConfig) -> None:
    cfg.result_output = _streamtext_output_path(cfg.result_output)
    cfg.batch.output = _streamtext_output_path(cfg.batch.output)
    cfg.batch.worker_log_dir = _streamtext_output_path(cfg.batch.worker_log_dir)
    cfg.trace.output_root = _streamtext_output_path(cfg.trace.output_root)


def _streamtext_output_path(value: str) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return value
    parts = path.parts
    if not parts or any(part.lower() == "streamtext" for part in parts):
        return value
    if parts[0] == "outputs":
        return str(Path("outputs") / "streamtext" / Path(*parts[1:]))
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=("ovo", "streamingbench"), default="")
    parser.add_argument("--backend", choices=("mock", "openai", "vllm", "openai_compatible", "local", "gemini"), default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="debug only: stop each sample after N stream steps")
    parser.add_argument("--output", default="")
    parser.add_argument("--print-steps", action="store_true", help="print each step raw and applied output")
    return parser.parse_args()


if __name__ == "__main__":
    run_cli()
