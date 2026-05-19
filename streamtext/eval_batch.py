#!/usr/bin/env python3
"""Multi-process StreamText evaluation entry point."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.factory import create_backend
from evaluation.rollout_metrics import rollout_metrics_from_trace
from streamweave.config import EvalConfig, eval_config_from_dict, load_config
from streamweave.frame_store import FrameStore
from streamweave.schemas import BenchmarkSample, QueryEvent

from streamtext.rollout import StreamTextRunner
from streamtext.runner import (
    error_result,
    ensure_streamtext_config,
    load_samples,
    result_from_trace,
    write_summary,
)


_PROGRESS_QUEUE: Any = None
_TASK_QUEUE: Any = None


def main() -> None:
    args = parse_args()
    config_data = load_config(args.config)
    cfg = eval_config_from_dict(config_data)
    _apply_overrides(cfg, args)
    _apply_loader_limit(cfg, args)
    ensure_streamtext_config(cfg)
    _validate_postprocess_mode(cfg)

    samples = load_samples(cfg)
    limit = args.limit or cfg.batch.limit
    if limit and not _limit_applied_by_loader(cfg):
        samples = samples[:limit]
    expected_sample_ids = _expected_result_sample_ids(samples)

    output_path = Path(
        args.output
        or cfg.batch.output
        or cfg.result_output
        or (Path(cfg.trace.output_root) / cfg.trace.experiment_name / f"{cfg.benchmark}_results.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output_path.parent / f".{output_path.stem}_parts"
    worker_log_dir = Path(
        args.worker_log_dir
        or ((output_path.parent / "worker_logs") if args.output else "")
        or cfg.batch.worker_log_dir
        or output_path.parent / "worker_logs"
    )
    if args.resume:
        shard_dir.mkdir(parents=True, exist_ok=True)
        worker_log_dir.mkdir(parents=True, exist_ok=True)
    else:
        _prepare_run_dir(shard_dir, "part_*.jsonl")
        _prepare_run_dir(worker_log_dir, "worker_*.log")

    endpoints = _configured_endpoints(cfg, args)
    workers = args.workers or cfg.batch.workers or (len(endpoints) if endpoints else 1)
    workers = max(1, workers)
    manifest_path = shard_dir / "run_manifest.json"
    manifest = _build_manifest(cfg, args, samples, output_path, expected_sample_ids)
    if args.resume:
        _validate_manifest(manifest_path, manifest)

    completed_results: dict[int, dict[str, Any]] = {}
    if args.resume:
        completed_results = _load_completed_results(output_path, shard_dir, expected_sample_ids)
        pending_tasks = _pending_task_count(samples, completed_results)
        print(
            f"resume: completed_rows={len(completed_results)} pending_tasks={pending_tasks} "
            f"total_tasks={len(samples)} total_rows={len(expected_sample_ids)}",
            flush=True,
        )
    _write_manifest(manifest_path, manifest)
    pending_indices = [
        idx for idx, sample in enumerate(samples) if not _task_is_completed(idx, sample, completed_results)
    ]
    jobs = []
    for worker_id in range(workers):
        endpoint = endpoints[worker_id % len(endpoints)] if endpoints else ""
        part_path = shard_dir / f"part_{worker_id:03d}.jsonl"
        log_path = worker_log_dir / f"worker_{worker_id:03d}.log"
        jobs.append(
            (
                worker_id,
                config_data,
                vars(args),
                endpoint,
                str(part_path),
                str(log_path),
                bool(args.resume),
            )
        )

    if not samples:
        output_path.write_text("", encoding="utf-8")
        write_summary(cfg.benchmark, [], output_path)
        return
    if not pending_indices:
        results = _merge_parts(
            shard_dir,
            output_path,
            expected_sample_ids=expected_sample_ids,
            include_existing_output=args.resume,
        )
        write_summary(cfg.benchmark, results, output_path)
        print(f"Saved merged results to {output_path}", flush=True)
        print(f"Saved worker logs to {worker_log_dir}", flush=True)
        return

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    task_queue = ctx.Queue()
    for sample_idx in pending_indices:
        task_queue.put(sample_idx)
    for _ in range(workers):
        task_queue.put(None)
    with ctx.Pool(processes=workers, initializer=_init_worker_queues, initargs=(progress_queue, task_queue)) as pool:
        async_results = [pool.apply_async(_worker_entry, (job,)) for job in jobs]
        _consume_progress(progress_queue, async_results, total=len(pending_indices), worker_log_dir=worker_log_dir)
        for result in async_results:
            result.get()

    results = _merge_parts(
        shard_dir,
        output_path,
        expected_sample_ids=expected_sample_ids,
        include_existing_output=args.resume,
    )
    write_summary(cfg.benchmark, results, output_path)
    print(f"Saved merged results to {output_path}", flush=True)
    print(f"Saved worker logs to {worker_log_dir}", flush=True)


def _init_worker_queues(progress_queue: Any, task_queue: Any) -> None:
    global _PROGRESS_QUEUE, _TASK_QUEUE
    _PROGRESS_QUEUE = progress_queue
    _TASK_QUEUE = task_queue


def _worker_entry(job: tuple[int, dict[str, Any], dict[str, Any], str, str, str, bool]) -> str:
    worker_id, config_data, args_data, endpoint, part_path, log_path, resume = job
    if _PROGRESS_QUEUE is None or _TASK_QUEUE is None:
        raise RuntimeError("Worker queues were not initialized.")
    log_mode = "a" if resume else "w"
    with open(log_path, log_mode, encoding="utf-8", buffering=1) as log_file:
        if resume:
            print(f"[resume] worker {worker_id} appending results to {part_path}", file=log_file, flush=True)
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            return _worker_run(worker_id, config_data, args_data, endpoint, part_path, _PROGRESS_QUEUE, _TASK_QUEUE, resume)


def _worker_run(
    worker_id: int,
    config_data: dict[str, Any],
    args_data: dict[str, Any],
    endpoint: str,
    part_path: str,
    progress_queue: Any,
    task_queue: Any,
    resume: bool,
) -> str:
    cfg = eval_config_from_dict(config_data)
    _apply_overrides(cfg, argparse.Namespace(**args_data))
    _apply_loader_limit(cfg, argparse.Namespace(**args_data))
    ensure_streamtext_config(cfg)
    _validate_postprocess_mode(cfg)
    if endpoint:
        cfg.backend.base_url = endpoint
    cfg.backend.endpoints = []

    samples = load_samples(cfg)
    limit = int(args_data.get("limit") or 0) or cfg.batch.limit
    if limit and not _limit_applied_by_loader(cfg):
        samples = samples[:limit]

    runner = StreamTextRunner(
        backend=create_backend(cfg.backend, cfg.runtime),
        frame_store=FrameStore(cfg.dataset),
        runtime=cfg.runtime,
        trace_config=cfg.trace,
        dataset_name=cfg.dataset.dataset_name or cfg.benchmark,
        prompt_profile=cfg.prompt.profile,
        postprocess_config=cfg.postprocess,
        reward_config=cfg.reward,
        memory_config=cfg.memory,
    )

    local_count = 0
    part_mode = "a" if resume else "w"
    with open(part_path, part_mode, encoding="utf-8") as handle:
        while True:
            sample_idx = task_queue.get()
            if sample_idx is None:
                break
            local_count += 1
            sample = samples[sample_idx]
            task_results = _run_task_results(cfg, runner, sample_idx, sample)
            for result_index, result in task_results:
                result.update(
                    {
                        "index": result_index,
                        "task_index": sample_idx,
                        "worker_id": worker_id,
                        "endpoint": endpoint,
                        "policy": "streamtext",
                        "prompt_type": cfg.prompt.profile,
                    }
                )
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            task_failed = any(bool(result.get("error")) for _, result in task_results)
            progress_queue.put(
                {
                    "type": "sample_done",
                    "worker_id": worker_id,
                    "local_count": local_count,
                    "index": sample_idx,
                    "sample_id": sample.sample_id,
                    "result_count": len(task_results),
                    "error": task_failed,
                }
            )
            print(
                f"[worker {worker_id}] {local_count} task_index={sample_idx} sample={sample.sample_id} "
                f"rows={len(task_results)} error={task_failed}",
                flush=True,
            )
    return f"[worker {worker_id}] done {local_count} samples endpoint={endpoint}"


def _run_task_results(
    cfg: EvalConfig,
    runner: StreamTextRunner,
    task_index: int,
    sample: Any,
) -> list[tuple[int, dict[str, Any]]]:
    if cfg.benchmark == "streamingbench" and _is_grouped_streamingbench_sample(sample):
        return _run_grouped_streamingbench(cfg, runner, sample)
    try:
        trace = runner.run_sample(sample)
        result = result_from_trace(cfg.benchmark, trace)
    except Exception as exc:
        result = error_result(cfg.benchmark, sample, repr(exc))
    return [(task_index, result)]


def _run_grouped_streamingbench(
    cfg: EvalConfig,
    runner: StreamTextRunner,
    sample: Any,
) -> list[tuple[int, dict[str, Any]]]:
    qa_list = [item for item in (getattr(sample, "metadata", {}) or {}).get("qa_list") or [] if isinstance(item, dict)]
    results: list[tuple[int, dict[str, Any]]] = []
    try:
        multi_trace = runner.run_streaming_qa_sample(sample)
        actual_metrics = _grouped_actual_metrics(multi_trace)
        for trace in multi_trace.qa_traces:
            result = result_from_trace(cfg.benchmark, trace)
            result.update(
                {
                    "grouped_eval": True,
                    "group_sample_id": sample.sample_id,
                    "group_video_id": sample.video_id,
                    "qa_id": trace.sample.metadata.get("qa_id"),
                    "qa_index": trace.sample.metadata.get("qa_index"),
                    "query_timestamp": trace.sample.metadata.get("query_timestamp"),
                    "target_timestamp": trace.sample.metadata.get("target_timestamp"),
                    "group_result_count": len(qa_list),
                    "group_actual_num_steps": actual_metrics.get("num_steps"),
                    "group_actual_model_call_count": actual_metrics.get("model_call_count"),
                    "group_actual_total_latency_seconds": actual_metrics.get("total_latency_seconds"),
                }
            )
            results.append((int(trace.sample.metadata.get("result_index")), result))
    except Exception as exc:
        results = []
        for qa_index, qa in enumerate(qa_list):
            result = error_result(cfg.benchmark, _sample_for_grouped_error(sample, qa, qa_index), repr(exc))
            result.update(
                {
                    "grouped_eval": True,
                    "group_sample_id": sample.sample_id,
                    "group_video_id": sample.video_id,
                    "qa_id": qa.get("qa_id") or f"qa_{qa_index:04d}",
                    "qa_index": qa_index,
                    "query_timestamp": qa.get("query_timestamp"),
                    "target_timestamp": qa.get("target_timestamp"),
                }
            )
            results.append((int(qa.get("result_index", qa_index)), result))

    seen = {index for index, _ in results}
    for qa_index, qa in enumerate(qa_list):
        result_index = int(qa.get("result_index", qa_index))
        if result_index in seen:
            continue
        result = error_result(cfg.benchmark, _sample_for_grouped_error(sample, qa, qa_index), "missing_grouped_qa_trace")
        result.update(
            {
                "grouped_eval": True,
                "group_sample_id": sample.sample_id,
                "group_video_id": sample.video_id,
                "qa_id": qa.get("qa_id") or f"qa_{qa_index:04d}",
                "qa_index": qa_index,
                "query_timestamp": qa.get("query_timestamp"),
                "target_timestamp": qa.get("target_timestamp"),
            }
        )
        results.append((result_index, result))
    return sorted(results, key=lambda item: item[0])


def _grouped_actual_metrics(multi_trace: Any) -> dict[str, Any]:
    transitions = list(getattr(multi_trace, "prefix_transitions", []) or [])
    for trace in getattr(multi_trace, "qa_traces", []) or []:
        trace_transitions = list(getattr(trace, "transitions", []) or [])
        if trace_transitions and trace_transitions[-1].sample_id == trace.sample.sample_id:
            transitions.append(trace_transitions[-1])
    return rollout_metrics_from_trace(SimpleNamespace(transitions=transitions))


def _sample_for_grouped_error(sample: Any, qa: dict[str, Any], qa_index: int) -> BenchmarkSample:
    qa_id = str(qa.get("qa_id") or f"qa_{qa_index:04d}")
    timestamp = _float_value(qa.get("query_timestamp"), _float_value(qa.get("target_timestamp"), 0.0))
    metadata = dict(getattr(sample, "metadata", {}) or {})
    metadata.pop("qa_list", None)
    metadata.update(
        {
            "is_streaming_qa_branch": True,
            "streamingbench_grouped_eval": True,
            "streamingbench_grouped_real": str(metadata.get("split", "")) == "real",
            "qa_id": qa_id,
            "qa_index": qa_index,
            "question": qa.get("question") if isinstance(qa.get("question"), dict) else {},
            "query_text": qa.get("query_text") or "",
            "query_timestamp": timestamp,
            "target_timestamp": timestamp,
            "result_index": qa.get("result_index", qa_index),
        }
    )
    return BenchmarkSample(
        sample_id=str(qa.get("sample_id") or f"{sample.sample_id}_{qa_id}"),
        video_id=str(getattr(sample, "video_id", "")),
        video_path=str(getattr(sample, "video_path", "")),
        query_events=[QueryEvent(text=str(qa.get("query_text") or ""), timestamp=timestamp)],
        metadata=metadata,
    )


def _apply_overrides(cfg: EvalConfig, args: argparse.Namespace) -> None:
    if getattr(args, "benchmark", ""):
        cfg.benchmark = args.benchmark
    if getattr(args, "backend", ""):
        cfg.backend.backend = args.backend
    if getattr(args, "model", ""):
        cfg.backend.model = args.model
    if getattr(args, "max_steps", 0):
        cfg.runtime.max_steps = args.max_steps


def _apply_loader_limit(cfg: EvalConfig, args: argparse.Namespace) -> None:
    limit = int(getattr(args, "limit", 0) or cfg.batch.limit or 0)
    if limit and _limit_applied_by_loader(cfg):
        cfg.benchmark_args["limit"] = limit


def _limit_applied_by_loader(cfg: EvalConfig) -> bool:
    return (
        cfg.benchmark == "streamingbench"
        and str(cfg.benchmark_args.get("split", "real")) in {"real", "omni"}
        and _truthy(cfg.benchmark_args.get("group_by_video", False))
    )


def _validate_postprocess_mode(cfg: EvalConfig) -> None:
    if cfg.postprocess.mode not in {"eval_repair", "rollout_repair"}:
        raise ValueError(f"StreamText eval supports eval_repair/rollout_repair only, got: {cfg.postprocess.mode}")


def _configured_endpoints(cfg: EvalConfig, args: argparse.Namespace) -> list[str]:
    endpoints = [item.strip() for item in str(getattr(args, "endpoints", "") or "").split(",") if item.strip()]
    if endpoints:
        return endpoints
    endpoints = [str(item).strip() for item in cfg.batch.endpoints if str(item).strip()]
    if endpoints:
        return endpoints
    return [item.strip() for item in cfg.backend.endpoints if item.strip()]


def _prepare_run_dir(path: Path, pattern: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old_path in path.glob(pattern):
        old_path.unlink()


def _build_manifest(
    cfg: EvalConfig,
    args: argparse.Namespace,
    samples: list[Any],
    output_path: Path,
    expected_sample_ids: list[str],
) -> dict[str, Any]:
    return {
        "benchmark": cfg.benchmark,
        "total_tasks": len(samples),
        "task_sample_ids": [str(sample.sample_id) for sample in samples],
        "total_samples": len(expected_sample_ids),
        "sample_ids": expected_sample_ids,
        "limit": int(getattr(args, "limit", 0) or cfg.batch.limit or 0),
        "model": cfg.backend.model,
        "output": str(output_path),
        "benchmark_args": _jsonable(cfg.benchmark_args),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _validate_manifest(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        print(f"resume: no manifest found at {path}; validating existing rows by sample_id", flush=True)
        return
    with path.open(encoding="utf-8") as handle:
        actual = json.load(handle)
    for key in ("benchmark", "total_samples", "sample_ids", "benchmark_args"):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"Cannot resume because {path} does not match current run for {key}. "
                "Use a fresh output directory or rerun without --resume."
            )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _expected_result_sample_ids(samples: list[Any]) -> list[str]:
    entries: list[tuple[int, str]] = []
    for task_index, sample in enumerate(samples):
        entries.extend(_sample_result_entries(task_index, sample))
    if not entries:
        return []
    max_index = max(index for index, _ in entries)
    sample_ids = [""] * (max_index + 1)
    for index, sample_id in entries:
        if index < 0:
            raise ValueError(f"Invalid negative result index {index} for sample_id={sample_id!r}.")
        if sample_ids[index]:
            raise ValueError(f"Duplicate result index {index} for sample_id={sample_id!r}.")
        sample_ids[index] = sample_id
    missing = [idx for idx, sample_id in enumerate(sample_ids) if not sample_id]
    if missing:
        raise ValueError(f"Missing result sample ids for index range; first gaps: {missing[:10]}")
    return sample_ids


def _sample_result_entries(task_index: int, sample: Any) -> list[tuple[int, str]]:
    if _is_grouped_streamingbench_sample(sample):
        entries: list[tuple[int, str]] = []
        for qa_index, qa in enumerate((getattr(sample, "metadata", {}) or {}).get("qa_list") or []):
            if not isinstance(qa, dict):
                continue
            result_index = int(qa.get("result_index", qa_index))
            sample_id = str(qa.get("sample_id") or f"{getattr(sample, 'sample_id', '')}_q{qa_index:03d}")
            entries.append((result_index, sample_id))
        return entries
    return [(task_index, str(getattr(sample, "sample_id", "")))]


def _task_result_indices(task_index: int, sample: Any) -> list[int]:
    return [index for index, _ in _sample_result_entries(task_index, sample)]


def _task_is_completed(task_index: int, sample: Any, completed_results: dict[int, dict[str, Any]]) -> bool:
    result_indices = _task_result_indices(task_index, sample)
    return bool(result_indices) and all(index in completed_results for index in result_indices)


def _pending_task_count(samples: list[Any], completed_results: dict[int, dict[str, Any]]) -> int:
    return sum(1 for task_index, sample in enumerate(samples) if not _task_is_completed(task_index, sample, completed_results))


def _is_grouped_streamingbench_sample(sample: Any) -> bool:
    metadata = getattr(sample, "metadata", {}) or {}
    return bool(metadata.get("streamingbench_grouped_eval") or metadata.get("streamingbench_grouped_real"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_completed_results(output_path: Path, shard_dir: Path, expected_sample_ids: list[str]) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    for source_path in _result_sources(output_path, shard_dir, include_existing_output=True):
        for index, result in _iter_result_rows(source_path, expected_sample_ids):
            completed[index] = result
    return completed


def _result_sources(output_path: Path, shard_dir: Path, *, include_existing_output: bool) -> list[Path]:
    sources: list[Path] = []
    if include_existing_output and output_path.exists():
        sources.append(output_path)
    sources.extend(sorted(shard_dir.glob("part_*.jsonl")))
    return sources


def _iter_result_rows(path: Path, expected_sample_ids: list[str]):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: ignored malformed JSON row {path}:{line_no}: {exc}", file=sys.stderr, flush=True)
                continue
            try:
                index = int(result["index"])
            except (KeyError, TypeError, ValueError):
                print(f"warning: ignored result row without valid index {path}:{line_no}", file=sys.stderr, flush=True)
                continue
            if index < 0 or index >= len(expected_sample_ids):
                print(f"warning: ignored out-of-range result index {index} in {path}:{line_no}", file=sys.stderr, flush=True)
                continue
            sample_id = str(result.get("sample_id", ""))
            expected_sample_id = str(expected_sample_ids[index])
            if not sample_id:
                print(f"warning: ignored result row without sample_id {path}:{line_no}", file=sys.stderr, flush=True)
                continue
            if sample_id != expected_sample_id:
                raise ValueError(
                    f"Cannot resume because result row {path}:{line_no} has index={index} "
                    f"sample_id={sample_id!r}, but current sample_id is {expected_sample_id!r}."
                )
            yield index, result


def _merge_parts(
    shard_dir: Path,
    output_path: Path,
    *,
    expected_sample_ids: list[str],
    include_existing_output: bool,
) -> list[dict[str, Any]]:
    results_by_index: dict[int, dict[str, Any]] = {}
    for source_path in _result_sources(output_path, shard_dir, include_existing_output=include_existing_output):
        for index, result in _iter_result_rows(source_path, expected_sample_ids):
            results_by_index[index] = result
    results = [results_by_index[index] for index in sorted(results_by_index)]
    if len(results) != len(expected_sample_ids):
        print(
            f"warning: merged {len(results)}/{len(expected_sample_ids)} results; summary will cover completed rows only",
            file=sys.stderr,
            flush=True,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for result in results:
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
    return results


def _consume_progress(progress_queue: Any, async_results: list[Any], *, total: int, worker_log_dir: Path) -> None:
    done = 0
    errors = 0
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    if tqdm is not None:
        with tqdm(total=total, desc="streamtext-eval", unit="sample", dynamic_ncols=True) as bar:
            while done < total:
                try:
                    event = progress_queue.get(timeout=0.5)
                except queue.Empty:
                    _raise_ready_worker_errors(async_results)
                    continue
                if event.get("type") != "sample_done":
                    continue
                done += 1
                errors += int(bool(event.get("error")))
                bar.update(1)
                bar.set_postfix(errors=errors)
        _raise_ready_worker_errors(async_results)
        return

    print(f"streamtext-eval: 0/{total} samples, worker logs -> {worker_log_dir}", flush=True)
    while done < total:
        try:
            event = progress_queue.get(timeout=5.0)
        except queue.Empty:
            _raise_ready_worker_errors(async_results)
            print(f"streamtext-eval: {done}/{total} samples, errors={errors}", flush=True)
            continue
        if event.get("type") != "sample_done":
            continue
        done += 1
        errors += int(bool(event.get("error")))
        print(f"streamtext-eval: {done}/{total} samples, errors={errors}", flush=True)
    _raise_ready_worker_errors(async_results)


def _raise_ready_worker_errors(async_results: list[Any]) -> None:
    for result in async_results:
        if result.ready():
            result.get(timeout=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=("ovo", "streamingbench"), default="")
    parser.add_argument("--backend", choices=("mock", "openai", "vllm", "openai_compatible", "local", "gemini"), default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoints", default="")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="debug only: stop each sample after N stream steps")
    parser.add_argument("--output", default="")
    parser.add_argument("--worker-log-dir", default="")
    parser.add_argument("--resume", action="store_true", help="skip completed rows from existing results/part files")
    return parser.parse_args()


if __name__ == "__main__":
    main()
