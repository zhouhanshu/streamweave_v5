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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.factory import create_backend
from streamweave.config import EvalConfig, eval_config_from_dict, load_config
from streamweave.frame_store import FrameStore

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
    ensure_streamtext_config(cfg)
    _validate_postprocess_mode(cfg)

    samples = load_samples(cfg)
    limit = args.limit or cfg.batch.limit
    if limit:
        samples = samples[:limit]

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
    manifest = _build_manifest(cfg, args, samples, output_path)
    if args.resume:
        _validate_manifest(manifest_path, manifest)

    completed_results: dict[int, dict[str, Any]] = {}
    if args.resume:
        completed_results = _load_completed_results(output_path, shard_dir, samples)
        print(
            f"resume: completed={len(completed_results)} pending={len(samples) - len(completed_results)} total={len(samples)}",
            flush=True,
        )
    _write_manifest(manifest_path, manifest)
    pending_indices = [idx for idx in range(len(samples)) if idx not in completed_results]
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
        results = _merge_parts(shard_dir, output_path, samples=samples, include_existing_output=args.resume)
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

    results = _merge_parts(shard_dir, output_path, samples=samples, include_existing_output=args.resume)
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
    ensure_streamtext_config(cfg)
    _validate_postprocess_mode(cfg)
    if endpoint:
        cfg.backend.base_url = endpoint
    cfg.backend.endpoints = []

    samples = load_samples(cfg)
    limit = int(args_data.get("limit") or 0) or cfg.batch.limit
    if limit:
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
            try:
                trace = runner.run_sample(sample)
                result = result_from_trace(cfg.benchmark, trace)
            except Exception as exc:
                result = error_result(cfg.benchmark, sample, repr(exc))
            result.update(
                {
                    "index": sample_idx,
                    "worker_id": worker_id,
                    "endpoint": endpoint,
                    "policy": "streamtext",
                    "prompt_type": cfg.prompt.profile,
                }
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            progress_queue.put(
                {
                    "type": "sample_done",
                    "worker_id": worker_id,
                    "local_count": local_count,
                    "index": sample_idx,
                    "sample_id": sample.sample_id,
                    "error": bool(result.get("error")),
                }
            )
            print(
                f"[worker {worker_id}] {local_count} index={sample_idx} sample={sample.sample_id} error={bool(result.get('error'))}",
                flush=True,
            )
    return f"[worker {worker_id}] done {local_count} samples endpoint={endpoint}"


def _apply_overrides(cfg: EvalConfig, args: argparse.Namespace) -> None:
    if getattr(args, "benchmark", ""):
        cfg.benchmark = args.benchmark
    if getattr(args, "backend", ""):
        cfg.backend.backend = args.backend
    if getattr(args, "model", ""):
        cfg.backend.model = args.model
    if getattr(args, "max_steps", 0):
        cfg.runtime.max_steps = args.max_steps


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


def _build_manifest(cfg: EvalConfig, args: argparse.Namespace, samples: list[Any], output_path: Path) -> dict[str, Any]:
    return {
        "benchmark": cfg.benchmark,
        "total_samples": len(samples),
        "sample_ids": [str(sample.sample_id) for sample in samples],
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
    for key in ("benchmark", "total_samples", "sample_ids"):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"Cannot resume because {path} does not match current run for {key}. "
                "Use a fresh output directory or rerun without --resume."
            )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_completed_results(output_path: Path, shard_dir: Path, samples: list[Any]) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    for source_path in _result_sources(output_path, shard_dir, include_existing_output=True):
        for index, result in _iter_result_rows(source_path, samples):
            completed[index] = result
    return completed


def _result_sources(output_path: Path, shard_dir: Path, *, include_existing_output: bool) -> list[Path]:
    sources: list[Path] = []
    if include_existing_output and output_path.exists():
        sources.append(output_path)
    sources.extend(sorted(shard_dir.glob("part_*.jsonl")))
    return sources


def _iter_result_rows(path: Path, samples: list[Any]):
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
            if index < 0 or index >= len(samples):
                print(f"warning: ignored out-of-range result index {index} in {path}:{line_no}", file=sys.stderr, flush=True)
                continue
            sample_id = str(result.get("sample_id", ""))
            expected_sample_id = str(samples[index].sample_id)
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
    samples: list[Any],
    include_existing_output: bool,
) -> list[dict[str, Any]]:
    results_by_index: dict[int, dict[str, Any]] = {}
    for source_path in _result_sources(output_path, shard_dir, include_existing_output=include_existing_output):
        for index, result in _iter_result_rows(source_path, samples):
            results_by_index[index] = result
    results = [results_by_index[index] for index in sorted(results_by_index)]
    if len(results) != len(samples):
        print(
            f"warning: merged {len(results)}/{len(samples)} results; summary will cover completed rows only",
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
