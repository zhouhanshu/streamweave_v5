#!/usr/bin/env python3
"""Multi-process evaluation entry point.

Endpoint assignment follows the V3 local-vLLM pattern: each process owns one
endpoint, chosen by worker_id % len(endpoints). Single-process eval never
round-robins endpoint lists.
"""

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
from streamweave.rollout import RolloutRunner

from evaluation.runner import error_result, load_samples, result_from_trace, write_summary


_PROGRESS_QUEUE: Any = None


def main() -> None:
    args = parse_args()
    config_data = load_config(args.config)
    cfg = eval_config_from_dict(config_data)
    _apply_overrides(cfg, args)

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
    worker_log_dir = Path(args.worker_log_dir or cfg.batch.worker_log_dir or output_path.parent / "worker_logs")
    _prepare_run_dir(shard_dir, "part_*.jsonl")
    _prepare_run_dir(worker_log_dir, "worker_*.log")

    endpoints = _configured_endpoints(cfg, args)
    workers = args.workers or cfg.batch.workers or (len(endpoints) if endpoints else 1)
    workers = max(1, workers)
    jobs = []
    indices = list(range(len(samples)))
    for worker_id in range(workers):
        endpoint = endpoints[worker_id % len(endpoints)] if endpoints else ""
        part_path = shard_dir / f"part_{worker_id:03d}.jsonl"
        log_path = worker_log_dir / f"worker_{worker_id:03d}.log"
        jobs.append(
            (
                worker_id,
                indices[worker_id::workers],
                config_data,
                vars(args),
                endpoint,
                str(part_path),
                str(log_path),
            )
        )

    if not samples:
        output_path.write_text("", encoding="utf-8")
        write_summary(cfg.benchmark, [], output_path)
        return

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    with ctx.Pool(processes=workers, initializer=_init_worker_progress, initargs=(progress_queue,)) as pool:
        async_results = [pool.apply_async(_worker_entry, (job,)) for job in jobs]
        _consume_progress(progress_queue, async_results, total=len(samples), worker_log_dir=worker_log_dir)
        for result in async_results:
            result.get()

    results = _merge_parts(shard_dir, output_path)
    write_summary(cfg.benchmark, results, output_path)
    print(f"Saved merged results to {output_path}", flush=True)
    print(f"Saved worker logs to {worker_log_dir}", flush=True)


def _init_worker_progress(progress_queue: Any) -> None:
    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = progress_queue


def _worker_entry(job: tuple[int, list[int], dict[str, Any], dict[str, Any], str, str, str]) -> str:
    worker_id, indices, config_data, args_data, endpoint, part_path, log_path = job
    if _PROGRESS_QUEUE is None:
        raise RuntimeError("Worker progress queue was not initialized.")
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            return _worker_run(worker_id, indices, config_data, args_data, endpoint, part_path, _PROGRESS_QUEUE)


def _worker_run(
    worker_id: int,
    indices: list[int],
    config_data: dict[str, Any],
    args_data: dict[str, Any],
    endpoint: str,
    part_path: str,
    progress_queue: Any,
) -> str:
    cfg = eval_config_from_dict(config_data)
    _apply_overrides(cfg, argparse.Namespace(**args_data))
    if endpoint:
        cfg.backend.base_url = endpoint
    cfg.backend.endpoints = []

    samples = load_samples(cfg)
    limit = int(args_data.get("limit") or 0) or cfg.batch.limit
    if limit:
        samples = samples[:limit]

    runner = RolloutRunner(
        backend=create_backend(cfg.backend, cfg.runtime),
        frame_store=FrameStore(cfg.dataset),
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

    with open(part_path, "w", encoding="utf-8") as f:
        for local_count, sample_idx in enumerate(indices, start=1):
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
                    "policy": cfg.policy,
                    "prompt_type": cfg.prompt.profile,
                }
            )
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            progress_queue.put(
                {
                    "type": "sample_done",
                    "worker_id": worker_id,
                    "local_count": local_count,
                    "total": len(indices),
                    "sample_id": sample.sample_id,
                    "error": bool(result.get("error")),
                }
            )
            print(
                f"[worker {worker_id}] {local_count}/{len(indices)} sample={sample.sample_id} error={bool(result.get('error'))}",
                flush=True,
            )
    return f"[worker {worker_id}] done {len(indices)} samples endpoint={endpoint}"


def _apply_overrides(cfg: EvalConfig, args: argparse.Namespace) -> None:
    if getattr(args, "benchmark", ""):
        cfg.benchmark = args.benchmark
    if getattr(args, "backend", ""):
        cfg.backend.backend = args.backend
    if getattr(args, "model", ""):
        cfg.backend.model = args.model


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


def _merge_parts(shard_dir: Path, output_path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for part_path in sorted(shard_dir.glob("part_*.jsonl")):
        with part_path.open(encoding="utf-8") as part:
            for line in part:
                if line.strip():
                    results.append(json.loads(line))
    results.sort(key=lambda item: int(item.get("index", 0)))
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
        with tqdm(total=total, desc="eval", unit="sample", dynamic_ncols=True) as bar:
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

    print(f"eval: 0/{total} samples, worker logs -> {worker_log_dir}", flush=True)
    while done < total:
        try:
            event = progress_queue.get(timeout=5.0)
        except queue.Empty:
            _raise_ready_worker_errors(async_results)
            print(f"eval: {done}/{total} samples, errors={errors}", flush=True)
            continue
        if event.get("type") != "sample_done":
            continue
        done += 1
        errors += int(bool(event.get("error")))
        print(f"eval: {done}/{total} samples, errors={errors}", flush=True)
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
    parser.add_argument("--output", default="")
    parser.add_argument("--worker-log-dir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
