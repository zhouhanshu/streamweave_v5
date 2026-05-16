#!/usr/bin/env python3
"""Multi-process batch evaluation for VideoHallucer.

Mirrors evaluation/eval_batch.py's pattern (one endpoint per worker via
worker_id % len(endpoints)), but calls RolloutRunner.run_multi_qa_sample
so each entry's basic + hallucination probes share a single prefix rollout.

Workflow:
  1. Load entries from videohallucer.json
  2. Spawn N workers; each owns its own endpoint and RolloutRunner
  3. Task queue feeds entry indices to workers
  4. Each worker writes a shard file: per_qa_part_<wid>.jsonl
  5. Merge shards, aggregate pair-level + subset-level metrics
  6. Write per_qa.jsonl / per_pair.jsonl / summary.json / summary.txt

Resume:
  --resume reuses existing shard files; only entries with no recorded row
  for any of their qa_ids are re-scheduled.

Usage:
  python dataset3/eval_videohallucer_batch.py \\
      --config dataset3/configs/eval_videohallucer_anchor_delta.yaml \\
      --workers 8 \\
      --endpoints http://h1:8000/v1 http://h2:8000/v1 ... \\
      --model qwen3vl-8b-streamweave-sft
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.factory import create_backend  # noqa: E402
from streamweave.config import EvalConfig, eval_config_from_dict, load_config  # noqa: E402
from streamweave.frame_store import FrameStore  # noqa: E402
from streamweave.rollout import RolloutRunner  # noqa: E402

from dataset3.videohallucer_loader import load_samples  # noqa: E402
from dataset3.videohallucer_scorer import (  # noqa: E402
    aggregate,
    format_summary_table,
    score_qa_trace,
)


_PROGRESS_QUEUE: Any = None
_TASK_QUEUE: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--endpoints", nargs="*", default=None,
                        help="vLLM/OpenAI-compatible endpoints; one per worker via round-robin")
    parser.add_argument("--backend", default="", help="override cfg.backend.backend")
    parser.add_argument("--model", default="", help="override cfg.backend.model")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task", default="", help="restrict to a single subset")
    parser.add_argument("--output-name", default="", help="override cfg.trace.experiment_name")
    parser.add_argument("--resume", action="store_true", help="reuse shard files; only run pending entries")
    parser.add_argument("--worker-log-dir", default="", help="override cfg.batch.worker_log_dir")
    return parser.parse_args()


def _apply_overrides(cfg: EvalConfig, args: argparse.Namespace) -> None:
    if args.backend:
        cfg.backend.backend = args.backend
    if args.model:
        cfg.backend.model = args.model
    if args.output_name:
        cfg.trace.experiment_name = args.output_name
    if args.task:
        cfg.benchmark_args["task"] = args.task
    if args.limit:
        cfg.benchmark_args["limit"] = int(args.limit)


def _output_dir(cfg: EvalConfig) -> Path:
    return Path(cfg.trace.output_root) / cfg.trace.experiment_name


def _split_endpoint_arg(values: Any) -> list[str]:
    """Accept --endpoints in any of:
      --endpoints h1 h2 h3
      --endpoints "h1,h2,h3"     (matches existing evaluation/eval_batch.py convention)
      --endpoints "h1, h2"        (whitespace tolerated)
    """
    out: list[str] = []
    if not values:
        return out
    items = values if isinstance(values, (list, tuple)) else [values]
    for v in items:
        s = str(v or "").strip()
        if not s:
            continue
        for part in s.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _configured_endpoints(cfg: EvalConfig, args: argparse.Namespace) -> list[str]:
    endpoints = _split_endpoint_arg(getattr(args, "endpoints", None))
    if endpoints:
        return endpoints
    if cfg.batch.endpoints:
        return [str(e).strip() for e in cfg.batch.endpoints if str(e).strip()]
    if cfg.backend.endpoints:
        return [str(e).strip() for e in cfg.backend.endpoints if str(e).strip()]
    if cfg.backend.base_url:
        return [cfg.backend.base_url]
    return []


def _prepare_run_dir(path: Path, glob: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob(glob):
        old.unlink()


def _load_completed(shard_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Read existing shard files; return {pair_id: {branch: row}} for resume."""
    completed: dict[str, dict[str, dict[str, Any]]] = {}
    for part_path in sorted(shard_dir.glob("per_qa_part_*.jsonl")):
        with part_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                completed.setdefault(row.get("pair_id", ""), {})[row.get("branch", "")] = row
    return completed


def _entry_completed(entry_qa_list: list[dict[str, Any]], completed: dict[str, dict[str, dict[str, Any]]]) -> bool:
    if not entry_qa_list:
        return True
    pair_id = entry_qa_list[0]["pair_id"]
    branches = completed.get(pair_id, {})
    return all(qa["branch"] in branches for qa in entry_qa_list)


def main() -> None:
    args = parse_args()
    config_data = load_config(args.config)
    cfg = eval_config_from_dict(config_data)
    _apply_overrides(cfg, args)

    samples = load_samples(
        cfg.benchmark_args["json_path"],
        cfg.benchmark_args["video_root"],
        sample_ids=cfg.benchmark_args.get("sample_ids") or [],
        task=cfg.benchmark_args.get("task") or "",
        limit=int(cfg.benchmark_args.get("limit") or 0),
    )
    print(f"Loaded {len(samples)} entries", flush=True)

    output_dir = _output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / ".shards"
    worker_log_dir = Path(args.worker_log_dir or cfg.batch.worker_log_dir or (output_dir / "worker_logs"))

    if args.resume:
        shard_dir.mkdir(parents=True, exist_ok=True)
        worker_log_dir.mkdir(parents=True, exist_ok=True)
    else:
        _prepare_run_dir(shard_dir, "per_qa_part_*.jsonl")
        _prepare_run_dir(worker_log_dir, "worker_*.log")

    endpoints = _configured_endpoints(cfg, args)
    workers = args.workers or cfg.batch.workers or (len(endpoints) if endpoints else 1)
    workers = max(1, workers)
    if not endpoints:
        endpoints = [""]  # mock backend or single configured base_url
    print(f"Workers: {workers}, Endpoints: {endpoints}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Shard dir:  {shard_dir}", flush=True)

    completed = _load_completed(shard_dir) if args.resume else {}
    pending_indices = [
        idx for idx, sample in enumerate(samples)
        if not _entry_completed(sample.metadata.get("qa_list") or [], completed)
    ]
    print(f"Pending entries: {len(pending_indices)} / {len(samples)}", flush=True)

    if pending_indices:
        ctx = mp.get_context("spawn")
        progress_queue = ctx.Queue()
        task_queue = ctx.Queue()
        for idx in pending_indices:
            task_queue.put(idx)
        for _ in range(workers):
            task_queue.put(None)
        jobs = []
        for worker_id in range(workers):
            endpoint = endpoints[worker_id % len(endpoints)]
            part_path = shard_dir / f"per_qa_part_{worker_id:03d}.jsonl"
            log_path = worker_log_dir / f"worker_{worker_id:03d}.log"
            jobs.append((worker_id, config_data, vars(args), endpoint, str(part_path), str(log_path), bool(args.resume)))

        with ctx.Pool(processes=workers, initializer=_init_queues, initargs=(progress_queue, task_queue)) as pool:
            async_results = [pool.apply_async(_worker_entry, (job,)) for job in jobs]
            _consume_progress(progress_queue, async_results, total=len(pending_indices))
            for res in async_results:
                res.get()

    # Merge shards
    per_qa_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for part_path in sorted(shard_dir.glob("per_qa_part_*.jsonl")):
        with part_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (row.get("pair_id", ""), row.get("branch", ""))
                if key in seen:
                    continue
                seen.add(key)
                per_qa_rows.append(row)

    per_qa_path = output_dir / "per_qa.jsonl"
    per_pair_path = output_dir / "per_pair.jsonl"
    summary_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"

    with per_qa_path.open("w", encoding="utf-8") as f:
        for row in per_qa_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    agg = aggregate(per_qa_rows)
    with per_pair_path.open("w", encoding="utf-8") as f:
        for row in agg["per_pair"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps({"per_subset": agg["per_subset"], "overall": agg["overall"]}, indent=2, ensure_ascii=False)
    )
    table = format_summary_table(agg)
    summary_txt_path.write_text(table + "\n")

    print()
    print(table, flush=True)
    print()
    print(f"Wrote {per_qa_path}", flush=True)
    print(f"Wrote {per_pair_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {summary_txt_path}", flush=True)


def _init_queues(progress_queue: Any, task_queue: Any) -> None:
    global _PROGRESS_QUEUE, _TASK_QUEUE
    _PROGRESS_QUEUE = progress_queue
    _TASK_QUEUE = task_queue


def _worker_entry(job: tuple) -> str:
    worker_id, config_data, args_data, endpoint, part_path, log_path, resume = job
    log_mode = "a" if resume else "w"
    with open(log_path, log_mode, encoding="utf-8", buffering=1) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            try:
                return _worker_run(worker_id, config_data, args_data, endpoint, part_path, resume)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                _PROGRESS_QUEUE.put({"type": "worker_crash", "worker_id": worker_id, "error": repr(exc)})
                raise


def _worker_run(
    worker_id: int,
    config_data: dict[str, Any],
    args_data: dict[str, Any],
    endpoint: str,
    part_path: str,
    resume: bool,
) -> str:
    cfg = eval_config_from_dict(config_data)
    _apply_overrides(cfg, argparse.Namespace(**args_data))
    if endpoint:
        cfg.backend.base_url = endpoint
    cfg.backend.endpoints = []

    samples = load_samples(
        cfg.benchmark_args["json_path"],
        cfg.benchmark_args["video_root"],
        sample_ids=cfg.benchmark_args.get("sample_ids") or [],
        task=cfg.benchmark_args.get("task") or "",
        limit=int(cfg.benchmark_args.get("limit") or 0),
    )
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

    print(f"[worker {worker_id}] starting endpoint={endpoint} model={cfg.backend.model}", flush=True)

    part_mode = "a" if resume else "w"
    local_count = 0
    with open(part_path, part_mode, encoding="utf-8") as f:
        while True:
            sample_idx = _TASK_QUEUE.get()
            if sample_idx is None:
                break
            local_count += 1
            sample = samples[sample_idx]
            qa_list = sample.metadata.get("qa_list") or []
            try:
                multi = runner.run_multi_qa_sample(sample)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                for qa in qa_list:
                    row = _error_row(sample, qa, err)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                _PROGRESS_QUEUE.put({"type": "sample_done", "worker_id": worker_id, "sample_id": sample.sample_id, "error": True})
                continue

            if multi.task_failed:
                for qa in qa_list:
                    row = _error_row(sample, qa, multi.failure_reason or "task_failed")
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                _PROGRESS_QUEUE.put({"type": "sample_done", "worker_id": worker_id, "sample_id": sample.sample_id, "error": True})
                continue

            ok_count = 0
            for trace in multi.qa_traces:
                qa_meta = _qa_meta_from_branch(trace.sample.metadata)
                row = score_qa_trace(trace, qa_meta)
                if row["hit"]:
                    ok_count += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            _PROGRESS_QUEUE.put({
                "type": "sample_done",
                "worker_id": worker_id,
                "sample_id": sample.sample_id,
                "ok_branches": ok_count,
                "total_branches": len(multi.qa_traces),
                "error": False,
            })
    print(f"[worker {worker_id}] done {local_count} samples", flush=True)
    return f"worker {worker_id} ok"


def _qa_meta_from_branch(branch_metadata: dict[str, Any]) -> dict[str, Any]:
    raw = branch_metadata.get("raw_annotation") or {}
    return {
        "task": raw.get("task") or branch_metadata.get("task") or "",
        "pair_id": raw.get("pair_id") or branch_metadata.get("pair_id") or "",
        "branch": raw.get("branch") or branch_metadata.get("qa_id") or "",
        "question": raw.get("question") or branch_metadata.get("question") or "",
        "gt": raw.get("gt") or branch_metadata.get("gt") or "",
    }


def _error_row(sample, qa: dict[str, Any], err: str) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "task": qa["task"],
        "pair_id": qa["pair_id"],
        "branch": qa["branch"],
        "question": qa["question"],
        "gt": qa["gt"],
        "pred": "",
        "hit": 0,
        "final_answer": "",
        "raw_output": "",
        "task_failed": True,
        "failure_reason": err,
    }


def _consume_progress(progress_queue: Any, async_results: list, total: int) -> None:
    import queue as _queue
    done = 0
    fail = 0
    while done + fail < total and not all(r.ready() for r in async_results):
        try:
            msg = progress_queue.get(timeout=5)
        except _queue.Empty:
            continue
        if msg.get("type") == "worker_crash":
            print(f"[CRASH] worker {msg['worker_id']} {msg['error']}", flush=True)
            continue
        if msg.get("error"):
            fail += 1
        else:
            done += 1
        if (done + fail) % 10 == 0 or (done + fail) == total:
            print(f"[progress] done={done} fail={fail} / total={total}", flush=True)
    # drain any remaining messages
    while True:
        try:
            progress_queue.get_nowait()
        except Exception:
            break


if __name__ == "__main__":
    main()
