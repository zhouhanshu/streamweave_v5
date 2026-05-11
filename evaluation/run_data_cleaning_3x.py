#!/usr/bin/env python3
"""Run repeated StreamWeave inference for dataset cleaning.

The script reuses the normal evaluation adapters and rollout runner, but writes
one aggregate row per sample.  Each row contains N independent sampled runs plus
success-rate and protocol-stability fields for downstream filtering.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import queue
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.factory import create_backend
from streamweave.config import EvalConfig, eval_config_from_dict, load_config
from streamweave.frame_store import FrameStore
from streamweave.rollout import RolloutRunner

from evaluation.eval_batch import (
    _apply_overrides,
    _configured_endpoints,
    _consume_progress,
    _iter_result_rows,
    _prepare_run_dir,
    _write_manifest,
)
from evaluation import dataset2_adapter
from evaluation.runner import (
    error_result as runner_error_result,
    load_samples as runner_load_samples,
    result_from_trace as runner_result_from_trace,
)


_PROGRESS_QUEUE: Any = None
_TASK_QUEUE: Any = None


def main() -> None:
    args = parse_args()
    config_data = load_config(args.config)
    cfg = eval_config_from_dict(config_data)
    _apply_cleaning_overrides(cfg, args)

    samples = load_samples_for_cleaning(cfg)
    limit = args.limit or cfg.batch.limit
    if limit:
        samples = samples[:limit]

    output_path = Path(args.output or cfg.batch.output or cfg.result_output or "outputs/data_cleaning_0510/scores_3x.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output_path.parent / f".{output_path.stem}_parts"
    worker_log_dir = Path(args.worker_log_dir or cfg.batch.worker_log_dir or output_path.parent / "worker_logs")
    if args.resume:
        shard_dir.mkdir(parents=True, exist_ok=True)
        worker_log_dir.mkdir(parents=True, exist_ok=True)
    else:
        _prepare_run_dir(shard_dir, "part_*.jsonl")
        _prepare_run_dir(worker_log_dir, "worker_*.log")

    endpoints = _configured_endpoints(cfg, args)
    workers = args.workers or cfg.batch.workers or (len(endpoints) if endpoints else 1)
    workers = max(1, workers)

    manifest = _build_cleaning_manifest(cfg, args, samples, output_path)
    manifest_path = shard_dir / "run_manifest.json"
    if args.resume:
        _validate_cleaning_manifest(manifest_path, manifest)
    _write_manifest(manifest_path, manifest)

    completed = (
        _load_completed_aggregates(output_path, shard_dir, samples, include_existing_output=cfg.benchmark != "dataset2")
        if args.resume
        else {}
    )
    pending_indices = [idx for idx in range(len(samples)) if idx not in completed]
    if not samples:
        output_path.write_text("", encoding="utf-8")
        write_cleaning_summary([], output_path)
        return
    if not pending_indices:
        rows = _merge_aggregate_parts(cfg, shard_dir, output_path, samples=samples, include_existing_output=args.resume)
        write_cleaning_summary(rows, output_path)
        print(f"Saved merged cleaning results to {output_path}", flush=True)
        return

    jobs = []
    for worker_id in range(workers):
        endpoint = endpoints[worker_id % len(endpoints)] if endpoints else ""
        part_path = shard_dir / f"part_{worker_id:03d}.jsonl"
        log_path = worker_log_dir / f"worker_{worker_id:03d}.log"
        jobs.append((worker_id, config_data, vars(args), endpoint, str(part_path), str(log_path), bool(args.resume)))

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

    rows = _merge_aggregate_parts(cfg, shard_dir, output_path, samples=samples, include_existing_output=args.resume)
    write_cleaning_summary(rows, output_path)
    print(f"Saved merged cleaning results to {output_path}", flush=True)
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
    args = argparse.Namespace(**args_data)
    _apply_cleaning_overrides(cfg, args)
    if endpoint:
        cfg.backend.base_url = endpoint
    cfg.backend.endpoints = []

    samples = load_samples_for_cleaning(cfg)
    limit = int(args_data.get("limit") or 0) or cfg.batch.limit
    if limit:
        samples = samples[:limit]

    base_experiment = cfg.trace.experiment_name
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

    repeats = max(1, int(args_data.get("repeats") or 3))
    part_mode = "a" if resume else "w"
    local_count = 0
    with open(part_path, part_mode, encoding="utf-8") as f:
        while True:
            sample_idx = task_queue.get()
            if sample_idx is None:
                break
            sample = samples[sample_idx]
            local_count += 1
            runs = []
            for repeat_idx in range(repeats):
                runner.trace_config.experiment_name = _repeat_experiment_name(base_experiment, repeat_idx)
                runs.append(_run_repeat(cfg, runner, sample_idx, sample, repeat_idx, endpoint))
            row = aggregate_sample_runs(cfg, sample_idx, sample, runs)
            row.update({"worker_id": worker_id, "endpoint": endpoint, "policy": cfg.policy, "prompt_type": cfg.prompt.profile})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            progress_queue.put(
                {
                    "type": "sample_done",
                    "worker_id": worker_id,
                    "local_count": local_count,
                    "index": sample_idx,
                    "sample_id": sample.sample_id,
                    "error": bool(row.get("all_runs_failed")),
                }
            )
            print(
                f"[worker {worker_id}] {local_count} index={sample_idx} sample={sample.sample_id} "
                f"pass_rate={row.get('pass_rate')} difficulty={row.get('difficulty')}",
                flush=True,
            )
    return f"[worker {worker_id}] done {local_count} samples endpoint={endpoint}"


def _run_repeat(
    cfg: EvalConfig,
    runner: RolloutRunner,
    sample_idx: int,
    sample: Any,
    repeat_idx: int,
    endpoint: str,
) -> dict[str, Any]:
    if cfg.benchmark == "dataset2" and (getattr(sample, "metadata", {}) or {}).get("is_multi_qa"):
        return _run_multi_qa_repeat(cfg, runner, sample_idx, sample, repeat_idx, endpoint)
    trace_dir = _trace_dir_for_sample(runner, sample)
    try:
        trace = runner.run_sample(sample)
        result = result_from_trace_for_cleaning(cfg.benchmark, trace)
        result.update(_quality_metrics_from_trace(trace))
    except Exception as exc:
        result = error_result_for_cleaning(cfg.benchmark, sample, repr(exc))
        result.update(
            {
                "parser_ok_rate": 0.0,
                "quality_valid_rate": 0.0,
                "issue_codes": [],
                "first_issue_codes": [],
            }
        )
    result.update(
        {
            "index": sample_idx,
            "repeat_index": repeat_idx,
            "endpoint": endpoint,
            "trace_dir": str(trace_dir),
            "trace_jsonl": str(trace_dir / "trace.jsonl"),
            "trace_txt": str(trace_dir / "trace.txt"),
        }
    )
    return result


def _run_multi_qa_repeat(
    cfg: EvalConfig,
    runner: RolloutRunner,
    sample_idx: int,
    sample: Any,
    repeat_idx: int,
    endpoint: str,
) -> dict[str, Any]:
    del cfg
    trace_dir = _trace_dir_for_sample(runner, sample)
    qa_results: list[dict[str, Any]] = []
    try:
        multi_trace = runner.run_multi_qa_sample(sample)
        if multi_trace.task_failed:
            qa_results = [
                _multi_qa_error_result(sample, qa, idx, multi_trace.failure_reason)
                for idx, qa in enumerate((sample.metadata or {}).get("qa_list") or [])
                if isinstance(qa, dict)
            ]
        else:
            for trace in multi_trace.qa_traces:
                result = dataset2_adapter.score_trace(trace)
                if trace.task_failed:
                    result["error"] = trace.failure_reason or "task_failed"
                    result["error_type"] = "task_failed"
                else:
                    result["error"] = ""
                    result["error_type"] = ""
                result.update(_quality_metrics_from_trace(trace))
                meta = dict(getattr(trace.sample, "metadata", {}) or {})
                qa_id = str(meta.get("qa_id") or "")
                qa_trace_dir = trace_dir / _safe_trace_name(qa_id)
                result.update(
                    {
                        "qa_id": qa_id,
                        "qa_index": meta.get("qa_index"),
                        "question": meta.get("question", ""),
                        "answer": meta.get("answer", ""),
                        "trace_dir": str(qa_trace_dir),
                        "trace_jsonl": str(qa_trace_dir / "trace.jsonl"),
                        "trace_txt": str(qa_trace_dir / "trace.txt"),
                    }
                )
                qa_results.append(result)
    except Exception as exc:
        qa_results = [
            _multi_qa_error_result(sample, qa, idx, repr(exc))
            for idx, qa in enumerate((sample.metadata or {}).get("qa_list") or [])
            if isinstance(qa, dict)
        ]

    scored = [result for result in qa_results if result.get("score") is not None]
    qa_correct_count = sum(float(result.get("score") or 0.0) for result in scored)
    score = (qa_correct_count / len(scored)) if scored else None
    error_count = sum(1 for result in qa_results if result.get("error"))
    result = {
        "index": sample_idx,
        "repeat_index": repeat_idx,
        "sample_id": str(getattr(sample, "sample_id", "")),
        "video_id": str(getattr(sample, "video_id", "")),
        "endpoint": endpoint,
        "trace_dir": str(trace_dir),
        "prefix_trace_dir": str(trace_dir / "_prefix"),
        "score": score,
        "qa_count": len(qa_results),
        "scored_qa_count": len(scored),
        "qa_correct_count": qa_correct_count,
        "error_count": error_count,
        "error": "multi_qa_errors" if error_count == len(qa_results) and qa_results else "",
        "error_type": "multi_qa_errors" if error_count == len(qa_results) and qa_results else "",
        "parser_ok_rate": _mean([item.get("parser_ok_rate") for item in qa_results]),
        "quality_valid_rate": _mean([item.get("quality_valid_rate") for item in qa_results]),
        "qa_results": qa_results,
    }
    return result


def _multi_qa_error_result(sample: Any, qa: dict[str, Any], qa_index: int, error: str) -> dict[str, Any]:
    qa_id = str(qa.get("qa_id") or qa.get("source_annotation_id") or f"qa_{qa_index:04d}")
    return {
        "sample_id": f"{getattr(sample, 'sample_id', '')}_{qa_id}",
        "video_id": str(getattr(sample, "video_id", "")),
        "qa_id": qa_id,
        "qa_index": qa_index,
        "question": qa.get("query_text") or qa.get("question") or "",
        "answer": qa.get("answer", ""),
        "response": "",
        "prediction": None,
        "ground_truth": qa.get("ground_truth") or dataset2_adapter.expected_option_letter(qa),
        "score": 0,
        "num_steps": 0,
        "task_failed": True,
        "failure_reason": error,
        "error": error,
        "error_type": "multi_qa_sample_error",
        "parser_ok_rate": 0.0,
        "quality_valid_rate": 0.0,
        "issue_codes": [],
        "first_issue_codes": [],
    }


def _quality_metrics_from_trace(trace: Any) -> dict[str, Any]:
    transitions = list(getattr(trace, "transitions", []) or [])
    if not transitions:
        return {"parser_ok_rate": 0.0, "quality_valid_rate": 0.0, "issue_codes": [], "first_issue_codes": []}
    parser_ok = []
    quality_valid = []
    issue_codes: list[str] = []
    first_issue_codes: list[str] = []
    for transition in transitions:
        quality = getattr(transition, "quality", None)
        parser_ok.append(bool(getattr(quality, "parser_ok", False)))
        quality_valid.append(bool(getattr(quality, "valid", False)))
        codes = [str(issue.code) for issue in (getattr(quality, "issues", []) or [])]
        issue_codes.extend(codes)
        if codes and not first_issue_codes:
            first_issue_codes = codes
    total = len(transitions)
    return {
        "parser_ok_rate": sum(parser_ok) / total,
        "quality_valid_rate": sum(quality_valid) / total,
        "issue_codes": sorted(set(issue_codes)),
        "first_issue_codes": first_issue_codes,
    }


def aggregate_sample_runs(cfg: EvalConfig, index: int, sample: Any, runs: list[dict[str, Any]]) -> dict[str, Any]:
    if cfg.benchmark == "dataset2":
        return dataset2_adapter.part_row(index, sample, runs)
    scored = [run for run in runs if run.get("score") is not None]
    pass_count = sum(int(run.get("score") or 0) for run in scored)
    scored_runs = len(scored)
    pass_rate = (pass_count / scored_runs) if scored_runs else None
    parser_ok_rate = _mean([run.get("parser_ok_rate") for run in runs])
    quality_valid_rate = _mean([run.get("quality_valid_rate") for run in runs])
    answers = [str(run.get("response") or "").strip() for run in runs]
    answer_consistency = _answer_consistency(answers)
    error_count = sum(1 for run in runs if run.get("error"))
    metadata = dict(getattr(sample, "metadata", {}) or {})
    duration_seconds = _sample_duration_seconds(sample, runs)
    return {
        "index": index,
        "sample_id": str(getattr(sample, "sample_id", "")),
        "video_id": str(getattr(sample, "video_id", "")),
        "benchmark": cfg.benchmark,
        "category": metadata.get("category", ""),
        "task": metadata.get("task", ""),
        "question": _sample_question(sample),
        "ground_truth": metadata.get("ground_truth"),
        "query_timestamp": metadata.get("query_timestamp"),
        "target_timestamp": metadata.get("target_timestamp"),
        "duration_seconds": duration_seconds,
        "repeats": len(runs),
        "scored_runs": scored_runs,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "difficulty": difficulty_from_pass_rate(pass_rate),
        "parser_ok_rate": parser_ok_rate,
        "quality_valid_rate": quality_valid_rate,
        "answer_consistency": answer_consistency,
        "answers": answers,
        "score_values": [run.get("score") for run in runs],
        "error_count": error_count,
        "all_runs_failed": error_count == len(runs),
        "filter_decision": initial_filter_decision(pass_rate, parser_ok_rate, quality_valid_rate),
        "filter_reason": initial_filter_reason(pass_rate, parser_ok_rate, quality_valid_rate),
        "runs": runs,
    }


def difficulty_from_pass_rate(pass_rate: float | None) -> str:
    if pass_rate is None:
        return "unscored"
    if pass_rate >= 1.0:
        return "easy"
    if pass_rate >= 2.0 / 3.0:
        return "medium"
    if pass_rate >= 1.0 / 3.0:
        return "hard"
    return "unsolved"


def initial_filter_decision(
    pass_rate: float | None,
    parser_ok_rate: float | None,
    quality_valid_rate: float | None,
) -> str:
    if pass_rate is None:
        return "keep_review"
    parser_ok_rate = float(parser_ok_rate or 0.0)
    quality_valid_rate = float(quality_valid_rate or 0.0)
    if parser_ok_rate <= 0.0 or quality_valid_rate <= 0.0:
        return "drop_invalid"
    if pass_rate >= 1.0 and parser_ok_rate >= 1.0 and quality_valid_rate >= 1.0:
        return "keep_strong"
    if pass_rate >= 2.0 / 3.0:
        return "keep_review"
    if pass_rate <= 1.0 / 3.0:
        return "drop_weak"
    return "keep_review"


def initial_filter_reason(
    pass_rate: float | None,
    parser_ok_rate: float | None,
    quality_valid_rate: float | None,
) -> str:
    if pass_rate is None:
        return "no automatic score; requires manual review"
    if float(parser_ok_rate or 0.0) <= 0.0:
        return "all repeated runs failed parser checks"
    if float(quality_valid_rate or 0.0) <= 0.0:
        return "all repeated runs failed StreamWeave quality checks"
    return f"pass_rate={pass_rate:.3f}, parser_ok_rate={float(parser_ok_rate or 0.0):.3f}, quality_valid_rate={float(quality_valid_rate or 0.0):.3f}"


def write_cleaning_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    summary = summarize_cleaning_rows(rows)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    table_path = output_path.with_name(f"{output_path.stem}_summary.md")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    table_path.write_text(format_cleaning_summary(summary) + "\n", encoding="utf-8")
    print(format_cleaning_summary(summary), flush=True)
    print(f"Saved cleaning summary to {summary_path} and {table_path}", flush=True)


def summarize_cleaning_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_rates = [float(row["pass_rate"]) for row in rows if row.get("pass_rate") is not None]
    return {
        "count": len(rows),
        "avg_pass_rate": (sum(pass_rates) / len(pass_rates)) if pass_rates else None,
        "pass_count_counts": dict(Counter(str(row.get("pass_count", "")) for row in rows if "pass_count" in row)),
        "difficulty_counts": dict(Counter(str(row.get("difficulty", "")) for row in rows)),
        "filter_decision_counts": dict(Counter(str(row.get("filter_decision", "")) for row in rows)),
        "category_counts": dict(Counter(str(row.get("category", "")) for row in rows)),
        "task_counts": dict(Counter(str(row.get("task", "")) for row in rows)),
        "parser_ok_rate_mean": _mean([row.get("parser_ok_rate") for row in rows]),
        "quality_valid_rate_mean": _mean([row.get("quality_valid_rate") for row in rows]),
        "answer_consistency_mean": _mean([row.get("answer_consistency") for row in rows]),
    }


def format_cleaning_summary(summary: dict[str, Any]) -> str:
    lines = ["# Data Cleaning Summary", ""]
    lines.append(f"- count: {summary.get('count', 0)}")
    lines.append(f"- avg_pass_rate: {_fmt(summary.get('avg_pass_rate'))}")
    lines.append(f"- parser_ok_rate_mean: {_fmt(summary.get('parser_ok_rate_mean'))}")
    lines.append(f"- quality_valid_rate_mean: {_fmt(summary.get('quality_valid_rate_mean'))}")
    lines.append(f"- answer_consistency_mean: {_fmt(summary.get('answer_consistency_mean'))}")
    for key in ("pass_count_counts", "difficulty_counts", "filter_decision_counts", "category_counts", "task_counts"):
        lines.append("")
        lines.append(f"## {key}")
        for name, count in sorted((summary.get(key) or {}).items()):
            lines.append(f"- {name or '<empty>'}: {count}")
    return "\n".join(lines)


def _apply_cleaning_overrides(cfg: EvalConfig, args: argparse.Namespace) -> None:
    _apply_overrides(cfg, args)
    if cfg.benchmark == "dataset2":
        dataset_path = str(getattr(args, "dataset_path", "") or cfg.benchmark_args.get("dataset_path") or "").strip()
        if not dataset_path:
            raise ValueError("dataset2 cleaning requires --dataset-path or benchmark_args.dataset_path")
        path = Path(dataset_path)
        cfg.benchmark_args["dataset_path"] = str(path)
        if path.suffix == ".jsonl":
            cfg.dataset.dataset_root = str(path.parent.parent)
            cfg.dataset.dataset_name = path.parent.name
        else:
            cfg.dataset.dataset_root = str(path.parent)
            cfg.dataset.dataset_name = path.name
        cfg.dataset.video_root = ""
        cfg.dataset.frame_id_base = 0
        cfg.dataset.image_ext = "jpg"
    if getattr(args, "temperature", None) is not None:
        cfg.backend.temperature = float(args.temperature)
    if getattr(args, "top_p", None) is not None:
        cfg.backend.top_p = float(args.top_p)
    if getattr(args, "max_tokens", None):
        cfg.backend.max_tokens = int(args.max_tokens)


def _build_cleaning_manifest(cfg: EvalConfig, args: argparse.Namespace, samples: list[Any], output_path: Path) -> dict[str, Any]:
    return {
        "benchmark": cfg.benchmark,
        "total_samples": len(samples),
        "sample_ids": [str(sample.sample_id) for sample in samples],
        "limit": int(getattr(args, "limit", 0) or cfg.batch.limit or 0),
        "model": cfg.backend.model,
        "output": str(output_path),
        "benchmark_args": _jsonable(cfg.benchmark_args),
        "repeats": int(getattr(args, "repeats", 3) or 3),
        "temperature": cfg.backend.temperature,
        "top_p": cfg.backend.top_p,
    }


def _validate_cleaning_manifest(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        print(f"resume: no manifest found at {path}; validating existing rows by sample_id", flush=True)
        return
    with path.open(encoding="utf-8") as handle:
        actual = json.load(handle)
    for key in (
        "benchmark",
        "total_samples",
        "sample_ids",
        "model",
        "benchmark_args",
        "repeats",
        "temperature",
        "top_p",
    ):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"Cannot resume because {path} does not match current run for {key}. "
                "Use a fresh output directory or rerun without --resume."
            )


def _load_completed_aggregates(
    output_path: Path,
    shard_dir: Path,
    samples: list[Any],
    *,
    include_existing_output: bool,
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    for source_path in _result_sources(output_path, shard_dir, include_existing_output=include_existing_output):
        for index, result in _iter_result_rows(source_path, samples):
            completed[index] = result
    return completed


def _merge_aggregate_parts(
    cfg: EvalConfig,
    shard_dir: Path,
    output_path: Path,
    *,
    samples: list[Any],
    include_existing_output: bool,
) -> list[dict[str, Any]]:
    include_existing_output = include_existing_output and cfg.benchmark != "dataset2"
    rows_by_index: dict[int, dict[str, Any]] = {}
    for source_path in _result_sources(output_path, shard_dir, include_existing_output=include_existing_output):
        for index, result in _iter_result_rows(source_path, samples):
            rows_by_index[index] = result
    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    if len(rows) != len(samples):
        print(
            f"warning: merged {len(rows)}/{len(samples)} aggregate rows; summary will cover completed rows only",
            file=sys.stderr,
            flush=True,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            if cfg.benchmark == "dataset2":
                row = dict(row.get("annotation") or {})
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_samples_for_cleaning(cfg: EvalConfig):
    if cfg.benchmark == "dataset2":
        return dataset2_adapter.load_samples(cfg.benchmark_args)
    return runner_load_samples(cfg)


def result_from_trace_for_cleaning(benchmark: str, trace: Any) -> dict[str, Any]:
    if benchmark == "dataset2":
        result = dataset2_adapter.score_trace(trace)
        if trace.task_failed:
            result["error"] = trace.failure_reason or "task_failed"
            result["error_type"] = "task_failed"
        else:
            result["error"] = ""
            result["error_type"] = ""
        return result
    return runner_result_from_trace(benchmark, trace)


def error_result_for_cleaning(benchmark: str, sample: Any, error: str) -> dict[str, Any]:
    if benchmark == "dataset2":
        return dataset2_adapter.error_result(sample, error)
    return runner_error_result(benchmark, sample, error)


def _result_sources(output_path: Path, shard_dir: Path, *, include_existing_output: bool) -> list[Path]:
    sources = []
    if include_existing_output and output_path.exists():
        sources.append(output_path)
    sources.extend(sorted(shard_dir.glob("part_*.jsonl")))
    return sources


def _repeat_experiment_name(base: str, repeat_idx: int) -> str:
    suffix = f"repeat_{repeat_idx + 1:02d}"
    return f"{base}/{suffix}" if base else suffix


def _trace_dir_for_sample(runner: RolloutRunner, sample: Any) -> Path:
    trace_dir = Path(runner.trace_config.output_root) / runner.trace_config.experiment_name / str(sample.video_id)
    if str(sample.sample_id) != str(sample.video_id):
        trace_dir = trace_dir / str(sample.sample_id)
    return trace_dir


def _sample_question(sample: Any) -> str:
    events = list(getattr(sample, "query_events", []) or [])
    if not events:
        return ""
    return str(getattr(events[0], "text", "") or "")


def _sample_duration_seconds(sample: Any, runs: list[dict[str, Any]]) -> float | None:
    metadata = dict(getattr(sample, "metadata", {}) or {})
    for key in ("duration_seconds", "video_duration", "duration", "target_timestamp", "query_timestamp"):
        value = metadata.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    step_counts = [int(run.get("num_steps", 0) or 0) for run in runs]
    if step_counts:
        return float(max(step_counts))
    return None


def _answer_consistency(answers: list[str]) -> float | None:
    normalized = [_normalize_answer(answer) for answer in answers if answer.strip()]
    if not normalized:
        return None
    counts = Counter(normalized)
    return max(counts.values()) / len(answers)


def _normalize_answer(answer: str) -> str:
    return " ".join(str(answer).strip().lower().split())


def _safe_trace_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value).strip())
    return safe or "qa"


def _mean(values: list[Any]) -> float | None:
    nums = []
    for value in values:
        if value is None:
            continue
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            continue
    return (sum(nums) / len(nums)) if nums else None


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=("ovo", "streamingbench", "dataset2"), default="")
    parser.add_argument("--backend", choices=("mock", "openai", "vllm", "openai_compatible", "local", "gemini"), default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoints", default="")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--worker-log-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--dataset-path", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
