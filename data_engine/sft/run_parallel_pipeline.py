#!/usr/bin/env python3
"""Run SFT synthesis with dynamic multiprocessing workers.

Workers claim one sample at a time from a local SQLite queue. Each sample is
written as one JSON file, so interrupted runs can resume from completed files.
Global JSONL/ShareGPT files are rebuilt after workers finish.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.sft.io_utils import write_json, write_jsonl
from data_engine.sft.rollout_sft import iter_sft_sample_records
from data_engine.sft.run_pipeline import (
    make_backend,
    make_source_config,
    make_synthesis_config,
    output_paths,
    relative_path,
    run_sharegpt,
    safe_file_stem,
    sample_manifest_row,
    training_step_rows,
)
from data_engine.sft.sample_sources import load_sample_source, source_input_path


DEFAULT_INPUT = Path(
    "/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/streamweave_data/annotations_qa_filter_answered.jsonl"
)
DEFAULT_RAW_DATA_ROOT = Path(
    "/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/streamweave_data"
)
COMPLETED_STATUSES = {"accepted", "failed", "error"}


@dataclass(slots=True)
class ClaimedJob:
    task_index: int
    sample_id: str
    sample_path: Path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_dir)
    paths.sample_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / "sft_jobs.sqlite"

    if args.overwrite:
        clear_output_state(paths=paths, db_path=db_path)
        paths.sample_dir.mkdir(parents=True, exist_ok=True)

    samples = load_sample_source(make_source_config(args))
    init_queue(args=args, db_path=db_path, samples=samples)
    total = job_count(db_path)
    if total == 0:
        raise SystemExit("No SFT samples selected.")

    print(
        f"[queue] total={total} input={source_input_path(make_source_config(args))} output_dir={args.output_dir}",
        flush=True,
    )
    run_workers(args=args, db_path=db_path)
    summary = finalize_from_jobs(args=args, db_path=db_path)
    if args.sharegpt:
        share_args = argparse.Namespace(**vars(args))
        share_args.overwrite = True
        run_sharegpt(share_args, output_paths(args.output_dir))
    print(
        f"[done] accepted={summary['num_accepted_samples']} failed={summary['num_failed_samples']} "
        f"error={summary['num_error_samples']} steps={summary['num_steps']}",
        flush=True,
    )


def init_queue(*, args: argparse.Namespace, db_path: Path, samples: list[Any]) -> None:
    conn = connect_db(db_path)
    try:
        create_schema(conn)
        conn.execute("DELETE FROM jobs")
        now = time.time()
        for task_index, sample in enumerate(samples):
            sample_path = paths_sample_path(args.output_dir, task_index, sample.sample_id)
            record = load_existing_sample_record(sample_path)
            status = "pending"
            failure_reason = ""
            answer_correct = None
            usable_for_sft = None
            num_steps = None
            num_expected_steps = None
            error = ""
            if record and args.resume:
                record_status = str(record.get("status") or "")
                if record_status in COMPLETED_STATUSES:
                    status = record_status
                    if args.rerun_failed and status in {"failed", "error"}:
                        status = "pending"
                    else:
                        failure_reason = str(record.get("failure_reason") or "")
                        answer_correct = int(bool(record.get("answer_correct")))
                        usable_for_sft = int(bool(record.get("usable_for_sft")))
                        num_steps = int(record.get("num_steps", 0) or 0)
                        num_expected_steps = int(record.get("num_expected_steps", 0) or 0)
                        error = str(record.get("error") or "")
            conn.execute(
                """
                INSERT INTO jobs (
                    task_index, sample_id, video_id, qa_id, question_type, status, sample_path,
                    worker_id, started_at, updated_at, finished_at, failure_reason,
                    answer_correct, usable_for_sft, num_steps, num_expected_steps, error, attempts
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    task_index,
                    sample.sample_id,
                    sample.video_id,
                    sample.qa_id,
                    sample.task,
                    status,
                    relative_path(sample_path, args.output_dir),
                    now,
                    failure_reason,
                    answer_correct,
                    usable_for_sft,
                    num_steps,
                    num_expected_steps,
                    error,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS jobs (
            task_index INTEGER PRIMARY KEY,
            sample_id TEXT NOT NULL UNIQUE,
            video_id TEXT,
            qa_id TEXT,
            question_type TEXT,
            status TEXT NOT NULL,
            sample_path TEXT NOT NULL,
            worker_id INTEGER,
            started_at REAL,
            updated_at REAL,
            finished_at REAL,
            failure_reason TEXT,
            answer_correct INTEGER,
            usable_for_sft INTEGER,
            num_steps INTEGER,
            num_expected_steps INTEGER,
            error TEXT,
            attempts INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        """
    )
    conn.commit()


def reset_crashed_jobs(db_path: Path) -> int:
    """Move any jobs still in 'running' state to 'error' (worker was killed before finish_job)."""
    conn = connect_db(db_path)
    try:
        cursor = conn.execute(
            "UPDATE jobs SET status = 'error', failure_reason = 'worker_crash' WHERE status = 'running'"
        )
        n = cursor.rowcount
        conn.commit()
        return n
    finally:
        conn.close()


def run_workers(*, args: argparse.Namespace, db_path: Path) -> None:
    workers = [
        mp.Process(target=worker_loop, args=(worker_id, vars(args), str(db_path)), daemon=False)
        for worker_id in range(args.num_workers)
    ]
    started = time.time()
    for process in workers:
        process.start()

    try:
        render_progress_loop(db_path=db_path, workers=workers, started=started, interval=args.progress_interval)
    finally:
        for process in workers:
            process.join()

    failed = [(idx, process.exitcode) for idx, process in enumerate(workers) if process.exitcode not in {0, None}]
    if failed:
        for worker_id, exitcode in failed:
            print(f"[worker {worker_id}] exited with code {exitcode}", file=sys.stderr, flush=True)

    crashed = reset_crashed_jobs(db_path)
    if crashed:
        print(f"[pipeline] reset {crashed} crashed job(s) from running→error (worker was killed)", file=sys.stderr, flush=True)

    if failed:
        raise SystemExit(1)

    counts = status_counts(db_path)
    if counts.get("pending", 0) or counts.get("running", 0):
        raise SystemExit(f"Queue did not finish cleanly: {counts}")


def worker_loop(worker_id: int, args_data: dict[str, Any], db_path_text: str) -> None:
    args = argparse.Namespace(**args_data)
    db_path = Path(db_path_text)
    conn = connect_db(db_path)
    try:
        samples = load_sample_source(make_source_config(args))
        backend = make_backend(args)
        source_config = make_source_config(args)
        config = make_synthesis_config(args, source_config)
        while True:
            job = claim_job(conn, worker_id=worker_id, output_dir=args.output_dir)
            if job is None:
                return
            try:
                sample = samples[job.task_index]
                sample_record = next(iter_sft_sample_records([sample], backend, config))
                sample_record["path"] = relative_path(job.sample_path, args.output_dir)
                write_json_atomic(sample_record, job.sample_path, worker_id=worker_id)
                finish_job(conn, job=job, sample_record=sample_record)
            except Exception as exc:
                error_record = build_error_record(
                    sample=samples[job.task_index],
                    sample_path=job.sample_path,
                    output_dir=args.output_dir,
                    error=short_error(exc),
                )
                write_json_atomic(error_record, job.sample_path, worker_id=worker_id)
                finish_job(conn, job=job, sample_record=error_record, error=traceback.format_exc(limit=8))
    finally:
        conn.close()


def claim_job(conn: sqlite3.Connection, *, worker_id: int, output_dir: Path) -> ClaimedJob | None:
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            """
            SELECT task_index, sample_id, sample_path
            FROM jobs
            WHERE status = 'pending'
            ORDER BY task_index
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            """
            UPDATE jobs
            SET status = 'running', worker_id = ?, started_at = ?, updated_at = ?, attempts = attempts + 1
            WHERE task_index = ?
            """,
            (worker_id, now, now, row["task_index"]),
        )
        conn.commit()
        return ClaimedJob(
            task_index=int(row["task_index"]),
            sample_id=str(row["sample_id"]),
            sample_path=output_dir / str(row["sample_path"]),
        )
    except Exception:
        conn.rollback()
        raise


def finish_job(
    conn: sqlite3.Connection,
    *,
    job: ClaimedJob,
    sample_record: dict[str, Any],
    error: str = "",
) -> None:
    status = str(sample_record.get("status") or "error")
    if status not in COMPLETED_STATUSES:
        status = "error"
    now = time.time()
    conn.execute(
        """
        UPDATE jobs
        SET status = ?, updated_at = ?, finished_at = ?, failure_reason = ?, answer_correct = ?,
            usable_for_sft = ?, num_steps = ?, num_expected_steps = ?, error = ?
        WHERE task_index = ?
        """,
        (
            status,
            now,
            now,
            str(sample_record.get("failure_reason") or ""),
            int(bool(sample_record.get("answer_correct"))),
            int(bool(sample_record.get("usable_for_sft"))),
            int(sample_record.get("num_steps", 0) or 0),
            int(sample_record.get("num_expected_steps", 0) or 0),
            error or str(sample_record.get("error") or ""),
            job.task_index,
        ),
    )
    conn.commit()


def finalize_from_jobs(*, args: argparse.Namespace, db_path: Path) -> dict[str, Any]:
    paths = output_paths(args.output_dir)
    conn = connect_db(db_path)
    try:
        rows = conn.execute("SELECT * FROM jobs ORDER BY task_index").fetchall()
    finally:
        conn.close()

    manifest_rows = []
    accepted_steps = []
    accepted_samples = 0
    failed_samples = 0
    error_samples = 0
    attempted_steps = 0
    variant_rescue_rows = 0
    variant_rescue_variants = 0

    for row in rows:
        sample_path = args.output_dir / str(row["sample_path"])
        if not sample_path.exists():
            continue
        with sample_path.open(encoding="utf-8") as f:
            sample_record = json.load(f)
        if isinstance(sample_record, dict):
            sample_record["path"] = relative_path(sample_path, args.output_dir)
        manifest_rows.append(sample_manifest_row(sample_record))
        attempted_steps += int(sample_record.get("num_steps", 0) or 0)
        rows_to_export = training_step_rows(sample_record)
        if sample_record.get("usable_for_sft"):
            accepted_samples += 1
        elif sample_record.get("status") == "error":
            error_samples += 1
        else:
            failed_samples += 1
        accepted_steps.extend(rows_to_export)
        for step in rows_to_export:
            metadata = step.get("metadata") or {}
            if isinstance(metadata, dict) and metadata.get("variant_rescue"):
                variant_rescue_rows += 1
                variant_rescue_variants += 1 + len(step.get("answer_variants") or [])

    write_jsonl(manifest_rows, paths.sample_manifest)
    write_jsonl(accepted_steps, paths.intermediate)
    summary = {
        "output": str(paths.intermediate),
        "sample_manifest": str(paths.sample_manifest),
        "sample_dir": str(paths.sample_dir),
        "job_db": str(db_path),
        "num_samples": len(rows),
        "num_accepted_samples": accepted_samples,
        "num_failed_samples": failed_samples,
        "num_error_samples": error_samples,
        "num_steps": len(accepted_steps),
        "num_attempted_steps": attempted_steps,
        "num_variant_rescue_rows": variant_rescue_rows,
        "num_variant_rescue_variants": variant_rescue_variants,
        "prompt_type": args.prompt_type,
        "policy": args.policy,
        "backend": args.backend,
        "model": args.model,
        "source": args.source,
        "input": str(source_input_path(make_source_config(args))),
    }
    write_json(summary, paths.summary)
    print(
        f"[finalize] accepted {accepted_samples}/{len(rows)} sample(s), "
        f"failed={failed_samples}, error={error_samples}, saved {len(accepted_steps)} step row(s) -> {paths.intermediate}",
        flush=True,
    )
    print(f"[finalize] sample manifest -> {paths.sample_manifest}", flush=True)
    return summary


def render_progress_loop(
    *,
    db_path: Path,
    workers: list[mp.Process],
    started: float,
    interval: float,
) -> None:
    last_render = 0.0
    while True:
        now = time.time()
        alive = any(process.is_alive() for process in workers)
        if now - last_render >= max(0.5, interval) or not alive:
            print_progress(db_path=db_path, started=started, final=not alive)
            last_render = now
        if not alive:
            return
        time.sleep(0.5)


def print_progress(*, db_path: Path, started: float, final: bool = False) -> None:
    counts = status_counts(db_path)
    total = sum(counts.values())
    done = sum(counts.get(status, 0) for status in COMPLETED_STATUSES)
    running = counts.get("running", 0)
    pending = counts.get("pending", 0)
    elapsed = max(time.time() - started, 1e-6)
    rate_per_min = done / elapsed * 60.0
    remaining = pending + running
    eta_seconds = remaining / max(done / elapsed, 1e-9) if done else None
    bar = progress_bar(done, total)
    body = (
        f"[progress] {bar} {done}/{total} "
        f"accepted={counts.get('accepted', 0)} failed={counts.get('failed', 0)} "
        f"error={counts.get('error', 0)} running={running} pending={pending} "
        f"rate={rate_per_min:.2f}/min elapsed={format_duration(elapsed)} eta={format_duration(eta_seconds)}"
    )
    is_tty = sys.stdout.isatty()
    message = ("\r" + body.ljust(160)) if is_tty else body
    end = "\n" if final or not is_tty else ""
    print(message, end=end, flush=True)


def progress_bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(width * done / total)
    filled = min(width, max(0, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def status_counts(db_path: Path) -> dict[str, int]:
    conn = connect_db(db_path)
    try:
        rows = conn.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status").fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}
    finally:
        conn.close()


def job_count(db_path: Path) -> int:
    conn = connect_db(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) AS count FROM jobs").fetchone()
        return int(row["count"] if row else 0)
    finally:
        conn.close()


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def paths_sample_path(output_dir: Path, task_index: int, sample_id: str) -> Path:
    return output_dir / "samples" / f"{task_index:06d}_{safe_file_stem(sample_id)}.json"


def load_existing_sample_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            value = json.load(f)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def write_json_atomic(data: Any, path: Path, *, worker_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.worker{worker_id}.tmp")
    write_json(data, tmp_path)
    tmp_path.replace(path)


def build_error_record(*, sample: Any, sample_path: Path, output_dir: Path, error: str) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "qa_id": sample.qa_id,
        "question_type": sample.task,
        "path": relative_path(sample_path, output_dir),
        "status": "error",
        "usable_for_sft": False,
        "answer_correct": False,
        "failure_reason": "worker_exception",
        "failure_step_index": None,
        "num_steps": 0,
        "num_expected_steps": 0,
        "checks": {"all_steps_valid": False, "answer_correct": False},
        "metadata": {"answer_time": sample.answer_time, "annotation": sample.metadata},
        "steps": [],
        "error": error,
    }


def clear_output_state(*, paths: Any, db_path: Path) -> None:
    for path in (
        db_path,
        db_path.with_suffix(db_path.suffix + "-wal"),
        db_path.with_suffix(db_path.suffix + "-shm"),
        paths.intermediate,
        paths.sample_manifest,
        paths.sharegpt,
        paths.dataset_info,
        paths.summary,
    ):
        if path.exists():
            path.unlink()
    if paths.sample_dir.exists():
        for path in paths.sample_dir.glob("*.json"):
            path.unlink()


def short_error(exc: Exception, limit: int = 2000) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def format_duration(value: float | None) -> str:
    if value is None:
        return "--"
    seconds = max(0, int(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source", choices=("frames", "ovo"), default="frames")
    parser.add_argument("--raw-data-root", type=Path, default=DEFAULT_RAW_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("data_engine/sft/outputs/gemini_final_full"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true", help="Start fresh and clear this output directory's job/sample files.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    parser.add_argument("--rerun-failed", action="store_true", help="Re-run samples whose existing JSON status is failed or error.")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--sharegpt", dest="sharegpt", action="store_true", default=True)
    parser.add_argument("--no-sharegpt", dest="sharegpt", action="store_false")

    parser.add_argument("--ovo-anno-path", type=Path, default=Path("/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json"))
    parser.add_argument("--ovo-video-dir", type=Path, default=Path("/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/chunked_videos"))
    parser.add_argument("--ovo-task", default="")
    parser.add_argument("--frame-dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--frame-dataset-name", default="ovo")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=0)

    parser.add_argument("--prompt-type", choices=("teacher", "teacher_synthesis", "teacher_eval", "production"), default="teacher_synthesis")
    parser.add_argument("--policy", default="streamweave")
    parser.add_argument("--frames-per-step", dest="frames_per_step", type=int, default=5)
    parser.add_argument("--chunks-per-step", dest="frames_per_step", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--memory-window", type=float, default=180.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--keep-invalid", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-notes-per-step", type=int, default=1)
    parser.add_argument("--bridge-note-reminder-seconds", type=float, default=20.0)
    parser.add_argument("--answer-step-rollouts", type=int, default=5)
    parser.add_argument("--answer-step-temperature", type=float, default=0.3)
    parser.add_argument("--answer-step-top-p", type=float, default=0.95)

    parser.add_argument("--backend", default="gemini")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--base-url", default="http://127.0.0.1:8082/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--image-quality", type=int, default=85)
    parser.add_argument("--dataset-name", default="streamweave_sft")
    parser.add_argument(
        "--train-prompt-type",
        default="production",
        choices=("production", "teacher_synthesis", "teacher_eval", "teacher", "eval", "final", "recorded"),
    )

    args = parser.parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")
    return args


if __name__ == "__main__":
    main()
