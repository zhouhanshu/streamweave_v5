#!/usr/bin/env python3
"""Repeated Gemini cleaning for RL query_events JSONL files.

This is the 0516 RL-data counterpart of evaluation/run_data_cleaning_3x.py:
it runs normal StreamWeave streaming inference multiple times, scores the
answer emitted at each answer_events[].time, and writes pass-rate fields back
to the original RL JSONL files.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = PROJECT_ROOT / "RL"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from backend.factory import create_backend
from streamweave.config import (
    BackendConfig,
    DatasetConfig,
    MemoryConfig,
    PostprocessConfig,
    RewardConfig,
    RuntimeConfig,
    SynthesisConfig,
    TraceConfig,
)
from streamweave.frame_store import FrameStore
from streamweave.rollout import RolloutRunner, _timestamp_to_frame_id
from streamweave.schemas import BenchmarkSample, ContentItem, QueryEvent, RolloutTrace, Transition
from streamweave_rl.scorers import score_answer


DEFAULT_RL_DIR = Path("dataset2/0516data/rl")
DEFAULT_INPUTS = [
    "rl_0514_normalized.jsonl",
    "rl_0514_unable_normalized.jsonl",
    "rl_0514_one_normalized.jsonl",
    "rl_0514_multi_normalized.jsonl",
    "rl_0515_yesno.jsonl",
]


@dataclass(frozen=True)
class SourceRow:
    source_path: Path
    source_file: str
    source_line: int
    row: dict[str, Any]


@dataclass(frozen=True)
class InferenceJob:
    source: SourceRow
    run_id: int


_THREAD_LOCAL = threading.local()
_WRITE_LOCK = threading.Lock()


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_paths(args)
    run_output = args.output
    summary_output = args.summary or run_output.with_suffix(run_output.suffix + ".summary.json")

    if args.stage in {"all", "infer"}:
        run_inference(args, input_paths, run_output)

    if args.stage in {"all", "writeback"}:
        write_back_scores(
            input_paths=input_paths,
            run_output=run_output,
            expected_repeats=args.repeats,
            allow_partial=args.allow_partial,
            write_sample_summary=args.write_sample_summary,
        )

    if args.stage in {"all", "writeback", "summarize"}:
        summary = summarize_sources(input_paths)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"[summary] saved {summary_output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-dir", type=Path, default=DEFAULT_RL_DIR)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument(
        "--stage",
        choices=["all", "infer", "writeback", "summarize"],
        default="all",
        help="infer writes per-answer run rows; writeback aggregates them into the source JSONL files.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RL_DIR / "rl_difficulty_gemini_runs.jsonl")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit over source rows before repeat expansion.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Allow writeback when not every answer has --repeats results.")
    parser.add_argument("--write-sample-summary", action="store_true", help="Also write top-level eval_* aggregate fields.")

    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--image-quality", type=int, default=85)
    parser.add_argument("--max-retries", type=int, default=2)

    parser.add_argument("--frames-per-step", type=int, default=RuntimeConfig().frames_per_step)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--memory-window-seconds", type=float, default=MemoryConfig().window_seconds)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--prompt-profile", default="teacher_eval")
    parser.add_argument("--postprocess-mode", default="eval_repair")
    parser.add_argument("--policy", default="streamweave")
    parser.add_argument("--text-scorer", choices=["gemini", "rule"], default="gemini")
    parser.add_argument("--judge-model", default="", help="Defaults to --model.")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=0.1)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--judge-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--trace-root", type=Path, default=Path("outputs/rl_difficulty_gemini_0516/traces"))
    parser.add_argument("--trace-experiment", default="rl_query_events_gemini")
    parser.add_argument("--no-trace-jsonl", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.inputs:
        path = Path(item)
        if not path.is_absolute():
            path = args.rl_dir / path
        if not path.exists():
            raise FileNotFoundError(f"missing input file: {path}")
        paths.append(path)
    return paths


def run_inference(args: argparse.Namespace, input_paths: list[Path], output_path: Path) -> None:
    sources = load_source_rows(input_paths)
    if args.limit:
        sources = sources[: args.limit]
    jobs = [InferenceJob(source=source, run_id=run_id) for source in sources for run_id in range(args.repeats)]
    expected_answer_counts = {
        (source.source_file, source.source_line): count_answer_events(source.row)
        for source in sources
    }
    completed = load_completed_jobs(output_path, expected_answer_counts) if args.resume else set()
    pending = [job for job in jobs if job_key(job) not in completed]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and output_path.exists() else "w"
    print(
        f"[infer] sources={len(sources)} jobs={len(jobs)} completed={len(completed)} pending={len(pending)} "
        f"output={output_path}",
        flush=True,
    )
    if not pending:
        return

    started = time.time()
    done = 0
    errors = 0
    with output_path.open(mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [executor.submit(run_one_job, job, args) for job in pending]
            for future in as_completed(futures):
                rows = future.result()
                with _WRITE_LOCK:
                    for row in rows:
                        out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    out.flush()
                done += 1
                if rows and rows[0].get("error"):
                    errors += 1
                if done % max(1, args.log_every) == 0 or done == len(pending):
                    rate = done / max(time.time() - started, 1e-6)
                    print(f"[infer] progress {done}/{len(pending)} errors={errors} rate={rate:.2f}/s", flush=True)


def run_one_job(job: InferenceJob, args: argparse.Namespace) -> list[dict[str, Any]]:
    source = job.source
    started = time.time()
    try:
        runner = thread_runner(args)
        runner.trace_config.experiment_name = repeat_experiment_name(args.trace_experiment, job.run_id)
        sample = sample_from_source(source, run_id=job.run_id)
        trace = runner.run_sample(sample)
        return score_trace_answer_events(
            source,
            job.run_id,
            trace,
            frames_per_step=max(1, int(args.frames_per_step)),
            sample_fps=float(args.sample_fps),
            args=args,
            trace_root=args.trace_root,
            trace_experiment=args.trace_experiment,
            latency=time.time() - started,
        )
    except Exception as exc:
        return error_rows_for_source(
            source,
            job.run_id,
            repr(exc),
            trace_root=args.trace_root,
            trace_experiment=args.trace_experiment,
            latency=time.time() - started,
        )


def thread_runner(args: argparse.Namespace) -> RolloutRunner:
    runner = getattr(_THREAD_LOCAL, "runner", None)
    if runner is not None:
        return runner

    dataset_root = PROJECT_ROOT / "dataset2"
    runtime = RuntimeConfig(
        sample_fps=float(args.sample_fps),
        frames_per_step=max(1, int(args.frames_per_step)),
        max_frames=0,
        max_steps=0,
        resolution=int(args.resolution),
    )
    dataset = DatasetConfig(
        dataset_root=str(dataset_root),
        dataset_name="",
        video_root="",
        frame_id_base=0,
        image_ext="jpg",
        jpeg_quality=95,
        overwrite_frames=False,
    )
    backend = BackendConfig(
        backend="gemini",
        model=str(args.model),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        timeout_seconds=float(args.timeout_seconds),
        image_quality=int(args.image_quality),
        max_retries=int(args.max_retries),
    )
    runner = RolloutRunner(
        backend=create_backend(backend, runtime),
        frame_store=FrameStore(dataset),
        runtime=runtime,
        trace_config=TraceConfig(
            output_root=str(args.trace_root),
            experiment_name=str(args.trace_experiment),
            write_jsonl=not bool(args.no_trace_jsonl),
        ),
        dataset_name="",
        prompt_profile=str(args.prompt_profile),
        policy=str(args.policy),
        postprocess_config=PostprocessConfig(mode=str(args.postprocess_mode)),
        reward_config=RewardConfig(),
        synthesis_config=SynthesisConfig(max_attempts=3),
        memory_config=MemoryConfig(window_seconds=float(args.memory_window_seconds)),
    )
    _THREAD_LOCAL.runner = runner
    return runner


def thread_text_judge_backend(args: argparse.Namespace):
    backend = getattr(_THREAD_LOCAL, "text_judge_backend", None)
    if backend is not None:
        return backend

    runtime = RuntimeConfig(
        sample_fps=float(args.sample_fps),
        frames_per_step=max(1, int(args.frames_per_step)),
        max_frames=0,
        max_steps=0,
        resolution=int(args.resolution),
    )
    backend_config = BackendConfig(
        backend="gemini",
        model=str(args.judge_model or args.model),
        max_tokens=int(args.judge_max_tokens),
        temperature=float(args.judge_temperature),
        top_p=float(args.judge_top_p),
        timeout_seconds=float(args.judge_timeout_seconds),
        image_quality=int(args.image_quality),
        max_retries=int(args.judge_max_retries),
    )
    backend = create_backend(backend_config, runtime)
    _THREAD_LOCAL.text_judge_backend = backend
    return backend


def sample_from_source(source: SourceRow, *, run_id: int) -> BenchmarkSample:
    row = source.row
    dataset = nonempty_str(row.get("dataset"), default="")
    if not dataset:
        raise ValueError(f"{source.source_file}:{source.source_line} missing dataset")
    sample_fps = float(row.get("sample_fps", 1.0) or 1.0)
    frame_id_base = int(row.get("frame_id_base", 0) or 0)
    if abs(sample_fps - 1.0) > 1e-6:
        raise ValueError(f"{source.source_file}:{source.source_line} sample_fps={sample_fps}; this cleaner expects 1fps")
    if frame_id_base != 0:
        raise ValueError(f"{source.source_file}:{source.source_line} frame_id_base={frame_id_base}; this cleaner expects 0")

    query_events = []
    for query in row.get("query_events") or []:
        if not isinstance(query, dict):
            continue
        query_events.append(QueryEvent(text=str(query.get("content") or "").strip(), timestamp=float(query.get("time") or 0.0)))
    if not query_events:
        raise ValueError(f"{source.source_file}:{source.source_line} has no query_events")

    sample_id = f"{Path(source.source_file).stem}_line{source.source_line:06d}_run{run_id}"
    metadata = dict(row)
    metadata.update(
        {
            "frame_dataset_name": dataset,
            "target_timestamp": float(row.get("frame_count") or max_answer_time(row) or 0.0),
            "_source_file": source.source_file,
            "_source_line": source.source_line,
            "_run_id": run_id,
        }
    )
    return BenchmarkSample(
        sample_id=sample_id,
        video_id=str(row.get("video_id") or "").strip(),
        video_path=str(row.get("video_path") or ""),
        query_events=query_events,
        metadata=metadata,
    )


def score_trace_answer_events(
    source: SourceRow,
    run_id: int,
    trace: RolloutTrace,
    *,
    frames_per_step: int,
    sample_fps: float,
    args: argparse.Namespace,
    trace_root: Path,
    trace_experiment: str,
    latency: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = source_identity(source, run_id)
    if trace.task_failed:
        return error_rows_for_source(
            source,
            run_id,
            trace.failure_reason or "task_failed",
            trace_root=trace_root,
            trace_experiment=trace_experiment,
            latency=latency,
        )
    for query_index, query in enumerate(source.row.get("query_events") or []):
        if not isinstance(query, dict):
            continue
        for answer_index, answer_event in enumerate(query.get("answer_events") or []):
            if not isinstance(answer_event, dict):
                continue
            transition = transition_at_time(
                trace,
                float(answer_event.get("time") or 0.0),
                row=source.row,
                frames_per_step=frames_per_step,
                sample_fps=sample_fps,
            )
            response = answer_from_transition(transition)
            score, status = score_answer_event(
                response=response,
                row=source.row,
                query=query,
                answer_event=answer_event,
                args=args,
            )
            rows.append(
                {
                    **base,
                    "_query_index": query_index,
                    "_answer_index": answer_index,
                    "qid": str(query.get("qid") or f"q{query_index}"),
                    "answer_time": float(answer_event.get("time") or 0.0),
                    "answer_type": str(query.get("answer_type") or ""),
                    "response": response,
                    "ground_truth": answer_ground_truth(query, answer_event),
                    "score": score,
                    "correct": bool(score >= 1.0),
                    "score_status": status,
                    "step_index": transition.step_index if transition is not None else None,
                    "step_start": transition.step_start if transition is not None else None,
                    "step_end": transition.step_end if transition is not None else None,
                    "parser_ok": bool(transition.quality.parser_ok) if transition is not None else False,
                    "quality_valid": bool(transition.quality.valid) if transition is not None else False,
                    "trace_dir": str(trace_dir_for_source(source, run_id, trace_root=trace_root, trace_experiment=trace_experiment)),
                    "latency_seconds": latency,
                    "error": "",
                    "error_type": "",
                }
            )
    return rows


def error_rows_for_source(
    source: SourceRow,
    run_id: int,
    error: str,
    *,
    trace_root: Path,
    trace_experiment: str,
    latency: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = source_identity(source, run_id)
    for query_index, query in enumerate(source.row.get("query_events") or []):
        if not isinstance(query, dict):
            continue
        for answer_index, answer_event in enumerate(query.get("answer_events") or []):
            if not isinstance(answer_event, dict):
                continue
            rows.append(
                {
                    **base,
                    "_query_index": query_index,
                    "_answer_index": answer_index,
                    "qid": str(query.get("qid") or f"q{query_index}"),
                    "answer_time": float(answer_event.get("time") or 0.0),
                    "answer_type": str(query.get("answer_type") or ""),
                    "response": "",
                    "ground_truth": answer_ground_truth(query, answer_event),
                    "score": 0.0,
                    "correct": False,
                    "score_status": "run_error",
                    "step_index": None,
                    "step_start": None,
                    "step_end": None,
                    "parser_ok": False,
                    "quality_valid": False,
                    "trace_dir": str(trace_dir_for_source(source, run_id, trace_root=trace_root, trace_experiment=trace_experiment)),
                    "latency_seconds": latency,
                    "error": error,
                    "error_type": "sample_error",
                }
            )
    return rows


def score_answer_event(
    *,
    response: str,
    row: dict[str, Any],
    query: dict[str, Any],
    answer_event: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, str]:
    if not truthy(answer_event.get("should_answer", True)):
        return (1.0 if not response.strip() else 0.0), "expected_silence"
    ground_truth = answer_ground_truth(query, answer_event)
    answer_type = str(query.get("answer_type") or "").strip().lower()
    if answer_type == "text" and str(args.text_scorer) == "gemini":
        return score_text_answer_with_gemini(
            response=response,
            ground_truth=ground_truth,
            row=row,
            query=query,
            answer_event=answer_event,
            args=args,
        )

    metadata = {**row, **query, **answer_event, "ground_truth": ground_truth}
    scorer = "mcq" if answer_type == "mcq" else "text"
    try:
        score = score_answer(response, ground_truth, scorer=scorer, metadata=metadata)
    except Exception as exc:
        return 0.0, f"scorer_error:{type(exc).__name__}"
    return min(max(float(score), 0.0), 1.0), "scored"


def score_text_answer_with_gemini(
    *,
    response: str,
    ground_truth: str,
    row: dict[str, Any],
    query: dict[str, Any],
    answer_event: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, str]:
    if not response.strip() or not ground_truth.strip():
        return 0.0, "text_judge_empty"
    try:
        backend = thread_text_judge_backend(args)
        judge_result = backend.generate(
            [ContentItem("text", text=build_text_judge_prompt(response, ground_truth, row, query, answer_event))],
            generate_kwargs={
                "temperature": float(args.judge_temperature),
                "top_p": float(args.judge_top_p),
                "max_output_tokens": int(args.judge_max_tokens),
                "response_mime_type": "application/json",
            },
        )
        correct = parse_text_judge_correct(judge_result.text)
    except Exception as exc:
        return 0.0, f"text_judge_error:{type(exc).__name__}"
    return (1.0 if correct else 0.0), ("text_judge_gemini_correct" if correct else "text_judge_gemini_incorrect")


def build_text_judge_prompt(
    response: str,
    ground_truth: str,
    row: dict[str, Any],
    query: dict[str, Any],
    answer_event: dict[str, Any],
) -> str:
    payload = {
        "dataset": row.get("dataset", ""),
        "video_id": row.get("video_id", ""),
        "qid": query.get("qid", ""),
        "question": query.get("content", ""),
        "reference_answer": ground_truth,
        "model_answer": response,
        "answer_time": answer_event.get("time"),
    }
    return (
        "You are grading a short-answer video QA response.\n"
        "Decide whether the model answer is semantically correct with respect to the reference answer and the question.\n"
        "Accept paraphrases, synonyms, different word order, and concise answers.\n"
        "Reject answers that contradict the reference, miss the key entity/action/state, answer a different question, "
        "or add incompatible hallucinated details.\n"
        "Do not require exact wording. Output only valid JSON with this schema: {\"correct\": true} or {\"correct\": false}.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_text_judge_correct(text: str) -> bool:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= start:
        raw = raw[start : end + 1]
    data = json.loads(raw)
    value = data.get("correct", data.get("is_correct", data.get("score")))
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) >= 0.5
    return str(value).strip().lower() in {"true", "yes", "y", "1", "correct"}


def answer_ground_truth(query: dict[str, Any], answer_event: dict[str, Any]) -> str:
    answer_type = str(query.get("answer_type") or "").strip().lower()
    if answer_type == "mcq":
        return str(answer_event.get("gt") or answer_event.get("answer") or answer_event.get("content") or "").strip()
    return str(answer_event.get("answer") or answer_event.get("content") or answer_event.get("gt") or "").strip()


def transition_at_time(
    trace: RolloutTrace,
    timestamp: float,
    *,
    row: dict[str, Any],
    frames_per_step: int,
    sample_fps: float,
) -> Transition | None:
    transitions = list(trace.transitions or [])
    frame_count = int(row.get("frame_count") or 0)
    if transitions and frame_count > 0:
        frame_id_base = int(row.get("frame_id_base", 0) or 0)
        frame_id = _timestamp_to_frame_id(
            float(timestamp),
            sample_fps=float(sample_fps),
            frame_id_base=frame_id_base,
            min_frame_id=frame_id_base,
            max_frame_id=frame_id_base + frame_count - 1,
        )
        step_index = max(0, (frame_id - frame_id_base) // max(1, int(frames_per_step)))
        for transition in transitions:
            if int(transition.step_index) == step_index:
                return transition
    for transition in transitions:
        if float(transition.step_end) + 1e-6 >= float(timestamp):
            return transition
    return transitions[-1] if transitions else None


def answer_from_transition(transition: Transition | None) -> str:
    if transition is None:
        return ""
    if not transition.quality.parser_ok:
        return ""
    return str(transition.raw_action.answer or "").strip()


def write_back_scores(
    *,
    input_paths: list[Path],
    run_output: Path,
    expected_repeats: int,
    allow_partial: bool,
    write_sample_summary: bool,
) -> None:
    scores = aggregate_run_output(run_output)
    missing: list[str] = []
    pending_writes: list[tuple[Path, list[dict[str, Any]]]] = []
    for input_path in input_paths:
        rows = read_jsonl(input_path)
        updated_rows = []
        for source_line, row in enumerate(rows):
            updated = dict(row)
            total_runs = 0
            total_correct = 0
            query_events = []
            for query_index, query in enumerate(updated.get("query_events") or []):
                query_out = dict(query)
                answer_events = []
                for answer_index, answer_event in enumerate(query_out.get("answer_events") or []):
                    item = dict(answer_event)
                    key = (input_path.name, source_line, query_index, answer_index)
                    runs = scores.get(key, [])
                    if len(runs) < expected_repeats:
                        missing.append(f"{input_path.name}:{source_line}:q{query_index}:a{answer_index} runs={len(runs)}")
                    correct = sum(1 for result in runs if bool(result.get("correct")))
                    run_count = len(runs)
                    item["eval_runs"] = run_count
                    item["eval_correct"] = correct
                    item["eval_pass_rate"] = (correct / run_count) if run_count else None
                    total_runs += run_count
                    total_correct += correct
                    answer_events.append(item)
                query_out["answer_events"] = answer_events
                query_events.append(query_out)
            updated["query_events"] = query_events
            if write_sample_summary:
                updated["eval_runs"] = total_runs
                updated["eval_correct"] = total_correct
                updated["eval_pass_rate"] = (total_correct / total_runs) if total_runs else None
            updated_rows.append(updated)
        pending_writes.append((input_path, updated_rows))

    if missing and not allow_partial:
        preview = "\n".join(missing[:20])
        raise RuntimeError(
            f"Refusing writeback because {len(missing)} answer events have fewer than {expected_repeats} runs. "
            f"Use --allow-partial for debug runs.\n{preview}"
        )

    for input_path, updated_rows in pending_writes:
        write_jsonl(input_path, updated_rows)
        print(f"[writeback] updated {input_path}", flush=True)


def aggregate_run_output(path: Path) -> dict[tuple[str, int, int, int], list[dict[str, Any]]]:
    scores: dict[tuple[str, int, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        key = (
            str(row.get("_source_file") or ""),
            int(row.get("_source_line")),
            int(row.get("_query_index")),
            int(row.get("_answer_index")),
        )
        run_id = int(row.get("_run_id"))
        scores[key][run_id] = row
    return {key: [runs[idx] for idx in sorted(runs)] for key, runs in scores.items()}


def summarize_sources(input_paths: list[Path]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "files": {},
        "total_rows": 0,
        "total_query_events": 0,
        "total_answer_events": 0,
        "eval_runs_counts": {},
        "eval_pass_rate_counts": {},
    }
    run_counts = Counter()
    pass_rate_counts = Counter()
    for path in input_paths:
        rows = read_jsonl(path)
        file_answer_events = 0
        file_scored = 0
        for row in rows:
            for query in row.get("query_events") or []:
                summary["total_query_events"] += 1
                for answer in query.get("answer_events") or []:
                    file_answer_events += 1
                    runs = answer.get("eval_runs")
                    pass_rate = answer.get("eval_pass_rate")
                    run_counts[str(runs)] += 1
                    pass_rate_counts[format_pass_rate(pass_rate)] += 1
                    if runs is not None:
                        file_scored += 1
        summary["files"][path.name] = {
            "rows": len(rows),
            "answer_events": file_answer_events,
            "scored_answer_events": file_scored,
        }
        summary["total_rows"] += len(rows)
        summary["total_answer_events"] += file_answer_events
    summary["eval_runs_counts"] = dict(run_counts.most_common())
    summary["eval_pass_rate_counts"] = dict(pass_rate_counts.most_common())
    return summary


def load_completed_jobs(path: Path, expected_answer_counts: dict[tuple[str, int], int]) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, int, int]] = set()
    seen_answers: dict[tuple[str, int, int], int] = Counter()
    for row in read_jsonl(path):
        key = (str(row.get("_source_file") or ""), int(row.get("_source_line")), int(row.get("_run_id")))
        seen_answers[key] += 1
    for key, count in seen_answers.items():
        expected = expected_answer_counts.get((key[0], key[1]), 0)
        if expected > 0 and count >= expected:
            completed.add(key)
    return completed


def load_source_rows(paths: list[Path]) -> list[SourceRow]:
    sources: list[SourceRow] = []
    for path in paths:
        for line_index, row in enumerate(read_jsonl(path)):
            validate_rl_row(row, source_file=path.name, source_line=line_index)
            sources.append(SourceRow(source_path=path, source_file=path.name, source_line=line_index, row=row))
    return sources


def validate_rl_row(row: dict[str, Any], *, source_file: str, source_line: int) -> None:
    if not str(row.get("dataset") or "").strip():
        raise ValueError(f"{source_file}:{source_line} missing dataset")
    if not str(row.get("video_id") or "").strip():
        raise ValueError(f"{source_file}:{source_line} missing video_id")
    frame_count = int(row.get("frame_count") or 0)
    if frame_count <= 0:
        raise ValueError(f"{source_file}:{source_line} invalid frame_count={row.get('frame_count')!r}")
    queries = row.get("query_events")
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{source_file}:{source_line} missing query_events")
    for query_index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f"{source_file}:{source_line} query_events[{query_index}] is not object")
        if not str(query.get("qid") or "").strip():
            raise ValueError(f"{source_file}:{source_line} query_events[{query_index}] missing qid")
        if not str(query.get("content") or "").strip():
            raise ValueError(f"{source_file}:{source_line} query_events[{query_index}] missing content")
        if str(query.get("answer_type") or "").strip().lower() not in {"mcq", "text"}:
            raise ValueError(f"{source_file}:{source_line} query_events[{query_index}] invalid answer_type")
        answers = query.get("answer_events")
        if not isinstance(answers, list):
            raise ValueError(f"{source_file}:{source_line} query_events[{query_index}].answer_events is not list")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def source_identity(source: SourceRow, run_id: int) -> dict[str, Any]:
    return {
        "_source_file": source.source_file,
        "_source_line": source.source_line,
        "_run_id": run_id,
        "dataset": source.row.get("dataset", ""),
        "video_id": source.row.get("video_id", ""),
        "sample_id": f"{Path(source.source_file).stem}_line{source.source_line:06d}",
        "frame_count": source.row.get("frame_count"),
    }


def job_key(job: InferenceJob) -> tuple[str, int, int]:
    return (job.source.source_file, job.source.source_line, job.run_id)


def repeat_experiment_name(base: str, run_id: int) -> str:
    return f"{base}/repeat_{run_id:02d}"


def trace_dir_for_source(source: SourceRow, run_id: int, *, trace_root: Path, trace_experiment: str) -> Path:
    return trace_root / repeat_experiment_name(trace_experiment, run_id) / str(
        source.row.get("video_id") or ""
    ) / f"{Path(source.source_file).stem}_line{source.source_line:06d}_run{run_id}"


def count_answer_events(row: dict[str, Any]) -> int:
    count = 0
    for query in row.get("query_events") or []:
        if not isinstance(query, dict):
            continue
        count += sum(1 for item in query.get("answer_events") or [] if isinstance(item, dict))
    return count


def max_answer_time(row: dict[str, Any]) -> float:
    values: list[float] = []
    for query in row.get("query_events") or []:
        for answer in query.get("answer_events") or []:
            try:
                values.append(float(answer.get("time")))
            except (TypeError, ValueError):
                pass
    return max(values) if values else 0.0


def nonempty_str(value: Any, *, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def format_pass_rate(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
