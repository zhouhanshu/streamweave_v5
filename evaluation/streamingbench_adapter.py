"""StreamingBench adapter."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.rollout_metrics import rollout_metrics_from_trace, summarize_rollout_metrics
from streamweave.ovo import OPTION_LABELS, extract_mcq
from streamweave.schemas import BenchmarkSample, QueryEvent, RolloutTrace


ANNO_FILES = {
    "real": "questions_real.json",
    "sqa": "questions_sqa.json",
    "proactive": "questions_proactive.json",
    "omni": "questions_omni_stream.json",
}


_LAST_LOAD_STATS: dict[str, Any] = {}


def load_samples(args: dict[str, Any]) -> list[BenchmarkSample]:
    anno_dir = Path(args["anno_dir"])
    video_dir = Path(args["video_dir"])
    split = args.get("split", "real")
    limit = int(args.get("limit", 0) or 0)
    task_filter = _task_filter(args.get("task_filter") or args.get("task_types"))
    if split not in ANNO_FILES:
        raise ValueError(f"Unknown StreamingBench split: {split}")
    with (anno_dir / ANNO_FILES[split]).open(encoding="utf-8") as f:
        entries = json.load(f)

    stats: dict[str, Any] = {
        "split": split,
        "task_filter": sorted(task_filter),
        "group_by_video": bool(split in {"real", "omni"} and _truthy(args.get("group_by_video", False))),
        "skipped_missing_videos": 0,
        "skipped_missing_samples": 0,
        "missing_video_files": [],
    }
    if split == "sqa":
        samples = _load_sqa_samples(entries, video_dir=video_dir, limit=limit, task_filter=task_filter)
    elif stats["group_by_video"]:
        samples = _load_grouped_samples(
            entries,
            video_dir=video_dir,
            split=split,
            limit=limit,
            stats=stats,
            task_filter=task_filter,
        )
    else:
        samples = _load_flat_samples(
            entries,
            video_dir=video_dir,
            split=split,
            limit=limit,
            stats=stats,
            task_filter=task_filter,
        )
    _LAST_LOAD_STATS.clear()
    _LAST_LOAD_STATS.update(stats)
    return samples


def _load_grouped_samples(
    entries: list[Any],
    *,
    video_dir: Path,
    split: str,
    limit: int,
    stats: dict[str, Any],
    task_filter: set[str],
) -> list[BenchmarkSample]:
    grouped: dict[str, dict[str, Any]] = {}
    question_result_index = 0
    video_question_counts: dict[str, int] = defaultdict(int)
    for raw_entry_index, raw_entry in enumerate(entries):
        entry_list = raw_entry if isinstance(raw_entry, list) else [raw_entry]
        for entry_index, entry in enumerate(entry_list):
            video_path = video_dir / os.path.basename(entry["video_path"])
            video_id = Path(entry["video_path"]).stem
            group = grouped.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "raw_entry_indices": [],
                    "questions": [],
                },
            )
            group["raw_entry_indices"].append(raw_entry_index)
            for q_idx, question in enumerate(entry.get("questions", [])):
                if not _question_matches_task_filter(question, task_filter):
                    continue
                if limit and question_result_index >= limit:
                    return _grouped_samples_from_groups(grouped, split=split, stats=stats)
                question = _normalized_question(question)
                ts = _ts_to_seconds(question["time_stamp"])
                video_question_index = video_question_counts[video_id]
                video_question_counts[video_id] += 1
                group["questions"].append(
                    {
                        "qa_id": f"q{video_question_index:03d}",
                        "sample_id": f"{video_id}_q{video_question_index:03d}",
                        "result_index": question_result_index,
                        "question": question,
                        "query_text": _build_query(question),
                        "query_timestamp": ts,
                        "target_timestamp": ts,
                        "raw_entry_index": raw_entry_index,
                        "entry_index": entry_index,
                        "question_index": q_idx,
                        "video_question_index": video_question_index,
                    }
                )
                question_result_index += 1
    return _grouped_samples_from_groups(grouped, split=split, stats=stats)


def _grouped_samples_from_groups(grouped: dict[str, dict[str, Any]], *, split: str, stats: dict[str, Any]) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    output_rows = 0
    for group_index, group in enumerate(grouped.values()):
        questions = [item for item in group["questions"] if isinstance(item, dict)]
        if not questions:
            continue
        target_timestamp = max(float(item.get("target_timestamp", 0.0) or 0.0) for item in questions)
        output_rows += len(questions)
        samples.append(
            BenchmarkSample(
                sample_id=str(group["video_id"]),
                video_id=str(group["video_id"]),
                video_path=str(group["video_path"]),
                query_events=[],
                metadata={
                    "split": split,
                    "streamingbench_grouped_eval": True,
                    "streamingbench_grouped_real": split == "real",
                    "group_index": group_index,
                    "qa_list": questions,
                    "question_count": len(questions),
                    "raw_entry_indices": sorted(set(int(idx) for idx in group.get("raw_entry_indices", []))),
                    "query_timestamp": target_timestamp,
                    "target_timestamp": target_timestamp,
                },
            )
        )
    stats["grouped_video_count"] = len(samples)
    stats["output_row_count"] = output_rows
    return samples


def _load_flat_samples(
    entries: list[Any],
    *,
    video_dir: Path,
    split: str,
    limit: int,
    stats: dict[str, Any],
    task_filter: set[str],
) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    video_question_counts: dict[str, int] = defaultdict(int)
    missing_video_files: set[str] = set()
    for raw_entry_index, raw_entry in enumerate(entries):
        entry_list = raw_entry if isinstance(raw_entry, list) else [raw_entry]
        for entry_index, entry in enumerate(entry_list):
            video_path = video_dir / os.path.basename(entry["video_path"])
            video_id = Path(entry["video_path"]).stem
            if split == "proactive" and not video_path.exists():
                missing_video_files.add(video_path.name)
                stats["skipped_missing_samples"] += len(entry.get("questions", []))
                continue
            for q_idx, question in enumerate(entry.get("questions", [])):
                if not _question_matches_task_filter(question, task_filter):
                    continue
                question = _normalized_question(question)
                ts = _ts_to_seconds(question["time_stamp"])
                query_timestamp = ts
                target_timestamp = ts
                start_timestamp = None
                if split == "proactive":
                    start_timestamp = ts
                    query_timestamp = ts + 1.0
                    target_timestamp = _ts_to_seconds(question["ground_truth_time_stamp"]) + 4.0
                video_question_index = video_question_counts[video_id]
                video_question_counts[video_id] += 1
                samples.append(
                    BenchmarkSample(
                        sample_id=f"{video_id}_q{video_question_index:03d}",
                        video_id=video_id,
                        video_path=str(video_path),
                        query_events=[QueryEvent(timestamp=query_timestamp, text=_build_query(question))],
                        metadata={
                            "question": question,
                            "split": split,
                            "raw_entry_index": raw_entry_index,
                            "entry_index": entry_index,
                            "question_index": q_idx,
                            "video_question_index": video_question_index,
                            "query_timestamp": query_timestamp,
                            "target_timestamp": target_timestamp,
                            "start_timestamp": start_timestamp,
                        },
                    )
                )
                if limit and len(samples) >= limit:
                    stats["skipped_missing_videos"] = len(missing_video_files)
                    stats["missing_video_files"] = sorted(missing_video_files)
                    return samples
    stats["skipped_missing_videos"] = len(missing_video_files)
    stats["missing_video_files"] = sorted(missing_video_files)
    return samples


def _load_sqa_samples(entries: list[Any], *, video_dir: Path, limit: int, task_filter: set[str]) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    video_question_counts: dict[str, int] = defaultdict(int)
    for sequence_index, sequence in enumerate(entries):
        entry_list = sequence if isinstance(sequence, list) else [sequence]
        history: list[dict[str, Any]] = []
        for entry_index, entry in enumerate(entry_list):
            video_path = video_dir / os.path.basename(entry["video_path"])
            video_id = Path(entry["video_path"]).stem
            for q_idx, raw_question in enumerate(entry.get("questions", [])):
                if not _question_matches_task_filter(raw_question, task_filter):
                    continue
                question = _normalized_question(raw_question)
                ts = _ts_to_seconds(question["time_stamp"])
                video_question_index = video_question_counts[video_id]
                video_question_counts[video_id] += 1
                query_events = _sqa_query_events(history, question, timestamp=ts)
                samples.append(
                    BenchmarkSample(
                        sample_id=f"{video_id}_q{video_question_index:03d}",
                        video_id=video_id,
                        video_path=str(video_path),
                        query_events=query_events,
                        metadata={
                            "question": question,
                            "split": "sqa",
                            "raw_entry_index": sequence_index,
                            "entry_index": entry_index,
                            "question_index": q_idx,
                            "video_question_index": video_question_index,
                            "sqa_history_count": len(history),
                            "query_timestamp": ts,
                            "target_timestamp": ts,
                        },
                    )
                )
                if limit and len(samples) >= limit:
                    return samples
                history.append(question)
    return samples


def score_trace(trace: RolloutTrace) -> dict[str, Any]:
    q = trace.sample.metadata["question"]
    task_type = q.get("task_type", "")
    if task_type == "Proactive Output":
        proactive_score = _score_proactive_trace(trace, q)
        response = proactive_score["response"]
        gt = q.get("ground_truth_output", "")
        score = int(bool(proactive_score["answer_correct"]))
    else:
        response = trace.final_answer()
        gt = q.get("answer", "")
        max_options = len(q.get("options") or []) or 5
        pred = _extract_mcq(response, max_options=max_options)
        gold = _extract_mcq(gt, max_options=max_options)
        score = int(bool(gold) and pred == gold)
    result = {
        "sample_id": trace.sample.sample_id,
        "video_id": trace.sample.video_id,
        "split": trace.sample.metadata.get("split", ""),
        "task": task_type,
        "required_ability": q.get("required_ability", ""),
        "response": response,
        "ground_truth": gt,
        "score": score,
        "query_timestamp": trace.sample.metadata.get("query_timestamp"),
        "target_timestamp": trace.sample.metadata.get("target_timestamp"),
        "video_question_index": trace.sample.metadata.get("video_question_index"),
        "num_steps": len(trace.transitions),
        "task_failed": trace.task_failed,
        "failure_reason": trace.failure_reason,
    }
    if task_type == "Proactive Output":
        result.update(proactive_score)
    result.update(rollout_metrics_from_trace(trace))
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in results if row.get("score") is not None]
    correct = sum(int(row.get("score") or 0) for row in scored)
    task_rows: list[dict[str, Any]] = []
    by_task: dict[str, dict[str, Any]] = {}
    task_order: list[str] = []
    for row in scored:
        task = str(row.get("task") or "")
        if task not in by_task:
            by_task[task] = {"total": 0, "correct": 0, "accuracy": 0.0}
            task_order.append(task)
        by_task[task]["total"] += 1
        by_task[task]["correct"] += int(row.get("score") or 0)
    for task in task_order:
        stats = by_task[task]
        total = int(stats["total"])
        stats["accuracy"] = (int(stats["correct"]) / total) if total else 0.0
        task_rows.append(
            {
                "task": task,
                "total": total,
                "correct": int(stats["correct"]),
                "accuracy": float(stats["accuracy"]),
            }
        )
    return {
        "count": len(results),
        "scored_count": len(scored),
        "correct": correct,
        "accuracy": (correct / len(scored)) if scored else None,
        "task_rows": task_rows,
        "by_task": by_task,
        "proactive": _summarize_proactive(results),
        "grouped_actual_rollout_metrics": _summarize_grouped_actual(results),
        "load_stats": dict(_LAST_LOAD_STATS),
        "rollout_metrics": summarize_rollout_metrics(results),
    }


def _build_query(question: dict[str, Any]) -> str:
    if question.get("task_type") == "Proactive Output":
        output = str(question.get("ground_truth_output", "")).strip()
        return (
            f"{question.get('question', '')}\n\n"
            f'Output "{output}" in <answer> only when the condition is satisfied. '
            "Before that moment, keep <answer></answer> empty. Do not answer yes/no."
        )
    options = _format_options(question.get("options", []))
    return f"{question.get('question', '')}\nOptions:\n{options}\n\nAnswer with one option letter."


def _sqa_query_events(
    history: list[dict[str, Any]],
    current_question: dict[str, Any],
    *,
    timestamp: float,
) -> list[QueryEvent]:
    events: list[QueryEvent] = []
    for question in history:
        events.append(QueryEvent(timestamp=timestamp, text=_build_sqa_history_question(question), role="q"))
        events.append(QueryEvent(timestamp=timestamp, text=str(question.get("answer", "")).strip(), role="a"))
    events.append(QueryEvent(timestamp=timestamp, text=_build_query(current_question), role="q"))
    return events


def _build_sqa_history_question(question: dict[str, Any]) -> str:
    options = _format_options(question.get("options", []))
    return f"At timestamp {question.get('time_stamp', '')}, Question: {question.get('question', '')}\nOptions:\n{options}"


def _normalized_question(question: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(question)
    if "options" in normalized:
        normalized["options"] = _normalize_options(normalized.get("options") or [])
    return normalized


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _task_filter(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw_items = value.replace(";", ",").replace("\n", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    return {str(item).strip().lower() for item in raw_items if str(item).strip()}


def _question_matches_task_filter(question: dict[str, Any], task_filter: set[str]) -> bool:
    if not task_filter:
        return True
    candidates = {
        str(question.get("task_type", "")).strip().lower(),
        str(question.get("required_ability", "")).strip().lower(),
    }
    return bool(candidates & task_filter)


def _normalize_options(options: list[Any]) -> list[str]:
    values = [str(option).strip() for option in options]
    if not values:
        return []
    if values[0].startswith("A."):
        return values
    return [
        f"{OPTION_LABELS[index]}. {value}" if index < len(OPTION_LABELS) else value
        for index, value in enumerate(values)
    ]


def _format_options(options: list[Any]) -> str:
    return "\n".join(_normalize_options(options))


def _ts_to_seconds(ts: str) -> float:
    parts = [float(part) for part in ts.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def _extract_mcq(response: str | None, *, max_options: int = 5) -> str:
    return extract_mcq(response, max_options=max_options) or ""


def _score_proactive_trace(trace: RolloutTrace, question: dict[str, Any]) -> dict[str, Any]:
    response = ""
    pred_step_index = None
    pred_step_start = None
    pred_step_end = None
    for transition in trace.transitions:
        answer = transition.applied.action.answer.strip()
        if not answer:
            continue
        response = answer
        pred_step_index = transition.step_index
        pred_step_start = float(transition.step_start)
        pred_step_end = float(transition.step_end)
        break

    gt_time = _ts_to_seconds(question["ground_truth_time_stamp"])
    window_start = gt_time - 2.0
    window_end = gt_time + 2.0
    time_correct = False
    if pred_step_start is not None and pred_step_end is not None:
        time_correct = pred_step_end >= window_start and pred_step_start <= window_end

    gt_output = str(question.get("ground_truth_output", "")).strip()
    answer_correct = bool(time_correct and gt_output and gt_output in response)
    return {
        "response": response,
        "ground_truth_timestamp": gt_time,
        "ground_truth_output": gt_output,
        "pred_step_index": pred_step_index,
        "pred_step_start": pred_step_start,
        "pred_step_end": pred_step_end,
        "time_window_start": window_start,
        "time_window_end": window_end,
        "time_correct": int(time_correct),
        "answer_correct": int(answer_correct),
    }


def _summarize_proactive(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [row for row in results if row.get("task") == "Proactive Output"]
    if not rows:
        return None
    total = len(rows)
    time_correct = sum(int(row.get("time_correct") or 0) for row in rows)
    answer_correct = sum(int(row.get("answer_correct") or 0) for row in rows)
    return {
        "total": total,
        "time_correct": time_correct,
        "time_accuracy": time_correct / total if total else None,
        "answer_correct": answer_correct,
        "answer_accuracy": answer_correct / total if total else None,
        "skipped_missing_videos": int(_LAST_LOAD_STATS.get("skipped_missing_videos") or 0),
        "skipped_missing_samples": int(_LAST_LOAD_STATS.get("skipped_missing_samples") or 0),
        "missing_video_files": list(_LAST_LOAD_STATS.get("missing_video_files") or []),
    }


def _summarize_grouped_actual(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    groups: dict[str, dict[str, Any]] = {}
    for row in results:
        if not row.get("grouped_eval"):
            continue
        group_id = str(row.get("group_sample_id") or row.get("video_id") or "")
        if not group_id or group_id in groups:
            continue
        groups[group_id] = row
    if not groups:
        return None
    rows = list(groups.values())
    total_calls = sum(int(row.get("group_actual_model_call_count") or 0) for row in rows)
    total_steps = sum(int(row.get("group_actual_num_steps") or 0) for row in rows)
    total_latency = sum(float(row.get("group_actual_total_latency_seconds") or 0.0) for row in rows)
    logical_calls = sum(int(row.get("model_call_count") or 0) for row in results if row.get("grouped_eval"))
    return {
        "num_groups": len(rows),
        "num_rows": sum(1 for row in results if row.get("grouped_eval")),
        "actual_num_steps": total_steps,
        "actual_model_call_count": total_calls,
        "actual_total_latency_seconds": total_latency,
        "logical_row_model_call_count": logical_calls,
        "estimated_call_saving_ratio": (logical_calls / total_calls) if total_calls else None,
    }
