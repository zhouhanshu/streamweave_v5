"""OVO-Bench adapter and scoring."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.rollout_metrics import rollout_metrics_from_trace, summarize_rollout_metrics
from streamweave.ovo import (
    BACKWARD_TASKS,
    CATEGORY_TASKS,
    FORWARD_TASKS,
    REAL_TIME_TASKS,
    TASK_CATEGORY,
    build_forward_query,
    build_mcq_query,
    category_for_task,
    extract_mcq,
    extract_yes_no,
    is_mcq_task,
    option_letter_from_gt,
)
from streamweave.schemas import BenchmarkSample, QueryEvent, RolloutTrace


def load_samples(args: dict[str, Any]) -> list[BenchmarkSample]:
    anno_path = Path(args["anno_path"])
    video_dir = Path(args["video_dir"])
    sample_ids = {str(x) for x in args.get("sample_ids", []) if str(x)}
    task_filter = args.get("task", "")
    limit = int(args.get("limit", 0) or 0)
    with anno_path.open(encoding="utf-8") as f:
        annotations = json.load(f)

    samples: list[BenchmarkSample] = []
    for anno in annotations:
        if task_filter and anno.get("task") != task_filter:
            continue
        anno_samples = _anno_to_samples(anno, video_dir)
        if sample_ids:
            anno_id = str(anno["id"])
            if anno_id not in sample_ids:
                anno_samples = [sample for sample in anno_samples if sample.sample_id in sample_ids]
            if not anno_samples:
                continue
        samples.extend(anno_samples)
        if limit and len(samples) >= limit:
            return samples[:limit]
    return samples


def score_trace(trace: RolloutTrace) -> dict[str, Any]:
    meta = trace.sample.metadata
    task = str(meta.get("task", ""))
    response = trace.final_answer()
    ground_truth = meta.get("ground_truth")
    score = score_response(task, response, ground_truth)
    result = {
        "sample_id": trace.sample.sample_id,
        "video_id": trace.sample.video_id,
        "annotation_id": meta.get("annotation_id", trace.sample.sample_id),
        "test_index": meta.get("test_index"),
        "category": category_for_task(task),
        "task": task,
        "response": response,
        "ground_truth": ground_truth,
        "score": score,
        "num_steps": len(trace.transitions),
        "task_failed": trace.task_failed,
        "failure_reason": trace.failure_reason,
    }
    result.update(rollout_metrics_from_trace(trace))
    return result


def score_response(task: str, response: str | None, ground_truth: Any) -> int:
    if task in BACKWARD_TASKS or task in REAL_TIME_TASKS:
        return score_mcq(response, str(ground_truth))
    if task == "REC":
        return score_rec(response, ground_truth)
    if task in {"SSR", "CRR"}:
        return score_yes_no(response, ground_truth)
    raise ValueError(f"Unknown OVO task: {task}")


def score_mcq(response: str | None, ground_truth: str) -> int:
    pred = extract_mcq(response)
    return int(pred is not None and pred.upper() == ground_truth.upper())


def score_rec(response: str | None, ground_truth: Any) -> int:
    if response is None or not str(response).strip():
        return 0
    nums = re.findall(r"\d+", str(response))
    return int("".join(nums) == str(ground_truth)) if nums else 0


def score_yes_no(response: str | None, ground_truth: Any) -> int:
    pred = extract_yes_no(response)
    if pred is None:
        return 0
    gt = int(ground_truth)
    return int(pred == gt)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores_by_task: dict[str, dict[str, list[int]]] = {
        category: defaultdict(list) for category in CATEGORY_TASKS
    }
    for result in results:
        task = str(result.get("task", ""))
        if task not in TASK_CATEGORY:
            continue
        score = result.get("score")
        if score is None:
            continue
        scores_by_task[TASK_CATEGORY[task]][task].append(int(score))

    task_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for category, tasks in CATEGORY_TASKS.items():
        task_accuracies: list[float] = []
        for task in tasks:
            scores = scores_by_task[category].get(task, [])
            if not scores:
                continue
            correct = sum(scores)
            total = len(scores)
            accuracy = 100.0 * correct / total
            task_accuracies.append(accuracy)
            task_rows.append({"category": category, "task": task, "correct": correct, "total": total, "accuracy": accuracy})
        if task_accuracies:
            category_rows.append({"category": category, "task": "AVG", "correct": None, "total": None, "accuracy": sum(task_accuracies) / len(task_accuracies)})

    total_avg = sum(float(row["accuracy"]) for row in category_rows) / len(category_rows) if category_rows else None
    return {
        "task_rows": task_rows,
        "category_rows": category_rows,
        "total_avg": total_avg,
        "rollout_metrics": summarize_rollout_metrics(results),
    }


def format_summary_table(summary: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    category_rows = {row["category"]: row for row in summary.get("category_rows", [])}
    for category in CATEGORY_TASKS:
        rows.extend(row for row in summary.get("task_rows", []) if row["category"] == category)
        if category in category_rows:
            rows.append(category_rows[category])
    if summary.get("total_avg") is not None:
        rows.append(
            {
                "category": "total",
                "task": "AVG",
                "correct": None,
                "total": None,
                "accuracy": float(summary["total_avg"]),
            }
        )

    headers = ("Category", "Task", "Correct", "Total", "Accuracy")
    table_rows = [
        (
            _title(row["category"]),
            str(row["task"]),
            "-" if row["correct"] is None else str(row["correct"]),
            "-" if row["total"] is None else str(row["total"]),
            f"{float(row['accuracy']):.2f}%",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in table_rows)) if table_rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [
        "OVO-Bench Summary",
        separator,
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |",
        separator,
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))) + " |")
    lines.append(separator)
    return "\n".join(lines)


def write_summary_files(results: list[dict[str, Any]], output_path: Path) -> tuple[dict[str, Any], str, Path, Path]:
    summary = summarize_results(results)
    table = format_summary_table(summary)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    table_path = output_path.with_name(f"{output_path.stem}_summary.txt")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    table_path.write_text(table + "\n", encoding="utf-8")
    return summary, table, summary_path, table_path


def _anno_to_samples(anno: dict[str, Any], video_dir: Path) -> list[BenchmarkSample]:
    task = anno["task"]
    if task in FORWARD_TASKS:
        return [_forward_sample(anno, video_dir, idx, info) for idx, info in enumerate(anno.get("test_info", []))]
    return [_mcq_sample(anno, video_dir)]


def _forward_sample(anno: dict[str, Any], video_dir: Path, idx: int, info: dict[str, Any]) -> BenchmarkSample:
    task = anno["task"]
    target_timestamp = float(info.get("realtime", 0.0))
    # Official OVO offline evaluation creates one chunk per test_info item and
    # asks once at the chunk end, i.e. test_info.realtime. REC is intentionally
    # target-aware here: the counting target is available from the start so the
    # model can maintain a running count through QA History.
    query_timestamp = 0.0 if task == "REC" else target_timestamp
    ground_truth = info.get("count", 0) if task == "REC" else info.get("type", 0)
    sample_id = f"{anno['id']}_{idx}"
    return BenchmarkSample(
        sample_id=sample_id,
        video_id=sample_id,
        video_path=str(video_dir / f"{sample_id}.mp4"),
        query_events=[QueryEvent(timestamp=query_timestamp, text=_build_query(task, anno, idx))],
        metadata={
            "annotation_id": str(anno["id"]),
            "test_index": idx,
            "category": category_for_task(task),
            "task": task,
            "ground_truth": ground_truth,
            "query_timestamp": query_timestamp,
            "target_timestamp": target_timestamp,
        },
    )


def _mcq_sample(anno: dict[str, Any], video_dir: Path) -> BenchmarkSample:
    task = anno["task"]
    video_id = str(anno["id"])
    query_timestamp = float(anno.get("realtime", 0.0))
    return BenchmarkSample(
        sample_id=video_id,
        video_id=video_id,
        video_path=str(video_dir / f"{video_id}.mp4"),
        query_events=[QueryEvent(timestamp=query_timestamp, text=_build_query(task, anno, 0))],
        metadata={
            "annotation_id": video_id,
            "test_index": None,
            "category": category_for_task(task),
            "task": task,
            "ground_truth": option_letter_from_gt(anno["gt"]),
            "query_timestamp": query_timestamp,
            "target_timestamp": query_timestamp,
        },
    )


def _build_query(task: str, anno: dict[str, Any], index: int) -> str:
    if is_mcq_task(task):
        return build_mcq_query(str(anno["question"]), anno["options"])
    if task in FORWARD_TASKS:
        info = anno["test_info"][index] if task == "SSR" else None
        return build_forward_query(task, anno, info)
    raise ValueError(f"Unknown OVO task: {task}")


def _title(text: str) -> str:
    if text == "realtime":
        return "Realtime"
    return text.capitalize()
