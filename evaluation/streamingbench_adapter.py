"""StreamingBench adapter."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from evaluation.rollout_metrics import rollout_metrics_from_trace, summarize_rollout_metrics
from streamweave.schemas import BenchmarkSample, QueryEvent, RolloutTrace


ANNO_FILES = {
    "real": "questions_real.json",
    "sqa": "questions_sqa.json",
    "proactive": "questions_proactive.json",
}


def load_samples(args: dict[str, Any]) -> list[BenchmarkSample]:
    anno_dir = Path(args["anno_dir"])
    video_dir = Path(args["video_dir"])
    split = args.get("split", "real")
    limit = int(args.get("limit", 0) or 0)
    with (anno_dir / ANNO_FILES[split]).open(encoding="utf-8") as f:
        entries = json.load(f)

    samples: list[BenchmarkSample] = []
    for raw_entry in entries:
        entry_list = raw_entry if isinstance(raw_entry, list) else [raw_entry]
        for entry in entry_list:
            video_path = video_dir / os.path.basename(entry["video_path"])
            video_id = Path(entry["video_path"]).stem
            for q_idx, question in enumerate(entry.get("questions", [])):
                ts = _ts_to_seconds(question["time_stamp"])
                samples.append(
                    BenchmarkSample(
                        sample_id=f"{video_id}_{q_idx}",
                        video_id=video_id,
                        video_path=str(video_path),
                        query_events=[QueryEvent(timestamp=ts, text=_build_query(question))],
                        metadata={"question": question, "split": split},
                    )
                )
                if limit and len(samples) >= limit:
                    return samples
    return samples


def score_trace(trace: RolloutTrace) -> dict[str, Any]:
    q = trace.sample.metadata["question"]
    response = trace.final_answer()
    task_type = q.get("task_type", "")
    if task_type == "Proactive Output":
        score = None
        gt = q.get("ground_truth_output", "")
    else:
        gt = q.get("answer", "")
        score = int(_extract_mcq(response) == gt)
    result = {
        "sample_id": trace.sample.sample_id,
        "video_id": trace.sample.video_id,
        "task": task_type,
        "response": response,
        "ground_truth": gt,
        "score": score,
        "num_steps": len(trace.transitions),
        "task_failed": trace.task_failed,
        "failure_reason": trace.failure_reason,
    }
    result.update(rollout_metrics_from_trace(trace))
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in results if row.get("score") is not None]
    correct = sum(int(row.get("score") or 0) for row in scored)
    return {
        "count": len(results),
        "scored_count": len(scored),
        "correct": correct,
        "accuracy": (correct / len(scored)) if scored else None,
        "rollout_metrics": summarize_rollout_metrics(results),
    }


def _build_query(question: dict[str, Any]) -> str:
    if question.get("task_type") == "Proactive Output":
        return str(question.get("question", ""))
    options = "\n".join(question.get("options", []))
    return f"{question.get('question', '')}\n{options}\nAnswer with one option letter."


def _ts_to_seconds(ts: str) -> float:
    parts = [float(part) for part in ts.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def _extract_mcq(response: str) -> str:
    match = re.search(r"\b([A-D])\b", response.upper())
    return match.group(1) if match else ""
