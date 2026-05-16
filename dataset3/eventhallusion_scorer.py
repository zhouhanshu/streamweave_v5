"""Score EventHallusion rollout traces.

Per-qa: extract yes/no via startswith() on the trace's final answer
(matching official extract_pred in dataset3/raw/eventhallusion/eval.py).
Per-split: question-level accuracy.
Overall: macro-average across splits (matching official summary protocol).

We do NOT compute description accuracy here — that requires GPT-4o judge
on per-video model captions and lives outside this scorer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def extract_yes_no(text: str) -> str:
    """Mirror official extract_pred (dataset3/raw/eventhallusion/eval.py):

        video_llm_pred = video_llm_pred.lower()
        if video_llm_pred.startswith("yes"): return "Yes."
        elif video_llm_pred.startswith("no"): return "No."
        else: return None

    Notably NO .strip() before startswith — leading whitespace makes the
    match fail, same as official behaviour. Returns 'yes' / 'no' / ''.
    """
    if not text:
        return ""
    t = text.lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return ""


def score_qa_trace(trace, qa_meta: dict[str, Any]) -> dict[str, Any]:
    """Score a single qa-branch RolloutTrace from run_multi_qa_sample."""
    final_answer = trace.final_answer() if hasattr(trace, "final_answer") else ""
    raw_output = ""
    if getattr(trace, "transitions", None):
        last = trace.transitions[-1]
        backend = getattr(last, "backend_result", None)
        raw_output = getattr(backend, "text", "") or ""
    # Prefer <answer> tag; fall back to raw output.
    candidate_text = final_answer or raw_output
    pred = extract_yes_no(candidate_text)
    gt = (qa_meta.get("gt") or "").strip().lower()
    hit = int(pred != "" and pred == gt)
    return {
        "sample_id": trace.sample.sample_id,
        "video_id": trace.sample.video_id,
        "task": qa_meta["task"],
        "source_id": qa_meta["source_id"],
        "qa_id": qa_meta["qa_id"],
        "type": qa_meta.get("type", "hallucination"),
        "question": qa_meta["question"],
        "gt": gt,
        "pred": pred,
        "hit": hit,
        "final_answer": final_answer,
        "raw_output": raw_output,
        "task_failed": bool(getattr(trace, "task_failed", False)),
        "failure_reason": getattr(trace, "failure_reason", "") or "",
    }


def aggregate(per_qa_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-split and overall metrics.

    Returns:
      {
        "per_split": {task: {qs_count, qs_correct, qs_acc, video_count, no_match_count}},
        "overall": {qs_count, qs_correct, qs_acc (macro), qs_acc_micro, no_match_rate}
      }
    """
    per_split: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_qa_rows:
        grouped[row["task"]].append(row)

    for task, rows in grouped.items():
        n_q = len(rows)
        n_hit = sum(r["hit"] for r in rows)
        n_nomatch = sum(1 for r in rows if r["pred"] == "" and not r["task_failed"])
        n_videos = len({r["source_id"] for r in rows})
        per_split[task] = {
            "qs_count": n_q,
            "qs_correct": n_hit,
            "qs_acc": round(n_hit / n_q, 6) if n_q else 0.0,
            "video_count": n_videos,
            "no_match_count": n_nomatch,
        }

    total_q = sum(s["qs_count"] for s in per_split.values())
    total_hit = sum(s["qs_correct"] for s in per_split.values())
    total_nomatch = sum(s["no_match_count"] for s in per_split.values())
    n_split = len(per_split)
    overall = {
        "qs_count": total_q,
        "qs_correct": total_hit,
        "qs_acc_macro": round(sum(s["qs_acc"] for s in per_split.values()) / n_split, 6) if n_split else 0.0,
        "qs_acc_micro": round(total_hit / total_q, 6) if total_q else 0.0,
        "no_match_count": total_nomatch,
        "no_match_rate": round(total_nomatch / total_q, 6) if total_q else 0.0,
    }
    return {"per_split": per_split, "overall": overall}


def format_summary_table(agg: dict[str, Any]) -> str:
    rows = []
    header = f"{'Split':<14}{'videos':>9}{'questions':>11}{'correct':>10}{'qs_acc':>10}{'no_match':>10}"
    rows.append(header)
    rows.append("-" * len(header))
    for task in sorted(agg["per_split"]):
        s = agg["per_split"][task]
        rows.append(
            f"{task:<14}{s['video_count']:>9}{s['qs_count']:>11}"
            f"{s['qs_correct']:>10}{s['qs_acc']:>10.4f}{s['no_match_count']:>10}"
        )
    rows.append("-" * len(header))
    o = agg["overall"]
    rows.append(
        f"{'OVERALL (macro)':<14}{'':>9}{o['qs_count']:>11}"
        f"{o['qs_correct']:>10}{o['qs_acc_macro']:>10.4f}{o['no_match_count']:>10}"
    )
    rows.append(
        f"{'OVERALL (micro)':<14}{'':>9}{o['qs_count']:>11}"
        f"{o['qs_correct']:>10}{o['qs_acc_micro']:>10.4f}{'':>10}"
    )
    return "\n".join(rows)
