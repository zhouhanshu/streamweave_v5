"""Score VideoHallucer rollout traces.

Per-qa: official VideoHallucer match — `re.search(r'\\b{gt}\\b', pred, IGNORECASE)`.
This is lenient: if the gt word appears anywhere in pred (with word boundaries),
the answer counts as hit. We also record a separate `pred` field showing the
FIRST yes/no word found, for human diagnostics.
Per-pair: 'overall hit' iff both basic and hallucination are hit (the official
Hallucination Rate metric, equivalent to 'no-hallucination rate').
Per-subset and overall: Basic Acc / Hallucinated Acc / Overall Acc (averaged
over pairs); the per-subset numbers are macro-averaged for the overall row,
matching the official evaluation script.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


_FIRST_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def extract_yes_no(text: str) -> str:
    """Return the first 'yes' / 'no' word found (for diagnostics only).

    Does NOT decide hit/miss — see `matches_gt` for that. We surface the first
    word the model produced so a human can sanity-check unexpected behaviour
    (e.g., model said 'no' but lenient gt match still scored it 'yes' because
    pred was 'No, but actually yes').
    """
    if not text:
        return ""
    m = _FIRST_YES_NO_RE.search(text)
    return m.group(1).lower() if m else ""


def matches_gt(text: str, gt: str) -> int:
    """Official VideoHallucer scoring: does pred contain the gt word?

    Mirrors evaluations/evaluation_utils.py:
        basic_answer_pattern = r'\\b('+basic_answer+ r')\\b'
        if re.search(basic_answer_pattern, basic_predict, re.IGNORECASE):
            basic_hit = 1
    """
    if not text or gt not in {"yes", "no"}:
        return 0
    pattern = re.compile(r"\b" + re.escape(gt) + r"\b", re.IGNORECASE)
    return 1 if pattern.search(text) else 0


def score_qa_trace(trace, qa_meta: dict[str, Any]) -> dict[str, Any]:
    """Score a single qa-branch RolloutTrace produced by run_multi_qa_sample."""
    final_answer = trace.final_answer() if hasattr(trace, "final_answer") else ""
    raw_output = ""
    if getattr(trace, "transitions", None):
        last = trace.transitions[-1]
        backend = getattr(last, "backend_result", None)
        raw_output = getattr(backend, "text", "") or ""
    # Prefer the answer extracted from the model's <answer> tag; fall back to
    # raw output if that's empty (e.g., parser dropped it).
    candidate_text = final_answer or raw_output
    gt = (qa_meta.get("gt") or "").strip().lower()
    hit = matches_gt(candidate_text, gt)
    pred = extract_yes_no(candidate_text)  # diagnostic only, not used for hit
    return {
        "sample_id": trace.sample.sample_id,
        "video_id": trace.sample.video_id,
        "task": qa_meta["task"],
        "pair_id": qa_meta["pair_id"],
        "branch": qa_meta["branch"],
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
    """Compute per-pair and per-subset / overall metrics.

    Returns a dict with the structure::

        {
          "per_pair": [...],
          "per_subset": {subset: {basic_acc, halluc_acc, overall_acc, num_pairs}},
          "overall": {basic_acc, halluc_acc, overall_acc, num_pairs}
        }
    """
    # Index qa rows by (pair_id, branch). semantic_detail pairs have 2 entries
    # (basic on video A, hallucination on video B) with the same pair_id.
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    pair_task: dict[str, str] = {}
    for row in per_qa_rows:
        by_pair[row["pair_id"]][row["branch"]] = row
        pair_task[row["pair_id"]] = row["task"]

    per_pair_rows: list[dict[str, Any]] = []
    for pair_id, branches in by_pair.items():
        basic = branches.get("basic")
        hall = branches.get("hallucination")
        basic_hit = int(basic is not None and basic["hit"])
        hall_hit = int(hall is not None and hall["hit"])
        overall_hit = int(basic_hit and hall_hit)
        per_pair_rows.append(
            {
                "pair_id": pair_id,
                "task": pair_task[pair_id],
                "basic_hit": basic_hit,
                "halluc_hit": hall_hit,
                "overall_hit": overall_hit,
                "basic_present": basic is not None,
                "halluc_present": hall is not None,
                "basic_pred": basic["pred"] if basic else "",
                "halluc_pred": hall["pred"] if hall else "",
            }
        )

    # Per-subset metrics
    per_subset: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_pair_rows:
        grouped[row["task"]].append(row)
    for task, rows in grouped.items():
        n = len(rows)
        per_subset[task] = {
            "num_pairs": n,
            "basic_acc": round(sum(r["basic_hit"] for r in rows) / n, 6) if n else 0.0,
            "halluc_acc": round(sum(r["halluc_hit"] for r in rows) / n, 6) if n else 0.0,
            "overall_acc": round(sum(r["overall_hit"] for r in rows) / n, 6) if n else 0.0,
        }

    # Overall = macro-average across subsets (official protocol)
    if per_subset:
        n_sub = len(per_subset)
        overall = {
            "num_pairs": sum(s["num_pairs"] for s in per_subset.values()),
            "num_subsets": n_sub,
            "basic_acc": round(sum(s["basic_acc"] for s in per_subset.values()) / n_sub, 6),
            "halluc_acc": round(sum(s["halluc_acc"] for s in per_subset.values()) / n_sub, 6),
            "overall_acc": round(sum(s["overall_acc"] for s in per_subset.values()) / n_sub, 6),
        }
    else:
        overall = {"num_pairs": 0, "num_subsets": 0, "basic_acc": 0.0, "halluc_acc": 0.0, "overall_acc": 0.0}

    return {"per_pair": per_pair_rows, "per_subset": per_subset, "overall": overall}


def format_summary_table(agg: dict[str, Any]) -> str:
    rows = []
    header = f"{'Subset':<22}{'pairs':>8}{'basic':>10}{'halluc':>10}{'overall':>10}"
    rows.append(header)
    rows.append("-" * len(header))
    for task in sorted(agg["per_subset"]):
        s = agg["per_subset"][task]
        rows.append(
            f"{task:<22}{s['num_pairs']:>8}"
            f"{s['basic_acc']:>10.4f}{s['halluc_acc']:>10.4f}{s['overall_acc']:>10.4f}"
        )
    rows.append("-" * len(header))
    o = agg["overall"]
    rows.append(
        f"{'OVERALL (macro)':<22}{o['num_pairs']:>8}"
        f"{o['basic_acc']:>10.4f}{o['halluc_acc']:>10.4f}{o['overall_acc']:>10.4f}"
    )
    return "\n".join(rows)
