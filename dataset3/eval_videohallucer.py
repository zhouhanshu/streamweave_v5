#!/usr/bin/env python3
"""VideoHallucer evaluation CLI.

Loads converted entries from videohallucer.json, runs RolloutRunner.run_multi_qa_sample
per entry (one prefix rollout, multiple last-step branches for basic + hallucination),
extracts yes/no from each branch, then computes per-subset and overall metrics aligned
with the official VideoHallucer scorer (Basic Acc / Hallucinated Acc / Overall Acc).

Usage:
    python dataset3/eval_videohallucer.py \\
        --config dataset3/configs/eval_videohallucer_anchor_delta.yaml \\
        --limit 5

Outputs (written under cfg.trace.output_root / cfg.trace.experiment_name):
    per_qa.jsonl      one line per (entry, branch)
    per_pair.jsonl    one line per pair (basic_hit, halluc_hit, overall_hit)
    summary.json      per-subset + overall metrics
    summary.txt       human-readable table
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.factory import create_backend  # noqa: E402
from streamweave.config import load_eval_config  # noqa: E402
from streamweave.frame_store import FrameStore  # noqa: E402
from streamweave.rollout import RolloutRunner  # noqa: E402

from dataset3.videohallucer_loader import load_samples  # noqa: E402
from dataset3.videohallucer_scorer import (  # noqa: E402
    aggregate,
    format_summary_table,
    score_qa_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=0, help="cap on entries (0 = all)")
    parser.add_argument("--task", default="", help="only run a single subset (e.g. object_relation)")
    parser.add_argument("--backend", default="", help="override cfg.backend.backend (mock/openai_compatible/...)")
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--output-name", default="", help="override cfg.trace.experiment_name")
    return parser.parse_args()


def _build_runner(cfg) -> RolloutRunner:
    backend = create_backend(cfg.backend, cfg.runtime)
    frame_store = FrameStore(cfg.dataset)
    return RolloutRunner(
        backend=backend,
        frame_store=frame_store,
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


def _qa_meta_from_sample_metadata(qa_branch_metadata: dict[str, Any]) -> dict[str, Any]:
    raw = qa_branch_metadata.get("raw_annotation") or {}
    return {
        "task": raw.get("task") or qa_branch_metadata.get("task") or "",
        "pair_id": raw.get("pair_id") or qa_branch_metadata.get("pair_id") or "",
        "branch": raw.get("branch") or qa_branch_metadata.get("qa_id") or "",
        "question": raw.get("question") or qa_branch_metadata.get("question") or "",
        "gt": raw.get("gt") or qa_branch_metadata.get("gt") or "",
    }


def main() -> int:
    args = parse_args()
    cfg = load_eval_config(args.config)

    if args.backend:
        cfg.backend.backend = args.backend
    if args.model:
        cfg.backend.model = args.model
    if args.endpoint:
        cfg.backend.base_url = args.endpoint
        cfg.backend.endpoints = []
    if args.output_name:
        cfg.trace.experiment_name = args.output_name

    output_dir = Path(cfg.trace.output_root) / cfg.trace.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    per_qa_path = output_dir / "per_qa.jsonl"
    per_pair_path = output_dir / "per_pair.jsonl"
    summary_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"

    samples = load_samples(
        cfg.benchmark_args["json_path"],
        cfg.benchmark_args["video_root"],
        sample_ids=cfg.benchmark_args.get("sample_ids") or [],
        task=args.task or cfg.benchmark_args.get("task") or "",
        limit=args.limit or int(cfg.benchmark_args.get("limit") or 0),
    )
    print(f"Loaded {len(samples)} entries (policy={cfg.policy}, backend={cfg.backend.backend}, model={cfg.backend.model})", flush=True)
    print(f"Output dir: {output_dir}", flush=True)

    runner = _build_runner(cfg)

    per_qa_rows: list[dict[str, Any]] = []
    with per_qa_path.open("w", encoding="utf-8") as f_qa:
        for entry_index, sample in enumerate(samples, 1):
            qa_list = sample.metadata.get("qa_list") or []
            try:
                multi = runner.run_multi_qa_sample(sample)
            except Exception as exc:  # noqa: BLE001
                # Emit error rows for every probe so pair-level scoring still sees them.
                err = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                for qa in qa_list:
                    row = {
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
                    per_qa_rows.append(row)
                    f_qa.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_qa.flush()
                print(f"[{entry_index}/{len(samples)}] {sample.sample_id} ERROR: {err}", flush=True)
                continue

            if multi.task_failed:
                for qa in qa_list:
                    row = {
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
                        "failure_reason": multi.failure_reason or "task_failed",
                    }
                    per_qa_rows.append(row)
                    f_qa.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_qa.flush()
                print(f"[{entry_index}/{len(samples)}] {sample.sample_id} TASK_FAILED: {multi.failure_reason}", flush=True)
                continue

            preds_summary = []
            for trace in multi.qa_traces:
                qa_meta = _qa_meta_from_sample_metadata(trace.sample.metadata)
                row = score_qa_trace(trace, qa_meta)
                per_qa_rows.append(row)
                f_qa.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_qa.flush()
                preds_summary.append(f"{row['branch']}={row['pred'] or '?'}/{row['gt']}({row['hit']})")
            print(
                f"[{entry_index}/{len(samples)}] {sample.sample_id} task={sample.metadata['task']} "
                f"{' '.join(preds_summary)}",
                flush=True,
            )

    agg = aggregate(per_qa_rows)

    with per_pair_path.open("w", encoding="utf-8") as f_pair:
        for row in agg["per_pair"]:
            f_pair.write(json.dumps(row, ensure_ascii=False) + "\n")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
