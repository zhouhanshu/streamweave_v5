#!/usr/bin/env python3
"""EventHallusion evaluation CLI.

Loads converted entries from eventhallusion.json, runs RolloutRunner.run_multi_qa_sample
per entry (one prefix rollout, multiple last-step branches per question), extracts
yes/no via startswith() matching, then computes per-split and overall accuracy aligned
with the official EventHallusion eval.py (QA accuracy only — desc judgement is skipped).

Usage:
    python dataset3/eval_eventhallusion.py \\
        --config dataset3/configs/eval_eventhallusion_anchor_delta.yaml \\
        --limit 5
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

from dataset3.eventhallusion_loader import load_samples  # noqa: E402
from dataset3.eventhallusion_scorer import (  # noqa: E402
    aggregate,
    format_summary_table,
    score_qa_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task", default="", help="restrict to a single split: entire / mix / misleading")
    parser.add_argument("--backend", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--output-name", default="")
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


def _qa_meta_from_branch(branch_metadata: dict[str, Any]) -> dict[str, Any]:
    raw = branch_metadata.get("raw_annotation") or {}
    return {
        "task": raw.get("task") or branch_metadata.get("task") or "",
        "source_id": raw.get("source_id") or branch_metadata.get("source_id") or "",
        "qa_id": raw.get("qa_id") or branch_metadata.get("qa_id") or "",
        "type": raw.get("type", "hallucination"),
        "question": raw.get("question") or branch_metadata.get("question") or "",
        "gt": raw.get("gt") or branch_metadata.get("gt") or "",
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
                err = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                for qa in qa_list:
                    row = {
                        "sample_id": sample.sample_id,
                        "video_id": sample.video_id,
                        "task": qa["task"],
                        "source_id": qa["source_id"],
                        "qa_id": qa["qa_id"],
                        "type": qa.get("type", "hallucination"),
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
                        "source_id": qa["source_id"],
                        "qa_id": qa["qa_id"],
                        "type": qa.get("type", "hallucination"),
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
                qa_meta = _qa_meta_from_branch(trace.sample.metadata)
                row = score_qa_trace(trace, qa_meta)
                per_qa_rows.append(row)
                f_qa.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_qa.flush()
                preds_summary.append(f"{row['qa_id']}={row['pred'] or '?'}/{row['gt']}({row['hit']})")
            print(
                f"[{entry_index}/{len(samples)}] {sample.sample_id} task={sample.metadata['task']} "
                f"{' '.join(preds_summary)}",
                flush=True,
            )

    agg = aggregate(per_qa_rows)
    summary_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    table = format_summary_table(agg)
    summary_txt_path.write_text(table + "\n")

    print()
    print(table, flush=True)
    print()
    print(f"Wrote {per_qa_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {summary_txt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
