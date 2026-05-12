#!/usr/bin/env python3
"""Build the second-round StreamWeave RL cleaning pool.

The script starts from already student-scored dataset2 rl_tmp outputs. It does
not rerun the student 3x cleaning. Expensive Gemini stages are explicit and
cached as intermediate files so ratio sampling can be rerun cheaply.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    "outputs/data_cleaning_0510/NeXTVideo_rl_tmp/annotations_with_pass_count.jsonl",
    "outputs/data_cleaning_0510/PerceptionTest_rl_tmp/annotations_with_pass_count.jsonl",
    "outputs/data_cleaning_0510/YouTube_rl_tmp/annotations_with_pass_count.jsonl",
]
DEFAULT_GOOGLE_APPLICATION_CREDENTIALS = (
    "/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json"
)
DEFAULT_CHILD_PYTHON = "/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python"
DIFFICULTY_ORDER = ("veryhard", "hard", "medium", "easy")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = stage_paths(args.output_dir)
    stages = expand_stages(args.stage)

    if "extract_unsolved" in stages:
        extract_unsolved(args, paths)
    if "gemini_rescue" in stages:
        run_gemini_rescue(args, paths)
    if "build_candidates" in stages:
        build_candidates(args, paths)
    if "last4" in stages:
        run_last4_scoring(args, paths)
    if "build_pool" in stages:
        build_pool(args, paths)
    if "sample_ratio" in stages:
        sample_ratio(args, paths)

    write_run_manifest(args, paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student-scored-inputs",
        nargs="+",
        type=Path,
        default=[Path(item) for item in DEFAULT_INPUTS],
        help="Student-scored video-level rl_tmp jsonl files. Defaults to the three 0510 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset2/rl_cleaning_0512"),
        help="Directory for all v2 intermediate and final files.",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "extract_unsolved",
            "gemini_rescue",
            "build_candidates",
            "last4",
            "build_pool",
            "sample_ratio",
        ],
        default="all",
    )
    parser.add_argument("--python", default=default_child_python(), help="Python executable for child scripts.")
    parser.add_argument("--cleaning-config", type=Path, default=Path("configs/data_cleaning_dataset2.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset2"))
    parser.add_argument("--seed", type=int, default=511)
    parser.add_argument(
        "--google-application-credentials",
        default=DEFAULT_GOOGLE_APPLICATION_CREDENTIALS,
        help="Credentials file passed to Gemini child processes as GOOGLE_APPLICATION_CREDENTIALS.",
    )

    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--gemini-workers", type=int, default=128)
    parser.add_argument("--gemini-repeats", type=int, default=2)
    parser.add_argument("--gemini-temperature", type=float, default=0.3)
    parser.add_argument("--gemini-top-p", type=float, default=0.95)
    parser.add_argument("--gemini-max-tokens", type=int, default=2048)
    parser.add_argument("--gemini-resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--last4-model", default="gemini-2.5-flash")
    parser.add_argument("--last4-workers", type=int, default=128)
    parser.add_argument("--last4-keep-pass-rate", type=float, default=0.2)
    parser.add_argument("--last4-resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--ratio",
        default="3,3,3,1",
        help="Difficulty ratio in veryhard,hard,medium,easy order, e.g. 3,3,3,1.",
    )
    parser.add_argument(
        "--ratio-output",
        type=Path,
        default=None,
        help="Optional output path for the sampled ratio file. Default is output-dir/rl_ratio_<ratio>.jsonl.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for ratio sampling. Uses the largest whole ratio unit not exceeding this count.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing cheap intermediates.")
    parser.add_argument("--dry-run", action="store_true", help="Print external commands without running them.")
    return parser.parse_args()


def default_child_python() -> str:
    path = Path(DEFAULT_CHILD_PYTHON)
    return str(path) if path.exists() else sys.executable


def expand_stages(stage: str) -> set[str]:
    if stage == "all":
        return {
            "extract_unsolved",
            "gemini_rescue",
            "build_candidates",
            "last4",
            "build_pool",
            "sample_ratio",
        }
    return {stage}


def stage_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "unsolved": output_dir / "unsolved_for_gemini.jsonl",
        "gemini_scored": output_dir / "gemini_2x_scored.jsonl",
        "candidates": output_dir / "candidates.jsonl",
        "last4_input": output_dir / "last4_input_by_video.jsonl",
        "last4_grouped": output_dir / "last4_grouped_scored.jsonl",
        "last4_scored": output_dir / "last4_scored.jsonl",
        "pool": output_dir / "pool.jsonl",
        "manifest": output_dir / "run_manifest.json",
    }


def extract_unsolved(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    output_path = paths["unsolved"]
    if output_path.exists() and not args.overwrite:
        print(f"[extract_unsolved] reuse {output_path}")
        return

    rows_by_video: dict[tuple[str, str], dict[str, Any]] = {}
    input_counts: Counter[str] = Counter()
    unsolved_counts: Counter[str] = Counter()

    for input_path in args.student_scored_inputs:
        for row_index, row in enumerate(read_jsonl(input_path)):
            dataset = dataset_name(row)
            qa_list = [qa for qa in row.get("qa_list") or [] if isinstance(qa, dict)]
            input_counts[dataset] += len(qa_list)
            unsolved_qas = []
            for qa in qa_list:
                student_pass = int(qa.get("pass_count") or 0)
                if student_pass != 0:
                    continue
                item = dict(qa)
                item["student_pass_count"] = student_pass
                item["student_scored_input"] = str(input_path)
                item["student_source_row_index"] = row_index
                item["qa_id"] = qa_id(item, len(unsolved_qas))
                unsolved_qas.append(item)
                unsolved_counts[dataset] += 1

            if not unsolved_qas:
                continue
            key = (dataset, str(row.get("video_id") or ""))
            parent = rows_by_video.get(key)
            if parent is None:
                parent = parent_without_qa(row)
                parent["qa_list"] = []
                rows_by_video[key] = parent
            parent["qa_list"].extend(unsolved_qas)

    rows = sorted(rows_by_video.values(), key=lambda row: (dataset_name(row), str(row.get("video_id") or "")))
    for row in rows:
        qa_list = [dict(qa) for qa in row.get("qa_list") or [] if isinstance(qa, dict)]
        for idx, qa in enumerate(qa_list):
            qa["qa_index"] = idx
        row["qa_list"] = qa_list
        row["qa_count"] = len(qa_list)

    write_jsonl(output_path, rows)
    write_summary(output_path, rows, mode="video")
    print(f"[extract_unsolved] wrote {output_path} videos={len(rows)} qa={sum(len(r.get('qa_list') or []) for r in rows)}")
    print(f"[extract_unsolved] input_qa={dict(input_counts)} unsolved_qa={dict(unsolved_counts)}")


def run_gemini_rescue(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    input_path = paths["unsolved"]
    output_path = paths["gemini_scored"]
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}; run --stage extract_unsolved first.")
    ensure_dataset_root_compatible(args, input_path)

    cmd = [
        args.python,
        str(PROJECT_ROOT / "evaluation" / "run_data_cleaning_3x.py"),
        "--config",
        str(args.cleaning_config),
        "--benchmark",
        "dataset2",
        "--dataset-path",
        str(input_path),
        "--backend",
        "gemini",
        "--model",
        args.gemini_model,
        "--workers",
        str(args.gemini_workers),
        "--repeats",
        str(args.gemini_repeats),
        "--temperature",
        str(args.gemini_temperature),
        "--top-p",
        str(args.gemini_top_p),
        "--max-tokens",
        str(args.gemini_max_tokens),
        "--output",
        str(output_path),
        "--worker-log-dir",
        str(args.output_dir / "gemini_2x_worker_logs"),
    ]
    if args.gemini_resume:
        cmd.append("--resume")
    run_command(
        cmd,
        dry_run=args.dry_run,
        title="gemini_rescue",
        credentials=args.google_application_credentials,
    )


def build_candidates(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    output_path = paths["candidates"]
    if output_path.exists() and not args.overwrite:
        print(f"[build_candidates] reuse {output_path}")
        return
    gemini_path = paths["gemini_scored"]
    if not gemini_path.exists():
        raise FileNotFoundError(f"Missing {gemini_path}; run --stage gemini_rescue first.")

    flat_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for input_path in args.student_scored_inputs:
        for row_index, row in enumerate(read_jsonl(input_path)):
            qa_list = [qa for qa in row.get("qa_list") or [] if isinstance(qa, dict)]
            for qa_index, qa in enumerate(qa_list):
                student_pass = int(qa.get("pass_count") or 0)
                if student_pass <= 0:
                    continue
                difficulty = difficulty_from_student_pass(student_pass)
                item = flatten_qa_row(
                    parent=row,
                    qa=qa,
                    qa_index=qa_index,
                    difficulty=difficulty,
                    student_pass_count=student_pass,
                    gemini_pass_count=None,
                    source_row_index=row_index,
                    source_file=input_path,
                )
                if item["sample_id"] in seen_ids:
                    continue
                seen_ids.add(item["sample_id"])
                flat_rows.append(item)

    rescued = 0
    dropped_unsolved = 0
    for row_index, row in enumerate(read_jsonl(gemini_path)):
        qa_list = [qa for qa in row.get("qa_list") or [] if isinstance(qa, dict)]
        for qa_index, qa in enumerate(qa_list):
            gemini_pass = int(qa.get("pass_count") or 0)
            student_pass = int(qa.get("student_pass_count") or 0)
            if student_pass != 0:
                student_pass = 0
            if gemini_pass <= 0:
                dropped_unsolved += 1
                continue
            item = flatten_qa_row(
                parent=row,
                qa=qa,
                qa_index=qa_index,
                difficulty="veryhard",
                student_pass_count=student_pass,
                gemini_pass_count=gemini_pass,
                source_row_index=row_index,
                source_file=gemini_path,
            )
            item["gemini_repeats"] = int(row.get("repeats") or args.gemini_repeats)
            if item["sample_id"] in seen_ids:
                continue
            seen_ids.add(item["sample_id"])
            flat_rows.append(item)
            rescued += 1

    flat_rows.sort(key=flat_sort_key)
    write_jsonl(output_path, flat_rows)
    write_summary(output_path, flat_rows, mode="flat")
    print(f"[build_candidates] wrote {output_path} rows={len(flat_rows)} rescued_veryhard={rescued} dropped_unsolved={dropped_unsolved}")


def run_last4_scoring(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    candidates_path = paths["candidates"]
    grouped_input = paths["last4_input"]
    grouped_output = paths["last4_grouped"]
    flat_output = paths["last4_scored"]
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing {candidates_path}; run --stage build_candidates first.")

    if args.overwrite or not grouped_input.exists():
        candidates = read_jsonl(candidates_path)
        grouped = group_flat_rows_for_qa(candidates)
        write_jsonl(grouped_input, grouped)
        write_summary(grouped_input, grouped, mode="video")
        print(f"[last4] wrote grouped input {grouped_input} videos={len(grouped)} qa={len(candidates)}")
    else:
        print(f"[last4] reuse grouped input {grouped_input}")

    cmd = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "score_dataset2_last4_gemini.py"),
        "--input",
        str(grouped_input),
        "--output",
        str(grouped_output),
        "--image-root",
        str(args.dataset_root),
        "--model",
        args.last4_model,
        "--workers",
        str(args.last4_workers),
        "--log-every",
        "100",
    ]
    if args.last4_resume:
        cmd.append("--resume")
    run_command(
        cmd,
        dry_run=args.dry_run,
        title="last4",
        credentials=args.google_application_credentials,
    )

    if args.dry_run:
        return
    flatten_grouped_last4(grouped_output, flat_output)
    write_summary(flat_output, read_jsonl(flat_output), mode="flat")
    print(f"[last4] wrote flat scored {flat_output}")


def build_pool(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    input_path = paths["last4_scored"]
    output_path = paths["pool"]
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}; run --stage last4 first.")
    if output_path.exists() and not args.overwrite:
        print(f"[build_pool] reuse {output_path}")
        return

    rows = read_jsonl(input_path)
    rng = random.Random(args.seed)
    keep_ids: set[str] = set()

    for difficulty in DIFFICULTY_ORDER:
        bucket = [row for row in rows if row.get("difficulty_v2") == difficulty]
        static_pass = [row for row in bucket if int(row.get("last4_pass") or 0) == 1]
        static_keep_count = int(len(static_pass) * float(args.last4_keep_pass_rate))
        if static_keep_count and static_pass:
            static_keep = set(r["sample_id"] for r in rng.sample(static_pass, static_keep_count))
        else:
            static_keep = set()
        for row in bucket:
            sample_id = str(row.get("sample_id") or "")
            last4_pass = int(row.get("last4_pass") or 0)
            if last4_pass == 0 or sample_id in static_keep:
                keep_ids.add(sample_id)

    pool = []
    for row in rows:
        item = dict(row)
        sample_id = str(item.get("sample_id") or "")
        item["last4_keep"] = sample_id in keep_ids
        item["last4_keep_rule"] = "pass0_all_pass1_bucket_sample"
        if item["last4_keep"]:
            pool.append(item)

    write_jsonl(output_path, pool)
    write_summary(output_path, pool, mode="flat")
    print(f"[build_pool] wrote {output_path} rows={len(pool)}")


def sample_ratio(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    input_path = paths["pool"]
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}; run --stage build_pool first.")

    ratio_values = parse_ratio(args.ratio)
    ratio_name = "".join(str(ratio_values[name]) for name in DIFFICULTY_ORDER)
    output_path = args.ratio_output or (args.output_dir / f"rl_ratio_{ratio_name}.jsonl")

    rows = read_jsonl(input_path)
    by_diff: dict[str, list[dict[str, Any]]] = {name: [] for name in DIFFICULTY_ORDER}
    for row in rows:
        difficulty = str(row.get("difficulty_v2") or "")
        if difficulty in by_diff:
            by_diff[difficulty].append(row)

    unit_candidates = []
    for difficulty, ratio in ratio_values.items():
        if ratio <= 0:
            continue
        unit_candidates.append(len(by_diff[difficulty]) // ratio)
    unit = min(unit_candidates) if unit_candidates else 0
    if args.max_samples > 0:
        ratio_total = sum(ratio for ratio in ratio_values.values() if ratio > 0)
        capped_unit = args.max_samples // ratio_total if ratio_total else 0
        unit = min(unit, capped_unit)

    rng = random.Random(args.seed)
    selected_ids: set[str] = set()
    target_counts: dict[str, int] = {}
    for difficulty, ratio in ratio_values.items():
        target = ratio * unit
        target_counts[difficulty] = target
        bucket = by_diff[difficulty]
        if target >= len(bucket):
            selected = bucket
        else:
            selected = rng.sample(bucket, target)
        selected_ids.update(str(row.get("sample_id") or "") for row in selected)

    sampled = [dict(row) for row in rows if str(row.get("sample_id") or "") in selected_ids]
    for row in sampled:
        row["ratio_name"] = ratio_name
        row["ratio_unit"] = unit

    write_jsonl(output_path, sampled)
    write_summary(
        output_path,
        sampled,
        mode="flat",
        extra={
            "ratio": ratio_values,
            "ratio_unit": unit,
            "target_counts": target_counts,
            "max_samples": args.max_samples,
        },
    )
    print(f"[sample_ratio] wrote {output_path} rows={len(sampled)} unit={unit} targets={target_counts}")


def ensure_dataset_root_compatible(args: argparse.Namespace, input_path: Path) -> None:
    dataset_root = (PROJECT_ROOT / args.dataset_root).resolve() if not args.dataset_root.is_absolute() else args.dataset_root.resolve()
    inferred_root = input_path.resolve().parent.parent
    if inferred_root != dataset_root:
        raise ValueError(
            "Gemini full-rollout rescue uses evaluation/run_data_cleaning_3x.py, which infers dataset_root from "
            f"--dataset-path.parent.parent. Put --output-dir under {args.dataset_root} or adjust --dataset-root. "
            f"Current input={input_path}, inferred_root={inferred_root}, expected={dataset_root}."
        )


def run_command(cmd: list[str], *, dry_run: bool, title: str, credentials: str) -> None:
    env = gemini_env(credentials)
    printable = " ".join(shell_quote(part) for part in cmd)
    credentials = env.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    prefix = f"GOOGLE_APPLICATION_CREDENTIALS={shell_quote(credentials)} " if credentials else ""
    print(f"[{title}] {prefix}{printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def gemini_env(credentials: str) -> dict[str, str]:
    env = dict(os.environ)
    credentials = credentials or env.get("GOOGLE_APPLICATION_CREDENTIALS") or DEFAULT_GOOGLE_APPLICATION_CREDENTIALS
    if credentials:
        path = Path(credentials)
        if not path.exists():
            raise FileNotFoundError(
                f"Gemini credentials file does not exist: {path}. "
                "Set --google-application-credentials or export GOOGLE_APPLICATION_CREDENTIALS."
            )
        env["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    return env


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=,+-")
    if all(ch in safe for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def parent_without_qa(row: dict[str, Any]) -> dict[str, Any]:
    parent = {key: value for key, value in row.items() if key not in {"qa_list", "qa_count"}}
    normalize_video_paths(parent)
    return parent


def normalize_video_paths(row: dict[str, Any]) -> None:
    dataset = dataset_name(row)
    for key in ("video", "frames_dir"):
        value = str(row.get(key) or "").strip().strip("/")
        if value and value.startswith("video/") and dataset:
            row[key] = f"{dataset}/{value}"


def dataset_name(row: dict[str, Any]) -> str:
    return str(row.get("dataset") or row.get("source_dataset") or "").strip()


def qa_id(qa: dict[str, Any], idx: int) -> str:
    for key in ("qa_id", "sample_id", "source_annotation_id"):
        value = str(qa.get(key) or "").strip()
        if value:
            return value
    return f"qa_{idx:04d}"


def difficulty_from_student_pass(pass_count: int) -> str:
    if pass_count == 1:
        return "hard"
    if pass_count == 2:
        return "medium"
    if pass_count == 3:
        return "easy"
    raise ValueError(f"student pass_count must be 1/2/3 for non-rescued rows, got {pass_count}")


def flatten_qa_row(
    *,
    parent: dict[str, Any],
    qa: dict[str, Any],
    qa_index: int,
    difficulty: str,
    student_pass_count: int,
    gemini_pass_count: int | None,
    source_row_index: int,
    source_file: Path,
) -> dict[str, Any]:
    base = parent_without_qa(parent)
    qid = qa_id(qa, qa_index)
    sample_id = f"{base.get('video_id')}__{qid}"
    item = dict(base)
    for key, value in qa.items():
        if key in {"pass_count", "last4_pass"}:
            continue
        item[key] = value
    source_scored_file = qa.get("source_scored_file", str(source_file))
    source_scored_row_index = qa.get("source_scored_row_index", source_row_index)
    item.update(
        {
            "sample_id": sample_id,
            "qa_id": qid,
            "qa_index": int(qa.get("qa_index", qa_index) or 0),
            "question": str(qa.get("question") or qa.get("query_text") or ""),
            "answer": qa.get("answer", ""),
            "options": qa.get("options"),
            "gt": qa.get("gt", qa.get("ground_truth")),
            "pass_count": int(student_pass_count),
            "student_pass_count": int(student_pass_count),
            "gemini_pass_count": gemini_pass_count,
            "difficulty_v2": difficulty,
            "parent_qa_count": int(parent.get("qa_count") or len(parent.get("qa_list") or [])),
            "source_scored_file": str(source_scored_file),
            "source_scored_row_index": int(source_scored_row_index),
        }
    )
    if "task" not in item or not item["task"]:
        item["task"] = qa.get("task", "")
    return item


def flat_sort_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("video_id") or ""),
        int(row.get("qa_index") or 0),
        str(row.get("qa_id") or ""),
    )


def group_flat_rows_for_qa(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        dataset = dataset_name(row)
        video_id = str(row.get("video_id") or "")
        key = (dataset, video_id)
        parent = groups.get(key)
        if parent is None:
            parent = {key_name: value for key_name, value in row.items() if key_name not in qa_only_fields()}
            parent["qa_list"] = []
            groups[key] = parent
        qa = {key_name: row.get(key_name) for key_name in qa_fields_for_grouping() if key_name in row}
        qa["qa_id"] = str(row.get("qa_id") or "")
        qa["qa_index"] = int(row.get("qa_index") or len(parent["qa_list"]))
        qa["sample_id"] = str(row.get("sample_id") or "")
        qa["student_pass_count"] = int(row.get("student_pass_count") or 0)
        qa["gemini_pass_count"] = row.get("gemini_pass_count")
        qa["difficulty_v2"] = row.get("difficulty_v2")
        parent["qa_list"].append(qa)

    grouped = sorted(groups.values(), key=lambda row: (dataset_name(row), str(row.get("video_id") or "")))
    for row in grouped:
        row["qa_list"] = sorted(row["qa_list"], key=lambda qa: (int(qa.get("qa_index") or 0), str(qa.get("qa_id") or "")))
        row["qa_count"] = len(row["qa_list"])
    return grouped


def qa_only_fields() -> set[str]:
    return {
        "sample_id",
        "qa_id",
        "qa_index",
        "question",
        "answer",
        "options",
        "gt",
        "ground_truth",
        "query_text",
        "student_pass_count",
        "gemini_pass_count",
        "difficulty_v2",
        "last4_pass",
        "source_line_index",
        "source_annotation_id",
        "source_turn_index",
        "source_answer_raw",
        "source_gt_letter",
        "source_scored_file",
        "source_scored_row_index",
        "gemini_repeats",
    }


def qa_fields_for_grouping() -> tuple[str, ...]:
    return (
        "question",
        "answer",
        "options",
        "gt",
        "ground_truth",
        "task",
        "query_text",
        "source_line_index",
        "source_annotation_id",
        "source_turn_index",
        "source_answer_raw",
        "source_gt_letter",
        "pass_count",
        "source_scored_file",
        "source_scored_row_index",
        "gemini_repeats",
    )


def flatten_grouped_last4(input_path: Path, output_path: Path) -> None:
    flat: list[dict[str, Any]] = []
    for row_index, row in enumerate(read_jsonl(input_path)):
        qa_list = [qa for qa in row.get("qa_list") or [] if isinstance(qa, dict)]
        for qa_index, qa in enumerate(qa_list):
            student_pass = int(qa.get("student_pass_count") or 0)
            gemini_value = qa.get("gemini_pass_count")
            gemini_pass = int(gemini_value) if gemini_value is not None else None
            difficulty = str(qa.get("difficulty_v2") or "")
            item = flatten_qa_row(
                parent=row,
                qa=qa,
                qa_index=qa_index,
                difficulty=difficulty,
                student_pass_count=student_pass,
                gemini_pass_count=gemini_pass,
                source_row_index=row_index,
                source_file=input_path,
            )
            item["last4_pass"] = int(qa.get("last4_pass") or 0)
            flat.append(item)
    flat.sort(key=flat_sort_key)
    write_jsonl(output_path, flat)


def parse_ratio(value: str) -> dict[str, int]:
    text = value.strip()
    if "=" in text:
        out = {name: 0 for name in DIFFICULTY_ORDER}
        for part in text.split(","):
            if not part.strip():
                continue
            key, raw = part.split("=", 1)
            key = key.strip()
            if key not in out:
                raise ValueError(f"Unknown difficulty in ratio: {key}")
            out[key] = int(raw)
        return out
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != len(DIFFICULTY_ORDER):
        raise ValueError(f"--ratio needs {len(DIFFICULTY_ORDER)} values in {DIFFICULTY_ORDER} order")
    return dict(zip(DIFFICULTY_ORDER, parts, strict=True))


def write_summary(path: Path, rows: list[dict[str, Any]], *, mode: str, extra: dict[str, Any] | None = None) -> None:
    summary = summarize(rows, mode=mode)
    if extra:
        summary.update(extra)
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    flat_rows: list[dict[str, Any]]
    if mode == "video":
        flat_rows = []
        for row in rows:
            for qa in row.get("qa_list") or []:
                if not isinstance(qa, dict):
                    continue
                item = dict(row)
                item.pop("qa_list", None)
                item.update(qa)
                item["dataset"] = dataset_name(row)
                flat_rows.append(item)
    else:
        flat_rows = rows

    return {
        "rows": len(rows),
        "qa": len(flat_rows),
        "by_dataset": dict(Counter(dataset_name(row) for row in flat_rows)),
        "by_difficulty_v2": dict(Counter(str(row.get("difficulty_v2") or "") for row in flat_rows)),
        "by_student_pass_count": dict(Counter(str(row.get("student_pass_count", row.get("pass_count", ""))) for row in flat_rows)),
        "by_gemini_pass_count": dict(Counter(str(row.get("gemini_pass_count", "")) for row in flat_rows)),
        "by_last4_pass": dict(Counter(str(row.get("last4_pass", "")) for row in flat_rows)),
    }


def write_run_manifest(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    manifest = {
        "student_scored_inputs": [str(path) for path in args.student_scored_inputs],
        "output_dir": str(args.output_dir),
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "google_application_credentials": args.google_application_credentials,
        "gemini_model": args.gemini_model,
        "gemini_repeats": args.gemini_repeats,
        "last4_model": args.last4_model,
        "last4_keep_pass_rate": args.last4_keep_pass_rate,
        "ratio": args.ratio,
        "max_samples": args.max_samples,
        "paths": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
