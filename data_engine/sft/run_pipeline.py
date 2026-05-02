#!/usr/bin/env python3
"""Run the StreamWeave SFT synthesis pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.factory import create_backend
from streamweave.config import BackendConfig, RuntimeConfig

from data_engine.sft.export_llamafactory import dataset_info, export_sharegpt
from data_engine.sft.io_utils import read_jsonl, write_json, write_jsonl
from data_engine.sft.rollout_sft import SFTMockBackend, SFTSynthesisConfig, iter_sft_sample_records
from data_engine.sft.sample_sources import (
    SampleSourceConfig,
    load_sample_source,
    source_input_path,
    source_media_dir,
)


@dataclass(slots=True)
class PipelinePaths:
    intermediate: Path
    sample_manifest: Path
    sample_dir: Path
    sharegpt: Path
    dataset_info: Path
    summary: Path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_dir)
    stages = resolve_stages(args.stage)

    if "intermediate" in stages:
        run_intermediate(args, paths)
    if "finalize" in stages:
        run_finalize(args, paths)
    if "sharegpt" in stages:
        run_sharegpt(args, paths)


def run_intermediate(args: argparse.Namespace, paths: PipelinePaths) -> None:
    paths.sample_dir.mkdir(parents=True, exist_ok=True)
    source_config = make_source_config(args)
    samples = load_sample_source(source_config)
    backend = make_backend(args)
    config = make_synthesis_config(args, source_config)
    written = 0
    skipped = 0
    for sample in samples:
        sample_path = paths.sample_dir / f"{safe_file_stem(sample.sample_id)}.json"
        if sample_path.exists() and not args.overwrite:
            skipped += 1
            continue
        for sample_record in iter_sft_sample_records([sample], backend, config):
            sample_record["path"] = relative_path(sample_path, args.output_dir)
            write_json(sample_record, sample_path)
            written += 1
    print(
        f"[intermediate] wrote {written} sample file(s), skipped {skipped} existing sample file(s) -> {paths.sample_dir}",
        flush=True,
    )


def run_finalize(args: argparse.Namespace, paths: PipelinePaths) -> dict[str, int | str]:
    sample_records = load_sample_records(paths)
    manifest_rows = []
    accepted_steps = []
    accepted_samples = 0
    failed_samples = 0
    attempted_steps = 0
    for sample_record in sample_records:
        path = sample_record.get("path")
        if not path:
            sample_id = str(sample_record.get("sample_id") or "sample")
            sample_record["path"] = f"samples/{safe_file_stem(sample_id)}.json"
        manifest_rows.append(sample_manifest_row(sample_record))
        attempted_steps += int(sample_record.get("num_steps", 0) or 0)
        if sample_record.get("usable_for_sft"):
            accepted_samples += 1
            for row in sample_record.get("steps", []):
                if not row.get("task_failed"):
                    accepted_steps.append(row)
        else:
            failed_samples += 1

    write_jsonl(manifest_rows, paths.sample_manifest)
    write_jsonl(accepted_steps, paths.intermediate)
    summary = {
        "output": str(paths.intermediate),
        "sample_manifest": str(paths.sample_manifest),
        "sample_dir": str(paths.sample_dir),
        "num_samples": len(sample_records),
        "num_accepted_samples": accepted_samples,
        "num_failed_samples": failed_samples,
        "num_steps": len(accepted_steps),
        "num_attempted_steps": attempted_steps,
        "prompt_type": args.prompt_type,
        "policy": args.policy,
        "backend": args.backend,
        "model": args.model,
    }
    try:
        source_config = make_source_config(args)
        summary.update(
            {
                "source": source_config.source,
                "input": str(source_input_path(source_config)),
                "ovo_video_dir": str(source_config.ovo_video_dir) if source_config.source == "ovo" else "",
                "frame_dataset_root": str(source_config.frame_dataset_root) if source_config.source == "ovo" else "",
                "frame_dataset_name": source_config.frame_dataset_name if source_config.source == "ovo" else "",
            }
        )
    except Exception as exc:
        summary["source_error"] = f"{type(exc).__name__}: {exc}"
    write_json(
        summary,
        paths.summary,
    )
    print(
        f"[finalize] accepted {accepted_samples}/{len(sample_records)} sample(s), saved {len(accepted_steps)} step row(s) -> {paths.intermediate}",
        flush=True,
    )
    print(f"[finalize] sample manifest -> {paths.sample_manifest}", flush=True)
    return summary


def run_sharegpt(args: argparse.Namespace, paths: PipelinePaths) -> None:
    if paths.sharegpt.exists() and not args.overwrite:
        print(f"[sharegpt] skip existing {paths.sharegpt}", flush=True)
        return
    if not paths.sample_manifest.exists() or not paths.intermediate.exists():
        run_finalize(args, paths)
    rows = accepted_step_rows(paths)
    result = export_sharegpt(
        rows,
        paths.sharegpt,
        dataset_name=args.dataset_name,
        train_prompt_type=args.train_prompt_type,
    )
    write_json(dataset_info(args.dataset_name, paths.sharegpt.name), paths.dataset_info)
    print(
        f"[sharegpt] saved {result['count']} row(s) with train_prompt_type={args.train_prompt_type} -> {paths.sharegpt}",
        flush=True,
    )
    print(f"[sharegpt] dataset_info snippet -> {paths.dataset_info}", flush=True)


def make_backend(args: argparse.Namespace):
    if args.backend == "mock":
        return SFTMockBackend()
    return create_backend(
        BackendConfig(
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            image_quality=args.image_quality,
        ),
        RuntimeConfig(resolution=args.max_image_side),
    )


def make_source_config(args: argparse.Namespace) -> SampleSourceConfig:
    return SampleSourceConfig(
        source=args.source,
        input=args.input,
        raw_data_root=args.raw_data_root,
        ovo_anno_path=args.ovo_anno_path,
        ovo_video_dir=args.ovo_video_dir,
        ovo_task=args.ovo_task,
        frame_dataset_root=args.frame_dataset_root,
        frame_dataset_name=args.frame_dataset_name,
        fps=args.fps,
        max_frames=args.max_frames,
        offset=args.offset,
        limit=args.limit,
        sample_ids=set(args.sample_ids),
    )


def make_synthesis_config(args: argparse.Namespace, source_config: SampleSourceConfig) -> SFTSynthesisConfig:
    return SFTSynthesisConfig(
        prompt_type=args.prompt_type,
        policy=args.policy,
        frames_per_step=args.frames_per_step,
        memory_window=args.memory_window,
        max_steps=args.max_steps,
        media_dir=source_media_dir(source_config),
        keep_invalid=args.keep_invalid,
        max_attempts=args.max_attempts,
    )


def output_paths(output_dir: Path) -> PipelinePaths:
    return PipelinePaths(
        intermediate=output_dir / "sft_steps.jsonl",
        sample_manifest=output_dir / "sample_manifest.jsonl",
        sample_dir=output_dir / "samples",
        sharegpt=output_dir / "llamafactory_sharegpt.jsonl",
        dataset_info=output_dir / "dataset_info_streamweave_sft.json",
        summary=output_dir / "summary.json",
    )


def safe_file_stem(value: str) -> str:
    import re

    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "sample"


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sample_manifest_row(sample_record: dict) -> dict:
    return {
        "sample_id": sample_record.get("sample_id"),
        "video_id": sample_record.get("video_id"),
        "qa_id": sample_record.get("qa_id"),
        "question_type": sample_record.get("question_type"),
        "path": sample_record.get("path"),
        "status": sample_record.get("status"),
        "usable_for_sft": bool(sample_record.get("usable_for_sft")),
        "answer_correct": bool(sample_record.get("answer_correct")),
        "failure_reason": sample_record.get("failure_reason"),
        "failure_step_index": sample_record.get("failure_step_index"),
        "num_steps": sample_record.get("num_steps"),
        "num_expected_steps": sample_record.get("num_expected_steps"),
        "checks": sample_record.get("checks", {}),
    }


def load_sample_records(paths: PipelinePaths) -> list[dict]:
    records: list[dict] = []
    if not paths.sample_dir.exists():
        return records
    for path in sorted(paths.sample_dir.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            record = json.load(f)
        if isinstance(record, dict):
            record["path"] = relative_path(path, paths.sample_manifest.parent)
            records.append(record)
    return records


def accepted_step_rows(paths: PipelinePaths) -> list[dict]:
    if not paths.sample_manifest.exists():
        rows: list[dict] = []
        for sample_record in load_sample_records(paths):
            if sample_record.get("usable_for_sft"):
                rows.extend(row for row in sample_record.get("steps", []) if not row.get("task_failed"))
        return rows
    rows: list[dict] = []
    for manifest in read_jsonl(paths.sample_manifest):
        if not manifest.get("usable_for_sft"):
            continue
        sample_path = paths.sample_manifest.parent / str(manifest.get("path") or "")
        if not sample_path.exists():
            raise FileNotFoundError(f"Accepted sample JSON does not exist: {sample_path}")
        import json

        with sample_path.open(encoding="utf-8") as f:
            sample_record = json.load(f)
        for row in sample_record.get("steps", []):
            if not row.get("task_failed"):
                rows.append(row)
    return rows


def resolve_stages(stage: str) -> list[str]:
    if stage == "all":
        return ["intermediate", "finalize", "sharegpt"]
    return [stage]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data_engine/synthesize/outputs/annotations_qa.jsonl"))
    parser.add_argument("--source", choices=("frames", "ovo"), default="frames")
    parser.add_argument("--raw-data-root", type=Path, default=Path("raw_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_engine/sft/outputs"))
    parser.add_argument("--stage", choices=("all", "intermediate", "finalize", "sharegpt"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument(
        "--ovo-anno-path",
        type=Path,
        default=Path("/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json"),
    )
    parser.add_argument(
        "--ovo-video-dir",
        type=Path,
        default=Path("/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/chunked_videos"),
    )
    parser.add_argument("--ovo-task", default="")
    parser.add_argument("--frame-dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--frame-dataset-name", default="ovo")
    parser.add_argument("--fps", type=float, default=None)

    parser.add_argument(
        "--prompt-type",
        choices=("teacher", "teacher_synthesis", "teacher_eval", "production"),
        default="teacher_synthesis",
    )
    parser.add_argument("--policy", default="streamweave")
    parser.add_argument(
        "--frames-per-step",
        dest="frames_per_step",
        type=int,
        default=5,
        help="Number of extracted frames per model step.",
    )
    parser.add_argument("--chunks-per-step", dest="frames_per_step", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--memory-window", type=float, default=180.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--keep-invalid", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)

    parser.add_argument("--backend", default="mock")
    parser.add_argument("--model", default="mock")
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
        help="Prompt used as the ShareGPT user input. Teacher generation still uses --prompt-type.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
