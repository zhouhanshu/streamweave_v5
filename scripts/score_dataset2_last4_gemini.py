#!/usr/bin/env python3
"""Score dataset2 QA using only the final four video frames with Gemini.

The script preserves the input video-level JSONL structure and annotates each
QA item with one field:

  last4_pass: 1 if the answer is correct, otherwise 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.factory import create_backend
from evaluation.dataset2_adapter import expected_option_letter, normalize_text
from streamweave.config import BackendConfig, RuntimeConfig
from streamweave.ovo import extract_mcq
from streamweave.schemas import ContentItem


DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class QAJob:
    row_index: int
    qa_index: int
    row: dict[str, Any]
    qa: dict[str, Any]
    frame_paths: tuple[Path, ...]


_THREAD_LOCAL = threading.local()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    completed_rows = load_completed_rows(output_path) if args.resume else {}

    jobs: list[QAJob] = []
    output_rows: list[dict[str, Any] | None] = [None] * len(rows)
    skipped_rows = 0
    missing_frame_rows = 0
    for row_index, row in enumerate(rows):
        row_key = row_identity(row, row_index)
        if row_key in completed_rows:
            output_rows[row_index] = completed_rows[row_key]
            skipped_rows += 1
            continue
        try:
            frame_paths = tuple(last_frame_paths(row, image_root=Path(args.image_root), count=args.num_frames))
        except FileNotFoundError as exc:
            missing_frame_rows += 1
            output_rows[row_index] = annotate_row_error(row, str(exc))
            continue
        qa_list = [item for item in row.get("qa_list") or [] if isinstance(item, dict)]
        if not qa_list:
            output_rows[row_index] = annotate_row_error(row, "missing qa_list")
            continue
        for qa_index, qa in enumerate(qa_list):
            jobs.append(QAJob(row_index=row_index, qa_index=qa_index, row=row, qa=qa, frame_paths=frame_paths))

    print(
        f"[last4] input={input_path} rows={len(rows)} skipped_rows={skipped_rows} "
        f"missing_frame_rows={missing_frame_rows} qa_jobs={len(jobs)} output={output_path}",
        flush=True,
    )

    results_by_row: dict[int, dict[int, dict[str, Any]]] = {}
    started = time.time()
    done = 0
    errors = 0
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [executor.submit(score_qa_job, job, args) for job in jobs]
            for future in as_completed(futures):
                job, result = future.result()
                results_by_row.setdefault(job.row_index, {})[job.qa_index] = result
                done += 1
                if result.get("_error"):
                    errors += 1
                if done % max(1, args.log_every) == 0 or done == len(jobs):
                    rate = done / max(time.time() - started, 1e-6)
                    print(
                        f"[last4] progress {done}/{len(jobs)} errors={errors} rate={rate:.2f}/s",
                        flush=True,
                    )

    for row_index, row in enumerate(rows):
        if output_rows[row_index] is not None:
            continue
        qa_results = results_by_row.get(row_index, {})
        output_rows[row_index] = merge_row_results(row, qa_results)

    write_path = output_path
    if output_path.resolve() == input_path.resolve():
        write_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with write_path.open("w", encoding="utf-8") as out:
        for row in output_rows:
            if row is None:
                continue
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    if write_path != output_path:
        write_path.replace(output_path)

    summary = summarize_rows([row for row in output_rows if row is not None])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[last4] saved output={output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input dataset2 JSONL, usually a rl_filtered.jsonl file.")
    parser.add_argument("--output", default="", help="Output JSONL. Default: update --input in place.")
    parser.add_argument("--image-root", default="dataset2", help="Root used to resolve row['frames_dir'] or row['video'].")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--image-quality", type=int, default=85)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_index} is not an object: {path}")
            rows.append(row)
    return rows


def load_completed_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row_has_last4_pass(row):
                completed[row_identity(row, line_index)] = row
    return completed


def row_identity(row: dict[str, Any], fallback_index: int) -> str:
    return str(row.get("sample_id") or row.get("video_id") or fallback_index)


def last_frame_paths(row: dict[str, Any], *, image_root: Path, count: int) -> list[Path]:
    frame_dir = resolve_frame_dir(row, image_root=image_root)
    paths = sorted_frame_paths(frame_dir)
    if not paths:
        raise FileNotFoundError(f"no frames found under {frame_dir}")
    return paths[-max(1, count) :]


def resolve_frame_dir(row: dict[str, Any], *, image_root: Path) -> Path:
    for key in ("frames_dir", "video"):
        value = str(row.get(key) or "").strip().strip("/")
        if not value:
            continue
        path = Path(value)
        if path.is_absolute():
            return path
        return image_root / path
    dataset = str(row.get("dataset") or row.get("source_dataset") or "").strip()
    video_id = str(row.get("video_id") or "").strip()
    if dataset and video_id:
        return image_root / dataset / "video" / video_id
    raise FileNotFoundError("row has no frames_dir/video and no dataset+video_id fallback")


def sorted_frame_paths(frame_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(frame_dir.glob(pattern))
    return sorted(paths, key=frame_sort_key)


def frame_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return (int(digits), path.name)
    return (0, path.name)


def score_qa_job(job: QAJob, args: argparse.Namespace) -> tuple[QAJob, dict[str, Any]]:
    expected = expected_option_letter(job.qa)
    if expected is None:
        return job, {"last4_pass": 0, "_error": "cannot resolve expected option"}
    try:
        response = gemini_generate(prompt_content(job), args)
        prediction = prediction_letter(response, job.qa)
        return job, {"last4_pass": int(prediction == expected)}
    except Exception as exc:
        return job, {"last4_pass": 0, "_error": f"{type(exc).__name__}: {exc}"}


def gemini_generate(content: list[ContentItem], args: argparse.Namespace) -> str:
    backend = get_thread_backend(args)
    result = backend.generate(content)
    return result.text.strip()


def get_thread_backend(args: argparse.Namespace):
    backend = getattr(_THREAD_LOCAL, "backend", None)
    if backend is not None:
        return backend
    backend_config = BackendConfig(
        backend="gemini",
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_seconds=args.timeout_seconds,
        image_quality=args.image_quality,
        max_retries=args.max_retries,
    )
    runtime_config = RuntimeConfig(resolution=args.resolution)
    backend = create_backend(backend_config, runtime_config)
    _THREAD_LOCAL.backend = backend
    return backend


def prompt_content(job: QAJob) -> list[ContentItem]:
    qa = job.qa
    options = qa.get("options") or []
    option_lines = "\n".join(f"{chr(ord('A') + idx)}. {str(option).strip()}" for idx, option in enumerate(options))
    text = (
        "You are answering a multiple-choice video question using only the final four frames provided below. "
        "The frames are in chronological order from older to newer. "
        "Answer with exactly one option letter and no explanation.\n\n"
        f"Question: {str(qa.get('question') or '').strip()}\n"
        f"Options:\n{option_lines}\n\n"
        "Final answer:"
    )
    content = [ContentItem(type="text", text=text)]
    for idx, path in enumerate(job.frame_paths, start=1):
        content.append(ContentItem(type="text", text=f"Frame {idx}:"))
        content.append(ContentItem(type="image", image_path=path))
    return content


def prediction_letter(response: str, qa: dict[str, Any]) -> str | None:
    options = qa.get("options") or []
    prediction = extract_mcq(response, max_options=len(options))
    if prediction is not None:
        return prediction.upper()
    response_norm = normalize_text(response)
    for idx, option in enumerate(options):
        if response_norm == normalize_text(option):
            return chr(ord("A") + idx)
    return None


def merge_row_results(row: dict[str, Any], qa_results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    out = dict(row)
    qa_list = [dict(item) for item in row.get("qa_list") or [] if isinstance(item, dict)]
    for qa_index, qa in enumerate(qa_list):
        result = qa_results.get(qa_index)
        if result is None:
            qa["last4_pass"] = 0
        else:
            qa["last4_pass"] = int(result.get("last4_pass") or 0)
    out["qa_list"] = qa_list
    return out


def annotate_row_error(row: dict[str, Any], error: str) -> dict[str, Any]:
    out = dict(row)
    qa_list = [dict(item) for item in row.get("qa_list") or [] if isinstance(item, dict)]
    for qa in qa_list:
        qa["last4_pass"] = 0
    out["qa_list"] = qa_list
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qa_total = 0
    pass_total = 0
    by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset = str(row.get("dataset") or row.get("source_dataset") or "unknown")
        bucket = by_dataset.setdefault(dataset, {"videos": 0, "qa": 0, "pass": 0})
        bucket["videos"] += 1
        for qa in row.get("qa_list") or []:
            if not isinstance(qa, dict):
                continue
            qa_total += 1
            bucket["qa"] += 1
            passed = int(qa.get("last4_pass") or 0)
            pass_total += passed
            bucket["pass"] += passed
    for bucket in by_dataset.values():
        bucket["last4_accuracy"] = bucket["pass"] / bucket["qa"] if bucket["qa"] else None
    return {
        "videos": len(rows),
        "qa": qa_total,
        "last4_accuracy": pass_total / qa_total if qa_total else None,
        "by_dataset": by_dataset,
    }


def row_has_last4_pass(row: dict[str, Any]) -> bool:
    qa_list = [item for item in row.get("qa_list") or [] if isinstance(item, dict)]
    return bool(qa_list) and all("last4_pass" in qa for qa in qa_list)


if __name__ == "__main__":
    main()
