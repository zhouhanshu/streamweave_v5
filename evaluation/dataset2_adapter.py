"""Adapter for dataset2 annotation JSONL files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from streamweave.ovo import OPTION_LABELS, build_mcq_query, extract_mcq, option_label
from streamweave.schemas import BenchmarkSample, QueryEvent


def load_samples(args: dict[str, Any]) -> list[BenchmarkSample]:
    dataset_path = Path(str(args.get("dataset_path") or "")).expanduser()
    if not dataset_path:
        raise ValueError("dataset2 benchmark requires benchmark_args.dataset_path or --dataset-path")
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    annotation_path = dataset_path / "annotations.jsonl" if dataset_path.is_dir() else dataset_path
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing dataset2 annotations: {annotation_path}")

    require_options = bool(args.get("require_options", True))
    task_filter = str(args.get("task") or "").strip().lower()
    limit = int(args.get("limit") or 0)
    samples: list[BenchmarkSample] = []
    for line_index, row in _iter_jsonl(annotation_path):
        if task_filter and str(row.get("task", "")).strip().lower() != task_filter:
            if not _row_has_qa_list(row):
                continue
            row = _filter_qa_list_by_task(row, task_filter)
            if not row.get("qa_list"):
                continue
        if _row_has_qa_list(row):
            if require_options and not _all_qa_have_options(row):
                continue
            sample = multi_qa_sample_from_row(row, line_index=line_index, dataset_path=dataset_path)
            samples.append(sample)
            if limit and len(samples) >= limit:
                break
            continue
        sample = sample_from_row(row, line_index=line_index, dataset_path=dataset_path)
        options = row.get("options")
        if require_options and not isinstance(options, list):
            continue
        samples.append(sample)
        if limit and len(samples) >= limit:
            break
    return samples


def sample_from_row(row: dict[str, Any], *, line_index: int, dataset_path: Path) -> BenchmarkSample:
    video_id = str(row.get("video_id") or "").strip()
    if not video_id:
        raise ValueError(f"dataset2 row {line_index} missing video_id")
    sample_id = _sample_id(row, line_index)
    options = row.get("options")
    question = str(row.get("question") or "").strip()
    if isinstance(options, list) and options:
        query_text = build_mcq_query(question, options)
    else:
        query_text = question
    timestamp = _float_or(row.get("realtime", row.get("query_timestamp")), 0.0)
    raw_annotation = dict(row)
    metadata = dict(row)
    metadata.update(
        {
            "dataset_path": str(dataset_path),
            "dataset2_line_index": line_index,
            "raw_annotation": raw_annotation,
            "ground_truth": expected_option_letter(row),
            "frame_dataset_name": _frame_dataset_name(row, dataset_path),
            "query_timestamp": timestamp,
            "target_timestamp": timestamp,
        }
    )
    return BenchmarkSample(
        sample_id=sample_id,
        video_id=video_id,
        video_path="",
        query_events=[QueryEvent(text=query_text, timestamp=timestamp)],
        metadata=metadata,
    )


def multi_qa_sample_from_row(row: dict[str, Any], *, line_index: int, dataset_path: Path) -> BenchmarkSample:
    video_id = str(row.get("video_id") or "").strip()
    if not video_id:
        raise ValueError(f"dataset2 row {line_index} missing video_id")
    sample_id = str(row.get("sample_id") or video_id)
    raw_qa_list = [dict(item) for item in row.get("qa_list") or [] if isinstance(item, dict)]
    qa_list = [_normalize_qa_item(item, idx) for idx, item in enumerate(raw_qa_list)]
    timestamp = _float_or(row.get("realtime", row.get("query_timestamp")), 0.0)
    raw_annotation = dict(row)
    raw_annotation["qa_list"] = raw_qa_list
    frame_dataset_name = _frame_dataset_name(row, dataset_path)
    metadata = dict(row)
    metadata.update(
        {
            "dataset_path": str(dataset_path),
            "dataset2_line_index": line_index,
            "raw_annotation": raw_annotation,
            "qa_list": qa_list,
            "is_multi_qa": True,
            "frame_dataset_name": frame_dataset_name,
            "query_timestamp": timestamp,
            "target_timestamp": timestamp,
            "ground_truth": None,
        }
    )
    return BenchmarkSample(
        sample_id=sample_id,
        video_id=video_id,
        video_path="",
        query_events=[],
        metadata=metadata,
    )


def score_trace(trace: Any) -> dict[str, Any]:
    sample = trace.sample
    meta = dict(sample.metadata or {})
    options = meta.get("options")
    expected = expected_option_letter(meta)
    response = trace.final_answer()
    if not isinstance(options, list) or not options or expected is None:
        score = None
        prediction = None
    else:
        prediction = extract_mcq(response, max_options=len(options))
        score = int(prediction == expected)
    return {
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "response": response,
        "prediction": prediction,
        "ground_truth": expected,
        "score": score,
        "num_steps": len(trace.transitions),
        "task_failed": bool(trace.task_failed),
        "failure_reason": trace.failure_reason,
        "task": meta.get("task", ""),
        "category": meta.get("task", ""),
        "dataset": meta.get("dataset", ""),
    }


def error_result(sample: BenchmarkSample, error: str) -> dict[str, Any]:
    meta = dict(sample.metadata or {})
    return {
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "response": "",
        "prediction": None,
        "ground_truth": meta.get("ground_truth"),
        "score": 0,
        "num_steps": 0,
        "task_failed": True,
        "failure_reason": error,
        "error": error,
        "error_type": "sample_error",
        "task": meta.get("task", ""),
        "category": meta.get("task", ""),
        "dataset": meta.get("dataset", ""),
    }


def expected_option_letter(row: dict[str, Any]) -> str | None:
    options = row.get("options")
    if not isinstance(options, list) or not options:
        return None
    idx = expected_option_index(row.get("gt"), options, str(row.get("answer") or ""), row)
    return option_label(idx) if idx is not None else None


def expected_option_index(gt: Any, options: list[Any], answer: str, row: dict[str, Any]) -> int | None:
    option_texts = [clean_str(option) for option in options]
    answer_norm = normalize_text(answer)
    source_letter = clean_str(row.get("source_gt_letter")).upper()
    if source_letter and len(source_letter) == 1 and source_letter in OPTION_LABELS:
        idx = OPTION_LABELS.index(source_letter)
        if 0 <= idx < len(option_texts):
            return idx
    if isinstance(gt, str):
        text = gt.strip()
        if len(text) == 1 and text.upper() in OPTION_LABELS:
            idx = OPTION_LABELS.index(text.upper())
            return idx if 0 <= idx < len(option_texts) else None
        try:
            raw = int(text)
        except ValueError:
            gt_norm = normalize_text(text)
            for idx, option in enumerate(option_texts):
                if normalize_text(option) == gt_norm:
                    return idx
            return None
    else:
        try:
            raw = int(gt)
        except (TypeError, ValueError):
            return None

    candidates: list[int] = []
    if 0 <= raw < len(option_texts):
        candidates.append(raw)
    if 1 <= raw <= len(option_texts):
        candidates.append(raw - 1)
    if answer_norm:
        for idx in candidates:
            if normalize_text(option_texts[idx]) == answer_norm:
                return idx
    return candidates[0] if candidates else None


def annotation_with_pass_count(sample: BenchmarkSample, runs: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [run for run in runs if run.get("score") is not None]
    pass_count = sum(int(run.get("score") or 0) for run in scored)
    row = dict((sample.metadata or {}).get("raw_annotation") or {})
    row["pass_count"] = pass_count
    return row


def part_row(index: int, sample: BenchmarkSample, runs: list[dict[str, Any]]) -> dict[str, Any]:
    if (sample.metadata or {}).get("is_multi_qa"):
        return multi_qa_part_row(index, sample, runs)
    row = annotation_with_pass_count(sample, runs)
    scored = [run for run in runs if run.get("score") is not None]
    scored_runs = len(scored)
    pass_rate = (row["pass_count"] / scored_runs) if scored_runs else None
    return {
        "index": index,
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "pass_count": row["pass_count"],
        "pass_rate": pass_rate,
        "difficulty": _difficulty_from_pass_rate(pass_rate),
        "scored_runs": scored_runs,
        "parser_ok_rate": _mean(run.get("parser_ok_rate") for run in runs),
        "quality_valid_rate": _mean(run.get("quality_valid_rate") for run in runs),
        "annotation": row,
        "runs": runs,
    }


def multi_qa_part_row(index: int, sample: BenchmarkSample, runs: list[dict[str, Any]]) -> dict[str, Any]:
    row = multi_qa_annotation_with_scores(sample, runs)
    qa_items = [item for item in row.get("qa_list") or [] if isinstance(item, dict)]
    avg_pass_count = _mean(item.get("pass_count") for item in qa_items)
    return {
        "index": index,
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "qa_count": row.get("qa_count"),
        "pass_rate": row.get("video_pass_rate"),
        "difficulty": _difficulty_from_pass_rate(row.get("video_pass_rate")),
        "video_pass_rate": row.get("video_pass_rate"),
        "avg_pass_count": avg_pass_count,
        "scored_qa_count": len([item for item in qa_items if item.get("pass_count") is not None]),
        "parser_ok_rate": _mean(run.get("parser_ok_rate") for run in runs),
        "quality_valid_rate": _mean(run.get("quality_valid_rate") for run in runs),
        "annotation": row,
        "runs": runs,
    }


def multi_qa_annotation_with_scores(sample: BenchmarkSample, runs: list[dict[str, Any]]) -> dict[str, Any]:
    row = dict((sample.metadata or {}).get("raw_annotation") or {})
    qa_items = [dict(item) for item in row.get("qa_list") or [] if isinstance(item, dict)]
    qa_scores: dict[str, list[dict[str, Any]]] = {str(_qa_id(item, idx)): [] for idx, item in enumerate(qa_items)}
    for run in runs:
        for result in run.get("qa_results") or []:
            qa_id = str(result.get("qa_id") or "")
            if qa_id in qa_scores:
                qa_scores[qa_id].append(result)

    pass_rates: list[float] = []
    updated_qa = []
    for idx, qa in enumerate(qa_items):
        item = dict(qa)
        qa_id = str(_qa_id(item, idx))
        scored = [result for result in qa_scores.get(qa_id, []) if result.get("score") is not None]
        pass_count = sum(int(result.get("score") or 0) for result in scored)
        scored_runs = len(scored)
        pass_rate = (pass_count / scored_runs) if scored_runs else None
        item["pass_count"] = pass_count
        if pass_rate is not None:
            pass_rates.append(pass_rate)
        updated_qa.append(item)

    row["qa_list"] = updated_qa
    row["repeats"] = len(runs)
    row["qa_count"] = len(updated_qa)
    row["video_pass_rate"] = (sum(pass_rates) / len(pass_rates)) if pass_rates else None
    return row


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"dataset2 row {line_index} is not an object: {path}")
            yield line_index, row


def _sample_id(row: dict[str, Any], line_index: int) -> str:
    for key in ("sample_id", "source_annotation_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    video_id = str(row.get("video_id") or "").strip()
    source_row = row.get("source_row_index")
    source_turn = row.get("source_turn_index")
    if source_row is not None and source_turn is not None:
        return f"{video_id}_row{int(source_row):06d}_turn{int(source_turn):02d}"
    return f"{video_id}_line{line_index:06d}"


def _row_has_qa_list(row: dict[str, Any]) -> bool:
    return isinstance(row.get("qa_list"), list)


def _all_qa_have_options(row: dict[str, Any]) -> bool:
    qa_list = row.get("qa_list")
    return isinstance(qa_list, list) and all(isinstance(item, dict) and isinstance(item.get("options"), list) for item in qa_list)


def _filter_qa_list_by_task(row: dict[str, Any], task_filter: str) -> dict[str, Any]:
    out = dict(row)
    out["qa_list"] = [
        item for item in row.get("qa_list") or [] if isinstance(item, dict) and str(item.get("task", "")).strip().lower() == task_filter
    ]
    return out


def _normalize_qa_item(item: dict[str, Any], idx: int) -> dict[str, Any]:
    qa = dict(item)
    qa["qa_index"] = int(qa.get("qa_index", idx) or 0)
    qa["qa_id"] = _qa_id(qa, idx)
    question = str(qa.get("question") or "").strip()
    options = qa.get("options")
    qa["query_text"] = build_mcq_query(question, options) if isinstance(options, list) and options else question
    qa["ground_truth"] = expected_option_letter(qa)
    return qa


def _qa_id(qa: dict[str, Any], idx: int) -> str:
    for key in ("qa_id", "sample_id", "source_annotation_id"):
        value = str(qa.get(key) or "").strip()
        if value:
            return value
    return f"qa_{idx:04d}"


def _frame_dataset_name(row: dict[str, Any], dataset_path: Path) -> str:
    video = str(row.get("frames_dir") or row.get("video") or "").strip().strip("/")
    if "/" in video:
        prefix = video.split("/", 1)[0]
        if prefix and prefix != "video":
            return prefix
    if dataset_path.is_dir():
        return dataset_path.name
    dataset = str(row.get("dataset") or row.get("source_dataset") or "").strip()
    if dataset:
        return dataset
    return dataset_path.stem


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(value: Any) -> str:
    text = clean_str(value).lower()
    return re.sub(r"\s+", " ", text).strip(" .。")


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values) -> float | None:
    nums: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            continue
    return (sum(nums) / len(nums)) if nums else None


def _difficulty_from_pass_rate(pass_rate: Any) -> str:
    if pass_rate is None:
        return "unscored"
    try:
        value = float(pass_rate)
    except (TypeError, ValueError):
        return "unscored"
    if value >= 1.0:
        return "easy"
    if value >= 2.0 / 3.0:
        return "medium"
    if value >= 1.0 / 3.0:
        return "hard"
    return "unsolved"
