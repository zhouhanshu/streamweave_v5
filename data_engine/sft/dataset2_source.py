"""dataset2 annotation adapter for StreamWeave SFT synthesis."""

from __future__ import annotations

import warnings
from pathlib import Path

from .io_utils import JsonDict, read_jsonl
from .schemas import FrameRef, QueryPlan, SamplePlan


def load_sample_plans(
    annotation_path: str | Path,
    *,
    raw_data_root: str | Path,
    sample_fps: float | None = None,
    offset: int = 0,
    limit: int = 0,
    sample_ids: set[str] | None = None,
    max_frames: int = 0,
) -> list[SamplePlan]:
    rows = _read_dataset2_rows(Path(annotation_path))
    if sample_ids:
        rows = [(idx, row, base) for idx, row, base in rows if _matches_sample_id(row, sample_ids, idx)]
    rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    return [
        annotation_to_sample_plan(
            row,
            line_index=line_index,
            input_base_dir=input_base_dir,
            raw_data_root=raw_data_root,
            sample_fps=sample_fps,
            max_frames=max_frames,
        )
        for line_index, row, input_base_dir in rows
    ]


def annotation_to_sample_plan(
    row: JsonDict,
    *,
    line_index: int,
    input_base_dir: Path,
    raw_data_root: str | Path,
    sample_fps: float | None = None,
    max_frames: int = 0,
) -> SamplePlan:
    video_id = _video_id(row, line_index)
    frames = _load_frame_refs(
        row,
        raw_data_root=Path(raw_data_root),
        input_base_dir=input_base_dir,
        sample_fps=sample_fps,
    )
    if max_frames and max_frames < len(frames):
        raise ValueError(
            f"dataset2 row {line_index} declares frame_count={len(frames)} but max_frames={max_frames} "
            "is tighter; data record is the source of truth — drop --max-frames or raise it."
        )
    if not frames:
        raise ValueError(f"dataset2 row {line_index} has no readable frames.")

    if _row_has_qa_list(row):
        metadata = _multi_qa_metadata(row, line_index=line_index, input_base_dir=input_base_dir)
        timestamp = _timestamp(row)
        return SamplePlan(
            sample_id=str(row.get("sample_id") or video_id),
            video_id=video_id,
            qa_id=str(row.get("qa_id") or video_id),
            task=str(row.get("task") or "multi_qa"),
            query_events=[],
            question_text="",
            query_time=timestamp,
            answer_time=timestamp,
            frames=frames,
            metadata=metadata,
        )

    question = format_question(row)
    timestamp = _timestamp(row)
    query_events = [QueryPlan(text=question, timestamp=timestamp)] if question else []
    qa_id = str(row.get("qa_id") or row.get("sample_id") or f"{video_id}_line{line_index:06d}")
    metadata = dict(row)
    metadata.update(
        {
            "dataset2_line_index": line_index,
            "input_base_dir": str(input_base_dir),
            "query_timestamp": timestamp,
            "target_timestamp": timestamp,
            "raw_annotation": dict(row),
        }
    )
    return SamplePlan(
        sample_id=str(row.get("sample_id") or f"{qa_id}_{_task(row)}"),
        video_id=video_id,
        qa_id=qa_id,
        task=_task(row),
        query_events=query_events,
        question_text=question,
        query_time=timestamp,
        answer_time=timestamp,
        frames=frames,
        metadata=metadata,
    )


def format_question(row: JsonDict) -> str:
    question = str(row.get("query_text") or row.get("question") or "").strip()
    if not question:
        return ""
    options = row.get("options")
    if isinstance(options, list) and options:
        lines = []
        for idx, option in enumerate(options):
            label = chr(ord("A") + idx)
            text = str(option).strip()
            if text[:2].upper() == f"{label}.":
                lines.append(text)
            else:
                lines.append(f"{label}. {text}")
        return (
            f"Question: {question}\n"
            "Options:\n"
            + "\n".join(lines)
            + "\n\nRespond only with the letter corresponding to your chosen option."
        )
    return question


def _read_dataset2_rows(path: Path) -> list[tuple[int, JsonDict, Path]]:
    if path.is_file():
        return [(idx, row, path.parent) for idx, row in enumerate(read_jsonl(path))]
    if not path.is_dir():
        raise FileNotFoundError(f"dataset2 SFT input does not exist: {path}")

    direct = path / "sft.jsonl"
    if not direct.exists():
        raise FileNotFoundError(
            f"No direct sft.jsonl found in {path}. Pass one dataset directory or one concrete sft.jsonl; "
            "merge multiple datasets explicitly after per-dataset export."
        )
    return [(idx, row, direct.parent) for idx, row in enumerate(read_jsonl(direct))]


def _multi_qa_metadata(row: JsonDict, *, line_index: int, input_base_dir: Path) -> JsonDict:
    timestamp = _timestamp(row)
    raw_qa_list = [dict(item) for item in row.get("qa_list") or [] if isinstance(item, dict)]
    qa_list = [_normalize_qa_item(item, idx) for idx, item in enumerate(raw_qa_list)]
    metadata = dict(row)
    metadata.update(
        {
            "dataset2_line_index": line_index,
            "input_base_dir": str(input_base_dir),
            "is_multi_qa": True,
            "qa_list": qa_list,
            "raw_annotation": {**dict(row), "qa_list": raw_qa_list},
            "query_timestamp": timestamp,
            "target_timestamp": timestamp,
            "ground_truth": None,
        }
    )
    return metadata


def _normalize_qa_item(item: JsonDict, idx: int) -> JsonDict:
    qa = dict(item)
    qa["qa_index"] = int(qa.get("qa_index", idx) or 0)
    qa["qa_id"] = _qa_id(qa, idx)
    qa["query_text"] = format_question(qa)
    if qa.get("ground_truth") is None:
        qa["ground_truth"] = _expected_option_letter(qa)
    return qa


def _load_frame_refs(
    row: JsonDict,
    *,
    raw_data_root: Path,
    input_base_dir: Path,
    sample_fps: float | None,
) -> list[FrameRef]:
    frames_dir = _frames_dir(row, raw_data_root=raw_data_root, input_base_dir=input_base_dir)
    pattern = str(row.get("frame_name_format") or "{frame_id:06d}.jpg")
    base = int(row.get("frame_id_base", 0) or 0)
    fps = _resolve_sample_fps(row, sample_fps)
    seconds_per_frame = 1.0 / fps
    frame_count = int(row.get("frame_count") or row.get("sampled_frames") or 0)
    if frame_count <= 0:
        raise ValueError(
            f"dataset2 row {_video_id(row, 0)} is missing frame_count/sampled_frames; "
            "step count must be derived from the data record, not the frame folder."
        )
    paths: list[tuple[int, Path]] = []
    missing_paths: list[Path] = []
    for offset in range(frame_count):
        frame_id = base + offset
        path = frames_dir / pattern.format(frame_id=frame_id)
        if not path.is_file():
            missing_paths.append(path)
            continue
        paths.append((frame_id, path))
    if missing_paths:
        warnings.warn(
            f"dataset2 row {_video_id(row, 0)} skipped {len(missing_paths)} missing frame file(s); "
            f"first missing: {missing_paths[0]}",
            RuntimeWarning,
            stacklevel=2,
        )

    refs: list[FrameRef] = []
    for frame_index, (frame_id, path) in enumerate(paths):
        start = (frame_id - base) * seconds_per_frame
        refs.append(
            FrameRef(
                global_frame_id=frame_id,
                start_time=start,
                end_time=start + seconds_per_frame,
                image_path=path,
                frame_index=frame_index,
            )
        )
    return refs


def _frames_dir(row: JsonDict, *, raw_data_root: Path, input_base_dir: Path) -> Path:
    raw = str(row.get("frames_dir") or row.get("video") or row.get("video_frame_dir") or "").strip()
    if not raw:
        raise ValueError(f"dataset2 row {_video_id(row, 0)} is missing frames_dir/video.")
    path = Path(raw)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"Frame directory does not exist: {path}")
        return path

    candidates = [
        raw_data_root / path,
        input_base_dir / path,
    ]
    dataset = str(row.get("dataset") or row.get("source_dataset") or "").strip()
    if dataset:
        candidates.append(raw_data_root / dataset / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Frame directory does not exist. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _resolve_sample_fps(row: JsonDict, sample_fps: float | None) -> float:
    value = sample_fps
    if value is None:
        value = row.get("sample_fps", row.get("fps", 1.0))
    fps = float(value)
    if fps <= 0:
        raise ValueError(f"sample_fps must be positive for dataset2 row {_video_id(row, 0)}: {fps}")
    return fps


def _timestamp(row: JsonDict) -> float:
    for key in ("target_timestamp", "realtime", "query_timestamp", "ask_time", "clue_time", "answer_time"):
        if key in row and row[key] is not None:
            return float(row[key])
    return 0.0


def _task(row: JsonDict) -> str:
    return str(row.get("task") or row.get("type") or row.get("question_type") or "backward").strip()


def _video_id(row: JsonDict, line_index: int) -> str:
    value = str(row.get("video_id") or "").strip()
    if value:
        return value
    raw = str(row.get("frames_dir") or row.get("video") or "").strip().strip("/")
    if raw:
        return Path(raw).name
    return f"dataset2_line{line_index:06d}"


def _qa_id(qa: JsonDict, idx: int) -> str:
    for key in ("qa_id", "sample_id", "source_annotation_id"):
        value = str(qa.get(key) or "").strip()
        if value:
            return value
    return f"qa_{idx:04d}"


def _row_has_qa_list(row: JsonDict) -> bool:
    return isinstance(row.get("qa_list"), list)


def _matches_sample_id(row: JsonDict, sample_ids: set[str], line_index: int) -> bool:
    video_id = _video_id(row, line_index)
    sample_id = str(row.get("sample_id") or video_id)
    if video_id in sample_ids or sample_id in sample_ids:
        return True
    return any(_qa_id(qa, idx) in sample_ids for idx, qa in enumerate(row.get("qa_list") or []) if isinstance(qa, dict))


def _expected_option_letter(row: JsonDict) -> str | None:
    options = row.get("options")
    if not isinstance(options, list) or not options:
        return None
    idx = _expected_option_index(row.get("gt"), options, str(row.get("answer") or ""))
    return chr(ord("A") + idx) if idx is not None else None


def _expected_option_index(gt: object, options: list[object], answer: str) -> int | None:
    if isinstance(gt, str):
        text = gt.strip()
        if len(text) == 1 and text.upper().isalpha():
            idx = ord(text.upper()) - ord("A")
            return idx if 0 <= idx < len(options) else None
        try:
            raw = int(text)
        except ValueError:
            return _option_index_by_text(text, options)
    else:
        try:
            raw = int(gt)
        except (TypeError, ValueError):
            return _option_index_by_text(answer, options)

    answer_norm = _norm(answer)
    candidates = []
    if 0 <= raw < len(options):
        candidates.append(raw)
    if 1 <= raw <= len(options):
        candidates.append(raw - 1)
    if answer_norm:
        for idx in candidates:
            if _norm(options[idx]) == answer_norm:
                return idx
    return candidates[0] if candidates else None


def _option_index_by_text(text: object, options: list[object]) -> int | None:
    target = _norm(text)
    if not target:
        return None
    for idx, option in enumerate(options):
        if _norm(option) == target:
            return idx
    return None


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split()).strip(" .。")


