"""Legacy extracted-frame annotation adapter for StreamWeave SFT."""

from __future__ import annotations

from pathlib import Path

from .io_utils import JsonDict, read_json_or_jsonl
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
    rows = read_json_or_jsonl(annotation_path)
    if sample_ids:
        rows = [row for row in rows if _matches_sample_id(row, sample_ids)]
    rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    return [
        annotation_to_sample_plan(row, raw_data_root=raw_data_root, sample_fps=sample_fps, max_frames=max_frames)
        for row in rows
    ]


def annotation_to_sample_plan(
    row: JsonDict,
    *,
    raw_data_root: str | Path,
    sample_fps: float | None = None,
    max_frames: int = 0,
) -> SamplePlan:
    video_id = _video_id(row)
    row_id = _row_id(row)
    task = _task(row)
    query_events = _query_events(row, task)
    question_text = query_events[0].text if query_events else ""
    query_time = query_events[0].timestamp if query_events else None
    answer_time = _answer_time(row)
    frames = _load_frame_refs(row, raw_data_root=raw_data_root, sample_fps=sample_fps)
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"Annotation {row_id} has no readable frames.")

    qa_id = str(row.get("qa_id") or f"{video_id}_{row_id}")
    sample_id = str(row.get("sample_id") or f"{qa_id}_{task}")
    return SamplePlan(
        sample_id=sample_id,
        video_id=video_id,
        qa_id=qa_id,
        task=task,
        query_events=query_events,
        question_text=question_text,
        query_time=query_time,
        answer_time=answer_time,
        frames=frames,
        metadata=dict(row),
    )


def format_question(row: JsonDict) -> str:
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError(f"Annotation {_row_id(row)} is missing question.")
    options = row.get("options")
    if isinstance(options, list) and options:
        option_lines = []
        for idx, option in enumerate(options):
            label = chr(ord("A") + idx)
            text = str(option).strip()
            if text[:2].upper() == f"{label}.":
                option_lines.append(text)
            else:
                option_lines.append(f"{label}. {text}")
        return (
            f"Question: {question}\n"
            "Options:\n"
            + "\n".join(option_lines)
            + "\n\nRespond only with the letter corresponding to your chosen option."
        )
    return question


def _load_frame_refs(row: JsonDict, *, raw_data_root: str | Path, sample_fps: float | None = None) -> list[FrameRef]:
    frames_dir = _frames_dir(row, raw_data_root=raw_data_root)
    pattern = str(row.get("frame_name_format") or "{frame_id:06d}.jpg")
    base = int(row.get("frame_id_base", 0) or 0)
    fps = _resolve_sample_fps(row, sample_fps)
    seconds_per_frame = 1.0 / fps
    frame_count = int(row.get("frame_count") or row.get("sampled_frames") or 0)
    paths: list[tuple[int, Path]] = []
    if frame_count > 0:
        for offset in range(frame_count):
            frame_id = base + offset
            path = frames_dir / pattern.format(frame_id=frame_id)
            paths.append((frame_id, path))
    else:
        for path in sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png")):
            frame_id = _parse_frame_id(path, fallback=len(paths) + base)
            paths.append((frame_id, path))

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


def _resolve_sample_fps(row: JsonDict, sample_fps: float | None) -> float:
    value = sample_fps
    if value is None:
        value = row.get("sample_fps", row.get("fps", 1.0))
    fps = float(value)
    if fps <= 0:
        raise ValueError(f"sample_fps must be positive for annotation {_row_id(row)}: {fps}")
    return fps


def _frames_dir(row: JsonDict, *, raw_data_root: str | Path) -> Path:
    raw = row.get("frames_dir") or row.get("video") or row.get("video_frame_dir")
    if not raw:
        raise ValueError(f"Annotation {_row_id(row)} is missing frames_dir/video.")
    path = Path(str(raw))
    if not path.is_absolute():
        path = Path(raw_data_root) / path
    if not path.exists():
        raise FileNotFoundError(f"Frame directory does not exist: {path}")
    return path


def _video_id(row: JsonDict) -> str:
    if row.get("video_id"):
        return str(row["video_id"])
    raw = row.get("frames_dir") or row.get("video") or ""
    if raw:
        return Path(str(raw)).name
    return str(row.get("id", "unknown_video"))


def _task(row: JsonDict) -> str:
    return str(row.get("task") or row.get("type") or row.get("question_type") or "synthesis").strip()


def _row_id(row: JsonDict) -> str:
    return str(row.get("id") or row.get("qa_id") or _video_id(row))


def _matches_sample_id(row: JsonDict, sample_ids: set[str]) -> bool:
    if _row_id(row) in sample_ids or _video_id(row) in sample_ids:
        return True
    video_id = _video_id(row)
    qa_id = str(row.get("qa_id") or f"{video_id}_{_row_id(row)}")
    task = _task(row)
    sample_id = str(row.get("sample_id") or f"{qa_id}_{task}")
    return sample_id in sample_ids or qa_id in sample_ids


def _query_time(row: JsonDict, task: str) -> float:
    if "ask_time" in row:
        return float(row["ask_time"])
    if "realtime" in row:
        return float(row["realtime"])
    if "query_time" in row:
        return float(row["query_time"])
    if task == "forward" and "clue_time" in row:
        return max(float(row["clue_time"]) - 1.0, 0.0)
    return 0.0


def _query_events(row: JsonDict, task: str) -> list[QueryPlan]:
    raw_events = row.get("query_events")
    if isinstance(raw_events, list) and raw_events:
        events: list[QueryPlan] = []
        fallback_text = _format_question_or_empty(row)
        for item in raw_events:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("question") or "").strip()
            if not text:
                text = fallback_text
            if not text:
                continue
            timestamp = item.get("timestamp", item.get("query_time", item.get("ask_time", item.get("realtime", 0.0))))
            events.append(QueryPlan(text=text, timestamp=float(timestamp)))
        if events:
            return sorted(events, key=lambda event: event.timestamp)
    question = _format_question_or_empty(row)
    if question:
        return [QueryPlan(text=question, timestamp=_query_time(row, task))]
    return []


def _format_question_or_empty(row: JsonDict) -> str:
    if not str(row.get("question") or "").strip():
        return ""
    return format_question(row)


def _answer_time(row: JsonDict) -> float | None:
    for key in ("clue_time", "answer_time", "realtime"):
        if key in row:
            return float(row[key])
    return None


def _parse_frame_id(path: Path, *, fallback: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return fallback
