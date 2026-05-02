"""Shared IO helpers for VideoXum synthesis."""

from __future__ import annotations

import json
from json import JSONDecoder
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[JsonDict]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[JsonDict] = []
    with target.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json_or_jsonl(path: str | Path) -> list[JsonDict]:
    target = Path(path)
    if target.suffix.lower() == ".jsonl":
        return read_jsonl(target)
    value = read_json(target)
    if not isinstance(value, list):
        raise ValueError("Input annotations must be a JSON list or JSONL file.")
    return [item for item in value if isinstance(item, dict)]


def write_jsonl(rows: Iterable[JsonDict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(row: JsonDict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def frame_path(record: JsonDict, frame_id: int, *, raw_data_root: str | Path = "raw_data") -> Path:
    pattern = str(record.get("frame_name_format") or "{frame_id:06d}.jpg")
    frames_dir = Path(str(record["frames_dir"]))
    if not frames_dir.is_absolute():
        frames_dir = Path(raw_data_root) / frames_dir
    return frames_dir / pattern.format(frame_id=int(frame_id))


def frame_paths(
    record: JsonDict,
    frame_ids: Iterable[int],
    *,
    raw_data_root: str | Path = "raw_data",
) -> list[Path]:
    return [frame_path(record, frame_id, raw_data_root=raw_data_root) for frame_id in frame_ids]


def require_frame_paths(
    record: JsonDict,
    frame_ids: Iterable[int],
    *,
    raw_data_root: str | Path = "raw_data",
    context: str = "",
) -> list[Path]:
    paths = frame_paths(record, frame_ids, raw_data_root=raw_data_root)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        label = f" for {context}" if context else ""
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
        raise FileNotFoundError(f"Missing {len(missing)} frame(s){label}: {preview}{suffix}")
    return paths


def existing_frame_paths(
    record: JsonDict,
    frame_ids: Iterable[int],
    *,
    raw_data_root: str | Path = "raw_data",
) -> list[Path]:
    return [path for path in frame_paths(record, frame_ids, raw_data_root=raw_data_root) if path.exists()]


def parse_json_from_text(text: str) -> Any:
    """Parse JSON even when the model wraps it in markdown or extra text."""
    cleaned = strip_code_fence(text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[idx:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object or array found in model output.")


def strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def as_json_list(value: Any, *, key: str | None = None) -> list[JsonDict]:
    if key is not None and isinstance(value, dict):
        value = value.get(key, [])
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list.")
    rows = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Expected a list of JSON objects.")
        rows.append(item)
    return rows


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))
