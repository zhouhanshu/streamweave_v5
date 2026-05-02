"""Local IO helpers for StreamWeave SFT synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

JsonDict = dict[str, Any]


def read_json_or_jsonl(path: str | Path) -> list[JsonDict]:
    target = Path(path)
    if target.suffix.lower() == ".jsonl":
        return read_jsonl(target)
    with target.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list in {target}.")
    return [item for item in value if isinstance(item, dict)]


def read_jsonl(path: str | Path) -> list[JsonDict]:
    target = Path(path)
    rows: list[JsonDict] = []
    if not target.exists():
        return rows
    with target.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def write_jsonl(rows: Iterable[JsonDict], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(row: JsonDict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def reset_file(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)


def media_path(path: str | Path, media_dir: str | Path) -> str:
    target = Path(path)
    root = Path(media_dir)
    try:
        return target.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return target.as_posix()

