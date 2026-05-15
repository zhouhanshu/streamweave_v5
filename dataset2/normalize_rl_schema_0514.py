#!/usr/bin/env python3
"""Normalize the 0514 RL jsonl files to the agreed exp3 training schema."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

JOBS = (
    ("CogStream/rl_0514.jsonl", "CogStream/rl_0514_normalized.jsonl"),
    ("Streamo-Instruct-465K/rl_0514_unable.jsonl", "Streamo-Instruct-465K/rl_0514_unable_normalized.jsonl"),
    ("Streamo-Instruct-465K/rl_0514_one.jsonl", "Streamo-Instruct-465K/rl_0514_one_normalized.jsonl"),
    ("Streamo-Instruct-465K/rl_0514_multi.jsonl", "Streamo-Instruct-465K/rl_0514_multi_normalized.jsonl"),
)

TOP_KEYS = {
    "dataset",
    "video_id",
    "video",
    "frames_dir",
    "sample_fps",
    "frame_count",
    "frame_id_base",
    "query_events",
}
QUERY_KEYS = {"qid", "time", "content", "answer_type", "answer_policy", "options", "answer_events"}
ANSWER_KEYS = {"time", "gt", "answer", "content"}
VALID_POLICIES = {"answer_when_asked", "update_when_changed"}
PROACTIVE_PREFIX = "Please answer the following question based on the video content. You may update your answer multiple times."
MIN_SAME_EVENT_GAP_SECONDS = 7.0


def main() -> None:
    total = 0
    for src_rel, dst_rel in JOBS:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        rows = [normalize_row(row, src) for row in iter_jsonl(src)]
        validate_rows(rows)
        write_jsonl(dst, rows)
        total += len(rows)
        print(f"{dst_rel}: {len(rows)} rows")
    print(f"total: {total} rows")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def normalize_row(row: dict[str, Any], src: Path) -> dict[str, Any]:
    dataset = require_text(row.get("dataset"), src, "dataset")
    video_id = require_text(row.get("video_id") or row.get("sample_id"), src, "video_id")
    video = require_text(row.get("video") or row.get("frames_dir") or f"video/{video_id}", src, "video")
    frames_dir = require_text(row.get("frames_dir") or row.get("video") or video, src, "frames_dir")
    frame_count = int(round_time(row.get("frame_count") or row.get("realtime") or row.get("duration")))
    sample_fps = float(row.get("sample_fps", row.get("fps", 1.0)))
    frame_id_base = int(row.get("frame_id_base", 0) or 0)
    query_events = normalize_query_events(row.get("query_events"), src)
    return {
        "dataset": dataset,
        "video_id": video_id,
        "video": video,
        "frames_dir": frames_dir,
        "sample_fps": sample_fps,
        "frame_count": frame_count,
        "frame_id_base": frame_id_base,
        "query_events": query_events,
    }


def normalize_query_events(value: Any, src: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{src}: missing query_events")
    events: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{src}: query_events[{index}] is not an object")
        raw_answer_events = item.get("answer_events", [])
        policy = str(item.get("answer_policy") or infer_policy(raw_answer_events)).strip()
        if policy not in VALID_POLICIES:
            raise ValueError(f"{src}: invalid answer_policy={policy!r}")
        options = normalize_options(item.get("options"))
        answer_type = "mcq" if options else "text"
        content = require_text(
            item.get("content") or item.get("question") or item.get("query") or item.get("text"),
            src,
            f"query_events[{index}].content",
        )
        if policy == "update_when_changed":
            content = add_proactive_prefix(content)
        event: dict[str, Any] = {
            "qid": str(item.get("qid") or f"q{index}"),
            "time": round_time(item.get("time", item.get("timestamp", item.get("query_time", 0)))),
            "content": content,
            "answer_type": answer_type,
            "answer_policy": policy,
        }
        if answer_type == "mcq":
            event["options"] = options
        event["answer_events"] = normalize_answer_events(item.get("answer_events"), src, answer_type=answer_type)
        return_ordered = {key: event[key] for key in ("qid", "time", "content", "answer_type", "answer_policy")}
        if "options" in event:
            return_ordered["options"] = event["options"]
        return_ordered["answer_events"] = event["answer_events"]
        events.append(return_ordered)
    return events


def normalize_answer_events(value: Any, src: Path, *, answer_type: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{src}: missing answer_events")
    events: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{src}: answer_events[{index}] is not an object")
        event: dict[str, Any] = {
            "time": round_time(item.get("time", item.get("timestamp", item.get("evidence_time", 0)))),
        }
        answer = item.get("answer")
        content = item.get("content")
        if answer is None and content is None and item.get("gt") is None:
            raise ValueError(f"{src}: answer_events[{index}] has no answer/content/gt")
        if answer_type == "mcq":
            gt = str(item.get("gt") or "").strip().upper()
            if not gt or len(gt) != 1 or not gt.isalpha():
                raise ValueError(f"{src}: answer_events[{index}] has invalid mcq gt={item.get('gt')!r}")
            event["gt"] = gt
        if answer is not None and str(answer).strip():
            event["answer"] = str(answer).strip()
        if content is not None and str(content).strip():
            event["content"] = str(content).strip()
        elif answer is not None and str(answer).strip():
            event["content"] = str(answer).strip()
        events.append(event)
    return events


def add_proactive_prefix(content: str) -> str:
    if content.startswith(PROACTIVE_PREFIX):
        return content
    return f"{PROACTIVE_PREFIX}\n\n{content}"


def infer_policy(answer_events: Any) -> str:
    return "update_when_changed" if isinstance(answer_events, list) and len(answer_events) > 1 else "answer_when_asked"


def normalize_options(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def round_time(value: Any) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"negative time/frame_count: {value!r}")
    return float(int(math.floor(number + 0.5)))


def require_text(value: Any, src: Path, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{src}: missing {field}")
    return text


def validate_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        extra_top = set(row) - TOP_KEYS
        if extra_top:
            raise ValueError(f"{row.get('video_id')}: extra top-level keys {sorted(extra_top)}")
        video_id = str(row["video_id"])
        frame_count = row["frame_count"]
        frame_dir = ROOT / row["dataset"] / row["frames_dir"]
        if not frame_dir.is_dir():
            raise FileNotFoundError(f"{video_id}: missing frame dir {frame_dir}")
        query_times: list[float] = []
        answer_times: list[float] = []
        for query in row["query_events"]:
            extra_query = set(query) - QUERY_KEYS
            if extra_query:
                raise ValueError(f"{video_id}:{query.get('qid')}: extra query keys {sorted(extra_query)}")
            if not isinstance(query["time"], (int, float)):
                raise ValueError(f"{video_id}:{query.get('qid')}: query time is not numeric")
            if query["time"] > frame_count:
                raise ValueError(f"{video_id}:{query.get('qid')}: query time exceeds frame_count")
            query_times.append(float(query["time"]))
            answer_type = query.get("answer_type")
            if answer_type not in {"mcq", "text"}:
                raise ValueError(f"{video_id}:{query.get('qid')}: invalid answer_type={answer_type!r}")
            if answer_type == "mcq" and not query.get("options"):
                raise ValueError(f"{video_id}:{query.get('qid')}: mcq missing options")
            if answer_type == "text" and "options" in query:
                raise ValueError(f"{video_id}:{query.get('qid')}: text query should not have options")
            if not isinstance(query.get("answer_events"), list):
                raise ValueError(f"{video_id}:{query.get('qid')}: answer_events is not a list")
            for answer in query["answer_events"]:
                extra_answer = set(answer) - ANSWER_KEYS
                if extra_answer:
                    raise ValueError(f"{video_id}:{query.get('qid')}: extra answer keys {sorted(extra_answer)}")
                if not isinstance(answer["time"], (int, float)):
                    raise ValueError(f"{video_id}:{query.get('qid')}: answer time is not numeric")
                if answer["time"] > frame_count:
                    raise ValueError(f"{video_id}:{query.get('qid')}: answer time exceeds frame_count")
                answer_times.append(float(answer["time"]))
                if answer_type == "mcq" and "gt" not in answer:
                    raise ValueError(f"{video_id}:{query.get('qid')}: mcq answer missing gt")
                if answer_type == "text" and "gt" in answer:
                    raise ValueError(f"{video_id}:{query.get('qid')}: text answer should not have gt")
        validate_min_gap(query_times, f"{video_id}:query")
        validate_min_gap(answer_times, f"{video_id}:answer")


def validate_min_gap(times: list[float], label: str) -> None:
    for prev, cur in zip(sorted(times), sorted(times)[1:]):
        if cur - prev < MIN_SAME_EVENT_GAP_SECONDS:
            raise ValueError(f"{label} events are too close: {prev} -> {cur}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
