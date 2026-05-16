#!/usr/bin/env python3
"""Build OVO-CRR-style yes/no evidence sufficiency data from unused Streamo one-QA rows."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RL_DIR = ROOT / "0516data" / "rl"
SOURCE_REL = "0516data/rl/rl_0514_one_normalized.jsonl"
DEFAULT_OUTPUT = RL_DIR / "rl_0515_yesno.jsonl"
DEFAULT_SUMMARY = RL_DIR / "rl_0515_yesno.summary.json"
RANDOM_SEED = 511
PAIR_COUNT = 200
MIN_GAP = 15.0
MAX_GAP = 30.0

SPLIT_SOURCES = (
    ("0516data/rl/rl_0514_pre.jsonl", 800, 20),
    ("0516data/rl/rl_0514_normalized.jsonl", 1000, 30),
    ("0516data/rl/rl_0514_unable_normalized.jsonl", 300, 15),
    ("0516data/rl/rl_0514_one_normalized.jsonl", 220, 15),
    ("0516data/rl/rl_0514_multi_normalized.jsonl", 280, 20),
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

PROACTIVE_PREFIX = "Please answer the following question based on the video content. You may update your answer multiple times."

CRR_PROMPT = """You're responsible for answering questions based on the video content.
The following question is relevant to the latest frames, i.e. the end of the video.
{question}
Decide whether the existing visual content, especially the latest frames near the end of the video, provides enough information for answering the question.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset2-root", type=Path, default=ROOT)
    parser.add_argument("--source", default=SOURCE_REL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    parser.add_argument("--min-gap", type=float, default=MIN_GAP)
    parser.add_argument("--max-gap", type=float, default=MAX_GAP)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def reconstruct_used_indices(dataset2_root: Path, *, seed: int, target_rel: str) -> set[int]:
    rng = random.Random(seed)
    used: set[int] = set()
    for rel_path, train_count, val_count in SPLIT_SOURCES:
        rows = read_jsonl(dataset2_root / rel_path)
        need = train_count + val_count
        if len(rows) < need:
            raise ValueError(f"{rel_path}: available={len(rows)}, requested={need}")
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        selected = set(indices[:need])
        if rel_path == target_rel:
            used = selected
    if not used:
        raise ValueError(f"target source {target_rel!r} is not present in SPLIT_SOURCES")
    return used


def clean_question(content: str) -> str:
    text = content.strip()
    if text.startswith(PROACTIVE_PREFIX):
        text = text[len(PROACTIVE_PREFIX) :].strip()
    if "\nOptions:" in text:
        text = text.split("\nOptions:", 1)[0].strip()
    if "\n\nUpdate your answer" in text:
        text = text.split("\n\nUpdate your answer", 1)[0].strip()
    if "\nUpdate your answer" in text:
        text = text.split("\nUpdate your answer", 1)[0].strip()
    return text


def round_second(value: Any) -> int:
    number = float(value)
    return int(number + 0.5)


def candidate_from_row(index: int, row: dict[str, Any], *, min_gap: float, max_gap: float) -> dict[str, Any] | None:
    events = row.get("query_events")
    if not isinstance(events, list) or len(events) != 1:
        return None
    query = events[0]
    if not isinstance(query, dict):
        return None
    answers = query.get("answer_events")
    if not isinstance(answers, list) or len(answers) != 1:
        return None
    answer = answers[0]
    if not isinstance(answer, dict):
        return None
    if str(query.get("answer_type") or "") != "mcq":
        return None
    if str(query.get("answer_policy") or "") != "update_when_changed":
        return None

    no_time = round_second(query.get("time"))
    yes_time = round_second(answer.get("time"))
    if no_time <= 0 or yes_time <= 0:
        return None
    gap = float(yes_time - no_time)
    if gap < min_gap or gap > max_gap:
        return None
    if yes_time > int(row.get("frame_count", 0)):
        return None

    question = clean_question(str(query.get("content") or ""))
    if not question:
        return None
    frames_dir = str(row.get("frames_dir") or row.get("video") or "")
    dataset = str(row.get("dataset") or "")
    if not dataset or not frames_dir:
        return None

    return {
        "source_index": index,
        "source_video_id": str(row.get("video_id") or ""),
        "question": question,
        "no_time": no_time,
        "yes_time": yes_time,
        "gap": gap,
        "row": row,
    }


def build_yesno_row(source: dict[str, Any], *, answer: str, frame_count: int) -> dict[str, Any]:
    row = source["row"]
    question = source["question"]
    time = float(frame_count)
    return {
        "dataset": row["dataset"],
        "video_id": row["video_id"],
        "video": row.get("video") or row.get("frames_dir"),
        "frames_dir": row.get("frames_dir") or row.get("video"),
        "sample_fps": float(row.get("sample_fps", 1.0)),
        "frame_count": frame_count,
        "frame_id_base": int(row.get("frame_id_base", 0) or 0),
        "query_events": [
            {
                "qid": "q0",
                "time": time,
                "content": CRR_PROMPT.format(question=question),
                "answer_type": "text",
                "answer_policy": "answer_when_asked",
                "answer_events": [
                    {
                        "time": time,
                        "answer": answer,
                        "content": answer,
                    }
                ],
            }
        ],
    }


def build_rows(candidates: list[dict[str, Any]], *, pairs: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(candidates) < pairs:
        raise ValueError(f"not enough candidates: available={len(candidates)}, requested_pairs={pairs}")
    selected = list(candidates)
    rng.shuffle(selected)
    selected = selected[:pairs]
    output: list[dict[str, Any]] = []
    for candidate in selected:
        output.append(build_yesno_row(candidate, answer="No", frame_count=int(candidate["no_time"])))
        output.append(build_yesno_row(candidate, answer="Yes", frame_count=int(candidate["yes_time"])))
    return output


def validate_rows(dataset2_root: Path, rows: list[dict[str, Any]]) -> None:
    for row_index, row in enumerate(rows):
        extra_top = set(row) - TOP_KEYS
        if extra_top:
            raise ValueError(f"row {row_index}: extra top-level keys {sorted(extra_top)}")
        frame_count = int(row["frame_count"])
        if frame_count <= 0:
            raise ValueError(f"row {row_index}: invalid frame_count={frame_count}")
        frames_dir = dataset2_root / str(row["dataset"]) / str(row["frames_dir"])
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"row {row_index}: missing frames dir {frames_dir}")
        events = row.get("query_events")
        if not isinstance(events, list) or len(events) != 1:
            raise ValueError(f"row {row_index}: expected exactly one query event")
        query = events[0]
        if query.get("answer_type") != "text":
            raise ValueError(f"row {row_index}: yes/no data must use answer_type=text")
        if "options" in query:
            raise ValueError(f"row {row_index}: text query must not contain options")
        if float(query.get("time")) != float(frame_count):
            raise ValueError(f"row {row_index}: query time must equal frame_count")
        answers = query.get("answer_events")
        if not isinstance(answers, list) or len(answers) != 1:
            raise ValueError(f"row {row_index}: expected exactly one answer event")
        answer = answers[0]
        if float(answer.get("time")) != float(frame_count):
            raise ValueError(f"row {row_index}: answer time must equal frame_count")
        if answer.get("answer") not in {"Yes", "No"} or answer.get("content") != answer.get("answer"):
            raise ValueError(f"row {row_index}: answer must be exactly Yes or No")


def summarize(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    used_indices: set[int],
    source_count: int,
    *,
    source_rel: str,
    seed: int,
    min_gap: float,
    max_gap: float,
) -> dict[str, Any]:
    answer_counts: Counter[str] = Counter()
    frame_counts: list[int] = []
    gaps: list[float] = []
    for row in rows:
        event = row["query_events"][0]
        answer_counts[str(event["answer_events"][0]["answer"])] += 1
        frame_counts.append(int(row["frame_count"]))
    for left, right in zip(rows[0::2], rows[1::2]):
        gaps.append(float(right["frame_count"] - left["frame_count"]))
    return {
        "source": source_rel,
        "random_seed": seed,
        "source_rows": source_count,
        "excluded_train_val_source_rows": len(used_indices),
        "candidate_policy": {
            "source_subset": "unused rows from 0516data/rl/rl_0514_one_normalized.jsonl",
            "gap_seconds": [min_gap, max_gap],
            "one_source_row_becomes": "two independent RL rows: No at query time, Yes at answer time",
        },
        "available_candidates": len(candidates),
        "pairs": len(rows) // 2,
        "rows": len(rows),
        "query_events": len(rows),
        "answer_events": len(rows),
        "answer_counts": dict(answer_counts),
        "frame_count_min": min(frame_counts) if frame_counts else None,
        "frame_count_max": max(frame_counts) if frame_counts else None,
        "gap_min": min(gaps) if gaps else None,
        "gap_max": max(gaps) if gaps else None,
    }


def main() -> None:
    args = parse_args()
    dataset2_root = args.dataset2_root.resolve()
    source_rel = str(args.source)
    source_rows = read_jsonl(dataset2_root / source_rel)
    used_indices = reconstruct_used_indices(dataset2_root, seed=args.seed, target_rel=source_rel)
    candidates = [
        candidate
        for index, row in enumerate(source_rows)
        if index not in used_indices
        for candidate in [candidate_from_row(index, row, min_gap=args.min_gap, max_gap=args.max_gap)]
        if candidate is not None
    ]
    rng = random.Random(args.seed)
    rows = build_rows(candidates, pairs=args.pairs, rng=rng)
    validate_rows(dataset2_root, rows)
    write_jsonl(args.output.resolve(), rows)

    summary = summarize(
        rows,
        candidates,
        used_indices,
        len(source_rows),
        source_rel=source_rel,
        seed=args.seed,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
    )
    summary["output"] = str(args.output.resolve())
    summary["summary_output"] = str(args.summary_output.resolve())
    write_json(args.summary_output.resolve(), summary)
    print(f"wrote {len(rows)} rows / {len(rows) // 2} pairs -> {args.output.resolve()}")
    print(f"available candidates: {len(candidates)}")
    print(f"answer counts: {dict(Counter(row['query_events'][0]['answer_events'][0]['answer'] for row in rows))}")


if __name__ == "__main__":
    main()
