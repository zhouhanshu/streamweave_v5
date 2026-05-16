"""Load EventHallusion entries as StreamWeave BenchmarkSample objects.

Same pattern as videohallucer_loader: each entry (one video, 1-N questions)
becomes one BenchmarkSample whose metadata.qa_list drives
RolloutRunner.run_multi_qa_sample to branch the last step per probe.

Differences from VideoHallucer:
- gt is "yes"/"no" but scorer uses startswith() (not regex word-boundary),
  matching official EventHallusion eval.py.
- Up to 7 questions per video (vs 2 max for VideoHallucer), but the framework
  loops over qa_list with no length cap.
- mix split's source_id ("mix_001") differs from video_id ("interleave_001"),
  already normalized at convert time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from streamweave.schemas import BenchmarkSample, QueryEvent


YESNO_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Answer with only 'Yes' or 'No'."
)


def _format_query(question: str) -> str:
    return YESNO_PROMPT_TEMPLATE.format(question=question.strip())


def _build_qa_list(entry: dict[str, Any]) -> list[dict[str, Any]]:
    qa_list: list[dict[str, Any]] = []
    for probe in entry["test_info"]:
        qa_list.append(
            {
                "qa_id": probe["qa_id"],            # "q0", "q1", ...
                "type": probe.get("type", "hallucination"),
                "question": probe["question"],
                "query_text": _format_query(probe["question"]),
                "gt": probe["gt"],
                "source_id": entry["source_id"],
                "task": entry["task"],
            }
        )
    return qa_list


def load_samples(
    json_path: str | Path,
    video_root: str | Path,
    *,
    sample_ids: list[str] | None = None,
    task: str = "",
    limit: int = 0,
) -> list[BenchmarkSample]:
    entries = json.loads(Path(json_path).read_text())
    sample_ids_set = {str(x) for x in (sample_ids or []) if str(x)}
    video_root = Path(video_root)

    samples: list[BenchmarkSample] = []
    for entry in entries:
        if task and entry["task"] != task:
            continue
        entry_id = str(entry["id"])
        if sample_ids_set and entry_id not in sample_ids_set:
            continue

        qa_list = _build_qa_list(entry)
        sentinel_query = QueryEvent(text=qa_list[0]["query_text"], timestamp=0.0)

        sample = BenchmarkSample(
            sample_id=f"eh_{entry_id:0>6}_{entry['task']}_{entry['source_id']}",
            video_id=entry["video_id"],
            video_path=str(video_root / entry["video"]),
            query_events=[sentinel_query],
            metadata={
                "benchmark": "eventhallusion",
                "task": entry["task"],
                "source_id": entry["source_id"],
                "category": entry.get("category", ""),
                "length": entry.get("length", ""),
                "qa_list": qa_list,
            },
        )
        samples.append(sample)
        if limit and len(samples) >= limit:
            break
    return samples
