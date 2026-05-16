"""Load VideoHallucer entries as StreamWeave BenchmarkSample objects.

Each entry in videohallucer.json (one video, 1-2 yes/no probes) becomes one
BenchmarkSample whose metadata.qa_list drives RolloutRunner.run_multi_qa_sample
to branch the last step per probe.
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
        qa_id = probe["branch"]  # "basic" or "hallucination" — guaranteed unique per entry
        qa_list.append(
            {
                "qa_id": qa_id,
                "branch": probe["branch"],
                "question": probe["question"],
                "query_text": _format_query(probe["question"]),
                "gt": probe["gt"],
                "pair_id": entry["pair_id"],
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
        # The QueryEvent below is only a sentinel — run_multi_qa_sample ignores
        # sample.query_events and uses metadata.qa_list instead.
        sentinel_query = QueryEvent(text=qa_list[0]["query_text"], timestamp=0.0)

        sample = BenchmarkSample(
            sample_id=f"vh_{entry_id:0>6}_{entry['task']}",
            video_id=entry["video_id"],
            video_path=str(video_root / entry["video"]),
            query_events=[sentinel_query],
            metadata={
                "benchmark": "videohallucer",
                "task": entry["task"],
                "pair_id": entry["pair_id"],
                "subtype": entry.get("subtype", ""),
                "qa_list": qa_list,
                # Leave target_timestamp unset so all frames are used; questions
                # are always asked at the last step regardless.
            },
        )
        samples.append(sample)
        if limit and len(samples) >= limit:
            break
    return samples
