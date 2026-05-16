"""Convert raw VideoHallucer subsets into a single OVO-style JSON.

Output: dataset3/videohallucer/videohallucer.json
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

RAW_ROOT = Path("/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset3/raw/videohallucer")
OUT_DIR = Path("/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset3/videohallucer")
OUT_JSON = OUT_DIR / "videohallucer.json"

SUBSETS = (
    "object_relation",
    "temporal",
    "semantic_detail",
    "interaction",
    "external_factual",
    "external_nonfactual",
    "fact_detect",
)


def _load_subset(subset: str) -> list[dict]:
    return json.loads((RAW_ROOT / subset / f"{subset}.json").read_text())


def _normalize_answer(ans: str) -> str:
    return ans.strip().lower().rstrip(".")


def _video_id(video_filename: str) -> str:
    return Path(video_filename).stem


def _video_relpath(subset: str, video_filename: str) -> str:
    return f"{subset}/videos/{video_filename}"


def _convert_subset(subset: str, next_id: int) -> tuple[list[dict], int]:
    raw = _load_subset(subset)
    entries: list[dict] = []
    if subset == "semantic_detail":
        # different video for basic vs hallucination -> 2 entries per pair
        for pair_idx, item in enumerate(raw):
            pair_id = f"{subset}_{pair_idx}"
            for branch in ("basic", "hallucination"):
                side = item[branch]
                entry = {
                    "id": next_id,
                    "task": subset,
                    "video_id": _video_id(side["video"]),
                    "video": _video_relpath(subset, side["video"]),
                    "pair_id": pair_id,
                    "test_info": [
                        {
                            "branch": branch,
                            "question": side["question"],
                            "gt": _normalize_answer(side["answer"]),
                        }
                    ],
                }
                if "type" in item:
                    entry["subtype"] = item["type"]
                entries.append(entry)
                next_id += 1
    else:
        # same video for basic + hallucination -> 1 entry per pair, test_info has both
        for pair_idx, item in enumerate(raw):
            basic = item["basic"]
            hall = item["hallucination"]
            assert basic["video"] == hall["video"], (
                f"{subset} pair {pair_idx}: video mismatch in same-video subset"
            )
            pair_id = f"{subset}_{pair_idx}"
            entry = {
                "id": next_id,
                "task": subset,
                "video_id": _video_id(basic["video"]),
                "video": _video_relpath(subset, basic["video"]),
                "pair_id": pair_id,
                "test_info": [
                    {
                        "branch": "basic",
                        "question": basic["question"],
                        "gt": _normalize_answer(basic["answer"]),
                    },
                    {
                        "branch": "hallucination",
                        "question": hall["question"],
                        "gt": _normalize_answer(hall["answer"]),
                    },
                ],
            }
            if "type" in item:
                entry["subtype"] = item["type"]
            entries.append(entry)
            next_id += 1
    return entries, next_id


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict] = []
    next_id = 0
    per_subset_counts: dict[str, int] = {}
    for subset in SUBSETS:
        entries, next_id = _convert_subset(subset, next_id)
        per_subset_counts[subset] = len(entries)
        all_entries.extend(entries)

    OUT_JSON.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2))

    unique_video_ids = OrderedDict()
    for e in all_entries:
        unique_video_ids[e["video_id"]] = None
    total_questions = sum(len(e["test_info"]) for e in all_entries)

    print(f"Wrote {OUT_JSON}")
    print(f"Entries: {len(all_entries)}")
    print(f"Unique video_ids (after dedup across subsets): {len(unique_video_ids)}")
    print(f"Total questions: {total_questions}")
    print("Per-subset entry counts:")
    for subset, count in per_subset_counts.items():
        print(f"  {subset}: {count}")


if __name__ == "__main__":
    main()
