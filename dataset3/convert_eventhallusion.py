"""Convert raw EventHallusion question JSONs into a single OVO-style JSON.

Output: dataset3/eventhallusion/eventhallusion.json

Notes:
  - 3 splits: entire / mix / misleading. We use the JSON's split name as `task`.
  - mix JSON ids are `mix_xxx`, but the video files live under `interleave/`
    and are named `interleave_xxx.mp4`; we translate on the fly so video_id
    matches the actual file stem.
  - gt is normalized to lower-case "yes"/"no" (without the trailing period
    used in the raw answers); scorer applies startswith() matching.
  - desc task is intentionally skipped (would require GPT-4o judge).
"""

from __future__ import annotations

import json
from pathlib import Path

RAW_ROOT = Path("/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset3/raw/eventhallusion")
QUESTIONS_DIR = RAW_ROOT / "questions"
VIDEOS_ROOT = RAW_ROOT / "videos" / "videos"        # raw zip nested twice
OUT_DIR = Path("/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset3/eventhallusion")
OUT_JSON = OUT_DIR / "eventhallusion.json"

# split → (json filename, video subdirectory under VIDEOS_ROOT, id prefix in JSON)
SPLITS: list[tuple[str, str, str, str]] = [
    # (task name, json file, video subdir, json id prefix)
    ("entire", "entire_questions.json", "entire", "entire"),
    ("mix", "mix_questions.json", "interleave", "mix"),
    ("misleading", "misleading_questions.json", "misleading", "misleading"),
]


def _normalize_answer(ans: str) -> str:
    """'No.' / 'no.' / 'Yes' → 'no' / 'yes'."""
    return ans.strip().lower().rstrip(".")


def _video_id_for(task: str, source_id: str, video_subdir: str, id_prefix: str) -> str:
    """Map source_id ('mix_001') → actual video stem ('interleave_001')."""
    if id_prefix == video_subdir:
        return source_id
    # mix → interleave; suffix is the part after the underscore
    suffix = source_id.split("_", 1)[1]
    return f"{video_subdir}_{suffix}"


def _video_relpath(video_subdir: str, video_id: str) -> str:
    return f"{video_subdir}/{video_id}.mp4"


def _convert_split(task: str, json_name: str, video_subdir: str, id_prefix: str, next_id: int) -> tuple[list[dict], int]:
    raw = json.loads((QUESTIONS_DIR / json_name).read_text())
    entries: list[dict] = []
    for item in raw:
        source_id = item["id"]
        video_id = _video_id_for(task, source_id, video_subdir, id_prefix)

        test_info = []
        for qa_idx, qa in enumerate(item.get("questions") or []):
            test_info.append(
                {
                    "qa_id": f"q{qa_idx}",
                    "type": qa.get("type", "hallucination"),
                    "question": qa["question"],
                    "gt": _normalize_answer(qa["answer"]),
                }
            )
        if not test_info:
            continue

        entry = {
            "id": next_id,
            "task": task,
            "source_id": source_id,
            "video_id": video_id,
            "video": _video_relpath(video_subdir, video_id),
            "category": item.get("category", ""),
            "length": item.get("length", ""),
            "event_info": item.get("event_info", {}),
            "test_info": test_info,
        }
        entries.append(entry)
        next_id += 1
    return entries, next_id


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict] = []
    next_id = 0
    per_split_entries: dict[str, int] = {}
    per_split_questions: dict[str, int] = {}
    for task, json_name, video_subdir, id_prefix in SPLITS:
        entries, next_id = _convert_split(task, json_name, video_subdir, id_prefix, next_id)
        per_split_entries[task] = len(entries)
        per_split_questions[task] = sum(len(e["test_info"]) for e in entries)
        all_entries.extend(entries)

    # Sanity check video files exist on disk.
    missing: list[str] = []
    for entry in all_entries:
        path = VIDEOS_ROOT / entry["video"]
        if not path.exists():
            missing.append(str(path))

    OUT_JSON.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2))

    print(f"Wrote {OUT_JSON}")
    print(f"Entries: {len(all_entries)}")
    print(f"Total questions: {sum(per_split_questions.values())}")
    print("Per-split:")
    for task, _, _, _ in SPLITS:
        print(f"  {task}: entries={per_split_entries[task]}, questions={per_split_questions[task]}")
    print(f"Unique video_ids: {len({e['video_id'] for e in all_entries})}")
    if missing:
        print(f"\nWARNING: {len(missing)} video files missing on disk!")
        for p in missing[:5]:
            print(f"  {p}")


if __name__ == "__main__":
    main()
