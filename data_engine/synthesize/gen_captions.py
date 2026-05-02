#!/usr/bin/env python3
"""Step 2: generate window captions and image-derived fine visual facts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.synthesize.io_utils import (
    JsonDict,
    append_jsonl,
    parse_json_from_text,
    read_jsonl,
    require_frame_paths,
)
from data_engine.synthesize.vlm_client import VLMClient


def generate_caption_record(
    video_record: JsonDict,
    caption_client: VLMClient,
    *,
    facts_client: VLMClient | None = None,
    raw_data_root: Path = Path("raw_data"),
) -> JsonDict:
    facts_client = facts_client or caption_client
    windows: list[JsonDict] = []
    keyframe_facts: list[JsonDict] = []
    errors: list[JsonDict] = []
    for window in video_record.get("windows", []):
        window_out = dict(window)
        if window.get("type") == "keyframe":
            try:
                payload = generate_keyframe_window_payload(
                    video_record,
                    window,
                    facts_client,
                    raw_data_root=raw_data_root,
                )
                window_out.update(payload)
                keyframe_facts.extend(payload.get("fine_visual_facts", []))
            except Exception as exc:
                window_out["fine_visual_facts"] = []
                window_out["keyframe_payload_error"] = repr(exc)
                errors.append({"stage": "keyframe_payload", "window_id": window.get("window_id"), "error": repr(exc)})
        else:
            try:
                caption_payload = generate_window_caption(
                    video_record,
                    window,
                    caption_client,
                    raw_data_root=raw_data_root,
                )
                window_out.update(caption_payload)
            except Exception as exc:
                window_out["caption_error"] = repr(exc)
                errors.append({"stage": "caption", "window_id": window.get("window_id"), "error": repr(exc)})
        windows.append(window_out)

    return {
        "source_annotation": video_record.get("source_annotation", {}),
        "dataset": video_record.get("dataset", "streamweave_data"),
        "source_dataset": video_record.get("source_dataset", "VideoXum"),
        "split": video_record.get("split", ""),
        "video_id": video_record["video_id"],
        "video": video_record.get("video", video_record["frames_dir"]),
        "activitynet_video": video_record.get("activitynet_video", ""),
        "frames_dir": video_record["frames_dir"],
        "frame_name_format": video_record.get("frame_name_format", "{frame_id:06d}.jpg"),
        "sampled_frames": video_record["sampled_frames"],
        "selected_key_frame_ids": video_record.get("selected_key_frame_ids", []),
        "windows": windows,
        "keyframe_facts": keyframe_facts,
        "errors": errors,
    }


def generate_window_caption(
    video_record: JsonDict,
    window: JsonDict,
    client: VLMClient,
    *,
    raw_data_root: Path = Path("raw_data"),
) -> JsonDict:
    image_paths = require_frame_paths(
        video_record,
        window.get("frame_ids", []),
        raw_data_root=raw_data_root,
        context=f"{video_record['video_id']}:{window.get('window_id')}",
    )
    prompt = build_caption_prompt(video_record, window)
    raw = client.call([client.user_message(prompt, image_paths)], max_tokens=1200)
    parsed = parse_json_from_text(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Caption response must be a JSON object.")
    caption = str(parsed.get("caption", "")).strip()
    if not caption:
        raise ValueError("Normal-window caption response is missing 'caption'.")
    return {"caption": caption, "caption_raw": raw}


def generate_keyframe_window_payload(
    video_record: JsonDict,
    window: JsonDict,
    client: VLMClient,
    *,
    raw_data_root: Path = Path("raw_data"),
) -> JsonDict:
    image_paths = require_frame_paths(
        video_record,
        window.get("frame_ids", []),
        raw_data_root=raw_data_root,
        context=f"{video_record['video_id']}:{window.get('window_id')}:keyframe_payload",
    )
    prompt = build_keyframe_prompt(video_record, window)
    raw = client.call([client.user_message(prompt, image_paths)], max_tokens=2200)
    parsed = parse_json_from_text(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Keyframe response must be a JSON object.")
    required = ("before_keyframe_caption", "after_keyframe_caption", "whole_window_caption")
    missing = [key for key in required if not str(parsed.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Keyframe response is missing {missing}.")
    facts = parsed.get("fine_visual_facts", [])
    if not isinstance(facts, list):
        raise ValueError("'fine_visual_facts' must be a list.")
    cleaned = []
    allowed_keyframes = {int(k) for k in window.get("source_keyframe_ids", [])}
    allowed_frames = {int(frame_id) for frame_id in window.get("frame_ids", [])}
    for idx, fact in enumerate(facts):
        if not isinstance(fact, dict):
            continue
        evidence_frame_ids = coerce_evidence_frame_ids(fact.get("evidence_frame_ids", []), allowed_frames)
        if not evidence_frame_ids:
            continue
        source_keyframe_id = infer_source_keyframe_id(fact, evidence_frame_ids, allowed_keyframes)
        if source_keyframe_id not in allowed_keyframes:
            continue
        fact_text = str(fact.get("fact", "")).strip()
        category = str(fact.get("category", fact.get("fact_type", "other"))).strip() or "other"
        if not fact_text:
            continue
        cleaned.append(
            {
                "fact_id": f"{video_record['video_id']}_k{source_keyframe_id}_fact_{idx:03d}",
                "source_keyframe_id": source_keyframe_id,
                "fact": fact_text,
                "answer": str(fact.get("answer", "")).strip(),
                "category": category,
                "fact_type": category,
                "evidence_frame_ids": evidence_frame_ids,
                "evidence_interval": [
                    min(evidence_frame_ids),
                    max(evidence_frame_ids),
                ],
            }
        )
    return {
        "before_keyframe_caption": str(parsed["before_keyframe_caption"]).strip(),
        "after_keyframe_caption": str(parsed["after_keyframe_caption"]).strip(),
        "whole_window_caption": str(parsed["whole_window_caption"]).strip(),
        "pre_caption": str(parsed["before_keyframe_caption"]).strip(),
        "post_caption": str(parsed["after_keyframe_caption"]).strip(),
        "window_caption": str(parsed["whole_window_caption"]).strip(),
        "fine_visual_facts": cleaned,
        "caption_raw": raw,
    }


def coerce_evidence_frame_ids(value: object, allowed_frames: set[int]) -> list[int]:
    if not isinstance(value, list):
        return []
    frame_ids: list[int] = []
    for item in value:
        try:
            frame_id = int(item)
        except (TypeError, ValueError):
            continue
        if frame_id in allowed_frames:
            frame_ids.append(frame_id)
    return sorted(set(frame_ids))


def infer_source_keyframe_id(fact: JsonDict, evidence_frame_ids: list[int], allowed_keyframes: set[int]) -> int:
    try:
        source_keyframe_id = int(fact.get("source_keyframe_id"))
        if source_keyframe_id in allowed_keyframes:
            return source_keyframe_id
    except (TypeError, ValueError):
        pass
    if not allowed_keyframes:
        return -1
    if len(allowed_keyframes) == 1:
        return next(iter(allowed_keyframes))
    center = sum(evidence_frame_ids) / max(len(evidence_frame_ids), 1)
    return min(allowed_keyframes, key=lambda keyframe_id: abs(keyframe_id - center))


def build_caption_prompt(video_record: JsonDict, window: JsonDict) -> str:
    frame_ids = ", ".join(str(x) for x in window.get("frame_ids", []))
    if window.get("type") == "normal":
        return f"""\
You are a visual captioning assistant for short video clips.

Video ID: {video_record['video_id']}
Window type: normal
Window frame ids: [{frame_ids}]
Window time: {window.get('time')}

You will be given a short sequence of frames from one video window.
Describe only what is visually observable in these frames.

Rules:
1. Do not infer events outside the given frames.
2. Do not mention uncertainty unless the visual evidence is genuinely unclear.
3. Do not invent identities, intentions, relationships, or causal explanations.
4. Focus on actions, objects, people, scene layout, text visible in the image, and changes across frames.
5. If nothing important changes, say that the scene remains mostly stable.

Return JSON only:
{{
  "caption": "..."
}}
"""


def build_keyframe_prompt(video_record: JsonDict, window: JsonDict) -> str:
    frame_ids = ", ".join(str(x) for x in window.get("frame_ids", []))
    source_keyframes = ", ".join(str(x) for x in window.get("source_keyframe_ids", []))
    return f"""\
You are a fine-grained visual fact extractor for video QA generation.

Video ID: {video_record['video_id']}
Window type: keyframe
Key frame ids: [{source_keyframes}]
Window frame ids: [{frame_ids}]
Window time: {window.get('time')}

You will be given a short window of frames centered around one key frame.
The key frame is marked by the key frame ids above. Your job is to describe the local temporal context and extract fine-grained visual facts from the key frame and nearby frames.

Definitions:
- before_keyframe_caption: what is visually happening before the key frame.
- after_keyframe_caption: what is visually happening after the key frame.
- whole_window_caption: what happens across the full window.
- fine_visual_facts: atomic visual facts that can be verified from the image frames.

Fine visual facts should include, when visible:
1. objects held, touched, worn, opened, closed, moved, or pointed at;
2. colors, numbers, logos, signs, screen text, labels, or written words;
3. spatial relations between people and objects;
4. distinctive object attributes;
5. body pose, gaze direction, hand position, or interaction details;
6. changes that happen near the key frame.

Rules for fine_visual_facts:
1. Each fact must be directly visible.
2. Each fact must be atomic: one fact per sentence.
3. Prefer specific facts over generic facts.
4. Avoid facts that are obvious from language priors alone.
5. Do not infer intentions or unseen causes.
6. Do not use vague phrases such as "something", "maybe", "probably".
7. Do not include facts that require knowing the full video outside this window.
8. Facts should be useful for making multiple-choice visual questions.
9. If a visible text is present, quote only the visible text.
10. If no reliable fine visual fact exists, return an empty list.

Output JSON only:
{{
  "before_keyframe_caption": "...",
  "after_keyframe_caption": "...",
  "whole_window_caption": "...",
  "fine_visual_facts": [
    {{
      "fact": "...",
      "evidence_frame_ids": [45],
      "category": "object|attribute|text|spatial_relation|action|pose|count|other"
    }}
  ]
}}
"""

def run_cli(args: argparse.Namespace) -> None:
    records = read_jsonl(args.windows)
    if args.limit:
        records = records[: args.limit]
    caption_client = VLMClient.from_backend(args.caption_backend, max_tokens=args.max_tokens)
    facts_client = VLMClient.from_backend(args.facts_backend, max_tokens=args.max_tokens)
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    for idx, record in enumerate(records, start=1):
        result = generate_caption_record(
            record,
            caption_client,
            facts_client=facts_client,
            raw_data_root=args.raw_data_root,
        )
        append_jsonl(result, args.output)
        print(f"[{idx}/{len(records)}] captions video={record['video_id']} errors={len(result['errors'])}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("data_engine/synthesize/outputs/windows.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data_engine/synthesize/outputs/captions.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--raw-data-root", type=Path, default=Path("raw_data"))
    parser.add_argument("--caption-backend", default="qwen3vl")
    parser.add_argument("--facts-backend", default="qwen3vl")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_cli(parse_args())
