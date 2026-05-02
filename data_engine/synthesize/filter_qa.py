#!/usr/bin/env python3
"""Step 4: verify and filter generated streaming QA candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.synthesize import filter_prompts
from data_engine.synthesize.gen_qa import build_timeline_text, selected_key_frame_ids
from data_engine.synthesize.io_utils import JsonDict, append_jsonl, parse_json_from_text, read_jsonl, require_frame_paths
from data_engine.synthesize.mcq_utils import SUPPORTED_LABELS, coerce_confidence, correct_choice, extract_choice
from data_engine.synthesize.vlm_client import VLMClient


LANGUAGE_LEAK_CONFIDENCE = 0.55
TEXT_LEAK_CONFIDENCE = 0.50
VISUAL_SUPPORT_CONFIDENCE = 0.55
HISTORY_GUESS_CONFIDENCE = 0.70


def filter_qa_record(
    caption_record: JsonDict,
    qa_record: JsonDict,
    filter_client: VLMClient,
    *,
    keep_per_video: int = 2,
    max_history_keyframes: int = 8,
    raw_data_root: Path = Path("raw_data"),
) -> JsonDict:
    verified: list[JsonDict] = []
    dropped_qa: list[JsonDict] = []
    for qa in qa_record.get("qa_candidates", []):
        try:
            verified.append(
                verify_one_qa(
                    caption_record,
                    qa,
                    filter_client,
                    max_history_keyframes=max_history_keyframes,
                    raw_data_root=raw_data_root,
                )
            )
        except Exception as exc:
            dropped_qa.append({"qa_id": qa.get("qa_id"), "error": repr(exc)})

    accepted = [qa for qa in verified if qa.get("verification", {}).get("passed")]
    return {
        "video_id": qa_record["video_id"],
        "source_annotation": caption_record.get("source_annotation", {}),
        "video": qa_record.get("video", caption_record.get("video", caption_record.get("frames_dir", ""))),
        "frames_dir": qa_record.get("frames_dir", caption_record.get("frames_dir", "")),
        "selected_key_frame_ids": qa_record.get("selected_key_frame_ids", selected_key_frame_ids(caption_record)),
        "accepted_qa": accepted[:keep_per_video],
        "accepted_count": min(len(accepted), keep_per_video),
        "verified_qa": verified,
        "verified_count": len(verified),
        "dropped_qa": dropped_qa,
        "dropped_count": len(dropped_qa),
        "errors": [],
    }


def verify_one_qa(
    caption_record: JsonDict,
    qa: JsonDict,
    client: VLMClient,
    *,
    max_history_keyframes: int,
    raw_data_root: Path,
) -> JsonDict:
    qa_type = str(qa.get("type", "")).strip()
    gt = correct_choice(qa)
    if gt not in {"A", "B", "C", "D"}:
        raise ValueError("QA is missing a valid multiple-choice ground truth.")

    verification = common_audits(qa, client)
    common_passed = verification["option_quality_passed"]

    if qa_type == "realtime":
        add_realtime_checks(verification, caption_record, qa, client, gt, raw_data_root=raw_data_root)
        verification["passed"] = (
            common_passed
            and not verification["text_timeline_leaked"]
            and verification["visual_supported"]
            and verification["temporal"]
        )
    elif qa_type == "forward":
        add_forward_checks(
            verification,
            caption_record,
            qa,
            client,
            gt,
            max_history_keyframes=max_history_keyframes,
            raw_data_root=raw_data_root,
        )
        verification["passed"] = (
            common_passed
            and not verification["history_leaked"]
            and verification["visual_supported"]
            and verification["temporal"]
        )
    elif qa_type == "backward":
        add_backward_checks(
            verification,
            caption_record,
            qa,
            client,
            gt,
            max_history_keyframes=max_history_keyframes,
            raw_data_root=raw_data_root,
        )
        verification["passed"] = common_passed and verification["history_supported"] and verification["temporal"]
    else:
        verification["passed"] = False
        verification["error"] = f"Unknown QA type: {qa_type}"

    qa_out = dict(qa)
    qa_out["verification"] = verification
    return qa_out


def common_audits(qa: JsonDict, client: VLMClient) -> JsonDict:
    option_quality = call_json(client, filter_prompts.option_quality_prompt(qa), max_tokens=768)
    return {
        "option_quality": option_quality,
        "option_quality_passed": option_quality_passed(option_quality),
    }


def add_realtime_checks(
    verification: JsonDict,
    caption_record: JsonDict,
    qa: JsonDict,
    client: VLMClient,
    gt: str,
    *,
    raw_data_root: Path,
) -> None:
    text_timeline = verify_text_timeline(caption_record, qa, client)
    visual = verify_visual_support(
        caption_record,
        qa,
        client,
        prompt_builder=filter_prompts.realtime_visual_prompt,
        raw_data_root=raw_data_root,
        context="realtime_visual",
    )
    verification.update(
        {
            "text_timeline": text_timeline,
            "visual": visual,
            "text_timeline_leaked": text_timeline_leaked(text_timeline, gt),
            "visual_supported": visual_supported(visual, gt),
            "temporal": abs(int(qa["query_time"]) - int(qa["answer_time"])) <= 2,
        }
    )


def add_forward_checks(
    verification: JsonDict,
    caption_record: JsonDict,
    qa: JsonDict,
    client: VLMClient,
    gt: str,
    *,
    max_history_keyframes: int,
    raw_data_root: Path,
) -> None:
    history = verify_forward_history(
        caption_record,
        qa,
        client,
        max_history_keyframes=max_history_keyframes,
        raw_data_root=raw_data_root,
    )
    visual = verify_visual_support(
        caption_record,
        qa,
        client,
        prompt_builder=filter_prompts.forward_visual_prompt,
        raw_data_root=raw_data_root,
        context="forward_visual",
    )
    verification.update(
        {
            "history_before_ask_time": history,
            "visual_at_answer_time": visual,
            "history_leaked": forward_history_leaked(history, gt),
            "visual_supported": visual_supported(visual, gt),
            "temporal": int(qa["query_time"]) < int(qa["answer_time"]),
        }
    )


def add_backward_checks(
    verification: JsonDict,
    caption_record: JsonDict,
    qa: JsonDict,
    client: VLMClient,
    gt: str,
    *,
    max_history_keyframes: int,
    raw_data_root: Path,
) -> None:
    history = verify_backward_history(
        caption_record,
        qa,
        client,
        max_history_keyframes=max_history_keyframes,
        raw_data_root=raw_data_root,
    )
    verification.update(
        {
            "history_to_ask_time": history,
            "history_supported": backward_history_supported(history, gt),
            "temporal": int(qa["answer_time"]) < int(qa["query_time"]),
        }
    )


def verify_text_timeline(caption_record: JsonDict, qa: JsonDict, client: VLMClient) -> JsonDict:
    prompt = filter_prompts.text_timeline_prompt(build_timeline_text(caption_record), qa)
    return call_json(client, prompt, max_tokens=768)


def verify_visual_support(
    caption_record: JsonDict,
    qa: JsonDict,
    client: VLMClient,
    *,
    prompt_builder,
    raw_data_root: Path,
    context: str,
) -> JsonDict:
    frame_ids = evidence_or_answer_frames(caption_record, qa)
    image_paths = require_frame_paths(
        caption_record,
        frame_ids,
        raw_data_root=raw_data_root,
        context=f"{caption_record['video_id']}:{qa.get('qa_id')}:{context}",
    )
    result = call_json(client, prompt_builder(qa, frame_ids), image_paths=image_paths, max_tokens=768)
    result.setdefault("frame_ids", frame_ids)
    return result


def verify_forward_history(
    caption_record: JsonDict,
    qa: JsonDict,
    client: VLMClient,
    *,
    max_history_keyframes: int,
    raw_data_root: Path,
) -> JsonDict:
    frame_ids = history_keyframe_frame_ids(caption_record, int(qa["query_time"]), qa, max_history_keyframes=max_history_keyframes)
    image_paths = require_frame_paths(
        caption_record,
        frame_ids,
        raw_data_root=raw_data_root,
        context=f"{caption_record['video_id']}:{qa.get('qa_id')}:forward_history",
    )
    prompt = filter_prompts.forward_history_prompt(build_prefix_timeline_text(caption_record, int(qa["query_time"])), qa)
    result = call_json(client, prompt, image_paths=image_paths, max_tokens=768)
    result.setdefault("frame_ids", frame_ids)
    return result


def verify_backward_history(
    caption_record: JsonDict,
    qa: JsonDict,
    client: VLMClient,
    *,
    max_history_keyframes: int,
    raw_data_root: Path,
) -> JsonDict:
    frame_ids = history_keyframe_frame_ids(caption_record, int(qa["query_time"]), qa, max_history_keyframes=max_history_keyframes)
    image_paths = require_frame_paths(
        caption_record,
        frame_ids,
        raw_data_root=raw_data_root,
        context=f"{caption_record['video_id']}:{qa.get('qa_id')}:backward_history",
    )
    prompt = filter_prompts.backward_history_prompt(build_prefix_timeline_text(caption_record, int(qa["query_time"])), qa)
    result = call_json(client, prompt, image_paths=image_paths, max_tokens=768)
    result.setdefault("frame_ids", frame_ids)
    return result


def call_json(
    client: VLMClient,
    prompt: str,
    *,
    image_paths: list[Path] | None = None,
    max_tokens: int = 768,
) -> JsonDict:
    raw = client.call([client.user_message(prompt, image_paths or [])], max_tokens=max_tokens)
    parsed = parse_json_from_text(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Verifier response must be a JSON object.")
    parsed["raw"] = raw
    return parsed


def option_quality_passed(result: JsonDict) -> bool:
    return (
        bool(result.get("is_valid_mcq"))
        and bool(result.get("has_single_correct_answer"))
        and not bool(result.get("has_option_bias"))
        and bool(result.get("is_visual_question"))
    )


def text_timeline_leaked(result: JsonDict, gt: str) -> bool:
    choice = extract_choice(str(result.get("choice", "")))
    confidence = coerce_confidence(result.get("confidence"))
    support = str(result.get("text_support", "")).strip()
    return (choice == gt and confidence >= TEXT_LEAK_CONFIDENCE) or (choice == gt and support in SUPPORTED_LABELS)


def visual_supported(result: JsonDict, gt: str) -> bool:
    choice = extract_choice(str(result.get("choice", "")))
    confidence = coerce_confidence(result.get("confidence"))
    support = str(result.get("visual_support", "")).strip()
    return choice == gt and confidence >= VISUAL_SUPPORT_CONFIDENCE and support in SUPPORTED_LABELS


def forward_history_leaked(result: JsonDict, gt: str) -> bool:
    choice = extract_choice(str(result.get("choice", "")))
    confidence = coerce_confidence(result.get("confidence"))
    support = str(result.get("history_support", "")).strip()
    if bool(result.get("is_answer_available_before_ask_time")):
        return True
    if choice == gt and confidence >= LANGUAGE_LEAK_CONFIDENCE and support != "not_supported":
        return True
    return choice == gt and confidence >= HISTORY_GUESS_CONFIDENCE


def backward_history_supported(result: JsonDict, gt: str) -> bool:
    choice = extract_choice(str(result.get("choice", "")))
    confidence = coerce_confidence(result.get("confidence"))
    support = str(result.get("history_support", "")).strip()
    return choice == gt and confidence >= VISUAL_SUPPORT_CONFIDENCE and support in SUPPORTED_LABELS


def build_prefix_timeline_text(caption_record: JsonDict, query_time: int) -> str:
    lines = []
    for window in caption_record.get("windows", []):
        start, end = [int(x) for x in window.get("time", [0, 0])]
        if end > query_time:
            continue
        caption = window_caption(window)
        lines.append(f"[{start}-{end}] {caption or ''}")
    return "\n".join(lines) or "<empty>"


def window_caption(window: JsonDict) -> str:
    if window.get("type") == "keyframe":
        return str(window.get("whole_window_caption") or window.get("window_caption") or window.get("caption") or "")
    return str(window.get("caption") or "")


def evidence_or_answer_frames(caption_record: JsonDict, qa: JsonDict) -> list[int]:
    sampled_frames = int(caption_record["sampled_frames"])
    evidence = [int(x) for x in qa.get("evidence_frame_ids", []) if 0 <= int(x) < sampled_frames]
    if evidence:
        return evidence
    answer_time = int(qa["answer_time"])
    start = max(0, answer_time - 2)
    end = min(sampled_frames - 1, answer_time + 2)
    return list(range(start, end + 1))


def history_keyframe_frame_ids(
    caption_record: JsonDict,
    query_time: int,
    qa: JsonDict,
    *,
    max_history_keyframes: int,
) -> list[int]:
    sampled_frames = int(caption_record["sampled_frames"])
    latest_visible_frame = max(0, min(query_time, sampled_frames - 1))
    selected_keyframes = sorted(
        frame_id for frame_id in selected_key_frame_ids(caption_record) if 0 <= frame_id <= latest_visible_frame
    )
    historical_evidence = sorted(
        int(frame_id)
        for frame_id in qa.get("evidence_frame_ids", [])
        if 0 <= int(frame_id) <= latest_visible_frame
    )
    if max_history_keyframes <= 0:
        return sorted(set(selected_keyframes) | set(historical_evidence))

    required = set(historical_evidence)
    slots = max(max_history_keyframes - len(required), 0)
    recent_keyframes = []
    if slots > 0:
        recent_keyframes = [frame_id for frame_id in selected_keyframes if frame_id not in required][-slots:]
    return sorted(required | set(recent_keyframes))


def records_by_video_id(rows: list[JsonDict]) -> dict[str, JsonDict]:
    return {str(row["video_id"]): row for row in rows}


def run_cli(args: argparse.Namespace) -> None:
    caption_by_id = records_by_video_id(read_jsonl(args.captions))
    qa_records = read_jsonl(args.qa)
    if args.limit:
        qa_records = qa_records[: args.limit]
    filter_client = VLMClient.from_backend(args.filter_backend, max_tokens=args.max_tokens)
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    for idx, qa_record in enumerate(qa_records, start=1):
        caption_record = caption_by_id[qa_record["video_id"]]
        result = filter_qa_record(
            caption_record,
            qa_record,
            filter_client,
            keep_per_video=args.keep_per_video,
            max_history_keyframes=args.max_history_keyframes,
            raw_data_root=args.raw_data_root,
        )
        append_jsonl(result, args.output)
        print(
            f"[{idx}/{len(qa_records)}] filter video={qa_record['video_id']} "
            f"accepted={result['accepted_count']} errors={len(result['errors'])}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, default=Path("data_engine/synthesize/outputs/captions.jsonl"))
    parser.add_argument("--qa", type=Path, default=Path("data_engine/synthesize/outputs/qa_candidates.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data_engine/synthesize/outputs/qa_filtered.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--raw-data-root", type=Path, default=Path("raw_data"))
    parser.add_argument("--filter-backend", default="qwen3vl")
    parser.add_argument("--keep-per-video", type=int, default=2)
    parser.add_argument(
        "--max-history-keyframes",
        "--max-history-images",
        dest="max_history_keyframes",
        type=int,
        default=8,
        help="Maximum historical keyframe images for prefix/history verification.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_cli(parse_args())
