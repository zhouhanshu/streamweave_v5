#!/usr/bin/env python3
"""Step 3: generate streaming QA candidates from captions and fine facts."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
from data_engine.synthesize.mcq_utils import (
    choice_for_answer_text,
    extract_choice,
    normalize_options,
    option_text_for_choice,
    strip_option_prefix,
)
from data_engine.synthesize.vlm_client import VLMClient


DEFAULT_QA_MAX_ATTEMPTS = 3
DEFAULT_RAW_RESPONSE_CHARS = 12000


@dataclass(slots=True)
class QAContext:
    eligible_keyframes: list[int]
    image_paths: list[Path]
    prompt: str


def generate_qa_record(
    caption_record: JsonDict,
    client: VLMClient,
    *,
    num_questions: int = 6,
    max_attempts: int = DEFAULT_QA_MAX_ATTEMPTS,
    raw_response_chars: int = DEFAULT_RAW_RESPONSE_CHARS,
    raw_data_root: Path = Path("raw_data"),
) -> JsonDict:
    max_attempts = max(1, int(max_attempts))
    candidates: list[JsonDict] = []
    errors: list[JsonDict] = []
    attempts: list[JsonDict] = []
    eligible_keyframes: list[int] = []
    try:
        context = build_qa_context(
            caption_record,
            num_questions=num_questions,
            raw_data_root=raw_data_root,
        )
        eligible_keyframes = context.eligible_keyframes
        candidates, attempts = generate_qa_with_retries(
            caption_record,
            client,
            context,
            num_questions=num_questions,
            max_attempts=max_attempts,
            raw_response_chars=raw_response_chars,
        )
        if not candidates:
            errors.append(
                {
                    "stage": "gen_qa",
                    "error": f"No valid QA candidates after {max_attempts} attempt(s).",
                }
            )
    except Exception as exc:
        errors.append({"stage": "gen_qa", "error": repr(exc)})
    return {
        "video_id": caption_record["video_id"],
        "video": caption_record.get("video", caption_record["frames_dir"]),
        "frames_dir": caption_record["frames_dir"],
        "frame_name_format": caption_record.get("frame_name_format", "{frame_id:06d}.jpg"),
        "sampled_frames": caption_record["sampled_frames"],
        "selected_key_frame_ids": selected_key_frame_ids(caption_record),
        "eligible_key_frame_ids": eligible_keyframes,
        "requested_questions": num_questions,
        "max_attempts": max_attempts,
        "qa_candidates": candidates,
        "qa_generation_attempts": attempts,
        "errors": errors,
    }


def generate_qa_for_video(
    caption_record: JsonDict,
    client: VLMClient,
    *,
    num_questions: int = 6,
    max_attempts: int = DEFAULT_QA_MAX_ATTEMPTS,
    raw_response_chars: int = DEFAULT_RAW_RESPONSE_CHARS,
    raw_data_root: Path = Path("raw_data"),
) -> list[JsonDict]:
    context = build_qa_context(caption_record, num_questions=num_questions, raw_data_root=raw_data_root)
    candidates, _ = generate_qa_with_retries(
        caption_record,
        client,
        context,
        num_questions=num_questions,
        max_attempts=max_attempts,
        raw_response_chars=raw_response_chars,
    )
    if not candidates:
        raise ValueError(f"QA response did not contain any valid QA object after {max_attempts} attempt(s).")
    return candidates


def build_qa_context(
    caption_record: JsonDict,
    *,
    num_questions: int,
    raw_data_root: Path,
) -> QAContext:
    eligible_keyframes = [
        int(keyframe_id)
        for keyframe_id in selected_key_frame_ids(caption_record)
        if facts_for_keyframe(caption_record, int(keyframe_id))
    ]
    if not eligible_keyframes:
        raise ValueError("No selected keyframe has fine_visual_facts.")
    image_ids = qa_context_frame_ids(caption_record, eligible_keyframes)
    image_paths = require_frame_paths(
        caption_record,
        image_ids,
        raw_data_root=raw_data_root,
        context=f"{caption_record['video_id']}:qa_context",
    )
    prompt = build_qa_prompt(caption_record, eligible_keyframes, num_questions=num_questions)
    return QAContext(eligible_keyframes=eligible_keyframes, image_paths=image_paths, prompt=prompt)


def generate_qa_with_retries(
    caption_record: JsonDict,
    client: VLMClient,
    context: QAContext,
    *,
    num_questions: int,
    max_attempts: int,
    raw_response_chars: int,
) -> tuple[list[JsonDict], list[JsonDict]]:
    attempts: list[JsonDict] = []
    max_attempts = max(1, int(max_attempts))
    for attempt_index in range(1, max_attempts + 1):
        prompt = context.prompt
        if attempt_index > 1:
            prompt = f"{context.prompt}\n\n{retry_instruction(num_questions)}"
        try:
            raw = client.call([client.user_message(prompt, context.image_paths)], max_tokens=client.max_tokens)
            candidates, attempt_log = parse_and_normalize_qa_response(
                caption_record,
                raw,
                raw_response_chars=raw_response_chars,
            )
        except Exception as exc:
            attempt_log = {
                "attempt": attempt_index,
                "status": "call_or_parse_error",
                "error": repr(exc),
            }
            candidates = []

        attempt_log["attempt"] = attempt_index
        attempts.append(attempt_log)
        if candidates:
            attempt_log["status"] = "accepted"
            return candidates[:num_questions], attempts

    return [], attempts


def parse_and_normalize_qa_response(
    caption_record: JsonDict,
    raw: str,
    *,
    raw_response_chars: int,
) -> tuple[list[JsonDict], JsonDict]:
    attempt_log: JsonDict = {
        "status": "started",
        "raw_response_chars": len(raw),
        "raw_response": compact_text(raw, raw_response_chars),
    }
    try:
        parsed = parse_json_from_text(raw)
    except Exception as exc:
        attempt_log["status"] = "parse_error"
        attempt_log["error"] = repr(exc)
        return [], attempt_log

    qas, container = extract_qa_candidates(parsed)
    attempt_log["parsed_container"] = container
    if not isinstance(qas, list):
        attempt_log["status"] = "invalid_schema"
        attempt_log["error"] = "QA response must be a JSON object with 'qa_candidates' or a JSON list."
        return [], attempt_log

    normalized: list[JsonDict] = []
    rejections: list[JsonDict] = []
    for idx, qa in enumerate(qas):
        if not isinstance(qa, dict):
            rejections.append({"index": idx, "reason": "candidate_is_not_object"})
            continue
        item, reason = normalize_qa_with_reason(caption_record, qa, idx)
        if item is not None:
            normalized.append(item)
        else:
            rejections.append({"index": idx, "reason": reason, "raw_candidate": qa})

    attempt_log.update(
        {
            "parsed_candidate_count": len(qas),
            "valid_candidate_count": len(normalized),
            "rejected_candidate_count": len(rejections),
            "rejected_candidates": rejections[:20],
        }
    )
    if not normalized:
        attempt_log["status"] = "no_valid_candidates"
    return normalized, attempt_log


def extract_qa_candidates(parsed: object) -> tuple[object, str]:
    if isinstance(parsed, list):
        return parsed, "json_list"
    if isinstance(parsed, dict):
        if "qa_candidates" in parsed:
            return parsed.get("qa_candidates"), "json_object.qa_candidates"
        if "qas" in parsed:
            return parsed.get("qas"), "json_object.qas"
    return None, type(parsed).__name__


def compact_text(text: str, max_chars: int) -> str:
    if max_chars < 0:
        return text
    if max_chars == 0:
        return ""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {omitted} chars]"


def retry_instruction(num_questions: int) -> str:
    return f"""\
Previous response did not produce valid QA candidates for the required schema.
Try again and return JSON only.
The top-level object must contain "qa_candidates".
Generate up to {num_questions} candidates. If a candidate is weak, omit that candidate, but do not return prose.
Every candidate must include: type, question, options, gt, answer, ask_time, answer_time, evidence_frame_ids, source_facts, rationale.
"""


def normalize_qa(caption_record: JsonDict, qa: JsonDict, idx: int) -> JsonDict | None:
    item, _ = normalize_qa_with_reason(caption_record, qa, idx)
    return item


def normalize_qa_with_reason(caption_record: JsonDict, qa: JsonDict, idx: int) -> tuple[JsonDict | None, str]:
    target_keyframe_id = coerce_int(qa.get("source_keyframe_id"))
    if target_keyframe_id is None:
        target_keyframe_id = coerce_int(qa.get("answer_time"))
    if target_keyframe_id is None:
        target_keyframe_id = coerce_int(qa.get("clue_time"))
    if target_keyframe_id is None:
        return None, "missing_source_keyframe_id_answer_time_or_clue_time"

    qa_type = str(qa.get("type", "")).strip()
    answer_format = str(qa.get("answer_format", "multiple-choice")).strip()
    question = str(qa.get("question", "")).strip()
    answer = str(qa.get("answer", "")).strip()
    if qa_type not in {"realtime", "forward", "backward"}:
        return None, "invalid_type"
    if answer_format != "multiple-choice":
        return None, "invalid_answer_format"
    if not question or not answer:
        return None, "missing_question_or_answer"

    options = normalize_options(qa.get("options", []))
    answer_choice = extract_choice(str(qa.get("gt", ""))) or extract_choice(answer)
    if not answer_choice and answer:
        answer_choice = choice_for_answer_text(options, answer)
    if len(options) != 4 or answer_choice not in {"A", "B", "C", "D"}:
        return None, "invalid_options_or_gt"
    answer_is_only_choice = extract_choice(answer) == answer.strip().upper()
    answer_text = str(qa.get("answer_text", "")).strip()
    if not answer_text and answer and not answer_is_only_choice:
        answer_text = strip_option_prefix(answer)
    if not answer_text:
        answer_text = option_text_for_choice(options, answer_choice)
    if not answer_text:
        return None, "missing_answer_text"

    normalization_warnings: list[str] = []
    query_time = coerce_int(qa.get("ask_time", qa.get("query_time")))
    if query_time is None:
        query_time = target_keyframe_id
        normalization_warnings.append("missing_query_time_defaulted_to_source_keyframe_id")

    answer_time = coerce_int(qa.get("answer_time"))
    if answer_time is None:
        answer_time = coerce_int(qa.get("clue_time"))
    if answer_time is None:
        answer_time = target_keyframe_id
        normalization_warnings.append("missing_answer_time_defaulted_to_source_keyframe_id")

    evidence_frame_ids = coerce_int_list(qa.get("evidence_frame_ids", []))
    if not evidence_frame_ids:
        evidence_frame_ids = [target_keyframe_id]
        normalization_warnings.append("missing_evidence_frame_ids_defaulted_to_source_keyframe_id")

    source_facts = normalize_source_facts(qa.get("source_facts", []))
    visual_fact = str(qa.get("visual_fact", "")).strip() or "; ".join(source_facts)
    qa_id = f"{caption_record['video_id']}_k{target_keyframe_id}_{qa_type}_{idx:03d}"
    normalized = {
        "video_id": caption_record["video_id"],
        "video": caption_record.get("video", caption_record["frames_dir"]),
        "frames_dir": caption_record["frames_dir"],
        "qa_id": qa_id,
        "source_keyframe_id": target_keyframe_id,
        "type": qa_type,
        "question_subtype": str(qa.get("question_subtype", "")).strip(),
        "answer_format": answer_format,
        "question": question,
        "answer": answer_choice,
        "answer_text": answer_text,
        "options": options,
        "query_time": query_time,
        "answer_time": answer_time,
        "evidence_frame_ids": evidence_frame_ids,
        "evidence_interval": qa.get("evidence_interval", [min(evidence_frame_ids), max(evidence_frame_ids)]),
        "visual_fact": visual_fact,
        "source_facts": source_facts,
        "rationale": str(qa.get("rationale", "")).strip(),
        "raw_qa": qa,
    }
    if normalization_warnings:
        normalized["normalization_warnings"] = normalization_warnings
    return normalized, ""


def normalize_source_facts(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    facts = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("fact", "")).strip()
        else:
            text = str(item).strip()
        if text:
            facts.append(text)
    return facts


def coerce_int(value: object) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def coerce_int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    values: list[int] = []
    for item in value:
        coerced = coerce_int(item)
        if coerced is not None:
            values.append(coerced)
    return values


def build_qa_prompt(caption_record: JsonDict, eligible_keyframes: list[int], *, num_questions: int = 6) -> str:
    selected_keyframes = selected_key_frame_ids(caption_record)
    timeline = build_timeline_text(caption_record)
    target_specs = format_target_specs(caption_record, eligible_keyframes, selected_keyframes)
    all_facts = format_facts(caption_record.get("keyframe_facts", []))
    return f"""\
You are a video QA generation assistant.

You will be given:
1. a timeline of window captions for one video;
2. selected key frames and their timestamps/frame ids;
3. fine-grained visual facts extracted around key frames;
4. key-frame images or nearby evidence images.

Your task is to generate high-quality multiple-choice QA candidates for video understanding.

Only generate questions that are answerable from the provided visual evidence.
Do not generate open-ended questions, yes/no questions, or questions requiring outside knowledge.
Frame id equals timestamp in seconds for this dataset.

=== Video ID ===
{caption_record['video_id']}

=== Video Timeline Captions ===
{timeline}

=== Eligible Target Keyframes ===
{target_specs}

=== All Image-Derived Facts ===
{all_facts}

Generate exactly {num_questions} QA candidates in total, not per key frame. It is acceptable to return fewer only when there are not enough reliable visual facts.

Question types:
1. realtime:
   The question is asked near the answer time. It should require inspecting the current or nearby visual evidence.
2. forward:
   The question is asked before the answer is visually available. It should only become answerable at a later key frame.
3. backward:
   The question is asked after the answer has already appeared. It should require using earlier visual history.

Rules for all questions:
1. Each question must have exactly four options: A, B, C, D.
2. Exactly one option must be correct.
3. The correct answer must be directly supported by the evidence frames.
4. Distractors must be plausible but visually wrong.
5. Do not make the correct option longer, more specific, or stylistically different than the distractors.
6. The question and options alone must not reveal the answer.
7. Avoid options like "all of the above", "none of the above", "cannot be determined", or "unknown".
8. Avoid commonsense-only questions that can be answered without looking at the video.
9. Avoid questions where the answer can be guessed from the wording.
10. Avoid asking about invisible intentions, emotions, causes, or identities.
11. Prefer questions grounded in fine visual facts: object attributes, text, colors, spatial relations, hand-object interaction, pose, count, or local temporal change.
12. Do not generate a question if the visual fact is too weak, ambiguous, or not uniquely answerable.

Additional rules for realtime questions:
1. ask_time should be close to answer_time.
2. The question should require the evidence image, not just the text timeline.
3. The timeline description alone should not make the answer obvious.

Additional rules for forward questions:
1. ask_time must be earlier than answer_time.
2. The answer must not be knowable from visual evidence before ask_time.
3. The question should sound natural at ask_time.
4. The clue_time should be the later time/frame where the answer becomes visible.

Additional rules for backward questions:
1. ask_time must be after the answer evidence appears.
2. The answer should be supported by earlier visual history.
3. The question should not require future frames after ask_time.

Temporal constraints:
- For realtime, ask_time must be inside realtime_query_range and answer_time must equal source_keyframe_id.
- For forward, ask_time must be inside forward_query_range and answer_time must equal source_keyframe_id.
- For backward, ask_time must be inside backward_query_range and answer_time must equal source_keyframe_id.
- Do not create a forward or backward question when the corresponding query range is null.

Return JSON only:
{{
  "qa_candidates": [
    {{
      "type": "realtime|forward|backward",
      "question": "...",
      "options": {{
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
      }},
      "gt": "A|B|C|D",
      "answer": "...",
      "ask_time": 0,
      "answer_time": 45,
      "evidence_frame_ids": [45],
      "source_facts": ["..."],
      "rationale": "Brief explanation of why the answer is supported by the visual evidence."
    }}
  ]
}}

Before writing the final JSON, silently check each candidate:
- Can the answer be inferred from the question and options alone?
- Can the answer be inferred from the text timeline alone?
- Is there exactly one correct option?
- Are all distractors plausible?
- Is the evidence visual and specific?
Only include candidates that pass these checks.
Do not output the checklist.
"""


def format_target_specs(caption_record: JsonDict, eligible_keyframes: list[int], selected_keyframes: list[int]) -> str:
    sampled_frames = int(caption_record["sampled_frames"])
    lines = []
    for keyframe_id in eligible_keyframes:
        previous_keyframe, next_keyframe = previous_next_keyframe(selected_keyframes, keyframe_id)
        forward_range = None
        backward_range = None
        if previous_keyframe is not None and 10 < keyframe_id - previous_keyframe < 60:
            forward_range = [previous_keyframe + 1, keyframe_id - 1]
        if next_keyframe is not None and 10 < next_keyframe - keyframe_id < 60:
            backward_range = [keyframe_id + 1, next_keyframe - 1]
        realtime_range = [max(0, keyframe_id - 1), min(sampled_frames - 1, keyframe_id + 1)]
        visible_window = [max(0, keyframe_id - 2), min(sampled_frames - 1, keyframe_id + 2)]
        lines.append(
            f"- source_keyframe_id={keyframe_id}; "
            f"visible_window={visible_window}; "
            f"realtime_query_range={realtime_range}; "
            f"forward_query_range={forward_range}; "
            f"backward_query_range={backward_range}; "
            f"facts={facts_for_keyframe(caption_record, keyframe_id)}"
        )
    return "\n".join(lines)


def qa_context_frame_ids(caption_record: JsonDict, eligible_keyframes: list[int]) -> list[int]:
    sampled_frames = int(caption_record["sampled_frames"])
    frame_ids: set[int] = set()
    for keyframe_id in eligible_keyframes:
        start = max(0, keyframe_id - 2)
        end = min(sampled_frames - 1, keyframe_id + 2)
        frame_ids.update(range(start, end + 1))
    return sorted(frame_ids)


def build_timeline_text(caption_record: JsonDict) -> str:
    lines = []
    for window in caption_record.get("windows", []):
        start, end = window.get("time", ["?", "?"])
        if window.get("type") == "keyframe":
            caption = window.get("whole_window_caption") or window.get("window_caption") or window.get("caption") or ""
            keyframes = ",".join(str(x) for x in window.get("source_keyframe_ids", []))
            lines.append(f"[{start}-{end}] keyframe_window(k={keyframes}): {caption}")
        else:
            lines.append(f"[{start}-{end}] normal: {window.get('caption', '')}")
    return "\n".join(lines)


def facts_for_keyframe(caption_record: JsonDict, keyframe_id: int) -> list[JsonDict]:
    return [
        fact
        for fact in caption_record.get("keyframe_facts", [])
        if int(fact.get("source_keyframe_id", -1)) == int(keyframe_id)
    ]


def selected_key_frame_ids(caption_record: JsonDict) -> list[int]:
    raw = caption_record.get("selected_key_frame_ids", caption_record.get("selected_keyframe_ids", []))
    if not isinstance(raw, list):
        return []
    return [int(frame_id) for frame_id in raw]


def format_facts(facts: list[JsonDict]) -> str:
    if not facts:
        return "- <none>"
    lines = []
    for fact in facts:
        lines.append(
            "- "
            f"k={fact.get('source_keyframe_id')}; "
            f"fact={fact.get('fact')}; "
            f"category={fact.get('category', fact.get('fact_type'))}; "
            f"evidence={fact.get('evidence_frame_ids')}"
        )
    return "\n".join(lines)


def find_window_for_keyframe(caption_record: JsonDict, keyframe_id: int) -> JsonDict | None:
    for window in caption_record.get("windows", []):
        if int(keyframe_id) in {int(k) for k in window.get("source_keyframe_ids", [])}:
            return window
    return None


def previous_next_keyframe(selected_keyframes: list[int], target_keyframe_id: int) -> tuple[int | None, int | None]:
    ordered = sorted(selected_keyframes)
    previous_values = [k for k in ordered if k < target_keyframe_id]
    next_values = [k for k in ordered if k > target_keyframe_id]
    previous_keyframe = previous_values[-1] if previous_values else None
    next_keyframe = next_values[0] if next_values else None
    return previous_keyframe, next_keyframe


def run_cli(args: argparse.Namespace) -> None:
    records = read_jsonl(args.captions)
    if args.limit:
        records = records[: args.limit]
    client = VLMClient.from_backend(args.qa_backend, max_tokens=args.max_tokens)
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    for idx, record in enumerate(records, start=1):
        result = generate_qa_record(
            record,
            client,
            num_questions=args.num_questions,
            max_attempts=args.max_attempts,
            raw_response_chars=args.raw_response_chars,
            raw_data_root=args.raw_data_root,
        )
        append_jsonl(result, args.output)
        print(
            f"[{idx}/{len(records)}] qa video={record['video_id']} "
            f"candidates={len(result['qa_candidates'])} errors={len(result['errors'])}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, default=Path("data_engine/synthesize/outputs/captions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data_engine/synthesize/outputs/qa_candidates.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--raw-data-root", type=Path, default=Path("raw_data"))
    parser.add_argument("--qa-backend", default="gemini")
    parser.add_argument("--num-questions", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_QA_MAX_ATTEMPTS)
    parser.add_argument(
        "--raw-response-chars",
        type=int,
        default=DEFAULT_RAW_RESPONSE_CHARS,
        help="Characters of each raw QA model response to store. Use -1 for full text, 0 to omit.",
    )
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_cli(parse_args())
