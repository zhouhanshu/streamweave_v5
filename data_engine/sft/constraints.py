"""SFT-only validation rules layered on top of StreamWeave XML parsing."""

from __future__ import annotations

import re

from streamweave.schemas import ModelAction, QualityReport, ValidationIssue

from .io_utils import JsonDict
from .schemas import FrameRef as PlanFrameRef
from .schemas import SamplePlan
from .timing import target_window_for_time

QA_TIME_TOLERANCE = 0.51


def note_reminder_context(latest_bridge_seconds: float | None, threshold_seconds: float) -> str:
    if threshold_seconds <= 0:
        return ""
    if latest_bridge_seconds is None or latest_bridge_seconds <= threshold_seconds:
        return ""
    return (
        "There has not been a recent <note>. If the current frames contain useful visual evidence, "
        "please preserve one representative current frame with a <note> as soon as possible."
    )


def apply_note_count_constraint(
    *,
    quality: QualityReport,
    raw_action: ModelAction,
    max_notes_per_step: int,
) -> None:
    if max_notes_per_step <= 0:
        return

    note_times = [
        _interval_key(event.start_time, event.end_time)
        for event in raw_action.events
        if event.kind == "note"
    ]
    quality.metrics["note_count"] = len(note_times)
    quality.metrics["max_notes_per_step"] = max_notes_per_step
    if len(note_times) > max_notes_per_step:
        quality.issues.append(
            ValidationIssue(
                "too_many_notes_in_step",
                (
                    f"This step outputs {len(note_times)} notes, but at most {max_notes_per_step} note is allowed. "
                    f"Output note time ranges: {_format_interval_keys(note_times)}. "
                    "Retry by keeping only the single most representative current-frame note."
                ),
            )
        )
        quality.valid = False
        quality.rewards.note_bridge_timing_reward = 0


def apply_qa_answer_constraints(
    *,
    quality: QualityReport,
    raw_action: ModelAction,
    qa_before: list[JsonDict],
    plan_frames: list[PlanFrameRef],
    sample: SamplePlan,
    frames_per_step: int,
) -> None:
    expected = _expected_qa_output(sample, qa_before, plan_frames, frames_per_step)
    if expected is None:
        return

    actual_answer = raw_action.answer.strip()
    answer_required = expected["answer_required"]
    if answer_required is None:
        answer_ok = True
    elif answer_required:
        answer_ok = bool(actual_answer)
    else:
        answer_ok = actual_answer == ""

    quality.metrics["expected_answer_required"] = answer_required
    quality.metrics["expected_answer_window"] = expected.get("target_window")
    quality.metrics["qa_schedule_reason"] = expected["reason"]

    if answer_ok:
        return

    template = _format_answer_template(expected)
    actual_summary = f"actual answer={_actual_answer_state(actual_answer)}"
    reason = expected.get("reason") or ""
    quality.issues.append(
        ValidationIssue(
            "qa_answer_mismatch",
            (
                "QA answer mismatch. "
                f"Required answer tag:\n{template}\n"
                f"Reason: {reason}. {actual_summary}."
            ),
        )
    )
    quality.valid = False
    quality.rewards.format_reward = 0


def check_sample_answer(sample: SamplePlan, steps: list[JsonDict]) -> JsonDict:
    expected = _expected_sample_answer(sample)
    model_answers = _sample_model_answers(steps)
    if not expected["applicable"]:
        correct = not model_answers
        return {
            **expected,
            "model_answers": model_answers,
            "answer_correct": correct,
            "reason": "no GT answer is available; accepted only if the sample emitted no answer" if correct else "sample emitted an answer but no GT answer is available",
        }

    matches = [_answer_matches(item["answer"], expected) for item in model_answers]
    correct = bool(model_answers) and all(matches)
    return {
        **expected,
        "model_answers": model_answers,
        "answer_correct": correct,
        "reason": "all emitted answers match GT" if correct else "missing answer or emitted answer does not match GT",
    }


def answer_matches_metadata(metadata: JsonDict, task: str, answer: str) -> bool | None:
    expected = _expected_answer_from_metadata(_merge_nested_metadata(metadata), task)
    if not expected["applicable"]:
        return None
    return _answer_matches(answer, expected)


def _expected_sample_answer(sample: SamplePlan) -> JsonDict:
    metadata = _answer_metadata(sample)
    return _expected_answer_from_metadata(metadata, sample.task)


def _expected_answer_from_metadata(metadata: JsonDict, sample_task: str) -> JsonDict:
    task = str(metadata.get("task") or sample_task or "").strip()
    options = metadata.get("options")
    if not isinstance(options, list):
        options = []
    options = [str(option).strip() for option in options]
    answer_text = str(metadata.get("answer") or "").strip()
    gt = metadata.get("gt")
    if gt is None:
        gt = metadata.get("ground_truth")
    option_index = _expected_option_index(gt, options, answer_text)
    if option_index is not None:
        return {
            "applicable": True,
            "gt": gt,
            "expected_option_index": option_index,
            "expected_letter": chr(ord("A") + option_index),
            "expected_answer": options[option_index],
            "options": options,
        }
    expected_letter = _expected_letter_without_options(gt)
    if expected_letter:
        return {
            "applicable": True,
            "gt": gt,
            "expected_option_index": None,
            "expected_letter": expected_letter,
            "expected_answer": "",
            "options": options,
        }
    expected_answer = _expected_non_option_answer(task, gt)
    if expected_answer:
        return {
            "applicable": True,
            "gt": gt,
            "expected_option_index": None,
            "expected_letter": "",
            "expected_answer": expected_answer,
            "options": options,
        }
    if answer_text:
        return {
            "applicable": True,
            "gt": gt,
            "expected_option_index": None,
            "expected_letter": "",
            "expected_answer": answer_text,
            "options": options,
        }
    return {
        "applicable": False,
        "gt": gt,
        "expected_option_index": None,
        "expected_letter": "",
        "expected_answer": "",
        "options": options,
    }


def _answer_metadata(sample: SamplePlan) -> JsonDict:
    return _merge_nested_metadata(sample.metadata)


def _merge_nested_metadata(metadata: JsonDict) -> JsonDict:
    metadata = dict(metadata)
    nested = metadata.get("sample_metadata")
    if isinstance(nested, dict):
        merged = dict(nested)
        merged.update(metadata)
        return merged
    return metadata


def _expected_letter_without_options(gt: object) -> str:
    if not isinstance(gt, str):
        return ""
    text = gt.strip().upper()
    return text if len(text) == 1 and "A" <= text <= "Z" else ""


def _expected_non_option_answer(task: str, gt: object) -> str:
    if gt is None:
        return ""
    task = task.upper()
    if task == "REC":
        return str(gt).strip()
    if task in {"SSR", "CRR"}:
        try:
            return "Yes" if int(gt) == 1 else "No"
        except (TypeError, ValueError):
            text = str(gt).strip().lower()
            if text in {"yes", "no"}:
                return text.capitalize()
    return ""


def _expected_option_index(gt: object, options: list[str], answer_text: str) -> int | None:
    if not options:
        return None
    if isinstance(gt, str):
        text = gt.strip()
        if len(text) == 1 and text.upper().isalpha():
            index = ord(text.upper()) - ord("A")
            return index if 0 <= index < len(options) else None
        try:
            raw_index = int(text)
        except ValueError:
            normalized = _normalize_answer_text(text)
            for index, option in enumerate(options):
                if _normalize_answer_text(option) == normalized:
                    return index
            return None
    else:
        try:
            raw_index = int(gt)
        except (TypeError, ValueError):
            return None

    answer_norm = _normalize_answer_text(answer_text)
    zero_based_ok = 0 <= raw_index < len(options)
    one_based_ok = 1 <= raw_index <= len(options)
    if answer_norm:
        if zero_based_ok and _normalize_answer_text(options[raw_index]) == answer_norm:
            return raw_index
        if one_based_ok and _normalize_answer_text(options[raw_index - 1]) == answer_norm:
            return raw_index - 1
    if zero_based_ok:
        return raw_index
    if one_based_ok:
        return raw_index - 1
    return None


def _sample_model_answers(steps: list[JsonDict]) -> list[JsonDict]:
    answers: list[JsonDict] = []
    for row in steps:
        metadata = row.get("metadata") or {}
        raw_action = metadata.get("raw_action") if isinstance(metadata, dict) else {}
        answer = ""
        if isinstance(raw_action, dict):
            answer = str(raw_action.get("answer") or "").strip()
        if not answer:
            answer = _extract_answer_from_xml(str(row.get("target_xml") or ""))
        if answer:
            answers.append({"step_index": row.get("step_index"), "answer": answer})
    return answers


def _extract_answer_from_xml(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return (match.group(1).strip() if match else "")


def _answer_matches(answer: str, expected: JsonDict) -> bool:
    answer_norm = _normalize_answer_text(answer)
    expected_letter = str(expected.get("expected_letter") or "").strip().upper()
    if expected_letter:
        letter_match = re.match(r"^(?:option\s*)?([A-Z])(?:[).:]|\s|$)", answer.strip(), flags=re.IGNORECASE)
        if letter_match and letter_match.group(1).upper() == expected_letter:
            return True
    expected_answer = _normalize_answer_text(str(expected.get("expected_answer") or ""))
    return bool(expected_answer and answer_norm == expected_answer)


def _normalize_answer_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"^(?:option\s*)?[a-z][).:]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .。")


def _interval_key(start: float, end: float) -> tuple[float, float]:
    return (round(float(start), 3), round(float(end), 3))


def _format_interval_keys(values: list[tuple[float, float]] | set[tuple[float, float]]) -> list[str]:
    return [f"{start:.1f}-{end:.1f}" for start, end in values]


def _expected_qa_output(
    sample: SamplePlan,
    qa_before: list[JsonDict],
    frames: list[PlanFrameRef],
    frames_per_step: int = 1,
) -> JsonDict | None:
    task = sample.task.lower().strip()
    has_question = any(str(item.get("role") or "") == "q" for item in qa_before if isinstance(item, dict))
    if not has_question:
        return _qa_expectation(False, "no question is present in QA History, so answer must stay empty")
    if task in {"realtime", "backward"}:
        window_start, window_end = target_window_for_time(_first_query_time(sample, qa_before), sample, frames, frames_per_step)
        if _has_answer(qa_before):
            return _qa_expectation(
                None,
                f"{task} question has already been answered; later answer updates are optional",
                target_window=(window_start, window_end),
            )
        return _qa_expectation(
            True,
            f"{task} question is active and unanswered, so answer now",
            target_window=(window_start, window_end),
        )
    if task == "forward":
        clue_time = _forward_clue_time(sample)
        if clue_time is None:
            return None
        window_start, window_end = target_window_for_time(clue_time, sample, frames, frames_per_step)
        if _has_answer(qa_before):
            return _qa_expectation(
                None,
                "forward question has already been answered; later answer updates are optional",
                target_window=(window_start, window_end),
            )
        due = frames[-1].end_time >= window_end - QA_TIME_TOLERANCE if frames else False
        if due:
            return _qa_expectation(
                True,
                "forward question has reached the clue window, so answer now",
                target_window=(window_start, window_end),
            )
        return _qa_expectation(
            False,
            "forward question is active before clue_time, so keep answer empty",
            target_window=(window_start, window_end),
        )
    return None


def _qa_expectation(
    answer_required: bool | None,
    reason: str,
    target_window: tuple[float, float] | None = None,
) -> JsonDict:
    return {
        "answer_required": answer_required,
        "reason": reason,
        "target_window": [target_window[0], target_window[1]] if target_window is not None else None,
    }


def _has_answer(qa_before: list[JsonDict]) -> bool:
    return any(str(item.get("role") or "") == "a" for item in qa_before if isinstance(item, dict))


def _actual_answer_state(actual_answer: str) -> str:
    return "non-empty" if actual_answer else "empty"


def _first_query_time(sample: SamplePlan, qa_before: list[JsonDict]) -> float:
    if sample.query_time is not None:
        return float(sample.query_time)
    if sample.query_events:
        return float(sample.query_events[0].timestamp)
    for item in qa_before:
        if isinstance(item, dict) and str(item.get("role") or "") == "q":
            return float(item.get("t", item.get("timestamp", 0.0)) or 0.0)
    return 0.0


def _forward_clue_time(sample: SamplePlan) -> float | None:
    for key in ("clue_time", "answer_time"):
        if key in sample.metadata and sample.metadata[key] is not None:
            return float(sample.metadata[key])
    return sample.answer_time


def _format_answer_template(expected: JsonDict) -> str:
    answer_required = expected.get("answer_required")
    if answer_required:
        return "<answer>...your answer based on QA History, Memory, and Current frames...</answer>"
    if answer_required is None:
        return "<answer></answer> or <answer>...updated answer...</answer>"
    return "<answer></answer>"
