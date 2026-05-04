"""Pluggable answer scorers for StreamWeave RL."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from typing import Any


ScoreFn = Callable[[str, Any, Mapping[str, Any]], float]

BACKWARD_TASKS = {"EPM", "ASI", "HLD"}
REAL_TIME_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
FORWARD_TASKS = {"REC", "SSR", "CRR"}
OVO_TASKS = BACKWARD_TASKS | REAL_TIME_TASKS | FORWARD_TASKS


def score_answer(
    answer: str,
    ground_truth: Any,
    *,
    scorer: str = "auto",
    metadata: Mapping[str, Any] | None = None,
    fallback_mode: str = "exact_or_contains",
) -> float:
    meta = metadata or {}
    name = (scorer or "auto").strip()
    if ":" in name:
        return float(_load_custom_scorer(name)(answer, ground_truth, meta))
    if name == "auto":
        name = _infer_scorer(meta)
    if name in {"default", "exact", "contains", "exact_or_contains"}:
        mode = fallback_mode if name == "default" else name
        return _score_default(answer, ground_truth, mode=mode)
    if name in {"streamweave", "streamweave_mcq", "mcq"}:
        return _score_streamweave_mcq(answer, ground_truth, meta)
    if name == "ovo":
        return _score_ovo(answer, ground_truth, meta)
    raise ValueError(f"Unknown StreamWeave success scorer: {scorer}")


def _infer_scorer(metadata: Mapping[str, Any]) -> str:
    dataset = str(metadata.get("dataset") or metadata.get("benchmark") or "").lower()
    task = str(metadata.get("task") or "").strip()
    if dataset == "ovo" or task in OVO_TASKS:
        return "ovo"
    if metadata.get("options") is not None or metadata.get("gt") is not None:
        return "streamweave_mcq"
    return "default"


def _load_custom_scorer(spec: str) -> ScoreFn:
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"Configured scorer is not callable: {spec}")
    return func


def _score_default(answer: str, ground_truth: Any, *, mode: str) -> float:
    answer_norm = _normalize_answer_text(answer)
    gt_norm = _normalize_answer_text(str(ground_truth or ""))
    if not answer_norm or not gt_norm:
        return 0.0
    if mode == "exact":
        return 1.0 if answer_norm == gt_norm else 0.0
    if mode == "contains":
        return 1.0 if gt_norm in answer_norm else 0.0
    if mode == "exact_or_contains":
        return 1.0 if answer_norm == gt_norm or gt_norm in answer_norm or answer_norm in gt_norm else 0.0
    raise ValueError(f"Unknown default success mode: {mode}")


def _score_streamweave_mcq(answer: str, ground_truth: Any, metadata: Mapping[str, Any]) -> float:
    options = _normalize_options(metadata.get("options"))
    answer_text = str(metadata.get("answer") or "").strip()
    gt = metadata.get("gt", ground_truth)
    option_index = _expected_option_index(gt, options, answer_text)
    if option_index is not None:
        expected_letter = chr(ord("A") + option_index)
        expected_answer = options[option_index]
        return 1.0 if _answer_matches(answer, expected_letter=expected_letter, expected_answer=expected_answer) else 0.0
    expected = answer_text or str(ground_truth or "").strip()
    return _score_default(answer, expected, mode="exact")


def _score_ovo(answer: str, ground_truth: Any, metadata: Mapping[str, Any]) -> float:
    task = str(metadata.get("task") or "").strip()
    gt = metadata.get("ground_truth", ground_truth)
    if task in BACKWARD_TASKS or task in REAL_TIME_TASKS:
        pred = _extract_mcq(answer)
        return 1.0 if pred is not None and pred.upper() == str(gt).strip().upper() else 0.0
    if task == "REC":
        nums = re.findall(r"\d+", str(answer or ""))
        return 1.0 if nums and "".join(nums) == str(gt).strip() else 0.0
    if task in {"SSR", "CRR"}:
        text = str(answer or "").strip().upper()
        try:
            gt_int = int(gt)
        except (TypeError, ValueError):
            return 0.0
        if (text == "N" or "NO" in text) and gt_int == 0:
            return 1.0
        if (text == "Y" or "YES" in text) and gt_int == 1:
            return 1.0
        return 0.0
    return _score_default(answer, gt, mode="exact_or_contains")


def _normalize_options(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        items = [value.get(choice) for choice in ("A", "B", "C", "D")]
    elif isinstance(value, list):
        items = value
    else:
        return []
    out: list[str] = []
    for item in items:
        if item is None:
            continue
        out.append(str(item).strip())
    return out


def _expected_option_index(gt: Any, options: list[str], answer_text: str) -> int | None:
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


def _answer_matches(answer: str, *, expected_letter: str, expected_answer: str) -> bool:
    pred = _extract_mcq(answer)
    if pred is not None and pred.upper() == expected_letter:
        return True
    return bool(_normalize_answer_text(answer) == _normalize_answer_text(expected_answer))


def _extract_mcq(answer: str | None) -> str | None:
    if answer is None or not str(answer).strip():
        return None
    text = str(answer).strip()
    letter = re.search(r"\b([A-D])\b", text.upper())
    if letter:
        return letter.group(1)
    number = re.search(r"\b([1-4])\b", text)
    if number:
        return chr(64 + int(number.group(1)))
    return None


def _normalize_answer_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"^(?:option\s*)?[a-z][).:]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")
