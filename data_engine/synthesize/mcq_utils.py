"""Utilities for normalizing and checking multiple-choice QA fields."""

from __future__ import annotations

import re
from typing import Any


CHOICES = ("A", "B", "C", "D")
SUPPORTED_LABELS = {"fully_supported", "partially_supported"}


def normalize_options(value: object) -> list[str]:
    """Return options as ["A. ...", "B. ...", "C. ...", "D. ..."]."""
    if isinstance(value, dict):
        return [
            f"{choice}. {strip_option_prefix(str(value.get(choice, '')))}"
            for choice in CHOICES
            if str(value.get(choice, "")).strip()
        ]
    if not isinstance(value, list):
        return []

    options: list[str] = []
    for idx, item in enumerate(value[: len(CHOICES)]):
        choice = CHOICES[idx]
        text = str(item).strip()
        if not text:
            continue
        if text[:2].upper() in {f"{choice}.", f"{choice})"}:
            options.append(text)
        else:
            options.append(f"{choice}. {strip_option_prefix(text)}")
    return options


def option_map(value: object) -> dict[str, str]:
    """Return a choice->text map without A./B. prefixes."""
    options = normalize_options(value)
    return {
        choice: strip_option_prefix(options[idx]) if idx < len(options) else ""
        for idx, choice in enumerate(CHOICES)
    }


def format_options(value: object) -> str:
    options = option_map(value)
    return "\n".join(f"{choice}. {options[choice]}" for choice in CHOICES)


def strip_option_prefix(text: str) -> str:
    return re.sub(r"^[A-D][.)]\s*", "", text.strip(), flags=re.IGNORECASE).strip()


def extract_choice(text: str) -> str:
    stripped = text.strip().upper()
    if stripped in CHOICES:
        return stripped
    for choice in CHOICES:
        if stripped.startswith(f"{choice}.") or stripped.startswith(f"{choice})"):
            return choice
    match = re.search(r"\b([A-D])\b", stripped)
    return match.group(1) if match else ""


def correct_choice(qa: dict[str, Any]) -> str:
    return extract_choice(str(qa.get("gt", ""))) or extract_choice(str(qa.get("answer", "")))


def choice_to_index(choice: str) -> int | None:
    if choice not in CHOICES:
        return None
    return CHOICES.index(choice)


def option_text_for_choice(options: list[str], choice: str) -> str:
    index = choice_to_index(choice)
    if index is None or index >= len(options):
        return ""
    return strip_option_prefix(options[index])


def choice_for_answer_text(options: list[str], answer_text: str) -> str:
    normalized_answer = normalize_text(answer_text)
    for idx, option in enumerate(options):
        if normalize_text(strip_option_prefix(option)) == normalized_answer:
            return CHOICES[idx]
    return ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))
