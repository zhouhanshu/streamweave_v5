"""Shared OVO-Bench task and scoring helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


BACKWARD_TASKS = ("EPM", "ASI", "HLD")
REAL_TIME_TASKS = ("OCR", "ACR", "ATR", "STU", "FPD", "OJR")
FORWARD_TASKS = ("REC", "SSR", "CRR")

CATEGORY_TASKS = {
    "backward": BACKWARD_TASKS,
    "realtime": REAL_TIME_TASKS,
    "forward": FORWARD_TASKS,
}
TASK_CATEGORY = {task: category for category, tasks in CATEGORY_TASKS.items() for task in tasks}
OVO_TASKS = set(TASK_CATEGORY)
MCQ_TASKS = set(BACKWARD_TASKS) | set(REAL_TIME_TASKS)
YES_NO_TASKS = {"SSR", "CRR"}
OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
YES_NO_RE = re.compile(r"\b(YES|Y|NO|N)\b", flags=re.IGNORECASE)

BR_PROMPT_TEMPLATE = """\
Question: {question}
Options:
{options}

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response.
"""

REC_PROMPT_TEMPLATE = """\
You're watching a video in which people may perform a certain type of action repetitively.
The person performing this kind of action is referred to as "they" in the following statement.
Your task is to track the running total of how many times different people in the video have performed this kind of action in total.
One complete motion counts as one.
Target counting question: {question}
Keep this target action in mind throughout the video.
Whenever the running total changes, answer with the updated total count as a single number (e.g., 0, 1, 2, 3...).
If the running total has not changed in the current frames, keep <answer></answer> empty and preserve useful counting evidence in Memory.
The final non-empty answer should be the total count up to the latest observed frame.
"""

SSR_PROMPT_TEMPLATE = """\
You're watching a tutorial video which contains a sequence of steps.
The following is one step from the whole procedure:
{step}
Your task is to determine whether the current visual content belongs to this step.
Answer "Yes" if the current frames show the person preparing for, starting, continuing, completing, or showing the immediate result of this step, or if the current frames have any reasonable relation to this step.
Answer "No" only if the current frames are completely a different step or provide no visual evidence related to this step.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""

CRR_PROMPT_TEMPLATE = """\
You're responsible for answering questions based on the video content.
Target question, do not answer it directly:
{question}
Only judge whether the existing visual content, especially the latest frames near the end of the video, provides enough information to answer the target question.
Answer "Yes" if the answer to the target question can be inferred from the visible content, or if there is any visual clue related to the target question.
Answer "No" only if the necessary visual evidence is not visible, or if the visible content remains ambiguous and completely unrelated to the target question. Do not be too strict.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""


def category_for_task(task: str) -> str:
    try:
        return TASK_CATEGORY[task]
    except KeyError as exc:
        raise ValueError(f"Unknown OVO task: {task}") from exc


def is_mcq_task(task: str) -> bool:
    return task in MCQ_TASKS


def option_label(index: int) -> str:
    if index < 0 or index >= len(OPTION_LABELS):
        raise ValueError(f"Option index out of range: {index}")
    return OPTION_LABELS[index]


def option_letter_from_gt(gt: Any) -> str:
    return option_label(int(gt))


def format_options(options: Sequence[Any]) -> str:
    return "; ".join(f"{option_label(index)}. {option}" for index, option in enumerate(options)) + ";"


def build_mcq_query(question: str, options: Sequence[Any]) -> str:
    return BR_PROMPT_TEMPLATE.format(question=question, options=format_options(options))


def build_forward_query(task: str, row: dict[str, Any], info: dict[str, Any] | None = None) -> str:
    info = info or {}
    if task == "REC":
        return REC_PROMPT_TEMPLATE.format(question=f"How many times did they {row.get('activity', '')}?")
    if task == "SSR":
        return SSR_PROMPT_TEMPLATE.format(step=info.get("step", ""))
    if task == "CRR":
        return CRR_PROMPT_TEMPLATE.format(question=row.get("question", ""))
    raise ValueError(f"Unknown OVO task: {task}")


def extract_mcq(response: str | None, *, max_options: int = 5) -> str | None:
    if response is None or not str(response).strip():
        return None
    if max_options <= 0:
        return None
    max_options = min(max_options, len(OPTION_LABELS))
    labels = OPTION_LABELS[:max_options]
    text = str(response).strip()
    letter = re.search(rf"\b([{re.escape(labels)}])\b", text.upper())
    if letter:
        return letter.group(1)
    for number in re.finditer(r"\b(\d+)\b", text):
        value = int(number.group(1))
        if 1 <= value <= max_options:
            return option_label(value - 1)
    return None


def extract_yes_no(response: str | None) -> int | None:
    if response is None or not str(response).strip():
        return None
    polarities = []
    for match in YES_NO_RE.finditer(str(response).upper()):
        token = match.group(1)
        polarities.append(1 if token in {"Y", "YES"} else 0)
    if not polarities:
        return None
    unique = set(polarities)
    if len(unique) > 1:
        return None
    return polarities[0]
