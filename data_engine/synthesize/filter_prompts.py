"""Prompt builders for QA filtering and auditing."""

from __future__ import annotations

from data_engine.synthesize.io_utils import JsonDict
from data_engine.synthesize.mcq_utils import correct_choice, format_options


def option_quality_prompt(qa: JsonDict) -> str:
    return f"""\
You are a multiple-choice QA quality auditor.

You will be given a question, four options, and the intended correct option.
Your job is to detect flaws in the question and options.

Check for:
1. more than one plausible correct answer;
2. no clearly valid correct answer;
3. correct option is much longer or more detailed than distractors;
4. distractors are silly, impossible, or grammatically mismatched;
5. options are not parallel in style;
6. question is yes/no-like or open-ended despite having options;
7. question can be answered by commonsense without visual evidence;
8. question contains ambiguous references such as "it", "that", "there" without clear context;
9. question asks about invisible intent, emotion, identity, or causality;
10. options include "unknown", "none of the above", or "all of the above".

Question:
{qa['question']}

Options:
{format_options(qa.get('options', []))}

Intended correct option: {correct_choice(qa)}

Output JSON only:
{{
  "is_valid_mcq": true,
  "has_single_correct_answer": true,
  "has_option_bias": false,
  "is_visual_question": true,
  "problems": [
    "..."
  ]
}}
"""


def text_timeline_prompt(timeline: str, qa: JsonDict) -> str:
    return f"""\
You are a multiple-choice QA verifier.

You will be given:
1. a video timeline described only in text;
2. a multiple-choice question;
3. four options.

You must answer the question using only the text timeline.
Do not use images.
Do not use outside knowledge.
Do not answer UNKNOWN.
You must choose exactly one option from A, B, C, or D.

After choosing, estimate whether the text timeline alone supports the answer.

Text timeline:
{timeline}

Question:
{qa['question']}

Options:
{format_options(qa.get('options', []))}

Output JSON only:
{{
  "choice": "A|B|C|D",
  "confidence": 0.0,
  "text_support": "fully_supported|partially_supported|not_supported",
  "evidence_from_text": "...",
  "reason": "..."
}}
"""


def realtime_visual_prompt(qa: JsonDict, frame_ids: list[int]) -> str:
    return f"""\
You are a visual multiple-choice QA verifier.

You will be given:
1. one or more evidence frames from a video;
2. a multiple-choice question;
3. four options.

Answer the question using only the provided image frames.
Do not use outside knowledge.
Do not answer UNKNOWN.
You must choose exactly one option from A, B, C, or D.

Choose the option best supported by the visual evidence.
If the image is ambiguous, still choose the best option, but lower your confidence.

Evidence frame ids: {frame_ids}

Question:
{qa['question']}

Options:
{format_options(qa.get('options', []))}

Output JSON only:
{{
  "choice": "A|B|C|D",
  "confidence": 0.0,
  "visual_support": "fully_supported|partially_supported|not_supported",
  "evidence_frame_ids": [0],
  "visual_evidence": "...",
  "reason": "..."
}}
"""


def forward_history_prompt(history_timeline: str, qa: JsonDict) -> str:
    return f"""\
You are a forward-question leakage verifier.

You will be given:
1. the video history before the question ask_time;
2. a multiple-choice question;
3. four options.

The question is intended to be a forward-looking question.
At ask_time, the answer should not yet be knowable.

Use only the history before ask_time.
Do not use later frames.
Do not answer UNKNOWN.
You must choose exactly one option from A, B, C, or D.

Your task:
1. Make a forced-choice guess.
2. Decide whether the history before ask_time already supports the correct answer.

ask_time: {qa['query_time']}
History timeline before ask_time:
{history_timeline}

Question:
{qa['question']}

Options:
{format_options(qa.get('options', []))}

Output JSON only:
{{
  "choice": "A|B|C|D",
  "confidence": 0.0,
  "history_support": "fully_supported|partially_supported|not_supported",
  "is_answer_available_before_ask_time": true,
  "reason": "..."
}}
"""


def forward_visual_prompt(qa: JsonDict, frame_ids: list[int]) -> str:
    return f"""\
You are a visual verifier for a forward-looking multiple-choice video question.

You will be given:
1. evidence frames from the answer/clue time;
2. a multiple-choice question;
3. four options.

Use only the provided evidence frames.
Do not use outside knowledge.
Do not answer UNKNOWN.
You must choose exactly one option from A, B, C, or D.

Evidence frame ids: {frame_ids}

Question:
{qa['question']}

Options:
{format_options(qa.get('options', []))}

Output JSON only:
{{
  "choice": "A|B|C|D",
  "confidence": 0.0,
  "visual_support": "fully_supported|partially_supported|not_supported",
  "visual_evidence": "...",
  "reason": "..."
}}
"""


def backward_history_prompt(history_timeline: str, qa: JsonDict) -> str:
    return f"""\
You are a backward-looking video QA verifier.

You will be given:
1. the video history before the question ask_time;
2. optionally, evidence frames from before ask_time;
3. a multiple-choice question;
4. four options.

The question is intended to be answerable from past visual history.
Use only information before ask_time.
Do not use future frames.
Do not use outside knowledge.
Do not answer UNKNOWN.
You must choose exactly one option from A, B, C, or D.

ask_time: {qa['query_time']}
History timeline before ask_time:
{history_timeline}

Question:
{qa['question']}

Options:
{format_options(qa.get('options', []))}

Output JSON only:
{{
  "choice": "A|B|C|D",
  "confidence": 0.0,
  "history_support": "fully_supported|partially_supported|not_supported",
  "evidence_frame_ids": [0],
  "evidence": "...",
  "reason": "..."
}}
"""
