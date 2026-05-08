"""Strict raw validation and lenient action extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import ModelAction, ModelEvent, ValidationIssue


TOKEN_RE = re.compile(
    r"<state>(?P<state>.*?)</state>"
    r"|<answer>(?P<answer>.*?)</answer>"
    r'|<note\s+t="(?P<note_t>[^"]+)">\s*</note>'
    r'|<bridge\s+t="(?P<bridge_t>[^"]+)">(?P<bridge_text>.*?)</bridge>',
    flags=re.DOTALL,
)

SELF_CLOSING_NOTE_RE = re.compile(r'<note\b[^>]*\bt="[^"]+"[^>]*/>', flags=re.DOTALL)


@dataclass(slots=True)
class ParseResult:
    action: ModelAction
    parser_ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)


def strict_validate_raw_output(raw: str) -> ParseResult:
    issues: list[ValidationIssue] = []
    tokens = []
    pos = 0
    for match in TOKEN_RE.finditer(raw):
        if raw[pos : match.start()].strip():
            issues.append(ValidationIssue("text_outside_tags", "Raw output contains text outside allowed XML tags."))
        tokens.append(match)
        pos = match.end()
    if raw[pos:].strip():
        issues.append(ValidationIssue("text_outside_tags", "Raw output contains trailing text outside XML tags."))

    if not tokens:
        action = ModelAction(state="", answer="", events=[], raw=raw, state_present=False, answer_present=False)
        issues.append(ValidationIssue("no_tags", "No valid XML tags found."))
        return ParseResult(action=action, parser_ok=False, issues=issues)

    action, parse_issues = _tokens_to_action(raw, tokens)
    issues.extend(parse_issues)
    kinds = [_token_kind(match) for match in tokens]
    if SELF_CLOSING_NOTE_RE.search(raw):
        issues.append(
            ValidationIssue(
                "note_tag_format",
                'Current output notes must use paired tags like <note t="36.0-37.0"></note>, not self-closing <note .../>.',
            )
        )
    if kinds.count("state") != 1:
        issues.append(ValidationIssue("state_count", "Raw output must contain exactly one <state> tag."))
    if action.state_present and not action.state.strip():
        issues.append(ValidationIssue("state_empty", "Raw output <state> must summarize the current state and QA decision."))
    if action.state_present and re.search(r"</?\w", action.state):
        issues.append(ValidationIssue("state_contains_xml", "Raw output <state> must not contain XML tags."))
    if kinds.count("answer") != 1:
        issues.append(ValidationIssue("answer_count", "Raw output must contain exactly one <answer> tag."))
    if not action.events:
        issues.append(ValidationIssue("missing_observation", "Raw output must contain at least one note or bridge tag."))
    if len(kinds) >= 2:
        if kinds[0] != "state" or kinds[1] != "answer":
            issues.append(ValidationIssue("tag_order", "Raw output must start with <state> then <answer>."))
    else:
        issues.append(ValidationIssue("tag_order", "Raw output must start with <state> then <answer>."))
    for kind in kinds[2:]:
        if kind not in {"note", "bridge"}:
            issues.append(ValidationIssue("tag_order", "Only note/bridge tags may appear after <answer>."))

    return ParseResult(action=action, parser_ok=not issues, issues=issues)


def parse_for_repair(raw: str) -> ParseResult:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    tokens = list(TOKEN_RE.finditer(cleaned))
    action, issues = _tokens_to_action(raw, tokens)
    return ParseResult(action=action, parser_ok=bool(tokens), issues=issues)


def _tokens_to_action(raw: str, tokens: list[re.Match[str]]) -> tuple[ModelAction, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    state = ""
    answer = ""
    state_present = False
    answer_present = False
    events: list[ModelEvent] = []
    for index, match in enumerate(tokens):
        kind = _token_kind(match)
        try:
            if kind == "state":
                state_present = True
                state = (match.group("state") or "").strip()
            elif kind == "answer":
                answer_present = True
                answer = (match.group("answer") or "").strip()
            elif kind == "note":
                note_t = match.group("note_t")
                if note_t is None:
                    raise ValueError("Missing note time attribute")
                start, end = _parse_interval(note_t)
                events.append(
                    ModelEvent(
                        kind="note",
                        start_time=start,
                        end_time=end,
                    )
                )
            elif kind == "bridge":
                start, end = _parse_interval(match.group("bridge_t"))
                events.append(
                    ModelEvent(
                        kind="bridge",
                        start_time=start,
                        end_time=end,
                        text=(match.group("bridge_text") or "").strip(),
                    )
                )
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    "tag_parse_error",
                    f"Could not parse {kind} tag #{index + 1}: {exc}.",
                )
            )
    return (
        ModelAction(
            state=state,
            answer=answer,
            events=events,
            raw=raw,
            state_present=state_present,
            answer_present=answer_present,
        ),
        issues,
    )


def _token_kind(match: re.Match[str]) -> str:
    if match.group("state") is not None:
        return "state"
    if match.group("answer") is not None:
        return "answer"
    if match.group("note_t") is not None:
        return "note"
    return "bridge"


def _parse_interval(text: str) -> tuple[float, float]:
    match = re.match(r"\s*([0-9.]+)\s*[-\u2013]\s*([0-9.]+)\s*$", text)
    if not match:
        raise ValueError(f"Invalid interval: {text!r}")
    start = float(match.group(1))
    end = float(match.group(2))
    if start >= end:
        raise ValueError(f"Invalid non-positive interval: {text!r}")
    return start, end
