"""Strict raw validation and lenient action extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import ModelAction, ModelEvent, ValidationIssue


TOKEN_RE = re.compile(
    r"<eta>(?P<eta>.*?)</eta>"
    r"|<answer>(?P<answer>.*?)</answer>"
    r'|<note\s+t="(?P<note_t>[^"]+)"\s+frame="(?P<note_frame>\d+)">\s*</note>'
    r'|<note\s+t="(?P<self_note_t>[^"]+)"\s+frame="(?P<self_note_frame>\d+)"\s*/>'
    r'|<bridge\s+t="(?P<bridge_t>[^"]+)">(?P<bridge_text>.*?)</bridge>',
    flags=re.DOTALL,
)


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
        action = ModelAction(eta=None, answer="", events=[], raw=raw, eta_present=False, answer_present=False)
        issues.append(ValidationIssue("no_tags", "No valid XML tags found."))
        return ParseResult(action=action, parser_ok=False, issues=issues)

    action, parse_issues = _tokens_to_action(raw, tokens)
    issues.extend(parse_issues)
    kinds = [_token_kind(match) for match in tokens]
    for match in tokens:
        if _is_self_closing_note(match):
            issues.append(
                ValidationIssue(
                    "note_tag_format",
                    'Current output notes must use paired tags like <note t="36.0-37.0" frame="2"></note>, not self-closing <note .../>.',
                )
            )
    if kinds.count("eta") != 1:
        issues.append(ValidationIssue("eta_count", "Raw output must contain exactly one <eta> tag."))
    if kinds.count("answer") != 1:
        issues.append(ValidationIssue("answer_count", "Raw output must contain exactly one <answer> tag."))
    if not action.events:
        issues.append(ValidationIssue("missing_observation", "Raw output must contain at least one note or bridge tag."))
    if len(kinds) >= 2:
        if kinds[0] != "eta" or kinds[1] != "answer":
            issues.append(ValidationIssue("tag_order", "Raw output must start with <eta> then <answer>."))
    else:
        issues.append(ValidationIssue("tag_order", "Raw output must start with <eta> then <answer>."))
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
    eta = None
    answer = ""
    eta_present = False
    answer_present = False
    events: list[ModelEvent] = []
    for index, match in enumerate(tokens):
        kind = _token_kind(match)
        try:
            if kind == "eta":
                eta_present = True
                value = (match.group("eta") or "").strip()
                if value:
                    eta = float(value)
            elif kind == "answer":
                answer_present = True
                answer = (match.group("answer") or "").strip()
            elif kind == "note":
                note_t = match.group("note_t") or match.group("self_note_t")
                note_frame = match.group("note_frame") or match.group("self_note_frame")
                if note_t is None or note_frame is None:
                    raise ValueError("Missing note time or frame attribute")
                start, end = _parse_interval(note_t)
                events.append(
                    ModelEvent(
                        kind="note",
                        start_time=start,
                        end_time=end,
                        frame_index=int(note_frame) - 1,
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
            eta=eta,
            answer=answer,
            events=events,
            raw=raw,
            eta_present=eta_present,
            answer_present=answer_present,
        ),
        issues,
    )


def _token_kind(match: re.Match[str]) -> str:
    if match.group("eta") is not None:
        return "eta"
    if match.group("answer") is not None:
        return "answer"
    if match.group("note_t") is not None or match.group("self_note_t") is not None:
        return "note"
    return "bridge"


def _is_self_closing_note(match: re.Match[str]) -> bool:
    return match.group("self_note_t") is not None


def _parse_interval(text: str) -> tuple[float, float]:
    match = re.match(r"\s*([0-9.]+)\s*[-\u2013]\s*([0-9.]+)\s*$", text)
    if not match:
        raise ValueError(f"Invalid interval: {text!r}")
    start = float(match.group(1))
    end = float(match.group(2))
    if start >= end:
        raise ValueError(f"Invalid non-positive interval: {text!r}")
    return start, end
