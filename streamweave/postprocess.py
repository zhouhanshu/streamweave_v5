"""Deterministic postprocessing for eval/rollout."""

from __future__ import annotations

import re

from .parser import parse_for_repair
from .quality import QualityContext
from .schemas import AppliedAction, BridgeRecord, ModelAction, ModelEvent, NoteRecord, QARecord


def repair_for_execution(raw: str, context: QualityContext) -> AppliedAction:
    """Extract a usable action once. This is not a retry loop."""
    parsed = parse_for_repair(raw)
    action = parsed.action
    repaired_events: list[ModelEvent] = []
    notes: list[NoteRecord] = []
    bridges: list[BridgeRecord] = []
    repair_types: list[str] = []
    replace_open_tail = False

    events = list(action.events)
    repaired_from_malformed = _dedupe_events(events, _extract_malformed_notes(raw, context))
    if repaired_from_malformed:
        events.extend(repaired_from_malformed)
        repair_types.append("repair_malformed_note_tags")

    seen_bridge = False
    for event in events:
        if event.kind == "note":
            if event.frame_index is None or not (0 <= event.frame_index < len(context.frames)):
                repair_types.append("drop_note_frame_oob")
                continue
            frame = context.frames[event.frame_index]
            if abs(event.start_time - frame.start_time) > 0.51 or abs(event.end_time - frame.end_time) > 0.51:
                repair_types.append("repair_note_time")
            repaired = ModelEvent(
                kind="note",
                start_time=frame.start_time,
                end_time=frame.end_time,
                frame_index=event.frame_index,
            )
            repaired_events.append(repaired)
            notes.append(
                NoteRecord(
                    start_time=frame.start_time,
                    end_time=frame.end_time,
                    image_path=frame.image_path,
                    global_frame_index=frame.global_index,
                )
            )
            continue

        if event.kind != "bridge":
            continue
        text = event.text.strip()
        if not text:
            repair_types.append("drop_empty_bridge")
            continue
        is_first_bridge = not seen_bridge
        seen_bridge = True
        start = event.start_time
        end = event.end_time
        if is_first_bridge and context.open_tail_bridge is not None:
            if abs(start - context.open_tail_bridge.start_time) > 0.51:
                repair_types.append("repair_open_tail_start")
            start = context.open_tail_bridge.start_time
            if end > context.step_end:
                repair_types.append("clamp_open_tail_end")
                end = context.step_end
            if end < context.step_start:
                repair_types.append("extend_open_tail_end")
                end = context.step_end
            replace_open_tail = True
        else:
            new_start = max(start, context.step_start)
            new_end = min(end, context.step_end)
            if new_start != start or new_end != end:
                repair_types.append("clamp_bridge_interval")
            start, end = new_start, new_end
        if start >= end:
            repair_types.append("drop_invalid_bridge_interval")
            continue
        repaired = ModelEvent(kind="bridge", start_time=start, end_time=end, text=text)
        repaired_events.append(repaired)
        bridges.append(BridgeRecord(start_time=start, end_time=end, text=text))

    answer_text = action.answer.strip()
    answer = QARecord(timestamp=context.step_end, text=answer_text, role="a") if answer_text else None
    repaired_action = ModelAction(
        eta=action.eta,
        answer=answer_text,
        events=sorted(repaired_events, key=lambda event: (event.start_time, event.end_time, event.kind != "bridge")),
        raw=raw,
        eta_present=action.eta_present,
        answer_present=action.answer_present,
    )
    return AppliedAction(
        action=repaired_action,
        notes=notes,
        bridges=bridges,
        answer=answer,
        replace_open_tail=replace_open_tail,
        repair_count=len(repair_types),
        repair_types=repair_types,
    )


def apply_raw_action(action: ModelAction, context: QualityContext) -> AppliedAction:
    notes: list[NoteRecord] = []
    bridges: list[BridgeRecord] = []
    replace_open_tail = False
    for event_index, event in enumerate(action.events):
        if event.kind == "note" and event.frame_index is not None and 0 <= event.frame_index < len(context.frames):
            frame = context.frames[event.frame_index]
            notes.append(
                NoteRecord(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    image_path=frame.image_path,
                    global_frame_index=frame.global_index,
                )
            )
        elif event.kind == "bridge":
            if (
                event_index == 0
                and context.open_tail_bridge is not None
                and abs(event.start_time - context.open_tail_bridge.start_time) <= 0.51
            ):
                replace_open_tail = True
            bridges.append(BridgeRecord(start_time=event.start_time, end_time=event.end_time, text=event.text.strip()))
    answer_text = action.answer.strip()
    answer = QARecord(timestamp=context.step_end, text=answer_text, role="a") if answer_text else None
    return AppliedAction(action=action, notes=notes, bridges=bridges, answer=answer, replace_open_tail=replace_open_tail)


def synthesis_feedback(
    issues: list[object],
    raw_output: str,
    *,
    step_start: float | None = None,
    step_end: float | None = None,
    memory_tail_kind: str | None = None,
    memory_tail_end: float | None = None,
    open_tail_start: float | None = None,
) -> str:
    """Build a focused retry message.

    The previous raw output is intentionally NOT echoed back verbatim. Returning the model's own
    bad output anchors it into repeating the same mistake on the next attempt.
    """
    codes = [str(getattr(issue, "code", "")) for issue in issues]
    messages = [str(getattr(issue, "message", "")) for issue in issues]

    fixes: list[str] = []
    fixes.extend(
        _bridge_fixes(
            codes=codes,
            messages=messages,
            step_start=step_start,
            step_end=step_end,
            memory_tail_kind=memory_tail_kind,
            memory_tail_end=memory_tail_end,
            open_tail_start=open_tail_start,
        )
    )
    fixes.extend(_zero_duration_bridge_fixes(codes, messages, raw_output))
    fixes.extend(_unclosed_note_fixes(codes, raw_output))

    other_errors = [
        f"- {code}: {message}"
        for code, message in zip(codes, messages)
        if code
        not in {
            "bridge_time_oob",
            "event_overlap",
            "missing_bridge_gap",
            "invalid_bridge_gap",
            "duplicate_bridge_gap",
            "open_tail_start_mismatch",
            "missing_open_tail_bridge",
            "open_tail_end_oob",
            "tag_parse_error",
            "text_outside_tags",
        }
    ]

    sections = ["Previous attempt failed validation. Apply the corrections below and produce a fresh full XML answer."]
    if fixes:
        sections.append("Corrections:")
        sections.extend(fixes)
    if other_errors:
        sections.append("Other reported errors:")
        sections.extend(other_errors)
    sections.append(
        "Output format: <eta>...</eta>, <answer>...</answer>, then bridge/note tags in chronological order, "
        "no overlap and no text outside tags. Notes use paired <note t=\"...\" frame=\"N\"></note>."
    )
    return "\n".join(sections)


_BRIDGE_FIX_CODES = {
    "bridge_time_oob",
    "event_overlap",
    "missing_bridge_gap",
    "invalid_bridge_gap",
    "duplicate_bridge_gap",
    "open_tail_start_mismatch",
    "missing_open_tail_bridge",
    "open_tail_end_oob",
}


def _bridge_fixes(
    *,
    codes: list[str],
    messages: list[str],
    step_start: float | None,
    step_end: float | None,
    memory_tail_kind: str | None,
    memory_tail_end: float | None,
    open_tail_start: float | None,
) -> list[str]:
    code_set = set(codes)
    if not code_set & _BRIDGE_FIX_CODES:
        return []

    required_gaps: list[str] = []
    invalid_gaps: list[str] = []
    duplicate_gaps: list[str] = []
    for message in messages:
        # Anchor on full prefixes so duplicate_bridge_gap ("Multiple bridges cover required gap X-Y")
        # is not mistaken for a missing gap.
        required_gaps.extend(re.findall(r"Missing bridge for required gap ([0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?)", message))
        invalid_gaps.extend(re.findall(r"Bridge interval ([0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?)", message))
        duplicate_gaps.extend(re.findall(r"Multiple bridges cover required gap ([0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?)", message))
    required_gaps = _unique_keep_order(required_gaps)
    invalid_gaps = _unique_keep_order(invalid_gaps)
    duplicate_gaps = _unique_keep_order(duplicate_gaps)

    lines: list[str] = []
    paired = _pair_invalid_to_required(invalid_gaps, required_gaps)
    used_required: set[str] = set()
    used_invalid: set[str] = set()
    for invalid, required in paired:
        if not _is_safe_replacement(invalid, required):
            # Likely two unrelated bridges that happen to share counts; let the
            # "Add required" + "Drop invalid" branches below handle them so we
            # don't tell the model to copy text from a different gap.
            continue
        lines.append(f'- Bridge `t="{invalid}"` should be `t="{required}"`. Replace it.')
        used_invalid.add(invalid)
        used_required.add(required)

    for required in required_gaps:
        if required in used_required:
            continue
        lines.append(f'- Add a `<bridge t="{required}">...</bridge>` for the missing gap.')

    for invalid in invalid_gaps:
        if invalid in used_invalid:
            continue
        lines.append(f'- Bridge `t="{invalid}"` does not match any required gap; drop it.')

    for duplicate in duplicate_gaps:
        lines.append(f'- Multiple bridges cover gap `t="{duplicate}"`; keep only one.')

    if memory_tail_kind == "note" and memory_tail_end is not None:
        lines.append(
            f"- Memory ends with a <note ending at {_fmt_time(memory_tail_end)}>, so do NOT inherit any bridge. "
            f"Your first new bridge cannot start before {_fmt_time(memory_tail_end)}."
        )
    elif memory_tail_kind == "bridge" and ("missing_open_tail_bridge" in code_set or "open_tail_start_mismatch" in code_set):
        anchor = _fmt_time(open_tail_start) if open_tail_start is not None else "the original Memory bridge start"
        lines.append(
            f"- Memory ends with an open-tail <bridge>, so your first event must be a <bridge> starting at t={anchor} "
            "(rewrite/extend that bridge; do not start a new one)."
        )

    if step_start is not None and step_end is not None and not lines:
        # No specific gaps to point at, but bridge timing was wrong — give the structural anchor.
        lines.append(
            f"- This step covers {_fmt_time(step_start)}-{_fmt_time(step_end)}. "
            "Bridge `t` is structural (must align with note/window boundaries), not a description of when the action started."
        )
    return lines


def _zero_duration_bridge_fixes(codes: list[str], messages: list[str], raw_output: str) -> list[str]:
    if "tag_parse_error" not in codes:
        return []
    intervals: list[str] = []
    for message in messages:
        intervals.extend(re.findall(r"Invalid non-positive interval: '([0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?)'", message))
    if not intervals:
        return []
    unique = _unique_keep_order(intervals)
    label = ", ".join(f'`t="{value}"`' for value in unique)
    return [
        f"- Drop the zero-duration bridge {label}. "
        "A step does not need to end with a bridge: if the last frame in this step is anchored as a <note>, stop after that note.",
    ]


def _unclosed_note_fixes(codes: list[str], raw_output: str) -> list[str]:
    if "text_outside_tags" not in codes and "tag_parse_error" not in codes:
        return []
    open_tags = re.findall(r'<note\b[^>]*frame="(\d+)"[^/>]*>', raw_output)
    closed_count = len(re.findall(r"</note>", raw_output))
    if not open_tags or len(open_tags) <= closed_count:
        return []
    unclosed_frames = open_tags[closed_count:]
    label = ", ".join(f"frame={frame}" for frame in _unique_keep_order(unclosed_frames))
    return [
        f"- Note tag for {label} was opened but never closed. "
        'Always pair: `<note t="..." frame="N"></note>`.',
    ]


def _pair_invalid_to_required(invalid: list[str], required: list[str]) -> list[tuple[str, str]]:
    """Pair a wrong interval with the closest required interval, when sizes match.

    Most failures emit (wrong-start, correct-end) - e.g. "9.0-11.0" vs required "10.0-11.0".
    Pair greedily by matching end-times, then by overall closeness.
    """
    if not invalid or not required or len(invalid) != len(required):
        return []
    pairs: list[tuple[str, str]] = []
    remaining = list(required)
    for inv in invalid:
        best_index = -1
        best_score = float("inf")
        for index, req in enumerate(remaining):
            score = _interval_distance(inv, req)
            if score < best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            return []
        pairs.append((inv, remaining.pop(best_index)))
    return pairs


def _interval_distance(a: str, b: str) -> float:
    parts_a = _parse_pair(a)
    parts_b = _parse_pair(b)
    if parts_a is None or parts_b is None:
        return float("inf")
    return abs(parts_a[0] - parts_b[0]) + abs(parts_a[1] - parts_b[1])


def _is_safe_replacement(invalid: str, required: str, *, endpoint_tol: float = 0.51, max_drift: float = 1.0) -> bool:
    """Whether telling the model to Replace `invalid` with `required` is safe.

    A pairing is safe when (a) at least one endpoint already matches — typical of
    "model extended/shrunk one side" edits — or (b) both endpoints drift by no more
    than `max_drift`. Outside these regimes the two intervals likely refer to
    different bridges, and a Replace instruction would tell the model to keep the
    text from the wrong gap.
    """
    a = _parse_pair(invalid)
    b = _parse_pair(required)
    if a is None or b is None:
        return False
    start_match = abs(a[0] - b[0]) <= endpoint_tol
    end_match = abs(a[1] - b[1]) <= endpoint_tol
    if start_match or end_match:
        return True
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= max_drift


def _parse_pair(text: str) -> tuple[float, float] | None:
    match = re.match(r"\s*([0-9.]+)-([0-9.]+)\s*$", text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _fmt_time(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"{value:.1f}"
    return f"{value:g}"


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


NOTE_TAG_RE = re.compile(r"<note\b(?P<attrs>[^>]*)>(?P<body>.*?)</note>|<note\b(?P<self_attrs>[^>]*)/>", re.DOTALL)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def _extract_malformed_notes(raw: str, context: QualityContext) -> list[ModelEvent]:
    events: list[ModelEvent] = []
    for match in NOTE_TAG_RE.finditer(raw):
        attrs_text = match.group("attrs") if match.group("attrs") is not None else match.group("self_attrs")
        attrs = {key: value for key, value in ATTR_RE.findall(attrs_text or "")}
        frame_text = attrs.get("frame") or attrs.get("id")
        if not frame_text or not frame_text.isdigit():
            continue
        frame_index = int(frame_text) - 1
        if not (0 <= frame_index < len(context.frames)):
            continue
        if attrs.get("t"):
            interval = _parse_interval(attrs["t"])
            if interval is None:
                continue
            start, end = interval
        else:
            frame = context.frames[frame_index]
            start, end = frame.start_time, frame.end_time
        events.append(ModelEvent(kind="note", start_time=start, end_time=end, frame_index=frame_index))
    return events


def _parse_interval(text: str) -> tuple[float, float] | None:
    match = re.match(r"\s*([0-9.]+)\s*[-\u2013]\s*([0-9.]+)\s*$", text)
    if not match:
        return None
    start = float(match.group(1))
    end = float(match.group(2))
    if start >= end:
        return None
    return start, end


def _dedupe_events(existing: list[ModelEvent], candidates: list[ModelEvent]) -> list[ModelEvent]:
    seen = {
        (event.kind, event.frame_index, round(event.start_time, 3), round(event.end_time, 3), event.text)
        for event in existing
    }
    out: list[ModelEvent] = []
    for event in candidates:
        key = (event.kind, event.frame_index, round(event.start_time, 3), round(event.end_time, 3), event.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out
