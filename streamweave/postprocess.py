"""Deterministic postprocessing for eval/rollout."""

from __future__ import annotations

import re

from .parser import parse_for_repair
from .quality import QualityContext
from .schemas import AppliedAction, BridgeRecord, FrameRef, ModelAction, ModelEvent, NoteRecord, QARecord


NOTE_TIME_TOLERANCE = 0.11
BRIDGE_GAP_TOLERANCE = 1e-6
BRIDGE_INTERVAL_TOLERANCE = 0.11
FALLBACK_BRIDGE_TEXT = "The video continues through this interval, but no reliable bridge description was recovered."


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

    for event_index, event in enumerate(events):
        if event.kind == "note":
            frame, match_issue = _match_note_frame(event, context.frames)
            if frame is None:
                repair_types.append(f"drop_note_time_{match_issue}")
                continue
            repaired = ModelEvent(
                kind="note",
                start_time=frame.start_time,
                end_time=frame.end_time,
            )
            repaired_events.append(repaired)
            continue

        if event.kind != "bridge":
            continue
        text = event.text.strip()
        if not text:
            repair_types.append("drop_empty_bridge")
            continue
        is_first_event_bridge = event_index == 0
        start = event.start_time
        end = event.end_time
        if context.open_tail_bridge is not None and abs(start - context.open_tail_bridge.start_time) <= 0.51:
            start = context.open_tail_bridge.start_time
            if end > context.step_end:
                repair_types.append("clamp_open_tail_end")
                end = context.step_end
            if end < context.step_start:
                repair_types.append("extend_open_tail_end")
                end = context.step_end
            replace_open_tail = True
        else:
            if is_first_event_bridge and context.open_tail_bridge is not None:
                repair_types.append("skip_open_tail_start_repair")
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

    repaired_events, notes, bridges, normalized_replace_open_tail = _normalize_observation_events(
        repaired_events,
        context,
        repair_types,
    )
    replace_open_tail = replace_open_tail or normalized_replace_open_tail

    answer_text = action.answer.strip()
    answer = QARecord(timestamp=context.step_end, text=answer_text, role="a") if answer_text else None
    state_text = action.state.strip()
    state_present = action.state_present
    if not state_text:
        state_text = "No usable state was recovered from the raw output."
        state_present = True
        repair_types.append("fill_empty_state")
    repaired_action = ModelAction(
        state=state_text,
        answer=answer_text,
        events=repaired_events,
        raw=raw,
        state_present=state_present,
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
        if event.kind == "note":
            frame, _match_issue = _match_note_frame(event, context.frames)
            if frame is None:
                continue
            notes.append(
                NoteRecord(
                    start_time=frame.start_time,
                    end_time=frame.end_time,
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


def _match_note_frame(event: ModelEvent, frames: list[FrameRef]) -> tuple[FrameRef | None, str]:
    matches = [
        frame
        for frame in frames
        if abs(event.start_time - frame.start_time) <= NOTE_TIME_TOLERANCE
        and abs(event.end_time - frame.end_time) <= NOTE_TIME_TOLERANCE
    ]
    if len(matches) == 1:
        return matches[0], ""
    if not matches:
        return None, "unmatched"
    return None, "ambiguous"


def _normalize_observation_events(
    events: list[ModelEvent],
    context: QualityContext,
    repair_types: list[str],
) -> tuple[list[ModelEvent], list[NoteRecord], list[BridgeRecord], bool]:
    notes = _dedupe_note_events(events, repair_types)
    source_bridges = [event for event in events if event.kind == "bridge" and event.text.strip()]
    expected_gaps = _expected_bridge_gaps(notes, context)
    normalized_bridges: list[ModelEvent] = []
    used_bridge_indexes: set[int] = set()

    for gap_start, gap_end in expected_gaps:
        match_index = _find_matching_bridge(source_bridges, gap_start, gap_end, used_bridge_indexes)
        text = ""
        if match_index is not None:
            used_bridge_indexes.add(match_index)
            source = source_bridges[match_index]
            text = source.text.strip()
            if not _same_interval(source.start_time, source.end_time, gap_start, gap_end):
                repair_types.append("adjust_bridge_gap")
            if (
                context.open_tail_bridge is not None
                and abs(gap_start - context.open_tail_bridge.start_time) <= BRIDGE_INTERVAL_TOLERANCE
                and source.start_time > context.open_tail_bridge.start_time + BRIDGE_INTERVAL_TOLERANCE
            ):
                text = _merge_bridge_text(context.open_tail_bridge.text, text)
                repair_types.append("merge_open_tail_bridge_text")
        elif (
            context.open_tail_bridge is not None
            and abs(gap_start - context.open_tail_bridge.start_time) <= BRIDGE_INTERVAL_TOLERANCE
            and context.open_tail_bridge.text.strip()
        ):
            text = context.open_tail_bridge.text.strip()
            repair_types.append("add_missing_bridge_gap")
        else:
            text = FALLBACK_BRIDGE_TEXT
            repair_types.append("add_missing_bridge_gap")
        normalized_bridges.append(ModelEvent(kind="bridge", start_time=gap_start, end_time=gap_end, text=text))

    if len(used_bridge_indexes) < len(source_bridges):
        repair_types.append("drop_invalid_bridge_gap")

    normalized_events = sorted(
        [*notes, *normalized_bridges],
        key=lambda event: (event.start_time, event.end_time, event.kind != "bridge"),
    )
    note_records = _note_records_from_events(notes, context)
    bridge_records = [
        BridgeRecord(start_time=event.start_time, end_time=event.end_time, text=event.text.strip())
        for event in normalized_bridges
    ]
    replace_open_tail = bool(
        normalized_bridges
        and context.open_tail_bridge is not None
        and abs(normalized_bridges[0].start_time - context.open_tail_bridge.start_time) <= BRIDGE_INTERVAL_TOLERANCE
    )
    return normalized_events, note_records, bridge_records, replace_open_tail


def _dedupe_note_events(events: list[ModelEvent], repair_types: list[str]) -> list[ModelEvent]:
    notes: list[ModelEvent] = []
    seen: set[tuple[float, float]] = set()
    for event in sorted(
        (event for event in events if event.kind == "note"),
        key=lambda item: (item.start_time, item.end_time),
    ):
        key = (round(event.start_time, 3), round(event.end_time, 3))
        if key in seen:
            repair_types.append("drop_duplicate_note")
            continue
        seen.add(key)
        notes.append(event)
    return notes


def _expected_bridge_gaps(notes: list[ModelEvent], context: QualityContext) -> list[tuple[float, float]]:
    if not notes:
        start = context.open_tail_bridge.start_time if context.open_tail_bridge is not None else context.step_start
        return [(start, context.step_end)] if context.step_end > start + BRIDGE_GAP_TOLERANCE else []

    gaps: list[tuple[float, float]] = []
    first_note = notes[0]
    if context.open_tail_bridge is not None:
        if first_note.start_time > context.open_tail_bridge.start_time + BRIDGE_GAP_TOLERANCE:
            gaps.append((context.open_tail_bridge.start_time, first_note.start_time))
    elif first_note.start_time > context.step_start + BRIDGE_GAP_TOLERANCE:
        gaps.append((context.step_start, first_note.start_time))

    previous_end = first_note.end_time
    for note in notes[1:]:
        if note.start_time > previous_end + BRIDGE_GAP_TOLERANCE:
            gaps.append((previous_end, note.start_time))
        previous_end = note.end_time

    if context.step_end > previous_end + BRIDGE_GAP_TOLERANCE:
        gaps.append((previous_end, context.step_end))
    return gaps


def _find_matching_bridge(
    bridges: list[ModelEvent],
    gap_start: float,
    gap_end: float,
    used_indexes: set[int],
) -> int | None:
    for index, bridge in enumerate(bridges):
        if index in used_indexes:
            continue
        if _same_interval(bridge.start_time, bridge.end_time, gap_start, gap_end):
            return index

    best_index: int | None = None
    best_score: tuple[float, float] | None = None
    for index, bridge in enumerate(bridges):
        if index in used_indexes:
            continue
        overlap = min(bridge.end_time, gap_end) - max(bridge.start_time, gap_start)
        start_close = abs(bridge.start_time - gap_start) <= 0.51
        end_close = abs(bridge.end_time - gap_end) <= 0.51
        if overlap <= 0 and not (start_close or end_close):
            continue
        distance = abs(bridge.start_time - gap_start) + abs(bridge.end_time - gap_end)
        score = (-max(overlap, 0.0), distance)
        if best_score is None or score < best_score:
            best_score = score
            best_index = index
    return best_index


def _same_interval(start: float, end: float, expected_start: float, expected_end: float) -> bool:
    return (
        abs(start - expected_start) <= BRIDGE_INTERVAL_TOLERANCE
        and abs(end - expected_end) <= BRIDGE_INTERVAL_TOLERANCE
    )


def _merge_bridge_text(old_text: str, new_text: str) -> str:
    old = old_text.strip()
    new = new_text.strip()
    if not old:
        return new or FALLBACK_BRIDGE_TEXT
    if not new or new == old:
        return old
    return f"{old} {new}"


def _note_records_from_events(notes: list[ModelEvent], context: QualityContext) -> list[NoteRecord]:
    records: list[NoteRecord] = []
    for note in notes:
        frame, _match_issue = _match_note_frame(note, context.frames)
        if frame is None:
            continue
        records.append(
            NoteRecord(
                start_time=frame.start_time,
                end_time=frame.end_time,
                image_path=frame.image_path,
                global_frame_index=frame.global_index,
            )
        )
    return records


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
        "Output format: <state>...</state>, <answer>...</answer>, then bridge/note tags in chronological order, "
        "no overlap and no text outside tags. Notes use paired <note t=\"...\"></note>."
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
    open_tags = re.findall(r'<note\b[^>/]*\bt="([^"]+)"[^>/]*>', raw_output)
    closed_count = len(re.findall(r"</note>", raw_output))
    if not open_tags or len(open_tags) <= closed_count:
        return []
    unclosed_times = open_tags[closed_count:]
    label = ", ".join(f't="{time_range}"' for time_range in _unique_keep_order(unclosed_times))
    return [
        f"- Note tag for {label} was opened but never closed. "
        'Always pair: `<note t="..."></note>`.',
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


NOTE_TAG_RE = re.compile(r"<note\b(?P<attrs>[^>]*)>(?P<body>.*?)</note>", re.DOTALL)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def _extract_malformed_notes(raw: str, context: QualityContext) -> list[ModelEvent]:
    events: list[ModelEvent] = []
    for match in NOTE_TAG_RE.finditer(raw):
        attrs_text = match.group("attrs")
        attrs = {key: value for key, value in ATTR_RE.findall(attrs_text or "")}
        if "frame" in attrs or "id" in attrs:
            continue
        time_text = attrs.get("t")
        if not time_text:
            continue
        interval = _parse_interval(time_text)
        if interval is None:
            continue
        start, end = interval
        events.append(ModelEvent(kind="note", start_time=start, end_time=end))
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
    seen = {(event.kind, round(event.start_time, 3), round(event.end_time, 3), event.text) for event in existing}
    out: list[ModelEvent] = []
    for event in candidates:
        key = (event.kind, round(event.start_time, 3), round(event.end_time, 3), event.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out
