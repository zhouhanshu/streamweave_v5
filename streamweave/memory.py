"""Memory state for anchors, deltas, and QA history."""

from __future__ import annotations

from .policies import MemoryPolicy
from .schemas import BridgeRecord, ContentItem, NoteRecord, QARecord


class MemoryStore:
    def __init__(self, *, memory_window: float = 120.0) -> None:
        self.memory_window = memory_window
        self.notes: list[NoteRecord] = []
        self.bridges: list[BridgeRecord] = []
        self.qa_history: list[QARecord] = []

    def add_note(self, note: NoteRecord) -> None:
        self.notes.append(note)

    def add_bridge(self, bridge: BridgeRecord) -> None:
        self.bridges.append(bridge)

    def replace_bridge(self, old: BridgeRecord, new: BridgeRecord) -> None:
        for idx, bridge in enumerate(self.bridges):
            if bridge is old:
                self.bridges[idx] = new
                return
        self.bridges.append(new)

    def add_qa(self, qa: QARecord) -> None:
        self.qa_history.append(qa)

    def has_question(self) -> bool:
        return any(qa.role == "q" for qa in self.qa_history)

    def evict(self, current_time: float) -> None:
        cutoff = current_time - self.memory_window
        for note in self.notes:
            if note.end_time < cutoff:
                note.image_available = False

    def open_tail_bridge(self, *, include_notes: bool = True, include_bridges: bool = True) -> BridgeRecord | None:
        events: list[tuple[float, int, BridgeRecord | NoteRecord]] = []
        if include_bridges:
            for bridge in self.bridges:
                events.append((bridge.end_time, 0, bridge))
        if include_notes:
            for note in self.notes:
                events.append((note.end_time, 1, note))
        if not events:
            return None
        _, _, latest = max(events, key=lambda item: (item[0], item[1]))
        return latest if isinstance(latest, BridgeRecord) else None

    def build_memory_content(self, policy: MemoryPolicy) -> list[ContentItem]:
        if not policy.read_memory:
            return []
        events: list[tuple[float, int, str, object]] = []
        if policy.read_bridges:
            events.extend((bridge.start_time, 0, "bridge", bridge) for bridge in self.bridges)
        if policy.read_notes:
            events.extend((note.start_time, 1, "note", note) for note in self.notes)
        events.sort(key=lambda item: (item[0], item[1]))

        content: list[ContentItem] = []
        for _, _, kind, item in events:
            if kind == "bridge":
                bridge = item
                assert isinstance(bridge, BridgeRecord)
                content.append(
                    ContentItem(
                        "text",
                        text=f'<delta t="{bridge.start_time:.1f}-{bridge.end_time:.1f}">{bridge.text}</delta>',
                    )
                )
            else:
                note = item
                assert isinstance(note, NoteRecord)
                if note.image_available:
                    content.append(ContentItem("text", text=f'<anchor t="{note.start_time:.1f}-{note.end_time:.1f}">'))
                    content.append(ContentItem("image", image_path=note.image_path))
                    content.append(ContentItem("text", text="</anchor>"))
                else:
                    content.append(ContentItem("text", text=f'<anchor t="{note.start_time:.1f}-{note.end_time:.1f}"></anchor>'))
        return content

    def build_qa_text(self) -> str:
        if not self.qa_history:
            return "<empty/>"
        lines = []
        for qa in sorted(self.qa_history, key=lambda item: item.timestamp):
            lines.append(f'<qa t="{qa.timestamp:.1f}" role="{qa.role}">{qa.text}</qa>')
        return "\n".join(lines)

    def dump_text(self) -> str:
        lines = ["Memory:"]
        events: list[tuple[float, int, str]] = []
        for bridge in self.bridges:
            events.append((bridge.start_time, 0, f'delta {bridge.start_time:.1f}-{bridge.end_time:.1f}: {bridge.text}'))
        for note in self.notes:
            status = str(note.image_path) if note.image_available else f"{note.image_path} (image_evicted)"
            events.append((note.start_time, 1, f"anchor {note.start_time:.1f}-{note.end_time:.1f}: {status}"))
        for qa in self.qa_history:
            events.append((qa.timestamp, 2, f"qa {qa.timestamp:.1f} {qa.role}: {qa.text}"))
        for _, _, text in sorted(events, key=lambda item: (item[0], item[1])):
            lines.append(text)
        return "\n".join(lines)
