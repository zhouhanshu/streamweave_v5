"""Memory/observation policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MemoryPolicy:
    name: str
    read_notes: bool = True
    read_bridges: bool = True
    read_memory: bool = True
    commit_notes: bool = True
    commit_bridges: bool = True
    use_open_tail: bool = True
    use_recent_frames: bool = False
    recent_frame_count: int = 5


def make_policy(name: str) -> MemoryPolicy:
    key = name.lower()
    if key in {"streamweave", "full", "note_bridge"}:
        return MemoryPolicy(name="streamweave", read_notes=True, read_bridges=True)
    if key in {"keyframe_only", "note_only"}:
        return MemoryPolicy(
            name="keyframe_only",
            read_notes=True,
            read_bridges=False,
            commit_notes=True,
            commit_bridges=True,
            use_open_tail=False,
        )
    if key == "bridge_only":
        return MemoryPolicy(
            name="bridge_only",
            read_notes=False,
            read_bridges=True,
            commit_notes=True,
            commit_bridges=True,
            use_open_tail=True,
        )
    if key in {"recent_frames", "recent5", "recent_5"}:
        return MemoryPolicy(
            name="recent_frames",
            read_notes=False,
            read_bridges=False,
            read_memory=False,
            commit_notes=True,
            commit_bridges=True,
            use_open_tail=False,
            use_recent_frames=True,
            recent_frame_count=5,
        )
    if key in {"no_memory", "none"}:
        return MemoryPolicy(
            name="no_memory",
            read_notes=False,
            read_bridges=False,
            read_memory=False,
            commit_notes=True,
            commit_bridges=True,
            use_open_tail=False,
        )
    raise ValueError(f"Unknown policy: {name}")
