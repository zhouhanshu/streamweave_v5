"""Real-time trace writer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schemas import AppliedAction, BackendResult, FrameRef, QualityReport, Transition


class TraceWriter:
    def __init__(self, root: str | Path, *, write_jsonl: bool = True) -> None:
        self.root = Path(root)
        self.write_jsonl = write_jsonl
        self.root.mkdir(parents=True, exist_ok=True)
        self.trace_txt = self.root / "trace.txt"
        self.trace_jsonl = self.root / "trace.jsonl"
        self.memory_txt = self.root / "memory.txt"
        for path in (self.trace_txt, self.trace_jsonl, self.memory_txt):
            if path.exists():
                path.unlink()

    def write_step_start(
        self,
        *,
        step_index: int,
        prompt_text: str,
        prompt_images: list[str],
        memory_before: str,
        current_frames: list[FrameRef],
    ) -> None:
        with self.trace_txt.open("a", encoding="utf-8") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"Step {step_index} Input\n")
            f.write(f"{'=' * 60}\n")
            f.write("Current frame paths:\n")
            for frame in current_frames:
                f.write(f"- local={frame.step_local_id} global={frame.global_index} t={frame.start_time:.1f}-{frame.end_time:.1f} path={frame.image_path}\n")
            f.write("Prompt images:\n")
            for image in prompt_images:
                f.write(f"- {image}\n")
            f.write("Memory before:\n")
            f.write(memory_before + "\n")
            f.write("Prompt:\n")
            f.write(prompt_text + "\n")
            f.flush()

    def write_backend_done(self, *, step_index: int, backend_result: BackendResult) -> None:
        with self.trace_txt.open("a", encoding="utf-8") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"Step {step_index} Raw Output\n")
            f.write(f"{'=' * 60}\n")
            f.write(
                f"backend endpoint={backend_result.endpoint_id} attempts={backend_result.attempt_count} "
                f"latency={backend_result.latency_seconds:.3f}s\n"
            )
            if backend_result.retry_errors:
                f.write("backend retry errors:\n")
                for error in backend_result.retry_errors:
                    f.write(f"- {error}\n")
            f.write(backend_result.text + "\n")
            f.flush()

    def write_env_done(
        self,
        *,
        step_index: int,
        quality: QualityReport,
        applied: AppliedAction,
        memory_after: str,
    ) -> None:
        with self.trace_txt.open("a", encoding="utf-8") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"Step {step_index} Quality / Applied Action\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"valid={quality.valid} parser_ok={quality.parser_ok}\n")
            f.write(f"rewards={asdict(quality.rewards)}\n")
            f.write(f"metrics={quality.metrics}\n")
            if quality.issues:
                f.write("issues:\n")
                for issue in quality.issues:
                    f.write(f"- [{issue.severity}] {issue.code}: {issue.message}\n")
            f.write(f"repair_count={applied.repair_count} repair_types={applied.repair_types}\n")
            f.write("applied output:\n")
            f.write(_action_to_text(applied) + "\n")
            f.write("Memory after:\n")
            f.write(memory_after + "\n\n")
            f.flush()
        self.memory_txt.write_text(memory_after + "\n", encoding="utf-8")

    def write_attempt_failed(
        self,
        *,
        step_index: int,
        attempt_index: int,
        quality: QualityReport,
        feedback: str,
        memory_after: str,
    ) -> None:
        with self.trace_txt.open("a", encoding="utf-8") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"Step {step_index} Attempt {attempt_index} Rejected\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"valid={quality.valid} parser_ok={quality.parser_ok}\n")
            f.write(f"rewards={asdict(quality.rewards)}\n")
            f.write(f"metrics={quality.metrics}\n")
            if quality.issues:
                f.write("issues:\n")
                for issue in quality.issues:
                    f.write(f"- [{issue.severity}] {issue.code}: {issue.message}\n")
            f.write("retry feedback:\n")
            f.write(feedback + "\n")
            f.write("Memory after rejected attempt:\n")
            f.write(memory_after + "\n\n")
            f.flush()
        self.memory_txt.write_text(memory_after + "\n", encoding="utf-8")

    def write_transition(self, transition: Transition) -> None:
        if not self.write_jsonl:
            return
        with self.trace_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(transition), ensure_ascii=False, default=_json_default) + "\n")
            f.flush()


def _action_to_text(applied: AppliedAction) -> str:
    action = applied.action
    lines = [
        f"<eta>{'' if action.eta is None else action.eta}</eta>",
        f"<answer>{action.answer}</answer>",
    ]
    for event in action.events:
        if event.kind == "note":
            frame = "" if event.frame_index is None else event.frame_index + 1
            lines.append(f'<note t="{event.start_time:.1f}-{event.end_time:.1f}" frame="{frame}"></note>')
        elif event.kind == "bridge":
            lines.append(f'<bridge t="{event.start_time:.1f}-{event.end_time:.1f}">{event.text}</bridge>')
    return "\n".join(lines)


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return f"<{type(value).__name__}>"
