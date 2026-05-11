#!/usr/bin/env python3
"""Export StreamWeave SFT intermediate rows to LLaMAFactory ShareGPT JSONL."""

from __future__ import annotations

import argparse
from copy import deepcopy
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.sft.constraints import answer_matches_metadata
from data_engine.sft.io_utils import JsonDict, read_jsonl, write_json, write_jsonl
from streamweave.prompts import PromptContext, build_prompt
from streamweave.schemas import ContentItem, FrameRef


def export_sharegpt(
    rows: Iterable[JsonDict],
    output_path: str | Path,
    *,
    dataset_name: str = "streamweave_sft",
    train_prompt_type: str = "production",
) -> dict[str, int | str]:
    expanded_rows = list(expand_training_rows(rows))
    out_rows = [to_sharegpt_row(row, train_prompt_type=train_prompt_type) for row in expanded_rows]
    count = write_jsonl(out_rows, output_path)
    return {
        "dataset_name": dataset_name,
        "count": count,
        "output_path": str(output_path),
        "train_prompt_type": train_prompt_type,
    }


def expand_training_rows(rows: Iterable[JsonDict]) -> Iterable[JsonDict]:
    for row in rows:
        yield row
        yield from _variant_rows(row)


def to_sharegpt_row(row: JsonDict, *, train_prompt_type: str = "production") -> JsonDict:
    content, images = _select_training_prompt(row, train_prompt_type=train_prompt_type)
    target_xml = str(row.get("target_xml") or "").strip()
    if not content:
        raise ValueError(f"Row {row.get('sample_id')} has an empty prompt.")
    if not target_xml:
        raise ValueError(f"Row {row.get('sample_id')} has an empty target_xml.")
    _validate_image_placeholders(row, content, images)
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": target_xml},
        ],
        "images": images,
    }


def _variant_rows(row: JsonDict) -> Iterable[JsonDict]:
    variants = row.get("answer_variants") or []
    if not isinstance(variants, list):
        return
    for variant in variants:
        if not isinstance(variant, dict) or not _variant_is_exportable(row, variant):
            continue
        out = deepcopy(row)
        out["sample_id"] = f"{row.get('sample_id')}_answer_variant_{int(variant.get('variant_index', 0)):04d}"
        out["target_xml"] = str(variant.get("target_xml") or variant.get("raw_teacher_xml") or "").strip()
        out["raw_teacher_xml"] = out["target_xml"]
        out["target_raw_output"] = out["target_xml"]
        out["quality"] = {"raw": variant.get("quality", {}), "target": variant.get("quality", {})}
        out["metadata"] = dict(out.get("metadata") or {})
        out["metadata"]["answer_variant"] = variant
        yield out


def _variant_is_exportable(row: JsonDict, variant: JsonDict) -> bool:
    if not variant.get("accepted"):
        return False
    target_xml = str(variant.get("target_xml") or variant.get("raw_teacher_xml") or "").strip()
    answer = str(variant.get("answer") or "").strip()
    if not target_xml or not answer:
        return False
    metadata = row.get("metadata") or {}
    annotation = metadata.get("annotation") if isinstance(metadata, dict) else {}
    if not isinstance(annotation, dict):
        annotation = {}
    match = answer_matches_metadata(annotation, str(row.get("question_type") or ""), answer)
    return True if match is None else bool(match)


def dataset_info(dataset_name: str, file_name: str) -> JsonDict:
    return make_dataset_info(dataset_name, file_name)


def make_dataset_info(dataset_name: str, file_name: str) -> JsonDict:
    return {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }


def _select_training_prompt(row: JsonDict, *, train_prompt_type: str) -> tuple[str, list[str]]:
    if train_prompt_type == "recorded":
        return _select_recorded_prompt(row)
    content = build_prompt(train_prompt_type, _prompt_context_from_row(row))
    return _render_content_for_sharegpt(content)


def _select_recorded_prompt(row: JsonDict) -> tuple[str, list[str]]:
    prompt = row.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError(f"Row {row.get('sample_id')} is missing prompt metadata.")
    content = str(prompt.get("base_content") or prompt.get("content") or "")
    images = [str(path) for path in (prompt.get("base_images") or prompt.get("images") or [])]
    return content, images


def _prompt_context_from_row(row: JsonDict) -> PromptContext:
    return PromptContext(
        memory_content=_memory_content_from_row(row),
        qa_text=_qa_text_from_row(row),
        frames=_current_frames_from_row(row),
        extra_context=_teacher_context_from_row(row),
    )


def _memory_content_from_row(row: JsonDict) -> list[ContentItem]:
    content: list[ContentItem] = []
    memory = row.get("memory_before") or []
    if not isinstance(memory, list):
        return content
    for item in memory:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "")
        start, end = _interval(item)
        if kind == "bridge":
            text = str(item.get("text") or "").strip()
            content.append(ContentItem("text", text=f'<delta t="{start:.1f}-{end:.1f}">{text}</delta>'))
        elif kind == "note":
            image_path = item.get("image_path")
            if item.get("image_available", True) and image_path:
                content.append(ContentItem("text", text=f'<anchor t="{start:.1f}-{end:.1f}">'))
                content.append(ContentItem("image", image_path=Path(str(image_path))))
                content.append(ContentItem("text", text="</anchor>"))
            else:
                content.append(ContentItem("text", text=f'<anchor t="{start:.1f}-{end:.1f}"></anchor>'))
    return content


def _qa_text_from_row(row: JsonDict) -> str:
    qa_history = row.get("qa_history") or []
    if not isinstance(qa_history, list):
        return ""
    lines: list[str] = []
    for item in qa_history:
        if not isinstance(item, dict):
            continue
        timestamp = float(item.get("t", item.get("timestamp", 0.0)) or 0.0)
        role = str(item.get("role") or "q")
        text = str(item.get("text") or "")
        lines.append(f'<qa t="{timestamp:.1f}" role="{role}">{text}</qa>')
    return "\n".join(lines)


def _current_frames_from_row(row: JsonDict) -> list[FrameRef]:
    current_frames = row.get("current_frames") or []
    if not isinstance(current_frames, list):
        return []
    video_id = str(row.get("video_id") or "")
    frames: list[FrameRef] = []
    for index, item in enumerate(current_frames, start=1):
        if not isinstance(item, dict):
            continue
        start, end = _interval(item)
        image_path = str(item.get("image_path") or "")
        if not image_path:
            raise ValueError(f"Row {row.get('sample_id')} current frame #{index} is missing image_path.")
        frames.append(
            FrameRef(
                video_id=video_id,
                global_index=int(item.get("global_frame_id", item.get("frame_id", index)) or index),
                start_time=start,
                end_time=end,
                image_path=Path(image_path),
                step_local_id=int(item.get("frame_id", index) or index),
            )
        )
    return frames


def _teacher_context_from_row(row: JsonDict) -> str:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        return ""
    annotation = metadata.get("annotation") or {}
    if not isinstance(annotation, dict):
        return ""
    return str(annotation.get("teacher_context") or "")


def _render_content_for_sharegpt(content: list[ContentItem]) -> tuple[str, list[str]]:
    rendered: list[str] = []
    images: list[str] = []
    for item in content:
        if item.type == "text":
            rendered.append(item.text)
        elif item.type == "image" and item.image_path is not None:
            rendered.append("<image>")
            images.append(_path_to_string(item.image_path))
    return "".join(rendered), images


def _path_to_string(path: str | Path) -> str:
    return Path(path).as_posix()


def _interval(item: JsonDict) -> tuple[float, float]:
    value = item.get("t") or item.get("time") or [0.0, 0.0]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    if isinstance(value, str) and "-" in value:
        start, end = value.split("-", 1)
        return float(start), float(end)
    return 0.0, 0.0


def _validate_image_placeholders(row: JsonDict, content: str, images: list[str]) -> None:
    num_placeholders = content.count("<image>")
    if num_placeholders != len(images):
        raise ValueError(
            f"Row {row.get('sample_id')} has {num_placeholders} <image> placeholders "
            f"but {len(images)} image paths."
        )


def run_cli(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    result = export_sharegpt(
        rows,
        args.output,
        dataset_name=args.dataset_name,
        train_prompt_type=args.train_prompt_type,
    )
    if args.dataset_info:
        write_json(dataset_info(args.dataset_name, args.output.name), args.dataset_info)
    print(
        f"[sharegpt] saved {result['count']} row(s) with train_prompt_type={args.train_prompt_type} -> {args.output}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data_engine/sft/outputs/sft_steps.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data_engine/sft/outputs/llamafactory_sharegpt.jsonl"))
    parser.add_argument("--dataset-info", type=Path, default=Path("data_engine/sft/outputs/dataset_info_streamweave_sft.json"))
    parser.add_argument("--dataset-name", default="streamweave_sft")
    parser.add_argument(
        "--train-prompt-type",
        default="production",
        choices=("production", "teacher_synthesis", "teacher_eval", "teacher", "eval", "final", "recorded"),
        help="Prompt used as the ShareGPT user input. Use recorded to keep the prompt stored in sft_steps.jsonl.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_cli(parse_args())
