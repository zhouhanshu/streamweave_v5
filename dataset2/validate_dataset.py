#!/usr/bin/env python3
"""Validate a dataset2 StreamWeave-style dataset directory."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_TASKS = {"backward", "realtime", "forward"}
OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IMAGE_RE = re.compile(r"^(\d{6})\.(jpg|jpeg|png)$", flags=re.IGNORECASE)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    validator = DatasetValidator(
        dataset_path,
        check_frames=args.check_frames,
        max_rows=args.max_rows,
        max_examples=args.max_examples,
    )
    report = validator.run()
    output_path = Path(args.output) if args.output else dataset_path / "validation_report.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(format_report(report, output_path), flush=True)
    if report["errors"]["total"] and not args.no_fail:
        raise SystemExit(1)


class DatasetValidator:
    def __init__(self, root: Path, *, check_frames: str, max_rows: int, max_examples: int) -> None:
        self.root = root
        self.check_frames = check_frames
        self.max_rows = max_rows
        self.max_examples = max_examples
        self.errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.warnings: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.error_counts: Counter[str] = Counter()
        self.warning_counts: Counter[str] = Counter()
        self.stats: dict[str, Any] = {
            "dataset_root": str(root),
            "annotations": 0,
            "videos_in_index": 0,
            "unique_annotation_videos": 0,
            "tasks": Counter(),
            "splits": Counter(),
            "datasets": Counter(),
            "option_counts": Counter(),
            "duration_buckets": Counter(),
            "frame_check_mode": check_frames,
        }
        self.video_index: dict[str, dict[str, Any]] = {}
        self.annotation_video_ids: set[str] = set()
        self.frame_checks_done: set[str] = set()

    def run(self) -> dict[str, Any]:
        self._check_root()
        self._load_video_index()
        self._validate_annotations()
        self._finalize_cross_checks()
        return self._report()

    def _check_root(self) -> None:
        if not self.root.exists():
            self._error("missing_dataset_root", path=str(self.root))
            return
        if not self.root.is_dir():
            self._error("dataset_path_not_dir", path=str(self.root))
            return
        for name in ("annotations.jsonl", "video_index.jsonl"):
            if not (self.root / name).exists():
                self._error("missing_required_file", path=str(self.root / name))
        if not (self.root / "video").exists():
            self._error("missing_video_dir", path=str(self.root / "video"))

    def _load_video_index(self) -> None:
        path = self.root / "video_index.jsonl"
        if not path.exists():
            return
        for line_no, row in iter_jsonl(path, self.errors, self.error_counts, code="bad_video_index_json"):
            video_id = clean_str(row.get("video_id"))
            if not video_id:
                self._error("video_index_missing_video_id", file=str(path), line=line_no)
                continue
            if video_id in self.video_index:
                self._error("duplicate_video_index_id", video_id=video_id, file=str(path), line=line_no)
                continue
            self.video_index[video_id] = row
            self._validate_video_index_row(row, line_no)
        self.stats["videos_in_index"] = len(self.video_index)

    def _validate_video_index_row(self, row: dict[str, Any], line_no: int) -> None:
        video_id = clean_str(row.get("video_id"))
        frame_count = as_int(row.get("frame_count"))
        sample_fps = as_float(row.get("sample_fps", row.get("fps")))
        frame_id_base = as_int(row.get("frame_id_base", 0))
        frames_dir = clean_str(row.get("frames_dir") or row.get("video") or f"video/{video_id}")
        if frame_count is None or frame_count <= 0:
            self._error("video_index_bad_frame_count", video_id=video_id, frame_count=row.get("frame_count"), line=line_no)
        if sample_fps is None or sample_fps <= 0:
            self._error("video_index_bad_sample_fps", video_id=video_id, sample_fps=row.get("sample_fps"), line=line_no)
        if frame_id_base is None or frame_id_base < 0:
            self._error("video_index_bad_frame_id_base", video_id=video_id, frame_id_base=row.get("frame_id_base"), line=line_no)
        if frames_dir:
            path = self.resolve_path(frames_dir)
            if not path.exists():
                self._error("video_index_missing_frames_dir", video_id=video_id, frames_dir=frames_dir, line=line_no)

    def _validate_annotations(self) -> None:
        path = self.root / "annotations.jsonl"
        if not path.exists():
            return
        seen_sample_ids: set[str] = set()
        row_count = 0
        for line_no, row in iter_jsonl(path, self.errors, self.error_counts, code="bad_annotation_json"):
            row_count += 1
            if self.max_rows and row_count > self.max_rows:
                break
            self.stats["annotations"] += 1
            self._validate_annotation_row(row, line_no, seen_sample_ids)

    def _validate_annotation_row(self, row: dict[str, Any], line_no: int, seen_sample_ids: set[str]) -> None:
        video_id = clean_str(row.get("video_id"))
        sample_id = clean_str(row.get("sample_id"))
        task = clean_str(row.get("task")).lower()
        dataset_name = clean_str(row.get("dataset"))
        split = clean_str(row.get("split"))
        question = clean_str(row.get("question"))
        answer = clean_str(row.get("answer"))
        gt = row.get("gt")
        frame_count = as_int(row.get("frame_count"))
        sample_fps = as_float(row.get("sample_fps", row.get("fps")))
        frame_id_base = as_int(row.get("frame_id_base", 0))
        realtime = as_float(row.get("realtime", row.get("query_timestamp")))
        frames_dir = clean_str(row.get("frames_dir") or row.get("video"))

        self.stats["tasks"][task or "<missing>"] += 1
        self.stats["splits"][split or "<missing>"] += 1
        self.stats["datasets"][dataset_name or "<missing>"] += 1
        self.annotation_video_ids.add(video_id)

        if not video_id:
            self._error("annotation_missing_video_id", line=line_no)
        if sample_id:
            if sample_id in seen_sample_ids:
                self._error("duplicate_sample_id", sample_id=sample_id, line=line_no)
            seen_sample_ids.add(sample_id)
        if task not in ALLOWED_TASKS:
            self._error("bad_task", task=task, line=line_no, video_id=video_id)
        if not question:
            self._error("empty_question", line=line_no, video_id=video_id)
        if not answer:
            self._error("empty_answer", line=line_no, video_id=video_id)
        if gt is None or clean_str(gt) == "":
            self._error("missing_gt", line=line_no, video_id=video_id)
        if frame_count is None or frame_count <= 0:
            self._error("bad_frame_count", frame_count=row.get("frame_count"), line=line_no, video_id=video_id)
        if sample_fps is None or sample_fps <= 0:
            self._error("bad_sample_fps", sample_fps=row.get("sample_fps"), line=line_no, video_id=video_id)
        if frame_id_base is None or frame_id_base < 0:
            self._error("bad_frame_id_base", frame_id_base=row.get("frame_id_base"), line=line_no, video_id=video_id)
        if realtime is None:
            self._error("missing_realtime", line=line_no, video_id=video_id)

        if frame_count and sample_fps and realtime is not None:
            max_time = (frame_count - 1) / sample_fps
            if realtime < -1e-6 or realtime > max_time + 1.01:
                self._error(
                    "realtime_out_of_range",
                    realtime=realtime,
                    max_time=max_time,
                    frame_count=frame_count,
                    sample_fps=sample_fps,
                    line=line_no,
                    video_id=video_id,
                )
            self.stats["duration_buckets"][duration_bucket(max_time)] += 1

        for key in ("ask_time", "clue_time", "target_timestamp", "duration"):
            if key in row and row[key] is not None:
                value = as_float(row[key])
                if value is None or value < -1e-6:
                    self._error("bad_time_field", key=key, value=row[key], line=line_no, video_id=video_id)

        options = row.get("options")
        if options is not None:
            self._validate_options(row, line_no)

        if video_id:
            index_row = self.video_index.get(video_id)
            if index_row is None:
                self._error("annotation_video_not_in_index", video_id=video_id, line=line_no)
            else:
                self._cross_check_index(row, index_row, line_no)

        if frames_dir and video_id and video_id not in self.frame_checks_done:
            self._check_frame_dir(video_id=video_id, frames_dir=frames_dir, frame_count=frame_count, frame_id_base=frame_id_base, line_no=line_no)
            self.frame_checks_done.add(video_id)

    def _validate_options(self, row: dict[str, Any], line_no: int) -> None:
        video_id = clean_str(row.get("video_id"))
        options = row.get("options")
        answer = clean_str(row.get("answer"))
        if not isinstance(options, list):
            self._error("options_not_list", line=line_no, video_id=video_id)
            return
        self.stats["option_counts"][str(len(options))] += 1
        if not (2 <= len(options) <= len(OPTION_LABELS)):
            self._error("bad_option_count", option_count=len(options), line=line_no, video_id=video_id)
            return
        if any(clean_str(option) == "" for option in options):
            self._error("empty_option", line=line_no, video_id=video_id)
        expected = expected_option_index(row.get("gt"), options, answer, row)
        if expected is None:
            self._error("gt_not_compatible_with_options", gt=row.get("gt"), answer=answer, options=options[:6], line=line_no, video_id=video_id)
            return
        if answer and normalize_text(answer) != normalize_text(options[expected]):
            self._warning(
                "answer_text_differs_from_gt_option",
                gt=row.get("gt"),
                expected_option=options[expected],
                answer=answer,
                line=line_no,
                video_id=video_id,
            )

    def _cross_check_index(self, row: dict[str, Any], index_row: dict[str, Any], line_no: int) -> None:
        video_id = clean_str(row.get("video_id"))
        for key in ("frame_count", "sample_fps", "frame_id_base"):
            if key not in row or key not in index_row:
                continue
            if str(row.get(key)) != str(index_row.get(key)):
                self._error(
                    "annotation_index_field_mismatch",
                    key=key,
                    annotation_value=row.get(key),
                    index_value=index_row.get(key),
                    line=line_no,
                    video_id=video_id,
                )
        row_frames = clean_str(row.get("frames_dir") or row.get("video"))
        idx_frames = clean_str(index_row.get("frames_dir") or index_row.get("video"))
        if row_frames and idx_frames and row_frames != idx_frames:
            self._error("annotation_index_frames_dir_mismatch", annotation_frames_dir=row_frames, index_frames_dir=idx_frames, line=line_no, video_id=video_id)

    def _check_frame_dir(self, *, video_id: str, frames_dir: str, frame_count: int | None, frame_id_base: int | None, line_no: int) -> None:
        if self.check_frames == "none":
            return
        path = self.resolve_path(frames_dir)
        if not path.exists():
            self._error("missing_frames_dir", video_id=video_id, frames_dir=frames_dir, line=line_no)
            return
        if not path.is_dir():
            self._error("frames_path_not_dir", video_id=video_id, frames_dir=frames_dir, line=line_no)
            return
        manifest = path / "manifest.json"
        if not manifest.exists():
            self._warning("missing_manifest", video_id=video_id, frames_dir=frames_dir, line=line_no)
        else:
            try:
                manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception as exc:
                self._error("bad_manifest_json", video_id=video_id, path=str(manifest), error=str(exc), line=line_no)
            else:
                if manifest_data.get("status") not in {None, "complete"}:
                    self._error("manifest_not_complete", video_id=video_id, status=manifest_data.get("status"), line=line_no)
                if frame_count is not None and manifest_data.get("frame_count") is not None and int(manifest_data.get("frame_count")) != frame_count:
                    self._error("manifest_frame_count_mismatch", video_id=video_id, manifest_frame_count=manifest_data.get("frame_count"), frame_count=frame_count, line=line_no)

        if frame_count is None or frame_id_base is None:
            return
        expected_ids = [frame_id_base, frame_id_base + frame_count // 2, frame_id_base + frame_count - 1]
        for frame_id in sorted(set(expected_ids)):
            if not self._frame_exists(path, frame_id):
                self._error("missing_sample_frame", video_id=video_id, frame_id=frame_id, frames_dir=frames_dir, line=line_no)
        if self.check_frames == "full":
            ids = sorted(frame_ids_in_dir(path))
            expected = set(range(frame_id_base, frame_id_base + frame_count))
            actual = set(ids)
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing:
                self._error("missing_frames", video_id=video_id, count=len(missing), examples=missing[:10], frames_dir=frames_dir, line=line_no)
            if extra:
                self._warning("extra_frames", video_id=video_id, count=len(extra), examples=extra[:10], frames_dir=frames_dir, line=line_no)
            if len(actual & expected) != frame_count:
                self._error("frame_count_mismatch", video_id=video_id, expected=frame_count, actual=len(actual & expected), frames_dir=frames_dir, line=line_no)

    def _frame_exists(self, frames_dir: Path, frame_id: int) -> bool:
        for ext in ("jpg", "jpeg", "png"):
            if (frames_dir / f"{frame_id:06d}.{ext}").exists():
                return True
        return False

    def _finalize_cross_checks(self) -> None:
        self.stats["unique_annotation_videos"] = len(self.annotation_video_ids)
        if self.video_index and not self.max_rows:
            unused = sorted(set(self.video_index) - self.annotation_video_ids)
            if unused:
                self._warning("video_index_rows_without_annotations", count=len(unused), examples=unused[: self.max_examples])

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.root / path

    def _error(self, code: str, **data: Any) -> None:
        self.error_counts[code] += 1
        if len(self.errors[code]) < self.max_examples:
            self.errors[code].append(data)

    def _warning(self, code: str, **data: Any) -> None:
        self.warning_counts[code] += 1
        if len(self.warnings[code]) < self.max_examples:
            self.warnings[code].append(data)

    def _report(self) -> dict[str, Any]:
        error_counts = dict(sorted(self.error_counts.items()))
        warning_counts = dict(sorted(self.warning_counts.items()))
        return {
            "ok": not bool(error_counts),
            "dataset_root": str(self.root),
            "stats": jsonable_stats(self.stats),
            "errors": {
                "total": sum(error_counts.values()),
                "by_code": error_counts,
                "examples": dict(sorted(self.errors.items())),
            },
            "warnings": {
                "total": sum(warning_counts.values()),
                "by_code": warning_counts,
                "examples": dict(sorted(self.warnings.items())),
            },
        }


def iter_jsonl(path: Path, errors: dict[str, list[dict[str, Any]]], error_counts: Counter[str], *, code: str):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                error_counts[code] += 1
                errors[code].append({"file": str(path), "line": line_no, "error": str(exc)})
                continue
            if not isinstance(row, dict):
                error_counts[code] += 1
                errors[code].append({"file": str(path), "line": line_no, "error": "row is not an object"})
                continue
            yield line_no, row


def expected_option_index(gt: Any, options: list[Any], answer: str, row: dict[str, Any]) -> int | None:
    option_texts = [clean_str(option) for option in options]
    answer_norm = normalize_text(answer)
    source_letter = clean_str(row.get("source_gt_letter")).upper()
    if source_letter and len(source_letter) == 1 and source_letter in OPTION_LABELS:
        idx = OPTION_LABELS.index(source_letter)
        if 0 <= idx < len(option_texts):
            return idx
    if isinstance(gt, str):
        text = gt.strip()
        if len(text) == 1 and text.upper() in OPTION_LABELS:
            idx = OPTION_LABELS.index(text.upper())
            return idx if 0 <= idx < len(option_texts) else None
        try:
            raw = int(text)
        except ValueError:
            gt_norm = normalize_text(text)
            for idx, option in enumerate(option_texts):
                if normalize_text(option) == gt_norm:
                    return idx
            return None
    else:
        try:
            raw = int(gt)
        except (TypeError, ValueError):
            return None

    candidates = []
    if 0 <= raw < len(option_texts):
        candidates.append(raw)
    if 1 <= raw <= len(option_texts):
        candidates.append(raw - 1)
    if answer_norm:
        for idx in candidates:
            if normalize_text(option_texts[idx]) == answer_norm:
                return idx
    return candidates[0] if candidates else None


def frame_ids_in_dir(path: Path) -> list[int]:
    ids: list[int] = []
    for item in path.iterdir():
        if not item.is_file():
            continue
        match = IMAGE_RE.match(item.name)
        if match:
            ids.append(int(match.group(1)))
    return ids


def duration_bucket(seconds: float) -> str:
    if seconds < 30:
        return "0-30s"
    if seconds < 60:
        return "30-60s"
    if seconds < 120:
        return "60-120s"
    if seconds < 300:
        return "120-300s"
    if seconds < 600:
        return "300-600s"
    return ">600s"


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(value: Any) -> str:
    text = clean_str(value).lower()
    return re.sub(r"\s+", " ", text).strip(" .。")


def as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def jsonable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, Counter):
            out[key] = dict(sorted(value.items()))
        else:
            out[key] = value
    return out


def format_report(report: dict[str, Any], output_path: Path) -> str:
    stats = report["stats"]
    lines = [
        f"dataset: {report['dataset_root']}",
        f"ok: {report['ok']}",
        f"annotations: {stats.get('annotations', 0)}",
        f"videos_in_index: {stats.get('videos_in_index', 0)}",
        f"unique_annotation_videos: {stats.get('unique_annotation_videos', 0)}",
        f"errors: {report['errors']['total']} {report['errors']['by_code']}",
        f"warnings: {report['warnings']['total']} {report['warnings']['by_code']}",
        f"report: {output_path}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset directory, e.g. dataset2/NeXTVideo")
    parser.add_argument(
        "--check-frames",
        choices=("none", "sample", "full"),
        default="sample",
        help="none: skip frame files; sample: check manifest and first/middle/last frame; full: scan every frame directory.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Debug limit. 0 means all rows.")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--output", default="", help="Default: <dataset>/validation_report.json")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 even when errors are found.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
