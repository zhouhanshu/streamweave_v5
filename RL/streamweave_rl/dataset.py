"""Dataset adapter for StreamWeave RL.

The dataset intentionally returns a dummy tensor plus metadata.  Actual
multimodal prompts are rendered inside the stateful agent loop because each
turn depends on StreamWeave memory and the current frame window.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class StreamWeaveAgentDataset(Dataset):
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer=None,
        config: Any | None = None,
        processor=None,
        max_samples: int = -1,
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = _to_plain(config)
        self.streamweave_config = dict(self.config.get("streamweave", {}) or {})
        self.rows = _load_rows(data_files)
        for idx, row in enumerate(self.rows):
            _validate_canonical_query_events(row, row_index=idx)
        if max_samples and max_samples > 0:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = dict(self.rows[idx])
        task = str(_first_present(row, ["task", "type", "question_type"], default="")).strip()
        question = _question_text(row)
        query_timestamp = _query_timestamp(row)
        ground_truth = _ground_truth(row)
        video_id = _video_id(row, idx)
        sample_id = _sample_id(row, video_id=video_id, task=task, idx=idx)

        row_cfg = _to_plain(row.get("streamweave_config") or row.get("rl_config") or {})
        streamweave_config = _deep_merge(self.streamweave_config, _streamweave_config_from_row(row))
        streamweave_config = _deep_merge(streamweave_config, row_cfg)
        sample_metadata = _sample_metadata(row, task=task, question=question, ground_truth=ground_truth)

        return {
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "agent_name": "streamweave_agent",
            "data_source": str(sample_metadata.get("dataset") or "streamweave"),
            "sample_id": sample_id,
            "video_id": video_id,
            "video_path": str(_first_present(row, ["video_path", "path", "video_file"], default="")),
            "question": str(question),
            "query_timestamp": query_timestamp,
            "ground_truth": ground_truth,
            "sample_metadata": sample_metadata,
            "streamweave_config": streamweave_config,
            "reward_model": {"ground_truth": ground_truth},
            "extra_info": {"index": idx},
            "index": idx,
            "raw_prompt": [{"role": "user", "content": "StreamWeave stateful video task."}],
        }


def _load_rows(data_files: str | list[str]) -> list[dict[str, Any]]:
    if isinstance(data_files, str):
        files = [item.strip() for item in data_files.split(",") if item.strip()]
    else:
        files = [str(item) for item in data_files]
    rows: list[dict[str, Any]] = []
    for file_name in files:
        path = Path(file_name).expanduser()
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            import pandas as pd

            frame = pd.read_parquet(path)
            rows.extend(_frame_to_records(frame))
        elif suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
        elif suffix == ".json":
            with path.open(encoding="utf-8") as f:
                loaded = json.load(f)
            rows.extend(_rows_from_json(loaded))
        elif suffix == ".csv":
            import pandas as pd

            frame = pd.read_csv(path)
            rows.extend(_frame_to_records(frame))
        else:
            raise ValueError(f"Unsupported StreamWeave RL data file: {path}")
    return rows


def _frame_to_records(frame: Any) -> list[dict[str, Any]]:
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def _rows_from_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        for key in ("data", "annotations", "samples"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [dict(item) for item in nested if isinstance(item, Mapping)]
        return [dict(value)]
    raise ValueError("JSON dataset must contain an object, a list of objects, or a data/annotations/samples list.")


def _first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _question_text(row: dict[str, Any]) -> str:
    event = _first_query_event(row)
    return str(event.get("question") or event.get("content") or "").strip()


def _query_timestamp(row: dict[str, Any]) -> float:
    return float(_first_query_event(row)["time"])


def _ground_truth(row: dict[str, Any]) -> Any:
    target = _last_answer_event(row)
    if target is None:
        return ""
    query = _query_for_answer_event(row, target)
    if _answer_type(query) == "mcq":
        return target.get("gt", target.get("answer", target.get("content", "")))
    return target.get("answer", target.get("content", target.get("gt", "")))


def _first_query_event(row: dict[str, Any]) -> Mapping[str, Any]:
    return row["query_events"][0]


def _last_answer_event(row: dict[str, Any]) -> Mapping[str, Any] | None:
    for query in reversed(row["query_events"]):
        answers = query.get("answer_events")
        if isinstance(answers, list) and answers:
            return answers[-1]
    return None


def _query_for_answer_event(row: dict[str, Any], target: Mapping[str, Any]) -> Mapping[str, Any]:
    for query in row["query_events"]:
        answers = query.get("answer_events")
        if isinstance(answers, list) and any(answer is target for answer in answers):
            return query
    return _first_query_event(row)


def _validate_canonical_query_events(row: dict[str, Any], *, row_index: int) -> None:
    events = row.get("query_events")
    if not isinstance(events, list) or not events:
        raise ValueError(
            f"RL row {row_index} must use the canonical query_events list; "
            "legacy top-level question/query_timestamp/response fields are not accepted."
        )
    for query_index, query in enumerate(events):
        if not isinstance(query, Mapping):
            raise ValueError(f"RL row {row_index} query_events[{query_index}] must be an object")
        _require_nonempty(query, "qid", row_index=row_index, query_index=query_index)
        _require_float_time(query, "time", row_index=row_index, query_index=query_index)
        _require_nonempty(query, "content", row_index=row_index, query_index=query_index)
        answer_type = _answer_type(query)
        if answer_type == "mcq" and not _normalize_options(query.get("options")):
            raise ValueError(f"RL row {row_index} query_events[{query_index}] has answer_type=mcq but no options")
        answers = query.get("answer_events")
        if not isinstance(answers, list):
            raise ValueError(f"RL row {row_index} query_events[{query_index}].answer_events must be a list")
        for answer_index, answer in enumerate(answers):
            if not isinstance(answer, Mapping):
                raise ValueError(
                    f"RL row {row_index} query_events[{query_index}].answer_events[{answer_index}] must be an object"
                )
            _require_float_time(answer, "time", row_index=row_index, query_index=query_index, answer_index=answer_index)
            if answer_type == "mcq":
                _require_nonempty(answer, "gt", row_index=row_index, query_index=query_index, answer_index=answer_index)
                _require_nonempty(answer, "answer", row_index=row_index, query_index=query_index, answer_index=answer_index)
            elif not any(str(answer.get(key) or "").strip() for key in ("answer", "content", "gt")):
                raise ValueError(
                    f"RL row {row_index} query_events[{query_index}].answer_events[{answer_index}] "
                    "must contain answer/content/gt"
                )


def _answer_type(query: Mapping[str, Any]) -> str:
    explicit = str(query.get("answer_type") or "").strip().lower()
    if explicit in {"mcq", "multiple_choice", "multiple-choice"}:
        return "mcq"
    if explicit in {"text", "freeform", "natural_language", "natural-language"}:
        return "text"
    raise ValueError("query_events[].answer_type is required and must be 'mcq' or 'text'")


def _normalize_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _require_nonempty(
    obj: Mapping[str, Any],
    key: str,
    *,
    row_index: int,
    query_index: int,
    answer_index: int | None = None,
) -> None:
    if str(obj.get(key) or "").strip():
        return
    location = f"RL row {row_index} query_events[{query_index}]"
    if answer_index is not None:
        location += f".answer_events[{answer_index}]"
    raise ValueError(f"{location} requires non-empty {key!r}")


def _require_float_time(
    obj: Mapping[str, Any],
    key: str,
    *,
    row_index: int,
    query_index: int,
    answer_index: int | None = None,
) -> None:
    if key not in obj:
        _require_nonempty(obj, key, row_index=row_index, query_index=query_index, answer_index=answer_index)
    try:
        float(obj[key])
    except (TypeError, ValueError) as exc:
        location = f"RL row {row_index} query_events[{query_index}]"
        if answer_index is not None:
            location += f".answer_events[{answer_index}]"
        raise ValueError(f"{location}.{key} must be numeric") from exc


def _video_id(row: dict[str, Any], idx: int) -> str:
    value = _first_present(row, ["video_id"], default=None)
    if value is not None:
        return str(value)
    if row.get("video") is not None:
        return Path(str(row["video"])).stem if Path(str(row["video"])).suffix else Path(str(row["video"])).name
    return str(_first_present(row, ["id"], default=f"sample_{idx}"))


def _sample_id(row: dict[str, Any], *, video_id: str, task: str, idx: int) -> str:
    explicit = _first_present(row, ["sample_id", "qa_id"], default=None)
    if explicit is not None:
        return str(explicit)
    row_id = _first_present(row, ["id"], default=None)
    if row_id is not None and str(row_id) != video_id:
        return str(row_id)
    return f"{video_id}_{task or idx}"


def _streamweave_config_from_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    dataset_name = str(row.get("dataset") or "").strip()
    if dataset_name:
        out["dataset_name"] = dataset_name
        out["dataset"] = {"dataset_name": dataset_name}
    fps = _first_present(row, ["sample_fps", "fps"], default=None)
    if fps is not None:
        out.setdefault("runtime", {})["sample_fps"] = float(fps)
    if row.get("frame_id_base") is not None:
        out.setdefault("dataset", {})["frame_id_base"] = int(row.get("frame_id_base") or 0)
    return out


def _sample_metadata(row: dict[str, Any], *, task: str, question: str, ground_truth: Any) -> dict[str, Any]:
    metadata = _to_plain_value(row)
    if not isinstance(metadata, dict):
        metadata = dict(row)
    metadata["task"] = task
    metadata["question"] = question
    metadata["ground_truth"] = ground_truth
    if not metadata.get("dataset"):
        metadata["dataset"] = metadata.get("benchmark") or "streamweave"
    return metadata


def _bool_config(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _to_plain(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            converted = OmegaConf.to_container(value, resolve=True)
            return converted if isinstance(converted, dict) else {}
    except Exception:
        pass
    plain = _to_plain_value(value)
    return plain if isinstance(plain, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _to_plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_plain_value(item) for item in value)
    return value
