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

BACKWARD_TASKS = {"EPM", "ASI", "HLD"}
REAL_TIME_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
FORWARD_TASKS = {"REC", "SSR", "CRR"}

BR_PROMPT_TEMPLATE = """\
Question: {question}
Options:
{options}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C).
Do not include any additional text or explanation in your response.
"""

REC_PROMPT_TEMPLATE = """\
You're watching a video in which people may perform a certain type of action repetitively.
The person performing this kind of action is referred to as "they" in the following statement.
Your task is to count how many times different people in the video have performed this kind of action in total.
One complete motion counts as one.
Now, answer the following question: {question}
Provide your answer as a single number (e.g., 0, 1, 2, 3...) indicating the total count.
Do not include any additional text or explanation in your response.
"""

SSR_PROMPT_TEMPLATE = """\
You're watching a tutorial video which contains a sequence of steps.
The following is one step from the whole procedure:
{step}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""

CRR_PROMPT_TEMPLATE = """\
You're responsible for answering questions based on the video content.
The following question is relevant to the latest frames, i.e. the end of the video.
{question}
Decide whether the existing visual content, especially the latest frames near the end of the video, provides enough information for answering the question.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""


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
        self.rows = _expand_rows(_load_rows(data_files))
        if _bool_config(self.streamweave_config.get("require_question", True)):
            self.rows = [row for row in self.rows if _row_has_question(row)]
        if max_samples and max_samples > 0:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = dict(self.rows[idx])
        task = str(_first_present(row, ["task", "type", "question_type"], default="")).strip()
        question = _question_text(row)
        query_timestamp = _query_timestamp(row, task)
        ground_truth = _ground_truth(row, task)
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


def _expand_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        task = str(row.get("task") or "").strip()
        if task in FORWARD_TASKS and isinstance(row.get("test_info"), list):
            for index, info in enumerate(row["test_info"]):
                if not isinstance(info, Mapping):
                    continue
                item = dict(row)
                item.pop("test_info", None)
                item["benchmark"] = item.get("benchmark") or "ovo"
                item["dataset"] = item.get("dataset") or "ovo"
                item["test_index"] = index
                item["sample_id"] = f"{row.get('id')}_{index}"
                item["video_id"] = item["sample_id"]
                item["query_timestamp"] = float(row.get("ask_time", 0.0)) if task == "CRR" else 0.0
                item["target_timestamp"] = float(info.get("realtime", 0.0))
                item["ground_truth"] = info.get("count", 0) if task == "REC" else info.get("type", 0)
                item["question"] = _ovo_forward_question(task, row, info)
                expanded.append(item)
            continue
        if task in BACKWARD_TASKS | REAL_TIME_TASKS:
            item = dict(row)
            item["benchmark"] = item.get("benchmark") or "ovo"
            item["dataset"] = item.get("dataset") or "ovo"
            expanded.append(item)
            continue
        expanded.append(row)
    return expanded


def _first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _row_has_question(row: dict[str, Any]) -> bool:
    if str(_first_present(row, ["question", "query", "query_text"], default="")).strip():
        return True
    events = row.get("query_events")
    if isinstance(events, list):
        return any(isinstance(item, Mapping) and str(_first_present(dict(item), ["text", "question"], default="")).strip() for item in events)
    return False


def _question_text(row: dict[str, Any]) -> str:
    events = row.get("query_events")
    if isinstance(events, list) and events:
        for item in events:
            if not isinstance(item, Mapping):
                continue
            text = str(_first_present(dict(item), ["text", "question"], default="")).strip()
            if text:
                return text
    question = str(_first_present(row, ["question", "query", "query_text"], default="")).strip()
    if not question:
        return ""
    task = str(row.get("task") or "").strip()
    if task in BACKWARD_TASKS | REAL_TIME_TASKS and isinstance(row.get("options"), list):
        options = "; ".join(f"{chr(65 + idx)}. {option}" for idx, option in enumerate(row["options"])) + ";"
        return BR_PROMPT_TEMPLATE.format(question=question, options=options)
    options = row.get("options")
    if isinstance(options, list) and options:
        option_lines = []
        for idx, option in enumerate(options):
            label = chr(ord("A") + idx)
            text = str(option).strip()
            option_lines.append(text if text[:2].upper() == f"{label}." else f"{label}. {text}")
        return (
            f"Question: {question}\n"
            "Options:\n"
            + "\n".join(option_lines)
            + "\n\nRespond only with the letter corresponding to your chosen option."
        )
    return question


def _query_timestamp(row: dict[str, Any], task: str) -> float:
    events = row.get("query_events")
    if isinstance(events, list) and events:
        for item in events:
            if not isinstance(item, Mapping):
                continue
            value = _first_present(dict(item), ["timestamp", "query_time", "ask_time", "realtime"], default=None)
            if value is not None:
                return float(value)
    for key in ("query_timestamp", "query_time", "ask_time", "realtime", "timestamp"):
        if key in row and row[key] is not None:
            return float(row[key])
    if task == "forward" and row.get("clue_time") is not None:
        return max(float(row["clue_time"]) - 1.0, 0.0)
    return 0.0


def _ground_truth(row: dict[str, Any], task: str) -> Any:
    for key in ("ground_truth", "target"):
        if key in row and row[key] is not None:
            return row[key]
    if task in BACKWARD_TASKS | REAL_TIME_TASKS and row.get("gt") is not None:
        try:
            return chr(65 + int(row["gt"]))
        except (TypeError, ValueError):
            return row["gt"]
    if row.get("gt") is not None:
        return row["gt"]
    return _first_present(row, ["answer"], default="")


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


def _ovo_forward_question(task: str, row: dict[str, Any], info: Mapping[str, Any]) -> str:
    if task == "REC":
        return REC_PROMPT_TEMPLATE.format(question=f"How many times did they {row.get('activity', '')}?")
    if task == "SSR":
        return SSR_PROMPT_TEMPLATE.format(step=info.get("step", ""))
    if task == "CRR":
        return CRR_PROMPT_TEMPLATE.format(question=row.get("question", ""))
    return str(row.get("question") or "")


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
