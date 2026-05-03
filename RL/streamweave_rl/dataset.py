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

import pandas as pd
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
        if max_samples and max_samples > 0:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = dict(self.rows[idx])
        question = _first_present(row, ["question", "query", "query_text"], default="")
        query_timestamp = float(_first_present(row, ["query_timestamp", "query_time", "timestamp"], default=0.0) or 0.0)
        ground_truth = str(_first_present(row, ["ground_truth", "answer", "target"], default=""))
        video_id = str(_first_present(row, ["video_id", "id"], default=f"sample_{idx}"))
        sample_id = str(_first_present(row, ["sample_id"], default=video_id))

        row_cfg = _to_plain(row.get("streamweave_config") or row.get("rl_config") or {})
        streamweave_config = _deep_merge(self.streamweave_config, row_cfg)

        return {
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "agent_name": "streamweave_agent",
            "data_source": "streamweave",
            "sample_id": sample_id,
            "video_id": video_id,
            "video_path": str(_first_present(row, ["video_path", "path"], default="")),
            "question": str(question),
            "query_timestamp": query_timestamp,
            "ground_truth": ground_truth,
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
            frame = pd.read_parquet(path)
        elif suffix in {".jsonl", ".json"}:
            frame = pd.read_json(path, lines=suffix == ".jsonl")
        elif suffix == ".csv":
            frame = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported StreamWeave RL data file: {path}")
        rows.extend(json.loads(frame.to_json(orient="records")))
    return rows


def _first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


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
