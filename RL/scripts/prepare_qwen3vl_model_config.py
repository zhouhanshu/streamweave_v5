#!/usr/bin/env python3
"""Prepare a Qwen3-VL model directory compatible with the local Transformers build."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: prepare_qwen3vl_model_config.py SOURCE_MODEL_DIR OUTPUT_DIR", file=sys.stderr)
        return 2
    source = Path(sys.argv[1]).expanduser().resolve()
    output = Path(sys.argv[2]).expanduser().resolve()
    config_path = source / "config.json"
    if not config_path.is_file():
        print(str(source))
        return 0

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    text_config = config.get("text_config")
    needs_patch = (
        config.get("model_type") == "qwen3_vl"
        and isinstance(text_config, dict)
        and text_config.get("rope_scaling") is None
        and isinstance(text_config.get("rope_parameters"), dict)
    )
    if not needs_patch:
        print(str(source))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = output / child.name
        if child.name == "config.json":
            continue
        if target.exists() or target.is_symlink():
            continue
        os.symlink(child, target, target_is_directory=child.is_dir())

    rope_parameters = dict(text_config["rope_parameters"])
    if "rope_theta" in rope_parameters:
        text_config["rope_theta"] = rope_parameters["rope_theta"]
    text_config["rope_scaling"] = {
        key: value
        for key, value in rope_parameters.items()
        if key != "rope_theta"
    }
    tmp_config = output / "config.json.tmp"
    with tmp_config.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_config.replace(output / "config.json")
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
