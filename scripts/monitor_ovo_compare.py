#!/usr/bin/env python3
import argparse
import glob
import json
import os
import time
from pathlib import Path


def natural_key(sample_id: str):
    head = sample_id.split("_", 1)[0]
    return (int(head) if head.isdigit() else 10**12, sample_id)


def load_jsonl(paths):
    if isinstance(paths, (str, Path)):
        paths = [str(paths)]

    rows = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = row.get("sample_id") or row.get("annotation_id") or row.get("id")
                if sample_id is not None:
                    rows[str(sample_id)] = row
    return rows


def load_current(output_dir: Path):
    result_file = output_dir / "results.jsonl"
    if result_file.exists():
        return load_jsonl(result_file)
    return load_jsonl(sorted(glob.glob(str(output_dir / ".results_parts" / "part_*.jsonl"))))


def score(row):
    if not row:
        return None
    if row.get("score") is not None:
        try:
            return int(float(row["score"]))
        except (TypeError, ValueError):
            pass
    if "correct" in row:
        return int(bool(row["correct"]))
    return None


def answer(row):
    if not row:
        return "-"
    return str(row.get("response") or row.get("answer") or row.get("prediction") or "")


def ground_truth(row):
    if not row:
        return "-"
    return str(row.get("ground_truth") or row.get("gt") or row.get("label") or "")


def accuracy(rows, ids=None):
    if ids is None:
        ids = list(rows)
    valid = [sample_id for sample_id in ids if score(rows.get(sample_id)) is not None]
    correct = sum(1 for sample_id in valid if score(rows[sample_id]) == 1)
    return correct, len(valid), (100.0 * correct / len(valid) if valid else 0.0)


def render_once(args):
    current = load_current(args.output_dir)
    refs = {name: load_jsonl(path) for name, path in args.ref}
    ids = sorted(current, key=natural_key)
    cur_correct, cur_total, cur_acc = accuracy(current, ids)

    print("=" * 100)
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"current: {cur_correct}/{cur_total} acc={cur_acc:.2f}% progress={cur_total}/{args.total}")

    for name, rows in refs.items():
        overlap = [sample_id for sample_id in ids if sample_id in rows]
        correct, total, acc = accuracy(rows, overlap)
        cur_wins = [sample_id for sample_id in overlap if score(current[sample_id]) == 1 and score(rows[sample_id]) == 0]
        cur_loses = [sample_id for sample_id in overlap if score(current[sample_id]) == 0 and score(rows[sample_id]) == 1]
        both_correct = sum(1 for sample_id in overlap if score(current[sample_id]) == 1 and score(rows[sample_id]) == 1)
        both_wrong = sum(1 for sample_id in overlap if score(current[sample_id]) == 0 and score(rows[sample_id]) == 0)
        print(
            f"{name}: {correct}/{total} acc={acc:.2f}% "
            f"cur_wins={len(cur_wins)} {cur_wins[:args.show_ids]} "
            f"cur_loses={len(cur_loses)} {cur_loses[:args.show_ids]} "
            f"both_correct={both_correct} both_wrong={both_wrong}"
        )

    print("recent:")
    for sample_id in ids[-args.recent :]:
        row = current[sample_id]
        cells = [f"{sample_id}: cur {answer(row)}/{ground_truth(row)} s={score(row)}"]
        for name, rows in refs.items():
            ref_row = rows.get(sample_id)
            cells.append(f"{name} {answer(ref_row)}/{ground_truth(ref_row)} s={score(ref_row)}" if ref_row else f"{name} -")
        print(" | ".join(cells))
    print(flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ovo_qwen3_rl_exp8_step40_1of8_6gpu"),
    )
    parser.add_argument("--total", type=int, default=364)
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--recent", type=int, default=12)
    parser.add_argument("--show-ids", type=int, default=20)
    parser.add_argument(
        "--ref",
        nargs=2,
        action="append",
        metavar=("NAME", "RESULTS_JSONL"),
        default=[
            ("gemini_state", "outputs/ovo_gemini_1of8_state_note_t/results.jsonl"),
            ("gemini_retry", "outputs/ovo_gemini_full_retry/results.jsonl"),
            ("base", "outputs/ovo_qwen3vl8b_base_full_state_note_t/results.jsonl"),
        ],
    )
    args = parser.parse_args()

    while True:
        render_once(args)
        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
