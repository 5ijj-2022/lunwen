"""Validate and split question JSONL without leaking paraphrases across splits."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_records(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not (row.get("query") or row.get("question")):
            raise ValueError(f"record {index} needs a query or question field")
    return rows


def split_records(rows: list[dict[str, Any]], seed: int,
                  train_ratio: float, dev_ratio: float) -> dict[str, list[dict[str, Any]]]:
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("train_ratio must be > 0 and train_ratio + dev_ratio must be < 1")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = str(row.get("target_id") or row.get("id") or f"row-{index}")
        groups[group].append(row)
    keys = list(groups)
    if len(keys) < 3:
        raise ValueError("at least three target groups are required for train/dev/test splits")
    random.Random(seed).shuffle(keys)
    total = len(keys)
    train_end = min(total - 2, max(1, round(total * train_ratio)))
    dev_end = min(total - 1, max(train_end + 1, train_end + round(total * dev_ratio)))
    result = {
        "train": [row for key in keys[:train_end] for row in groups[key]],
        "dev": [row for key in keys[train_end:dev_end] for row in groups[key]],
        "test": [row for key in keys[dev_end:] for row in groups[key]],
    }
    for split, split_rows in result.items():
        for row in split_rows:
            row["split"] = split
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe DisarmCache question splits")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = split_records(read_records(args.input), args.seed, args.train_ratio, args.dev_ratio)
    summary = {"seed": args.seed, "groups": {}, "records": {}}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        summary["records"][split] = len(rows)
        summary["groups"][split] = len({str(row.get("target_id") or row.get("id")) for row in rows})
    (output / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
