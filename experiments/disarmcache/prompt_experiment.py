"""Run prompt variants over the same questions for a controlled ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_dataset import load_records, run_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DisarmCache prompt variants")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080/query")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["evidence_only", "conflict_aware", "disarmcache_guard"],
    )
    parser.add_argument("--query-order", choices=["cache-first", "retrieval-first"], default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    records = load_records(args.questions, limit=args.limit)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for variant in args.variants:
        summaries[variant] = run_records(
            records,
            args.url,
            output / f"{variant}.jsonl",
            method=f"prompt:{variant}",
            prompt_variant=variant,
            query_order=args.query_order,
            timeout=args.timeout,
        )
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
