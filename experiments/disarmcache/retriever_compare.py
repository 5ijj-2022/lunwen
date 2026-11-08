"""Compare clean and edited retriever evidence through the /retrieve endpoint."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_dataset import load_records, post_json, record_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retriever evidence and scores")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--clean-url", required=True, help="Service base URL, e.g. http://127.0.0.1:8081")
    parser.add_argument("--edited-url", required=True, help="Service base URL, e.g. http://127.0.0.1:8082")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    records = load_records(args.questions, limit=args.limit)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    services = {"clean": args.clean_url.rstrip("/"), "edited": args.edited_url.rstrip("/")}
    counts = {name: 0 for name in services}
    with output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            query = record_query(record)
            for method, base_url in services.items():
                started = time.perf_counter()
                row: dict[str, Any] = {
                    "request_index": index,
                    "method": method,
                    "record": record,
                    "query": query,
                }
                try:
                    response = post_json(f"{base_url}/retrieve", {"query": query}, args.timeout)
                    docs = response.get("retrieval", {}).get("documents", [])
                    row["retriever_version"] = response.get("retriever_version")
                    row["retrieved_ids"] = [str(doc.get("id")) for doc in docs]
                    row["retrieved_scores"] = [doc.get("score") for doc in docs]
                    row["poisoned_ids"] = [
                        str(doc.get("id")) for doc in docs if bool(doc.get("poisoned", False))
                    ]
                    counts[method] += 1
                except Exception as exc:
                    row["error"] = str(exc)
                row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"questions": len(records), "services": services, "successful_requests": counts,
               "output": str(output)}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
