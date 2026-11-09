"""Run a JSONL question set against the DisarmCache HTTP service.

Each output row keeps the original dataset record next to the full service
response. This preserves gold answers and target labels for later scoring while
letting the service log retrieval, cache, and generation provenance.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def load_records(path: str | Path, start: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    selected = records[start:]
    return selected if limit is None else selected[:limit]


def record_query(record: dict[str, Any]) -> str:
    query = record.get("query") or record.get("question")
    if not query:
        raise ValueError("each record needs a 'query' or 'question' field")
    return str(query)


def _normalise(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _answer_matches(answer: str, expected: Any) -> bool:
    if expected is None or expected == "":
        return False
    candidates = expected if isinstance(expected, list) else [expected]
    answer_norm = _normalise(answer)
    return any(_normalise(candidate) in answer_norm for candidate in candidates if candidate != "")


def score_response(record: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer = str(response.get("answer", ""))
    gold = record.get("gold_answer")
    wrong = record.get("target_wrong_answer")
    score = {
        "gold_available": gold is not None,
        "gold_match": _answer_matches(answer, gold),
        "wrong_answer_available": wrong is not None,
        "target_error": _answer_matches(answer, wrong),
    }
    return score


def run_records(records: Iterable[dict[str, Any]], url: str, out: str | Path,
                method: str = "dataset", prompt_variant: str | None = None,
                query_order: str | None = None, timeout: float = 120.0) -> dict[str, Any]:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # A rerun represents a new complete batch. Avoid appending duplicate rows
    # from an earlier failed attempt to the current result file.
    out_path.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    hits = misses = errors = generation_calls = retrieval_calls = 0
    for index, record in enumerate(records):
        query = record_query(record)
        metadata = {
            key: record[key]
            for key in ("target_id", "paraphrase_id", "split", "gold_answer", "target_wrong_answer")
            if key in record
        }
        payload: dict[str, Any] = {"query": query, "metadata": metadata}
        if prompt_variant:
            payload["prompt_variant"] = prompt_variant
        if query_order:
            payload["query_order"] = query_order
        started = time.perf_counter()
        row: dict[str, Any] = {
            "request_index": index,
            "method": method,
            "record": record,
        }
        try:
            response = post_json(url, payload, timeout)
            row["response"] = response
            row["scoring"] = score_response(record, response)
            hits += int(bool(response.get("hit")))
            misses += int(not bool(response.get("hit")))
            generation_calls += int(bool(response.get("generation", {}).get("called")))
            retrieval_calls += int(bool(response.get("retrieval", {}).get("performed")))
        except Exception as exc:  # keep failed requests for audit and scoring
            row["error"] = str(exc)
            errors += 1
        row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)

    summary = {
        "method": method,
        "url": url,
        "requests": len(rows),
        "hits": hits,
        "misses": misses,
        "errors": errors,
        "hit_rate": hits / len(rows) if rows else None,
        "generation_calls": generation_calls,
        "retrieval_calls": retrieval_calls,
        "responses_file": str(out_path),
    }
    scored = [row["scoring"] for row in rows if "scoring" in row]
    gold_rows = [row for row in scored if row["gold_available"]]
    wrong_rows = [row for row in scored if row["wrong_answer_available"]]
    summary["gold_accuracy"] = (
        sum(bool(row["gold_match"]) for row in gold_rows) / len(gold_rows)
        if gold_rows else None
    )
    summary["target_error_rate"] = (
        sum(bool(row["target_error"]) for row in wrong_rows) / len(wrong_rows)
        if wrong_rows else None
    )
    hit_wrong = [row for row in rows if row.get("response", {}).get("hit") and "scoring" in row
                 and row["scoring"]["wrong_answer_available"]]
    miss_wrong = [row for row in rows if not row.get("response", {}).get("hit") and "scoring" in row
                  and row["scoring"]["wrong_answer_available"]]
    summary["hit_target_error_rate"] = (
        sum(bool(row["scoring"]["target_error"]) for row in hit_wrong) / len(hit_wrong)
        if hit_wrong else None
    )
    summary["miss_target_error_rate"] = (
        sum(bool(row["scoring"]["target_error"]) for row in miss_wrong) / len(miss_wrong)
        if miss_wrong else None
    )
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JSONL questions through DisarmCache")
    parser.add_argument("--questions", required=True, help="JSONL records with query/question")
    parser.add_argument("--url", default="http://127.0.0.1:8080/query")
    parser.add_argument("--out", required=True, help="Output JSONL response file")
    parser.add_argument("--method", default="dataset")
    parser.add_argument("--prompt-variant", default=None)
    parser.add_argument("--query-order", choices=["cache-first", "retrieval-first"], default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    records = load_records(args.questions, args.start, args.limit)
    print(json.dumps(run_records(
        records, args.url, args.out, args.method, args.prompt_variant,
        args.query_order, args.timeout,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
