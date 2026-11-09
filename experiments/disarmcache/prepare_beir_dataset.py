"""Convert a BEIR dataset into the DisarmCache JSONL layout.

The converter keeps the qrels-positive documents for the selected queries and
optionally samples additional corpus documents as retrieval distractors. It
does not invent answer strings: BEIR qrels identify relevant documents, but
they do not provide extractive QA answers. The resulting question records can
therefore be used for retrieval/cache metrics; add verified gold answers from
the original QA annotation before reporting answer accuracy.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def load_qrels(path: Path, split: str) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"qrels file not found: {path}")
    result: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if parts[0].lower() in {"query-id", "query_id"}:
                continue
            if len(parts) >= 4:
                row_split, query_id, doc_id, score = parts[:4]
                if row_split != split:
                    continue
            elif len(parts) == 3:
                query_id, doc_id, score = parts
            else:
                raise ValueError(f"{path}:{line_number} has fewer than 3 columns")
            try:
                relevant = float(score) > 0
            except ValueError as exc:
                raise ValueError(f"invalid qrel score at {path}:{line_number}") from exc
            if relevant:
                result.setdefault(str(query_id), []).append(str(doc_id))
    return result


def load_queries(path: Path, qrels: dict[str, list[str]], limit: int) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        query_id = str(row.get("_id", row.get("id", "")))
        if query_id not in qrels:
            continue
        text = str(row.get("text", row.get("query", ""))).strip()
        if not text:
            continue
        queries.append({
            "query_id": query_id,
            "text": text,
            "positive_doc_ids": sorted(set(qrels[query_id])),
        })
    queries.sort(key=lambda row: row["query_id"])
    if limit > 0:
        queries = queries[:limit]
    if not queries:
        raise ValueError("no queries matched the selected qrels split")
    return queries


def document_record(row: dict[str, Any]) -> dict[str, Any]:
    doc_id = str(row.get("_id", row.get("id", "")))
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", row.get("contents", ""))).strip()
    combined = "\n".join(part for part in (title, text) if part)
    if not doc_id or not combined:
        raise ValueError("corpus record needs a document id and non-empty text")
    return {"id": doc_id, "text": combined, "poisoned": False}


def select_documents(path: Path, required: set[str], distractors: int,
                     seed: int) -> list[dict[str, Any]]:
    """Select all required docs plus a deterministic reservoir sample.

    `distractors=-1` keeps the complete corpus. A non-negative value bounds
    the output to required documents plus that many random distractors.
    """
    if distractors < -1:
        raise ValueError("distractors must be -1 or a non-negative integer")
    if distractors == -1:
        output: list[dict[str, Any]] = []
        found: set[str] = set()
        for row in read_jsonl(path):
            doc = document_record(row)
            output.append(doc)
            if doc["id"] in required:
                found.add(doc["id"])
        missing = sorted(required - found)
        if missing:
            raise ValueError(f"qrels reference {len(missing)} missing corpus documents")
        return output

    selected: dict[str, dict[str, Any]] = {}
    reservoir: list[dict[str, Any]] = []
    non_required_seen = 0
    rng = random.Random(seed)
    for row in read_jsonl(path):
        doc = document_record(row)
        if doc["id"] in required:
            selected[doc["id"]] = doc
            continue
        if distractors == 0:
            continue
        non_required_seen += 1
        if len(reservoir) < distractors:
            reservoir.append(doc)
        else:
            replacement = rng.randrange(non_required_seen)
            if replacement < distractors:
                reservoir[replacement] = doc

    missing = sorted(required - selected.keys())
    if missing:
        raise ValueError(f"qrels reference {len(missing)} missing corpus documents")
    return list(selected.values()) + reservoir


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BEIR data to DisarmCache JSONL")
    parser.add_argument("--source", required=True, help="BEIR dataset directory")
    parser.add_argument("--dataset", required=True, help="output dataset name, e.g. nq")
    parser.add_argument("--out", required=True, help="data/datasets/<dataset> output directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--query-limit", type=int, default=100,
                        help="queries to keep; 0 keeps all queries")
    parser.add_argument("--distractors", type=int, default=10000,
                        help="extra corpus docs; -1 keeps the full corpus")
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    corpus_path = source / "corpus.jsonl"
    queries_path = source / "queries.jsonl"
    qrels_path = source / "qrels" / f"{args.split}.tsv"
    for path in (corpus_path, queries_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(f"required BEIR file not found: {path}")

    qrels = load_qrels(qrels_path, args.split)
    queries = load_queries(queries_path, qrels, args.query_limit)
    required = {doc_id for query in queries for doc_id in query["positive_doc_ids"]}
    documents = select_documents(corpus_path, required, args.distractors, args.seed)

    question_rows = [
        {
            "target_id": query["query_id"],
            "paraphrase_id": f"{query['query_id']}-p0",
            "question": query["text"],
            "positive_doc_ids": query["positive_doc_ids"],
            "split": args.split,
            "source_dataset": args.dataset,
        }
        for query in queries
    ]
    question_count = write_jsonl(out / "questions" / f"{args.split}.jsonl", question_rows)
    corpus_count = write_jsonl(out / "corpus" / "clean.jsonl", documents)
    metadata = {
        "dataset": args.dataset,
        "source": "BEIR",
        "source_directory": str(source),
        "split": args.split,
        "seed": args.seed,
        "query_limit": args.query_limit,
        "distractors": args.distractors,
        "questions": question_count,
        "corpus_documents": corpus_count,
        "positive_documents": len(required),
        "gold_answer_note": "BEIR qrels provide relevant document ids, not QA answer strings.",
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
