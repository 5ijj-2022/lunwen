"""Inspect the per-question retrieval trace written by integrated_service.py.

Unlike the batch summary (which only shows scores), this viewer answers the two
questions that matter during an audit:

  * what did the system actually ANSWER for this question?
  * which documents did it RETRIEVE (or reuse from the cache) to do so?

On a cache hit the live retriever is bypassed by design.  The trace still
records the documents stored in the matched cache entry and labels them with
document_source=cache_entry, so nothing is silently missing.

Examples
--------
  # show everything in a trace file
  python3 show_retrieval.py runs/real/bitext_cancel_subscription/F3/retrieval_trace.jsonl

  # only questions whose question/answer/document text contains a keyword
  python3 show_retrieval.py <trace> --grep "newsletter"

  # only cache hits / only cache misses
  python3 show_retrieval.py <trace> --path hit

  # show one question by index and print the full document text / prompt
  python3 show_retrieval.py <trace> --index 3 --full

  # one-line-per-question overview
  python3 show_retrieval.py <trace> --list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _haystack(row: dict[str, Any]) -> str:
    parts: list[str] = [str(row.get("query", "")), str(row.get("answer", ""))]
    retrieval = row.get("retrieval") or {}
    for doc in retrieval.get("documents") or []:
        parts.append(str(doc.get("text", "")))
        parts.append(str(doc.get("id", "")))
    target = row.get("target") or {}
    parts.extend(str(v) for v in target.values() if isinstance(v, (str, int, float)))
    return "\n".join(parts).lower()


def _shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def print_record(row: dict[str, Any], index: int, full: bool, text_chars: int) -> None:
    retrieval = row.get("retrieval") or {}
    cache = row.get("cache") or {}
    prompt = row.get("prompt") or {}
    documents = retrieval.get("documents") or []
    sep = "=" * 68

    print(sep)
    print(f"[{index}] 时间: {row.get('timestamp', '')}   路径: "
          f"{str(row.get('path', '')).upper()}   缓存命中: {bool(row.get('hit'))}")
    print(f"问题: {row.get('query', '')}")
    print(f"回答: {row.get('answer', '')}")
    print(f"查询顺序: {row.get('query_order', '')}    文档来源: "
          f"{retrieval.get('document_source', '')}    "
          f"本次是否检索: {retrieval.get('performed_this_request')}")
    print(f"检索器版本: {row.get('answer_retriever_version', '')}    "
          f"当前检索器: {row.get('active_retriever_version', '')}")
    print(f"缓存相似度: {cache.get('similarity')}    阈值: {cache.get('threshold')}    "
          f"匹配阶段: {cache.get('match_basis', '')}    条目: {cache.get('entry_id', '')}")

    target = row.get("target")
    if target:
        print(f"目标: {target.get('target_id', '')}    "
              f"正确: {target.get('gold_answer', '')}    "
              f"攻击: {target.get('target_wrong_answer', '')}")

    print(f"检索文档数: {retrieval.get('document_count', len(documents))}    "
          f"投毒文档数: {retrieval.get('poisoned_count', 0)}    "
          f"目标指令数: {retrieval.get('target_instruction_count', 0)}")
    if retrieval.get("document_source") == "cache_entry":
        print("  (本次为缓存命中，文档来自命中缓存条目存储的历史证据，而非实时检索)")
    if documents:
        for i, doc in enumerate(documents, 1):
            score = doc.get("score")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            print(f"  {i}. {doc.get('id', '')}  rank={doc.get('rank')}  score={score_text}  "
                  f"poisoned={doc.get('poisoned', False)}  type={doc.get('poison_type', '')}")
            text = str(doc.get("text", ""))
            shown = text if full else _shorten(text, text_chars)
            print(f"     {shown}")
    else:
        print("  (无检索文档)")

    if full:
        print(f"提示词变体: {prompt.get('variant', '')}")
        print(f"提示词 system: {prompt.get('system', '')}")
        print("提示词 user:")
        print(prompt.get("user", ""))
        print(f"metadata: {row.get('metadata', {})}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DisarmCache retrieval traces")
    parser.add_argument("trace", help="path to retrieval_trace.jsonl")
    parser.add_argument("--grep", default=None, help="only records containing this keyword")
    parser.add_argument("--path", choices=["hit", "miss"], default=None,
                        help="only cache hits or cache misses")
    parser.add_argument("--index", type=int, default=None,
                        help="show only the Nth matching record (0-based, as in --list)")
    parser.add_argument("--list", action="store_true",
                        help="one line per question: index, path, hit, keyword, doc count")
    parser.add_argument("--full", action="store_true",
                        help="print full document text and prompt")
    parser.add_argument("--text-chars", type=int, default=240,
                        help="document text length when not --full (default 240)")
    args = parser.parse_args()

    rows = load_trace(args.trace)
    if not rows:
        print(f"No records found in {args.trace}", file=sys.stderr)
        return

    selected: list[tuple[int, dict[str, Any]]] = []
    for i, row in enumerate(rows):
        if args.path and row.get("path") != args.path:
            continue
        if args.grep and args.grep.lower() not in _haystack(row):
            continue
        selected.append((i, row))

    if args.list:
        for i, row in selected:
            retrieval = row.get("retrieval") or {}
            print(f"[{i}] {row.get('timestamp', '')}  "
                  f"{str(row.get('path', '')).upper():<4}  hit={bool(row.get('hit'))!s:<5}  "
                  f"docs={retrieval.get('document_count', 0)}  "
                  f"src={retrieval.get('document_source', '')}  "
                  f"Q={_shorten(str(row.get('query', '')), 80)}")
        print(f"-- {len(selected)} / {len(rows)} records --")
        return

    if args.index is not None:
        matches = [(i, row) for i, row in selected if i == args.index]
        if not matches:
            print(f"No record with index {args.index} after filtering", file=sys.stderr)
            return
        selected = matches

    if not selected:
        print("No matching records", file=sys.stderr)
        return
    for i, row in selected:
        print_record(row, i, args.full, args.text_chars)
    print(f"-- {len(selected)} / {len(rows)} records --")


if __name__ == "__main__":
    main()
