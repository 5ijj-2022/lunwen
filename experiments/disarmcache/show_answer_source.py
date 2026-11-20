"""Verify that cache hits return the matched cached answer verbatim.

For every response row, look up its ``response.entry_id`` in the cache file and
compare ``response.answer`` with that entry's stored ``answer``. On a cache hit
they must be identical, because the service sets ``answer = entry.answer``.

Usage (on the server):
    python3 show_answer_source.py <responses.jsonl>
    python3 show_answer_source.py <responses.jsonl> --cache <cache.jsonl>
    python3 show_answer_source.py <responses.jsonl> --hit-only --limit 10
    python3 show_answer_source.py <responses.jsonl> --mismatch-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("responses", type=Path, help="*.responses.jsonl to inspect")
    ap.add_argument("--cache", type=Path, default=None,
                    help="cache.jsonl (default: <responses parent>/cache.jsonl)")
    ap.add_argument("--limit", type=int, default=20, help="max rows to print")
    ap.add_argument("--hit-only", action="store_true", help="only cache hits")
    ap.add_argument("--mismatch-only", action="store_true", help="only rows where answer != cached answer")
    args = ap.parse_args()

    if args.cache:
        cache_path = args.cache
    else:
        # Look for cache.jsonl next to the responses, then one/two levels up.
        # Responses in archive/ keep the cache at the experiment root, and the
        # root cache.jsonl is a symlink to the current launch's cache.
        candidates = [
            args.responses.parent / "cache.jsonl",
            args.responses.parent.parent / "cache.jsonl",
            args.responses.parent.parent.parent / "cache.jsonl",
        ]
        cache_path = next((c for c in candidates if c.exists()), candidates[0])
    cache_rows = load_jsonl(cache_path)
    by_id = {str(r.get("entry_id")): r for r in cache_rows}
    rows = load_jsonl(args.responses)
    if not rows:
        raise SystemExit(f"no response rows found: {args.responses}")

    hits = same = diff = missing = 0
    shown = 0
    for i, row in enumerate(rows, 1):
        resp = row.get("response") or {}
        is_hit = bool(resp.get("hit"))
        entry_id = str(resp.get("entry_id") or "")
        answer = str(resp.get("answer", ""))
        entry = by_id.get(entry_id)
        cached_answer = str(entry.get("answer", "")) if entry else None
        identical = (cached_answer == answer) if entry is not None else None

        if is_hit:
            hits += 1
            if identical is True:
                same += 1
            elif identical is False:
                diff += 1
            elif identical is None:
                missing += 1
        if args.hit_only and not is_hit:
            continue
        if args.mismatch_only and identical is not False:
            continue
        if shown >= args.limit:
            continue
        shown += 1

        print(f"\n===== 第 {i} 条 [{'命中' if is_hit else '未命中'}] "
              f"entry_id={entry_id[:12]} =====")
        print("问:", (row.get("record") or {}).get("query", ""))
        print("答:", answer)
        if entry is None:
            print("缓存原文: (在缓存文件中未找到该 entry_id——可能缓存文件不是本次运行的)")
        else:
            print("缓存原文:", cached_answer)
            if is_hit:
                print("是否一致:", "✅ 完全一致（命中即复用缓存答案）" if identical
                      else "❌ 不一致（异常，请检查）")

    print("\n" + "=" * 60)
    print(f"缓存文件: {cache_path}  (共 {len(cache_rows)} 条)")
    print(f"命中总数: {hits}  回答==缓存: {same}  不一致: {diff}  "
          f"缓存中找不到entry: {missing}")
    if hits and same == hits:
        print("结论: 所有命中回答都与缓存原文一致，行为正常。")
    elif diff:
        print("结论: 存在不一致，检查是否用了错误的缓存文件（换 --cache 指定本次缓存）。")


if __name__ == "__main__":
    main()
