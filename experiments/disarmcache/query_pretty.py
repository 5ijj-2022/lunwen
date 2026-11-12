"""Send one query to the service and print a compact, readable report.

The HTTP response remains complete JSON for experiments. This client is only
for interactive checks, so terminal output does not get buried in the full
retrieval and cache provenance objects.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print one DisarmCache query")
    parser.add_argument("query")
    parser.add_argument("--url", default="http://127.0.0.1:8080/query")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--max-prompt-chars", type=int, default=2400,
                        help="limit displayed prompt text; --json keeps the full prompt")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="print the complete JSON response")
    args = parser.parse_args()

    payload = json.dumps({"query": args.query}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise SystemExit(f"request failed: {type(exc).__name__}: {exc}") from exc

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if result.get("error"):
        print(f"错误: {result['error']}")
        if result.get("error_type"):
            print(f"错误类型: {result['error_type']}")
        raise SystemExit(1)

    cache = result.get("cache") or {}
    retrieval = result.get("retrieval") or {}
    generation = result.get("generation") or {}
    path = str(result.get("path", "unknown")).upper()
    print("=== DisarmCache 查询 ===")
    print(f"问题: {result.get('query', args.query)}")
    print(f"答案: {result.get('answer', '')}")
    print(f"路径: {path}    命中: {bool(result.get('hit'))}")
    print(f"缓存相似度: {cache.get('similarity', result.get('similarity', 0.0))}")
    print(f"缓存键: {cache.get('key_mode', result.get('cache_key_mode', 'query'))}"
          f"    匹配阶段: {cache.get('match_basis', '')}")
    print(f"检索器版本: {result.get('retriever_version', '')}")
    print(f"检索执行: {bool(retrieval.get('performed'))}    生成执行: {bool(generation.get('called'))}")
    if result.get("target"):
        target = result["target"]
        print(f"DisarmRAG 目标: {target.get('target_id', '')}")
        print(f"目标正确答案: {target.get('gold_answer', '')}")
        print(f"目标攻击答案: {target.get('target_wrong_answer', '')}")
    if result.get("retriever_diagnostics"):
        diag = result["retriever_diagnostics"]
        print(f"检索向量诊断: rows={diag.get('rows')} dim={diag.get('dim')} "
              f"row_std={diag.get('row_std')} collapsed={diag.get('collapsed')}")

    # Keep the cache provenance visible in the interactive report.  The
    # batch response and JSONL event log still retain the complete object;
    # this section only makes the database entry readable line by line.
    print("缓存库:")
    print(f"  条目数量: {cache.get('entry_count', 0)}")
    print(f"  匹配阶段: {cache.get('match_basis', '')}")
    cache_entry = cache.get("matched_entry") or cache.get("entry")
    if cache_entry:
        print(f"  entry_id: {cache_entry.get('entry_id', '')}")
        print(f"  query: {cache_entry.get('query', '')}")
        print(f"  answer: {cache_entry.get('answer', '')}")
        print(f"  retriever_version: {cache_entry.get('retriever_version', '')}")
        print(f"  evidence_ids: {cache_entry.get('evidence_ids', [])}")
        print(f"  poisoned_source: {cache_entry.get('poisoned_source', False)}")
        print(f"  cache_key_mode: {cache_entry.get('cache_key_mode', cache.get('key_mode', ''))}")
        key_text = cache_entry.get("cache_key_text")
        if key_text:
            print("  Q+D 缓存键内容:")
            for line in str(key_text).splitlines():
                print(f"    {line}")
    else:
        print("  本次没有可显示的缓存条目")

    prompt = result.get("prompt") or {}
    if prompt:
        print(f"提示词变体: {prompt.get('variant', result.get('prompt_variant', ''))}")
        system = " ".join(str(prompt.get("system", "")).split())
        user = str(prompt.get("user", ""))
        if args.max_prompt_chars >= 0 and len(user) > args.max_prompt_chars:
            user = user[: max(0, args.max_prompt_chars - 3)] + "..."
        print(f"提示词 system: {system}")
        print("提示词 user:")
        print(user)

    documents = retrieval.get("documents") or []
    if not documents:
        cached_entry = cache.get("entry") or cache.get("matched_entry")
        documents = (cached_entry or {}).get("evidence_documents") or []
        if documents:
            print("知识库证据: 使用缓存条目中的 Q+D 文档")
    if documents:
        if retrieval.get("documents"):
            print("知识库检索结果:")
        for index, document in enumerate(documents[: max(0, args.max_docs)], 1):
            text = " ".join(str(document.get("text", "")).split())
            if len(text) > 240:
                text = text[:237] + "..."
            print(f"  {index}. {document.get('id', '')}  score={document.get('score', 0.0):.4f}")
            if document.get("poison_type"):
                print(f"     类型: {document.get('poison_type')}")
            print(f"     {text}")
    else:
        print("证据文档: 本次请求未执行检索或没有返回文档")


if __name__ == "__main__":
    main()
