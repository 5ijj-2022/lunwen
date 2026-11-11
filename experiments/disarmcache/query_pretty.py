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
            print("证据文档: 使用缓存条目中的 Q+D 文档")
    if documents:
        if retrieval.get("documents"):
            print("证据文档:")
        for index, document in enumerate(documents[: max(0, args.max_docs)], 1):
            text = " ".join(str(document.get("text", "")).split())
            if len(text) > 240:
                text = text[:237] + "..."
            print(f"  {index}. {document.get('id', '')}  score={document.get('score', 0.0):.4f}")
            print(f"     {text}")
    else:
        print("证据文档: 本次请求未执行检索或没有返回文档")


if __name__ == "__main__":
    main()
