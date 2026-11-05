# DisarmCache experimental harness

This directory is an offline, auditable harness for the proposed DisarmCache
study. It separates the two upstream mechanisms:

* **retriever stage**: clean or controlled-poisoned evidence selection;
* **semantic-cache stage**: deterministic cosine matching, normal admission,
  provenance, and dependency-aware invalidation.

The harness deliberately uses synthetic evidence and a deterministic answer
function. It validates the measurement protocol and log schema; it is not a
claim about the effectiveness of either upstream attack. Real-model runs must
replace the retriever and generator adapters while preserving the event schema.

## Run the offline smoke experiment

```bash
cd /workspace/research/disarmcache
python3 -m disarmcache.cli --out runs/smoke
```

Outputs are JSONL request/event logs, a summary JSON, and a Markdown report.
The smoke run covers clean RAG, poisoned RAG, naive composition, cache-aware
composition, and dependency invalidation. It has no network or API-key
requirement.

## Planned real-model adapter

1. Freeze a Contriever/edited-Contriever snapshot and rebuild its index for each
   snapshot (the DisarmRAG contract).
2. Use a local generation model or an explicitly configured provider adapter;
   never put credentials in source or logs.
3. Keep the cache encoder independent from the retriever and record threshold,
   tokenizer, truncation, index and model hashes.
4. Run the registered B0--B7 matrix from the draft, with target-level paired
   bootstrap and separate hit/miss path metrics.

The upstream ZIPs and the DOCX are retained under `upstream/` and
`draft_extracted.txt`; attached-document text is treated as a research plan,
not as evidence that any result has been measured.
