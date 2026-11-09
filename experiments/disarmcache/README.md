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
cd /data1/lab409/wwj/lunwen-main-git/experiments/disarmcache
python3 -m disarmcache.cli --out runs/smoke
```

服务器从 GitHub 下载后，如果项目目录名为 `lunwen-main-git`，直接使用上面的路径。
如果目录名不同，只修改项目根路径，不要修改实验代码或内部相对路径。

Outputs are JSONL request/event logs, a summary JSON, and a Markdown report.
The smoke run covers clean RAG, poisoned RAG, naive composition, cache-aware
composition, and dependency invalidation. It has no network or API-key
requirement.

## Run B2/B3 on any registered dataset

For real-model runs, use the shared dataset/experiment entry points:

```bash
./start_experiment.sh nq B2 8080
./run_experiment.sh nq B2
```

`B2` always selects `cache-first` and `B3` always selects
`retrieval-first`. To run another dataset, replace only `nq` and provide the
same two files under `data/datasets/<dataset>/`:

```text
corpus/clean.jsonl
questions/test.jsonl
```

For BEIR-formatted NQ/HotpotQA downloads, run
`prepare_beir_dataset.py` first. It creates the two files above and records the
selection parameters in `metadata.json`; the exact download and conversion
commands are in [DATASET_LAYOUT.md](DATASET_LAYOUT.md).

Results are isolated under `runs/real/<dataset>/B2/` or
`runs/real/<dataset>/B3/`. The full server workflow and environment variables
are documented in [DATASET_LAYOUT.md](DATASET_LAYOUT.md).

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

For the actual model-backed integration, see [REAL_RUN.md](REAL_RUN.md) and
`disarmcache/integrated_service.py`. It connects a clean or DisarmRAG-edited
Contriever checkpoint to an independent cache encoder and a local generator.
