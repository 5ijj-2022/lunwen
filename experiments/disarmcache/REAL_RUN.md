# 真实整合运行方式

服务器项目目录名是 `lunwen-main-git`。完整的 Conda、模型、语料、API、tmux、批量请求和恢复流程请
按同目录下的 [实验运行说明书.md](实验运行说明书.md) 执行；下面只保留最小启动命令。

该入口已经把检索器、生成器、语义缓存和来源追踪接通。它读取 JSONL 语料，加载真实 HuggingFace 模型，并通过标准输入逐行处理问题；每次请求都会先查语义缓存，未命中才检索和生成，生成结果随后正常写入缓存。

```bash
cd /data1/lab409/wwj/lunwen-main-git/experiments/disarmcache
export PROJECT_ROOT=/data1/lab409/wwj/lunwen-main-git
export MODEL_DIR=/data1/lab409/wwj/models
export HF_HOME="$MODEL_DIR/hf-cache"
PYTHONPATH=. python3 -m disarmcache.integrated_service \
  --retriever-model "$MODEL_DIR/contriever-edited" \
  --retriever-backend disarmrag \
  --disarmrag-source-zip "$PROJECT_ROOT/references/DisarmRAG-poisoning-attack-main.zip" \
  --cache-model "$MODEL_DIR/all-MiniLM-L6-v2" \
  --generator-model "$MODEL_DIR/Qwen2.5-7B-Instruct" \
  --corpus data/corpus.jsonl \
  --retriever-version disarmrag-edit-<sha> \
  --device cuda \
  --threshold 0.80 \
  --query-order cache-first \
  --cache-file runs/real/cache.jsonl \
  --log-file runs/real/events.jsonl
```

然后逐行输入问题。`--retriever-model` 可以指向干净 Contriever，也可以指向 DisarmRAG 编辑后的 Contriever；每个版本必须重建索引并使用不同的 `--retriever-version`。恢复时调用 `ProvenanceCache.invalidate({"disarmrag-edit-<sha>"})`，再用干净检索器启动新实例。

这条真实路径需要 GPU、模型权重和数据集；仓库内的 smoke 测试不会伪装成真实模型结果。

要做先检索后缓存对照，只把请求顺序改为：

```bash
--query-order retrieval-first
```

服务运行后查看缓存库和检索结果：

```bash
curl -sS http://127.0.0.1:8080/cache | python3 -m json.tool
curl -sS -X POST http://127.0.0.1:8080/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"your question"}' | python3 -m json.tool
```

完整的数据集、提示词和检索器对比步骤见 `EXPERIMENT_RETRIEVAL_FIRST.md`。
