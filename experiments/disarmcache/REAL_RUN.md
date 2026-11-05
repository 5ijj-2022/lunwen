# 真实整合运行方式

该入口已经把检索器、生成器、语义缓存和来源追踪接通。它读取 JSONL 语料，加载真实 HuggingFace 模型，并通过标准输入逐行处理问题；每次请求都会先查语义缓存，未命中才检索和生成，生成结果随后正常写入缓存。

```bash
cd /data1/lab409/wwj/lunwen-main/experiments/disarmcache
python3 -m pip install torch transformers sentence-transformers numpy
PYTHONPATH=. python3 -m disarmcache.integrated_service \
  --retriever-model /models/contriever-edited \
  --retriever-backend disarmrag \
  --disarmrag-source-zip ../../references/DisarmRAG-poisoning-attack-main.zip \
  --cache-model sentence-transformers/all-MiniLM-L6-v2 \
  --generator-model /models/Qwen2.5-7B-Instruct \
  --corpus data/corpus.jsonl \
  --retriever-version disarmrag-edit-<sha> \
  --device cuda \
  --threshold 0.80 \
  --cache-file runs/real/cache.jsonl \
  --log-file runs/real/events.jsonl
```

然后逐行输入问题。`--retriever-model` 可以指向干净 Contriever，也可以指向 DisarmRAG 编辑后的 Contriever；每个版本必须重建索引并使用不同的 `--retriever-version`。恢复时调用 `ProvenanceCache.invalidate({"disarmrag-edit-<sha>"})`，再用干净检索器启动新实例。

这条真实路径需要 GPU、模型权重和数据集；仓库内的 smoke 测试不会伪装成真实模型结果。
