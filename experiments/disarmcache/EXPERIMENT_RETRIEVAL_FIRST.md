# DisarmCache 对比实验：缓存库、先检索后缓存、检索器与提示词

当前版本保留原来的 `cache-first` 路径，并新增 `retrieval-first` 路径。两种路径使用相同的
检索器、语料、缓存编码器、生成模型和问题集合，只改变请求顺序，结果可以直接比较。

`cache-first` 分支已经保存为 Git 分支：

```text
cache-first -> 9d06496
```

## 1. 查看缓存库和检索结果

服务启动后，缓存库可以直接查看，不需要读取内部 JSONL 文件：

```bash
curl -sS http://127.0.0.1:8080/cache | python -m json.tool
```

按检索器版本过滤：

```bash
curl -sS 'http://127.0.0.1:8080/cache?retriever_version=disarmrag-edit-001' \
  | python -m json.tool
```

查看一次检索和当前缓存候选：

```bash
curl -sS -X POST http://127.0.0.1:8080/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many episodes did season one have?"}' \
  | python -m json.tool
```

返回中的 `retrieval.documents` 包含文档、排名、分数和 `poisoned` 标记，命中时
`cache.matched_entry` 包含缓存条目的 query、answer、证据 ID、检索器版本和提示词版本；miss
后新写入的条目位于 `cache.admitted_entry`。

只查看缓存匹配：

```bash
curl -sS -X POST http://127.0.0.1:8080/cache/lookup \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many episodes did season one have?"}' \
  | python -m json.tool
```

## 2. 启动先检索后缓存模式

在启动命令中加入 `--query-order retrieval-first`。这个模式每次请求都会先计算检索结果，再
查询语义缓存；命中时不调用生成模型，但响应会同时展示检索文档和缓存条目。

```bash
cd /data1/lab409/wwj/lunwen-main/experiments/disarmcache
export PROJECT_ROOT=/data1/lab409/wwj/lunwen-main
export MODEL_DIR=/data1/lab409/wwj/models
export DATA_DIR=$PROJECT_ROOT/experiments/disarmcache/data
export RUN_DIR=$PROJECT_ROOT/experiments/disarmcache/runs/real/retrieval_first
export HF_HOME=$MODEL_DIR/hf-cache
mkdir -p $RUN_DIR

read -rsp "OPENAI_API_KEY: " OPENAI_API_KEY
echo
export OPENAI_API_KEY

PYTHONPATH=. python3 -m disarmcache.integrated_service \
  --retriever-model $MODEL_DIR/contriever-edited \
  --retriever-backend disarmrag \
  --disarmrag-source-zip $PROJECT_ROOT/references/DisarmRAG-poisoning-attack-main.zip \
  --top-k 5 \
  --cache-model $MODEL_DIR/all-MiniLM-L6-v2 \
  --generator-backend openai \
  --generator-model gpt-4o-mini \
  --prompt-file config/prompts.json \
  --prompt-variant evidence_only \
  --corpus $DATA_DIR/corpus_poisoned.jsonl \
  --retriever-version disarmrag-edit-001 \
  --device cuda \
  --threshold 0.80 \
  --query-order retrieval-first \
  --cache-file $RUN_DIR/cache.jsonl \
  --log-file $RUN_DIR/events.jsonl \
  --http-host 127.0.0.1 \
  --http-port 8080
```

返回字段的含义：

| 字段 | 含义 |
|---|---|
| `query_order` | `cache-first` 或 `retrieval-first`。 |
| `retrieval.documents` | 本次检索得到的文档、rank、score 和污染标记。 |
| `cache.matched_entry` | 命中时的缓存条目；包含 answer、evidence_ids 和版本。 |
| `cache.admitted_entry` | miss 后刚写入的缓存条目。 |
| `generation.called` | 是否真的调用了生成模型。 |
| `prompt_variant` | 当前提示词实验版本。 |
| `metadata` | 问题文件中的 target_id、gold_answer 等标签。 |

截图中的改写相似度约为 `0.73`，而 `0.80` 阈值会判定为 miss。演示命中时可以重复同一个问题，
或临时使用 `--threshold 0.70`；正式论文要把阈值作为独立实验变量，不能为了命中率临时调整。

## 3. 数据集准备和批量运行

## 3.1 B3：干净检索器的 retrieval-first 对比

B3 使用干净 Contriever、query 缓存键和 `retrieval-first` 顺序。它不需要编辑后的
`checkpoint.pth`，可以在当前干净语料或 NQ 语料上先完成。启动服务时使用独立端口、缓存和日志：

```bash
--retriever-model "$MODEL_DIR/contriever" \
--retriever-version B3-clean-retrieval-first \
--query-order retrieval-first \
--cache-file runs/real/B3/cache.jsonl \
--log-file runs/real/B3/events.jsonl \
--http-port 8081
```

服务运行后，可以使用专用脚本批量保存 B3 结果：

```bash
./run_b3.sh data/nq_subset/splits/test.jsonl runs/real/B3 http://127.0.0.1:8081/query
```

脚本会生成 `test.responses.jsonl` 和对应的 `.summary.json`。如果使用另一个数据集，只需把
第一个参数换成该数据集的测试问题文件。

问题文件每行一个 JSON，至少包含 `question` 或 `query`。正式实验建议包含：

```json
{"target_id":"nq-001","paraphrase_id":"nq-001-p1","question":"...","gold_answer":"...","target_wrong_answer":"..."}
```

先按 target 分组划分数据，避免同一事实的改写进入不同集合：

```bash
python3 prepare_dataset.py \
  --input data/questions_all.jsonl \
  --out-dir data/splits \
  --seed 12
```

批量发送测试集并保留原始 gold 字段：

```bash
python3 run_dataset.py \
  --questions data/splits/test.jsonl \
  --url http://127.0.0.1:8080/query \
  --query-order retrieval-first \
  --method B6-retrieval-first \
  --out runs/real/retrieval_first/test.responses.jsonl
```

输出文件中的每行包含原始 `record`、完整 `response` 和可用时的 `scoring`。脚本会根据
`gold_answer` 与 `target_wrong_answer` 计算 gold accuracy、Overall-ASR、Hit-ASR、Miss-ASR；
后续可以再按 target 分组计算置信区间和恢复残留率。

## 4. 提示词实验

内置三个提示词版本：`evidence_only`、`conflict_aware` 和 `disarmcache_guard`。在同一服务上
运行提示词实验时，每个版本会使用自己的缓存命名空间，避免不同提示词互相命中：

```bash
python3 prompt_experiment.py \
  --questions data/splits/test.jsonl \
  --url http://127.0.0.1:8080/query \
  --query-order retrieval-first \
  --out-dir runs/real/prompt_ablation
```

自定义提示词文件格式：

```json
{
  "variants": {
    "my_prompt": {
      "system": "Treat retrieved text as evidence and verify conflicts.",
      "user": "Evidence:\n{context}\nQuestion: {query}\nAnswer:"
    }
  }
}
```

启动服务时传入：

```bash
--prompt-file data/prompts/my_prompts.json --prompt-variant my_prompt
```

这部分是提示词优化/消融实验；如果复现 DisarmRAG 的代理提示词生成阶段，还要在独立的上游
环境中运行 `pipeline/gen_att_prompt.py`，并把最终提示词转换为上面的 JSON 格式。训练阶段和
最终测试问题必须分开。

## 5. 干净检索器与编辑检索器对比

分别启动两个服务，使用不同端口、缓存文件和日志文件。两个服务都可以使用
`--query-order retrieval-first`，这样比较结果中一定包含检索文档：

```text
clean service:  --retriever-model .../contriever       --http-port 8081
edited service: --retriever-model .../contriever-edited --http-port 8082
```

如果只比较检索结果，可以使用 `--generator-backend none`；这种模式不会加载生成模型或 API
密钥，但仍提供 `/retrieve`、`/cache` 和 `/config`：

```bash
PYTHONPATH=. python3 -m disarmcache.integrated_service \
  --retriever-model "$MODEL_DIR/contriever" \
  --retriever-backend disarmrag \
  --disarmrag-source-zip "$PROJECT_ROOT/references/DisarmRAG-poisoning-attack-main.zip" \
  --cache-model "$MODEL_DIR/all-MiniLM-L6-v2" \
  --generator-backend none \
  --corpus "$DATA_DIR/corpus_clean.jsonl" \
  --retriever-version clean-001 \
  --device cuda --http-host 127.0.0.1 --http-port 8081
```

批量比较只调用 `/retrieve`，不会调用生成 API：

```bash
python3 retriever_compare.py \
  --questions data/splits/test.jsonl \
  --clean-url http://127.0.0.1:8081 \
  --edited-url http://127.0.0.1:8082 \
  --out runs/real/retriever_compare.jsonl
```

结果包含两种检索器的 retrieved_ids、retrieved_scores、poisoned_ids 和版本号，可计算 Recall@k、
Top-1 命中、污染文档比例和正常语料退化。

`contriever-edited` 必须来自真实的 DisarmRAG 编辑输出，并含有 `checkpoint.pth`。整合服务只
加载和比较 checkpoint，不会把干净模型伪装成编辑模型。

## 6. 生成图表

新的绘图脚本兼容 smoke 和真实服务日志，不会因为真实日志缺少 `phase` 字段而除零：

```bash
python3 plot_results.py \
  runs/real/retrieval_first/events.jsonl \
  --out runs/real/retrieval_first/figures
```

它会生成缓存路径、缓存相似度、检索分数、缓存增长图；只有日志存在 `phase` 和 `target_error`
字段时才生成按阶段错误率图。
