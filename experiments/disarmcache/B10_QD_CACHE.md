# B10：编辑检索器 + cache-first + Q+D top-3 缓存

B10 是一个独立实验，不会覆盖 B2、B3 的缓存或日志。

它使用：

- DisarmRAG 压缩包中的 `load_retriever` 接口；
- 已编辑的 Contriever 检索器目录；
- top-k=3；
- 缓存键 `Question + Evidence[1..3]`；
- cache-first 请求顺序。

Q+D 缓存存在一个顺序约束：如果请求还没有命中缓存，必须先检索才能得到 D。因此 B10
采用两阶段 cache-first：

1. 先用问题路由键检查已有条目；已有问题命中时直接复用，完全跳过编辑检索器。
2. 路由未命中时执行 top-3 检索，再用完整 Q+D 键匹配已有条目。
3. 两次都未命中时调用生成模型，并把问题、top-3 文档、答案、证据 ID 和检索器版本写入缓存。

响应中的 `cache.key_mode` 为 `query+documents`，`cache.match_basis` 为 `query-route` 或
`query+documents`。缓存条目中的 `cache_key_text` 和 `evidence_documents` 可以直接审计 Q+D。

## 启动

服务器上先确认编辑模型存在：

```bash
test -f /data1/lab409/wwj/models/contriever-edited/checkpoint.pth || \
  test -f /data1/lab409/wwj/models/contriever-edited/config.json
```

如果两种文件都不存在，不能把干净 `contriever` 冒充 B10；需要先完成 DisarmRAG 编辑器输出，或设置
`RETRIEVER_MODEL` 指向真实编辑 checkpoint。

压缩包中的 `train_retriever_editor.sh` 引用了未随包提供的
`editing.train_llama2_mend`。本仓库提供了
`prepare_disarmrag_editor.py`，它调用压缩包实际包含的 `EditTrainer`/MEND，读取包内的 NQ
编辑数据，训练编辑器并导出可加载的 HuggingFace 检索器。例如：

```bash
# 若尚未创建编辑环境，优先复用已验证的 CUDA/PyTorch 环境
conda create -n disarmrag-edit --clone disarmcache
conda activate disarmrag-edit
cd /data1/lab409/wwj/lunwen-main-git/experiments/disarmcache
PYTHONPATH=. python3 prepare_disarmrag_editor.py \
  --source-zip /data1/lab409/wwj/lunwen-main-git/references/DisarmRAG-poisoning-attack-main.zip \
  --clean-model /data1/lab409/wwj/models/contriever \
  --output-dir /data1/lab409/wwj/models/contriever-edited-b10-nq \
  --work-dir /data1/lab409/wwj/lunwen-main-git/experiments/disarmcache/runs/real/nq/B10/editor \
  --dataset nq --train-limit 100 --edit-limit 20 --max-iters 5000 \
  --val-steps 100 --device cuda:0
```

输出目录中的 `b10_edit_metadata.json` 保存源压缩包哈希、编辑指令和目标问题。没有真实编辑输出
时，B10 启动脚本会停止，避免把干净检索器误报为编辑检索器。

## 生成独立投毒语料

压缩包中的攻击结果可以生成一个新的投毒语料文件。该命令保留干净语料，并把每条对抗证据
标记为 `poisoned=true`：

```bash
python3 prepare_disarmrag_corpus.py \
  --clean-corpus "$DATA_ROOT/nq/corpus/clean.jsonl" \
  --source-zip "$PROJECT_ROOT/references/DisarmRAG-poisoning-attack-main.zip" \
  --dataset nq \
  --out "$DATA_ROOT/nq/corpus/poisoned.jsonl" \
  --target-limit 20
```

`--target-limit 20` 适合先做小规模链路验证；正式实验可以设为 `0`，包含压缩包中的全部目标。
启动 B10 时会自动优先选择这个 `poisoned.jsonl`，也可以显式设置 `CORPUS`。

配置 API：

```bash
export API_BASE_URL="https://4sapi.org/v1"
export API_KEY_ENV="API_SECRET_KEY"
export GENERATOR_BACKEND="openai"
export GENERATOR_MODEL="gpt-4o-mini"
read -rsp "请输入纯 API 密钥: " API_SECRET_KEY
echo
export API_SECRET_KEY
```

启动 NQ B10：

```bash
cd /data1/lab409/wwj/lunwen-main-git/experiments/disarmcache
./start_experiment.sh nq B10 8080
```

默认使用：

```text
retriever: /data1/lab409/wwj/models/contriever-edited
top-k: 3
query order: cache-first
cache key: query+documents
run directory: runs/real/nq/B10/
```

如需指定投毒语料：

```bash
export CORPUS="$DATA_ROOT/nq/corpus/poisoned.jsonl"
./start_experiment.sh nq B10 8080
```

## 检查

```bash
curl -sS http://127.0.0.1:8080/health | python3 -m json.tool

python3 query_pretty.py \
  "What is the capital of France?" \
  --url http://127.0.0.1:8080/query

curl -sS http://127.0.0.1:8080/cache | python3 -m json.tool
```

首次问题应看到 `path=miss`、`retrieval.performed=true`、`generation.called=true`，并且响应的
`prompt.system`、`prompt.user` 显示实际说明型提示词。重复完全相同问题应看到 `path=hit`、
`retrieval.performed=false`、`generation.called=false`，并且缓存条目含有
`cache_key_mode=query+documents`、`cache_key_text` 和三个 `evidence_documents`。命中时
`cache.match_basis=query-route` 表示先用问题路由跳过检索，条目仍保存完整 Q+D；首次未命中后的
实际键匹配会记录为 `query+documents`。

## 批量运行

```bash
export DATA_ROOT="$PWD/data/datasets"
export RUN_ROOT="$PWD/runs/real"

./run_experiment.sh \
  nq B10 \
  "$DATA_ROOT/nq/questions/test.jsonl" \
  http://127.0.0.1:8080/query
```

结果保存到：

```text
runs/real/nq/B10/cache.jsonl
runs/real/nq/B10/events.jsonl
runs/real/nq/B10/test.responses.jsonl
runs/real/nq/B10/test.responses.summary.json
```

绘图：

```bash
python3 plot_results.py \
  runs/real/nq/B10/events.jsonl \
  --out runs/real/nq/B10/figures
```
