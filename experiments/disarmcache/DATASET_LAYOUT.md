# 数据集目录和扩展规范

核心服务只接收两个 JSONL 输入：问题文件和文档语料文件。新增数据集时，在 `data/datasets/`
下新增一个数据集目录，不修改 `integrated_service.py`。

推荐目录：

```text
data/datasets/
├── nq/
│   ├── raw/
│   ├── questions/
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   └── test.jsonl
│   ├── corpus/
│   │   ├── clean.jsonl
│   │   └── poisoned.jsonl
│   └── metadata.json
├── hotpotqa/
│   ├── questions/
│   ├── corpus/
│   └── metadata.json
├── beir/
│   ├── questions/
│   ├── corpus/
│   └── metadata.json
└── synthetic/
    ├── questions/
    ├── corpus/
    └── metadata.json
```

问题记录至少包含：

```json
{"target_id":"nq-001","paraphrase_id":"nq-001-p1","question":"...","gold_answer":"...","target_wrong_answer":"...","split":"test"}
```

文档记录至少包含：

```json
{"id":"doc-001","text":"...","poisoned":false}
```

`target_wrong_answer` 只有在真实投毒目标已经定义时才填写。没有真实错误目标时留空，不能用
人工编造的错误答案冒充攻击结果。

小规模 NQ 子集可以暂时放在 `data/nq_subset/`；正式实验建议迁移到
`data/datasets/nq/`，并在 `metadata.json` 中记录数据集版本、来源、样本数和随机种子。

每个实验都通过 `--corpus` 和 `--questions` 指定输入，不把数据集路径写死在代码中。例如：

```bash
--corpus data/datasets/nq/corpus/clean.jsonl
--questions data/datasets/nq/questions/test.jsonl
```

## 下载并转换 BEIR/NQ

DisarmRAG 上游项目使用 BEIR 的 NQ、MS MARCO 和 HotpotQA 下载地址。先安装解压工具，创建原始
数据目录并断点下载 NQ：

```bash
cd /data1/lab409/wwj/lunwen-main-git/experiments/disarmcache
conda activate disarmcache
export DATA_ROOT="$PWD/data/datasets"
mkdir -p "$DATA_ROOT/raw/beir"
cd "$DATA_ROOT/raw/beir"

wget -c --tries=10 --timeout=120 \
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
unzip -q nq.zip
```

确认原始文件存在：

```bash
test -s "$DATA_ROOT/raw/beir/nq/corpus.jsonl"
test -s "$DATA_ROOT/raw/beir/nq/queries.jsonl"
test -s "$DATA_ROOT/raw/beir/nq/qrels/test.tsv"
```

第一次建议先做一个可运行的小子集：保留 100 个问题、每个问题的相关文档和 10,000 个干扰文档。
这一步不会修改原始下载文件：

```bash
PYTHONPATH=. python3 prepare_beir_dataset.py \
  --source "$DATA_ROOT/raw/beir/nq" \
  --dataset nq \
  --out "$DATA_ROOT/nq" \
  --split test \
  --query-limit 100 \
  --distractors 10000 \
  --seed 12
```

转换后应有：

```text
data/datasets/nq/
├── corpus/clean.jsonl
├── questions/test.jsonl
└── metadata.json
```

BEIR 的 qrels 只提供“相关文档 ID”，不提供 QA 答案字符串。因此这一步可以直接测检索、缓存命中、
延迟和生成调用；要报告答案准确率，必须再把经过核验的 NQ gold answer 合并到问题文件，不能
人工编造答案。

正式实验可以增加规模：

```bash
PYTHONPATH=. python3 prepare_beir_dataset.py \
  --source "$DATA_ROOT/raw/beir/nq" \
  --dataset nq \
  --out "$DATA_ROOT/nq-full" \
  --split test \
  --query-limit 0 \
  --distractors -1 \
  --seed 12
```

完整 NQ 语料会占用大量磁盘、内存和 GPU 时间。先完成 100 问题子集的链路验证，再运行完整语料。
`nq-full` 是另一个数据集目录，避免覆盖前面的可复现实验。

其他 BEIR 数据集只需替换压缩包名称和输出数据集名称：

```bash
wget -c --tries=10 --timeout=120 \
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip
unzip -q hotpotqa.zip

PYTHONPATH=. python3 prepare_beir_dataset.py \
  --source "$DATA_ROOT/raw/beir/hotpotqa" \
  --dataset hotpotqa \
  --out "$DATA_ROOT/hotpotqa" \
  --split test \
  --query-limit 100 \
  --distractors 10000 \
  --seed 12
```

## 固定 B2/B3 逻辑，只更换数据集

B2 和 B3 的实验逻辑已经封装在两个通用入口中，不需要为 NQ、HotpotQA 或其他数据集复制
Python 文件。目录名只表达两个独立维度：数据集名称和实验编号。

```text
runs/real/
├── nq/
│   ├── B2/
│   └── B3/
├── hotpotqa/
│   ├── B2/
│   └── B3/
└── synthetic/
    ├── B2/
    └── B3/
```

先启动服务。服务启动时会读取语料，因此更换数据集后需要重新启动服务：

```bash
export PROJECT_ROOT=/data1/lab409/wwj/lunwen-main-git
cd "$PROJECT_ROOT/experiments/disarmcache"
chmod +x start_experiment.sh run_experiment.sh

export MODEL_DIR=/data1/lab409/wwj/models
export DATA_ROOT="$PWD/data/datasets"
export RUN_ROOT="$PWD/runs/real"
export GENERATOR_BACKEND=openai
export GENERATOR_MODEL=gpt-4o-mini
export API_BASE_URL="$BASE_URL"       # 例如 https://4sapi.org/v1
export API_KEY_ENV=API_SECRET_KEY       # 与服务器中保存密钥的变量一致

./start_experiment.sh nq B2 8080
```

B3 使用相同入口，只改变实验编号和端口：

```bash
./start_experiment.sh nq B3 8081
```

入口自动固定 `B2=cache-first`、`B3=retrieval-first`，并分别写入：

```text
runs/real/nq/B2/cache.jsonl
runs/real/nq/B2/events.jsonl
runs/real/nq/B3/cache.jsonl
runs/real/nq/B3/events.jsonl
```

在另一个终端批量发送测试问题：

```bash
cd "$PROJECT_ROOT/experiments/disarmcache"
export DATA_ROOT="$PWD/data/datasets"
export RUN_ROOT="$PWD/runs/real"

./run_experiment.sh nq B2
./run_experiment.sh nq B3 http://127.0.0.1:8081/query
```

换数据集时只修改 `nq`，并把文件放到对应目录：

```text
data/datasets/nq/corpus/clean.jsonl
data/datasets/nq/questions/test.jsonl
data/datasets/hotpotqa/corpus/clean.jsonl
data/datasets/hotpotqa/questions/test.jsonl
```

例如运行 HotpotQA 的 B3：

```bash
./start_experiment.sh hotpotqa B3 8081
./run_experiment.sh hotpotqa B3 http://127.0.0.1:8081/query
```

不要在不同数据集或不同实验编号之间复用缓存文件。每个组合都有独立的缓存、事件日志和批量
响应文件，后续可以直接用 `plot_results.py` 分别统计或汇总。
