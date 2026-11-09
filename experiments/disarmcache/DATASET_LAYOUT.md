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
