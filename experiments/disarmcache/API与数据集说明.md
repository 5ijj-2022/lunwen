# API 与数据集方案

当前实现支持两类生成器：服务器本地 HuggingFace 模型，以及 OpenAI 兼容聊天 API。代码此前没有实际调用 API；真实主实验应由服务器设置密钥后运行。

## 推荐模型分工

主结果建议固定一个 API 生成模型，例如 `gpt-4o-mini` 或你可用的同等级模型；迁移实验再使用第二个模型。生成温度固定为 0。API 只负责 RAG 最终回答，检索器仍使用本地 Contriever/DisarmRAG checkpoint，缓存编码器独立使用本地 Sentence-Transformers 模型。这样每次请求都能记录检索、缓存和生成的路径。

BI 候选优化如果复现 DisarmRAG 原论文，可以使用其配置的代理模型；这些调用属于离线攻击构造阶段，必须和最终测试问题分开，并单独记录调用次数和 token 成本。不要让 API 读取最终测试改写后再调参。

## API 启动

在服务器环境设置 `OPENAI_API_KEY`，不要写入命令行、代码、JSONL 或 Git：

```bash
export OPENAI_API_KEY='服务器安全注入的值'
cd /data1/lab409/wwj/lunwen-main/experiments/disarmcache
PYTHONPATH=. python3 -m disarmcache.integrated_service \
  --retriever-model /models/contriever-edited \
  --retriever-backend disarmrag \
  --cache-model sentence-transformers/all-MiniLM-L6-v2 \
  --generator-backend openai \
  --generator-model gpt-4o-mini \
  --corpus data/nq_subset.jsonl \
  --retriever-version disarmrag-edit-001 \
  --device cuda --threshold 0.80 \
  --http-host 0.0.0.0 --http-port 8080
```

如果使用兼容服务，增加 `--api-base-url https://兼容服务地址/v1`。日志只保存模型名称、请求 ID、路径和 token 用量摘要；不保存密钥。

## 数据集用途

`Natural Questions (NQ)` 是主事实问答集，用于目标事实、改写和目标错误判定；建议先取 20–30 个目标做链路试验，再做 100 个目标的正式结果。`HotpotQA` 用于多跳外部验证，单独报告，不能和 NQ 未分层混平均。`BEIR` 用于检查编辑检索器对正常检索质量的影响，报告 nDCG@10 和 Recall@100。另建一个小型合成事实集，用于准确验证错误答案、相似但答案不同问题和恢复撤销。

正式数据应分为编辑训练集、阈值/防御开发集和最终测试集。测试改写不能进入 API 调参、缓存阈值选择或 BI 优化。每个 JSONL 文档至少包含 `id`、`text`、`poisoned`；问题文件包含 `target_id`、`question`、`paraphrase_id`、`gold_answer`、`target_wrong_answer`。

## 成本控制

先用 20 个目标和 3 个改写验证 API 线路，再扩大到主实验规模。所有 B0–B7 使用相同问题数和生成预算，缓存命中请求不调用生成 API；日志统计 API 调用数和输入/输出 token，论文报告成本和 p95 延迟。
