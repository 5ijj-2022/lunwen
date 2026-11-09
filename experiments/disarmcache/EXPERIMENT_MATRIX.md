# 论文实验执行矩阵

真实服务器上固定同一批问题、同一生成模型、同一预算和随机种子，依次运行：

| 编号 | 检索器 | 缓存顺序 | 缓存键 | 用途 | 当前状态 |
|---|---|---|---|---|---|
| B0 | 无 | 无 | 无 | LLM-only | 手动基线 |
| B1 | 干净 | 无 | 无 | 正常 RAG | 可运行 |
| B2 | 干净 | cache-first | query | 普通 RAG + 缓存 | 可运行 |
| B3 | 干净 | retrieval-first | query | 检索后缓存 | 可运行 |
| B4 | 干净 | retrieval-first | query+documents | NDS 风格正常缓存 | 计划 |
| B5 | 编辑后 | 无 | 无 | 检索器投毒 | 等待 checkpoint |
| B6 | 编辑后 | cache-first | query | 朴素串联 | 等待 checkpoint |
| B7 | 编辑后 | retrieval-first | query | 检索器 + 语义缓存 | 等待 checkpoint |
| B8 | 编辑后 | retrieval-first | query+documents | DisarmCache 主方法 | 计划 |
| B9 | 编辑后 | provenance | query+documents | 防御和撤销 | 计划 |

每个配置都保存 `events.jsonl`。运行完成后生成图表：

```bash
PYTHONPATH=. python3 plot_results.py runs/real/events.jsonl --out runs/real/figures
```

新增请求顺序对比：同一检索器、语料、阈值和问题集合分别使用
`--query-order cache-first` 与 `--query-order retrieval-first`。后者每次先计算检索结果，
再查缓存，返回中同时保留 `retrieval.documents` 和 `cache.matched_entry`（miss 时为
`cache.admitted_entry`）。

缓存库查看接口：

```bash
curl -sS http://127.0.0.1:8080/cache
curl -sS -X POST http://127.0.0.1:8080/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"question"}'
```

论文中报告 Overall-ASR、Hit-ASR、Miss-ASR、Hit Rate、PHR、Seed-ASR、正常准确率、p95 延迟、生成调用数和恢复残留率。每个指标保留请求数、目标数和随机种子；不同问题改写按目标分组计算置信区间。

阈值实验将 `--threshold` 依次设为 `0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95`，每个阈值使用独立缓存目录。消融实验分别关闭 BI、检索器编辑、缓存感知选择和 provenance；恢复实验分别执行保留缓存、全量清空和按 `retriever_version` 撤销。

文件职责：`core.py` 提供离线对象、缓存和指标；`cli.py` 运行可复核 smoke；`integrated_service.py` 连接 DisarmRAG loader、真实生成模型、独立缓存编码器和 HTTP API；`run_dataset.py` 批量运行带 gold 标签的问题；`prepare_dataset.py` 按 target 分组划分数据；`prompt_experiment.py` 运行提示词变体；`retriever_compare.py` 比较干净与编辑检索器；`plot_results.py` 从事件日志绘图；`REAL_RUN.md` 是服务器启动说明；`EXPERIMENT_PLAN.md` 是论文设计与公平比较约束。
