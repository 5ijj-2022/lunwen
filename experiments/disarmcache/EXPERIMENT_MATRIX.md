# 论文实验执行矩阵

真实服务器上固定同一批问题、同一生成模型、同一预算和随机种子，依次运行：

| 编号 | 配置 | 用途 |
|---|---|---|
| B0 | LLM only | 原始错误率 |
| B1 | Clean RAG | 干净检索质量 |
| B2 | Clean RAG + cache | 正常缓存效用 |
| B3 | DisarmRAG edited retriever | 检索器投毒效果 |
| B4 | Clean retriever + cache collision | 缓存机制效果 |
| B5 | B3 + 普通缓存 | 朴素串联 |
| B6 | B3 + provenance cache | 本文方法 |
| B7 | B6 + dependency invalidation | 防御和恢复 |

每个配置都保存 `events.jsonl`。运行完成后生成图表：

```bash
PYTHONPATH=. python3 plot_results.py runs/real/events.jsonl --out runs/real/figures
```

论文中报告 Overall-ASR、Hit-ASR、Miss-ASR、Hit Rate、PHR、Seed-ASR、正常准确率、p95 延迟、生成调用数和恢复残留率。每个指标保留请求数、目标数和随机种子；不同问题改写按目标分组计算置信区间。

阈值实验将 `--threshold` 依次设为 `0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95`，每个阈值使用独立缓存目录。消融实验分别关闭 BI、检索器编辑、缓存感知选择和 provenance；恢复实验分别执行保留缓存、全量清空和按 `retriever_version` 撤销。

文件职责：`core.py` 提供离线对象、缓存和指标；`cli.py` 运行可复核 smoke；`integrated_service.py` 连接 DisarmRAG loader、真实生成模型、独立缓存编码器和 HTTP API；`plot_results.py` 从事件日志绘图；`REAL_RUN.md` 是服务器启动说明；`EXPERIMENT_PLAN.md` 是论文设计与公平比较约束。
