# lunwen
第一篇论文尝试。

## DisarmCache 实验

`experiments/disarmcache/` 保存了检索器投毒与语义缓存协同研究的离线可审计验证器、实验计划和 smoke 结果。运行：

```bash
cd experiments/disarmcache
./run_smoke.sh
```

`references/` 保存论文初稿和两份上游源码压缩包。上游源码用于复现与接口对照；当前 smoke 结果使用合成数据，不代表真实模型实验结果。
