# lunwen
第一篇论文尝试。

## DisarmCache 实验

`experiments/disarmcache/` 保存了检索器投毒与语义缓存协同研究的离线可审计验证器、实验计划和 smoke 结果。运行：

```bash
cd /data1/lab409/wwj/lunwen-main/experiments/disarmcache
./run_smoke.sh
```

服务器从 GitHub 下载后，项目根目录名是 `lunwen-main`。如果实际上传位置不是
`/data1/lab409/wwj`，只需把命令中的这一段改成服务器上的实际父目录；项目内部目录名和代码不需要改动。

`references/` 保存论文初稿和两份上游源码压缩包。上游源码用于复现与接口对照；当前 smoke 结果使用合成数据，不代表真实模型实验结果。

新的缓存库查看、`retrieval-first` 路径、数据集批量运行、提示词消融和干净/编辑检索器对比流程见
[`experiments/disarmcache/EXPERIMENT_RETRIEVAL_FIRST.md`](experiments/disarmcache/EXPERIMENT_RETRIEVAL_FIRST.md)。

B0–B9 配置和可扩展数据集目录见
[`experiments/disarmcache/DATASET_LAYOUT.md`](experiments/disarmcache/DATASET_LAYOUT.md) 和
`experiments/disarmcache/config/experiments.json`。
