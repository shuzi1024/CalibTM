# CalibTM：网络流量矩阵补全研究

本次发布整理了 **2026-09-16 联合 SPIN／Direct／Sync 模型**的源码、完整实验结果和中文内部论文初稿。两套 WAN、两个变体、两个初始化，共八条 120 轮训练轨迹，以及 48 格推理成本测量均已完成。

## 阅读入口

- **独立研究咨询**：[从这里开始](review_packet/spin_direct_sync_20260916/START_HERE.md)。先阅读中性方法说明、完整结果及曲线、源码和原论文来源，再按需阅读已有论文与评审。
- **结果与复核材料**：[本轮研究索引](analysis/spin_sync_unified_20260916/README_CN.md)、[最终精度与成本表](analysis/spin_sync_unified_20260916/FINAL_RESULTS_CN.md)、[机器可读结果](analysis/spin_sync_unified_20260916/FINAL_RESULTS.json)。
- **中文内部初稿**：[四页 PDF](paper/spin_context_direct_cn_20260916/build/manuscript.pdf)、[LaTeX 与构建说明](paper/spin_context_direct_cn_20260916/README_CN.md)。采用 IEEE 会议双栏样式，属于内部研究稿。
- **已有研究判断**：[独立 ICC 视角审阅](analysis/spin_sync_unified_20260916/INDEPENDENT_ICC_REVIEW_CN.md)。独立咨询时可在形成自己的判断后再阅读。

## 当前模型与实验范围

Full 从头联合训练一层 32 维 SPIN 时空上下文、Direct 本流前后最近真实观测直读，以及四头双向同步矩阵记忆。每方向将 Direct 与记忆读出相加，再与上下文查询拼接解码。等系数相加不表示两路贡献相同。去 Sync 变体保留 SPIN、Direct、查询和解码器，使用相同共有参数初值独立训练。

任务为 **50 槽窗口、20% 观测率下的离线补全**，考察均匀、不均匀和共享缺口三种条件。NMAE 为主指标，NRMSE 为辅助。新增 Sync 后，两 WAN 的平均开发 NMAE 分别变化 +0.0382% 和 +1.4063%，NRMSE 分别变化 −0.8592% 和 −1.2402%；B8 摊销每窗耗时分别增加约 64.4% 和 62.2%。完整结果保留旧 SPIN、旧 Direct＋Sync 和 Linear 参考。

这些分数来自用于选取检查点的开发范围。旧参考的层数、训练配方和重复次数不同；固定 120 轮是同一轨迹的训练时长诊断，不能视为充分优化或独立最终测试。GEANT 使用固定版本的处理后快照，其原生时间戳和物理单位映射尚未核实。

## 主要源码

| 内容 | 入口 |
|---|---|
| 联合模型及去 Sync 变体 | [models.py](experiments/spin_sync_unified_v1/models.py) |
| 初始训练与开发评价 | [runner.py](experiments/spin_sync_unified_v1/runner.py) |
| 固定 120 轮接续 | [fixed120 runner](experiments/spin_sync_fixed120_v1/runner.py) |
| 数据范围、掩码与指标实现 | [data.py](experiments/sync_delta_v1/data.py)、[engine.py](experiments/sync_delta_v1/engine.py) |
| SPIN 任务适配与上游来源 | [SPIN 说明](experiments/spin_comparison_v1/README_CN.md) |
| 完整推理成本测量 | [benchmark_inference.py](analysis/spin_sync_unified_20260916/benchmark_inference.py) |

发布内容包含训练与测量入口所需的本地模型包。依赖中保留的旧候选名称反映代码复用；当前联合模型本身没有 LLM，不需要加载 GPT-2。

## 环境与重现范围

这是**源代码和结果快照**。原始流量数据、模型权重、历史运行资产、机器缓存和编译器二进制不随本次发布提供。已完成实验使用 Python 3.10.18、PyTorch 2.2.2+cu121、NumPy 1.22.4；推理测量使用 NVIDIA H20、FP32 并关闭 TF32。

训练入口还校验历史 registry、基线结果、checkpoint 和固定数据身份；接续入口需要父阶段的模型、优化器与随机状态。因此，克隆仓库和安装依赖并不能直接完成全部重跑，重新下载同名数据也不自动满足这些绑定。历史记录中的本机路径用于说明来源，不是通用执行路径。

根目录 [requirements.txt](requirements.txt) 同时包含较早 ARI-LLM 分支的依赖。各实验的环境、数据和资产要求应结合对应说明核对；论文编译另需 LaTeX 引擎及中文字体，见论文目录说明。

## 目录与历史内容

- `experiments/`：当前模型、复用实现及部分历史实验源码。
- `analysis/spin_sync_unified_20260916/`：本轮结果、成本与复核材料。
- `paper/spin_context_direct_cn_20260916/`：本轮中文论文及构建说明。
- `review_packet/spin_direct_sync_20260916/`：用于网页阅读或文件上传的独立咨询材料。
- `Imputation/`、`plan/`、`setup/` 及其他旧目录保留原项目内容；其中 ARI-LLM、value_only、GapCalib 等方向属于历史研究，不代表本轮待执行计划。

## 来源与许可

仓库保留原有 [LICENSE](LICENSE)。SPIN 和 Torch Spatiotemporal 的复用及改编部分另保留各自的 MIT 版权和许可声明；根目录许可不替代第三方声明。方法来源、固定上游版本及任务适配见 SPIN 说明与论文引用。
