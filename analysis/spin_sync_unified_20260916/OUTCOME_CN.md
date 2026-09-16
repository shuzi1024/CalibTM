# SPIN／Direct／Sync 研究结项

完成时间：2026-09-16 17:14:39 北京时间。

## 交付结论

本轮既定八条 120 轮训练轨迹、48 格统一推理成本测量、独立结果审计、ICC 视角审阅及四页中文 LaTeX 初稿均已完成。实验未扩展为新的模块、观测率或超参数搜索。

当前 Full 是一层 SPIN 上下文＋Direct 观测直读＋双向 Sync 矩阵记忆，全部从头联合训练。每方向 Direct 与 Sync 各 128 维，等系数 1:1 相加；这不等于预测贡献各 50%。Direct 自身也读取前后观测。Direct 独有参数 1,440，新增 Sync K/V 参数 24,960，其余主要为共享模块。

## 结果与方向

相对独立训练的 SPIN＋Direct，Full 的平均开发 NMAE 在 Abilene 增加 0.0382%，GEANT 增加 1.4063%；平均 NRMSE 分别降低 0.8592% 和 1.2402%。批量为 8 时摊销每窗耗时增加约 64.4% 和 62.2%。当前证据支持优先推进较简化的 SPIN＋Direct，完整保留 Full 的 NRMSE／成本取舍。

[独立 ICC 审阅](INDEPENDENT_ICC_REVIEW_CN.md) 对当前投稿说服力倾向 **Weak Reject**，同时认为预定问题已经有足够证据形成诚实内部初稿。当前数据是开发评价，Direct 的独立增量尚未隔离；旧参考配方不同，固定 120 轮不表示充分优化。后续若继续投稿研究，应优先补齐具体主张所需的最小证据，而非继续堆叠结构或实验矩阵。本轮不追加运行。

## 主要产物

- [中文初稿 PDF](../../paper/spin_context_direct_cn_20260916/build/manuscript.pdf)、[LaTeX 与构建说明](../../paper/spin_context_direct_cn_20260916/README_CN.md)：四页、IEEE 会议双栏、图表与字体已检查。
- [最终精度与成本表](FINAL_RESULTS_CN.md)、[机器可读结果](FINAL_RESULTS.json)。
- [八轨迹结果审计](FINAL_RESULT_AUDIT_CN.md)、[成本审计](INFERENCE_AUDIT_CN.md)、[论文科学复核](PAPER_SCIENTIFIC_REVIEW_CN.md)。
- [当前方法与最小范围](METHOD_AND_SCOPE_CN.md)、[详细汇总](summary_fixed120/SUMMARY_CN.md)。
- [GPU 关闭记录](GPU_CLOSEOUT.json)、[本次发布文件说明](README_CN.md)。

全部 GPU 训练于 16:48:06 北京时间结束，16:50:55 已核实两台主机没有 GPU 计算进程、本轮任务和本地队列均退出，早于 19:33:29 授权截止。没有新任务排队。记录的各轮训练与评估合计 12.1968 小时，不包括准备、IO、smoke、测速等开销，不作为完整占用计费量。旧 481 参数头论文保持原样。
