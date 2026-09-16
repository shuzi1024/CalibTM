# 2026-09-16 联合 SPIN／Direct／Sync 研究

## 当前证据

- [中性外部评审入口](../../review_packet/spin_direct_sync_20260916/START_HERE.md)，先读事实，再讨论已有研究判断。
- [最终精度与成本表](FINAL_RESULTS_CN.md)、[机器可读表](FINAL_RESULTS.json)。
- [八轨迹逐种子汇总](summary_fixed120/SUMMARY_CN.md)、[完整聚合数据](summary_fixed120/SUMMARY.json)。
- [成本汇总](inference_summary/INFERENCE_SUMMARY_CN.md)、[720 个原始计时样本](INFERENCE_BENCHMARK.json)。
- [简明方法与协议](METHOD_AND_SCOPE_CN.md)、[实现公式](METHOD_CN.md)、[评价范围审计](HOLDOUT_AUDIT_CN.md)。
- [完整结果审计](FINAL_RESULT_AUDIT_CN.md)、[成本审计](INFERENCE_AUDIT_CN.md)。
- [独立 ICC 审阅](INDEPENDENT_ICC_REVIEW_CN.md)、[论文科学复核](PAPER_SCIENTIFIC_REVIEW_CN.md)、[研究结项](OUTCOME_CN.md)。
- [中文初稿与构建说明](../../paper/spin_context_direct_cn_20260916/README_CN.md)。

## 原始记录与范围

`raw_outputs/` 保存早停父阶段，`raw_fixed120/` 保存同轨迹接续后的 result/history/config/status；两个阶段不是独立重复。当前共有两 WAN×两变体×两初始化八条完整 120 轮轨迹。旧参考记录在 `../research_7h_20260916/references/`。这里只复用该目录下的参考结果，不把早期残差头模型作为本轮模型。

结果是开发评价；旧参考配方不同；到达轮数上限不等于充分优化。训练曲线位于 `figures/`。SIX_JOB_AUDIT、STRUCTURAL_REVIEW 及固定轮数／第二种子说明是历史阶段记录，以最终完整结果为准。

本发布包含源码与保存的结果、聚合统计和误差记录，不包含训练权重、原始流量数组、机器环境、TeX 二进制、运行锁或后台启动队列。训练器仍会验证历史 registry、数据及基线检查点；代码依赖完整不表示克隆后可直接重跑完整历史。已有结果哈希保留原始身份，源码与模型未经重新训练或评分。

原记录可能含研究机器路径、主机标识、旧截止时间或未随包发布的历史路径，属于来源记录。当前发布清单与逐文件校验值见仓库根目录 `PUBLICATION_MANIFEST_20260916.json`；不使用旧本地交付清单替代当前文件身份。
