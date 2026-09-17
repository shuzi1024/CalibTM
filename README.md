# CalibTM：网络流量矩阵补全研究

## 当前状态：SPIN＋Direct 路径对照完成，两步细化结束

本次更新整理 2026-09-17 完成的两轮实验。主候选为**一层 SPIN 时空上下文＋本流双向真实观测读出**，不包含 Sync 或 LLM。两套 WAN 的单路径对照支持保留组合；随后固定两步细化未通过预定筛选，已停止该变体。

- **外部评审入口**：[START_HERE](review_packet/spin_direct_update_20260918/START_HERE.md)，或直接阅读[方法与完整结果合并页](review_packet/spin_direct_update_20260918/REVIEW_WEB_CN.md)。
- **组件实验**：[完整数值表](analysis/spin_path_controls_20260917/DATA_TABLES_CN.md)、[研究结论](analysis/spin_path_controls_20260917/OUTCOME_CN.md)。
- **两步细化**：[完整结果](analysis/spin_refine_20260917/report_geant/REPORT_CN.md)、[停止决定](analysis/spin_refine_20260917/DECISION.json)。
- **最新中文内部稿**：[五页 PDF](paper/spin_direct_path_controls_cn_20260917/build/manuscript.pdf)、[LaTeX 与构建说明](paper/spin_direct_path_controls_cn_20260917/README_CN.md)。

## 已完成的证据

路径对照共12条轨迹：两套 WAN × 三模型 × 两个初始化。其中8条单路径从头训练，4条组合从已有120轮末状态接续；全部达到160轮，保留120轮截面。不同轮数属于同一训练轨迹，不增加独立重复。

下表为三条件等权、两初始化平均的开发 NMAE，越低越好。

| 模型 | Abilene | GEANT |
|---|---:|---:|
| SPIN＋Direct | 0.161358 | 0.164810 |
| Context-only | 0.164053 | 0.168494 |
| Direct-only | 0.165011 | 0.172246 |

组合相对 Context-only 的 NMAE 改善约1.64%／2.19%，同卡 B8 每窗摊销耗时增加约3.82%／3.47%。相对 Direct-only 的 B8 耗时则为11.79／16.45倍；不能概括成组合同时具有最优精度与最低成本。两个种子、三个条件和末段轨迹均支持组合的主指标方向，辅助指标及掩码子组的例外完整保留。

固定两步细化在 GEANT 新训练4条轨迹，各160轮。匹配的一步／两步具有相同参数和配对初值；两步共享权重，仅末步监督，Direct始终仅读真实观测。平均 NMAE 从一步的0.164971变为两步的0.170527，劣化3.37%；B8耗时为1.95倍。两个种子和末段中位数均未通过筛选，因此未扩展到Abilene。这只评价本次固定实现，不是 ARI-LLM 多分辨率或 LLM 方法的复现。

## 任务范围与解释边界

本研究是具有完整历史监督数据的**离线补全**。固定50槽窗口，当前输入保留20%真实观测，使用均匀、不均匀、共享缺口三种条件；每流至少4个观测。完整拟合段用于监督、统计量和相关图。主指标是NMAE，NRMSE为辅助指标。

所有新分数均来自参与检查点选择和研究决策的开发范围。历史后续数据也曾被项目使用；新掩码不能恢复流量值的独立测试身份。GEANT处理后快照的原生时间轴和单位映射仍未核实。160轮是固定预算，不等于充分收敛；末段轨迹不是独立训练重复。

Context-only与Direct-only是当前骨架的路径对照，不是等有效容量控制，也不替代正常工作的外部SPIN基线。公平外部方法确认、评价来源核实及完整可运行的复现入口仍待完成；当前稿件属于内部研究稿。

## 代码与结果入口

| 内容 | 入口 |
|---|---|
| SPIN＋Direct 原骨架及历史 Full | [models.py](experiments/spin_sync_unified_v1/models.py) |
| 匹配路径模型／训练 | [models.py](experiments/spin_path_controls_v1/models.py)、[runner.py](experiments/spin_path_controls_v1/runner.py) |
| 固定一步／两步模型／训练 | [models.py](experiments/spin_refine_v1/models.py)、[runner.py](experiments/spin_refine_v1/runner.py) |
| 原始逐轮与逐条件记录 | [路径对照](analysis/spin_path_controls_20260917/raw_outputs)、[两步细化](analysis/spin_refine_20260917/raw_outputs) |
| GEANT／Abilene 汇总 | [GEANT](analysis/spin_path_controls_20260917/summary/SUMMARY.json)、[Abilene](analysis/spin_path_controls_20260917/summary_abilene/SUMMARY.json) |
| 既有评审（可后读） | [路径对照模拟审阅](analysis/spin_path_controls_20260917/FINAL_REVIEW_CN.md) |
| 历史联合记忆实验 | [2026-09-16 完整记录](review_packet/spin_direct_sync_20260916/START_HERE.md) |

## 发布与复现范围

本次提供源代码、已保存的指标／误差聚合／训练历史、检查记录、曲线以及论文。**不含原始流量数组、模型权重、机器凭据或编译器缓存。** 检查点哈希和历史路径是来源记录，不表示这些资产已经随仓库公开。某些归档清单描述作者工作区中曾核验的完整文件集，公开文件子集见[发布清单](PUBLICATION_MANIFEST_20260918.json)。

训练环境为Python 3.10.18、PyTorch 2.2.2+cu121、NumPy 1.22.4；神经计时在H20上使用FP32并关闭TF32。训练入口仍依赖固定数据身份、历史注册表及相关资产；克隆源码和安装依赖不足以重跑全部历史结果。根[依赖文件](requirements.txt)还包含较早ARI分支的需求，论文编译另需LaTeX引擎与中文字体。

本次更新不改变原有许可：[LICENSE](LICENSE)与[第三方声明](THIRD_PARTY_NOTICES.md)均保留。SPIN及Torch Spatiotemporal的复用遵循各自MIT声明。其他旧目录、过期授权与运行记录均为历史材料。
