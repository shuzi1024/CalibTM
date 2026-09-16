# 当前 Abilene / GEANT 评价范围审计

审计日期：2026-09-16。对象：本轮联合训练的 SPIN–Sync–Direct 模型是否还存在可确认未使用的后续最终验证范围。

**结论：在当前已固定的 Abilene 48384 行、GEANT 10772 行源序列内，没有确认到可作为本轮全新独立最终验证的后续范围。** 两个历史 gate 已经在早期 K3 研究中使用，又在 9 月 16 日凌晨实际完成冻结 SPIN 适配头评价并被报告；两个更晚的完整尾段已于 9 月 15 日实际完成全部 20 项计划评价，且进入 Direct–Sync 论文及后续研究判断。因此，本轮新模型尚未在这些范围运行，不等于这些范围尚未影响项目选型。此结论有肯定的访问记录支持，并非从“未发现文件”推断。

本次仅阅读现有代码、协议、注册表、完成标记、元数据和已生成报告；没有读取新评价样本、生成新分数、运行模型或使用 GPU。

## 1. 范围总表

以下均为固定源序列中的零基、左闭右开绝对行号；窗口长 T=50。序列行号不额外声称对应连续日历时间。

| 角色 | Abilene，144 flows | GEANT，462 flows | 已确认用途与本轮可用性 |
| --- | --- | --- | --- |
| fit | [0,33335) | [0,7023) | 训练窗口来源；完整 fit 用于均值/尺度与相关邻居估计，不能把未采到的训练窗口间隙重包装为测试。 |
| train gap | [33335,33384)，49 行 | [7023,7072)，49 行 | 原协议隔离带；单独不足一个 T50 窗口，且位于历史开发段之前。无证据支持改称最终测试。 |
| source_dev | [33384,33884)，10 窗 | [7072,7572)，10 窗 | checkpoint/开发评价组成部分，已有多轮结果。 |
| tune | [33884,37484)，72 窗 | [7572,8322)，15 窗 | checkpoint/模型选择组成部分，已有多轮结果。 |
| validation gap | [37484,37534)，50 行 | [8322,8372)，50 行 | 原协议隔离带；本审计不宣称其从未被任何历史代码访问。即使保持未评分，也各仅一窗，夹于 tune 与已消费 gate 之间，不构成有说服力的后续最终确认集。 |
| historical gate | [37534,41134)，72 窗 | [8372,9172)，16 窗 | 早期 K3 final gate；2026-09-16 再次完整评价并报告，明确 historical_gate_reuse / independent_test=false。 |
| later temporal tail | [41134,48384)，145 窗 | [9172,10772)，32 窗 | 2026-09-15 完整评价；覆盖当前固定源的全部剩余行，已消费。 |

开发评价合计分别 82 / 25 窗。范围定义来自 `experiments/sync_delta_v1/data.py:28`、`experiments/acil_innovation_v1/configs/protocol_v1.json`、`experiments/anchorcv_v1/gate_data.py` 及 `experiments/final_temporal_eval_v1/data.py:12`。当前 `final_temporal_eval_v3` 复用 v1 的尾段数据定义。

## 2. 后续完整尾段已经实际评价

主要证据目录：

`analysis/overnight_20260915/temporal_v3_completed_archive/`

- `registry.json` / `registry.sha256` 固定 20 项，两 WAN 各 10 项，包括 Linear、Direct–Sync、Sync、Direct-only、SPIN、ARI 与 Anchor 变体。注册表 SHA256：`cd3d9211a11b91695f326c8985179835e034b83f533396dd84a77d4a9cacee60`。
- `report/comparison.json` 明确 `state=complete, completed=20, total=20, pending=[]`。
- `abilene/FIRST_TAIL_ACCESS.json` 记录 2026-09-15 **07:22:55 北京时间**开始；`abilene/AVAILABLE_COMPLETE.json` 记录 **07:23:58**完成全部 10 项。其数据身份为 145 个窗口，起点 41134 至 48334、步长 50，完整覆盖 [41134,48384)。
- `geant/FIRST_TAIL_ACCESS.json` 记录 2026-09-15 **07:22:55 北京时间**开始；`geant/AVAILABLE_COMPLETE.json` 完成前 9 项；`geant/DEFERRED_COMPLETE.json` 记录当日 **18:57:58**完成最后 SPIN 项。数据身份为 32 个窗口，起点 9172 至 10722、步长 50，完整覆盖 [9172,10772)。GEANT 最后一个 status 仅表示延期项完成，不能据此误认为只有一个方法访问过尾段。
- `paper/direct_sync/manuscript.tex:91` 起明确记载 145/32 个后续窗口与 20 项全部完成；正文同时讨论这些结果及方法限制。因此这些分数已进入研究和写作，而非仅存放未读。

旧角色 `postfreeze_temporal_holdout_legacy_access_unverified` 描述的是当时冻结后的评价，且当时就未认证项目全历史独立性。即使假设更早没有访问，本轮模型决策发生于这些结果公开之后，也不能继承该次冻结之前的未见状态。`FIRST_TAIL_ACCESS` 文件名同样不构成项目所有早期工作的绝对首次访问证明。

## 3. 历史 gate 也有当日完成证据

主要证据：

- `analysis/research_7h_20260916/evaluate_gate.py:25`：固定 A [37534,41134)、G [8372,9172)，角色 `historical_gate_reuse`。
- `outputs/historical-gate-reuse-20260916/{abilene,geant}/evaluation_info.json`：记录绝对范围、72/16 窗、T50、三个缺失条件、mask seed 71001，以及 `independent_test=false`；来源说明明确写出旧 K3 final gate 已用过。
- 同目录 `status.json`：Abilene 于 **2026-09-16 03:16:38 北京时间**完成，GEANT 于 **03:16:35**完成。
- 当次冻结清单 `analysis/research_7h_20260916/FROZEN_SELECTION.json`，对应 SHA256 `a37101606d3ce967af7611593e87e20c2076b95a9ce992b31512e32680edd8e0`。
- `analysis/research_7h_20260916/FINAL_RESULTS_CN.md` 于 03:26 北京时间生成，明确 12/12 个适配头和两个 gate 报告完成；包含平均、分条件、分 seed、支持组等诊断。`paper/icc_cn_20260916/manuscript.tex` 及其输入章节采用了这批后续段结果。

当次 `selection_on_gate=false` 仅表示适配头的权重清单在该次评价前冻结，不能推出后来新模型设计未受这些结果影响。`source_dev/tune` 之后的父 validation 数组剩余部分，大部分正是上述 gate，不是尚未使用的秘密验证集。

## 4. 为什么不能再找“更晚的一段”

### 源序列终点有固定证据

`experiments/final_temporal_eval_v1/data.py` 固定源文件大小、SHA256 和总行数；`stream_tail` 检查总行数必须等于对应 stop。因此当前尾段已到达这两份固定源的末尾：

| 数据 | 固定源 | 结束行 / 源 SHA256 |
| --- | --- | --- |
| Abilene | `datasets/net_traffic/Abilene/abilene.csv` | 48384；`d1df6c8d00694415d393d150db9aecaa9e8cf9e2c29d6148fba71479d57f78e8` |
| GEANT | `downloads/raw/ari_llm/6a4f63656dbf/geant.csv` | 10772；`ab299a337087787541ece9c5422b81f4af03f816e38d9f031f6a328485057eba` |

来源元数据交叉印证：`analysis/final_evaluation_audit_20260914/abilene_identity_report.json` 记录官方 X01…X24 共 48384 行；`outputs/geant-recovery-20260912/data-restoration.json` 记录 ARI-LLM 提交 `6a4f63656dbf0964219c516ffd3e1e8fcf99ebed` 的 10772×462 CSV。完整原文件恢复、字节哈希和规范性解析属于技术访问，本身不必等同于基于分数的选择；这里判为已消费的关键是实际模型结果和论文记录。

### 旧标签、隔离带、另一份 GEANT 文件都不能自动补出测试集

`analysis/final_evaluation_audit_20260914/REPORT_CN.md` 曾发现 GEANT [9172,10772) 在旧 research card 中标记 `sealed_test`。这只能说明特定旧分支的声明；9 月 15 日实际完成记录已经更新其使用状态。

项目还曾存在 11460 行的 SNDlib GEANT 重建文件，但恢复报告与上述旧审计都指出其不等于本轮 10772 行 canonical CSV，且没有确证的时间/流轴映射。不能把 10772 之后的行号直接当作同一序列的未见延长段。

两条 gap 不能通过跨界拼窗、换随机 mask 或改名称变成新的独立后续数据。新 mask 可以验证固定值上的缺失模式变化，但不能恢复流量样本在项目层面的未见状态。

## 5. 对本轮收敛判断的最小影响

1. 继续本轮已授权 fit/dev 训练与预定比较；本审计不追加新数据集、不提出额外扫描或 GPU 任务。
2. 如果开发证据支持继续，固定新候选与 checkpoint 后，可复用一个已知评价范围作透明的历史 benchmark 复核。应明确写“已被历史研究访问的固定后续段复用”，不要写“全新独立测试”或用“本候选第一次运行”替代项目访问历史。
3. 论文能够如实讨论开发结果、冻结后复核和已有数据范围的局限；当前没有独立范围不等于必须继续扩张协议。是否值得成稿仍取决于本轮收益和机制证据，本审计不提前宣告模型收敛。

所有证据路径除本文件外均相对于 `/Users/agiuser/Documents/work/CalibTM`。审计止于现有元数据与已报告证据；未声称遍历过项目全部机器/历史日志，也未从局部缺少访问记录推断数据清白。
