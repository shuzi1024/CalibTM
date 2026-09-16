# 六个 fixed120 完成任务的独立数字审计

审计时间：2026-09-16T07:44:24.068845+00:00。所核对汇总快照：2026-09-16T07:36:49.707955+00:00。

## 1. 范围与结论

本次只读取现有 `raw_fixed120`、对应 `raw_outputs` 父历史及旧 `references`，独立计算后核对 `summary_fixed120`；未调用汇总程序、未运行模型或 GPU、未修改训练与汇总源码。快照包含 Abilene 的 Full/no_memory 各两个种子，以及 GEANT 的 no_memory 两个种子；两个 GEANT Full 尚未完成，不纳入均值或跨 WAN 的 Sync 结论。

**数字核对通过，但现有证据不支持“Sync 稳定改善主指标”的表述。** Abilene 的 Full 相对 no_memory，两个种子的 NMAE 差异方向相反，平均 NMAE 略差；平均 NRMSE 更好。no_memory 在两个 WAN 上均有较小的开发 NMAE 优势，但 GEANT 的平均 NRMSE 比三条旧参考都差。SPIN context + Direct 可以作为有初步经验支持的组合候选，尚不能据此单独归因 Direct 的贡献或宣称模型已经收敛。

## 2. 独立核算方法与一致性

每个条件先合并全部开发窗口缺失位置的误差和真值：`NMAE = Σ|error| / Σ|truth|`，`NRMSE = sqrt(Σerror² / Σtruth²)`。再对 uniform、unequal、unequal_gap 三个条件等权取平均。表中总体 NRMSE 是三个条件 NRMSE 的算术平均，不是把三条件合并后再开方。所有 NRMSE 都来自按 NMAE 选出的同一最佳检查点，没有另挑 NRMSE 最佳轮。跨种子统计使用两个种子的平均值与样本标准差（分母 `n−1`）；不把父任务和续跑当成四次重复。

核查通过的项目：

- 六份完成结果的角色均为 development_only；逐窗误差/真值总和、三条件指标、总体指标与结果文件一致。
- 每项历史均连续覆盖 1–120 轮；父历史与续跑历史前缀完全一致。历史中最小 selection_score、相应最佳轮、最终结果及该最佳轮的三条件 NMAE 一致。
- 六项 config.json 与 best.pt 的 SHA256 与 result.json 一致；结果文件指纹与汇总快照记录一致。
- 个体指标、方法均值、样本 SD、Abilene 配对差和六条旧参考指标与 SUMMARY.json 一致；数值比较使用相对容差 2×10⁻¹²、绝对容差 2×10⁻¹⁴。SUMMARY_CN.md 的九位小数显示也一致。
- Abilene 同种子 Full/no_memory 的数据、训练配置、源码指纹及 120 轮训练顺序/掩码记录一致。移除记忆也移除了参数：Abilene Full 为 98,693 参数、no_memory 为 73,733 参数；这不是等参数量对照。
- 新模型与旧参考按条件、cohort、window_start 对齐后的逐窗真值分母及缺失数一致；SPIN、旧 Direct+Sync 和新模型的开发掩码 SHA 一致。Linear 参考未存掩码 SHA，只存 mask_seed；本审计核对其逐窗分母与数量，没有重新加载位图。

### 个体结果

| WAN | 变体 | 种子 | 父轮数 → 总轮数 | 最佳轮 | 三条件平均 NMAE | 三条件平均 NRMSE |
|---|---|---:|---:|---:|---:|---:|
| Abilene | Full | 41001 | 89 → 120 | 120 | 0.162975999 | 0.218176260 |
| Abilene | Full | 41002 | 75 → 120 | 117 | 0.161885985 | 0.217960583 |
| Abilene | no_memory | 41001 | 76 → 120 | 118 | 0.163432869 | 0.220289858 |
| Abilene | no_memory | 41002 | 98 → 120 | 120 | 0.161305121 | 0.219626765 |
| GEANT | no_memory | 41001 | 30 → 120 | 119 | 0.167676249 | 0.229851096 |
| GEANT | no_memory | 41002 | 57 → 120 | 113 | 0.166465556 | 0.232527728 |

## 3. 两种子均值与样本标准差

| WAN | 方法 | NMAE 均值 ± SD | NRMSE 均值 ± SD |
|---|---|---:|---:|
| Abilene | Full | 0.162430992 ± 0.000770756 | 0.218068422 ± 0.000152506 |
| Abilene | no_memory | 0.162368995 ± 0.001504545 | 0.219958311 ± 0.000468878 |
| GEANT | no_memory | 0.167070903 ± 0.000856089 | 0.231189412 ± 0.001892664 |

SD 只描述当前两个训练种子的离散程度，不构成显著性检验或置信区间。Abilene Full 的 NMAE 均值比 no_memory 高 0.000061997，约差 0.0382%；这项均值差远小于两种方法各自的样本 SD。Full 的平均 NRMSE 低 0.001889890，相对 no_memory 改善约 0.8592%。

### Abilene 同种子配对差（Full − no_memory；负数表示 Full 更好）

| 种子 | 平均 NMAE 差 | 平均 NRMSE 差 |
|---:|---:|---:|
| 41001 | -0.000456870 | -0.002113598 |
| 41002 | +0.000580865 | -0.001666181 |

## 4. 三条件与误差权衡

下表仍为两种子的均值 ± 样本 SD。

| WAN | 方法 | 条件 | NMAE | NRMSE |
|---|---|---|---:|---:|
| Abilene | Full | uniform | 0.153066063 ± 0.000834748 | 0.207144251 ± 0.000031613 |
| Abilene | Full | unequal | 0.166127236 ± 0.000726443 | 0.222625312 ± 0.000340018 |
| Abilene | Full | unequal_gap | 0.168099677 ± 0.000751077 | 0.224435702 ± 0.000149114 |
| Abilene | no_memory | uniform | 0.153020537 ± 0.001557384 | 0.211577113 ± 0.000541952 |
| Abilene | no_memory | unequal | 0.166048460 ± 0.001457006 | 0.222986340 ± 0.001047734 |
| Abilene | no_memory | unequal_gap | 0.168037988 ± 0.001499245 | 0.225311482 ± 0.000900852 |
| GEANT | no_memory | uniform | 0.154175377 ± 0.001039202 | 0.211877125 ± 0.000474321 |
| GEANT | no_memory | unequal | 0.172739254 ± 0.000851604 | 0.241614131 ± 0.002215032 |
| GEANT | no_memory | unequal_gap | 0.174298077 ± 0.000677462 | 0.240076979 ± 0.002988641 |

- **Abilene 的 Sync 主指标效应不稳定。** 在三个条件下，41001 的 Full NMAE 都优于 no_memory，而 41002 的 Full NMAE 都更差；两个种子取平均后，Full 的三个条件 NMAE 均略高。不能仅用较好的 41001 说明 Sync 有稳定收益。
- **Abilene 的平方误差指标存在收益。** Full 的三个条件平均 NRMSE 均更低；逐种子、逐条件六个格子中五个更低，但 41002 的 unequal 条件反而高 0.000139403。因此“两个种子的总体 NRMSE 都改善”成立，“所有条件都改善”不成立。
- **no_memory 自身也有条件差异。** Abilene 的 no_memory 三条件平均 NMAE 都低于旧 SPIN，但 uniform 的 NRMSE 为 0.211577113，高于旧 SPIN 的 0.209904527；整体 NRMSE 的小幅优势来自其他条件。
- **GEANT 存在明确的 NMAE/NRMSE 权衡。** no_memory 三条件平均 NMAE 都略低于旧 SPIN 和 Linear；平均 NRMSE 却在三个条件都高于 Linear。相对旧 SPIN，uniform 的 NRMSE 较低，而 unequal 与 unequal_gap 较高，最终总体 NRMSE 更差。不能据 NMAE 单项较低写成“全面领先”。

## 5. 旧参考核对与可比性边界

`references/*/direct.json` 的实际 variant 是 **direct_sync**，即旧版 Direct+Sync，不是纯 Direct；`spin.json` 为旧的 spin_adapted。下表均为旧结果的原始值，未替换或重新训练。

| WAN | 旧参考 | 平均 NMAE | 平均 NRMSE | 训练轮数 / 上限 | 最佳轮 |
|---|---|---:|---:|---:|---:|
| Abilene | SPIN adapted | 0.164380137 | 0.220772517 | 137 / 300 | 97 |
| Abilene | 旧 Direct+Sync | 0.165638897 | 0.223397977 | 53 / 120 | 43 |
| Abilene | Linear | 0.171021478 | 0.241480181 | 无需训练 | — |
| GEANT | SPIN adapted | 0.168317410 | 0.229491749 | 300 / 300 | 284 |
| GEANT | 旧 Direct+Sync | 0.173335594 | 0.228000567 | 71 / 120 | 61 |
| GEANT | Linear | 0.167739045 | 0.228766611 | 无需训练 | — |

no_memory 相对旧 SPIN 的平均 NMAE 改善为 Abilene **1.2235%**、GEANT **0.7406%**；相对 Linear 为 **5.0593%**、**0.3983%**。但其平均 NRMSE 相对旧 SPIN，在 Abilene 改善 **0.3688%**，在 GEANT 恶化 **0.7397%**；GEANT 相对 Linear 恶化 **1.0591%**，相对旧 Direct+Sync 恶化 **1.3986%**。这些比例由原始值独立计算，方向与汇总一致。

旧可训练参考各只有 seed41001，本批方法有两个初始化；训练轮数、早停策略、模型结构和优化设置也不同。旧 SPIN 在 Abilene/GEANT 上分别训练 137、300 轮，旧 Direct+Sync 分别训练 53、71 轮，不能视为本批统一 120 轮结构的同预算组件消融。当前比较可描述已有参考上的表现差异，不能将差异全部归因于某一个新模块。

## 6. 现有证据能支持什么

**可以支持的有限表述：**“在当前已完成的开发实验中，保留 SPIN 上下文与 Direct 读出的 no_memory 组合，以较少参数达到与 Full 接近的 Abilene NMAE，并在两个 WAN 上取得略低于旧 SPIN 参考的平均 NMAE；Sync 在 Abilene 上的主要可见收益是平均 NRMSE，而非稳定的 NMAE 改善。”这里的“SPIN 上下文”是当前统一模型中的实现，不等同于原完整 SPIN 流程，也不表示借用组件本身是新贡献。

**仍不足以支持的表述：**“Direct 的独立贡献已证实”“SPIN 与 Direct 的协同机制已被识别”“Sync 在两个 WAN 上都无用/有效”“模型已取得稳定全面优势”。本批 Full/no_memory 配对只改变记忆分支及其参数，可以讨论该变化在已完成 Abilene 配对中的影响；它没有独立识别 SPIN context 或 Direct 的贡献。旧 SPIN 与旧 Direct+Sync 又同时存在结构和训练协议差异，不能补足这种归因。组合方案目前有经验依据，其单模块因果贡献及方法新颖性不由这些数字自动成立。

全部指标都是重复使用的开发集上选取最佳轮后的结果。fixed120 是在已有早停轨迹之后决定的训练时长诊断，不是新独立重复；六项中五项的最佳轮落在 116–120 轮，亦不能据此宣称训练已经充分收敛。当前文档仅给出六任务快照结论，保留尚未完成的两个 GEANT Full，不预判其结果，也不扩大实验矩阵。

## 7. 输入位置与快照指纹

- 原始完成结果及历史：`raw_fixed120/{abilene,geant}/{spin_sync_direct,spin_direct_no_memory}/seed{41001,41002}/{result,history}.json`，仅使用上表六项。
- 父历史：同层级的 `raw_outputs/.../history.json`。
- 旧参考：`../research_7h_20260916/references/{abilene,geant}/{spin,direct,linear}.json`。
- 核对目标：`summary_fixed120/SUMMARY.json` 与 `SUMMARY_CN.md`。
- 本次 SUMMARY.json SHA256：`5c802a7c37387e2af17d0efd1f781b225e5bd638dc8edcdb9e02961c8a4e7808`。

本审计未反序列化训练张量、未访问新增开发/测试窗口、未启动任何模型运行。
