# CalibTM 补充结果与 reviewer-facing 稳健性证据

**版本：2026-08-12**

本文件承接主文有意压缩的结果。它不改变 A/G historical final-gate 或 BRAIN frozen
confirmation 的原始裁决，也不把 post-gate 分析包装成新的独立测试。

## S1 证据角色

| 证据 | 身份 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| A/G K3 structured | development 后 historical evidence | 当前 operating point 在两网的 matched effect | pristine test、普遍泛化 |
| BRAIN confirmation | checkpoint/registry 冻结后打开 | 第三 WAN 上相对 Linear/Context-Free 的 effect confirmation | direct OD、长期时间推断、大型 baseline 排名 |
| 时间块稳健性 | post-gate read-only sensitivity | 物理 window 共享 draw 后 headline 仍为正 | 改写原注册 verdict、长期总体 CI |
| Frozen bound diagnostics | post-gate descriptive | 约束在现有 checkpoint 中是否实际活跃 | bounded 优于 unbounded/direct 的因果必要性 |
| 六方向 frozen transfer | post-gate descriptive transfer | 不更新权重时的完整六方向行为 | independent test、statistic-free universal zero-shot |
| RIPE availability replay | post-gate semi-synthetic replay | 实测 control-plane availability shapes 下的描述性行为 | native TM outage accuracy、第三 WAN |
| K2–K5 curve | development descriptive | 冻结模型可直接消费 K4/K5 额外观测 | 跨时间/跨网络稳定相对优势 |

## S2 物理时间块统计敏感性

A/G 每个 dataset×seed 先汇总 internal block 与 two-burst；三个冻结 seed 固定等权，
不作为三个现实数据样本。每个 bootstrap draw 在同一物理窗口上跨 methods、masks 与
seeds 共享。flow、target 和 mask 也不是独立重采样单位。

| 对比 | Point gain | Absolute ΔNMAE | count-4 CI | 约 1-day CI | 约 2-day CI |
|---|---:|---:|---:|---:|---:|
| CalibTM vs Context-Free | +1.0425% | +0.002426 | `[+0.8920,+1.2002]%` | `[+0.8955,+1.1930]%` | `[+0.8996,+1.1984]%` |
| CalibTM vs Linear | +1.5187% | +0.003546 | `[+1.3675,+1.6788]%` | `[+1.3551,+1.6881]%` | `[+1.3832,+1.6739]%` |

Abilene 的 T50 window 为 250 min，GEANT 为 750 min；约 1-day blocks 分别取
6/2 windows，约 2-day 取 12/4。所有方案均为 circular sensitivity，不能外推为无限
时间总体保证。

BRAIN 只有 28 个相邻但不重叠的 50-min windows。block 1/4/8/14 的 interval lower
bound 对 Context-Free 分别为 1.0525/1.0322/1.0372/1.0571%，对 Linear 为
3.1260/3.0265/2.9840/2.9980%。28/28 physical windows 对两个 comparator 都正，
median gain 为 +1.1533%/+3.1530%；但数据只跨约一天、只有两个 UTC start-day
clusters，cluster sign test 的双侧 p=0.5，不能声称长期自相关已被识别。

投稿图：`output/pdf/CalibTM_statistical_robustness.pdf`。

## S3 Frozen constraint usage 与能力边界

18 个 A/G dataset×seed×mask replay cells 的约束占用如下；occupancy 定义为实际
`|tanh(logit)|`，不是独立样本比例或显著性检验。

| Dataset | delta-r occupancy | internal-offset occupancy | edge-offset occupancy | r-hat clip | pre-projection negative |
|---|---:|---:|---:|---:|---:|
| Abilene | 1.67% | 59.51% | 92.52% | 0 | 2.19% |
| GEANT | 3.25% | 64.57% | 82.48% | 0 | 9.64% |

offset bounds 与 nonnegative projection 在冻结实现中数值上确实活跃；这不能替代
matched unbounded/direct arm。固定输入 function slices 在六个 checkpoints 间的
pairwise correlation median 为 internal 0.9856、edge 0.9910、delta-r 0.6769
（delta-r minimum -0.5690）。这些是固定坐标诊断切片，不是经验条件期望或因果曲线。

对 305,117 个单侧 edge segments 的逐段审计确认：290,993 个多点段内，active
features、Linear、edge offset 与最终 prediction 全部 bit-exact 常量。因此当前 edge
组件应称为 anchor-dependent、distance-invariant boundary-segment shift，而不是随外推
距离变化的 curve。

投稿图：`output/pdf/CalibTM_bound_usage_and_frozen_slices.pdf`。

## S4 六方向 frozen-weight cross-WAN transfer

所有方向在 outcome 前一并固定且全部报告；没有更新权重、target checkpoint selection、
集成或方向删除。每方向包含 3 source-model seeds × 3 target-mask seeds。

| Source → Target | Source value NMAE | vs source Context-Free | vs target Linear | 两组正 factors |
|---|---:|---:|---:|---:|
| Abilene → GEANT | 0.219641 | +0.5677% | +0.9966% | 9/9；9/9 |
| GEANT → Abilene | 0.235806 | +1.4027% | +1.9618% | 9/9；9/9 |
| Abilene → BRAIN | 0.906423 | +1.0945% | +2.3464% | 9/9；9/9 |
| GEANT → BRAIN | 0.903004 | +1.4240% | +2.7148% | 9/9；9/9 |
| BRAIN → Abilene | 0.235783 | +1.3023% | +1.9718% | 9/9；9/9 |
| BRAIN → GEANT | 0.219230 | +0.8544% | +1.1826% | 9/9；9/9 |

最差 structured factor 仍为 +0.4581% vs source Context-Free、+0.8088% vs Linear。
负面结果同样保留：A→G random 对 Context-Free pooled 仅 +0.1566%，且 6/9 factors
为正，所以 headline 只覆盖 structured masks。

正确名称是 **frozen-weight transfer under target-frozen observation-only preprocessing**。
它不是完全 statistic-free：std<1e-6 fallback 触发 A/G/BRAIN 的
2.41%/9.01%/40.44% flow-windows，对应 missing-truth mass 为
0.00409%/0.00225%/4.62562%。三个 target cohorts 也都已消费，因此这不是新的
independent confirmation。

投稿图：`output/pdf/CalibTM_cross_WAN_frozen_transfer.pdf`。

## S5 RIPE RIS measured-availability-shape replay

三次官方 collector maintenance 的 mask 只由每个 5-min gzip payload 解压后是否非空
定义，不使用 BGP value/update count，也不把 mask 强行改成 K3。每个事件固定 onset- 与
recovery-centered 两个 T50 windows。

- RRC18：68-slot empty run；两个窗口分别为 `1^25 0^25` 与 `0^25 1^25`；
- RRC15：28-slot empty run；在 T50 中生成与 RRC18 相同的两种 masks；
- RRC12：8-slot empty run；两个窗口为 `1^25 0^8 1^17` 与 `1^17 0^8 1^25`。

3 events × 2 phases × A/G × 3 seeds 共 36 cells。CalibTM 相对 Linear 为正 36/36，
相对 Context-Free 为正 35/36；唯一非正为 GEANT/RRC12-onset/seed1 的 -0.0813%。
RRC18/RRC15 同形结果逐 prediction/error deterministic 相等，因此六个 event-phase 只有
四个 unique masks，不能当作六次独立 replication。

真实的是 control-plane payload availability shape；TM values 与 flow axis 仍来自已消费
A/G。Abilene 与 RIS 都是 5-min bins，GEANT 仅转移 ordinal shape。它不是 native TM
outage、第三 WAN、accuracy confirmation 或 outage detection。

## S6 Observation budget K2–K5

冻结 checkpoint 在两个 development cohorts、random/internal nested masks 上直接运行
K2/K3/K4/K5。K2/K3 为 training-supported；K4/K5 没有重训，属于 observation-count
zero-shot。固定 common-K5 targets 上，CalibTM 的 K2→K5 NMAE 下降：

| Cohort | Random | Internal |
|---|---:|---:|
| mechanism | 16.57% | 16.43% |
| architecture | 18.90% | 19.06% |

Linear/Context-Free 也下降约 16%–19%，说明主要现象是更多真实观测普遍有益。更重要的
负面边界是 architecture/GEANT：两个 mask、两个 target scope、每个 K 对两个 comparator
都为负。因此该曲线只能说明 frozen model 能消费 K4/K5，不能声称跨时间/跨网络稳定优势。

## S7 其他移出主文的证据

- Intel Xeon Gold 6530、uncompiled FP32、seed-1 synthetic shapes 的 CPU grid：
  batch=1/one-thread 时 CalibTM 为 A/G 4.783/17.260 ms，ARI 为其
  283.8×/186.9×，ImputeFormer adapter 为 37.2×/27.9×；batch=32/eight-thread
  时 CalibTM 为 561.5/146.7 windows/s，是 ARI 的 174.0×/116.7×、adapter 的
  86.2×/30.9×。Linear 仍更便宜：batch=1 快 16.3×/22.9×，batch=32 throughput
  高 18.1×/24.3×。计时排除模型加载和输入构造；whole-process RSS 是 1-ms sampled
  separate pass，不能和 CUDA allocator memory 横向混用。
  完整表见 `experiments/value_only_cpu_pareto_v1/REPORT_2026-08-12_CN.md`，投稿图见
  `output/pdf/CalibTM_CPU_checked_pipeline_cost.pdf`。
- WS-DREAM RT 官方重建、service-disjoint discovery tune：retrained CalibTM 在 K4
  structured masks 下相对 Context-Free/Linear 为 +3.0108%/+7.4115%。它是 QoS-like
  applicability，不是第三 WAN、zero-shot、native missingness 或官方 test。
- H800 seed-1 training-update microbenchmark：ARI 在 A/G shape 的 complete-update latency
  为 CalibTM 的 8.05×/14.49×，steady peak allocated memory 为 42.82×/42.56×；不是
  完整训练 wall time 或 time-to-convergence。

## S8 主要 artifact

- statistics：`experiments/value_only_statistical_robustness_v1/results/analysis.json`
- bound diagnostics：`experiments/value_only_bound_diagnostics_v1/results/replay_cpu_fp32_r2.json`
- cross-WAN：`experiments/value_only_cross_wan_zero_shot_v1/results/formal_r1/adjudication.json`
- measured masks：`experiments/value_only_measured_availability_replay_v1/results/adjudication.json`
- K curve：`experiments/value_only_observation_budget_v1/results/adjudication.json`
- CPU Pareto：`experiments/value_only_cpu_pareto_v1/results/cpu_fp32_seed1.json`
