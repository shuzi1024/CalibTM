# CalibTM：极端时间稀疏下的轻量锚点保持流量矩阵校准补全

**CalibTM: Lightweight Anchor-Preserving Calibration for Traffic Matrix Completion under Extreme Temporal Sparsity**

**中文版方法论文研究稿 · v3 · 2026-08-12**

## 摘要

流量矩阵（Traffic Matrix, TM）历史档案可能因设备维护、采集服务中断或数据传输失败而不完整。本文研究一个刻意收窄的离线补全设置：在长度为 50 的时间窗口中，每条 OD flow 仅保留 3 个真实观测点。此时 canonical Linear 已由观测锚点确定一个零训练成本的可行骨架。我们考察一个具体问题：学习器能否不把缺失值参数化为无约束的自由预测，而只学习显式插值骨架上的局部有界校准，从而形成更轻量的 accuracy–efficiency operating point。

为此，我们提出 CalibTM：一个仅含 5,475 个参数的 observation-derived、scaffold-conditioned bounded calibrator。它只由真实观测构造 canonical Linear，再以共享小型网络产生尺度归一的内部点级与边界段级校准。输出在最后一步精确复制全部观测点，并保证缺失位置非负。CalibTM 以一次前向完成补全，不需要大型时空主干。

在 Abilene 和 GEANT 的 K=3 structured masks 上，CalibTM 相对参数量与初始化匹配的静态校准取得数据集等权的 1.0425% NMAE 改善，相对 Linear 改善 1.5187%。冻结方法在 SNDlib BRAIN 第三 WAN confirmation 上相对静态校准和 Linear 分别改善 1.1396% 和 3.2627%，三个 seed 与两类 structured mask 均同向。冻结 H800 与 CPU checked-pipeline 测量进一步表明，CalibTM 位于几乎零成本的 Linear 与大型 learned baselines 之间。实验支持的是一个场景限定的受约束校准 operating point，而不是“所有低自由度模型都优于自由重建”的普遍结论。

**关键词：** 流量矩阵补全；极端稀疏；受约束学习；插值校准；网络测量

<!-- FIGURE:METHOD -->

## 1 引言

流量矩阵描述一个时间段内源—目的节点对之间的流量，是容量规划、异常分析和流量工程的重要输入。完整获取所有 OD flow 的细粒度历史序列往往代价高昂；实际档案还可能受到设备升级、采集服务中断和数据传输失败的影响。因此，从少量可信观测恢复缺失位置一直是网络测量中的基本问题。

公开测量基础设施也表明“采集过程本身会留下空档”并非纯想象：RIPE Atlas 提供持续的 anchor full-mesh 测量及按时间下载结果的[官方接口](https://atlas.ripe.net/docs/apis/rest-api-manual/anchors/anchor-measurements/)；RIPE RIS 提供[时间索引的 MRT 档案规范](https://ris.ripe.net/docs/mrt/)，官方 [RRC18 维护通知](https://mailman.ripe.net/archives/list/ris-users%40ripe.net/thread/CYIFPVGKBE4EZKKFVZS4HQECUQ3G73UG/) 则明确记录 collector sessions 将中断且 MRT files 不会生成。RIPE RIS 不提供 traffic-volume TM truth；本文只在方法冻结后的描述性 replay 中，把三次维护事件的实测 payload-availability shape 转移到已消费的 A/G TM 窗口。它仍是 semi-synthetic measured-mask transfer，不是 native TM outage accuracy；主评估场景仍应准确称为 extreme artificial temporal thinning stress setting。

近年来，TM 补全方法从低秩矩阵/张量优化发展到神经张量补全、Transformer、扩散模型以及 LLM。ARI-LLM [1] 通过 flow-level tokenization 和 coarse-to-fine 自回归推理建模动态 TM；ImputeFormer [2] 把低秩先验嵌入 Transformer；Diffusion-TM [3] 和 3DDPS-TME [4] 则利用生成式分布学习或 routing-conditioned posterior sampling。这些方法分别利用跨 flow、绝对时间、拓扑或路由中的一种或多种结构。与之互补，本文关心在信息极少且 canonical Linear 已较强时，受约束的局部学习是否构成值得保留的成本—精度选择。

我们聚焦 **offline archive repair under extreme temporal thinning**：一个完整窗口在修复时可被双向查看，但每条 flow 仅有 K=3 个观测；模型不获得 topology、routing 或持久 flow identity，也不决定采样位置。canonical Linear 已经给出一个无需训练、严格通过观测锚点的结构先验。我们的核心问题是：

> 在同一组稀疏锚点下，围绕显式 Linear scaffold 学习尺度受限的局部校准，能否以很小的 learned-model 成本获得稳定增益？

CalibTM 将修正分解到内部区间和单侧边界，并通过 `tanh`、锚点差和观测尺度限制幅度；观测位置在最终一步逐位 hard-copy。它以一次前向的局部校准算子替换 ARI-LLM 的 Flow2Vec、LLM 和五阶段 rollout，但本文把这种替换视为一种设计选择，而非“约束模型普遍优于自由重建”的因果结论。

本文贡献有三点：

1. 提出一个 topology-free、anchor-preserving 的有界插值校准算子，把学习限制在 canonical Linear 邻域中的内部点级与边界段级形变。
2. 通过同参数、同初始化的 context-free calibrator（代码名 `static_zero`）隔离“向共享 MLP 暴露 scaffold-derived 通道”的增量，并用冻结 head intervention 与 bound diagnostics 定位实际起作用的校准路径。
3. 在 Abilene、GEANT 和冻结的 SNDlib BRAIN 第三 WAN confirmation 上报告多 seed/mask 证据，并联合分析时间块稳健性、冻结权重跨 WAN transfer、现代 stress comparators 与实测成本所形成的 operating point。

## 2 相关工作与方法边界

### 2.1 矩阵与张量补全

传统方法依赖低秩、周期性或张量结构。WTTC-TS [5] 通过时间切片和加权张量核范数处理连续缺失，在无需预训练的情况下追求较简单的优化形式；其后续工作 [6] 又结合 tensor nuclear norm-minus-Frobenius regularization 与 time slicing。NTC [7]、DATC [8] 和 PetTC [9] 分别借助神经张量交互、对抗学习和对比嵌入增强 TM 表示。此类方法通常把多个 flow 或多个时间片作为联合结构；CalibTM 则刻意不使用可识别的跨 flow 关系，只在单 flow 锚点定义的可行邻域内校准。它的目标不是取代低秩/周期方法在充分结构信息下的上限，而是提供一个信息边界更窄、成本更低的 operating point。

### 2.2 插值引导与约束式补全

线性、样条和核插值长期以来都是时间序列缺失处理的基本工具。经典 PCHIP [10] 通过局部斜率选择实现锚点精确和形状保持；近期 percentile slope-constrained Linear [11] 又从观测差分分布估计阈值，以逐步裁剪不合理斜率。LinAR [12] 则把 autoregressive variation 与 Linear interpolation 结合，在水文缺口内恢复局部波动并减轻端点跳变。这些工作说明 anchor preservation、bounded interpolation 和 Linear refinement 都不能单独作为新颖性。

学习式方法也早已利用插值或初始估计。最直接的非 TM 碰撞是 DCCN-SPF [13]：它以 Linear 表示 PM2.5 的总体趋势，再让因果卷积网络预测细节偏差，而非直接预测完整缺口。Interpolation-Prediction Networks [14] 用可学习核插值把稀疏不规则观测映射到参考时间点；PriSTI [15] 把插值条件编码为扩散先验；RDPI [16] 从确定性初值出发，以 residual 为扩散目标进行 refinement；ImputeINR [17] 则从稀疏观测学习连续隐式函数。

因此，CalibTM 的新颖性不建立在“首次使用 Linear prior”“首次学习插值偏差”或“小 MLP”上。本文的贡献限定于一个更窄的 TM operating point：在每条 flow 仅有三个锚点、不使用 flow identity/topology/routing 时，把 observation-only Linear 固定为显式 scaffold，只学习尺度归一、内部/边界分区的有界形变，并以最终 hard-copy 保证锚点精确；`static_zero` matched control 进一步隔离向 MLP 暴露样本条件通道的增量价值。这一定位强调特定受约束参数化及其在极端稀疏 TM 中的实证 operating point，而不把插值引导补全这一大类思想当作本文首创。

### 2.3 高容量时空与生成式补全

ImputeFormer [2] 将低秩投影与 Transformer 结合，以兼顾时空归纳偏置和表达力。Diffusion-TM [3] 使用扩散模型学习完整 TM 分布，并在极低已知比例下进行分析；3DDPS-TME [4] 将 TM 构造成三维张量，并通过 routing equations 引导扩散后验采样。它们分别依赖跨变量结构、完整分布或 routing matrix，和本文“不使用 topology/routing/flow identity”的输入边界不同。我们只对 ImputeFormer 做了一个 bounded official-architecture task adapter；Diffusion-TM 与 3DDPS-TME 在本文中属于相关工作，而非已经完成公平适配的数值基线。

### 2.4 LLM 与过程建模

ARI-LLM [1] 把一条长度为 T 的插值后 flow 经 Flow2Vec/LSTM 压成一个 token，再由 LLM 在 flow tokens 间建模，并通过五阶段 coarse-to-fine rollout 输出完整 TM；CalibTM 则省去这一主干，只保留局部确定性校准。Utimac [18] 从局部平稳窗口的生成过程出发，将 log-domain 流量分解为主统计成分和稀疏偏差，并进行共享参数推断与不确定性刻画。它展示了“先识别过程，再设计估计器”的更强问题驱动范式；其跨 flow covariance、概率假设和 uncertainty 目标也不同于本文。

### 2.5 网络测量场景

INT-MC [19] 从 in-band telemetry 的真实 overhead 与 path-selection 问题出发，并在可编程交换机原型中验证矩阵补全。它提醒我们，人工 K/T 比例不能直接等价为实际 telemetry 节省。本文只研究已有离线档案的缺失值修复，不控制观测路径，不声称 K=3/T=50 意味着 94% 的线上开销下降。

## 3 问题定义

令一个窗口内的 TM 为 X∈R^(F×T)，其中 F 为 flow 数，T=50 为时间长度。二值矩阵 M∈{0,1}^(F×T) 表示观测位置；主设置要求每条 flow 恰有 K=3 个观测。输入为 X_obs=M⊙X 与 M，目标是在 Ω_miss={(f,t):M_ft=0} 上预测 X。

本文采用 offline smoothing 而非 forecasting：修复某个内部缺口时允许使用其左右两侧已经存在于档案中的观测。边界位置不存在包围目标的左右锚点对；canonical Linear 使用单侧最近锚点延拓，而 CalibTM 的校准仍条件于该 flow 全部可见观测导出的均值、标准差与局部尺度。模型输出 X_hat 需要满足两个硬约束：

1. 锚点保持：对所有 M_ft=1，X_hat_ft=X_ft；
2. 非负性：对所有缺失位置，X_hat_ft≥0。

主指标为 ratio-of-sums NMAE：

`NMAE = Σ_(f,t∈Ω_miss) |X_hat_ft-X_ft| / Σ_(f,t∈Ω_miss) |X_ft|`。

所有正式方法比较共享相同窗口、mask 与 target 集合。Relative gain 定义为 `1 - NMAE(method)/NMAE(comparator)`，正值表示 method 更好。

### 3.1 场景边界

本文主实验的 K=3 masks 是人工 thinning，包括 random、internal block 和 two-burst；periodic polling 与 shared contiguous-outage 仅是具有 wall-clock 解释的 synthetic stress。它们不是 measured native TM missing trace。Abilene 为 5-minute bins，GEANT 为 15-minute bins，相同离散 mask 不代表相同物理时长。BRAIN 使用 1-minute bins，但其 OD demand 来自 link measurements 经 goals、path-flow LP 和取整反演，不是直接 OD collector。

## 4 CalibTM：锚点保持的有界插值校准

### 4.1 Observation-only canonical Linear

对目标位置 t，若左右锚点 `(t_L,x_L)` 与 `(t_R,x_R)` 均存在，先构造

`B_t = (1-r_t)x_L + r_t x_R,   r_t=(t-t_L)/(t_R-t_L)`。

边界缺口使用最近观测值延拓。所有涉及 Linear 的模型输入、prior feature 与 Linear baseline 共享同一纯函数；插值过程中不读取缺失真值。

### 4.2 归一化与条件输入

每条 flow 的均值和标准差只由当前观测计算；仅在整条 flow 无观测，或观测标准差低于 1e-6 时使用训练集冻结 fallback。CalibTM 在归一化坐标中运行。对位置 `(f,t)`，方法实际使用的条件信息定义为

`z_(f,t) = [B_(f,t), x^(obs)_(f,t), m_(f,t), vbar_f]`，

其中 `B` 为 canonical Linear，`s_f` 是在该 flow 逐观测标准化后重算的 `local_volatility`，而 `vbar_f=s_f/(max_g s_g+ε)` 是暴露给 MLP 的匿名窗口归一化特征。对正常的 K≥2 非退化 flow，`s_f` 近似常数；它主要在退化或 fallback 情形下变化，不应被解释为丰富的原始局部波动信号。在真正缺失位置，`x^(obs)=m=0`，因此逐位置变化的主要条件是 `B`。

主文将共享校准器记为四维语义映射 `g_θ(z_(f,t))`；冻结 checkpoint 的固定兼容接口与完整参数账本移至补充实现说明，它们不改变上述有效信息边界。`max_g s_g` 对 flow permutation 不变，不含 flow identity、topology、routing 或逐点 cross-flow attention；但它使预测依赖当前 window 的 flow composition。因而 CalibTM 不能识别特定 flow 间关系，却也不是 strictly flow-local 或 flow-subset-consistent。

### 4.3 共享校准网络

共享网络 `g_θ(z_(f,t))` 由一个小型归一化 MLP 与三个标量输出头构成；冻结 checkpoint 共含 5,475 个参数，所有 flow 与时间位置共享。精确层宽、兼容布局与参数账本见补充实现说明。三个 head 产生：

- 插值比例修正 `δr_t = 0.25 tanh(h_r)`；
- 内部偏移 `o_t = 0.10 tanh(h_o)(|x_R-x_L|+s_f+ε)`；
- 边界偏移 `e_t = 0.10 tanh(h_e)s_f`。

内部缺口使用

`r_t^ε=d_L/(d_L+d_R+ε),   r_hat_t = clip(r_t^ε+δr_t,0,1)`，

`x_t^int = (1-r_hat_t)x_L + r_hat_t x_R + o_t`。

边界缺口使用 `x_t^edge = x_anchor + e_t`。对同一 flow 的同一侧边界段，`B_t`、`x^(obs)`、`m` 与 flow-level 特征均不随位置变化，因此冻结 edge head 给出段内常量偏移；左右两侧可因锚点值不同而间接不同，但它不是随外推距离变化的 extrapolation curve。最后从归一化坐标恢复原尺度，对缺失预测做非负截断，并在最后一步以原始 bit-exact 值 hard-copy 所有观测。

这套参数化限制了归一化坐标中的修正幅度，但内部 offset 仍允许有限越界，因此它是 bounded calibration around Linear，而非严格凸包插值。恢复原坐标时乘回由当前观测计算的标准差，原始单位下的尺度适配主要来自这一步，而非 `s_f` 提供了丰富的 volatility 信号。冻结消融显示 `δr` 没有可测附加价值；本文保留它以忠实报告已冻结模型，但方法主张与主图聚焦于真正贡献收益的 internal/edge offsets。

**命题 1（锚点保持与有界形变）.** 对 `σ_f>0` 且原始观测锚点非负的有效输入，令 `y_L,y_R,B_t^(n),x_hat_t^(n)` 分别表示归一化坐标中的左/右锚点、canonical Linear 和最终输出。对内部缺失令 `D_t=d_L+d_R`、`Δ_t=|y_R-y_L|` 与 `η_t=|r_t^ε-r_t|=d_Lε/[D_t(D_t+ε)]`。对本文 K3 设置，冻结参数化保证

`|x_hat_t^(n)-B_t^(n)| ≤ (β_r+η_t)Δ_t+β_o(Δ_t+s_f+ε),   η_t<5×10^-7`；

对单侧边界缺失保证 `|x_hat_t^(n)-y_anchor|≤β_e s_f`，其中 `β_r=0.25`、`β_o=β_e=0.10`。恢复原量纲后，上述右端统一乘以观测定义的 `σ_f`；同时 `M_(f,t)=1 ⇒ X_hat_(f,t)=X_(f,t)`，且所有缺失输出非负。

**证明.** `|tanh(·)|≤1` 给出三个 head 的幅度界，`[0,1]` 投影不扩张距离，三角不等式即得两个界。反标准化后的边界锚点与内部 Linear scaffold 分别是非负观测值及其凸组合，因而最终 `Π_[0,+∞)` 同样非扩张；hard-copy 则直接给出锚点精确性。□

### 4.4 训练

冻结目标为 `L=0.5 L_K3 + 0.5 L_K2-middle-drop`。K2 由 K3 三个排序锚点中删除中间点产生，因此 K2 是训练覆盖条件，不能称 unseen-K。每个 checkpoint 训练 20 epochs、320 次 AdamW optimizer updates，并按 source-dev NMAE 选择最早达到最小值的 epoch。

### 4.5 为什么需要 matched context-free control

Context-Free Calibrator（代码名 `static_zero`）与 CalibTM 具有相同 5,475 参数、相同初始化、训练预算和有界外层公式，但共享网络接收的语义条件向量被代数置零。因此，共享 trunk 与 heads 只能学习全局固定系数；这些系数仍作用于同一个由样本锚点、归一化锚点差、`s_f` 与最终观测标准差缩放的外层约束公式。因而 CalibTM 相对该 control 的收益仅识别 **向 MLP 暴露注册 scaffold-derived 通道** 的增量价值，而非“增加一个 MLP”本身，也不是“完全去除所有样本条件”的对照。它不能单独识别 boundedness、Linear scaffold 或区域分解的因果必要性。精确兼容布局与参数账本见补充实现说明。早期 single-seed source-dev prototype probe 显示 `blinear_only` 已解释大部分输入通道增量；local-volatility 的独立增量很小。该 probe 不是三 seed final-gate 消融，本文因而只用它约束机制措辞，不把 rich geometry 或 volatility 解释为核心发现。

## 5 实验设计

### 5.1 数据与证据角色

| 数据集 | 规模/时间粒度 | 主角色 | 关键限制 |
|---|---|---|---|
| [Abilene](https://www.cs.utexas.edu/~yzhang/) | 144 flows；5-min bins | development 后的 historical final gate | 人工 K3；已消费证据 |
| GEANT [20] | 462 flows；15-min bins | development 后的 historical final gate | 人工 K3；已消费证据 |
| SNDlib BRAIN [21] | 161 nodes；14,311 directed pairs；1-min | 冻结后的第三 WAN effect confirmation（Linear/Context-Free） | LP-inverted OD；transductive schema；28 windows |

BRAIN 的 fit/source-dev/confirmation 分别含 136/28/28 个不跨 gap 的 T=50 windows。其 confirmation payload 只在三 seed、两方法的 checkpoint registry 冻结后打开。

### 5.2 Baselines

我们报告 canonical Linear、同参数 `static_zero`、PCHIP、原 INFOCOM 工作的 metric-matched ARI reimplementation，以及 ImputeFormer 官方架构的 bounded task adapter。注册 PCHIP 在内部区间使用标准 shape-preserving cubic slopes，边界则按 first/last anchor 做 nearest-value extension，随后执行非负截断与观测 hard projection；它不是 SciPy 默认边界外推。ARI 与 ImputeFormer 的比较只在 A/G exact historical P0 上完成；BRAIN 没有这两个方法的结果。ImputeFormer 可使用 flow identity、cross-flow attention 和绝对时间，信息类强于 CalibTM，因此本文把它定位为 **modern architecture stress comparator**，而不是信息匹配的机制 control 或跨网络 SOTA 排名依据。

### 5.3 统计与证据分层

A/G 主结果在 dataset×seed 内汇总 internal block 与 two-burst，再按 seed 和 dataset 等权。BRAIN 先在 seed 内汇总两类 structured masks，再对 seeds 7/8/9 等权。注册裁决保留原 paired bootstrap；另做 reviewer-facing 只读敏感性分析：固定三个 seed 等权，仅重采物理时间窗口/连续块，且每次 draw 跨 methods、masks 和 seeds 共享。flow、target、mask 与 seed 都不被当作独立现实样本。CalibTM 是在 A/G development 中通过 matched controls 选出的 operating point，因此 A/G 属于开发后的 historical evidence；BRAIN 是模型与 checkpoint 冻结后的新增 WAN confirmation。head/bound diagnostics、时间块敏感性、场景压力、六方向 frozen transfer 与 RIS measured-mask replay 均为 post-gate descriptive，不改变原 verdict，也不与 formal evidence 合并成 pooled CI。

## 6 结果

### 6.1 三 WAN：Linear、context-free 与 scaffold-conditioned calibration

| 数据/证据 | CalibTM NMAE | Context-Free | Linear | Gain vs context-free | Gain vs Linear |
|---|---:|---:|---:|---:|---:|
| Abilene K3 structured | **0.235979** | 0.239188 | 0.240525 | **+1.3413%** | **+1.8900%** |
| GEANT K3 structured | **0.219307** | 0.220950 | 0.221852 | **+0.7437%** | **+1.1474%** |
| A/G dataset-equal | **0.227643** | 0.230069 | 0.231189 | **+1.0425%** `[0.9175,1.1548]` | **+1.5187%** `[1.3316,1.6709]` |
| BRAIN frozen confirmation | **0.897916** | 0.908268 | 0.928203 | **+1.1396%** `[1.0511,1.2272]` | **+3.2627%** `[3.1243,3.3682]` |

A/G 的四个 structured dataset×mask cells 和三个 seed headline 均为正；但 792 个窗口级比较中有 48 个为负，因此不能声称逐窗口获胜。其 absolute `ΔNMAE` 相对 Context-Free/Linear 分别为 +0.00243/+0.00355；固定 seed、共享物理 window draw 的约 1 天 block 95% CI 分别为 `[+0.8955%,+1.1930%]` 与 `[+1.3551%,+1.6881%]`，约 2 天 block 仍全部为正。BRAIN 的 absolute `ΔNMAE` 为 +0.01035/+0.03029，28/28 物理窗口方向为正，中位 gain 为 +1.1533%/+3.1530%；但 confirmation 仅覆盖约一天、只有两个 UTC start-day clusters，不能声称已经控制日级或周级自相关。

表中的 A/G dataset-equal NMAE 是两个 seed-equal dataset NMAE 的算术平均，而正式 relative gain 是先在配对单元内计算、再按 seed/dataset 等权的 effect；因此不能用显示到六位小数的平均 NMAE 反推置信区间。传统 PCHIP 在同一 structured 协议中比 Linear 差 1.0900%，CalibTM 相对 PCHIP 改善 2.5800%，故它不是本设置中的 strongest baseline。

<!-- FIGURE:GAINS -->

### 6.2 大型 learned stress comparators 的网络异质性

| Dataset | Linear | Context-Free | CalibTM | ARI metric-matched | ImputeFormer adapter |
|---|---:|---:|---:|---:|---:|
| Abilene all masks | 0.237557 | 0.236011 | 0.232858 | 0.230258 | **0.222715** |
| GEANT all masks | 0.217642 | 0.216573 | **0.215160** | 0.227781 | 0.312429 |
| Dataset-equal | 0.227600 | 0.226292 | **0.224009** | 0.229020 | 0.267572 |

CalibTM 相对 ARI 的 all-mask dataset-equal gain 为 +2.19%，95% CI `[1.452,2.980]`，但 Abilene 为 -1.129%、GEANT 为 +5.541%。ImputeFormer 相对 CalibTM 在 Abilene 为 +4.351%，在 GEANT 为 -45.211%。这种大幅反转使 per-dataset 结果比 pooled 排名更重要，也提示 bounded adapter 在 GEANT 上可能存在适配局限。本文据此将两者用于刻画 dataset-sensitive accuracy–cost tradeoff，而不据其宣称稳定领先。

### 6.3 可测收益集中于区域化 bounded offsets

冻结 checkpoint 的 inference-only single-head-zero intervention 得到：

| 置零 head | 合法 target | Full over ablated | 95% CI | 结论 |
|---|---|---:|---:|---|
| `delta_r_head` | internal | -0.00377% | `[-0.02648,+0.02383]` | 无可测附加价值 |
| internal `offset_head` | internal | **+2.49701%** | `[+2.04227,+2.93543]` | 内部收益主要来源 |
| `edge_offset_head` | edge | **+1.12546%** | `[+1.01446,+1.23407]` | 边界收益来源 |

两个 offset 作用于不同区域，单头效应不能相加为严格 decomposition。冻结 checkpoint 的 intervention 表明插值比例修正没有可测附加价值，而内部和边界 offset 分别解释对应区域的主要收益。该分析发生在模型身份冻结之后，因此本文忠实报告原三头 checkpoint；coordinate warp 不作为核心贡献。

冻结 replay 还显示：`delta_r` 平均只占允许半径的 1.67%/3.25%（A/G），且 `r_hat` 从未 clip；internal/edge offset occupancy 为 59.51%/64.57% 与 92.52%/82.48%，非负投影前的负预测比例为 2.19%/9.64%。bounds 与 projection 在当前实现中数值上活跃，但这不是 bounded 相对 unbounded/direct predictor 的 matched 必要性证明。

### 6.4 Measured-availability shape replay

三次 RIPE RIS collector maintenance 的 5-minute payload availability 在 outcome 前固定为 onset/recovery windows，再转移到已消费的 A/G TM。CalibTM 相对 Linear 为正 36/36、相对 Context-Free 为正 35/36；唯一反向是 GEANT–RRC12 onset–seed 1 的 -0.0813%。RRC18/RRC15 同形，因此六个 event-phase 仅有四种 unique masks；RRC12 原样保留 42 observed/8 missing。真实的只有 control-plane availability shape，TM values 仍来自 A/G；这不是 native TM outage、第三 WAN 或 confirmation，GEANT 也只保留 ordinal shape。

### 6.5 Frozen-weight cross-WAN transfer（描述性）

一项全方向 post-gate replay 保持模型权重不变，只应用 target-frozen observation-only preprocessing。六个 structured source→target 方向相对 source Context-Free 的 gain 为 +0.5677% 至 +1.4240%，相对 target Linear 为 +0.9966% 至 +2.7148%；54/54 完整因素均正。三个 target cohorts 已消费，且 BRAIN 的 fit-std fallback 涉及约 4.63% missing-truth mass，因此这是 descriptive frozen-weight transfer，不是 independent test 或 statistic-free universal zero-shot。

### 6.6 Accuracy–efficiency operating point

CalibTM 有 5,475 个 total/trainable parameters，主 checkpoint 为 27,202 bytes；metric-matched ARI 有 84,784,178 个 total parameters、32,012,594 个 trainable parameters，主 checkpoint 为 354,505,976 bytes。参数与 checkpoint 比分别为 15,485.7× 和 13,032.3×，两者不能混为同一压缩指标。

在单张 H800、BF16、seed-1 synthetic shape、batch=8 的冻结 Python pipeline 中：

| Dataset | Linear | Context-Free | CalibTM | ARI | ImputeFormer |
|---|---:|---:|---:|---:|---:|
| Abilene latency ms | **0.340** | 2.867 | 3.171 | 37.372 | 5.836 |
| Abilene peak MiB | **4.835** | 55.413 | 55.413 | 548.876 | 171.361 |
| GEANT latency ms | **0.351** | 2.820 | 2.884 | 73.213 | 14.611 |
| GEANT peak MiB | **15.512** | 104.816 | 104.816 | 791.563 | 467.433 |

跨 batch 1/8/32，ARI 的 median latency 与 peak allocated memory 分别为 CalibTM 的 11.17×–35.12× 和 5.75×–12.58×；ImputeFormer adapter 分别为 1.81×–10.68× 和 1.74×–5.33×。与此同时，Linear 比 CalibTM 快 8.21×–13.91×，Context-Free 也略快且显存相同。正确结论是 CalibTM 位于 deterministic Linear 和大型 learned models 之间，而不是“同时最准且最低成本”。

在同一 Intel Xeon Gold 6530 的未编译 FP32 checked pipelines 上，单线程 batch=1 的 CalibTM latency 为 4.78/17.26 ms（A/G）；ARI 为其 283.8×/186.9×，ImputeFormer adapter 为 37.2×/27.9×。batch=32、8 threads 时 CalibTM 达 561.5/146.7 windows/s，分别为 ARI 的 174.0×/116.7× 和 adapter 的 86.2×/30.9×。Linear 仍更便宜：单窗口快 16.3×/22.9×，吞吐高 18.1×/24.3×。计时排除模型加载与输入构造；这些仍是 seed-1 synthetic shapes 的 pipeline results，不是官方或 architecture-only runtime。

<!-- FIGURE:PARETO -->

## 7 讨论

### 7.1 场景限定的设计原则

实验表明，当前 bounded instantiation 在每条 flow 只有三个锚点时构成一个小幅而可复现的 accuracy–efficiency operating point。Matched context-free control 识别了向 MLP 暴露 scaffold-derived 通道的增量作用；BRAIN confirmation 则表明该 operating point 能延伸到第三个 WAN。它们没有识别 boundedness、区域分解或 Linear scaffold 的因果必要性，也没有构成“低自由度必然优于自由重建”的普遍检验；ARI 和 ImputeFormer 在 Abilene 的优势正好说明高容量模型仍可能在特定网络占优。

### 7.2 与 ARI-LLM 的方法区别

ARI-LLM 的主路径是 dense flow vector → Flow2Vec/LSTM → LLM flow-token mixing → 五阶段 rollout。CalibTM 则采用 observation-only Linear → bounded local operator → hard copy 的单次前向路径。两者是不同的补全主干和信息—成本 operating point，CalibTM 并不依赖 ARI 输出。

### 7.3 新颖性边界

从数学实现看，CalibTM 是 canonical Linear 与小型有界校准网络的组合。它与已有 neural interpolation、prior-guided imputation 的差异不在 MLP，而在精确的任务参数化：锚点最终 hard-copy、内部/边界分区、尺度归一的有界形变，以及不使用可识别 flow 关系的单次前向算子。Matched context-free control 和成本曲线进一步说明该 operating point 在当前设置中具有实证价值。相较已经覆盖简单张量补全的 WTTC-TS、连续函数补全的 ImputeINR 和概率过程建模的 Utimac，本文保持更窄的 deterministic local calibration 定位。

### 7.4 局限

1. 三个 WAN 的主 masks 都是 artificial thinning，没有 native measured TM trace；BRAIN 还是 transductive-schema、LP-inverted OD，confirmation 仅约一天。
2. ARI/ImputeFormer 在 A/G 强烈异质且 BRAIN 无对应结果；H800/CPU 效率也只能解释为各自 checked pipelines，而非架构内禀速度。
3. 没有 matched unbounded/direct arm；edge 只是 distance-invariant segment shift，`vbar_f` 又带来 window-composition dependence，冻结 intervention 也不证明 two-head 重训非劣。
4. cross-WAN transfer 使用已消费 target 与 target-frozen preprocessing，只是 descriptive evidence；方法限于 offline bidirectional repair，且没有 uncertainty/abstention 输出。

## 8 结论

本文提出 CalibTM，一个 5,475 参数的锚点保持有界插值校准器。它用低自由度的一次前向算子整体替换 ARI-LLM 的 LLM 与自回归主干。在 Abilene、GEANT 和冻结 SNDlib BRAIN confirmation 上，CalibTM 相对 canonical Linear 与同参数 context-free calibration 取得小而一致的 structured NMAE 改善；冻结诊断把可测收益定位到 internal pointwise 与 boundary-segment offsets，并表明其幅度界和非负投影实际参与推理。大型 learned baseline 的准确率具有明显网络依赖，而 CalibTM 形成介于 Linear 与这些模型之间的 accuracy–efficiency operating point。

综合这些结果，在极端人工时间稀疏、锚点可信、允许离线双向修复且不使用 topology/routing 的条件下，先固定一个可行插值 scaffold、再学习小幅区域校准，是一种有竞争力的设计选择。该结论限定于本文的信息与评估边界；真实 missing process、direct-OD 外部数据和可信不确定性仍是后续工作。

## 参考文献

[1] F. Yan, K. Jiang, Y. Qiao, M. Li, Y. Li, P. Yu, and C. Feng. ARI-LLM: Autoregressive Imputation for Network Traffic Matrix via Large Language Models. IEEE INFOCOM, pp. 1–10, 2026. DOI: 10.1109/INFOCOM59046.2026.11571176.

[2] T. Nie et al. ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation. ACM KDD, pp. 2260–2271, 2024. DOI: 10.1145/3637528.3671751.

[3] X. Yuan et al. Diffusion Models Meet Network Management: Improving Traffic Matrix Analysis with Diffusion-based Approach. IEEE Transactions on Network and Service Management, 22(2):1259–1275, 2025. DOI: 10.1109/TNSM.2025.3527442.

[4] M. Li et al. 3DDPS: A Traffic Matrix Estimation Method Based on Three-Dimensional Diffusion Posterior Sampling. Computer Networks, 257:111007, 2025. DOI: 10.1016/j.comnet.2024.111007.

[5] T. Miyata. Traffic Matrix Completion by Weighted Tensor Nuclear Norm Minimization and Time Slicing. NOLTA, 15(2):311–323, 2024. DOI: 10.1587/nolta.15.311.

[6] T. Miyata. Completion of Traffic Matrix by Tensor Nuclear Norm Minus Frobenius Norm Minimization and Time Slicing. IEEE/IFIP NOMS, pp. 1–5, 2024. DOI: 10.1109/NOMS59830.2024.10575433.

[7] K. Xie et al. Neural Tensor Completion for Accurate Network Monitoring. IEEE INFOCOM, pp. 1688–1697, 2020. DOI: 10.1109/INFOCOM41043.2020.9155366.

[8] K. Xie et al. Deep Adversarial Tensor Completion for Accurate Network Traffic Measurement. IEEE/ACM Transactions on Networking, 31(5):2101–2116, 2023. DOI: 10.1109/TNET.2022.3233908.

[9] H. Wang et al. PetTC: Pairwise Joint Embedding Based Contrastive Tensor Completion for Network Traffic Monitoring Services. IEEE Transactions on Services Computing, 18(1):357–371, 2025. DOI: 10.1109/TSC.2024.3517331.

[10] F. N. Fritsch and R. E. Carlson. Monotone Piecewise Cubic Interpolation. SIAM Journal on Numerical Analysis, 17(2):238–246, 1980. DOI: 10.1137/0717021.

[11] S. Somnugpong, N. Butploy, and K. Khiewwan. Percentile-Based Slope-Constrained Linear Interpolation for Robust Imputation of Highly Volatile PM2.5 Time Series. MethodsX, 16:103859, 2026. DOI: 10.1016/j.mex.2026.103859.

[12] T. Niedzielski and M. Halicki. Improving Linear Interpolation of Missing Hydrological Data by Applying Integrated Autoregressive Models. Water Resources Management, 37(14):5707–5724, 2023. DOI: 10.1007/s11269-023-03625-7.

[13] P. Yuan, Y. Jiao, J. Li, and Y. Xia. A Densely Connected Causal Convolutional Network Separating Past and Future Data for Filling Missing PM2.5 Time Series Data. Heliyon, 10(2):e24738, 2024. DOI: 10.1016/j.heliyon.2024.e24738.

[14] S. N. Shukla and B. M. Marlin. Interpolation-Prediction Networks for Irregularly Sampled Time Series. ICLR, 2019.

[15] M. Liu, H. Huang, H. Feng, L. Sun, B. Du, and Y. Fu. PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation. IEEE ICDE, pp. 1927–1939, 2023. DOI: 10.1109/ICDE55515.2023.00150.

[16] Z. Liu, X. Zhao, and Y. Song. RDPI: A Refine Diffusion Probability Generation Method for Spatiotemporal Data Imputation. AAAI, 39(12):12255–12263, 2025. DOI: 10.1609/aaai.v39i12.33335.

[17] M. Li, K. Liu, J. Guo, J. Bu, H. Wang, and H. Wang. ImputeINR: Time Series Imputation via Implicit Neural Representations for Disease Diagnosis with Missing Data. IJCAI, pp. 9241–9249, 2025. DOI: 10.24963/ijcai.2025/1027.

[18] X. Liu et al. Rethinking Traffic Matrix Completion: Estimate the Process, Not the Entries. arXiv:2605.02225, 2026.

[19] X. Zeng et al. INT-MC: Low-Overhead In-Band Network-Wide Telemetry Based on Matrix Completion. Proc. ACM Meas. Anal. Comput. Syst., 8(3), Article 37, pp. 1–30, 2024. DOI: 10.1145/3700433.

[20] S. Uhlig, B. Quoitin, J. Lepropre, and S. Balon. Providing Public Intradomain Traffic Matrices to the Research Community. ACM SIGCOMM Computer Communication Review, 36(1):83–86, 2006. DOI: 10.1145/1111322.1111341.

[21] S. Orlowski, R. Wessäly, M. Pióro, and A. Tomaszewski. SNDlib 1.0—Survivable Network Design Library. Networks, 55(3):276–286, 2010. DOI: 10.1002/net.20371.
