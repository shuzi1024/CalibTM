# CalibTM v4 模块缝合实验协议（KAN / Local Attention / Mamba-lite）

版本：2026-09-11  
用途：在不改变问题边界的前提下，判断高级序列模块是否能把 GapCalib 从“有界残差 MLP”提升为有实证支持的方法组件。本文是一次性 bake-off 协议，不允许在结果出来后继续扩展候选模块或调参。

## 核心原则（轻量 CCF-C 版本）

模块可以构成组合创新，但必须满足：所有候选方法处理同一 gap 对象、看到完全相同的 observation-only 信息、使用相同的有界输出壳和 anchor-copy 规则；唯一系统性变化是 gap 内部的表示模块。这样结果能够回答“某个模块是否适合这个 gap operator”，而不是回答“某个模型搜索过程是否偶然找到更好的数字”。

v3 `value_only` 不被改写。v4 只在独立分支中运行；模块比较的目的，是判断是否值得把 GapCalib 升级成论文主模型，而不是开启新的无边界搜索。

## 统一的 GapCalib host

对每条 flow 的每个连续缺口先构造一个 observation-only descriptor，再由固定的四项 Bernstein basis 表示整段 gap 的 deformation。模块作用在 gap coefficient 上，而不是把每个缺失点当成独立预测任务。缺口信息只允许使用：

\[
q_j=[u_j,d,s,y_L/\sigma_f,y_R/\sigma_f,q_{LR},\log(1+\ell),\text{region}],
\]

其中 `u_j` 只用于最终 basis evaluation，`d` 是归一化 gap 长度，`s` 是左/右边界方向（内部缺口固定为 0），`y_L,y_R` 是相邻观测锚点，`q_LR` 是由锚点计算的局部斜率，`ell` 是缺口长度，`region` 是 internal/edge 指示。所有量均由该 flow 的观测锚点和训练集冻结尺度得到，不使用 topology、routing、flow identity、其他 flow 或目标值。四个 coefficient token 共享该 descriptor，并分别附带固定 basis index `k=0,...,3`。

主 scaffold 为：

\[
b_j=(1-u_j)y_L+u_jy_R
\]

默认 basis 固定为 cubic Bernstein：`B_k(u)=C(3,k)u^k(1-u)^(3-k)`，内部和边界共用这四个 basis；不在训练后改换 spline、Fourier、Chebyshev 或其他 basis。

内部与边界分别使用固定的 anchor-zero 因子：

\[
\phi_{int}(u)=u(1-u),\qquad \phi_{edge}(u)=u.
\]

每个候选模块只输出四个标准化 basis coefficients `c_0,...,c_3`，统一通过：

\[
\hat y_j=b_j+\sigma_f\beta_{region}\tanh\!\left(\phi_{region}(u_j)\sum_{k=0}^{3}c_kB_k(u_j)\right)
\]

再执行非负投影和观测位置 hard-copy。任何候选都不得改变这层 wrapper。边界候选必须显式使用 `u_j` 和方向，不能退化成当前 frozen implementation 的段内常量偏移。

## 四个预注册 arm

所有 arm 共享上述输入、输出壳、数据、损失、优化器和初始化种子。

1. `GapCalib-MLP`：两层 pointwise MLP，作为几何 gap operator 的基础实现。
2. `GapCalib-KAN`：仅把 coefficient-token 的 pointwise 映射替换为一层固定配置的 KAN edge function。固定 grid=4、spline degree=3、无自适应 grid、无额外 basis 搜索。
3. `GapCalib-Attn`：一个单头 local self-attention block，只在同一 gap 的四个 coefficient tokens 之间计算，`d_model=16`；禁止对 observed anchors 做 query-time attention，禁止跨 gap、跨 flow 和跨 window attention。这一限制避免退化为已有 interpolation-attention 范式。
4. `GapCalib-Mamba`：一个单层、固定四 token 顺序的 Mamba-lite selective state-space block，`d_model=16`、state size=4、expand=1；顺序由 basis index 固定。为避免方向偏置，正式实现同时跑正序与反序并平均两者输出，或使用预先固定的双向共享参数实现；不得为该选择另行搜索。

冻结的 v3 `value_only`、Context-Free 和 canonical Linear 作为外部参考臂只做推理，不参与 v4 训练或候选选择。这样可以区分“几何 host 本身的变化”和“KAN/Attention/Mamba 模块的增量”。

若正式 Mamba 依赖无法复现，则使用仓库内固定版本的最小 selective-SSM 实现，并在报告中明确写成 Mamba-lite，而不是泛称 Mamba。KAN、Attention 和 Mamba 的具体实现版本、配置和 commit hash 在训练前写入 registry。

## 参数与计算预算

目标预算为当前 `value_only` 的 5,475 个可训练参数。每个 arm 的可训练参数必须落在 `[5,200,5,750]`（约 ±5%）；LayerNorm、projection、embedding、SSM state 参数全部计入。禁止通过增加参数弥补模块差异。

宽度只允许使用训练前按参数公式确定的一组值，不根据结果回调。若某模块无法进入预算，则记录为 `budget-infeasible`，不放宽预算。除参数量外报告单窗口 FLOPs、CPU batch=1 latency 和 peak allocated memory，避免把高参数量误报成轻量改进。

## 训练和数据协议

- 观测协议固定为 `K=3,T=50`；主结果使用 registered random、internal-block、two-burst 三类 mask，分别报告，不能只汇总胜者。
- A/G source-dev 使用与 v3 相同的物理窗口抽样规则；每个 arm、mask、seed 共享同一批窗口和初始化 seed 映射。冻结的 v3 `value_only` 在这些窗口上仅作配对参考。
- 两到三个优化 seed（优先沿用 v3 的 `17,29,43`）；训练 epoch、update 数、batch、optimizer 和 checkpoint 选择规则沿用 v3。若运行成本有限，先用两个 seed 做筛选，再对胜者补第三个 seed。
- 损失、归一化、非负投影、hard-copy 和评估代码不因候选模块改变。
- 不做候选专属学习率、宽度、dropout、grid、state size 或 mask 调参。所有配置在第一条训练命令前写入 JSON registry 并计算 SHA。

## 预先固定的决策规则（不做复杂统计门槛）

开发阶段只允许在 A/G source-dev 上运行这一组四臂比较。以 paired window-level NMAE 为统计单位，所有方法对同一底层窗口配对。

先设一个 host gate：若 `GapCalib-MLP` 相对冻结的 `value_only` 没有约 `0.5%` 的 pooled structured 改善，或在 A/G 方向相反，则停止 v4；此时继续比较 KAN/Attention/Mamba 没有解释意义。

候选模块满足以下大多数条件即可进入一次小规模复核：

1. 相对 `GapCalib-MLP` 的 pooled structured gain 至少 `0.50%`，且 A、G 两个 WAN 的点估计均为正；
2. A/G structured 的配对差异方向基本一致；
3. 三类 mask 中至少两类同向，且没有明显灾难性退化；
4. anchor exactness、非负性和 deformation bound 三个基本性质通过；
5. 推理成本没有明显超过当前 MLP（建议不超过 3 倍）。

若多个模块通过，按以下固定顺序选一个：先比较 fresh-confirmation 前的 pooled gain，再比较 worst-5% gain，最后比较 CPU latency；不进行二次训练择优。若没有模块通过，v4 结论为“高级模块未带来可复现增益”，保留 `GapCalib-MLP` 或回到 v3，不继续添加候选。

## 小规模复核（可选，但建议做）

候选选择后固定代码和配置，在一个尚未用于模块筛选的时间片或 BRAIN 子集上复核即可。它的作用是避免把 A/G 的偶然排序写成主结论，不要求另建复杂数据治理流程。

复核使用相同的 K=3、mask 和指标。若胜者仍相对 MLP 与 Linear 保持同向小幅增益，就可以作为 v4 主结果；若复核失败，直接保留 v3，不再继续找新模块。

## 必须报告的结果

主表至少包括：absolute NMAE、relative gain、每个 WAN×mask cell、参数量和一次 latency 对比。若已有脚本方便，再补 FLOPs、memory 和 worst-5% gain；这些不是启动实验的前置条件。主图包括：

- 四臂在三个 mask 和两个 WAN 上的 paired gain；
- 一个 gap 内 residual 随 `u` 的曲线，展示 edge 是否真正距离感知；
- 若做小规模复核，再报告复核子集的 window-level scatter；
- accuracy–cost Pareto 图，保留 canonical Linear 作为最低成本参照。

任何“模块有效”的表述必须同时给出相对 `GapCalib-MLP` 和相对 Linear 的结果。不得把单一 seed、单一 mask 或单一数据集胜利写成方法结论。

## 停止与解释规则

- 四臂均失败：说明 gap geometry 尚不足以支持这些模块，停止 v4 模块搜索。
- 只有一个模块在 source-dev 获胜、fresh 失败：说明开发集过拟合或模块与网络分布不匹配，不修改协议、不换候选。
- 候选只在一个 WAN 获胜：可作为描述性异质性结果，不升级为方法创新。
- 候选提升来自更大的参数预算、不同预处理或更长训练：结果无效，回退并重跑统一预算。

即使某个模块通过，论文的创新表述也应是“geometry-explicit, anchor-preserving gap operator with [module]”，而不是“首次提出 KAN/Attention/Mamba 插值”。新颖性来自受约束 gap operator 与模块的共同设计；如果只有数字略好而没有机制差异，就仍按增量改进表述。

## 实际执行顺序（控制工作量）

1. 先做一次 data-free forward 检查，确认四臂都能保持锚点和非负输出。
2. 第一轮只在 A/G 的 `internal-block` 与 `two-burst` 两类代表性 mask 上跑四臂、两个 seed；不为任何模块单独调参。
3. 若某一模块相对 MLP 的 pooled gain 达到约 `0.5%` 且两个 WAN 不反向，再补 `random` mask、第三 seed和一个小规模复核。
4. 只有通过复核的模块才替换 v3 主模型；否则停止 v4，直接写 v3。整个过程最多一轮，不继续添加新结构。
