# CalibTM v4 方法创新探索（2026-09-11）

## 结论先行

当前 `value_only` 的方法新颖性确实有限。它最容易被概括为“canonical Linear + bounded residual MLP”，现有结果更适合支撑一个场景化 operating point，而不是新的学习原语。

如果目标是实质提高方法层面的创新，合理方向不是把热门模块缝到现有 MLP 上，而是重新定义校准对象：从逐点校准器改成**gap-level、几何显式、锚点零边界的受约束算子**。该方向必须作为独立 v4 研究，不能覆盖或改写当前 v3 证据。

## 为什么当前方法不够新

已有工作已经分别覆盖了以下思想：

- 插值先提供趋势、模型再学习残差或细节；
- 插值结果作为扩散或生成模型的先验；
- 形状保持、斜率约束和锚点精确；
- 低秩/张量结构下的 TM 补全；
- 过程建模和不确定性驱动的 TM completion。

近期 TM 工作仍在推进高容量的非连续缺失补全，例如 [PTC](https://www.sciopen.com/article/10.26599/BDMA.2026.9020029) 使用 PatchTransformer 和时空交互模块处理随机及整片时间缺失；[Utimac](https://arxiv.org/abs/2605.02225) 则把问题改写成局部过程参数估计并显式处理不确定性。因此继续添加一个 attention、KAN、Mamba 或 diffusion block，既不能避开已有碰撞，也不能解决当前方法的核心批评：输入几何贫乏、edge 段内修正为常量、没有证明 boundedness 的必要性。

## 推荐的 v4：GapCalib 算子

### 1. 输入对象

对每个连续缺口单独构造无量纲几何：

\[
u_t=\frac{t-t_L}{t_R-t_L},\quad
d=\frac{t_R-t_L}{T-1},\quad
\Delta=y_R-y_L,\quad
q=\frac{y_R-y_L}{t_R-t_L+\epsilon}.
\]

内部缺口输入左右锚点、归一化位置 `u`、gap length `d`、归一化端点差/斜率和观测尺度。边界缺口输入最近两个可见锚点、单侧距离 `u_edge`、方向 `s∈{-1,+1}`、gap length 和观测尺度。删除当前依赖 batch 内 `max_g s_g` 的 `vbar`，让每条 flow 的输出只由自身观测和显式 gap geometry 决定。

### 2. 输出参数化

内部缺口先计算 canonical Linear：

\[
b(u)=(1-u)y_L+uy_R.
\]

学习器输出低维 basis 系数 `c`，但残差强制在两个锚点处为零：

\[
r(u)=u(1-u)\,\sum_k c_k B_k(u,d,q),
\qquad
\hat y(u)=b(u)+\sigma_f\,\beta_o\tanh(r(u)).
\]

边界缺口使用单侧 basis，并令最近锚点处残差为零：

\[
r_{edge}(u)=u\,\sum_k c_k B_k(u_{edge},d,q,s),
\qquad
\hat y_{edge}=y_{anchor}+\sigma_f\,\beta_e\tanh(r_{edge}).
\]

最后继续执行非负投影与观测 hard-copy。`B_k` 可以是固定 cubic B-spline 或少量 Bernstein basis；第一版只允许一种 basis，禁止同时搜索 KAN、Chebyshev、Fourier 和 spline 变体。

### 3. 真正可证明的性质

若模型只接受单 flow 观测与该 gap 的几何量，v4 可以预注册并测试：

1. **Anchor exactness：** 缺口端点通过 `u(1-u)` 或 `u` 因子严格回到锚点；观测位置最终 hard-copy。
2. **Positive-scale equivariance：** 对 `x' = a x`、`a>0`，归一化前的预测满足 `f(x')=a f(x)`，直到非负投影或 fallback 触发。
3. **Reflection consistency：** 左右端点交换、`u→1-u` 后，内部预测曲线镜像一致。
4. **Flow-subset independence：** 增加或删除另一条 flow 不改变当前 flow 的输出；这一点要求移除 window-level cross-flow maximum。
5. **Bounded gap deformation：** 输出偏离 Linear 的幅度由显式 `β` 和观测尺度控制，而不是依赖输出层偶然学到的小数值。

这些是算子性质，不是“某个数据集上多了 1%”的经验包装。它们也能直接回应当前 frozen `value_only` 的两个具体限制：edge 段内 constant shift，以及 `vbar` 造成的 batch composition dependence。

## 预注册实验，不得边跑边改

在任何训练前固定以下内容：

- 方法：`GapCalib`、当前 `value_only`、`Context-Free`、canonical Linear；必要时加入一个参数匹配 direct/unbounded control；
- 输入：同一 K=3/T=50 协议，GapCalib 只能使用单 flow 观测、gap geometry 和观测尺度；
- 输出：内部与边界各一组 basis，不能再加 router、attention、cross-flow encoder 或 learned prior；
- 训练：相同 20 epochs、320 updates、三 seeds、相同 source-dev 选择规则；
- 新确认：必须使用当前 formal 结果未消费的时间段、数据集或明确独立的公开数据；不能把现有 A/G/BRAIN gate 重新当成 v4 confirmation；
- 主要指标：absolute NMAE、relative gain、tail-5% window degradation、hard-copy/非负/性质测试；
- 停止条件：若新算子没有在两个数据集上相对 Linear 同向，或相对当前 value_only 的改进只在单一数据集/单一 mask 出现，则停止 v4，不加特征、不调 basis、不改 bound。

## 第二候选：风险控制层

如果真正目标是提升论文问题贡献，而不是点预测器的结构新颖性，可以把 v4 改成“selective TM repair”：冻结当前点预测器，再用时间分块 conformal score 估计修复风险，在 CalibTM 与 Linear 之间选择，必要时 abstain。它可以回答“什么时候应该相信小幅校准”，但属于可靠性/风险控制论文，需要 coverage、interval width、risk-coverage 和 review budget；它不能作为当前 `value_only` 的点预测创新直接拼接。

## 不推荐的方向

- KAN：已经做过参数匹配 transplant，未超过原 MLP；继续换 basis 是结果驱动搜索。
- Mamba、Transformer、Diffusion、Neural CDE：会重新引入高容量、长序列或不规则路径假设，与当前极少锚点和 topology-free 边界不匹配。
- 单独删除 `delta_r` 或加入距离/斜率：可以改善现有模型表达力，但属于对 frozen 实现的局部修补，不能自动形成方法创新。
- 继续扫 `β`、K、mask phase 或 direct/unbounded 组合：除非先写独立 protocol 并获得新 confirmation，否则会回到二阶 auto-research。

## 最终路线选择

当前建议保留两个互不污染的分支：

1. **v3 投稿分支：** 立即成稿，主张场景化、轻量、有界 calibration operating point；接受方法新颖性中等。
2. **v4 探索分支：** 只有在导师明确要求提高技术创新时才启动；先冻结 GapCalib 的数学定义和性质测试，再决定是否投入训练与新确认。

不要把 v4 的潜在结果写回 v3，也不要在 v3 论文中暗示 GapCalib 已经存在。当前最重要的决策不是选择哪个热门模块，而是决定是否愿意为一次真正的方法重构支付新的数据、训练和失败成本。
