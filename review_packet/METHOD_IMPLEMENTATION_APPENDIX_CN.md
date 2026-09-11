# CalibTM checkpoint-compatible 实现说明

本文件记录中文版主文中有意移出的工程细节，用于复现冻结的 `value_only`
checkpoint；它不是额外方法贡献，也不进入当前论文 PDF。

## 有效信息与固定槽位

科学定义只使用四个语义量：canonical Linear `B_linear`、观测值
`x_obs`、mask `m` 和窗口 maximum 归一化后的 `vbar_f`。代码中第四个 active
slot 的字段名沿用 `local_volatility`，但其实际内容是 `vbar_f`；外层 geometry 中
同名字段则是未再除以窗口 maximum 的归一化坐标量 `s_f`。冻结实现将四个语义量写入
一个 16 槽的固定 masked feature layout：`B_linear`、`x_obs`、`m`、
`vbar_f` 分别位于 slots 0、1、2、13，其余 12 槽恒为零。零槽不提供
额外输入信息，但 LayerNorm 在 16 个槽上共同归一化，因此不能把现有
checkpoint 事后等价改写成 `Linear(4,64)`。

这一 16 槽布局来自早期 matched-control 兼容需求。论文方法身份仍由上述四个
语义量决定；若未来发布精简的 4 槽重训版本，应把它视为新模型并重新评估，
不能将现有结果直接迁移过去。

## 由有效槽位导出的能力边界

在缺失位置，`x_obs=m=0`。对同一 flow 的同一侧边界段，nearest-anchor
`B_linear` 与 `vbar_f` 也保持不变，所以四个有效槽、`edge_offset_head` 以及最终
prediction 均为段内常量。冻结模型因此实现的是 anchor-dependent、distance-invariant
的 boundary-segment shift，不是显式依赖距离或方向的外推曲线。

`vbar_f` 的分母是当前窗口内所有 flow 的 `s_f` maximum。它对 flow 排列不变，也不含
flow identity 或逐点相关路径；但加入或移除一个高 `s_f` flow 可能改变其他 flow 的
输入。因此冻结实现不是 strictly flow-local 或 flow-subset-consistent。主文据此不主张
distance-aware edge extrapolation，也不主张完全零 cross-flow aggregate。

## 参数账本

共享 trunk 的精确顺序为 `LayerNorm(16) -> Linear(16,64) -> GELU ->
Linear(64,64) -> GELU`，然后连接三个独立的 `Linear(64,1)` 标量 head。

- LayerNorm(16)：32 参数；
- Linear(16,64)：1,088 参数；
- Linear(64,64)：4,160 参数；
- 三个 Linear(64,1) 标量 head：195 参数；
- 合计：5,475 个 total/trainable parameters。

`static_zero` 使用完全相同的参数拓扑、初始化与训练预算，只把 16 个输入槽
全部代数置零。它因此保持严格的参数匹配，但并非完全去除所有
样本依赖：共享 trunk/heads 只学习全局固定系数，这些系数仍被代入同一个由当前
样本锚点、`anchor_delta` 和 `local_volatility` 缩放的外层约束公式。因而
`value_only` 相对 `static_zero` 的 matched contrast 仅隔离向 MLP 暴露样本条件
通道的增量价值。

## 冻结三头

冻结 checkpoint 包含 `delta_r_head`、`offset_head`（内部缺口）和
`edge_offset_head`。后验 single-head-zero intervention 显示
`delta_r_head` 没有可测附加价值，而两个 offset 分别贡献于内部与边界目标。
为避免消费结果后重新选模，当前论文忠实报告原三头 checkpoint；主文不把
coordinate warp 作为核心贡献。

## 冻结约束诊断

在已消费 A/G gate 上的 18-cell CPU-fp32 replay 中，`delta_r` 平均占允许半径的
1.67%/3.25%，且 `r_hat` clip 从未触发；internal offset occupancy 为
59.51%/64.57%，edge offset occupancy 为 92.52%/82.48%；非负投影前的负预测比例为
2.19%/9.64%。固定输入的 function slices 在六个 checkpoints 间，internal/edge 的
pairwise correlation median 为 0.9856/0.9910，`delta_r` 为 0.6769。

这些数字是 post-gate descriptive diagnostics，没有把重复 mask/seed 坐标当作独立统计
样本，也没有 CI。它们只说明 bounds 与 nonnegative projection 在冻结实现中数值上活跃；
不能证明 bounded 优于 unbounded/direct predictor，也不授权删 head、改 feature 或调
`beta` 后继续选择模型。完整报告与图见
`experiments/value_only_bound_diagnostics_v1/REPORT_2026-08-12_CN_r2.md` 与
`output/pdf/CalibTM_bound_usage_and_frozen_slices.pdf`。

## 代码真源

精确模型和外层约束以以下文件为准：

- `experiments/minimal_calibrator_v1/model.py`
- `experiments/acil_innovation_v1/acil.py`
- `experiments/acil_innovation_v1/acil_core.py`
- `experiments/minimal_calibrator_final_gate_v1/`

数值与证据 authority 见本目录 `README.md`。
