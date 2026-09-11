# CalibTM 内部 claim 与投稿检查清单

本文件供组内审稿使用，不进入论文 PDF。

## 当前可以主张

- K3 structured historical gate 上，CalibTM 跨 A/G 优于 Linear 和 matched static；
- frozen BRAIN confirmation 上对 Linear/static 的两个 structured contrast 均为正；
- historical dataset-equal 上优于 metric-matched ARI，但必须同时展示 Abilene 反转；
- 当前 checked H800 pipelines 中比 ARI/ImputeFormer adapter 更省 learned-model 成本；
- 当前 Xeon/FP32 checked CPU pipelines 中，单线程 latency 与 8-thread throughput 均显著优于 ARI/ImputeFormer adapter；同时必须报告 Linear 仍快 16–24×；
- internal/edge offset 有可测贡献，coordinate shift 没有；offset bounds 与非负投影在冻结 replay 中数值上活跃；
- 依赖结构更严格的物理时间块 sensitivity 中，A/G structured effect 仍为正；
- 六方向 frozen-weight transfer 在 target-frozen preprocessing 下均为正，但只能称 post-gate descriptive；
- 三次 RIPE maintenance 的 measured-availability-shape transfer 相对 Linear 为正 36/36、相对 matched control 为正 35/36，但只能称 semi-synthetic replay。

## 当前禁止主张

- 真实 outage、native missing 或线上 94% telemetry saving；
- 稳定超过所有 WAN、所有现代 baseline 或所有 LLM；
- BRAIN 是直接测量 OD，WS-DREAM 是第三 WAN；
- CalibTM 是最低成本方法，或效率倍率属于架构内禀；
- CPU 结果是 compiled/ONNX/官方实现 benchmark，或 whole-process RSS 等同于 allocator memory；
- 三个 head 都必要，local volatility 是核心机制；
- edge head 会按外推距离变化，或模型严格 flow-local / flow-subset-consistent；
- boundedness 已经因果优于 unbounded/direct predictor；
- cross-WAN 结果是 independent confirmation、statistic-free universal zero-shot，或 54 个独立现实样本；
- RIPE replay 是 native TM outage accuracy、第三 WAN，或六个独立 mask replication；
- 方法从一开始就由真实场景先验推导，或实验治理本身构成创新；
- matched static 已经因果证明“低自由度优于自由重建”。

## 当前投稿裁决

`PROCEED_CCF_C_WRITING`：模型身份冻结，不再搜索 backbone、gate、router、
residual、KAN、Mamba 或其他热门模块。当前 reviewer-facing robustness work 已完成；
后续只允许正式投稿模板、复现材料或在确有合法数据/代码时补一项公平 TM baseline，
不得依据现有结果加 edge feature、删 head 重训、扫 bound 半径或救活负结果。
