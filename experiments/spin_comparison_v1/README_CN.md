# SPIN 官方结构的 CalibTM 任务适配 v1

此目录只实现一个外部基线：**SPIN-adapted**。它对应 NeurIPS 2022 论文 *Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations*，不是自行设计的注意力网络借用 SPIN 名称，也不是原论文数据集主表复现。根任务负责 GPU 分配、正式注册、训练和评分，本目录不启动后台工作。

## 固定来源

- 作者仓库：https://github.com/Graph-Machine-Learning-Group/spin
- 固定提交：`7349ba31da7306e7e96c13668a3f1f0a4df90902`
- 论文：https://arxiv.org/abs/2205.13479
- 本发布包含 CPU oracle 实际读取的固定上游源码子集、配置及 SPIN/TSL 两份 MIT LICENSE；完整 tar 与论文 PDF 保留在本地研究归档，未随此次发布。原文可经 [官方入口](../../review_packet/spin_direct_sync_20260916/04_SPIN_SOURCE.md) 阅读。
- 上游声明依赖 `torch_spatiotemporal==0.1.1`；对应 `TorchSpatiotemporal/tsl` 的 `v0.1.1` 源码子集同样提供；完整源码为本地归档。该 tag 内部 `__version__` 仍写 0.1.0，来源标签和文件散列为身份依据。
- `source_model.py` 与官方 `spin/models/spin.py` 仅替换导入路径和类型别名；`positional.py` 与官方位置编码类仅替换导入路径；`scheduler.py` 与官方完全逐字一致。

`attention.py` 使用纯 PyTorch 实现原始加性时空注意力的相同计算，保留参数名称和形状。`primitives.py` 实现所需的 TSL/PyG 小算子，避免在 H20 的 Torch 2.2 环境安装旧版二进制扩展。`checks.py` 会实际执行归档的官方 SPIN/TSL 源码，并仅以 CPU gather/scatter shim 替代 PyG dispatch，再验证四层输出与所有参数梯度；并非只对照本实现的另一种调用。

## 原算法保留部分

使用官方 `config/imputation/spin.yaml`：hidden=32、4层、eta=3、message_layers=1、独立的本节点时间注意力与图边跨节点时间注意力、softmax 重权、输入 skip、层归一化、每层 MLP readout。

每条空间边分别对源时间点做加性注意力 softmax，然后对进入目标节点的边求和。前三层空间读取只使用真实观测位置；第四层按官方 eta 转换加入 valid/masked 节点嵌入，并允许空间传播已经构建的缺失位置表示。本节点时间注意力始终使用观测掩码且排除同一时刻。它不读取隐藏真值，也不把第四层的中间表示误称为真实观测。

保留官方 `TSL LayerNorm` 的 `(x-mean)/(std+eps)`，不擅自换成 Torch 默认的平方根方差公式。保留 TSL sparse-softmax 分母 `5e-8`。纯执行层面的分块与 activation checkpointing 不改变图、层数或监督目标；全图始终参与前向传播。空源集合额外返回零，避免 `-inf-(-inf)` 的 NaN；当前统一协议每条流均有多个真实观测，这个保护不改变正常样本。

## 必要任务适配

1. 输入是现有 Abilene 144 / GEANT 462 个 OD 流节点、原生间隔 T=50。复用既有完整 fit 历史及固定 fit/source_dev/tune 时间段、512个训练窗口、三种20%可见掩码和开发 mask71001；不引入新数据、观测率或图搜索。
2. 使用既有 fit-only 正相关邻流图，去掉显式 self-edge，保留每个目标的既定有向入邻居。Self 由官方独立时间注意力处理。官方交通传感器实验使用对称距离图，这里替换为 OD 流的 fit 相关图；这是明确的图输入适配，不是 SPIN 原文图复现。
3. 官方日/周日历协变量替换为统一任务可用的窗口内原生相对时刻 `t/49`，`u_size=1`。保留原节点嵌入和32维标准正弦位置编码。其他方法也不接收额外日历信号。
4. **内部归一化保留官方全局 StandardScaler**：官方 `axis=(0,1)` 对 `[time,nodes,channels]` 产生一个总体均值/标准差，而非逐流标准化。均值用登记的 fit `C`（非负流量下等于总体均值）；方差由 fit 的 `global_scale² + mean((mu-C)²)` 重构。`mu` 来自父协议 FP32 数组，只有该均值离散项继承 FP32 精度；不读取额外时间范围。
5. 使用统一三条件混合训练输入及缺失位置监督，替代官方随机选择20%/50%/80% whiten 的数据流程。训练和评估始终只有真实可见值作为输入；完整 fit 值只作已授权离线监督。
6. 保留官方 Adam lr=0.0008、weight_decay=0、有效 batch=8、clip=5、最多300 epochs、patience40，以及原12 epoch warmup / 三次 cosine restart / min_factor0.1 / linear_decay0.67。调度器按**完成的 epoch**更新。根运行器保存 scheduler state 以便精确续跑。每 epoch 使用同一512个登记窗口，因此64次有效更新；不把原仓库针对步长1大数据的 `batches_epoch=300` 变成重复抽样扩大训练数据。
7. 保留最终层 L1 加前三层 L1、每项权重1的深监督；L1 在内部全局标准化尺度上计算，仅监督统一缺失位置。最终输出在原始流量空间评分，统一非负投影并拷回真实观测。
8. checkpoint/early-stop 使用与其他方法相同的三个条件平均 missing-only NMAE，替代原数据集的 `val_mae`。本改动保证统一选型目标；300轮封顶且仍接近峰值时必须报告收敛限制。
9. 使用 FP32 以匹配本项目统一数值检查，原 YAML 为混合精度16。物理batch和节点分块只影响执行效率，最终实际数值写进根注册表。

## 根运行器 API

```python
from experiments.spin_comparison_v1.models import build_model, make_optimizer_scheduler, train_step

model = build_model(bundle.flows, init_seed=41001)
normalization_record = model.configure_from_bundle(bundle)
model.set_execution(node_chunk=32, gradient_checkpointing=True)
model = model.to(device)
optimizer, scheduler = make_optimizer_scheduler(model)
# 每个有效 batch 调 train_step；target_block 必须 >= bundle.flows。
# scheduler.step() 每个完整 epoch 末调用一次，保存/恢复其 state_dict。
```

`predict_block(...)` 接受原统一接口并返回 raw 预测；`return_aux=True` 另给四层标准化输出。图传播始终全图，评估必须 `target_block=bundle.flows`，避免无意义地为不同目标重复图传播。`train_step` 接受原统一参数签名，返回 loss/gradient_norm/missing_targets/supervised_readouts。

CPU 正确性检查：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spin_comparison_v1.checks
```

CPU oracle 通过且根任务的两 WAN 实际形状 GPU preflight 通过后，才可冻结正式队列。不得仅靠参数形状匹配或合成前向成功宣称适配正确。

## 实验解释边界

这是官方 SPIN 架构在同一 CalibTM 任务、图信息和训练历史上的适配。原论文的图、时间窗口、日历协变量和 missingness 任务均不同；无论胜负，都不能把本轮结果写成原论文复现失败或全面超越 SPIN。仅运行两个 WAN 的固定配置，不进行超参网格或另加外部方法。
