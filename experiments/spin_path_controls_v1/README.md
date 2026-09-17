# SPIN + Direct 的匹配路径对照

本目录新增独立训练的路径对照，复用旧模型，不修改历史实现与结果。

| variant | 解码器输入 | 保留路径 |
| --- | --- | --- |
| `spin_direct` | `[Df, Db, normalize(Q(H))]` | 原 `spin_direct_no_memory`，可直接加载旧模型及优化器状态 |
| `context_only` | `[0, 0, normalize(Q(H))]` | 一层 SPIN、上下文查询与原解码器；实际删除 Direct 参数 |
| `direct_only` | `[Df, Db, normalize(Q(P))]` | 原位置编码、本流双向真实观测读取与原解码器；实际删除观测编码器和 SPIN |

`H` 是整窗 SPIN 上下文；`P` 是时间位置与流身份编码。查询始终按原实现分为 4 个头、每头 32 维，分别归一化。Direct-only 的查询不读取流量值或掩码，目标流的预测不依赖其他流的当前窗口值、掩码或图边；拟合期全局均值与尺度仍由共同协议固定。

三种变体先按旧 no-memory 模型的完整次序初始化，再删除停用模块，因此同种子下所有共享参数初值逐位相同。`spin_direct` 的 `state_dict` 与 `named_parameters()` 顺序也与旧版本完全相同。没有新增参数。两个单路径模型不是等参数量对照，Context-only 也不替代官方 SPIN 基线。

接口：`build_model(num_flows, init_seed=41001, variant="spin_direct")`；保留 `configure_fit_statistics`、`configure_from_bundle`、`set_execution`、`predict_block` 与 `forward`。两个新增单路径对照从各自初值独立训练；组合延续原本从头训练的 120 轮轨迹，不使用推理时关闭分支代替训练。

Context-only 保留原解码接口，因此首层有 32,768 个权重对应恒为零的输入。总参数量不能解释为相同有效容量。本轮对照检验整条信息路径的使用价值，不隔离每一种容量或参数化影响。

## 合成检查

在安装了本项目 PyTorch 依赖的环境，从仓库根目录执行：

```bash
python -m experiments.spin_path_controls_v1.checks
```

检查仅使用 CPU 合成 T50 数据：旧模型正向、梯度和 AdamW 更新兼容；共享初始化；隐藏真值不影响预测；Direct-only 严格隔离别流与图、位置查询不读取上下文；Context-only 删除 Direct；有效梯度、全失观测安全性与状态重载。不会读取真实 WAN 数据或使用 GPU。

## 本轮训练协议

完整计划见 [PLAN_CN.md](../../analysis/spin_path_controls_20260917/PLAN_CN.md)。先 GEANT，三个模型、两个配对初始化；AdamW 学习率在第 1—120 轮为 0.001，第 121—160 轮为 0.0001，无早停。数据、掩码、损失和检查点选择沿用旧协议。

`spin_direct` 从旧 fixed120 `last.pt` 恢复模型、优化器及随机数状态；新对照独立完成 160 轮。新目录单独保存第 120 轮预算截面的 `result120.json`、`best120.pt`、`last120.pt`、`history120.json`，旧文件只读。

单任务入口如下；`RUN_DEADLINE_UNIX` 必须来自当前 GPU 授权，不能照搬已过期的执行记录：

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv/bin/python -m experiments.spin_path_controls_v1.runner \
  --dataset geant --variant context_only --seed 41001 \
  --deadline-unix "$RUN_DEADLINE_UNIX" --smoke

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv/bin/python -m experiments.spin_path_controls_v1.runner \
  --dataset geant --variant context_only --seed 41001 \
  --deadline-unix "$RUN_DEADLINE_UNIX"
```

已有本轮 `last.pt` 时使用 `--resume`。该研究入口仍依赖仓库历史数据注册表、规范数据与父检查点；只下载源码不能独立复现全部历史资产。本轮没有修改这些资产依赖。

结果属于开发阶段的路线筛选，160 轮不等于充分收敛。匹配比较包括完整种子、条件与轨迹；旧 Full/四层 SPIN 的不同预算与配方会单独标示。
