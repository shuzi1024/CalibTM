# Minimal Calibrator v1：Prototype 通过交接（2026-07-24）

## 结论

正式裁决为 `proceed`，进入 `value_only / local-scale conditioning` 分支。

这一轮支持的是一个很小的结论：在完全相同的插值公式、参数量和训练流程下，让校准器看到当前线性插值值与局部尺度，确实比只学习三个全局校准系数更好。它不支持重新引入 rich anchor/gap 特征，也不支持把单 seed 结果直接写成论文证据。

当前协议到此结束，不能直接追加 seed 2/3。下一步必须另冻 checkpoint-bank 和未打开 gate 的确认协议。

## 已确认事实

- 固定 prototype 网格为 bundle 1 × Abilene/GEANT × static-zero/Blinear-only/value-only，共 6/6 个 primary jobs。
- 六个任务均一次成功；无 missing、retry、replacement、OOM、NaN 或结果后改 seed。
- 每个任务均完成 20 epochs、每 epoch 64 physical batches、总计 320 optimizer updates。
- 三臂参数量均为 5,475，同一数据集内初始 `state_dict` 逐位一致。
- 三臂只改变共享 MLP 前的固定输入通道：
  - `static_zero`：16 个通道全零；
  - `blinear_only`：只保留 `B_linear`；
  - `value_only`：保留 `B_linear / x_obs / mask / local_volatility`。
- `static_zero` 不是“无几何模型”：外层公式仍使用左右 anchor、距离、anchor delta 和 local volatility。它表示三个不随样本变化的全局校准系数。
- `value_only` 与前一冻结协议中的同名实现，在参数、预测和梯度上 bit-exact。

### Source-dev checkpoint

| 数据集 | 方法 | best epoch | source-dev NMAE |
|---|---:|---:|---:|
| Abilene | static-zero | 8 | 0.2018181 |
| Abilene | Blinear-only | 18 | 0.1988745 |
| Abilene | value-only | 19 | 0.1986299 |
| GEANT | static-zero | 6 | 0.2217608 |
| GEANT | Blinear-only | 19 | 0.2192386 |
| GEANT | value-only | 18 | 0.2189468 |

所有 checkpoint 都已独立验证为 source-dev NMAE 的最早最小值。

## 冻结结果

### 主要对照：value-only 相对 static-zero

- structured dataset-equal 改善：`+0.8359%`
- paired circular-window bootstrap 95% 区间：`[+0.5658%, +1.1517%]`
- Abilene：`+1.5092%`
- GEANT：`+0.1626%`
- 两个数据集均为正，0.5% 主效应门槛与 CI 门槛均通过。
- 最差 structured cell 为 GEANT/internal-block：`-0.0136%`，通过 `-0.5%` no-harm 门槛。
- 最差 random dataset 为 GEANT：`-0.4168%`，通过 `-0.5%` no-harm 门槛。

### 相对普通线性插值

- value-only 相对 Linear：`+1.2354%`
- Abilene：`+1.9934%`
- GEANT：`+0.4774%`
- 1% 总体门槛与两个数据集均为正的门槛都通过。

### Blinear-only 简化探针

- Blinear-only 相对 static-zero：`+0.7652%`
- bootstrap 95% 区间：`[+0.5125%, +1.0594%]`
- Blinear-only 相对 value-only 点估计：`-0.07146%`
- 非劣 bootstrap 下界：`-0.104545%`
- 预注册非劣门槛：`-0.1%`

它只差约 `0.004545` 个百分点，但冻结规则仍要求判失败。因此不能在这一轮选择 Blinear-only；正式原因是 `value_scale_conditioning_supported`。

## 推断

- 前一阶段发现的约 1.25% learned-interpolation 增益，不只是三个全局静态校准系数：样本相关输入贡献了约 0.84% 的 structured 相对改善。
- `B_linear` 已解释大部分条件化收益，local volatility 带来的增量很小，但按预注册非劣规则仍不足以删掉。
- 因此当前最合适的方法不是大模型或 residual 网络，而是一个受插值先验约束的极小条件校准器。
- rich anchor/gap 特征没有必要；不能恢复此前已被否定的 ACIL feature-attribution 叙事。

## 证据边界与身份

- `git_available=false`
- `git_commit=null`
- source-tree SHA-256：`2dc6c977a52d133f9f37ea7e8be420a9ebf76ed04ca3d369ff57e83fc281dfd7`
- research-card SHA-256：`75b7e6f1e5ce71887d3f438e9f5961efba5ffab9dfbcc7fa529c9a714fefb13c`
- freeze 文件 SHA-256：`bd659ef11b93e3c99e9f59cbe89a923ebedfd336c1acb70d31d680eafbc4b4a4`
- freeze 权限：`0444`
- adjudication SHA-256：`9420b274dcca282fb7cd0d3c31597b1a7854a9236576f62f8a6cf06b742069b2`
- Abilene tune：72 个窗口 `[33884,37484)`。
- GEANT tune：15 个窗口 `[7572,8322)`。
- `test_access=false`；未读取 gate、GEANT test `[9172,10772)`、whole raw CSV 或旧 test cache。

## 验证与独立审计

- 新目录 focused tests：`24 passed`。
- 相关回归：`338 passed`。
- repo-level：`2274 passed, 21 failed, 4 errors, 1 skipped, 23 warnings`。
- repo-level 红项与此前历史基线相同：旧 operational inventory 与 OD-ORBIT runtime/provenance 漂移；本阶段新增回归为 0。
- 四卡 CUDA forward/backward smoke 通过；GPU2/GPU3 的 value-only loss、预测与梯度 digest 完全一致。
- 第一次 smoke 的诊断断言错误地要求 `x_obs/m` 在整个返回张量都为零；它们只在 missing targets 为零，在 observed hard-projection 位置非零。该失败已保留，冻结源码没有改变，第二次 smoke 通过。
- adjudication 用同一 10,000-draw 计划独立重跑，输出逐字节一致。
- 独立只读 evidence audit 重新加载六个 checkpoint，复算 result/manifest/data/mask/window/denominator、earliest-min checkpoint、ratio-of-sums、bootstrap 与全部 gates，结论同为 `PROCEED`。

## 未知与限制

- 当前只有一个训练 seed；window bootstrap 不能替代 seed 与 checkpoint-selection 不确定性。
- tune 已参与方法发现，`paper_evidence=false`。
- `value_only` 相对 static-zero 在 GEANT 上只有 `+0.1626%`，跨 seed 稳定性仍未知。
- local volatility 相对只看 Blinear 的增量约 `0.0715%`，方法解释应保持克制。
- 当前只覆盖 K=3、两个数据集和三种 mask；不同观测稀疏度与 WS-DREAM transfer 未知。
- 当前指标是 completion NMAE，不能外推为 routing 或 operational benefit。
- 未打开的 final gate 仍保持封存。

## 失败与停止动作

- 科学失败：Blinear-only 没过冻结的 value-only 非劣门槛。
- 正式 GPU 基础设施失败：无。
- smoke 诊断失败：1 次，原因与记录见上，未修改冻结源码。
- 当前协议中 seed 2/3：不得直接启动。
- 不允许根据本结果改变 prototype gate、补挑 seed 或恢复 rich feature 分支。
- GPU 队列退出后 occupancy 已自动恢复；占卡不属于研究证据。

## 关键产物

- 研究卡：`experiments/minimal_calibrator_v1/research_card.json`
- 只读 freeze：`experiments/minimal_calibrator_v1/freezes/minimal_calibrator_v1.json`
- 正式结果、manifests、logs 与 checkpoints：`experiments/minimal_calibrator_v1/results/prototype_r1/`
- 正式裁决：`experiments/minimal_calibrator_v1/results/prototype_r1/adjudication.json`
- 裁决 manifest：`experiments/minimal_calibrator_v1/results/prototype_r1/adjudication.json.manifest.json`
- queue 退出记录：`experiments/minimal_calibrator_v1/results/prototype_r1/queue_exit.json`
- 四卡 smoke：`experiments/minimal_calibrator_v1/smoke/preformal_cuda/smoke.json`
- 本交接：`experiments/minimal_calibrator_v1/handoffs/2026-07-24-prototype-proceed-handoff-zh.md`

## 复现与核验

核验 freeze：

```bash
PY=experiments/od_orbit_gpt_v1/.venv_runtime_v1/bin/python
PYTHONDONTWRITEBYTECODE=1 "$PY" - <<'PY'
from experiments.minimal_calibrator_v1.freeze import probe_identity
print(probe_identity())
PY
```

重新验证六个 primary results 并裁决到临时目录：

```bash
PY=experiments/od_orbit_gpt_v1/.venv_runtime_v1/bin/python
OUT="$(mktemp -d /tmp/minimal-calibrator-adjudication-XXXXXX)/adjudication.json"
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  -m experiments.minimal_calibrator_v1.adjudicate \
  --result-root experiments/minimal_calibrator_v1/results/prototype_r1 \
  --output "$OUT" \
  --bootstrap-draws 10000
sha256sum "$OUT"
```

预期 adjudication SHA-256：

```text
9420b274dcca282fb7cd0d3c31597b1a7854a9236576f62f8a6cf06b742069b2
```

## 研究者下一步应理解什么

现在第一次有了值得继续确认的方法信号：一个很小的输入条件化校准器，能稳定地胜过同参数量的静态系数控制，而不需要 LLM、residual 大网或 rich geometry 特征。但这仍只是 tune 上的单 seed 发现。

最干净的下一步不是再跑一轮 tune 选模型，而是：

1. 另冻 checkpoint-bank 协议；
2. 复用并重审 seed 1 的 value/static checkpoints；
3. 只训练 bundles 2/3 的 value/static 共 8 个 checkpoint-only jobs，训练只访问 fit/source-dev；
4. 冻结 12 个 checkpoint 的完整身份与 gate thresholds；
5. 最后一次性打开未使用的 gate，运行 3 bundles × 2 datasets 的 6 个 paired jobs。

只有 final gate 通过，才可以把这一信号升级为方法论文的核心证据。
