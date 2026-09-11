# ACIL Mechanism v1：Prototype 终止交接（2026-07-24）

## 结论

正式裁决为 `kill`。不得启动 bundles 2/3，也不得根据这批结果修改 feature mask、gate 或补挑 seed。

这次否定的是一个很窄的机制解释：ACIL 相对普通线性插值的收益，不能归因于 MLP 额外看到的 anchor/gap 特征通道。它没有否定“可学习插值”本身；三种 5,475 参数的小模型都比普通线性插值好，而删掉大部分特征后几乎没有损失。

## 已确认事实

- 固定网格为 seed bundle 1 × Abilene/GEANT × full/value-only/no-anchor，共 6/6 个 primary jobs。
- 六个任务均一次成功；无 missing、OOM、NaN、retry、替换 seed 或结果后调参。
- 每个任务均训练 20 epochs、完成 320 optimizer updates，参数量均为 5,475。
- 同一 dataset/seed 下三种方法初始权重逐位一致，训练数据、窗口顺序、mask、目标、预算、优化器和 checkpoint 选择规则完全一致。
- 唯一方法差异是进入共享 MLP 前的固定特征通道 mask。
- source-dev 最优 checkpoint：
  - Abilene：full/value-only/no-anchor 都是 epoch 19；
  - GEANT：full 为 epoch 19，value-only/no-anchor 为 epoch 18。

### 冻结 headline

- full 相对 Linear 的 structured dataset-equal 改善为 `+1.2505%`：
  - Abilene：`+2.0877%`；
  - GEANT：`+0.4134%`。
- full 相对 value-only 仅为 `+0.0160%`：
  - Abilene：`+0.0962%`；
  - GEANT：`-0.0643%`；
  - 配对 circular-window bootstrap 95% 区间：`[-0.0605%, +0.0919%]`。
- full 相对 no-anchor 仅为 `+0.0521%`：
  - Abilene：`+0.1011%`；
  - GEANT：`+0.0031%`；
  - 配对 circular-window bootstrap 95% 区间：`[-0.0036%, +0.1049%]`。
- GEANT random mask 上，full 相对 value-only 为 `-0.5149%`，略低于预注册的 `-0.5%` no-harm gate。
- 只有 Abilene 同时支持 full 优于两个 matched controls；预注册要求是 2/2 数据集。
- 11 个 gate 中 7 个失败，正式原因是 `acil_feature_attribution_gate_failure`。

## 推断

- ACIL 类模型相对普通线性插值的单 seed 优势约为 1.25%，但新增的 rich anchor/gap MLP 输入没有解释这部分优势。
- value-only 和 no-anchor 在相同参数量下几乎追平 full；因此不能把现有收益写成“显式局部几何特征让模型理解 gap/anchor 结构”。
- 当前更合理、也更需要先证伪的解释是：收益可能只来自一个很小的可学习校准器，甚至只是几个全局偏置，而不是复杂的观测自适应规则。
- 下一阶段应先做 bias-only / linear-only / value-only 的最小机制筛查。若 bias-only 也能追平，就应停止把 ACIL 包装成方法创新。

## 证据边界与身份

- `git_available=false`
- `git_commit=null`
- source-tree SHA-256：`1be43dbe9433f05b05e63dc2485b8c7835126ad262c693574ef31327a31a5598`
- research-card SHA-256：`8e9b229d45da892c8a573d519d65d9893eb4a788626d2d7bae4f6ecb5fd09170`
- freeze 文件 SHA-256：`26a6bcda73f8d42e12ab8505495e84deb53481c36c8a6160b6da70fabbf2ab34`
- freeze 权限为只读 `0444`。
- Abilene tune 为 72 个窗口 `[33884,37484)`。
- GEANT tune 为 15 个窗口 `[7572,8322)`。
- `test_access=false`；未读取 GEANT test `[9172,10772)`、whole raw CSV 或旧 test cache。

## 验证与独立审计

- focused tests：`19 passed`。
- 相关回归：`643 passed, 22 warnings`。
- repo-level：`2247 passed, 21 failed, 4 errors, 1 skipped, 23 warnings`；红项均属于已有 OD-ORBIT/旧实验环境漂移，本阶段新增回归为 0。
- 四卡 CUDA forward/backward smoke 通过；full 在 GPU0/GPU3 的 loss 与预测/梯度 digest 完全一致。
- stored adjudication 独立精确重建一致，SHA-256 为 `977a11b790b183d0bf4ba15552191040265b24c9b75eb9b4c04373877090b1cb`。
- 独立只读 evidence audit 重新核验六份 result/manifest/checkpoint、数据与 mask 身份、denominator、10,000 次 bootstrap 和全部 gates，结论同为 `KILL`，明确禁止 bundles 2/3。

## 未知与限制

- prototype 只有一个 seed；bootstrap 只描述固定 seed 下的窗口变动，不能代表 seed 或 checkpoint-selection 不确定性。
- tune 已用于 discovery，不能把本结果当作最终独立确认。
- 本阶段只覆盖 K=3、两个数据集和三种固定 mask。
- controls 只遮掉 MLP 可见的部分通道；外部插值公式仍使用 anchor、距离、gap 类型和尺度。因此本实验不能声称“去掉了所有几何信息”。
- 指标只有 completion NMAE，不能外推为 routing 或 operational benefit。

## 失败与停止动作

- 科学失败：额外 ACIL feature-channel attribution gate 明确失败。
- 基础设施失败：无。
- bundles 2/3：不得启动。
- 不允许在这个冻结协议内补特征、改 gate、换 seed 或扩大网络。
- GPU 正式队列结束后 occupancy 已恢复；占卡状态不属于研究证据。

## 关键产物

- 冻结研究卡：`experiments/acil_mechanism_v1/research_card.json`
- 不可变 freeze：`experiments/acil_mechanism_v1/freezes/acil_mechanism_v1.json`
- 六个正式结果、manifests、logs 与 checkpoints：`experiments/acil_mechanism_v1/results/prototype_r1/`
- 正式裁决：`experiments/acil_mechanism_v1/results/prototype_r1/adjudication.json`
- 裁决 manifest：`experiments/acil_mechanism_v1/results/prototype_r1/adjudication.json.manifest.json`
- 四卡 smoke：`experiments/acil_mechanism_v1/smoke/preformal_cuda/smoke.json`
- 本交接：`experiments/acil_mechanism_v1/handoffs/2026-07-24-prototype-kill-handoff-zh.md`

## 复现与核验命令

核验 freeze：

```bash
PY=experiments/od_orbit_gpt_v1/.venv_runtime_v1/bin/python
PYTHONDONTWRITEBYTECODE=1 "$PY" - <<'PY'
from experiments.acil_mechanism_v1.freeze import probe_identity
print(probe_identity())
PY
```

重新裁决现有六个 primary results 到临时目录：

```bash
PY=experiments/od_orbit_gpt_v1/.venv_runtime_v1/bin/python
OUT="$(mktemp -d /tmp/acil-mechanism-adjudication-XXXXXX)/adjudication.json"
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  -m experiments.acil_mechanism_v1.adjudicate \
  --stage prototype \
  --result experiments/acil_mechanism_v1/results/prototype_r1/abilene_seed1_full_gpu3.json \
  --result experiments/acil_mechanism_v1/results/prototype_r1/abilene_seed1_value_only_gpu3.json \
  --result experiments/acil_mechanism_v1/results/prototype_r1/abilene_seed1_no_anchor_gpu3.json \
  --result experiments/acil_mechanism_v1/results/prototype_r1/geant_seed1_full_gpu0.json \
  --result experiments/acil_mechanism_v1/results/prototype_r1/geant_seed1_value_only_gpu1.json \
  --result experiments/acil_mechanism_v1/results/prototype_r1/geant_seed1_no_anchor_gpu2.json \
  --output "$OUT"
sha256sum "$OUT"
```

不要对当前目录裸用 `--result-root`：该 CLI 会把同目录的 `launch_manifest.json` 也纳入 glob。stored adjudication 已由 manifest 显式绑定六个 primary inputs，因此不影响本阶段证据或结论。

## 研究者下一步应理解什么

这一轮最重要的不是“ACIL 有 1.25% 提升”，而是“这 1.25% 几乎不需要 rich feature channels”。在继续写方法故事之前，必须先回答最简单的问题：模型是否真的根据当前观测自适应，还是只学到了一个对所有样本都差不多的静态插值校正。这个问题只需要一个很小的 matched-control 实验，不需要 LLM、残差大网或复杂 selector。
