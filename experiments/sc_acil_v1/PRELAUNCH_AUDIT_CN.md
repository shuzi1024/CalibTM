# SC-ACIL v1 gate 前证据审计

审计日期：2026-07-16（UTC）。适用协议：`sc-acil-v1`。结论：在生成最终 source-tree freeze 且 CUDA smoke 重复一致后，可以启动冻结的 Stage A；本结论不授权 Stage B 或 sealed test。

## 泄漏与移动目标

- 旧 `acil_innovation_v1/runs` 中未发现 `formal_acil` 或 `formal_gate` 结果目录；seeds 4–6 和 gate cohort 尚未被该正式网格执行。
- 新 loader 只有 `(dataset, cohort)`，cohort 白名单为 fit/source-dev/gate，dataset 白名单为 Abilene/GEANT；没有 raw path、split、test 或 cache 参数。
- 底层只读四个规范 permitted arrays，并复算固定语义 SHA-256；不读取/哈希整张原始 CSV，不读取旧 `*_test*.npz`。
- worker 公共签名只有 `(stage, job_id, output_root)`；method、dataset、seed 都由 64 位 job ID 反解，不能从 CLI 覆盖。
- source-dev 只负责 checkpoint 选择；gate 不进入训练、早停或 variant 选择。相同 gate 上禁止修改模型、门槛或 seed。

## 方法与消融可解释性

- ACIL 明确标为 proposed method 的第一阶段和 `w/o self-calibration` 内部消融，不列作外部 baseline。
- `SC-ACIL` 与 `SC-ACIL-u0` 都有 451,017 个可训练参数；相同 seed 下全部初始 state tensors 相等，唯一干预是把 normalized scalar LOO innovation 通道置零。
- 两者都不使用 flow ID、跨流聚合、拓扑或路由；都保持非负输出与观测点 hard projection。
- missing truth 在 model input 中被 NaN 替换。虽然 batch 对象为 loss/metric 保留 truth，但模型调用只收到 `model_input, observed, fit_fallback`；hidden-payload invariance 有聚焦测试覆盖。

## 网格、载体与来源

- `fit_acil` 恰好 6 jobs：2 datasets × 3 preregistered seeds。
- `formal_gate` 恰好 12 jobs：2 residual methods × 2 datasets × 3 seeds。
- u0 job 唯一携带 Linear、ACIL-only、u0 记录；full job 只携带 full 记录。合并后每个 `method×seed×dataset×mask×window` 恰好一次。
- 每个 residual job 必须加载同 dataset×seed 的 ACIL checkpoint，并核对 manifest、checkpoint identity、文件和 tensor hashes；formal queue 在 6 个 ACIL jobs 全部成功前拒绝启动。
- NaN/OOM 作为 algorithmic failure 落盘；统计禁止只对幸存 jobs bootstrap。基础设施失败保留 partial artifact 和日志，只允许原 job ID 原命令重试。

## 指标与 gate

- 原始证据先按 window 对 flows 求和，保留 absolute-error 与 absolute-truth 两个 ratio-of-sums operands；不平均 per-flow NMAE。
- 主效应在每个 dataset 内 pool structured masks、seeds、windows 后算 improvement，再对两个 dataset 等权，防止 Abilene 的 72 个窗口压过 GEANT 的 16 个窗口。
- paired bootstrap 先重采样 seed，再按 dataset 用长度 4 的 circular window blocks 重采样；candidate/comparator 共用完全相同的 draw plan。
- 必须同时通过 full-over-ACIL、full-over-matched-u0、random no-harm、per-dataset、cell win/worst-cell 和 positive-seed 条件；不存在 gate-set revise 路径。
- Linear 只是 Stage-A 经典 sanity baseline。原始 ARI-LLM、ImputeFormer 和观测率/K 泛化必须在 Stage A proceed 后用新协议补齐，当前 Stage A 单独不能构成投稿证据。

## 验证状态

- 新包聚焦测试：19/19 passed。
- 旧 ACIL + 新 SC-ACIL 联合套件：295/295 passed，使用独立 fresh pytest basetemp。
- Python runtime 固定为现有实验环境 `experiments/od_orbit_gpt_v1/.venv_runtime_v1/bin/python`；它与旧成功 GPU jobs 的 runtime 一致。
- 尚待：最终 freeze 后的 CUDA deterministic smoke（两次 forward/backward hash 一致）。该项失败则不得启动正式队列。

## 审计判定

`PROVISIONAL PROCEED TO FROZEN CUDA SMOKE, THEN FORMAL STAGE A ONLY`。

这不是方法效果判定。效果判定只能来自完整 formal grid 与冻结 adjudicator；oracle/headroom 与旧 tune 结果不得替代 gate 结果。
