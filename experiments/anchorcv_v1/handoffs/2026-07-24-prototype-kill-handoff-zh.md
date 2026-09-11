# AnchorCV v1 Prototype 终止交接（2026-07-24）

## 结论

正式裁决为 `kill`。两项 prototype 作业及完整性审计均成功，但“隐藏一个中间观测点后比较两位专家误差”的路由信号接近随机，不能启动 bundles 2/3 extension。

## 已确认事实

- 正式网格为 seed bundle 1 × {Abilene, GEANT}，完成 2/2，失败 0，未替换 seed。
- 证据边界为 branch-sealed permitted fit/source-dev/tune；`test_access=false`，未读取 sealed test、旧 test cache 或 whole raw CSV。
- Expert N 完整训练 20 epochs / 320 optimizer updates。Abilene source-dev NMAE 从 0.20368 降至 best 0.19286；GEANT 从 0.23024 降至 best 0.22159。
- 四个 structured dataset×mask cell 的逐 case 隐藏真值 oracle 相对最佳单专家都有正空间，范围为 5.73%–7.90%，dataset-equal 汇总为 6.81%。
- deployable LOO 路由的 AUROC 为 0.5015，Spearman 为 -0.000068，几乎没有预测能力。
- deployable LOO 路由相对最佳单专家的 structured 汇总改善为 -2.52%；最差 cell 为 GEANT/internal-block，-3.74%。
- oracle gap capture 为 -0.374；七个冻结 method gates 中仅两个 oracle-headroom gates 通过，其余五个失败。
- 完整性审计 `valid=true`、`issues=[]`；checkpoint、evidence、mask、source/config/data/result hashes 均核对，并完成 CUDA checkpoint replay。

## 推断

- P/N 专家不是完全同质：Abilene 上 N 三种 mask 均优于 P；GEANT 上 P 优于 N 的 random/internal-block，而 N 优于 P 的 two-burst。
- 当前失败不应归因于“N 没训练起来”，而是单个中间 anchor 的 K2 误差不能代表 K3 missing complement 上的相对误差。
- 理论互补主要发生在 window×flow case 层面，不是稳定的 flow 身份：事后 static-flow oracle 只有约 0.46%–2.98% 空间；按前后半窗口交叉迁移后约为 -0.45%–0.11%。
- 对 LOO 分数做前后半交叉阈值校准也不能稳定改善；六个 cell 只有 GEANT/two-burst 为 +0.36%，其余持平或变差。因此不能把失败解释成“阈值 0 没调好”。

## 未知与限制

- 这里只否定 v1 的“sorted-middle single-anchor route”，不否定专家互补本身。
- 尚未检验多 anchor 交叉验证、训练集监督的风险估计或其他 observation-only 信号；它们需要新问题、新协议和新冻结，不能在 v1 内追加。
- prototype 只有 bundle 1；由于 method gate 已明确失败，按 stop rule 不增加 bundles 2/3。
- 发现是 branch-sealed，不是项目级 pristine holdout；项目其他目录存在历史 test artifacts。

## 失败记录

- 第一次 r1 smoke 启动命令漏设 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，四个进程均在首个 cuBLAS 运算前退出，未生成科学结果；占卡脚本自动恢复。以固定确定性环境重试后成功。
- AnchorCV focused/full tests：21/21、329/329 通过。
- 直接依赖测试：ACIL 276/276 通过；SC2 45 passed、1 skipped。
- 历史 ORBIT 全套为 869 passed、19 failed、4 errors；失败集中在其既有 runtime/provenance 固化与当前环境漂移（包版本、旧接口、source snapshot 分类），不由 AnchorCV 改动引起，也未在本阶段扩大范围修复。

## 冻结与产物

- 冻结记录：`experiments/anchorcv_v1/freezes/anchorcv_v1_full_pipeline_r1_20260724T0010Z.json`
- `git_available=false`
- `git_commit=null`
- config SHA-256：`f15b89cf0ae6ba041c7fb2561c457ef2ea2e0f2011ab4b263bcf0a5bb735a451`
- source-tree SHA-256：`36bede29f4ea2c3eae67985eb14aab76cf51d599075cdbcee689a08fce506f3c`
- freeze-file SHA-256：`105872ef1732819211b7d171fd33431e0f782aa9d950f0ff7a89c638540615b9`
- deterministic smoke：`experiments/anchorcv_v1/results/formal_v1r1/cuda_smoke/`
- 正式结果：`experiments/anchorcv_v1/results/formal_v1r1/prototype/`
- 裁决报告：`experiments/anchorcv_v1/results/formal_v1r1/reviews/prototype.json`
- GPU continuity log：`/gfs/space/private/suuuz/anchorcv_occupancy_operational.log`

## 复现命令

核验冻结：

```bash
experiments/od_orbit_gpt_v1/.venv_runtime_v1/bin/python -c \
  'from experiments.anchorcv_v1.freeze import verify_freeze_record; verify_freeze_record("experiments/anchorcv_v1/freezes/anchorcv_v1_full_pipeline_r1_20260724T0010Z.json"); print("ok")'
```

核验现有正式作业并输出队列汇总（不会重训已完整作业）：

```bash
experiments/od_orbit_gpt_v1/.venv_runtime_v1/bin/python \
  -m experiments.anchorcv_v1.launcher \
  --stage prototype \
  --output-root experiments/anchorcv_v1/results/formal_v1r1/prototype \
  --freeze-record experiments/anchorcv_v1/freezes/anchorcv_v1_full_pipeline_r1_20260724T0010Z.json \
  --gpus 0,1,2,3
```

重新审计（需要 CUDA 确定性环境）：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \
  experiments/od_orbit_gpt_v1/.venv_runtime_v1/bin/python \
  -m experiments.anchorcv_v1.review \
  --output-root experiments/anchorcv_v1/results/formal_v1r1/prototype \
  --freeze-record experiments/anchorcv_v1/freezes/anchorcv_v1_full_pipeline_r1_20260724T0010Z.json \
  --output /tmp/anchorcv-v1-review-repeat.json
```

## 研究者下一步应理解什么

AnchorCV v1 已经回答了一个有价值的否定问题：单点 LOO 适合提供“修正量”时，不一定适合在两个异质专家之间做 hard routing。下一轮不应给当前 selector 加层或补 seed；应先用便宜 probe 证明新的 observation-only 信号确实能预测 case-level expert regret，或者回到 repo 中已有稳定正证据的 Local-LOO observation-consistency correction，补齐 matched controls 与跨基线防御。
