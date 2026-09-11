# `value_only` / CalibTM 论文工作区

本目录是 2026-08-11 以后唯一允许继续编辑的论文入口。旧的
`experiments/*paper*`、`output/pdf/CAVR_*` 和 ARI-LLM+ 稿件只作历史档案，
不得把其中的 CAVR、rich-ACIL、router、residual 或诊断型结论迁入本文。

## 当前文件

- `main_cn.md`：中文版研究稿的内容真源；
- `METHOD_IMPLEMENTATION_APPENDIX_CN.md`：checkpoint-compatible 槽位、参数与三头说明；
- `SUPPLEMENTARY_RESULTS_CN.md`：时间块、bound、cross-WAN、measured-mask 与 K-curve 补充证据；
- `CLAIM_BOUNDARIES_CN.md`：内部 claim 与投稿检查清单，不进入论文 PDF；
- `refs.bib`：已核对的相关工作种子；
- `scripts/build_cn_pdf.py`：从 `main_cn.md` 生成内部审阅 PDF；
- `figures/`：构建时生成的图像预览；
- `rendered/`：构建后逐页渲染图，仅用于版面检查。

最终 PDF 输出到：

```text
output/pdf/CalibTM_中文方法论文_v3.pdf
```

## 构建

当前节点没有 TeX 工具链，因此中文版使用仓库已有隔离环境中的 ReportLab
生成；它不访问数据、checkpoint、sealed test 或旧 test cache：

```bash
.venv/bin/python \
  paper/value_only/scripts/build_cn_pdf.py
```

正式英文投稿稿应在目标会议及当年模板确定后新建 `main.tex`。不要把内部
中文版的单栏页数当作会议页数。

## 科学身份

论文工作名为 **CalibTM**，代码身份保持 `value_only`。它是 5,475 参数的
observation-derived、scaffold-conditioned bounded calibrator，不是 ARI-LLM 上的
附加模块。主结果只支持：在极端人工时间 thinning 的离线修复设置中，它相对
canonical Linear 和同参数 `static_zero` 取得小而稳定的精度收益，并在当前冻结
H800 Python pipelines 中位于 Linear 与大型 learned baselines 之间的成本点。

不得写成：真实 outage 普遍有效、跨网络稳定超过所有现代基线、直接 OD 实测、
架构内禀加速、最低成本、三个 correction head 都必要，或由 problem-first 假设
事前推导得到。`value_only` 是早期 ACIL 系统收缩中幸存的最小算子；后续
problem-first backbone 与 Tiny-KAN 检验均失败并被停止。

## 数值真源

总索引：`VALUE_ONLY_PAPER_PATH_HANDOFF_2026-08-11_CN.md`。主文数字必须可追溯到：

- A/G：`experiments/minimal_calibrator_final_gate_v1/results/final_gate_r1/final_adjudication.json`
- ARI：`experiments/eohq_v3/manifests/p0_incumbent.json`
- ImputeFormer：`experiments/imputeformer_modern_baseline_v1/results/formal_r1/paired_uncertainty.json`
- BRAIN：`experiments/minimal_calibrator_sndlib_brain_v1/results/formal_r1/adjudication.json`
- 推理效率：`experiments/value_only_inference_pareto_v1/results/h800_bf16_seed1.json`
- 训练更新效率：`experiments/value_only_training_step_efficiency_v1/results/h800_bf16_seed1.json`
- 三头消融：`experiments/value_only_head_ablation_v1/results/adjudication.json`
- 场景压力：`experiments/value_only_scenario_stress_v1/results/adjudication.json`
- 统计稳健性：`experiments/value_only_statistical_robustness_v1/results/analysis.json`
- Bound diagnostics：`experiments/value_only_bound_diagnostics_v1/results/replay_cpu_fp32_r2.json`
- Cross-WAN transfer：`experiments/value_only_cross_wan_zero_shot_v1/results/formal_r1/adjudication.json`
- Measured availability masks：`experiments/value_only_measured_availability_replay_v1/results/adjudication.json`
- CPU 推理效率：`experiments/value_only_cpu_pareto_v1/results/cpu_fp32_seed1.json`
