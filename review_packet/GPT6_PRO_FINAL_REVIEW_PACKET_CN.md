# GPT-6 Pro 最终方案审阅包

## 请先阅读的文件

1. `CalibTM_v3_CN.pdf`：当前 CalibTM v3 主论文。
2. `SUPPLEMENTARY_RESULTS_CN.md`：主要结果与效率证据（可选附件）。
3. `CALIBTM_V4_MODULE_BAKEOFF_PROTOCOL_2026-09-11_CN.md`：拟执行的 v4 模块实验协议。
4. `CALIBTM_V4_NOVELTY_EXPLORATION_2026-09-11_CN.md`：GapCalib 的创新定位与近邻风险。
5. 原始 ARI-LLM 论文 `ARI-LLM_paper.pdf`：只需重点理解问题定义、ARI-LLM 主流程和它与 CalibTM 的差异。

## 当前已知事实

- v3 `value_only` 约 5,475 个参数，使用 K=3/T=50 的极端时间稀疏设置。
- 它围绕 canonical Linear scaffold 做 observation-conditioned bounded calibration，保留观测点并执行非负投影。
- A/G structured 结果相对 matched static control 和 Linear 均有约 1% 的稳定增益；BRAIN 有冻结后的第三 WAN confirmation。
- ARI-LLM 与 ImputeFormer 的结果存在明显跨网络异质性；Linear 仍是最低成本基线。
- 当前没有声称普遍优于自由重建，也没有 native missing-TM trace。
- 已有 Tiny-KAN transplant 结果未显示相对 MLP 的明确增益，因此 KAN 只是预注册 challenger，不是预设主模型。

## 审阅目标

请判断：

1. Gap-level geometry + anchor-zero bounded wrapper 是否足以构成 CCF-C 方法贡献；
2. KAN、local Attention、Mamba-lite 是否适合只作为 coefficient generator；
3. 四个 coefficient tokens 是否过短，是否需要改成八个固定 shape tokens；
4. 第一轮实验是否已经足够，哪些建议属于过度工程化；
5. 是否建议冻结方案并开始实现，而不是继续搜索模块。

请以 reviewer 风险和论文可发表性为中心，不要泛泛建议增加更多模型。
