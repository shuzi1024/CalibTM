# CalibTM 组会文献速查

这份不是要在十分钟里逐条讲，而是防止老师追问“依据是什么”“别人做过没有”时只能凭印象回答。

## 最接近当前方法的工作

### DCCN-SPF：Linear 后再学习修正

- P. Yuan, Y. Jiao, J. Li, and Y. Xia, *A Densely Connected Causal Convolutional Network Separating Past and Future Data for Filling Missing PM2.5 Time Series Data*, Heliyon, 2024.
- DOI: <https://doi.org/10.1016/j.heliyon.2024.e24738>
- 原文链接：<https://www.sciencedirect.com/science/article/pii/S2405844024007692>
- 可以说：它先用 Linear 表示总体趋势，再让卷积网络学习剩余细节，说明“确定性底稿 + learned correction”有直接先例。
- 不能说：它证明了 CalibTM 的 bounded correction。它面向 PM2.5，读取缺口前后序列，网络和约束都不同。

### PriSTI：插值结果可以作为 learned imputation 的先验

- M. Liu et al., *PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation*, IEEE ICDE, 2023.
- DOI: <https://doi.org/10.1109/ICDE55515.2023.00150>
- 作者版：<https://arxiv.org/abs/2302.09746>
- 可以说：现代时空补全也会先构造粗插值/条件特征，再把它作为生成模型的 context prior。
- 不能说：PriSTI 支持 5K 逐点 MLP、hard-copy 或本文的 10% 修正界。

### RDPI：从确定性初值出发学习 residual

- Z. Liu, X. Zhao, and Y. Song, *RDPI: A Refine Diffusion Probability Generation Method for Spatiotemporal Data Imputation*, AAAI, 2025.
- 官方页面：<https://ojs.aaai.org/index.php/AAAI/article/view/33335>
- 可以说：它先产生 deterministic preliminary estimate，再把初值和真值之间的 residual 作为 diffusion target。
- 不能说：它的 residual 有硬上界或模型很轻；它仍是两阶段扩散模型。

## 约束、锚点与训练方式的依据

### PCHIP：形状保持插值是经典研究问题

- F. N. Fritsch and R. E. Carlson, *Monotone Piecewise Cubic Interpolation*, SIAM Journal on Numerical Analysis, 1980.
- DOI: <https://doi.org/10.1137/0717021>
- 可以说：插值不是只能追求平滑；主动保持数据形状、避免不合理振荡是经典思路。
- 不能说：PCHIP 等于 CalibTM 的最终 bit-exact hard-copy，也不能证明 learned offset 合理。

### Slope-Constrained Linear：在 Linear 周围加显式护栏有近期先例

- S. Somnugpong, N. Butploy, and K. Khiewwan, *Percentile-Based Slope-Constrained Linear Interpolation for Robust Imputation of Highly Volatile PM2.5 Time Series*, MethodsX, 2026.
- 全文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC13011222/>
- 可以说：该工作先做 Linear，再根据历史一阶差分阈值裁剪不合理斜率，说明“保留简单插值并限制异常变化”是合理研究路线。
- 不能说：它支持本文固定的 0.10/0.25；两者限制的量完全不同。

### SAITS：人工隐藏再监督是常见 masked-imputation 训练方式

- W. Du, D. Côté, and Y. Liu, *SAITS: Self-Attention-based Imputation for Time Series*, Expert Systems with Applications, 2023.
- 作者版：<https://arxiv.org/abs/2202.08516>
- 可以说：从已有观测中再隐藏一部分、使用隐藏真值监督模型，并不是临时为当前实验创造的训练逻辑。
- 不能说：SAITS 的网络、损失或 mask 协议与 CalibTM 完全相同。

## TM 领域中的位置

### SRMF：TM 恢复中早已有“联合结构 + 局部插值”

- M. Roughan, Y. Zhang, W. Willinger, and L. Qiu, *Spatio-Temporal Compressive Sensing and Internet Traffic Matrices (Extended Version)*, IEEE/ACM Transactions on Networking, 2012.
- DOI: <https://doi.org/10.1109/TNET.2011.2169424>
- 可以说：经典 TM 恢复会把低秩时空建模和 local interpolation 结合，插值在 TM 中不是随手选的预处理。
- 不能说：SRMF 证明单条 flow 的 Linear 在 K=3 下很强；它利用整个 TM 的联合结构。

### WTTC-TS：TM completion 也存在无需预训练的简单路线

- T. Miyata, *Traffic Matrix Completion by Weighted Tensor Nuclear Norm Minimization and Time Slicing*, NOLTA, 2024.
- 正式全文：<https://www.jstage.jst.go.jp/article/nolta/15/2/15_311/_pdf>
- 可以说：复杂深网不是 TM completion 的唯一选择；该工作强调无需预训练和更简单的问题形式。
- 不能直接说：它一定是低延迟推理。它仍涉及非凸优化和张量分解。

### ARI-LLM 与 ImputeFormer：当前直接比较对象

- F. Yan et al., *ARI-LLM: Autoregressive Imputation for Network Traffic Matrix via Large Language Models*, IEEE INFOCOM, 2026. DOI: <https://doi.org/10.1109/INFOCOM59046.2026.11571176>
- T. Nie et al., *ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation*, ACM KDD, 2024. 官方实现：<https://github.com/tongnie/ImputeFormer>
- 可以说：ARI-LLM 代表 Flow2Vec/LLM/多阶段自回归 TM 修复；ImputeFormer 代表带低秩归纳偏置的现代时空 Transformer。
- 必须补充：PPT 中 ARI 数值是同任务指标下的 reimplementation，ImputeFormer 数值是官方架构的本任务 adapter；不是照搬两篇论文的官方表格。

## 一句总括

> 这些文献支持的是研究动机和设计脉络，不是“别人已经证明 CalibTM 正确”。本文自己的证据仍然是同参数对照、三个 WAN 的结果和成本测量；本文也不把 Linear、MLP、插值先验、hard-copy 或约束单独当作创新。
