# 中文论文初稿：SPIN 上下文与 Direct 观测直读

[阅读四页 PDF](build/manuscript.pdf) · [LaTeX 主文件](manuscript.tex) · [最终实验表](../../analysis/spin_sync_unified_20260916/FINAL_RESULTS_CN.md) · [独立 ICC 审阅](../../analysis/spin_sync_unified_20260916/INDEPENDENT_ICC_REVIEW_CN.md)

## 本稿对应什么

本稿对应 2026-09-16 从头联合训练的一层 SPIN＋Direct＋Sync，以及独立训练的去 Sync 模型 SPIN＋Direct。两 WAN、两变体、两个初始化共八条 120 轮轨迹已完成。正文保持一幅结构图、一张精度表和一张成本表。

加入 Sync 后，Abilene 的平均 NMAE 基本持平，GEANT 增加约 1.41%；平均 NRMSE 分别改善约 0.86% 和 1.24%，批量为 8 时每窗耗时增加约 64% 和 62%。本稿据此优先讨论 SPIN＋Direct，同时保留完整模型的平方误差收益与成本。

这是**中文内部研究初稿**。独立 ICC 审阅倾向 Weak Reject：当前仅有开发评价，尚未单独证明 Direct 的增量，旧参考的层数与训练配方也不同；固定 120 轮结束不等于充分优化。初稿没有声称已完成独立最终测试或已具备投稿说服力。作者信息保留待填写。此前 `paper/icc_cn_20260916/` 的 481 参数残差头稿件属于另一模型，未被覆盖。

## 格式与构建

采用 `IEEEtran` 的 `conference,10pt,letterpaper` 双栏样式，依据 [IEEE 官方论文模板说明](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/) 和 [IEEE 会议 LaTeX 模板](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn)。中文使用 `xeCJK`；这是参考 IEEE 会议版式的中文工作稿，未核验特定届次 ICC 的页数、匿名或最终提交要求。

在仓库根目录运行：

```bash
python3 paper/spin_context_direct_cn_20260916/compile_pdf.py --compiler /path/to/tectonic --fetch
```

原研究环境默认使用本地 Tectonic 二进制与缓存离线编译；此发布不包含该机器的编译器和缓存，请安装 Tectonic 并用上述参数指定。可用 `--compiler /path/to/tectonic` 指定编译器，首次缓存缺失时加 `--fetch`。字体依赖包括 macOS 的 Songti SC、Heiti SC、Arial、Courier New，以及 TeX Gyre Termes；其它操作系统可安装相应字体或调整主文件字体设置。

在 macOS 上重新渲染、提取文字并检查字体嵌入：

```bash
swift paper/spin_context_direct_cn_20260916/build/render_pages.swift paper/spin_context_direct_cn_20260916/build/manuscript.pdf paper/spin_context_direct_cn_20260916/build
```

当前 PDF 共四页，全部使用字体已嵌入，最终离线编译通过。四页渲染均已逐页检查；日志没有缺字、溢出盒或未解析引用。两处行间疏松提示及 Songti 字体 script 元数据提示未影响可见版面，详见 [版面检查](build/LAYOUT_CHECK.json)、[PDF 检查](build/PDF_INSPECTION.json) 和 [构建记录](build/BUILD_INFO.json)。重编译后应重新检查，不能沿用旧 PDF 的版面记录。

## 内容与证据

- `intro_related.tex`：研究问题与 SPIN、SRMF、DeltaNet 相关工作。
- `method_section.tex`、`architecture.tex`：实际结构、公式与可见信息路径。
- `experiment_setup.tex`：固定协议、历史参考差异与开发评价范围。
- `results_tables.tex`、`results_section.tex`：最终精度/成本数字及有限结论。
- `references.bib`：文献及固定数据快照出处。

[独立科学内容复核](../../analysis/spin_sync_unified_20260916/PAPER_SCIENTIFIC_REVIEW_CN.md) 已逐项核对两表 60 个数值、正文变化率和实现公式；发现的表达问题均已修复。模型源码及结果文件保持冻结，完整证据入口见 [研究结项](../../analysis/spin_sync_unified_20260916/OUTCOME_CN.md)。
