# 浅层时空流量补全中的显式本流观测读出

这是使用 IEEEtran conference 双栏排版的中文内部初稿。它聚焦 SPIN+Direct 与两个独立训练的单路径对照，保留开发评价、历史参考配方不同及未完成确认性评价的限制。不是已经达到投稿标准的英文稿。

- [PDF](build/manuscript.pdf)
- [LaTeX 主文件](manuscript.tex)
- [本轮研究结论与模拟评审](../../analysis/spin_path_controls_20260917/OUTCOME_CN.md)
- [完整数值表](../../analysis/spin_path_controls_20260917/DATA_TABLES_CN.md)
- [固定实验计划](../../analysis/spin_path_controls_20260917/PLAN_CN.md)

## 结构与数据来源

方法主线是一层 SPIN 时空上下文与双向本流真实观测读出的联合训练。当前主模型不使用 Sync；历史 Full 的 120 轮结果单列保留。Context-only 与 Direct-only 是相同接口上的路径对照，不是官方 SPIN，也不是等有效容量控制。

结果来自两套 WAN、三个模型、两个配对初始化，共 12 条轨迹。其中 8 条新增单路径训练，4 条组合轨迹接续已有第 120 轮状态；120/160 是同一轨迹的两个预算截面，不增加独立重复。每个数据集的完整原始记录、评分校验、条件表和曲线均在 `analysis/spin_path_controls_20260917/`。

`results_tables.tex` 由下列命令直接从两个已完成的汇总和成本记录生成，来源校验值保存在 `TABLE_PROVENANCE.json`：

```bash
python3 analysis/spin_path_controls_20260917/build_tables.py
```

## 编译与检查

从仓库根目录执行：

```bash
python3 paper/spin_direct_path_controls_cn_20260917/compile_pdf.py
```

默认使用已有本地 Tectonic 与缓存，不联网下载；可通过 `--compiler` 指定其他 Tectonic 路径。字体沿用这台 Mac 的 Songti SC、Heiti SC 与 TeX Gyre Termes，换系统编译需要相应字体或显式替换。

`build/BUILD_INFO.json` 记录构建命令和 PDF 校验值；`build/PDF_INSPECTION.json` 记录页数、嵌入字体和渲染结果；最终人工版面核对记录在 `build/LAYOUT_CHECK.json`。

旧稿 `paper/spin_context_direct_cn_20260916/` 保留原样。训练仍依赖仓库的历史注册资产和规范数据；论文构建可执行不等于训练已具备脱离作者环境的完整复现入口。
