# 2026-09-18 更新评审入口

这里补充了上次2026-09-16评审之后完成的两轮实验。请先根据方法、原始结果和源码形成判断，再按需阅读既有评语。

## 先读这些

1. [方法、范围与完整结果合并页](REVIEW_WEB_CN.md)：含两WAN的120／160轮、全部种子、成本与预定义子组，以及固定两步细化结果。
2. [当前SPIN＋Direct骨架](../../experiments/spin_sync_unified_v1/models.py)与[路径对照实现](../../experiments/spin_path_controls_v1/models.py)；训练见[路径runner](../../experiments/spin_path_controls_v1/runner.py)。主候选是无记忆的SPIN＋Direct。
3. [两步模型](../../experiments/spin_refine_v1/models.py)、[训练入口](../../experiments/spin_refine_v1/runner.py)、[固定计划](../../analysis/spin_refine_20260917/PLAN_CN.md)。该变体已完成并停止。
4. [最新五页中文稿](../../paper/spin_direct_path_controls_cn_20260917/build/manuscript.pdf)；PDF解析受限时阅读[LaTeX主文件](../../paper/spin_direct_path_controls_cn_20260917/manuscript.tex)及其分节文件。该稿聚焦路径对照，两步负结果见本入口的另附材料。
5. [历史评价使用审计](../../analysis/spin_sync_unified_20260916/HOLDOUT_AUDIT_CN.md)：目前没有新增独立未见流量确认。训练源码仍依赖未随仓库提供的历史资产。

## 新增事实

- 路径对照：两WAN、三模型、两个初始化，共12条轨迹完成160轮。其中8条新训练，4条组合从120轮末状态接续；120／160属于同一轨迹。组合在两个预算、全部种子和条件的NMAE上均优于两条单路径。
- 成本：相对Context-only，组合B8每窗耗时增加约3.8%／3.5%；相对Direct-only则达到11.8／16.5倍。不能称为全面效率优势。
- 固定两步：GEANT四条新训练，各160轮。两步相对匹配一步平均NMAE劣化3.37%，B8耗时1.95倍，两个种子及末段方向均未通过规则；未扩展到Abilene。
- 全部仍为开发筛选；没有新独立测试、没有完成公平外部基线确认，也没有解决完整公开复现。

## 复核入口

- [组件完整数值表](../../analysis/spin_path_controls_20260917/DATA_TABLES_CN.md)
- [GEANT结构化汇总](../../analysis/spin_path_controls_20260917/summary/SUMMARY.json)、[Abilene结构化汇总](../../analysis/spin_path_controls_20260917/summary_abilene/SUMMARY.json)
- [路径对照原始记录](../../analysis/spin_path_controls_20260917/raw_outputs)
- [两步完整结果](../../analysis/spin_refine_20260917/report_geant/REPORT_CN.md)、[原始记录](../../analysis/spin_refine_20260917/raw_outputs)
- [GEANT曲线](../../analysis/spin_path_controls_20260917/figures_geant/development_nmae_curves.png)、[Abilene曲线](../../analysis/spin_path_controls_20260917/figures_abilene/development_nmae_curves.png)、[两步曲线](../../analysis/spin_refine_20260917/report_geant/geant_learning_curves.png)
- [来源和公开范围](../../PUBLICATION_MANIFEST_20260918.json)

## 需要你回答

请使用[本次提示词](00_PROMPT_CN.txt)，判断已得到什么增量、还缺哪项决定性证据，以及最多一组怎样的有限确认值得做。主指标、全部初始化和不利结果均保留，不要求预设接收结论。

形成独立意见后，可对照[已有模拟审稿](../../analysis/spin_path_controls_20260917/FINAL_REVIEW_CN.md)与[两步研究结论](../../analysis/spin_refine_20260917/OUTCOME_CN.md)。这些是作者侧研究判断，不是独立环境重新运行的复现认证。
