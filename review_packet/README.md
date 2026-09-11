# 网页版 GPT 审阅材料

整理日期：2026-09-11。本目录将两篇论文与此前推荐的方案文档集中在一起，方便下载和作为附件提交。模型权重与原始数据不在此包内。

## 建议上传的五个文件

| 文件 | 用途 |
| --- | --- |
| [ARI-LLM_paper.pdf](ARI-LLM_paper.pdf) | 原始 ARI-LLM 论文，供核对原方法与新方案的区别。 |
| [CalibTM_v3_CN.pdf](CalibTM_v3_CN.pdf) | 当前 CalibTM v3 论文，供核对已有证据与方法边界。 |
| [CALIBTM_V4_MODULE_BAKEOFF_PROTOCOL_2026-09-11_CN.md](CALIBTM_V4_MODULE_BAKEOFF_PROTOCOL_2026-09-11_CN.md) | 拟执行的 GapCalib 模块实验方案。 |
| [CALIBTM_V4_NOVELTY_EXPLORATION_2026-09-11_CN.md](CALIBTM_V4_NOVELTY_EXPLORATION_2026-09-11_CN.md) | 创新定位、近邻风险与前期探索背景。 |
| [GPT6_PRO_FINAL_REVIEW_PACKET_CN.md](GPT6_PRO_FINAL_REVIEW_PACKET_CN.md) | 审阅背景与问题清单，可作为主提示词。 |

## 可选补充

- [SUPPLEMENTARY_RESULTS_CN.md](SUPPLEMENTARY_RESULTS_CN.md)：补充结果和限制。
- [METHOD_IMPLEMENTATION_APPENDIX_CN.md](METHOD_IMPLEMENTATION_APPENDIX_CN.md)：实际条件输入、约束公式等实现细节。

打开文件页面后下载原始文件，再将文件作为附件提交；建议同时粘贴下面的说明：

> 请先阅读附件，区分已完成的 CalibTM v3 结果与尚待实施的 GapCalib v4 方案。以目标 CCF-C 论文的审稿视角，评估方法创新、模块是否合理、最小必要对照和工程工作量。优先指出影响实施方向的实质问题；不要求为了形式严谨扩建实验系统，也不要无边界推荐新模块。不同方案文档中的历史建议若不一致，请明确指出，不要假设它们已经被实现或验证。

## 版本说明

这里的 Markdown 是 `plan/` 与 `paper/value_only/` 中对应文件的审阅快照；审阅问题清单仅将文件引用改为本目录路径。v4 是待评估方案，不是新实验结果。后续修改主方案后，应同步更新本目录再提交下一轮审阅。

两篇 PDF 按原文件复制，未改写其正文或署名；上游论文的权利归其各自权利人，不因仓库代码许可证而改变。
