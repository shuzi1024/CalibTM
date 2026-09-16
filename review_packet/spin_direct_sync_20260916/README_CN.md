# 网页版 GPT-6 Pro 独立研究咨询材料

这是 2026-09-16 当前 SPIN／Direct／Sync 工作的独立咨询包，与上层目录中的 9 月 11 日旧方案分开。

## 网页入口

发布后可直接分享 [START_HERE.md](START_HERE.md)；[合并阅读页](REVIEW_WEB_CN.md) 集中提供方法、完整结果和评审问题。

## 使用顺序

1. 第一轮上传本目录 **01、02、03、04 四个文件**，然后复制 `00_PROMPT_CN.txt` 的全部内容作为提问。
2. 前四份提供方法、全部结果及曲线、模型源码和 SPIN 原论文；由评阅者先独立形成判断。所有不利结果和评价限制均保留。
3. 获得独立意见后，再上传 `optional/05_CHINESE_DRAFT.pdf` 和需要时的 `06_PRIOR_ICC_REVIEW_CN.md`，使用 `10_PAPER_FOLLOWUP_PROMPT_CN.txt` 检查论文定位和不同意见。
4. 只有当它需要进一步核查时，补充 07 协议源码、08 曲线图或 09 历史评价审计。曲线图对应的全部 120 轮数值已在 02 文件中。

压缩包用于集中取文件；解压后选择上述具体文件上传即可。无需提交整个仓库、模型权重、原始流量矩阵或全部历史对话。此包的源码供阅读，不宣称可脱离原仓库直接运行。

## 文件清单

| 文件 | 作用 | 建议 |
|---|---|---|
| 00_PROMPT_CN.txt | 可直接粘贴的研究决策问题 | 作为提问正文 |
| 01_METHOD_AND_CONTEXT_CN.md | 实际架构、融合、最小协议和证据边界 | 首轮 |
| 02_COMPLETE_RESULTS_CN.md | 两张完整表、24 个条件结果、八条完整学习曲线数值 | 首轮 |
| 03_MODEL_SOURCE.txt | 新模型、SPIN 依赖、Sync 更新及旧 Direct＋Sync 源码，含原路径和行号 | 首轮 |
| 04_SPIN_SOURCE.md | SPIN 官方原文链接，区分复用部件和增量 | 首轮 |
| optional/05_CHINESE_DRAFT.pdf | 当前四页内部初稿，含作者的方向判断 | 第二轮 |
| optional/06_PRIOR_ICC_REVIEW_CN.md | 既有评审意见，供比较分歧 | 第二轮可选 |
| optional/07_PROTOCOL_SOURCE.txt | 训练、接续、数据、指标和成本实现 | 按需 |
| optional/08_DEVELOPMENT_CURVES.pdf | 全八条开发 NMAE 曲线图 | 按需 |
| optional/09_HISTORICAL_EVALUATION_AUDIT_CN.md | 后续数据曾被使用的具体证据 | 按需 |
| optional/10_PAPER_FOLLOWUP_PROMPT_CN.txt | 论文与评审的第二轮提问 | 按需 |
| MANIFEST.json | 本包及原始来源哈希，便于核对版本 | 本地留存 |

提示词参考 [OpenAI 官方提示组织建议](https://developers.openai.com/api/docs/guides/latest-model)，明确研究范围、材料优先级和需要的输出；研究问题及证据选择按本项目实际情况制定。这里不对网页版型号、套餐或上传限额作假设。
