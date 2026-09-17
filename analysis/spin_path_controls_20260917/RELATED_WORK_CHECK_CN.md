# TM-LLM 与 Utimac：有限范围的相关工作核实

核实日期：2026-09-17。仅阅读原论文、作者所指代码仓库和作者机构出版记录；没有运行模型、下载流量数据、新增基线或改变当前实验。TM-LLM 的最终出版 PDF 已在临时目录下载并提取正文；Utimac 阅读 arXiv v1 全文。下述代码判断是静态阅读，不是复现认证。

**结论：两项均应在相关工作中定位；此次没有确认到能够直接用作本项目未消费日期的 WAN 数据。** TM-LLM 是同任务的预训练 LLM 适配路线；Utimac 是以数据中心为主要场景、估计局部统计分布并提供不确定性区间的路线。它们均不自动成为本轮必须新增的基线。

## 1. TM-LLM

### 书目信息与方法假设

- **正式论文**：Kaiwen Jiang、Fenglin Yan、Yan Qiao、Meng Li、Yuxuan Li、Mauro Conti，*Network Traffic Matrix Imputation via Large Language Models*，30th IEEE Symposium on Computers and Communications（ISCC 2025），7 页；DOI `10.1109/ISCC65549.2025.11326227`。作者机构记录明确为 2025 年已出版会议论文，会议于 2025-07-02 至 07-05 在 Bologna 举行。[作者机构出版记录](https://research.tudelft.nl/en/publications/network-traffic-matrix-imputation-via-large-language-models/)
- **输入与训练**：由部分可见 TM 开始，先用对抗训练的生成器预补全，再将 TM 嵌入与任务提示共同输入预训练 LLM，最后经 MLP 输出。原文冻结注意力、微调位置嵌入／归一化／前馈部分及任务层；缺失项真值参与训练目标。因此它依赖历史监督和预训练权重，并非只靠提示、无需训练的补全。[原论文第 III 节](https://repository.tudelft.nl/file/File_fa5073f1-ad2f-4d14-88e0-c4c7778e85e8)
- **数据与协议**：论文使用 Abilene 和 GÉANT；分别描述为 5 分钟、168 天与 15 分钟、112 天的源数据，但实际实验只选两者前 3000 行，按 8:1:1 划分。上述周期是该文数据描述，不能据此给本项目 GEANT 快照恢复时间戳。[原论文第 IV-A 节](https://repository.tudelft.nl/file/File_fa5073f1-ad2f-4d14-88e0-c4c7778e85e8)

### 官方代码与可复核边界

原文仓库 `FILingK/TM-LLM` 现重定向至 `JKevin17/TM-LLM`。本次固定读取提交 **`3f718caacad5cc67af69e18fd8d8f8ed36cea38b`**。README 声明提供 GPT-2、DeepSeek-R1-1.5B、Llama-3.1-8B 的使用示例。[官方仓库](https://github.com/JKevin17/TM-LLM/tree/3f718caacad5cc67af69e18fd8d8f8ed36cea38b)

- Loader 从无表头 CSV 读入并截取前 3000 行。官方代码给验证／测试起点减去 `seq_len`，再随机抽窗；这与我们非重叠开发窗的具体规则不同，不能直接横比论文数值。[固定提交的数据加载器](https://github.com/JKevin17/TM-LLM/blob/3f718caacad5cc67af69e18fd8d8f8ed36cea38b/ISCC_2025/Imputation/data_provider/data_loader.py#L148)
- GEANT 提示模板写为 **23 个节点、529 个 OD 项**，不同于我们当前 **22 个节点、462 条非对角流**。这里只确认模板内容，未读取 CSV 来认证其实际流轴。[固定提交的模型代码](https://github.com/JKevin17/TM-LLM/blob/3f718caacad5cc67af69e18fd8d8f8ed36cea38b/ISCC_2025/Imputation/models/TM_LLM.py#L154)
- **未来适配时需核查的输入依赖**：该提交 `test()` 第 555–559 行使用完整 `batch_x` 计算均值／标准差并标准化；第 568 行将这种标准化后的可见位置写回送给模型的 `inp`。由代码可推断，这条实现路径的模型输入依赖包含人为隐藏项的统计量。此次未运行隐藏值扰动检查，不能据此认定原论文全部结果，也不能据此主张我们的方法优越；当前只记录为复用该实现前需要审计的差异。[统计变换](https://github.com/JKevin17/TM-LLM/blob/3f718caacad5cc67af69e18fd8d8f8ed36cea38b/ISCC_2025/Imputation/exp/exp_imputation.py#L555)、[模型输入写回](https://github.com/JKevin17/TM-LLM/blob/3f718caacad5cc67af69e18fd8d8f8ed36cea38b/ISCC_2025/Imputation/exp/exp_imputation.py#L568)

### 是否提供可确认的新日期数据？

**未确认。** 仓库两个 CSV 在 Git 树中都是 133 字节的 Git LFS 指针；本次只读指针，未取流量对象：

| 数据 | LFS 对象 SHA256 | 声明字节数 |
| --- | --- | ---: |
| Abilene | `b6583be7c9314d5fd2a8b38709d25133449c74e89aa94f87b506c26e4e99704f` | 53,156,492 |
| GEANT | `a0655d8f962f5ebcb999b49eeef4b931ae0314dcdea8c6a4ed87d3a6621e32a0` | 52,320,959 |

来源：[Abilene 指针](https://raw.githubusercontent.com/JKevin17/TM-LLM/3f718caacad5cc67af69e18fd8d8f8ed36cea38b/ISCC_2025/datasets/net_traffic/Abilene/abilene.csv)、[GEANT 指针](https://raw.githubusercontent.com/JKevin17/TM-LLM/3f718caacad5cc67af69e18fd8d8f8ed36cea38b/ISCC_2025/datasets/net_traffic/GEANT/geant.csv)。

不同文件哈希不证明流量内容不重叠：格式、单位、节点与对角项保留方式均可能改变字节。已读资料没有建立这些文件与本项目快照之间的逐日、逐流映射，也未给出可直接认证为新增日期的清单；因此不能把更大的文件或后续行号当成独立测试来源。本项目现有范围的使用情况仍以[既有审计](../spin_sync_unified_20260916/HOLDOUT_AUDIT_CN.md)为准。

## 2. Utimac

### 书目信息与可确认状态

Xiyuan Liu、Zihao Wang、Guanzuo Liu、Xiucheng Tian、Wenting Wei，*Rethinking Traffic Matrix Completion: Estimate the Process, Not the Entries*，**arXiv:2605.02225v1**，2026-05-04 提交，15 页。arXiv 当前记录只列 v1，未列正式会议／期刊接收信息；因此本次只能称为 **2026 年预印本**，不能写成已接收论文。[作者提交的 arXiv 记录](https://arxiv.org/abs/2605.02225)

### 任务、输入和数据

- 在局部平稳窗内，以多帧部分观测估计共享统计参数；对数域采用高斯主体加稀疏偏差，通过正则化目标与块坐标下降补全，并输出不确定性区间。这是窗内统计推断路线，不是我们的从头监督训练上下文网络。关于完整实现是否还使用额外初始化资产，本次未能通过代码认证。[原文第 2–3 节](https://arxiv.org/html/2605.02225v1#S3)
- 主实验使用 Facebook-Pod-B、Facebook-ToR-A 数据中心流量；WAN 仅附录 GÉANT，22 节点、表内 462 条 OD 流，矩阵维度写为 484。可见率为 0.3–0.9，区别于我们固定 0.2；未发现 Abilene 实验。[原文实验与数据表](https://arxiv.org/html/2605.02225v1#S4)、[附录 D](https://arxiv.org/html/2605.02225v1#A4)
- 原文 GÉANT 附录给出 15 分钟间隔，但没有提供本项目所需的新增日期／逐流对应清单。**未确认未消费的 WAN 来源。** 两个 Facebook 数据集也不构成现有 WAN 的新增日期。[附录 F](https://arxiv.org/html/2605.02225v1#A6)

作者在论文中链接[匿名官方代码仓库](https://anonymous.4open.science/r/Utimac-0551/)。此次浏览器未取得可读仓库文件，直接只读请求亦发生 TLS 超时；因此未认证其代码、数据文件、下载可用性或精确时间划分。访问失败不等于资料不存在。

## 3. 对当前论文的有限作用

相关工作可以将 TM-LLM 放在“监督式预训练模型适配”路线，将 Utimac 放在“局部统计推断与不确定性估计”路线。我们要回答的仍是：**在固定的监督式离线补全任务中，浅层时空上下文与本流局部直读的组合是否值得其成本。** 这是本项目定位判断，不是两篇文献已经替我们证明的空白。

本次核实不支持宣称我们的模型优于这两项工作，也不改变预定 GEANT 路径对照。没有获得能够直接认证的新日期来源，不把下载另一份同名 WAN 文件列为已解决的最终评价方案。
