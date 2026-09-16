# 推理 benchmark 独立只读审计

审计时间：2026-09-16T08:01:05.396015+00:00。

## 1. 结论与核查范围

**48 格数字与现有报告一致，标签和主要测量口径正确。** 从 720 个原始正数计时样本独立重算了 192 项统计量（整批 median/P95、摊销 median/P95），与原始 JSON、48 行日志及 inference_summary 对应值完全一致，本次最大绝对差为 0。八组参数量按源码中的层尺寸逐项静态计数，与记录及完成结果元数据一致；未构建或运行模型。

本次只读已归档记录、完成结果、可用本地权重文件及相关源码；未运行 GPU、未新增性能测试、未修改 benchmark 或汇总源码，也未操作正在运行的训练任务。

## 2. 测量口径与标签

- 硬件记录为 **NVIDIA H20**；hostname 为 `d-20260907060201-gt68m`。启动记录的物理 GPU 为 1，`CUDA_VISIBLE_DEVICES=1`，进程内记为 `cuda:0`，两种编号一致；不能按 SSH 别名把型号写成 H200。环境为 Python 3.10.18、PyTorch 2.2.2+cu121、CUDA 12.1、NumPy 1.22.4。
- 2 WAN × 4 模型 × B1/B8 × 3 条件，恰好 48 格，无重复或缺格。每格 5 次暖身、15 次计时；两 WAN 分别使用前 8 个 fit 窗口，形状为 `[8,50,144]` 与 `[8,50,462]`，B1 只用其中首窗。观测率 20%，不计算任何准确率分数。
- 计时覆盖 `engine.predict_windows`：CPU→GPU 输入传输、全 OD 预测、GPU→CPU 输出传输、有限值检查、CPU 非负截断与观测值复制；每次前后显式同步 CUDA。检查点加载、fit 统计量/邻居图准备、掩码生成均在计时外。额外的输出复制一致性检查位于每格 15 次计时之后。
- FP32、TF32 关闭、eval/no_grad；Full、no_memory 与旧 SPIN 的内部 node_chunk 均为 32，推理关闭梯度检查点。旧 Direct+Sync 使用自身实现；四模型逐个加载，不把其它模型权重一起保留在 GPU 上。
- `spin_adapted` 显示为“旧 SPIN”，`direct_sync` 显示为“旧 Direct+Sync”，当前 Markdown 标签正确。Full 是本次统一 SPIN context + Direct + Sync；no_memory 去掉记忆分支，不等同于完整旧 SPIN。八个模型均绑定 seed41001 的已完成最佳检查点，本报告不是两个种子的平均性能。
- B1 汇总是三个条件的**各自整批中位数的算术平均**；B8 汇总先这样平均，再除以 8。B8 的 ms/窗是摊销量，不是单窗请求延迟，也不是把 45 个条件样本混在一起取中位数。

### P95

排序后的 15 个样本记为 x₁≤…≤x₁₅。NumPy 默认线性插值在本例为 `P95 = 0.7×x₁₄ + 0.3×x₁₅`，中位数为 x₈。48 格的这两个值及除以 batch 的结果均独立复算通过。P95 只有 15 个样本的尾部分辨率，容易受最慢一两次影响。例如 Abilene Full/B1/uniform 的 median 为 42.955 ms，P95 为 49.585 ms，最慢样本为 58.038 ms；本审计没有删去该样本。不能将这些 P95 解释为生产环境延迟保证。

## 3. 独立复算的主表

下表两项批延迟都按三条件中位数取算术平均。显存为 B8 三条件最大 `max_memory_allocated`，MiB = bytes / 2²⁰。

| WAN | 模型 | 参数 | B1 整批 ms | B8 整批 ms | B8 摊销 ms/窗 | B8 峰值 allocated MiB |
|---|---|---:|---:|---:|---:|---:|
| Abilene | Full | 98,693 | 42.879420 | 80.477908 | 10.059738 | 2896.947754 |
| Abilene | no_memory | 73,733 | 9.448313 | 48.939280 | 6.117410 | 2896.852539 |
| Abilene | 旧 SPIN | 65,364 | 27.387784 | 179.952425 | 22.494053 | 2897.614746 |
| Abilene | 旧 Direct+Sync | 79,137 | 37.015180 | 38.136602 | 4.767075 | 1219.578613 |
| GEANT | Full | 108,869 | 53.023260 | 246.795737 | 30.849467 | 3797.135742 |
| GEANT | no_memory | 83,909 | 22.421987 | 152.133344 | 19.016668 | 2976.750000 |
| GEANT | 旧 SPIN | 95,892 | 79.850093 | 570.842094 | 71.355262 | 2979.620117 |
| GEANT | 旧 Direct+Sync | 84,225 | 37.185946 | 98.027778 | 12.253472 | 3830.259277 |

## 4. 耗时比核算

比值严格定义为“分子耗时 / 分母耗时”；变化率为 `(比值−1)×100%`。低于 1 表示分子更快，不把耗时比直接称为加速倍数。B8 比值按摊销口径计算，与同一 B8 整批比值相同。

| WAN | 分子 / 分母 | B1 比值 | B1 变化率 | B8 比值 | B8 变化率 |
|---|---|---:|---:|---:|---:|
| Abilene | Full / no_memory | 4.538315 | +353.8315% | 1.644444 | +64.4444% |
| Abilene | no_memory / 旧 SPIN | 0.344983 | -65.5017% | 0.271957 | -72.8043% |
| GEANT | Full / no_memory | 2.364789 | +136.4789% | 1.622233 | +62.2233% |
| GEANT | no_memory / 旧 SPIN | 0.280801 | -71.9199% | 0.266507 | -73.3493% |

现有汇总中另外两行 Full/旧 SPIN 比值也核对通过。由原始样本可支持：在这个固定输入、固定实现与单设备测量中，no_memory 比旧 SPIN 的 B1 耗时少约 65.5%/71.9%（Abilene/GEANT），B8 摊销耗时少约 72.8%/73.3%；Full 相比 no_memory 的 B8 耗时增加约 64.4%/62.2%。

**不能扩大成“四种实现中始终最快”。** 两个 WAN 的 B8 测量中，旧 Direct+Sync 均更快：no_memory/旧 Direct+Sync 的 B8 耗时比分别为 **1.283263** 和 **1.551941**。B1 与 B8 的排序不同；原始表已保留这条参考，标签不可缩写成纯 Direct。

## 5. 参数量与显存的解释

### 参数量

按 Linear 的 `in×out+bias`、LayerNorm/TSLNorm、PReLU 与节点嵌入逐项静态相加，令 F 为 OD 流数，得到：

| 实现 | 精确参数公式 | 随 F 增长的部分 |
|---|---:|---|
| Full | 94,085 + 32F | 一张 32 维节点位置嵌入 |
| no_memory | 69,125 + 32F | 同上 |
| 旧 SPIN | 51,540 + 96F | 位置、valid、mask 三张 32 维节点嵌入 |
| 旧 Direct+Sync | 76,833 + 16F | 一张 16 维 flow embedding |

代入 F=144、462，八项计数均精确吻合。Full 比 no_memory 固定多 **24,960** 个参数，即 `key:32→128` 和 `value:32→128→128` 的权重/偏置；no_memory 保留 query、Direct 与 decoder。因此 no_memory 相对 Full 的参数更少，但相对旧 SPIN，Abilene **多 12.8037%**、GEANT **少 12.4964%**，不能统一写成“参数量比 SPIN 更少”。

参数总数不是执行次数。旧 SPIN 源码执行四层时空上下文及各层 readout；当前 no_memory 只执行一层上下文，但保留宽 128 的 query/decoder 等模块。Full 另外执行 K/V 投影和两个方向各 50 步的同步记忆扫描。这样的结构差异可以解释“权重数更少却不一定更快”为何不矛盾；当前测量没有逐算子 profile，不能把全部耗时差精确归因于某一个算子。

### 显存

48 格均满足 `peak ≥ resident ≥ 0` 且 `peak_above_resident = peak − resident`；原始 JSON、日志和汇总中的 bytes 一致。它们是 PyTorch **allocated** 口径，含常驻模型、buffer、fit 统计量与邻居表，以及被分配器计入的其它本进程常驻分配；不是纯权重大小，不是 reserved，也不是 nvidia-smi 整卡占用。

| WAN | 比较 | B8 峰值差 MiB（前者−后者） | B8 峰值变化率 |
|---|---|---:|---:|
| Abilene | Full / no_memory | +0.095215 | +0.003287% |
| Abilene | no_memory / 旧 SPIN | -0.762207 | -0.026305% |
| GEANT | Full / no_memory | +820.385742 | +27.559780% |
| GEANT | no_memory / 旧 SPIN | -2.870117 | -0.096325% |

Abilene Full 与 no_memory 的 B8 峰值仅差 **0.095214844 MiB**，恰等于 24,960 个 FP32 参数的字节差；二者的 `peak_above_resident` 都是 **2863.935546875 MiB**。GEANT 则不同，Full 比 no_memory 多约 **820.386 MiB**。因此去掉 Sync 并不保证各数据集都大幅节省推理峰值显存。

源码中的加性时空注意力会形成带两个时间轴的消息张量，规模随 `B×node_chunk×邻居数×T²×通道数` 增长；no_grad 不会消除这些前向临时张量。一次峰值还取决于哪些张量同时存活，而不是把四层的峰值简单相加。Abilene 峰值几乎相同与共享上下文临时张量占主导的解释一致；GEANT Full 还保留全目标的 K/V 及扫描中间量。这是源码支持的解释，当前峰值记录本身不能定位精确峰值算子。

no_memory 相对旧 SPIN 的 B8 峰值仅降低 **0.0263%/0.0963%**，远小于耗时降幅；不能把本次加速同时写成大幅省显存。

## 6. 证据边界与文件绑定

- 这是单次 benchmark 中同一组固定 fit 窗口、固定模型顺序的描述性测量。15 次是计时重复，不是独立训练重复；结论限于当前设备、FP32、B1/B8、T50 与执行实现，不外推到其它 batch、混合精度、设备或线上负载。
- 来源文件 benchmark_inference.py、engine.py、data.py 的当前 SHA 与测速记录一致；用于静态参数解释的 9 个模型源码文件与绑定配置中的源码指纹一致。主表、48 格统计、全部既有 Full 比值、48 行日志及 Markdown 显示精度均核对通过。
- 八项记录的参数/最佳轮/权重 SHA 与本地完成结果或旧 reference 完成记录一致；四项新模型的本地 best.pt/config/result 以及可用旧 SPIN 权重均重新核对 SHA。旧 Direct+Sync 权重及部分旧 registry 未在对应本地路径保存，本审计没有重新读取远端；其远端权重和来源校验由已绑定的 benchmark 源码在运行时执行，不能声称本审计再次读取了那些权重。
- 当前显示标签“旧 SPIN”“旧 Direct+Sync”与加载代码吻合。没有发现需修改的数值或标签；no_memory/旧 SPIN 比值是本审计补充复算的值，不是改写原始测量。
- 本任务不评价准确率，不从推理计时推断方法新颖性，不触碰仍在运行的 GEANT Full 训练。

输入：`INFERENCE_BENCHMARK.json`、`INFERENCE_BENCHMARK.log`、`INFERENCE_BENCHMARK_LAUNCH.json`、`inference_summary/INFERENCE_SUMMARY.json` 与 `INFERENCE_SUMMARY_CN.md`。

- 原始 benchmark JSON SHA256：`b19b2b4893fe3ea0897a61716b33d9d4b4e079a8372c571d0de3ec5c6990c03d`。
- 本次 summary JSON SHA256：`d0ed9e52d94de081ef76d53982fcd539ac7f1404eef0609683cbc299f0f20971`。
- 已审阅 benchmark 源码 SHA256：`d523e8bc84fcb5789a0ca712aea561c4f9fb42129d65bf8c6ab0b42a38c84679`。
