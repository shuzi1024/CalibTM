# ACIL-Innovation v1 研究卡

## 状态

`draft_protocol`。这是新的隔离实验分支，不继承 OD-Orbit 的研究结论，也不把旧 ACIL 报告当作正式证据。

## 可证伪问题

在每条流仅有 3/50 个观测点、且不使用拓扑、路由和绝对流身份时，从观测点构造的 ACIL leave-one-out innovation，能否让跨流集合模型在 structured missingness 下稳定优于冻结的 pointwise ACIL？

下列任一结果都足以否定相应层级的主张：

- hidden-truth Q oracle 相对 ACIL 的 structured pooled NMAE 改善不足 5%；
- observation-only global LOO 相对 ACIL 不足 1.5%，或相对 local LOO 不足 0.5%；
- Full-GPT2 相对 ACIL/Full-u0 中较强者不足 2%，或不能超过 LOO-DeepSets 0.5%；
- formal gate 没有达到预注册 effect、CI、cell win 和 no-harm 条件。

## 方法差异和贡献边界

ACIL 使用完整 K=3 观测集合产生基础补全。对每条流，将排序后的中间锚点临时隐藏，用同一冻结 ACIL checkpoint 在剩余两个严格包围该点的锚点上预测；观测值与该预测的差构成唯一合法 LOO innovation。跨流模块只读取这些 observation-only innovation、相对时间几何和 ACIL 基础信息，并通过锚点包络约束的残差解码器修正未观测点。

v1 不声称 multi-scale innovation。K=3 每条流只有一个严格双侧 LOO 锚点，K=5 不在本协议中。为避免 ACIL 在临时 K=2 输入上完全分布外，ACIL 基础 checkpoint 在 fit 上使用配对辅助项：同一个注册 K=3 mask 同时计算 K=3 主损失和“删去中间锚点”的 K=2 辅助损失，二者等权平均；checkpoint 仍只按 K=3 source-dev 三 mask 的 raw ratio-of-sums NMAE 选择。K=2 是训练支持和 LOO 内部构造，不是论文评估预算。

## 数据与证据边界

- window length 固定为 50；数据集为 Abilene 144 flows 和 GEANT 462 flows。
- cohort bounds、512 个 fit windows、source-dev、tune、validation gap 和 gate windows 全部由 `configs/protocol_v1.json` 固定。
- 当前是 branch-sealed discovery，不宣称项目级 pristine holdout。
- discovery 只接收注册 cohort 的许可数组；配置和 runner 不暴露原始数据路径、sealed split、旧 cache 或 split override。
- provenance 只能绑定许可解析数组和 cohort 副本，不能读取或散列包含不可访问字节的整份来源文件。

## Masks、LOO 与 Q/E

- families 固定为 `random`、`internal_block(length=8)`、`two_burst(length=3..8 each)`；K=3。
- 每个训练 job 不含 mask axis。第 `epoch` 轮中，epoch 排序位置 `p` 的窗口使用
  `families[(epoch*512+p+bundle-1)%3]`，所有方法共享同一身份规则。
- source-dev checkpoint 指标把三种 mask 的 raw absolute-error/truth sums 先合并，再计算一个 ratio-of-sums；禁止挑最好 mask 或平均三个 NMAE。
- deployable stages 的 E 始终是 K=3 观测集合的完整补集，共 47 个时间点，Q 为空。
- 只有 Stage H oracle 每条 flow 有一个 hidden-truth Q。Q 由 method-independent、truth-independent 的 domain-separated SHA-256 从缺失集合选择；structured masks 优先在 controlled gap 外选择，若候选为空才回退到全部缺失点。E 为其余 46 个缺失点。ACIL comparator 与 oracle 在完全相同的 E 上评分。

## 训练语义

- optimizer 固定为 AdamW：`betas=(0.9,0.999)`、`epsilon=1e-8`、weight decay `0.01`、constant learning-rate schedule、gradient clip norm `1.0`。bias 和 LayerNorm weight 不使用 weight decay，其余可训练参数使用注册 weight decay。
- new modules 和 scratch GPT blocks 的 learning rate 均为 `3e-4`；pretrained GPT blocks 为 `1e-5`。不允许按运行结果改变参数组或 schedule。
- Stage H 的 Truth-Q residual 训练 loss 和 source-dev checkpoint 都只在固定的 `E=U\Q` 上计算。所有 observation-only deployable residual methods（Local/Global-LOO、Full-u0、LOO-DeepSets、Full-Scratch、Full-GPT2）的训练 loss 和 checkpoint 都在完整的 `U` 上计算；Q 不存在，也不得从评分 target 中删点。

## 固定阶段

所有跨 stage checkpoint 依赖都按同一 dataset×seed bundle 绑定 source stage 通过 source-dev 规则选出的 best checkpoint，并且必须同时精确匹配 checkpoint identity、checkpoint file 和 checkpoint tensor 三个 SHA-256。只匹配文件名、epoch 或 method label 均不成立。

训练 job 与 headline result 的 carrier 关系也固定，避免给 deterministic/reused comparator 虚构训练 job：Stage-0 的 `acil` job 携带 Linear+ACIL；Stage-H 的 `truth_q_deepsets` job 携带 ACIL+Truth-Q；Stage-I 的 `local_loo` 携带 ACIL+Local，`global_loo` 只携带 Global；Full-tune 的 `full_u0` 携带 ACIL+LOO-DeepSets+Full-u0，Scratch/GPT2 各只携带自身；`formal_acil` 不携带 evaluation records；formal-gate 的 `full_u0` 携带 Linear+ACIL+Full-u0，其余六个 learned jobs 各只携带自身。对每个 dataset×bundle，所有 carriers 合并后必须恰好覆盖 stage method registry 一次，不能缺失、重复或新增 method。

### Stage 0：干净 ACIL 复现

- datasets: Abilene、GEANT；bundles 1–3；三 masks。
- methods: Linear fill、ACIL。
- gate: structured pooled ≥1%，两个 structured masks 均严格为正，paired 95% CI 下界 >0，至少 2/3 paired seeds 为正。
- failure: `kill` ACIL-centered 叙事。

### Stage H：truth-Q residual oracle

- methods: ACIL、Truth-Q DeepSets；oracle 仅为 diagnostic evidence。
- ACIL comparator 与 Truth-Q DeepSets 内嵌的 frozen ACIL base 都必须是同 dataset×bundle 的 Stage-0 ACIL best checkpoint；Stage H 不得另训或替换 base。
- gate: 主 effect 只 pool `internal_block` 和 `two_burst`，要求 ≥5%，两个 structured masks 均为正，paired CI 下界 >0，至少 2/3 seeds 为正；`random` 不进入主 effect，只单独要求 ≥−1%。
- failure: `kill` residual-field headroom 假设；不得继续训练 GPT。

### Stage I：observation-only 可辨识性

- methods: ACIL、Local-LOO、Global-LOO；主 effect 和 Global-over-Local 都只 pool `internal_block` 与 `two_burst`。Global 相对 ACIL 要求 ≥1.5%，相对 Local ≥0.5%，每数据集的 structured pooled effect 不低于 −0.5%，主 effect CI 下界 >0，且至少 2/3 seeds 为正。
- ACIL comparator、Local-LOO 的 frozen base 和 Global-LOO 的 frozen base 全部绑定同 dataset×bundle 的同一个 Stage-0 ACIL best checkpoint。
- derangement 只在 tune 上做，不重训。对每个 dataset×bundle×mask family，把已注册 windows×flows 的全部 scalar LOO innovations 按 registry 顺序展平，再使用一个固定的、无放回的联合 permutation；query geometry、middle indicator、truth 和 targets 原位不动。
- unshuffled candidate 始终沿用已计划的 method 名 `global_loo`；`global_loo_deranged` 只是由同一结果派生的 named diagnostic artifact，不进入 headline records，也不是额外训练 job 或可部署 method。
- permutation RNG 固定为 `PCG64DXSM`，domain 为 `acil-innovation-v1:stage-i-derangement:v1`。seed 是 `SHA-256(domain || NUL || canonical identity)` 前 128 bits 的 big-endian 整数；identity 依次绑定 protocol、dataset、cohort、window-schedule hash、seed bundle、mask family、evaluation-mask seed、registered-window count 和 flow count，明确不含 method outcome。
- structured pooled `gain_loss = unshuffled improvement over ACIL − deranged improvement over ACIL`，`fraction = gain_loss / unshuffled improvement over ACIL`。只有 unshuffled 主增益严格大于 0 时该比例才定义；要求 `gain_loss≥0.005` 且 `fraction≥0.30`。
- 若主改善在 `[0.5%,1.5%)` 且完整性通过，只允许一次新协议 revision；低于 0.5% 直接 `kill`。

### Full tune

- methods: ACIL、Full-u0、LOO-DeepSets、Full-Scratch、Full-GPT2；bundles 1–3。
- ACIL comparator 以及 Full-u0、Full-Scratch、Full-GPT2 各自内嵌的 frozen base 都绑定同 dataset×bundle 的 Stage-0 ACIL best checkpoint。`LOO-DeepSets` 只是 Stage-I `Global-LOO` 的论文方法标签重命名，必须精确复用同 dataset×bundle 的同一个 Global-LOO checkpoint，不重训、不复制成新的 checkpoint identity。
- 主 effect、over-DeepSets 和 pretraining attribution 都只 pool 两个 structured masks。Full-GPT2 相对在同一 structured pool 上 NMAE 更低的 ACIL/Full-u0 ≥2%，相对 LOO-DeepSets ≥0.5%，主 effect CI 下界 >0，至少 2/3 seeds 为正；`random` 只作为独立 no-harm floor，要求相对同样规则选出的 ACIL/Full-u0 不低于 −0.5%。
- 只有 Full-GPT2 相对同构 scratch 的 structured pooled effect ≥1% 且 CI 下界 >0 时，才能声称预训练贡献；若低于 1% 但不低于 −0.5%，方法结果与预训练主张分开裁决。

### 条件触发的 Formal-v2（不在 v1 active manifest 中）

v1 的 active manifest 只包含 Stage-0、Stage-H、Stage-I 和 Full-tune。只有 Full-tune 生成与当前 freeze 精确绑定的 `proceed` 裁决后，才允许建立独立 Formal-v2；Formal-v2 中所有 candidate 和 baseline 都从零成对训练/评估，不复用 v1 tune checkpoint。以下是预先记录的 v2 方案，不是 v1 可执行 job。

- 先运行 training-only `formal_acil`：bundles 4–6、只使用 fit/source-dev、不接触任何 evaluation cohort，生成 6 个 ACIL checkpoints。formal gate 的 ACIL comparator，以及 LOO-DeepSets、Full-u0、Full-Scratch、Full-GPT2 内嵌的 frozen ACIL base，都必须精确复用同 dataset×bundle 的该 formal-ACIL best artifact，不能在看到 gate 后重训或替换 base。
- formal gate 首次使用 gate cohort；其余训练仍只使用 fit/source-dev。
- methods: Linear、ACIL、ARI-LLM、LOO-DeepSets、Full-u0、Full-Scratch、Full-GPT2、ImputeFormer。原论文实验节中出现的一处“TM-LLM”是 ARI-LLM 的旧名/笔误，不将同一方法重复计为两个 baseline。
- Full-GPT2 相对在同一 structured pool 上 NMAE 更低的 ACIL/Full-u0 ≥2%，paired CI 下界 >0；每数据集的 structured pooled effect 不低于 −0.5%，至少 2/3 seeds 为正。`random` 不进入主 effect，只独立要求不低于 −0.5%。
- 6 个 dataset×mask cells（包含全部三 masks）至少 4 个为正。相对 ARI-LLM/ImputeFormer 中在同一 all-mask pool 上 NMAE 更低者，三 masks 合并的 effect 以及每个数据集各自的 all-mask pooled effect 都不得低于 −0.5%。
- formal failure: `kill`，不得在同一 gate 上改 variant、换 seed 或重定门槛。

## 指标和不确定性

Primary 为 raw-unit ratio-of-sums NMAE；NRMSE 为 secondary。paired improvement 为
`1 - NMAE(candidate)/NMAE(comparator)`。使用 seed-then-circular-window-block hierarchical bootstrap，block length 4、10,000 draws、PCG64DXSM seed 81001。所有阈值用 float64 原值比较，不预先四舍五入。

“一个正 seed”只有一个定义：对一个 seed bundle，先把两个数据集和该 effect 注册的 `pooling_masks` 的 raw error/truth sums 全部合并，计算一次 paired ratio-of-sums improvement；未经四舍五入的值严格大于 0 才计为正。不得用 cell 数、逐数据集多数票或 mask 平均替代。

## 完整性要求

- `git_available=false`、`git_commit=null`，每次冻结记录 deterministic source-tree SHA-256、config、许可数组、cohort、schedule、mask、target、checkpoint 和 result hashes。
- observed hard projection 精确成立，输出非负；扰动全部 missing truth 不得改变 deployable forward 输入和预测。
- Q/E 严格不交、并集固定；baseline 与 oracle 的 target hashes 必须逐记录一致。
- flow permutation equivariance、LOO 只使用中间已观测值、K=2 ACIL forward 数值有限，都必须有测试。
- 所有 missing/extra/duplicate/reordered records、NaN、OOM、算法失败和基础设施 retry 必须完整保留。基础设施 retry 只能复用相同 job ID、命令、seed 和 hashes。
- `configs/protocol_v1.json` 将结果目录发布契约精确注册为 `publication_contract="protocol_controlled_single_publisher_no_replace"`，只覆盖 frozen queue 控制的同 job 单 publisher。支持 `RENAME_NOREPLACE` 的文件系统使用内核原语；GPFS fallback 在最终 `lstat` 后使用 POSIX `rename`，并校验 source、父目录和 destination inode。非协作外部进程若恰在最终检查与 `rename` 之间创建空 destination directory，不在该契约内；POSIX 接口无法在保持 final path 原子可见的同时排除此竞态，不得把 fallback 描述为对任意外部 writer 的绝对保护。
- GPU 占用脚本和 operational logs 不进入科学 manifest；每个条件阶段结束后先审计再启动下一阶段。

## 当前结果

尚无。本卡只冻结研究问题和证据条件，不构成 method evidence。每个阶段结束必须输出中文 handoff，明确 facts、inferences、unknowns、failures、verdict、limitations、artifact paths 和精确复现命令。
