# SC-ACIL v1 正式确认卡（gate 首次访问前冻结）

## 一句话方法与问题

ACIL 先给出严格经过观测锚点的轨迹；SC-ACIL 再用同一条流上“遮掉中间锚点后能否预测回来”的部署合法误差，自校准整条缺失轨迹。问题是：这个 LOO innovation 是否同时优于 ACIL-only，以及参数量、初始化、训练完全相同但把 innovation 置零的 `SC-ACIL-u0`？

## 方法角色

- `SC-ACIL`：本论文完整方法。
- `ACIL-only`：本论文自研第一阶段，是 `w/o self-calibration` 内部消融，不能称为外部 baseline。
- `SC-ACIL-u0`：完全同容量的机制消融，仅将标量 LOO innovation 置零。
- Linear interpolation：经典外部基线；原始 ARI-LLM、ImputeFormer 等强外部方法属于通过本关之后的 Stage B。

本轮接受“不保留 LLM 为核心 backbone”这一选择。LLM 只在 Stage B 作为原始 ARI-LLM 外部比较，不从历史工程惯性获得主角地位。

## 冻结网格

- 数据：Abilene、GEANT；只读规范化 permitted train/validation arrays。
- 训练：fit；选择 checkpoint：source-dev；确认：此前未运行的 gate。
- 禁止：sealed test、旧 test cache、旧 test report、原始整表 CSV、任何 split/path override。
- masks：random、internal block、two burst；每流恰好 K=3；headline 只 pool 两个 structured masks。
- seed bundles：4、5、6；不按结果换 seed。
- jobs：6 个 ACIL fit jobs；随后 6 个 SC-ACIL-u0 与 6 个 SC-ACIL jobs。u0 job 唯一携带 Linear、ACIL-only、u0 gate 记录，full job只携带 full 记录。
- 训练：20 epochs、512 fit windows、AdamW、320 optimizer updates；source-dev 三 mask ratio-of-sums NMAE 选 checkpoint，平局取更早 epoch。

## 证据与门槛

原始证据单位为 `seed × dataset × mask × window`，先对 flows 求和。NMAE 是 `Σ|e|/Σ|y|`。主 effect 先在每个 dataset 内对 seeds、structured masks、windows 做 ratio-of-sums，再对两个 dataset 的 improvement 等权平均。置信区间是 paired hierarchical bootstrap：先重采样 seed，再在每个 dataset 内以长度 4 的 circular window block 重采样，共 10,000 draws。

全部条件同时满足才 `proceed`：

1. SC-ACIL 相对 ACIL-only 的 dataset-equal structured improvement ≥1.5%，paired 95% CI 下界严格 >0；
2. SC-ACIL 相对同容量 u0 ≥0.5%，paired 95% CI 下界严格 >0；
3. SC-ACIL 相对 ACIL-only 的 random dataset-equal improvement ≥−0.5%；
4. 两个 dataset 的 structured effect 均 ≥−0.5%；
5. 四个 dataset×structured-mask cells 至少三个严格为正，最差 cell ≥−1%；
6. 三个 seed bundle 的 dataset-equal structured effect 至少两个严格为正。

任一条件失败即 `kill_sc_acil_method_claim_no_gate_revision`：不得在相同 gate 上改模型、改门槛、挑 seed 或重新命名为成功。通过后才新建 Stage-B 协议，补 K/观测率和原始 ARI-LLM、ImputeFormer 等强外部 baseline；本卡本身不授权 Stage B，更不授权 sealed test。

## 已知限制

Stage A 只回答 K=3 下核心机制是否在新 cohort 复现、是否超出容量效应。它本身不足以投稿：通过后仍需观测率泛化、强外部 baseline、成本/参数量与必要消融。Gate 是 branch-sealed confirmation，不是项目级从未被任何人接触的 pristine test。

