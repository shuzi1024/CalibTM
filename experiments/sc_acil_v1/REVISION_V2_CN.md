# Freeze v2 修订说明

`freeze_v1` 在任何 formal training/evaluation job 启动前被 CUDA/CPU smoke 前置检查作废。

原因：smoke harness 第一次构造临时 `ACILBase` 时尚未进入确定性 seed boundary，导致 u0 的第一次与第二次临时 base 初始化不同。正式执行路径本来就会在训练 ACIL 前播种，且 residual job 加载完整 hash-bound ACIL checkpoint，因此该缺陷没有产生任何方法结果，也没有访问 gate。

唯一源码修订是在 smoke `_repeat` 中于临时 `ACILBase` 构造前调用同一 registered runtime seeder；模型、训练、数据、job grid、gate 与阈值均未改变。按项目规则重新运行聚焦/联合测试、重新计算全部 source/protocol/data identities，并生成 `freeze_v2`。`freeze_v1` 不得用于任何 job。

