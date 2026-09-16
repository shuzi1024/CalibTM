# 统一模型固定120轮时长诊断

状态：八条轨迹均已完成；见 [完整结果](../../analysis/spin_sync_unified_20260916/FINAL_RESULTS_CN.md)。下文保留当时的接续协议，不是新的启动指令。

仅取消spin_sync_unified_v1的早停；模型、AdamW、固定学习率、裁剪、数据和掩码不变。必须父任务已complete，恢复其停止轮last.pt、优化器及RNG，继续到总计120轮；这不是新种子或独立重复。新结果保留development_only并标明posthoc_training_duration_diagnostic。

新输出：outputs/spin-sync-unified-fixed120-20260916-v1。父文件只读。继承已验证父smoke，不生成新smoke；拒绝--smoke。参数--dataset/--variant/--seed/--device/--deadline-unix；本模块已存在last.pt时用--resume。所有进程仍受本轮7小时授权的同一绝对截止1789558409约束。

最初四项固定范围见analysis/spin_sync_unified_20260916/FIXED120_PLAN_CN.md。完成必须满120轮，并重载开发集最佳checkpoint后重新评分一致。中途截止只保存可恢复状态，不标完成。
