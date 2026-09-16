# SPIN / Direct / Sync 联合模型：保存结果汇总

生成：2026-09-16T08:50:52.946716+00:00；状态计数：{'complete': 8}。

只读快照，不探测进程存活。running/stopped 来自已保存 status；complete 需正式 result.json 标记完整且通过已有记录核对。smoke 仅为拟合路径/恢复检查，不是正式精度结果。
原始 V1 从头联合训练；fixed120 是取消早停后的事后训练时长诊断，从同一轨迹的父 last.pt 继续，不是新的独立重复。父输出只用于核查，不并入种子统计；整条1–120轮 history（零起始0–119）只计一次。
full 与 no_memory 按同一 seed 配对；移除记忆同时减少参数，因此不是等参数量控制。以下均为开发指标，不是独立测试。

## ABILENE

| 方法/种子 | 状态 | smoke | 已保存轮数 | 最佳轮 | 完成 NMAE | 完成 NRMSE |
|---|---|---|---:|---:|---:|---:|
| spin 旧参考 | complete | — | — | — | 0.164380137 | 0.220772517 |
| direct 旧参考 | complete | — | — | — | 0.165638897 | 0.223397977 |
| linear 旧参考 | complete | — | — | — | 0.171021478 | 0.241480181 |
| spin_sync_direct/41001 | complete | 继承通过 | 120 | 120 | 0.162975999 | 0.218176260 |
| spin_sync_direct/41002 | complete | 继承通过 | 120 | 117 | 0.161885985 | 0.217960583 |
| spin_direct_no_memory/41001 | complete | 继承通过 | 120 | 118 | 0.163432869 | 0.220289858 |
| spin_direct_no_memory/41002 | complete | 继承通过 | 120 | 120 | 0.161305121 | 0.219626765 |

核查说明：

- spin_sync_direct/41001：同轨迹诊断，父轮数 89，新增已保存轮数 31，累计 120；不新增 smoke、不算独立重复。
- spin_sync_direct/41002：同轨迹诊断，父轮数 75，新增已保存轮数 45，累计 120；不新增 smoke、不算独立重复。
- spin_direct_no_memory/41001：同轨迹诊断，父轮数 76，新增已保存轮数 44，累计 120；不新增 smoke、不算独立重复。
- spin_direct_no_memory/41002：同轨迹诊断，父轮数 98，新增已保存轮数 22，累计 120；不新增 smoke、不算独立重复。

### 已完成种子均值与样本标准差

| 方法 | 已完成/计划种子 | NMAE 均值 ± SD | NRMSE 均值 ± SD |
|---|---:|---:|---:|
| spin_sync_direct | 2/2 | 0.162430992 ± 0.000770756 | 0.218068422 ± 0.000152506 |
| spin_direct_no_memory | 2/2 | 0.162368995 ± 0.001504545 | 0.219958311 ± 0.000468878 |

只对该 WAN 计划/已发现种子全部完成的方法给出最终均值排名；部分均值仅描述已完成种子，不挑最好种子。

mean_nmae（低者较好）：spin_direct_no_memory 0.162368995 → spin_sync_direct 0.162430992 → spin 0.164380137 → direct 0.165638897 → linear 0.171021478。

mean_nrmse（低者较好）：spin_sync_direct 0.218068422 → spin_direct_no_memory 0.219958311 → spin 0.220772517 → direct 0.223397977 → linear 0.241480181。

### 已完成模型相对旧参考

正改善率 = 100×(参考−当前)/参考。

| 方法/种子 | 参考 | NMAE 改善 % | NRMSE 改善 % |
|---|---|---:|---:|
| spin_sync_direct/41001 | spin | +0.854 | +1.176 |
| spin_sync_direct/41001 | direct | +1.608 | +2.337 |
| spin_sync_direct/41001 | linear | +4.704 | +9.650 |
| spin_sync_direct/41002 | spin | +1.517 | +1.274 |
| spin_sync_direct/41002 | direct | +2.266 | +2.434 |
| spin_sync_direct/41002 | linear | +5.342 | +9.740 |
| spin_direct_no_memory/41001 | spin | +0.576 | +0.219 |
| spin_direct_no_memory/41001 | direct | +1.332 | +1.391 |
| spin_direct_no_memory/41001 | linear | +4.437 | +8.775 |
| spin_direct_no_memory/41002 | spin | +1.871 | +0.519 |
| spin_direct_no_memory/41002 | direct | +2.616 | +1.688 |
| spin_direct_no_memory/41002 | linear | +5.681 | +9.050 |

### 同 seed 的 full − no_memory

绝对差为负表示 full 较好；仅纳入两边均完成且配对核查通过的种子。

| seed | NMAE 差 | NRMSE 差 |
|---:|---:|---:|
| 41001 | -0.000456870 | -0.002113598 |
| 41002 | +0.000580865 | -0.001666181 |
## GEANT

| 方法/种子 | 状态 | smoke | 已保存轮数 | 最佳轮 | 完成 NMAE | 完成 NRMSE |
|---|---|---|---:|---:|---:|---:|
| spin 旧参考 | complete | — | — | — | 0.168317410 | 0.229491749 |
| direct 旧参考 | complete | — | — | — | 0.173335594 | 0.228000567 |
| linear 旧参考 | complete | — | — | — | 0.167739045 | 0.228766611 |
| spin_sync_direct/41001 | complete | 继承通过 | 120 | 120 | 0.168700654 | 0.228179198 |
| spin_sync_direct/41002 | complete | 继承通过 | 120 | 108 | 0.170140072 | 0.228465424 |
| spin_direct_no_memory/41001 | complete | 继承通过 | 120 | 119 | 0.167676249 | 0.229851096 |
| spin_direct_no_memory/41002 | complete | 继承通过 | 120 | 113 | 0.166465556 | 0.232527728 |

核查说明：

- spin_sync_direct/41001：同轨迹诊断，父轮数 28，新增已保存轮数 92，累计 120；不新增 smoke、不算独立重复。
- spin_sync_direct/41002：同轨迹诊断，父轮数 25，新增已保存轮数 95，累计 120；不新增 smoke、不算独立重复。
- spin_direct_no_memory/41001：同轨迹诊断，父轮数 30，新增已保存轮数 90，累计 120；不新增 smoke、不算独立重复。
- spin_direct_no_memory/41002：同轨迹诊断，父轮数 57，新增已保存轮数 63，累计 120；不新增 smoke、不算独立重复。

### 已完成种子均值与样本标准差

| 方法 | 已完成/计划种子 | NMAE 均值 ± SD | NRMSE 均值 ± SD |
|---|---:|---:|---:|
| spin_sync_direct | 2/2 | 0.169420363 ± 0.001017822 | 0.228322311 ± 0.000202392 |
| spin_direct_no_memory | 2/2 | 0.167070903 ± 0.000856089 | 0.231189412 ± 0.001892664 |

只对该 WAN 计划/已发现种子全部完成的方法给出最终均值排名；部分均值仅描述已完成种子，不挑最好种子。

mean_nmae（低者较好）：spin_direct_no_memory 0.167070903 → linear 0.167739045 → spin 0.168317410 → spin_sync_direct 0.169420363 → direct 0.173335594。

mean_nrmse（低者较好）：direct 0.228000567 → spin_sync_direct 0.228322311 → linear 0.228766611 → spin 0.229491749 → spin_direct_no_memory 0.231189412。

### 已完成模型相对旧参考

正改善率 = 100×(参考−当前)/参考。

| 方法/种子 | 参考 | NMAE 改善 % | NRMSE 改善 % |
|---|---|---:|---:|
| spin_sync_direct/41001 | spin | -0.228 | +0.572 |
| spin_sync_direct/41001 | direct | +2.674 | -0.078 |
| spin_sync_direct/41001 | linear | -0.573 | +0.257 |
| spin_sync_direct/41002 | spin | -1.083 | +0.447 |
| spin_sync_direct/41002 | direct | +1.844 | -0.204 |
| spin_sync_direct/41002 | linear | -1.431 | +0.132 |
| spin_direct_no_memory/41001 | spin | +0.381 | -0.157 |
| spin_direct_no_memory/41001 | direct | +3.265 | -0.812 |
| spin_direct_no_memory/41001 | linear | +0.037 | -0.474 |
| spin_direct_no_memory/41002 | spin | +1.100 | -1.323 |
| spin_direct_no_memory/41002 | direct | +3.963 | -1.986 |
| spin_direct_no_memory/41002 | linear | +0.759 | -1.644 |

### 同 seed 的 full − no_memory

绝对差为负表示 full 较好；仅纳入两边均完成且配对核查通过的种子。

| seed | NMAE 差 | NRMSE 差 |
|---:|---:|---:|
| 41001 | +0.001024405 | -0.001671898 |
| 41002 | +0.003674516 | -0.004062304 |

JSON 保留全部 history、smoke、个体及三条件指标、均值/样本SD、改善率、配对差和核查状态。SD 只描述本批训练种子，不构造显著性或置信区间。未复制的 ancillary/checkpoint 文件会明确标记，仅读取已有完成标记，不创建新训练协议。
