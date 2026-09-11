#!/usr/bin/env python3
"""Build the 2026-09-03 CalibTM group-meeting deck.

The deck deliberately uses a plain lab-meeting style: white background,
ordinary text boxes/tables, one blue accent, and no stock artwork.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE_PATH = ROOT / "experiments" / "paper_series" / "build_paper7_group_ppt.py"
OUT = ROOT / "output" / "ppt" / "CalibTM_暑假组会汇报_2026-09-03.pptx"

spec = importlib.util.spec_from_file_location("minimal_pptx", BASE_PATH)
assert spec and spec.loader
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


C = {
    "ink": "202124",
    "muted": "666666",
    "blue": "2F5597",
    "blue_light": "EAF0F8",
    "orange": "C55A11",
    "orange_light": "FCEFE6",
    "red": "A61C1C",
    "red_light": "F9EAEA",
    "green": "3D6B35",
    "green_light": "EDF4EA",
    "line": "BFBFBF",
    "line_light": "D9D9D9",
    "gray": "F4F4F4",
    "gray2": "E7E6E6",
    "white": "FFFFFF",
}


class PlainSlide(base.Slide):
    """A deliberately ordinary PowerPoint slide."""

    def add_title(self, title: str, subtitle: str | None) -> None:
        self.text(0.72, 0.35, 11.85, 0.55, [title], size=27, color=C["ink"], bold=True)
        if subtitle:
            self.text(0.75, 0.96, 11.65, 0.30, [subtitle], size=12, color=C["muted"])

    def footer(self, num: int) -> None:
        self.text(12.05, 7.15, 0.55, 0.20, [str(num)], size=9, color=C["muted"], align="r")

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        fill: str = C["white"],
        line: str = C["line"],
        size: int = 18,
        color: str = C["ink"],
        bold: bool = False,
        align: str = "ctr",
    ) -> None:
        self.shape(x, y, w, h, fill=fill, line=line)
        self.text(x + 0.06, y + 0.05, w - 0.12, h - 0.10, [text], size=size,
                  color=color, bold=bold, align=align, anchor="ctr")

    def strip(self, x: float, y: float, w: float, h: float, text: str,
              fill: str = C["gray"], color: str = C["ink"], size: int = 17,
              bold: bool = False) -> None:
        self.text(x + 0.15, y + 0.05, w - 0.30, h - 0.10, [text], size=size,
                  color=color, bold=bold, align="ctr", anchor="ctr")

    def plain_table(
        self,
        x: float,
        y: float,
        col_ws: list[float],
        row_h: float,
        rows: list[list[str]],
        *,
        font_size: int = 15,
        header_fill: str = C["gray2"],
        special_fills: dict[tuple[int, int], str] | None = None,
        row_fills: dict[int, str] | None = None,
        aligns: dict[int, str] | None = None,
    ) -> None:
        special_fills = special_fills or {}
        row_fills = row_fills or {}
        aligns = aligns or {}
        for r, row in enumerate(rows):
            xx = x
            for c, val in enumerate(row):
                fill = header_fill if r == 0 else row_fills.get(r, C["white"])
                fill = special_fills.get((r, c), fill)
                bold = r == 0 or (r, c) in special_fills
                align = aligns.get(c, "ctr")
                self.shape(xx, y + r * row_h, col_ws[c], row_h,
                           fill=fill, line=C["line_light"])
                self.text(xx + 0.05, y + r * row_h + 0.04,
                          col_ws[c] - 0.10, row_h - 0.08, [val],
                          size=font_size, color=C["ink"], bold=bold,
                          align=align, anchor="ctr")
                xx += col_ws[c]


class TitleSlide(PlainSlide):
    def add_title(self, title: str, subtitle: str | None) -> None:
        self.text(0.85, 1.48, 11.65, 0.75, [title], size=31,
                  color=C["ink"], bold=True, align="ctr", anchor="ctr")
        if subtitle:
            self.text(1.05, 2.42, 11.25, 0.58, [subtitle], size=21,
                      color=C["muted"], align="ctr", anchor="ctr")


def add_arrow(s: PlainSlide, x: float, y: float, text: str = "→") -> None:
    s.text(x, y, 0.40, 0.35, [text], size=24, color=C["muted"],
           bold=True, align="ctr", anchor="ctr")


def slide_deck() -> list[PlainSlide]:
    slides: list[PlainSlide] = []

    # 1. Title
    s = TitleSlide(
        "暑假阶段工作汇报",
        "极端稀疏 TM 修复：从 ARI-LLM 扩展尝试到轻量校准",
    )
    s.text(1.25, 3.70, 10.85, 0.40,
           ["方法暂名：CalibTM（calibration + traffic matrix）"], size=19,
           color=C["ink"], bold=True, align="ctr")
    s.text(1.25, 4.18, 10.85, 0.34,
           ["输入 Linear 初值，MLP 只输出受限修正量；真实观测最后原样保留"], size=15,
           color=C["muted"], align="ctr")
    s.text(1.25, 4.70, 10.85, 0.35, ["暑假两个月工作总结"], size=17,
           color=C["ink"], align="ctr")
    s.text(1.25, 5.11, 10.85, 0.32, ["苏梓坤    2026.09.03"], size=15,
           color=C["muted"], align="ctr")
    s.strip(2.05, 5.82, 9.25, 0.58,
            "本次汇报：任务与基线 / 方法与训练 / 对照实验 / 第三 WAN 与计算成本",
            fill=C["gray"], color=C["ink"], size=15)
    slides.append(s)

    # 2. Summer progression
    s = PlainSlide("这两个月完成的工作", "按时间顺序列出做过的事情和留下的材料")
    rows = [
        ["阶段", "做了什么", "留下的结果或材料"],
        ["任务与基线复核", "复现 ARI-LLM 的 K=3 任务；统一 Linear / ARI 指标口径",
         "A/G 分开看排序不同；all-mask 汇总为 0.2276 / 0.2290"],
        ["候选方法实验", "检查 backbone、跨 flow 信息和不同校准方式",
         "停止跨数据集不稳定的候选；固定 T=50、K=3"],
        ["固定方法后验证", "CalibTM 对比 Context-Free / Linear；增加 BRAIN",
         "A/G 各 3 seeds；BRAIN confirmation；统计与 CPU/GPU 测速"],
    ]
    s.plain_table(0.76, 1.55, [2.45, 4.75, 5.35], 1.08, rows,
                  font_size=14, aligns={0: "l", 1: "l", 2: "l"})
    s.text(0.93, 6.08, 11.50, 0.38,
           ["当前文件：固定方法实现、实验结果、中文论文稿、十分钟汇报材料。"],
           size=15, color=C["muted"], align="ctr")
    slides.append(s)

    # 3. Problem and the Linear observation
    s = PlainSlide("任务设置及 Linear / ARI 复算结果", "离线 TM 修复：T=50，每条 OD flow 仅保留 K=3 个真实观测")
    s.text(0.72, 1.38, 7.25, 0.36, ["一条长度 T=50 的 OD flow，只保留 3 个真实观测"],
           size=18, color=C["ink"], bold=True)
    start_x, y, w, gap = 0.75, 1.98, 0.185, 0.052
    anchors = {4, 24, 45}
    for i in range(50):
        fill = C["blue"] if i in anchors else C["gray2"]
        line = C["blue"] if i in anchors else C["line_light"]
        s.shape(start_x + i * (w + gap), y, w, 0.32, fill=fill, line=line)
    s.text(0.75, 2.39, 11.70, 0.56,
           ["蓝色是已知值，其余 47 个位置需要补回；因为是离线修复，可以同时看缺口左右两边",
            "K=3 规定保留几个点；三类 mask 只规定这 3 个点怎样放：随机 / 夹出一段内部缺口 / 夹出两段内部缺口"],
           size=12, color=C["muted"], align="ctr")
    s.text(0.82, 3.12, 5.20, 0.35, ["为什么先看 Linear？"], size=20,
           color=C["ink"], bold=True)
    s.text(0.98, 3.66, 5.00, 1.85,
           ["只连接真实观测点，不需要训练", "插值过程不使用被隐藏位置的真值", "同一套 canonical Linear 也作为 CalibTM 的初始底稿"],
           size=18, color=C["ink"], bullet=True)
    rows = [
        ["三类注册 mask 均纳入\nA/G 各 50%", "Linear", "ARI"],
        ["NMAE（越低越好）", "0.2276", "0.2290"],
    ]
    s.plain_table(6.65, 3.22, [2.65, 1.45, 1.45], 0.68, rows, font_size=15,
                  special_fills={(1, 1): C["blue_light"]})
    s.text(6.70, 4.58, 5.45, 0.28,
           ["每个网络内汇总三类 mask；3 seeds 等权，最后 A/G 各占 50%。"],
           size=10, color=C["muted"], align="ctr")
    s.text(6.70, 4.98, 5.45, 0.72,
           ["注意：Abilene 上 ARI 更好，GEANT 上 Linear 更好。", "这里只能说 Linear 在当前协议下很有竞争力。"],
           size=16, color=C["ink"], bullet=True)
    s.strip(1.25, 6.15, 10.85, 0.55,
            "后续固定比较：Linear、同参数 Context-Free、CalibTM",
            fill=C["gray"], color=C["ink"], size=17)
    s.text(0.82, 6.80, 11.55, 0.16,
           ["任务来源：ARI-LLM（Yan et al., INFOCOM 2026）；表中数值为当前项目同口径重算。"],
           size=8, color=C["muted"], align="l")
    slides.append(s)

    # 4. Closest literature and the resulting novelty boundary
    s = PlainSlide("与当前方法直接相关的工作", "保形插值、Linear 后的学习修正、插值先验和 TM 专用补全")
    rows = [
        ["研究线索", "代表工作", "已有工作说明了什么", "和当前工作的差别"],
        ["保形/受约束插值", "Monotone cubic, SIAM JNA 1980\nPCHIP rule, SIAM JSSC 1984\nSlope-Constrained Linear, MethodsX 2026",
         "插值可以显式保形；也可以在 Linear 周围加护栏",
         "前者无学习；后者限制斜率，并不能证明本文 10% bound"],
        ["Linear + 学习修正", "DCCN-SPF, Heliyon 2024",
         "可先用 Linear 表示趋势，再由网络补细节",
         "PM2.5 场景；使用卷积上下文，没有本文的硬边界与 TM 证据"],
        ["插值/先验引导补全", "IP-Net, ICLR 2019\nPriSTI, ICDE 2023",
         "插值结果或粗先验可以作为后续学习的条件",
         "分别是多变量插值网络和扩散模型，不是逐点小校准器"],
        ["TM 专用补全", "SRMF, IEEE/ACM ToN 2012\nWTTC-TS, NOLTA 2024",
         "TM 恢复可结合局部插值与低秩时空结构；也有无需预训练的简单路线",
         "当前方法刻意不使用 flow identity、topology 或跨 flow 建模"],
    ]
    s.plain_table(0.55, 1.43, [1.74, 2.70, 3.80, 4.00], 0.86, rows,
                  font_size=11, special_fills={(2, 0): C["blue_light"]},
                  aligns={0: "l", 1: "l", 2: "l", 3: "l"})
    s.strip(0.78, 6.07, 11.80, 0.73,
            "已有工作覆盖 Linear + learned refinement、shape constraints 和 interpolation prior。\n当前实验对象：K=3 TM、hard-copy、internal/edge bounds、5,475 参数及 matched control。",
            fill=C["gray"], color=C["ink"], size=14)
    slides.append(s)

    # 5. Method inputs and their rationale
    s = PlainSlide("MLP 输入：4 个有效通道", "每条 flow 先只用 3 个真实观测做归一化；表中列出缺失位置的实际取值")
    s.strip(0.82, 1.38, 11.70, 0.53,
            "μ_f、σ_f 只由当前 3 个锚点计算；标准化值 =（原始值 - μ_f）/ σ_f",
            fill=C["white"], color=C["ink"], size=16)
    rows = [
        ["输入", "它是什么", "原始设计意图", "在真正缺失的位置"],
        ["B_t", "标准化后的 Linear 初始值", "告诉 MLP 当前底稿是什么", "随位置变化，是主要逐点信号"],
        ["x_obs", "标准化后的当前位置观测值", "在统一接口中标记真点数值", "固定为 0，不会把锚点值传到缺失点"],
        ["m", "观测 mask：真点 1，缺失 0", "在统一接口中标记观测/缺失", "固定为 0；锚点保持实际由 hard-copy 保证"],
        ["v_bar", "锚点标准化后波动 / 当前窗口最大波动", "原设计用于区分 flow 的相对波动", "沿一条 flow 恒定；正常 flow≈1，近乎全平时较小"],
    ]
    s.plain_table(0.67, 2.16, [1.38, 3.05, 3.42, 4.15], 0.72, rows,
                  font_size=13, special_fills={(1, 0): C["blue_light"],
                                               (1, 3): C["blue_light"]},
                  aligns={0: "ctr", 1: "l", 2: "l", 3: "l"})
    s.strip(0.88, 5.91, 11.58, 0.66,
            "例：锚点 (t=5,20,40) 的原值为 (100,160,130)，则 μ=130、σ≈24.5。\n"
            "缺失 t=10：Linear=120，B_t=-0.408，而 x_obs=0、m=0；缺失 t=30：B_t=0.612，x_obs 仍为 0。",
            fill=C["gray"], color=C["ink"], size=12)
    s.text(0.92, 6.65, 11.46, 0.26,
           ["同一个逐点 MLP 用于所有 flow 和位置；不输入 flow identity、topology、routing，也不做跨时间聚合。"],
           size=13, color=C["muted"], align="ctr")
    slides.append(s)

    # 6. Method outputs and fixed constrained formulas
    s = PlainSlide("MLP 输出与缺失值计算公式", "x_L/x_R 是左右锚点，d_L/d_R 是距锚点距离，s_f 是可见锚点尺度；均为归一化量")
    s.strip(1.55, 1.37, 10.25, 0.50,
            "锚点与距离进入固定公式；共享 MLP 只输出 h_r、h_o、h_e，不直接输出最终缺失值",
            fill=C["white"], color=C["ink"], size=16)
    rows = [
        ["控制量", "计算方式", "用在哪里", "设计理由"],
        ["δr", "0.25 x tanh(h_r)", "中间缺口：微调插值比例 r", "允许沿两锚点方向小幅移动；范围不超过 ±0.25"],
        ["o", "0.10 x tanh(h_o)\nx (|x_R-x_L| + s_f + ε)", "中间缺口：上下偏移", "锚点差越大，允许的垂直修正相应增大，但仍有硬上限"],
        ["e", "0.10 x tanh(h_e) x s_f", "两端缺口：最近锚点 + e", "边界只有单侧锚点，因此单独处理；偏移限制在 flow 尺度内"],
    ]
    s.plain_table(0.72, 2.10, [1.25, 3.55, 2.72, 4.36], 0.72, rows,
                  font_size=12, special_fills={(1, 0): C["gray"],
                                               (2, 0): C["gray"],
                                               (3, 0): C["gray"]},
                  aligns={0: "ctr", 1: "l", 2: "l", 3: "l"})
    s.shape(0.82, 5.14, 7.17, 0.86, fill=C["white"], line=C["line"])
    s.text(0.96, 5.26, 6.89, 0.60,
           ["中间：r = d_L/(d_L+d_R)，r_hat = clip(r+δr, 0, 1)",
            "x_int = (1-r_hat)x_L + r_hat x_R + o"],
           size=13, color=C["ink"], align="l")
    s.shape(8.22, 5.14, 4.30, 0.86, fill=C["white"], line=C["line"])
    s.text(8.38, 5.28, 3.98, 0.55,
           ["边界：x_edge = 最近锚点 + e", "当前同一侧边界是整段常量偏移"],
           size=13, color=C["ink"], align="l")
    s.strip(0.85, 6.16, 11.62, 0.42,
            "最后：乘回 σ_f、加回 μ_f -> 负值截为 0 -> 3 个真实观测原样覆盖回来。",
            fill=C["gray"], color=C["ink"], size=14)
    s.text(0.91, 6.66, 11.48, 0.23,
           ["冻结 head 干预的相对改善：δr 约 −0.004%；internal offset 约 +2.50%；edge offset 约 +1.13%。"],
           size=12, color=C["muted"], align="ctr")
    slides.append(s)

    # 7. Training and matched control
    s = PlainSlide("训练流程与同参数对照", "正式评估仍是 K=3；K=2 只是训练时临时生成的辅助输入")
    train_flow = [
        (0.55, 1.57, 2.02, "完整历史窗口\n50 个真实值"),
        (2.95, 1.57, 2.10, "随机/成段遮挡\n只留下 K=3"),
        (5.43, 1.57, 2.20, "只用 3 个点\n生成 Linear"),
        (8.01, 1.57, 2.10, "MLP 修正\n预测另外 47 点"),
        (10.49, 1.57, 2.20, "只在缺失位置\n计算标准化 MAE"),
    ]
    for i, (x, y0, ww, txt) in enumerate(train_flow):
        fill = C["white"]
        s.box(x, y0, ww, 1.03, txt, fill=fill,
              line=C["line"], size=15,
              bold=i in (2, 3))
        if i < len(train_flow) - 1:
            add_arrow(s, x + ww + 0.07, 1.91)
    s.shape(0.76, 3.08, 5.78, 2.48, fill=C["white"], line=C["line"])
    s.text(0.96, 3.29, 5.38, 0.34, ["训练目标"], size=19,
           color=C["ink"], bold=True, align="ctr")
    s.text(1.03, 3.70, 5.22, 1.72,
           ["主任务：给 3 个锚点，误差算在其余 47 点", "辅助任务：复制同一样本，临时藏起按时间排序的中间锚点", "K=2 分支只看两端锚点；被藏点只作答案，不进入输入", "总损失 = 0.5 x K3误差 + 0.5 x K2误差；只更新 MLP"],
           size=12, color=C["ink"], bullet=True)
    s.shape(6.80, 3.08, 5.77, 2.48, fill=C["white"], line=C["line"])
    s.text(7.00, 3.29, 5.37, 0.34, ["同参数 Context-Free 对照"], size=19,
           color=C["ink"], bold=True, align="ctr")
    s.text(7.11, 3.78, 5.12, 1.52,
           ["同样的 Linear、外层公式和 5,475 参数", "同初始化、optimizer、训练量和数据", "唯一关键差别：它的 MLP 输入固定为 0", "因此只能学习一套全局固定控制量"],
           size=14, color=C["ink"], bullet=True)
    s.strip(0.90, 5.72, 11.54, 0.48,
            "K=2 的作用是增加一种更稀疏的训练视图；选择中间点，是因为它被左右两个锚点夹住。",
            fill=C["white"], color=C["ink"], size=14)
    s.text(0.92, 6.28, 11.48, 0.52,
           ["正式测试仍用 K=3。当前没有 K3-only 对照，不能说 K2 已被证明对 K3 必要。",
            "Context-Free 只改变 MLP 输入；bound、internal/edge 公式、训练量和隐藏真值均相同。"],
           size=11, color=C["muted"], align="ctr")
    s.text(0.94, 6.88, 11.44, 0.15,
           ["训练脉络：SAITS（Du et al., ESWA 2023）也采用从观测中再遮挡并监督被遮位置的 masked-imputation 任务。"],
           size=8, color=C["muted"], align="l")
    slides.append(s)

    # 8. Main results
    s = PlainSlide("A/G 成段缺失与 BRAIN 确认结果", "A/G：internal-block + two-burst；3 seeds 等权，A/G 各 50%；正数表示 NMAE 更低")
    rows = [
        ["数据集", "相对同参数固定校准", "相对 Linear"],
        ["Abilene", "+1.34%", "+1.89%"],
        ["GEANT", "+0.74%", "+1.15%"],
        ["A/G structured", "+1.04%", "+1.52%"],
        ["BRAIN", "+1.14%", "+3.26%"],
    ]
    s.plain_table(1.23, 1.52, [3.00, 3.75, 3.15], 0.72, rows,
                  font_size=17,
                  special_fills={(3, 1): C["gray"], (3, 2): C["gray"],
                                 (4, 1): C["gray"], (4, 2): C["gray"]},
                  aligns={0: "l"})
    s.text(1.26, 5.18, 9.90, 0.18,
           ["数据说明：GEANT（Uhlig et al., CCR 2006）；BRAIN 拓扑（Orlowski et al., Networks 2010）；Abilene 沿用 ARI 数据处理。"],
           size=8, color=C["muted"], align="l")
    s.text(0.94, 5.42, 11.35, 1.28,
           ["A/G 主结果只含两类成段缺失；Abilene、GEANT 分开看均为正方向。",
            "BRAIN：161 nodes、14,311 directed OD pairs；3 seeds × 2 类成段缺失均为正。",
            "方法/规则先固定；BRAIN 内训练并选 checkpoint，registry 冻结后评估 28 个窗口；不是 A/G→BRAIN zero-shot。"],
           size=14, color=C["ink"], bullet=True)
    slides.append(s)

    # 9. Large learned comparators and cost
    s = PlainSlide("A/G 全部缺失类型的精度与 A100 延迟估算", "NMAE：三类 mask 在各 dataset×seed 内合并，随后 3 seeds 等权；延迟为量级估算")
    rows = [
        ["方法", "Abilene", "GEANT", "A100 粗估"],
        ["Linear", "0.2376", "0.2176", "约 0.4-0.6 ms"],
        ["CalibTM", "0.2329", "0.2152", "约 3-5 ms"],
        ["ARI", "0.2303", "0.2278", "约 100-150 ms"],
        ["ImputeFormer", "0.2227", "0.3124", "约 20-40 ms"],
    ]
    s.plain_table(1.10, 1.48, [2.65, 2.35, 2.35, 3.05], 0.69, rows,
                  font_size=16,
                  special_fills={(1, 3): C["gray"], (2, 0): C["gray"],
                                 (2, 2): C["gray"]},
                  aligns={0: "l"})
    s.text(0.95, 5.23, 3.65, 0.32, ["分数据集结果"], size=18,
           color=C["ink"], bold=True, align="ctr")
    s.text(0.92, 5.62, 3.75, 0.68,
           ["Abilene：ImputeFormer 最低", "GEANT：CalibTM 最低"],
           size=15, color=C["ink"], bullet=True)
    s.text(4.83, 5.23, 3.65, 0.32, ["参数与延迟"], size=18,
           color=C["ink"], bold=True, align="ctr")
    s.text(4.85, 5.62, 3.70, 0.68,
           ["Linear：约 0.4-0.6 ms", "CalibTM：5,475 参数；约 3-5 ms"],
           size=15, color=C["ink"], bullet=True)
    s.text(8.70, 5.23, 3.70, 0.32, ["估算说明"], size=18,
           color=C["ink"], bold=True, align="ctr")
    s.text(8.72, 5.62, 3.68, 0.68,
           ["由 H800 实测换算", "不是 A100 实测"],
           size=15, color=C["ink"], bullet=True)
    s.strip(1.05, 6.48, 11.25, 0.34,
            "A100 数字由 H800 实测、官方峰值规格和 batch scaling 粗估，非实测；论文仍报告 H800/CPU 实测。",
            fill=C["gray"], color=C["muted"], size=13)
    s.text(1.08, 6.86, 11.18, 0.15,
           ["模型来源：ARI-LLM（INFOCOM 2026）；ImputeFormer（KDD 2024）。原始 H800 为统一环境实测，A100 仅为粗估。"],
           size=8, color=C["muted"], align="l")
    slides.append(s)

    # 10. Claims and limitations
    s = PlainSlide("实验覆盖范围与尚未覆盖内容", "已完成项和未完成项按同一行对应列出")
    rows = [
        ["已做过", "还没有做"],
        ["T=50、K=3；三类人工 mask", "真实 WAN TM 的原生缺失轨迹"],
        ["A/G：各 3 个训练 seed；另有 K=2--5 辅助检查", "其他 T；针对各 K 分别重训的完整曲线"],
        ["BRAIN：3 seeds × 2 masks；28 个 confirmation 窗口", "matched bounded vs unbounded / direct 对照"],
        ["CPU/H800 实测；另有已消费数据上的冻结跨 WAN replay", "BRAIN 大型 baseline；独立新数据上的跨 WAN zero-shot"],
    ]
    s.plain_table(0.82, 1.55, [5.85, 5.85], 0.92, rows,
                  font_size=16, aligns={0: "l", 1: "l"})
    s.strip(0.95, 6.38, 11.45, 0.48,
            "BRAIN 的 OD demand 由 link measurement 经 LP 反演；28 个连续窗口约覆盖一天。",
            fill=C["gray"], color=C["ink"], size=15)
    slides.append(s)

    # 11. Summary and next steps
    s = PlainSlide("已完成内容与下一步", "请老师和师兄判断投稿前还缺哪一项")
    rows = [
        ["项目", "当前内容"],
        ["任务与基线", "T=50、K=3；all-mask A/G：Linear 0.2276，ARI 0.2290"],
        ["当前实现", "5,475 参数；4 输入、3 head；hard-copy + nonnegative"],
        ["已完成实验", "A/G 各 3 seeds；BRAIN confirmation；同参数对照；CPU/GPU 测速"],
    ]
    s.plain_table(0.95, 1.48, [2.55, 8.95], 0.72, rows,
                  font_size=15, aligns={0: "l", 1: "l"})
    s.text(1.02, 4.62, 2.15, 0.35, ["下一步"], size=20,
           color=C["ink"], bold=True)
    s.text(1.18, 5.02, 10.80, 0.86,
           ["固定当前实现与现有结果，整理可复现材料", "把中文稿迁入正式英文双栏模板，重做主图和主表",
            "由老师决定投稿前只补一项：机制 matched control 或一个可靠 TM baseline"],
           size=16, color=C["ink"], bullet=True)
    s.strip(1.22, 6.12, 10.85, 0.52,
            "请老师和师兄帮我判断：目前这条线够不够形成一篇论文？还缺哪个最关键实验？",
            fill=C["white"], color=C["ink"], size=17)
    slides.append(s)

    # 12. Backup: distinguish the three notions of scale
    s = PlainSlide("备用：几个容易混淆的量", "主讲不用逐项念；老师追问归一化或尺度时再翻到这里")
    rows = [
        ["符号", "怎样计算", "实际作用", "需要注意"],
        ["μ_f", "当前 flow 的 3 个可见锚点均值", "标准化时减去；恢复时加回来", "不使用缺失位置真值"],
        ["σ_f", "当前 flow 的 3 个可见锚点原始标准差", "把不同数量级 OD flow 拉到相近范围；最后乘回来", "原始单位中的主要尺度适配来自它"],
        ["s_f", "标准化后，再对可见锚点计算标准差", "控制内部/边界 offset 的允许幅度", "常规 K=3 非退化 flow 中接近 1"],
        ["v_bar", "s_f / 当前窗口所有 flow 的最大 s", "作为 MLP 的第 4 个有效输入", "正常 K=3 flow≈1；全平锚点约 0.001；且受窗口组成影响"],
        ["B_t", "raw Linear 经 μ_f、σ_f 标准化", "告诉 MLP 当前需要校准的底稿", "边界段为最近锚点常值延拓"],
    ]
    s.plain_table(0.66, 1.46, [1.18, 3.42, 3.62, 3.78], 0.72, rows,
                  font_size=12, special_fills={(5, 0): C["blue_light"],
                                               (5, 2): C["blue_light"]},
                  aligns={0: "ctr", 1: "l", 2: "l", 3: "l"})
    s.strip(0.92, 6.15, 11.50, 0.54,
            "常规 K=3 非退化 flow 中 s_f 约为 1，v_bar 在 flow 内恒定；当前逐位置变化主要由 B_t 提供。",
            fill=C["gray"], color=C["ink"], size=14)
    slides.append(s)

    # 13. Backup: parameter ledger
    s = PlainSlide("备用：5,475 个参数从哪里来？", "16-slot 是冻结 checkpoint 的兼容接口；方法语义上只有前面解释的 4 个有效输入")
    s.box(0.82, 1.50, 11.70, 0.70,
          "LayerNorm(16) -> Linear(16,64) -> GELU -> Linear(64,64) -> GELU -> 3 个 Linear(64,1) head",
          fill=C["white"], line=C["line"], size=17, bold=True)
    rows = [
        ["部分", "参数计算", "参数量"],
        ["LayerNorm(16)", "weight 16 + bias 16", "32"],
        ["Linear 16 -> 64", "16 x 64 + bias 64", "1,088"],
        ["Linear 64 -> 64", "64 x 64 + bias 64", "4,160"],
        ["3 个输出 head", "3 x (64 + bias 1)", "195"],
        ["合计", "32 + 1,088 + 4,160 + 195", "5,475"],
    ]
    s.plain_table(1.70, 2.57, [3.18, 4.52, 2.25], 0.58, rows,
                  font_size=14, special_fills={(5, 0): C["blue_light"],
                                               (5, 2): C["blue_light"]},
                  aligns={0: "l", 1: "l", 2: "ctr"})
    s.strip(1.08, 6.31, 11.18, 0.47,
            "Checkpoint 保留 16-slot 接口；4 个槽接入上述变量，其余 12 个槽固定为 0。",
            fill=C["gray"], color=C["muted"], size=14)
    slides.append(s)

    # 14. Backup: honest implementation boundaries
    s = PlainSlide("备用：当前实现的已知边界", "用于回答模型输入、边界外推和 head 干预结果等追问")
    s.shape(0.78, 1.45, 5.78, 4.92, fill=C["white"], line=C["line"])
    s.shape(0.78, 1.45, 5.78, 0.58, fill=C["gray"], line=C["line"])
    s.text(0.98, 1.60, 5.38, 0.27, ["MLP 没有看到的信息"], size=19,
           color=C["ink"], bold=True, align="ctr")
    s.text(1.08, 2.30, 5.08, 3.60,
           ["没有直接输入左右锚点值", "没有输入相对位置、gap 长度或距离", "没有输入左右边界方向和 anchor slope",
            "不做整段时间聚合，也不做可识别的跨 flow 建模", "Abilene / GEANT / BRAIN 当前分别训练 checkpoint"],
           size=15, color=C["ink"], bullet=True)
    s.shape(6.82, 1.45, 5.73, 4.92, fill=C["white"], line=C["line"])
    s.shape(6.82, 1.45, 5.73, 0.58, fill=C["gray"], line=C["line"])
    s.text(7.02, 1.60, 5.33, 0.27, ["由当前实现直接得到的现象"], size=19,
           color=C["ink"], bold=True, align="ctr")
    s.text(7.12, 2.30, 5.03, 3.60,
            ["缺失点逐位置变化主要来自 B_t", "同一侧边界主要是整段常量偏移，不是趋势外推",
            "δr 的冻结干预接近 0；收益主要在 internal / edge offset", "Context-Free 对照只改变条件输入，未改变 bound",
            "若改输入、删 head 或改边界，需要重新训练并另设未使用确认集"],
           size=15, color=C["ink"], bullet=True)
    s.strip(1.02, 6.52, 11.28, 0.33,
            "本次汇报中的数值均来自当前冻结实现；未把事后建议算入现有方法。",
            fill=C["gray"], color=C["muted"], size=13)
    slides.append(s)

    for i, slide in enumerate(slides, start=1):
        if i > 1:
            slide.footer(i)
    return slides


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    base.slide_deck = slide_deck
    base.build()


if __name__ == "__main__":
    main()
