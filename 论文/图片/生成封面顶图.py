# -*- coding: utf-8 -*-
"""生成论文封面顶图：五层底层记忆化 AI 扮演架构（决策/扮演/情感主题）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12.5, 8.6), dpi=170)
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 8.6)
ax.axis("off")

# ── 背景渐变 ──
bg = np.linspace(0.96, 0.88, 256).reshape(-1, 1)
bg = np.repeat(bg, 1, axis=1)
bg_img = np.zeros((256, 1, 4))
bg_img[..., 0] = 0.075
bg_img[..., 1] = 0.11
bg_img[..., 2] = 0.20
bg_img[..., 3] = 1.0
ax.imshow(bg_img, extent=[0, 12.5, 0, 8.6], aspect="auto", interpolation="bicubic", zorder=0)

# ── 顶部标题区 ──
ax.text(6.25, 8.02, "底 层 记 忆 化  A I  扮 演 架 构", fontsize=25, fontweight="bold",
        ha="center", va="center", color="#ffffff", zorder=5)
ax.text(6.25, 7.55, "DECISION · PERSONA · EMOTION", fontsize=12, ha="center", va="center",
        color="#8fb4e8", zorder=5, family="Consolas")
ax.text(6.25, 7.18, "让大模型更有决策 · 更会扮演 · 更有情感", fontsize=14.5, ha="center", va="center",
        color="#ffd98a", zorder=5)

# ── 三大能力徽章 ──
def badge(x, text, color):
    b = FancyBboxPatch((x - 0.95, 6.42), 1.9, 0.52, boxstyle="round,pad=0.06,rounding_size=0.2",
                       linewidth=0, facecolor=color, zorder=6)
    ax.add_patch(b)
    ax.text(x, 6.68, text, fontsize=12, fontweight="bold", ha="center", va="center",
            color="#ffffff", zorder=7)
badge(2.4, "决  策  性", "#e0567d")
badge(6.25, "扮  演  一  致", "#4fd1ff")
badge(10.1, "情  感  表  达", "#9d7bff")

# ── 五层架构堆叠（自下而上 L0→L4） ──
layers = [
    ("L4", "外挂记忆 · 对话历史", "Ext. Memory · Dialogue History", "#5a8fc9"),
    ("L3", "RAG · 知识检索", "Knowledge Retrieval", "#4fb7a3"),
    ("L2", "LoRA · 风格外挂", "Style Adapter", "#e8b04b"),
    ("L1", "语义回响 · 思考链中断注入", "Semantic Echo · Think-Break Injection", "#4fd1ff"),
    ("L0", "模型底层 · 遗忘与身世内化", "Bottom-Layer · Amnesia & Persona Internalization", "#ff7a59"),
]
y_top = 6.05
bar_h = 0.88
gap = 0.14
for i, (tag, zh, en, color) in enumerate(layers):
    y = y_top - i * (bar_h + gap)
    # 高亮 L0 与 L1（核心）
    glow = color if i in (0, 4) else "#2a3b5e"
    b = FancyBboxPatch((1.6, y), 7.1, bar_h, boxstyle="round,pad=0.08,rounding_size=0.22",
                       linewidth=2.4, edgecolor=color, facecolor=glow, zorder=5,
                       linestyle="-")
    ax.add_patch(b)
    # 层级徽章
    bb = FancyBboxPatch((0.75, y + 0.12), 0.68, bar_h - 0.24, boxstyle="round,pad=0.06,rounding_size=0.15",
                        linewidth=0, facecolor=color, zorder=6)
    ax.add_patch(bb)
    ax.text(1.09, y + bar_h / 2, tag, fontsize=12.5, fontweight="bold", ha="center",
            va="center", color="#04121a", zorder=7)
    ax.text(2.1, y + bar_h * 0.66, zh, fontsize=12.5, fontweight="bold", ha="left",
            va="center", color="#ffffff", zorder=6)
    ax.text(2.1, y + bar_h * 0.30, en, fontsize=8.2, ha="left", va="center",
            color="#a9c4e8", zorder=6, family="Consolas")

# 层间箭头
for i in range(len(layers) - 1):
    y1 = y_top - i * (bar_h + gap) - gap * 0.35
    ax.add_patch(FancyArrowPatch((5.15, y1 + 0.02), (5.15, y1 - 0.02),
                                 arrowstyle="-|>", mutation_scale=14, linewidth=1.6,
                                 color="#5a7aa8", zorder=4))

# 右侧：两级记忆说明
ax.text(10.2, 5.75, "两级记忆体系", fontsize=11.5, fontweight="bold", color="#ffd98a", zorder=5)
ax.text(10.2, 5.30, "① 外挂记忆\n   对话历史（L4）\n   持续累积 · 1M 不够聊", fontsize=9,
        color="#cfe0f5", va="top", zorder=5)
ax.text(10.2, 3.95, "② 模型自身记忆\n   身世 / 世界观（L0）\n   一生固定 · 长在权重里", fontsize=9,
        color="#cfe0f5", va="top", zorder=5)
ax.text(10.2, 2.55, "提示词层大砍\n   仅留固定短锚点", fontsize=9, color="#ffb3a0", va="top", zorder=5)

# ── 底部标语 ──
ax.text(6.25, 0.62, "灾难性遗忘从「训练事故」变为「主动工程手段」", fontsize=13,
        ha="center", va="center", color="#ffd98a", fontweight="bold", zorder=5)
ax.text(6.25, 0.22, "扮演从「符号层」下沉到「模型底层」 ｜ 情感是剂量效应，不是开关", fontsize=9.5,
        ha="center", va="center", color="#8fb4e8", zorder=5)

plt.tight_layout()
out = r"i:\Desktop\语义回响\论文\图片\封面顶图_五层架构.png"
plt.savefig(out, bbox_inches="tight", facecolor="#13203a")
print("已生成:", out)
