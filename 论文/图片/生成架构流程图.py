# -*- coding: utf-8 -*-
"""生成四层一体化架构流程图（语义回响论文用）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 8.2), dpi=160)
ax.set_xlim(0, 12)
ax.set_ylim(0, 8.2)
ax.axis("off")

# 颜色
C1 = "#5a6a80"   # L1 基座 灰蓝
C2 = "#4fd1ff"   # L2 回响 青
C3 = "#fbbf24"   # L3 RAG 金
C4 = "#9d7bff"   # L4 LoRA 紫
CBG = "#f8fafc"
CLN = "#cbd5e1"

def box(x, y, w, h, color, title, sub, fc="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.25",
                       linewidth=2.2, edgecolor=color, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + 0.5, y + h * 0.68, title, fontsize=13, fontweight="bold", color="#0f172a", va="center")
    ax.text(x + 0.5, y + h * 0.30, sub, fontsize=9.5, color="#475569", va="center")
    return (x + w, y + h / 2)

def layer_badge(x, y, text, color):
    b = FancyBboxPatch((x, y), 1.05, 0.62, boxstyle="round,pad=0.06,rounding_size=0.2",
                       linewidth=0, facecolor=color)
    ax.add_patch(b)
    ax.text(x + 0.525, y + 0.31, text, fontsize=12, fontweight="bold", color="#04121a",
            ha="center", va="center")

# 标题
ax.text(6, 7.85, "语义回响 · 四层一体化推理架构", fontsize=17, fontweight="bold",
        ha="center", color="#0f172a")
ax.text(6, 7.5, "SemanticEcho — Four-Layer Integrated Inference Framework（1.5B 最优配置）",
        fontsize=10, ha="center", color="#64748b")

# 四层（从下到上：L1 在最底）
# L4 LoRA
box(2.6, 6.15, 6.8, 1.0, C4, "L4 · LoRA 语态适配器", "长期记忆 · gentle_v2 / emotion_v1 / emotion_7B\n热插拔 · r=8 α=16 · q/k/v/o_proj")
layer_badge(1.35, 6.34, "L4", C4)
# 箭头 L4 -> L3
ax.add_patch(FancyArrowPatch((6.0, 6.12), (6.0, 5.62), arrowstyle="-|>", mutation_scale=22, linewidth=2, color=CLN))

# L3 RAG
box(2.6, 4.62, 6.8, 1.0, C3, "L3 · RAG 向量检索", "中期记忆 · bge-small-zh-v1.5 + FAISS（151 条知识库）\n[参考信息] 前缀注入 · 缺失自动回退")
layer_badge(1.35, 4.81, "L3", C3)
ax.add_patch(FancyArrowPatch((6.0, 4.59), (6.0, 4.09), arrowstyle="-|>", mutation_scale=22, linewidth=2, color=CLN))

# L2 语义回响（核心，高亮）
box(2.6, 3.09, 6.8, 1.0, C2, "L2 · 语义回响 Echo（本次实证唯一启用层）", "短期记忆 · 回响池（指数衰减 + 情感词库筛选）\n最后 4 层 hidden_state 捕获 → 随机投影注入 logits")
layer_badge(1.35, 3.28, "L2", C2)
ax.add_patch(FancyArrowPatch((6.0, 3.06), (6.0, 2.56), arrowstyle="-|>", mutation_scale=22, linewidth=2, color=CLN))

# L1 微调基座
box(2.6, 1.56, 6.8, 1.0, C1, "L1 · 微调基座", "推理底座 · Qwen2.5-1.5B-Instruct（fp16 / 4bit NF4）\n权重不变 · 旁路注入 · 可插拔")
layer_badge(1.35, 1.75, "L1", C1)

# 右侧横切机制
ax.text(10.15, 5.9, "横切机制", fontsize=11, fontweight="bold", color="#0f172a")
ax.text(10.15, 5.45, "① 动态策略：情感密度 > 0.15\n   → τ 降至 0.05（策略 B）", fontsize=9, color="#334155", va="top")
ax.text(10.15, 4.35, "② λ 跨尺度归一化\n   λ × 896 / hidden_dim", fontsize=9, color="#334155", va="top")
ax.text(10.15, 3.3, "③ 思考阶段分离\n   思考 λ=0.08 / 正文 λ=0", fontsize=9, color="#334155", va="top")
ax.text(10.15, 2.3, "④ 跨轮持久回响池\n   会话级情绪连续性", fontsize=9, color="#334155", va="top")

# 底部输入输出
ax.add_patch(FancyArrowPatch((6.0, 1.53), (6.0, 1.1), arrowstyle="-|>", mutation_scale=22, linewidth=2, color=CLN))
b = FancyBboxPatch((3.3, 0.35), 5.4, 0.7, boxstyle="round,pad=0.08,rounding_size=0.2",
                   linewidth=0, facecolor="#eef2f7")
ax.add_patch(b)
ax.text(6.0, 0.7, "输出：更细腻、更像人的回复（人味增强）", fontsize=10.5,
        ha="center", va="center", color="#0f172a", fontweight="bold")

plt.tight_layout()
out = r"i:\Desktop\语义回响\论文\图片\四层架构流程图.png"
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("已生成:", out)
