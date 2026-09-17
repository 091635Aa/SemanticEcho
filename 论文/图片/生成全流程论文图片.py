# -*- coding: utf-8 -*-
"""全流程最终版论文图片生成（五层架构 / 思考链中断流程 / 两级记忆决策 / 实验对比）"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

图片目录 = r"i:\Desktop\语义回响\论文\图片"
os.makedirs(图片目录, exist_ok=True)

C0 = "#ff6b6b"   # L0 模型底层 红
C1 = "#4fd1ff"   # L1 回响 青
C2 = "#fbbf24"   # L2 LoRA 金
C3 = "#34d399"   # L3 RAG 绿
C4 = "#9d7bff"   # L4 外挂记忆 紫
CLN = "#94a3b8"


def box(ax, x, y, w, h, color, title, sub, fc="white", fs_t=12.5, fs_s=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.25",
                       linewidth=2.2, edgecolor=color, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h * 0.66, title, fontsize=fs_t, fontweight="bold",
            color="#0f172a", ha="center", va="center")
    ax.text(x + w / 2, y + h * 0.28, sub, fontsize=fs_s, color="#475569",
            ha="center", va="center")


def 图1_五层架构():
    fig, ax = plt.subplots(figsize=(12.5, 9.2), dpi=160)
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 9.2); ax.axis("off")
    ax.text(6.25, 8.9, "底层记忆化 AI 扮演架构 · 五层全景", fontsize=18, fontweight="bold",
            ha="center", color="#0f172a")
    ax.text(6.25, 8.5, "Bottom-Layer Memorized AI Acting Architecture — Five-Layer Overview (2026)",
            fontsize=10, ha="center", color="#64748b")

    # L4 外挂记忆
    box(ax, 2.8, 7.1, 6.6, 0.95, C4, "L4 · 外挂记忆", "对话历史（你我聊天记录）· 检索/记忆库\ncontext 1M 也不够装 → 必须外置", fs_t=12)
    ax.add_patch(FancyArrowPatch((6.1, 7.05), (6.1, 6.45), arrowstyle="-|>", mutation_scale=22, linewidth=2, color=CLN))
    # L3 RAG
    box(ax, 2.8, 5.5, 6.6, 0.95, C3, "L3 · RAG 向量检索", "知识检索 · bge-small-zh + FAISS\n命中质量低时自动关闭", fs_t=12)
    ax.add_patch(FancyArrowPatch((6.1, 5.45), (6.1, 4.85), arrowstyle="-|>", mutation_scale=22, linewidth=2, color=CLN))
    # L2 LoRA
    box(ax, 2.8, 3.9, 6.6, 0.95, C2, "L2 · LoRA 语态适配器", "风格外挂（可选）· peft 热切换\n失败自动回退基座", fs_t=12)
    ax.add_patch(FancyArrowPatch((6.1, 3.85), (6.1, 3.25), arrowstyle="-|>", mutation_scale=22, linewidth=2, color=CLN))
    # L1 语义回响
    box(ax, 2.8, 2.3, 6.6, 0.95, C1, "L1 · 语义回响（思考链中断注入）", "即时情感状态 · <think>捕获 → </think>硬中断 → 总体向量注入\n情感先在脑中想好，再表达", fs_t=12)
    ax.add_patch(FancyArrowPatch((6.1, 2.25), (6.1, 1.65), arrowstyle="-|>", mutation_scale=22, linewidth=2, color=CLN))
    # L0 模型底层（高亮）
    box(ax, 2.8, 0.65, 6.6, 1.0, C0, "L0 · 模型底层（直接调整模型本身）",
        "风格预训练 → 灾难性遗忘（忘掉自己是 AI）→ 身世内化（模型自身记忆）\n不依赖外挂 · 权重级改造 · 角色即模型", fs_t=12.5, fc="#fff5f5")
    ax.text(1.5, 4.6, "推\n理\n期\n外\n挂", fontsize=11, color="#64748b", ha="center", va="center", rotation=0)
    ax.text(1.5, 1.15, "训\n练\n期\n权\n重", fontsize=11, color="#dc2626", ha="center", va="center")

    # 右侧机制
    ax.text(10.3, 7.3, "两级记忆", fontsize=11.5, fontweight="bold", color="#0f172a")
    ax.text(10.3, 6.8, "外挂记忆：对话历史\n（第一步，context 装不下）", fontsize=9, color="#334155", va="top")
    ax.text(10.3, 5.4, "模型自身记忆：身世/背景\n（第二步，微调内化进权重）", fontsize=9, color="#334155", va="top")
    ax.text(10.3, 4.0, "提示词层（大砍）", fontsize=11.5, fontweight="bold", color="#0f172a")
    ax.text(10.3, 3.5, "仅固定短锚点：\n“你现在是XX，你在真实世界”", fontsize=9, color="#334155", va="top")
    ax.text(10.3, 2.2, "通用兼容", fontsize=11.5, fontweight="bold", color="#0f172a")
    ax.text(10.3, 1.7, "有预制参数→深度定制\n无参数→思考链兜底\n在线坍缩检测", fontsize=9, color="#334155", va="top")

    plt.tight_layout()
    out = os.path.join(图片目录, "五层架构图.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(); print("已生成:", out)


def 图2_思考链中断流程():
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=160)
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.text(6.5, 4.9, "思考链中断注入流程（L1 · 语义回响）", fontsize=15, fontweight="bold",
            ha="center", color="#0f172a")

    steps = [
        (0.4, C1, "① 思考阶段", "模型输出 <think>…\n思考时尽量用情感词", "捕获情感\nhidden_state 入池"),
        (2.9, C0, "② 硬中断", "检测到 </think>\n（token 级，Qwen3 id=151668）", "强行停止生成"),
        (5.4, C0, "③ 向量定格", "计算思考阶段全部向量\n的加权质心", "总体向量 = 情绪底色"),
        (7.9, C1, "④ 正文注入", "logits += 总体向量 @ 投影 × λ\n每个 token 固定偏置", "表示层携带情感"),
        (10.4, C4, "⑤ 输出", "正文自然表达\n不重复、不坍缩", "更有温度"),
    ]
    for i, (x, c, t, s1, s2) in enumerate(steps):
        b = FancyBboxPatch((x, 1.2), 2.25, 2.6, boxstyle="round,pad=0.1,rounding_size=0.3",
                           linewidth=2.2, edgecolor=c, facecolor="white")
        ax.add_patch(b)
        ax.text(x + 1.125, 3.35, t, fontsize=12.5, fontweight="bold", color=c, ha="center", va="center")
        ax.text(x + 1.125, 2.5, s1, fontsize=9, color="#334155", ha="center", va="center")
        ax.text(x + 1.125, 1.75, s2, fontsize=8.5, color="#64748b", ha="center", va="center")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 2.3, 2.5), (x + 2.82, 2.5),
                                         arrowstyle="-|>", mutation_scale=26, linewidth=2.4, color=CLN))
    # 底部说明
    b = FancyBboxPatch((2.0, 0.15), 9.0, 0.75, boxstyle="round,pad=0.08,rounding_size=0.2",
                       linewidth=0, facecolor="#eef2f7")
    ax.add_patch(b)
    ax.text(6.5, 0.52, "实测：Qwen3-1.7B 重复率 0.84（坍缩）→ 0.0036；Qwen3-4B 重 0.007，熵 +56%",
            fontsize=10, ha="center", va="center", color="#0f172a", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(图片目录, "思考链中断流程图.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(); print("已生成:", out)


def 图3_实验对比():
    fig, ax = plt.subplots(figsize=(10.5, 5), dpi=160)
    模型 = ["Qwen3-4B", "Qwen3-1.7B"]
    裸 = [0.0137, 0.0140]
    全面 = [0.0086, 0.84]
    思考链 = [0.0066, 0.0036]
    x = np.arange(len(模型)); w = 0.25
    b1 = ax.bar(x - w, 裸, w, label="裸模型", color="#cbd5e1")
    b2 = ax.bar(x, 全面, w, label="全面向量纠正（预制参数）", color="#fbbf24")
    b3 = ax.bar(x + w, 思考链, w, label="思考链纠正（中断注入）", color="#4fd1ff")
    for bars in (b1, b2, b3):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.01,
                    f"{r.get_height():.4f}".rstrip('0'), ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(模型)
    ax.set_ylabel("重复率（越低越好）")
    ax.set_title("语义回响各方案重复率对比（同种子 42 · runs=2 · 512 token）", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.set_ylim(0, 0.92)
    ax.axhline(0.15, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.text(1.55, 0.155, "坍缩线 0.15", color="#dc2626", fontsize=9)
    plt.tight_layout()
    out = os.path.join(图片目录, "思考链实验对比图.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(); print("已生成:", out)


if __name__ == "__main__":
    图1_五层架构()
    图2_思考链中断流程()
    图3_实验对比()
    print("全部图片生成完成")
