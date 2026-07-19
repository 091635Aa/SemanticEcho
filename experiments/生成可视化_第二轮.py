"""
生成可视化_第二轮.py — 读取两轮实验 JSON 数据，生成四张对比可视化图表。

图表列表:
  图A: 两轮实验语义熵对比柱状图
  图B: 情感筛选对语义熵的提升率柱状图
  图C: 情感命中率分布柱状图
  图D: 完整实验矩阵全景图（λ 与语义熵的关系）
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ── 全局 matplotlib 中文支持 ──────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 路径 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "实验数据")
OUTPUT_DIR = os.path.join(DATA_DIR, "可视化")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 所有实验文件路径
EXP_FILES: dict[str, str] = {
    "E1": os.path.join(DATA_DIR, "E1.json"),
    "E2": os.path.join(DATA_DIR, "E2.json"),
    "E3": os.path.join(DATA_DIR, "E3.json"),
    "E4": os.path.join(DATA_DIR, "E4.json"),
    "E5": os.path.join(DATA_DIR, "E5.json"),
    "E6": os.path.join(DATA_DIR, "E6.json"),
    "E7": os.path.join(DATA_DIR, "E7.json"),
    "E8": os.path.join(DATA_DIR, "E8.json"),
    "E9": os.path.join(DATA_DIR, "E9.json"),
    "E10": os.path.join(DATA_DIR, "E10.json"),
}

SUMMARY_R1 = os.path.join(DATA_DIR, "实验结果汇总.json")
SUMMARY_R2 = os.path.join(DATA_DIR, "实验结果汇总_第二轮.json")


# ── 数据加载与计算 ──────────────────────────────────────────────────────


def load_json(filepath: str) -> dict[str, Any]:
    """加载 JSON 文件，文件不存在或解析失败时抛出异常。

    Parameters
    ----------
    filepath : str
        JSON 文件路径。

    Returns
    -------
    dict[str, Any]
        解析后的字典。

    Raises
    ------
    FileNotFoundError
        指定的文件不存在。
    json.JSONDecodeError
        文件内容不是合法的 JSON。
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件未找到: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def calc_avg_entropy(filepath: str) -> float:
    """从单个实验 JSON 中计算所有重复结果的平均语义熵。

    遍历 '数据' → '重复结果' → '平均熵'，返回全局平均值。

    Parameters
    ----------
    filepath : str
        实验 JSON 文件路径。

    Returns
    -------
    float
        平均语义熵。

    Raises
    ------
    ValueError
        文件中未找到有效的平均熵数据。
    """
    data = load_json(filepath)
    avg_entropies: list[float] = []
    for entry in data.get("数据", []):
        for repeat in entry.get("重复结果", []):
            avg_val = repeat.get("平均熵")
            if avg_val is not None:
                avg_entropies.append(float(avg_val))
    if not avg_entropies:
        raise ValueError(f"未找到有效平均熵数据: {filepath}")
    return float(np.mean(avg_entropies))


def calc_avg_hit_rate(filepath: str) -> float:
    """从单个实验 JSON 的池统计中计算平均情感命中率。

    遍历 '数据' → '重复结果' → '池统计.情感命中率'，返回全局平均值。

    Parameters
    ----------
    filepath : str
        实验 JSON 文件路径。

    Returns
    -------
    float
        平均情感命中率。

    Raises
    ------
    ValueError
        文件中未找到情感命中率数据。
    """
    data = load_json(filepath)
    hit_rates: list[float] = []
    for entry in data.get("数据", []):
        for repeat in entry.get("重复结果", []):
            pool = repeat.get("池统计")
            if pool is not None and "情感命中率" in pool:
                hit_rates.append(float(pool["情感命中率"]))
    if not hit_rates:
        raise ValueError(f"未找到情感命中率数据: {filepath}")
    return float(np.mean(hit_rates))


def get_lambda_strength(filepath: str) -> float:
    """从实验 JSON 的统计中提取 lambda_strength。

    Parameters
    ----------
    filepath : str
        实验 JSON 文件路径。

    Returns
    -------
    float
        lambda 强度值。若为 null 则返回 0.0。
    """
    data = load_json(filepath)
    stat = data.get("统计", {})
    ls = stat.get("lambda_strength")
    return 0.0 if ls is None else float(ls)


# ── 图A: 两轮实验语义熵对比柱状图 ─────────────────────────────────────


def plot_figure_a() -> str:
    """生成图A：两轮实验语义熵对比柱状图。

    X轴为实验编号 (E1, E3, E4, E7, E8, E9, E10)，
    第一轮（E1/E3/E4）用蓝色，第二轮（E7/E8/E9/E10）用橙色。

    Returns
    -------
    str
        保存的文件路径。
    """
    first_round_ids = ["E1", "E3", "E4"]
    second_round_ids = ["E7", "E8", "E9", "E10"]
    all_ids = first_round_ids + second_round_ids

    values: list[float] = []
    bar_colors: list[str] = []
    for eid in all_ids:
        values.append(calc_avg_entropy(EXP_FILES[eid]))
        bar_colors.append("#4A90D9" if eid in first_round_ids else "#F5A623")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(all_ids))
    bars = ax.bar(
        x_pos,
        values,
        width=0.55,
        color=bar_colors,
        edgecolor="gray",
        linewidth=0.8,
        alpha=0.9,
    )

    # 柱子上标注数值
    for bar_item, val in zip(bars, values):
        ax.text(
            bar_item.get_x() + bar_item.get_width() / 2.0,
            bar_item.get_height() + 0.03,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_ids, fontsize=11, fontweight="bold")
    ax.set_title("两轮实验语义熵对比：情感筛选 vs 无筛选", fontsize=14, fontweight="bold")
    ax.set_xlabel("实验编号", fontsize=12)
    ax.set_ylabel("平均语义熵", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # 图例
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#4A90D9", alpha=0.9, label="第一轮（无情感筛选）"),
        Patch(color="#F5A623", alpha=0.9, label="第二轮（有情感筛选）"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="upper left")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图A_两轮熵对比.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图A] 已保存: {save_path}")
    return save_path


# ── 图B: 情感筛选提升率柱状图 ─────────────────────────────────────────


def plot_figure_b() -> str:
    """生成图B：情感筛选对语义熵的提升率柱状图。

    对比组: E7 vs E3 (λ=0.5), E8 vs E4 (λ=1.0), E10 vs E4 (λ=1.0+思考阶段)。
    提升率 = (第二轮熵 - 第一轮熵) / 第一轮熵 × 100%。

    Returns
    -------
    str
        保存的文件路径。
    """
    comparisons: list[tuple[str, str, str]] = [
        ("E7 vs E3\n(λ=0.5, 情感筛选)", "E7", "E3"),
        ("E8 vs E4\n(λ=1.0, 情感筛选)", "E8", "E4"),
        ("E10 vs E4\n(λ=1.0, 筛选+思考)", "E10", "E4"),
    ]

    labels: list[str] = []
    rates: list[float] = []
    for label, r2_id, r1_id in comparisons:
        r1_val = calc_avg_entropy(EXP_FILES[r1_id])
        r2_val = calc_avg_entropy(EXP_FILES[r2_id])
        rate = (r2_val - r1_val) / r1_val * 100.0
        labels.append(label)
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(9, 6))
    x_pos = np.arange(len(labels))
    bar_colors = ["#4ECDC4", "#FF6B6B", "#AA96DA"]
    bars = ax.bar(
        x_pos,
        rates,
        width=0.45,
        color=bar_colors,
        edgecolor="gray",
        linewidth=0.8,
        alpha=0.85,
    )

    # 标注数值
    for bar_item, rate_val in zip(bars, rates):
        y_offset = 1.5 if rate_val >= 0 else -4.5
        va = "bottom" if rate_val >= 0 else "top"
        ax.text(
            bar_item.get_x() + bar_item.get_width() / 2.0,
            bar_item.get_height() + y_offset,
            f"{rate_val:+.2f}%",
            ha="center",
            va=va,
            fontsize=11,
            fontweight="bold",
            color="black",
        )

    # 零线
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("情感筛选对语义熵的提升效果", fontsize=14, fontweight="bold")
    ax.set_ylabel("提升率 (%)", fontsize=12)
    ax.set_xlabel("对比组", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图B_筛选提升率.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图B] 已保存: {save_path}")
    return save_path


# ── 图C: 情感命中率分布 ────────────────────────────────────────────────


def plot_figure_c() -> str:
    """生成图C：不同配置下的情感词命中率柱状图。

    从 E7/E8/E9/E10 的池统计中提取平均情感命中率。

    Returns
    -------
    str
        保存的文件路径。
    """
    exp_ids = ["E7", "E8", "E9", "E10"]
    labels_map: dict[str, str] = {
        "E7": "E7\nλ=0.5\n筛选",
        "E8": "E8\nλ=1.0\n筛选",
        "E9": "E9\nλ=0.5\n筛选+思考",
        "E10": "E10\nλ=1.0\n筛选+思考",
    }

    values: list[float] = []
    labels: list[str] = []
    for eid in exp_ids:
        values.append(calc_avg_hit_rate(EXP_FILES[eid]))
        labels.append(labels_map[eid])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x_pos = np.arange(len(labels))
    bar_colors = ["#4ECDC4", "#FF6B6B", "#95E1D3", "#F38181"]
    bars = ax.bar(
        x_pos,
        values,
        width=0.5,
        color=bar_colors,
        edgecolor="gray",
        linewidth=0.8,
        alpha=0.85,
    )

    # 标注数值
    for bar_item, val in zip(bars, values):
        ax.text(
            bar_item.get_x() + bar_item.get_width() / 2.0,
            bar_item.get_height() + 0.005,
            f"{val:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("不同配置下的情感词命中率", fontsize=14, fontweight="bold")
    ax.set_xlabel("实验配置", fontsize=12)
    ax.set_ylabel("平均情感命中率", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图C_情感命中率.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图C] 已保存: {save_path}")
    return save_path


# ── 图D: 完整实验矩阵全景图 ──────────────────────────────────────────


def plot_figure_d() -> str:
    """生成图D：完整实验矩阵全景图 — λ 与语义熵的关系。

    第一轮（实线）: E1(λ=0), E3(λ=0.5), E4(λ=1.0), E5(λ=2.0)
    第二轮（虚线）: E7(λ=0.5), E8(λ=1.0)

    Returns
    -------
    str
        保存的文件路径。
    """
    # 第一轮数据
    r1_lambdas: list[float] = [0.0, 0.5, 1.0, 2.0]
    r1_ids = ["E1", "E3", "E4", "E5"]
    r1_entropies: list[float] = []
    for eid in r1_ids:
        r1_entropies.append(calc_avg_entropy(EXP_FILES[eid]))

    # 第二轮数据（仅含 λ=0.5 和 λ=1.0）
    r2_lambdas: list[float] = [0.5, 1.0]
    r2_ids = ["E7", "E8"]
    r2_entropies: list[float] = []
    for eid in r2_ids:
        r2_entropies.append(calc_avg_entropy(EXP_FILES[eid]))

    fig, ax = plt.subplots(figsize=(9, 6))

    # 第一轮 - 实线
    ax.plot(
        r1_lambdas,
        r1_entropies,
        marker="o",
        markersize=10,
        linewidth=2.5,
        color="#4A90D9",
        markerfacecolor="#4A90D9",
        markeredgecolor="black",
        markeredgewidth=1.5,
        linestyle="-",
        label="第一轮（无情感筛选）",
    )

    # 第二轮 - 虚线
    ax.plot(
        r2_lambdas,
        r2_entropies,
        marker="s",
        markersize=10,
        linewidth=2.5,
        color="#F5A623",
        markerfacecolor="#F5A623",
        markeredgecolor="black",
        markeredgewidth=1.5,
        linestyle="--",
        label="第二轮（有情感筛选）",
    )

    # 标注第一轮数值
    for lv, ev in zip(r1_lambdas, r1_entropies):
        ax.annotate(
            f"{ev:.4f}",
            (lv, ev),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#4A90D9",
            fontweight="bold",
        )

    # 标注第二轮数值
    for lv, ev in zip(r2_lambdas, r2_entropies):
        ax.annotate(
            f"{ev:.4f}",
            (lv, ev),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#F5A623",
            fontweight="bold",
        )

    ax.set_xlabel("λ 强度", fontsize=12)
    ax.set_ylabel("平均语义熵", fontsize=12)
    ax.set_title("语义回响完整实验矩阵：λ 与语义熵的关系", fontsize=14, fontweight="bold")
    ax.set_xticks([0.0, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0\n(Baseline)", "0.5", "1.0", "2.0"])
    ax.set_xlim(-0.2, 2.2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图D_完整实验矩阵.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图D] 已保存: {save_path}")
    return save_path


# ── 主函数 ───────────────────────────────────────────────────────────────


def main() -> None:
    """主入口：加载数据并生成全部四张图表。"""
    print("=" * 60)
    print("开始生成第二轮可视化图表...")
    print("=" * 60)

    # 图A: 两轮实验语义熵对比柱状图
    plot_figure_a()

    # 图B: 情感筛选提升率柱状图
    plot_figure_b()

    # 图C: 情感命中率分布
    plot_figure_c()

    # 图D: 完整实验矩阵全景图
    plot_figure_d()

    print("\n" + "=" * 60)
    print(f"全部图表已生成到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
