"""
生成增强可视化.py — 生成覆盖所有实验数据(E1-E10)的10种增强图表。

图表列表:
  图1: λ与语义熵的折线图（带误差带）—— 第一轮 vs 第二轮
  图2: 两轮实验语义熵对比柱状图
  图3: 情感命中率分布饼图
  图4: 各提示词语义熵散点图
  图5: 语义熵分布箱线图
  图6: 实验配置×情感维度的熵矩阵热力图
  图7: 多维度综合评估雷达图
  图8: 池质心范数演化趋势堆积面积图
  图9: 细腻度提升率对比条形图
  图10: λ-熵-情感命中率三维散点图
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ── 全局 matplotlib 设置 ──────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning)

# ── 路径 ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "实验数据")
OUTPUT_DIR = os.path.join(DATA_DIR, "可视化")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 实验文件映射 ──────────────────────────────────────────────────────────
EXPERIMENT_FILES: dict[str, str] = {
    f"E{i}": os.path.join(DATA_DIR, f"E{i}.json") for i in range(1, 11)
}

# ── 实验标签与描述 ────────────────────────────────────────────────────────
EXPERIMENT_LABELS: dict[str, str] = {
    "E1": "E1\nBaseline\n(top_p=0.9)",
    "E2": "E2\nBaseline\n(T=1.0)",
    "E3": "E3\nEcho\n(λ=0.5)",
    "E4": "E4\nEcho\n(λ=1.0)",
    "E5": "E5\nEcho\n(λ=2.0)",
    "E6": "E6\nEcho\n(λ=1.0,γ=0.01)",
    "E7": "E7\nEcho(λ=0.5)\n+筛选",
    "E8": "E8\nEcho(λ=1.0)\n+筛选",
    "E9": "E9\nEcho(λ=0.5)\n+筛选+思考",
    "E10": "E10\nEcho(λ=1.0)\n+筛选+思考",
}

EXPERIMENT_DESCRIPTION: dict[str, str] = {
    "E1": "Baseline (top_p=0.9)",
    "E2": "Baseline (temperature=1.0)",
    "E3": "Echo (λ=0.5, γ=0.05)",
    "E4": "Echo (λ=1.0, γ=0.1)",
    "E5": "Echo (λ=2.0, γ=0.5)",
    "E6": "Echo (λ=1.0, γ=0.01)",
    "E7": "Echo (λ=0.5) + 情感筛选",
    "E8": "Echo (λ=1.0) + 情感筛选",
    "E9": "Echo (λ=0.5) + 筛选 + 思考阶段",
    "E10": "Echo (λ=1.0) + 筛选 + 思考阶段",
}

# ── λ 配置映射 ────────────────────────────────────────────────────────────
LAMBDA_CONFIG: dict[str, float] = {
    "E1": 0.0,
    "E3": 0.5,
    "E4": 1.0,
    "E5": 2.0,
    "E7": 0.5,
    "E8": 1.0,
    "E9": 0.5,
    "E10": 1.0,
}

# ── 情感维度（按数据中出现顺序排列） ──────────────────────────────────────
EMOTION_DIMENSIONS: list[str] = ["开心", "悲伤", "愤怒", "中性", "复杂混合"]

# ── 统一配色 ──────────────────────────────────────────────────────────────
COLOR_BLUE = "#4A90D9"
COLOR_ORANGE = "#E8833A"
COLOR_BLUE_LIGHT = "#7BB3E0"
COLOR_ORANGE_LIGHT = "#F0A668"
COLOR_GREEN = "#2ECC71"
COLOR_RED = "#E74C3C"
COLOR_PURPLE = "#9B59B6"

ROUND1_COLORS = ["#4A90D9", "#5BA3E6", "#7BB3E0", "#A3C9ED", "#3A7BD5"]
ROUND2_COLORS = ["#E8833A", "#F0A668", "#F5C496", "#E67E22", "#D35400"]

FIRST_ROUND_IDS: list[str] = ["E1", "E2", "E3", "E4", "E5", "E6"]
SECOND_ROUND_IDS: list[str] = ["E7", "E8", "E9", "E10"]
ALL_EXP_IDS: list[str] = FIRST_ROUND_IDS + SECOND_ROUND_IDS


# ══════════════════════════════════════════════════════════════════════════
#  数据加载与提取
# ══════════════════════════════════════════════════════════════════════════


def load_json(filepath: str) -> dict[str, Any]:
    """加载 JSON 文件。

    Parameters
    ----------
    filepath : str
        JSON 文件绝对路径。

    Returns
    -------
    dict[str, Any]
        解析后的字典。

    Raises
    ------
    FileNotFoundError
        文件不存在时抛出。
    json.JSONDecodeError
        JSON 格式错误时抛出。
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件未找到: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def extract_experiment_data(
    filepath: str,
) -> tuple[list[float], list[dict[str, Any]], list[list[float]]]:
    """从单个实验 JSON 中提取所有重复结果的数据。

    提取每个重复结果的平均熵、池统计（若存在）和熵列表（每步熵）。

    Parameters
    ----------
    filepath : str
        JSON 文件路径。

    Returns
    -------
    entropies : list[float]
        所有（提示词×重复）的平均熵值。
    pool_stats : list[dict[str, Any]]
        每条重复结果中的池统计（若存在）。
    entropy_sequences : list[list[float]]
        每条重复结果中的熵序列（每步的熵值）。
    """
    data = load_json(filepath)
    entropies: list[float] = []
    pool_stats: list[dict[str, Any]] = []
    entropy_sequences: list[list[float]] = []

    for entry in data.get("数据", []):
        dimension = entry.get("维度", "未知")
        for repeat in entry.get("重复结果", []):
            avg_entropy = repeat.get("平均熵")
            if avg_entropy is not None:
                entropies.append(float(avg_entropy))

            entropy_list = repeat.get("熵列表")
            if entropy_list and isinstance(entropy_list, list):
                entropy_sequences.append([float(v) for v in entropy_list])

            pool = repeat.get("池统计")
            if pool is not None:
                enriched: dict[str, Any] = dict(pool)
                enriched["维度"] = dimension
                enriched["提示词"] = entry.get("提示词", "未知")
                pool_stats.append(enriched)

    return entropies, pool_stats, entropy_sequences


def load_all_experiments() -> dict[str, dict[str, Any]]:
    """加载所有实验数据。

    Returns
    -------
    dict[str, dict[str, Any]]
        {实验ID: {entropies, pool_stats, entropy_sequences, config}}。
    """
    result: dict[str, dict[str, Any]] = {}
    for exp_id in ALL_EXP_IDS:
        try:
            entropies, pool_stats, entropy_sequences = extract_experiment_data(
                EXPERIMENT_FILES[exp_id]
            )
            result[exp_id] = {
                "entropies": entropies,
                "pool_stats": pool_stats,
                "entropy_sequences": entropy_sequences,
                "config": load_json(EXPERIMENT_FILES[exp_id]).get("统计", {}),
            }
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"[警告] 无法加载 {exp_id}: {exc}")
    return result


def get_experiment_avg_entropy(
    exp_data: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """计算每个实验的平均语义熵。

    Parameters
    ----------
    exp_data : dict
        由 load_all_experiments() 返回的数据。

    Returns
    -------
    dict[str, float]
        {实验ID: 平均语义熵}。
    """
    return {
        eid: float(np.mean(entry["entropies"]))
        for eid, entry in exp_data.items()
        if entry["entropies"]
    }


def get_experiment_entropy_stats(
    exp_data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """计算每个实验的熵统计量（均值、标准差、中位数、Q1、Q3）。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    dict[str, dict[str, float]]
        {实验ID: {mean, std, median, q1, q3}}。
    """
    stats: dict[str, dict[str, float]] = {}
    for eid, entry in exp_data.items():
        vals = entry["entropies"]
        if not vals:
            continue
        stats[eid] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75)),
        }
    return stats


def get_emotion_entropy_matrix(
    exp_data: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], np.ndarray]:
    """构建实验×情感维度的熵矩阵。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    tuple[list[str], list[str], np.ndarray]
        (实验ID列表, 情感维度列表, 熵矩阵)。
    """
    # 对每个实验和维度，收集所有重复的平均熵
    matrix: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    exp_ids_in_order: list[str] = [
        eid for eid in ALL_EXP_IDS if eid in exp_data and exp_data[eid]["entropies"]
    ]

    for eid in exp_ids_in_order:
        filepath = EXPERIMENT_FILES[eid]
        data = load_json(filepath)
        for entry in data.get("数据", []):
            dim = entry.get("维度", "未知")
            for repeat in entry.get("重复结果", []):
                avg_entropy = repeat.get("平均熵")
                if avg_entropy is not None:
                    matrix[eid][dim].append(float(avg_entropy))

    dims = EMOTION_DIMENSIONS
    n_row, n_col = len(exp_ids_in_order), len(dims)
    entropy_grid = np.full((n_row, n_col), np.nan)

    for i, eid in enumerate(exp_ids_in_order):
        for j, dim in enumerate(dims):
            vals = matrix[eid].get(dim, [])
            if vals:
                entropy_grid[i, j] = float(np.mean(vals))

    return exp_ids_in_order, dims, entropy_grid


# ══════════════════════════════════════════════════════════════════════════
#  图1: 折线图 — λ与语义熵的关系（带误差带）
# ══════════════════════════════════════════════════════════════════════════


def plot_figure1(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图1：λ与语义熵的关系折线图（带误差带）。

    第一轮（E1→E3→E4→E5）：λ=[0,0.5,1.0,2.0]，实线。
    第二轮（E1→E7→E8）：λ=[0,0.5,1.0]，虚线。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    # 第一轮数据点
    round1_ids = ["E1", "E3", "E4", "E5"]
    round1_lambdas: list[float] = []
    round1_means: list[float] = []
    round1_stds: list[float] = []

    for eid in round1_ids:
        entry = exp_data.get(eid)
        if entry and entry["entropies"]:
            round1_lambdas.append(LAMBDA_CONFIG[eid])
            round1_means.append(float(np.mean(entry["entropies"])))
            round1_stds.append(float(np.std(entry["entropies"])))

    # 第二轮数据点
    round2_ids = ["E1", "E7", "E8"]
    round2_lambdas: list[float] = []
    round2_means: list[float] = []
    round2_stds: list[float] = []

    for eid in round2_ids:
        entry = exp_data.get(eid)
        if entry and entry["entropies"]:
            round2_lambdas.append(LAMBDA_CONFIG[eid])
            round2_means.append(float(np.mean(entry["entropies"])))
            round2_stds.append(float(np.std(entry["entropies"])))

    fig, ax = plt.subplots(figsize=(9, 6))

    # 第一轮 — 实线
    ax.plot(
        round1_lambdas,
        round1_means,
        marker="o",
        markersize=10,
        linewidth=2.5,
        color=COLOR_BLUE,
        markerfacecolor=COLOR_BLUE_LIGHT,
        markeredgecolor=COLOR_BLUE,
        markeredgewidth=1.5,
        linestyle="-",
        label="第一轮 (Echo)",
    )
    ax.fill_between(
        round1_lambdas,
        [m - s for m, s in zip(round1_means, round1_stds)],
        [m + s for m, s in zip(round1_means, round1_stds)],
        alpha=0.15,
        color=COLOR_BLUE,
        label="第一轮 ±1σ",
    )
    for lv, mv, sv in zip(round1_lambdas, round1_means, round1_stds):
        ax.annotate(
            f"{mv:.3f}",
            (lv, mv),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLOR_BLUE,
            fontweight="bold",
        )

    # 第二轮 — 虚线
    ax.plot(
        round2_lambdas,
        round2_means,
        marker="s",
        markersize=10,
        linewidth=2.5,
        color=COLOR_ORANGE,
        markerfacecolor=COLOR_ORANGE_LIGHT,
        markeredgecolor=COLOR_ORANGE,
        markeredgewidth=1.5,
        linestyle="--",
        label="第二轮 (Echo+筛选)",
    )
    ax.fill_between(
        round2_lambdas,
        [m - s for m, s in zip(round2_means, round2_stds)],
        [m + s for m, s in zip(round2_means, round2_stds)],
        alpha=0.15,
        color=COLOR_ORANGE,
        label="第二轮 ±1σ",
    )
    for lv, mv, sv in zip(round2_lambdas, round2_means, round2_stds):
        ax.annotate(
            f"{mv:.3f}",
            (lv, mv),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLOR_ORANGE,
            fontweight="bold",
        )

    ax.set_xlabel("λ 强度", fontsize=12)
    ax.set_ylabel("平均语义熵", fontsize=12)
    ax.set_title("λ 与语义熵的关系（带误差带）", fontsize=14, fontweight="bold")
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图1_λ与熵的折线图.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图1] 已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════
#  图2: 柱状图 — 两轮实验语义熵对比
# ══════════════════════════════════════════════════════════════════════════


def plot_figure2(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图2：两轮实验语义熵对比分组柱状图。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    round1_exps = ["E1", "E3", "E4"]
    round2_exps = ["E7", "E8", "E9", "E10"]
    all_exps = round1_exps + round2_exps

    means: list[float] = []
    errors: list[float] = []
    categories: list[str] = []
    group_colors: list[str] = []

    for eid in round1_exps:
        entry = exp_data.get(eid)
        if entry and entry["entropies"]:
            means.append(float(np.mean(entry["entropies"])))
            errors.append(float(np.std(entry["entropies"])))
            categories.append(EXPERIMENT_DESCRIPTION[eid])
            group_colors.append(COLOR_BLUE)

    for eid in round2_exps:
        entry = exp_data.get(eid)
        if entry and entry["entropies"]:
            means.append(float(np.mean(entry["entropies"])))
            errors.append(float(np.std(entry["entropies"])))
            categories.append(EXPERIMENT_DESCRIPTION[eid])
            group_colors.append(COLOR_ORANGE)

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(categories))

    bars = ax.bar(
        x_pos,
        means,
        yerr=errors,
        capsize=5,
        width=0.55,
        color=group_colors,
        alpha=0.85,
        edgecolor="gray",
        linewidth=0.5,
        error_kw={"linewidth": 1.5, "ecolor": "black"},
    )

    # 数值标注
    for bar_item, mean_val in zip(bars, means):
        ax.text(
            bar_item.get_x() + bar_item.get_width() / 2.0,
            bar_item.get_height() + 0.05,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    # 添加分组标识
    round1_end = len(round1_exps) - 0.5
    ax.axvline(x=round1_end, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(
        (round1_exps.__len__() - 1) / 2,
        max(means) * 1.08,
        "第一轮（纯Echo）",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=COLOR_BLUE,
    )
    ax.text(
        round1_exps.__len__() + (round2_exps.__len__() - 1) / 2,
        max(means) * 1.08,
        "第二轮（筛选/思考）",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=COLOR_ORANGE,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("平均语义熵", fontsize=12)
    ax.set_title("两轮实验语义熵对比", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图2_两轮熵对比柱状图.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图2] 已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════
#  图3: 饼图 — 情感命中率分布
# ══════════════════════════════════════════════════════════════════════════


def plot_figure3(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图3：各配置情感命中率分布饼图（E7-E10）。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    # 计算每个实验的平均情感命中率
    hit_rates: dict[str, float] = {}
    for eid in ["E7", "E8", "E9", "E10"]:
        entry = exp_data.get(eid)
        if entry is None:
            continue
        rates = [
            ps.get("情感命中率", 0)
            for ps in entry["pool_stats"]
            if ps.get("情感命中率") is not None
        ]
        if rates:
            hit_rates[eid] = float(np.mean(rates))

    if not hit_rates:
        print("[图3] 无情感命中率数据")
        save_path = os.path.join(OUTPUT_DIR, "图3_情感命中率饼图.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无可用数据", ha="center", va="center", fontsize=14)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 饼图
    labels_pie = [EXPERIMENT_DESCRIPTION[eid] for eid in hit_rates]
    sizes = list(hit_rates.values())
    colors_pie = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE]
    explode = [0.05] * len(sizes)

    wedges, texts_pie, autotexts = ax1.pie(
        sizes,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors_pie[: len(sizes)],
        explode=explode,
        pctdistance=0.75,
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    ax1.legend(
        wedges,
        labels_pie,
        title="实验配置",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
    )
    ax1.set_title("各配置情感命中率分布", fontsize=13, fontweight="bold")

    # 辅助柱状图
    bars_pos = np.arange(len(hit_rates))
    bar_colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE]
    ax2.bar(
        bars_pos,
        sizes,
        width=0.5,
        color=bar_colors[: len(sizes)],
        alpha=0.8,
        edgecolor="gray",
    )
    for i, (eid, rate) in enumerate(hit_rates.items()):
        ax2.text(
            i,
            rate + 0.01,
            f"{rate:.2%}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_xticks(bars_pos)
    ax2.set_xticklabels(
        [EXPERIMENT_DESCRIPTION[eid] for eid in hit_rates],
        rotation=15,
        ha="right",
        fontsize=8,
    )
    ax2.set_ylabel("平均情感命中率", fontsize=11)
    ax2.set_title("情感命中率数值对比", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图3_情感命中率饼图.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图3] 已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════
#  图4: 散点图 — 各提示词语义熵分布
# ══════════════════════════════════════════════════════════════════════════


def plot_figure4(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图4：各提示词语义熵分布散点图。

    按情感维度着色，展示每个提示词在不同实验中的语义熵分布。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    dim_colors = {
        "开心": "#E74C3C",
        "悲伤": "#3498DB",
        "愤怒": "#E67E22",
        "中性": "#2ECC71",
        "复杂混合": "#9B59B6",
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    # 收集每个实验中每个维度的平均熵
    all_x: list[float] = []
    all_y: list[float] = []
    all_dim: list[str] = []

    for exp_idx, eid in enumerate(ALL_EXP_IDS):
        entry = exp_data.get(eid)
        if entry is None:
            continue
        filepath = EXPERIMENT_FILES[eid]
        data = load_json(filepath)
        for entry_data in data.get("数据", []):
            dim = entry_data.get("维度", "未知")
            for repeat in entry_data.get("重复结果", []):
                avg_entropy = repeat.get("平均熵")
                if avg_entropy is not None:
                    all_x.append(float(exp_idx))
                    all_y.append(float(avg_entropy))
                    all_dim.append(dim)

    # 绘制散点
    for dim in sorted(set(all_dim)):
        mask = [d == dim for d in all_dim]
        x_vals = [all_x[i] for i in range(len(all_x)) if mask[i]]
        y_vals = [all_y[i] for i in range(len(all_y)) if mask[i]]
        color = dim_colors.get(dim, "#333333")
        ax.scatter(
            x_vals,
            y_vals,
            alpha=0.5,
            s=40,
            c=color,
            edgecolors="black",
            linewidths=0.3,
            label=dim,
        )

    ax.set_xticks(range(len(ALL_EXP_IDS)))
    ax.set_xticklabels(
        [EXPERIMENT_LABELS.get(eid, eid) for eid in ALL_EXP_IDS],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    ax.set_xlabel("实验编号", fontsize=12)
    ax.set_ylabel("平均语义熵", fontsize=12)
    ax.set_title("各提示词语义熵分布", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, title="情感维度", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图4_语义熵散点图.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图4] 已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════
#  图5: 箱线图 — 语义熵分布
# ══════════════════════════════════════════════════════════════════════════


def plot_figure5(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图5：所有实验语义熵分布箱线图。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    exp_ids_order = ALL_EXP_IDS
    all_entropies: list[list[float]] = []
    labels: list[str] = []

    for eid in exp_ids_order:
        entry = exp_data.get(eid)
        if entry is None or not entry["entropies"]:
            continue
        all_entropies.append(entry["entropies"])
        labels.append(
            EXPERIMENT_LABELS.get(eid, eid).replace("\n", " ")
        )

    if not all_entropies:
        print("[图5] 无有效数据")

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(
        all_entropies,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "red",
            "markersize": 6,
            "markeredgecolor": "darkred",
        },
    )

    # 按轮次配色
    colors = []
    for i, eid in enumerate(exp_ids_order[: len(all_entropies)]):
        if eid in SECOND_ROUND_IDS:
            colors.append(COLOR_ORANGE_LIGHT)
        else:
            colors.append(COLOR_BLUE_LIGHT)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 标注均值
    means = [float(np.mean(d)) for d in all_entropies]
    for i, mean_val in enumerate(means):
        ax.annotate(
            f"{mean_val:.3f}",
            xy=(i + 1, mean_val),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="red",
            fontweight="bold",
        )

    # 添加轮次分隔线
    round1_end = len(FIRST_ROUND_IDS)
    if round1_end < len(labels):
        ax.axvline(
            x=round1_end + 0.5,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )
        ax.text(
            round1_end / 2 + 0.5,
            ax.get_ylim()[1] * 0.98,
            "第一轮",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=COLOR_BLUE,
        )
        ax.text(
            round1_end + (len(exp_ids_order) - round1_end) / 2 + 0.5,
            ax.get_ylim()[1] * 0.98,
            "第二轮",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=COLOR_ORANGE,
        )

    ax.set_title("所有实验语义熵分布箱线图", fontsize=14, fontweight="bold")
    ax.set_ylabel("平均语义熵", fontsize=11)
    ax.set_xlabel("实验配置", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图5_语义熵箱线图.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图5] 已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════
#  图6: 热力图 — 实验配置×情感维度的熵矩阵
# ══════════════════════════════════════════════════════════════════════════


def plot_figure6(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图6：实验配置×情感维度的熵矩阵热力图。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    exp_ids_order, dims, entropy_grid = get_emotion_entropy_matrix(exp_data)

    if entropy_grid.size == 0:
        print("[图6] 无有效数据")

    fig, ax = plt.subplots(figsize=(12, 8))

    # 自定义颜色映射：低熵=蓝，高熵=红
    vmin = float(np.nanmin(entropy_grid))
    vmax = float(np.nanmax(entropy_grid))
    cmap = plt.cm.RdYlBu_r  # 红（高熵）→ 蓝（低熵）

    im = ax.imshow(entropy_grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # 标注数值
    for i in range(entropy_grid.shape[0]):
        for j in range(entropy_grid.shape[1]):
            val = entropy_grid[i, j]
            if not np.isnan(val):
                text_color = "white" if val > (vmin + vmax) / 2 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_yticks(range(len(exp_ids_order)))
    ax.set_yticklabels(
        [f"{eid}: {EXPERIMENT_DESCRIPTION.get(eid, eid)}" for eid in exp_ids_order],
        fontsize=8,
    )
    ax.set_xlabel("情感维度", fontsize=12)
    ax.set_ylabel("实验配置", fontsize=12)
    ax.set_title("实验配置 × 情感维度的语义熵矩阵", fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("平均语义熵", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图6_实验配置热力图.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图6] 已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════
#  图7: 雷达图/极坐标图 — 多维度综合评估
# ══════════════════════════════════════════════════════════════════════════


def plot_figure7(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图7：多维度综合评估雷达图（极坐标）。

    展示各实验在不同情感维度上的语义熵表现。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    exp_ids_order, dims, entropy_grid = get_emotion_entropy_matrix(exp_data)

    if entropy_grid.size == 0:
        print("[图7] 无有效数据")

    n_dims = len(dims)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # 选择有代表性的实验（不超过6个，避免太拥挤）
    selected_exps = ["E1", "E3", "E4", "E7", "E9", "E10"]
    colors_radar = [
        "#4A90D9",
        "#2ECC71",
        "#E74C3C",
        "#E8833A",
        "#9B59B6",
        "#1ABC9C",
    ]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    for idx, eid in enumerate(selected_exps):
        if eid not in exp_ids_order:
            continue
        row_idx = exp_ids_order.index(eid)
        values = [float(entropy_grid[row_idx, j]) for j in range(n_dims)]
        values += values[:1]  # close

        color = colors_radar[idx % len(colors_radar)]
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=1.8,
            color=color,
            label=f"{eid}: {EXPERIMENT_DESCRIPTION.get(eid, eid)}",
            alpha=0.8,
        )
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_ylim(0, max(entropy_grid[~np.isnan(entropy_grid)]) * 1.2)
    ax.set_title("多维度综合评估雷达图", fontsize=14, fontweight="bold", pad=25)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        fontsize=9,
        title="实验配置",
    )
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图7_多维度雷达图.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图7] 已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════
#  图8: 堆积面积图 — 池质心范数演化趋势
# ══════════════════════════════════════════════════════════════════════════


def plot_figure8(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图8：质心范数演化趋势（用熵序列代替，展示收敛趋势）。

    由于 JSON 数据中没有每一步的质心范数，使用熵序列（每步熵值）
    来展示生成过程的收敛趋势。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    # 选择典型实验展示收敛趋势
    target_exps = ["E1", "E3", "E4", "E5", "E7", "E8"]
    line_colors = {
        "E1": "#4A90D9",
        "E3": "#2ECC71",
        "E4": "#E74C3C",
        "E5": "#9B59B6",
        "E7": "#E8833A",
        "E8": "#1ABC9C",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for eid in target_exps:
        entry = exp_data.get(eid)
        if entry is None or not entry["entropy_sequences"]:
            continue

        # 计算平均熵序列（对齐到最短序列）
        sequences = entry["entropy_sequences"]
        min_len = min(len(seq) for seq in sequences)

        avg_seq = [
            float(np.mean([seq[step] for seq in sequences]))
            for step in range(min_len)
        ]

        color = line_colors.get(eid, "#333333")
        ax.plot(
            range(min_len),
            avg_seq,
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label=f"{eid}: {EXPERIMENT_DESCRIPTION.get(eid, eid)}",
        )

    ax.set_xlabel("生成步数", fontsize=12)
    ax.set_ylabel("平均熵值", fontsize=12)
    ax.set_title("生成过程中语义熵演化趋势", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, 200)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图8_质心范数趋势图.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图8] 已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════
#  图9: 对比条形图 — 细腻度提升率
# ══════════════════════════════════════════════════════════════════════════


def plot_figure9(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图9：各回响实验相对基线的细腻度提升率对比图。

    以 E1 (Baseline top_p=0.9) 为基线，计算各 Echo 实验的熵变化率。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    baseline_entry = exp_data.get("E1")
    if baseline_entry is None or not baseline_entry["entropies"]:
        raise ValueError("E1 基线数据缺失，无法计算提升率。")

    h_baseline = float(np.mean(baseline_entry["entropies"]))

    echo_exps = ["E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]
    exp_ids: list[str] = []
    rates: list[float] = []

    for eid in echo_exps:
        entry = exp_data.get(eid)
        if entry is None or not entry["entropies"]:
            print(f"[图9] {eid} 数据不足，跳过")
            continue
        h_echo = float(np.mean(entry["entropies"]))
        rate = (h_echo - h_baseline) / h_baseline * 100.0
        exp_ids.append(eid)
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(exp_ids))
    bar_colors = [COLOR_GREEN if r > 0 else COLOR_RED for r in rates]

    bars = ax.bar(
        x_pos,
        rates,
        width=0.55,
        color=bar_colors,
        edgecolor="gray",
        alpha=0.85,
        linewidth=0.5,
    )

    # 标注数值
    for bar_item, rate_val in zip(bars, rates):
        y_offset = 1.5 if rate_val >= 0 else -4.5
        ax.text(
            bar_item.get_x() + bar_item.get_width() / 2.0,
            bar_item.get_height() + y_offset,
            f"{rate_val:+.2f}%",
            ha="center",
            va="bottom" if rate_val >= 0 else "top",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    # 零线
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [EXPERIMENT_DESCRIPTION.get(eid, eid) for eid in exp_ids],
        rotation=20,
        ha="right",
        fontsize=8,
    )
    ax.set_title("各实验相对 E1 (Baseline) 的细腻度提升率", fontsize=14, fontweight="bold")
    ax.set_ylabel("熵变化率 (%)", fontsize=12)
    ax.set_xlabel("实验配置", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图9_细腻度提升率对比图.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图9] 已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════
#  图10: 三维散点图 — λ, 熵, 情感命中率关系
# ══════════════════════════════════════════════════════════════════════════


def plot_figure10(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图10：λ-语义熵-情感命中率三维散点图。

    使用 mplot3d 展示三个变量之间的关系，气泡大小表示情感命中率。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # 只选取有 λ 值和情感命中率的实验
    target_exps = ["E3", "E4", "E5", "E7", "E8", "E9", "E10"]

    lambda_vals: list[float] = []
    entropy_vals: list[float] = []
    hit_rate_vals: list[float] = []
    labels_3d: list[str] = []

    for eid in target_exps:
        entry = exp_data.get(eid)
        if entry is None:
            continue

        # 平均熵
        if not entry["entropies"]:
            continue
        avg_entropy = float(np.mean(entry["entropies"]))

        # 平均情感命中率
        rates = [
            ps.get("情感命中率", 0)
            for ps in entry["pool_stats"]
            if ps.get("情感命中率") is not None
        ]
        if not rates:
            continue
        avg_hit = float(np.mean(rates))

        lambda_vals.append(LAMBDA_CONFIG.get(eid, 0.0))
        entropy_vals.append(avg_entropy)
        hit_rate_vals.append(avg_hit)
        labels_3d.append(eid)

    if not lambda_vals:
        print("[图10] 无足够三维数据")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 气泡大小：映射情感命中率
    sizes = [max(hr * 800, 40) for hr in hit_rate_vals]

    # 颜色：按实验分组
    exp_colors = []
    for eid in labels_3d:
        if eid in SECOND_ROUND_IDS:
            exp_colors.append(COLOR_ORANGE)
        else:
            exp_colors.append(COLOR_BLUE)

    scatter = ax.scatter(
        lambda_vals,
        entropy_vals,
        hit_rate_vals,
        s=sizes,
        c=exp_colors,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
        depthshade=True,
    )

    # 标注实验ID
    for lv, ev, hr, label in zip(
        lambda_vals, entropy_vals, hit_rate_vals, labels_3d
    ):
        ax.text(
            lv,
            ev,
            hr,
            f"  {label}",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("λ 强度", fontsize=11, labelpad=10)
    ax.set_ylabel("平均语义熵", fontsize=11, labelpad=10)
    ax.set_zlabel("平均情感命中率", fontsize=11, labelpad=10)  # type: ignore[arg-type]
    ax.set_title("λ-语义熵-情感命中率 三维关系", fontsize=14, fontweight="bold")

    # 图例
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_BLUE,
            markersize=8,
            label="第一轮 (Echo)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_ORANGE,
            markersize=8,
            label="第二轮 (筛选/思考)",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

    # 调整视角
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图10_三维关系图.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图10] 已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """主入口：加载数据 → 生成全部10张增强图表。"""
    print("=" * 60)
    print("开始生成增强可视化图表（10张）...")
    print("=" * 60)

    exp_data = load_all_experiments()
    print(f"成功加载 {len(exp_data)} 个实验的数据\n")

    plot_figure1(exp_data)
    plot_figure2(exp_data)
    plot_figure3(exp_data)
    plot_figure4(exp_data)
    plot_figure5(exp_data)
    plot_figure6(exp_data)
    plot_figure7(exp_data)
    plot_figure8(exp_data)
    plot_figure9(exp_data)
    plot_figure10(exp_data)

    print("\n" + "=" * 60)
    print(f"全部 10 张增强图表已生成到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
