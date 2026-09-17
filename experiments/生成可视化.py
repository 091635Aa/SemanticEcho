"""
生成可视化.py — 读取实验数据 JSON 文件，生成五张可视化图表和一份文本对比表。

图表列表:
  图1: 语义熵分布对比（箱线图）
  图2: 细腻度提升率柱状图（以 E1 为基线）
  图3: λ 与平均熵的关系曲线
  图4: 质心范数柱状图（E3 vs E4 回响池强度）
  图5: 输出文本样例对比表（.txt）
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ── 全局 matplotlib 中文支持 ───────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 路径 ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "实验数据")
OUTPUT_DIR = os.path.join(DATA_DIR, "可视化")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENT_FILES: dict[str, str] = {
    "E1": os.path.join(DATA_DIR, "E1.json"),
    "E2": os.path.join(DATA_DIR, "E2.json"),
    "E3": os.path.join(DATA_DIR, "E3.json"),
    "E4": os.path.join(DATA_DIR, "E4.json"),
    "E5": os.path.join(DATA_DIR, "E5.json"),
    "E6": os.path.join(DATA_DIR, "E6.json"),
}
SUMMARY_FILE = os.path.join(DATA_DIR, "实验结果汇总.json")

EXPERIMENT_LABELS: dict[str, str] = {
    "E1": "E1\nBaseline\n(top_p=0.9)",
    "E2": "E2\nBaseline\n(T=1.0)",
    "E3": "E3\nEcho\n(λ=0.5, γ=0.05)",
    "E4": "E4\nEcho\n(λ=1.0, γ=0.1)",
    "E5": "E5\nEcho\n(λ=2.0, γ=0.5)",
    "E6": "E6\nEcho\n(λ=1.0, γ=0.01)",
}

# λ 配置（图3 专用）
LAMBDA_CONFIG: dict[str, float] = {
    "E1": 0.0,
    "E3": 0.5,
    "E4": 1.0,
    "E5": 2.0,
}


# ── 数据加载 ───────────────────────────────────────────────────────────────


def load_json(filepath: str) -> dict[str, Any]:
    """加载 JSON 文件，文件不存在或解析失败时抛出异常。"""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件未找到: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def extract_experiment_data(
    filepath: str,
) -> tuple[list[float], list[dict[str, Any]]]:
    """从单个实验 JSON 中提取所有重复结果的平均熵列表，以及池统计列表。

    Parameters
    ----------
    filepath : str
        JSON 文件路径。

    Returns
    -------
    entropies : list[float]
        所有（提示词×重复）的平均熵值。
    pool_stats : list[dict]
        每条重复结果中的池统计（若存在）。
    """
    data = load_json(filepath)
    entropies: list[float] = []
    pool_stats: list[dict[str, Any]] = []

    for entry in data.get("数据", []):
        for repeat in entry.get("重复结果", []):
            avg_entropy = repeat.get("平均熵")
            if avg_entropy is not None:
                entropies.append(float(avg_entropy))
            pool = repeat.get("池统计")
            if pool is not None:
                enriched: dict[str, Any] = dict(pool)
                enriched["维度"] = entry.get("维度", "未知")
                enriched["提示词"] = entry.get("提示词", "未知")
                pool_stats.append(enriched)

    return entropies, pool_stats


def load_all_experiments() -> dict[str, dict[str, Any]]:
    """加载所有实验数据，返回 {实验ID: {entropies, pool_stats, config}}。"""
    result: dict[str, dict[str, Any]] = {}
    for exp_id, filepath in EXPERIMENT_FILES.items():
        try:
            entropies, pool_stats = extract_experiment_data(filepath)
            result[exp_id] = {
                "entropies": entropies,
                "pool_stats": pool_stats,
                "config": load_json(filepath).get("统计", {}),
            }
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"[警告] 无法加载 {exp_id}: {exc}")
    return result


def load_summary_stats() -> dict[str, Any]:
    """加载实验结果汇总 JSON。"""
    return load_json(SUMMARY_FILE)


# ── 图1: 语义熵分布对比（箱线图） ──────────────────────────────────────


def plot_figure1(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图1：语义熵分布箱线图。

    Parameters
    ----------
    exp_data : dict
        由 load_all_experiments() 返回的数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    exp_ids = ["E1", "E2", "E3", "E4", "E5", "E6"]
    all_entropies: list[list[float]] = []
    labels: list[str] = []

    for eid in exp_ids:
        entry = exp_data.get(eid)
        if entry is None or not entry["entropies"]:
            print(f"[图1] {eid} 无有效数据，跳过")
            continue
        all_entropies.append(entry["entropies"])
        labels.append(EXPERIMENT_LABELS[eid])

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        all_entropies,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=5),
    )

    # 配色
    colors = ["#4ECDC4", "#FFE66D", "#FF6B6B", "#95E1D3", "#F38181", "#AA96DA"]
    for patch, color in zip(bp["boxes"], colors[: len(all_entropies)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("不同实验配置的语义熵分布对比", fontsize=14, fontweight="bold")
    ax.set_xlabel("实验配置", fontsize=11)
    ax.set_ylabel("平均语义熵", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # 在均值点上方标注数值
    means = [np.mean(d) for d in all_entropies]
    for i, mean_val in enumerate(means):
        ax.annotate(
            f"{mean_val:.3f}",
            xy=(i + 1, mean_val),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="red",
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图1_语义熵箱线图.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图1] 已保存: {save_path}")
    return save_path


# ── 图2: 细腻度提升率柱状图 ──────────────────────────────────────────


def plot_figure2(
    exp_data: dict[str, dict[str, Any]],
) -> str:
    """生成图2：以 E1 为基线的细腻度提升率柱状图。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    baseline = exp_data.get("E1")
    if baseline is None or not baseline["entropies"]:
        raise ValueError("E1 基线数据缺失，无法计算提升率。")
    h_baseline = np.mean(baseline["entropies"])

    echo_experiments = ["E3", "E4", "E5", "E6"]
    exp_ids: list[str] = []
    rates: list[float] = []

    for eid in echo_experiments:
        entry = exp_data.get(eid)
        if entry is None or not entry["entropies"]:
            print(f"[图2] {eid} 数据不足，跳过")
            continue
        h_echo = np.mean(entry["entropies"])
        rate = (h_echo - h_baseline) / h_baseline * 100.0
        exp_ids.append(eid)
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(9, 6))
    x_pos = np.arange(len(exp_ids))
    colors_bar = ["#FF6B6B", "#4ECDC4", "#95E1D3", "#AA96DA"]
    bars = ax.bar(x_pos, rates, width=0.5, color=colors_bar, edgecolor="gray", alpha=0.85)

    # 标注数值
    for bar_item, rate_val in zip(bars, rates):
        y_pos = rate_val + (1.5 if rate_val >= 0 else -4.5)
        ax.text(
            bar_item.get_x() + bar_item.get_width() / 2.0,
            y_pos,
            f"{rate_val:+.2f}%",
            ha="center",
            va="bottom" if rate_val >= 0 else "top",
            fontsize=11,
            fontweight="bold",
            color="black",
        )

    # 零线
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([EXPERIMENT_LABELS[e].replace("\n", "\n") for e in exp_ids])
    ax.set_title("语义回响对不同 λ 配置的细腻度提升率", fontsize=14, fontweight="bold")
    ax.set_ylabel("细腻度提升率 (%)", fontsize=11)
    ax.set_xlabel("实验配置", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图2_细腻度提升率.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图2] 已保存: {save_path}")
    return save_path


# ── 图3: λ 与平均熵的关系曲线 ────────────────────────────────────────


def plot_figure3(exp_data: dict[str, dict[str, Any]]) -> str:
    """生成图3：λ 强度对语义熵的非单调影响。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str
        保存的文件路径。
    """
    ordered = ["E1", "E3", "E4", "E5"]
    lambda_vals: list[float] = []
    entropy_means: list[float] = []
    entropy_stds: list[float] = []

    for eid in ordered:
        entry = exp_data.get(eid)
        if entry is None or not entry["entropies"]:
            print(f"[图3] {eid} 数据不足，跳过")
            continue
        lambda_vals.append(LAMBDA_CONFIG[eid])
        entropy_means.append(np.mean(entry["entropies"]))
        entropy_stds.append(np.std(entry["entropies"]))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # 折线
    ax.plot(
        lambda_vals,
        entropy_means,
        marker="o",
        markersize=10,
        linewidth=2.5,
        color="#FF6B6B",
        markerfacecolor="#4ECDC4",
        markeredgecolor="black",
        markeredgewidth=1.5,
        linestyle="-",
        label="平均语义熵",
    )
    # 误差带
    ax.fill_between(
        lambda_vals,
        [m - s for m, s in zip(entropy_means, entropy_stds)],
        [m + s for m, s in zip(entropy_means, entropy_stds)],
        alpha=0.15,
        color="#FF6B6B",
        label="±1 标准差",
    )

    # 标注数值
    for lv, mv, sv in zip(lambda_vals, entropy_means, entropy_stds):
        ax.annotate(
            f"{mv:.3f}\n±{sv:.3f}",
            (lv, mv),
            xytext=(0, -25),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="darkred",
        )

    ax.set_xlabel("λ 强度", fontsize=12)
    ax.set_ylabel("平均语义熵", fontsize=12)
    ax.set_title("λ 强度对语义熵的非单调影响", fontsize=14, fontweight="bold")
    ax.set_xticks(lambda_vals)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图3_λ与熵的关系.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图3] 已保存: {save_path}")
    return save_path


# ── 图4: 质心范数柱状图（E3 vs E4） ──────────────────────────────────


def plot_figure4(exp_data: dict[str, dict[str, Any]]) -> str | None:
    """生成图4：不同 λ 配置下回响池质心强度对比。

    对 E3 (λ=0.5) 和 E4 (λ=1.0) 的每条重复结果，汇总其 池统计.质心范数，
    按 prompt 维度绘制分组柱状图。

    Parameters
    ----------
    exp_data : dict
        实验数据。

    Returns
    -------
    str | None
        保存的文件路径；若数据不足则返回 None。
    """
    target_exps = {"E3": "λ=0.5", "E4": "λ=1.0"}
    # 提取每个实验的 质心范数（按 prompt + repeat 分组）
    # 按维度收集
    dim_data: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for eid, label_prefix in target_exps.items():
        entry = exp_data.get(eid)
        if entry is None:
            print(f"[图4] {eid} 无数据，跳过")
            continue
        for ps in entry["pool_stats"]:
            norm = ps.get("质心范数")
            if norm is not None:
                dim = ps.get("维度", "未知")
                dim_data[dim][eid].append(float(norm))

    if not dim_data:
        print("[图4] 无质心范数数据，跳过")
        return None

    # 按维度排序
    dims_sorted = sorted(dim_data.keys())
    e3_means: list[float] = []
    e4_means: list[float] = []
    e3_stds: list[float] = []
    e4_stds: list[float] = []

    for d in dims_sorted:
        e3_vals = dim_data[d].get("E3", [])
        e4_vals = dim_data[d].get("E4", [])
        e3_means.append(np.mean(e3_vals) if e3_vals else 0.0)
        e4_means.append(np.mean(e4_vals) if e4_vals else 0.0)
        e3_stds.append(np.std(e3_vals) if e3_vals else 0.0)
        e4_stds.append(np.std(e4_vals) if e4_vals else 0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dims_sorted))
    width = 0.3

    bars3 = ax.bar(
        x - width / 2,
        e3_means,
        width,
        yerr=e3_stds,
        capsize=4,
        label="E3 (λ=0.5)",
        color="#FF6B6B",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    bars4 = ax.bar(
        x + width / 2,
        e4_means,
        width,
        yerr=e4_stds,
        capsize=4,
        label="E4 (λ=1.0)",
        color="#4ECDC4",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    # 数值标注
    for bar_item in bars3:
        h = bar_item.get_height()
        if h > 0:
            ax.text(
                bar_item.get_x() + bar_item.get_width() / 2.0,
                h,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    for bar_item in bars4:
        h = bar_item.get_height()
        if h > 0:
            ax.text(
                bar_item.get_x() + bar_item.get_width() / 2.0,
                h,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(dims_sorted, rotation=30, ha="right")
    ax.set_title("不同 λ 配置下回响池质心强度", fontsize=14, fontweight="bold")
    ax.set_ylabel("质心范数", fontsize=11)
    ax.set_xlabel("情感维度", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "图4_质心范数.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图4] 已保存: {save_path}")
    return save_path


# ── 图5: 输出文本对比表 ─────────────────────────────────────────────


def plot_figure5() -> str:
    """生成图5：E1 和 E4 各维度代表性输出文本对比。

    对每个情感维度，选取第一个重复结果（重复次数=0）的文本，
    截取前 100 字进行对比。

    Returns
    -------
    str
        保存的 .txt 文件路径。
    """
    e1_path = EXPERIMENT_FILES["E1"]
    e4_path = EXPERIMENT_FILES["E4"]

    e1_data = load_json(e1_path)
    e4_data = load_json(e4_path)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("图5: E1 (Baseline) 与 E4 (Echo λ=1.0) 输出文本对比")
    lines.append("=" * 80)
    lines.append("")

    def _get_first_text(data: dict[str, Any]) -> dict[str, str]:
        """从实验数据中提取 {维度: 首个重复文本} 的映射。"""
        result: dict[str, str] = {}
        for entry in data.get("数据", []):
            dim = entry.get("维度", "未知")
            repeats = entry.get("重复结果", [])
            if repeats:
                text = repeats[0].get("文本", "")
                result[dim] = text
        return result

    e1_texts = _get_first_text(e1_data)
    e4_texts = _get_first_text(e4_data)

    all_dims = sorted(set(e1_texts.keys()) | set(e4_texts.keys()))

    for dim in all_dims:
        lines.append(f"{'─' * 80}")
        lines.append(f"▶ 维度: {dim}")
        lines.append(f"{'─' * 80}")
        lines.append("")

        txt1 = e1_texts.get(dim, "[无数据]")
        txt2 = e4_texts.get(dim, "[无数据]")

        lines.append("【E1 - Baseline (top_p=0.9)】")
        lines.append(f"  {txt1[:100]}")
        lines.append("")
        lines.append("【E4 - Echo (λ=1.0, γ=0.1)】")
        lines.append(f"  {txt2[:100]}")
        lines.append("")
        lines.append("")

    lines.append("=" * 80)

    save_path = os.path.join(OUTPUT_DIR, "图5_输出对比.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[图5] 已保存: {save_path}")
    return save_path


# ── 主函数 ─────────────────────────────────────────────────────────────────


def main() -> None:
    """主入口：加载数据 → 生成全部五张图表。"""
    print("=" * 60)
    print("开始生成可视化图表...")
    print("=" * 60)

    # 加载数据
    exp_data = load_all_experiments()
    print(f"成功加载 {len(exp_data)} 个实验的数据\n")

    # 图1: 箱线图
    plot_figure1(exp_data)

    # 图2: 提升率柱状图
    plot_figure2(exp_data)

    # 图3: λ-熵关系曲线
    plot_figure3(exp_data)

    # 图4: 质心范数
    plot_figure4(exp_data)

    # 图5: 文本对比
    plot_figure5()

    print("\n" + "=" * 60)
    print(f"全部图表已生成到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
