# -*- coding: utf-8 -*-
"""生成 哭鼻子 情绪二维波形图（PNG）：音频波形 + 情绪阶段彩色背景 + 关键事件标记"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import librosa

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

wav = r"i:\Desktop\语义回响\实验数据\哭鼻子分析\整段_0839_1513.wav"
输出 = r"i:\Desktop\语义回响\实验数据\哭鼻子分析\哭鼻子情绪波形图.png"
起点秒 = 519  # 8:39
时长秒 = 394   # 到 15:13

# 情绪阶段（起秒, 止秒, 名称, 颜色）
阶段 = [
    (0, 17, "平静\n8:39-8:56", "#c8e6c9"),
    (17, 39, "微伤感\n8:56-9:18", "#bbdefb"),
    (39, 73, "含泪→明显\n9:19-9:52", "#ffe0b2"),
    (73, 168, "互动回压\n9:53-11:27", "#bbdefb"),
    (168, 182, "再起波折\n11:28-11:41", "#ffe0b2"),
    (182, 263, "哭腔持续\n11:42-13:02", "#ef9a9a"),
    (263, 300, "决堤临界\n13:03-13:39", "#ef5350"),
    (300, 357, "委屈自嘲\n13:39-14:36", "#e1bee7"),
    (357, 394, "收尾\n14:37-15:13", "#cfd8dc"),
]

# 关键事件（秒, 标签, 颜色线）
事件 = [
    (17, "首次吸鼻子\n8:56", "#1565c0"),
    (52, "抽泣高峰\n9:31", "#ef6c00"),
    (188, "哭腔成型\n11:47", "#c62828"),
    (263, "持续吸鼻子\n13:02", "#b71c1c"),
    (300, "自述哭鼻子\n13:39", "#6a1b9a"),
    (357, "\u201c你们都不安慰我\u201d\n14:36", "#6a1b9a"),
]

def 时刻(x):
    """段内秒 → 绝对 分:秒"""
    t = 起点秒 + x
    return f"{t//60}:{t%60:02d}"

y, sr = librosa.load(wav, sr=22050, mono=True)
t = np.arange(len(y)) / sr

fig, ax = plt.subplots(figsize=(20, 7))
fig.suptitle("哭鼻子片段情绪波形图（8:39-15:13）· 男猫女猫向前冲 2026-08-05", fontsize=16, y=0.98)

# 波形
ax.plot(t, y, color="#37474f", linewidth=0.4, alpha=0.85)
ax.set_xlim(0, 时长秒)
ax.set_ylim(-0.6, 0.6)
ax.set_ylabel("振幅")

# 情绪背景色块
for 起, 止, 名, 色 in 阶段:
    ax.axvspan(起, 止, color=色, alpha=0.5, zorder=0)
    if 止 - 起 >= 15:
        ax.text((起 + 止) / 2, 0.52, 名, ha="center", va="top", fontsize=10,
                color="#263238", fontweight="bold")

# 关键事件
for 秒, 标签, 色 in 事件:
    ax.axvline(秒, color=色, linestyle="--", linewidth=1.2, alpha=0.9, zorder=2)
    ax.text(秒, -0.55, 标签, ha="center", va="top", fontsize=9.5,
            color=色, fontweight="bold", rotation=0)

# 图例
图例 = [mpatches.Patch(color="#c8e6c9", label="平静"),
        mpatches.Patch(color="#bbdefb", label="微伤感"),
        mpatches.Patch(color="#ffe0b2", label="含泪/哭腔"),
        mpatches.Patch(color="#ef9a9a", label="哭腔明显"),
        mpatches.Patch(color="#ef5350", label="决堤临界"),
        mpatches.Patch(color="#e1bee7", label="委屈/自嘲"),
        mpatches.Patch(color="#cfd8dc", label="收尾")]
ax.legend(handles=图例, loc="upper right", fontsize=10, framealpha=0.9)

# X 轴绝对时间刻度
刻度 = list(range(0, 时长秒 + 1, 30))
ax.set_xticks(刻度)
ax.set_xticklabels([时刻(x) for x in 刻度], fontsize=10)
ax.set_xlabel("时间（绝对，段内秒数 = 距 8:39 的秒数）")
ax.grid(axis="x", linestyle=":", alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(输出, dpi=150, bbox_inches="tight")
print("已生成:", 输出)
