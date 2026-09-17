# -*- coding: utf-8 -*-
"""生成 哭鼻子情绪波形图 v2：波形 + 情绪色带 + 歌曲分段条 + 关键歌词标注"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import librosa

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

wav = r"i:\Desktop\语义回响\实验数据\哭鼻子分析\整段_0839_1513.wav"
输出 = r"i:\Desktop\语义回响\实验数据\哭鼻子分析\哭鼻子情绪波形图_含歌词.png"
起点秒 = 519
时长秒 = 394

# 情绪阶段（秒）
阶段 = [
    (0, 17, "平静", "#c8e6c9"),
    (17, 39, "微伤感", "#bbdefb"),
    (39, 73, "含泪→明显", "#ffe0b2"),
    (73, 168, "互动回压", "#bbdefb"),
    (168, 182, "再起波折", "#ffe0b2"),
    (182, 263, "哭腔持续", "#ef9a9a"),
    (263, 300, "决堤临界", "#ef5350"),
    (300, 357, "委屈自嘲", "#e1bee7"),
    (357, 394, "收尾", "#cfd8dc"),
]

# 歌曲分段（秒, 歌名, 颜色）——基于 txt 时间轴
歌曲 = [
    (0, 41, "《幻听》许嵩", "#80deea"),
    (41, 76, "互动·点歌", "#b0bec5"),
    (76, 300, "《素颜》许嵩&何曼婷", "#ffab91"),
    (300, 394, "下播互动", "#b0bec5"),
]

# 关键歌词标注（秒, 歌词, 颜色）
歌词标注 = [
    (8, "「在远方的时候\n又想你到泪流」", "#006064"),
    (33, "「夜色多温柔\n你有多爱我」", "#00838f"),
    (62, "「如今一个人听歌\n总是会觉得失落」", "#00838f"),
    (110, "「又是一个安静的晚上\n一个人窝在摇椅里乘凉」", "#bf360c"),
    (180, "「如果再看你一眼\n是否还会有感觉」", "#bf360c"),
    (195, "「最真实的喜怒哀乐\n全都埋葬在昨天」", "#d84315"),
    (220, "「我怀念 别怀念\n怀念也回不到从前」", "#d84315"),
    (268, "「那些流逝了的\n就永远不会复现」", "#b71c1c"),
    (300, "「你们有听到刚刚我在\n唱歌的时候哭鼻子了吗」", "#6a1b9a"),
]

# 情绪事件标记（保留）
事件 = [
    (17, "首次吸鼻子 8:56", "#1565c0"),
    (52, "抽泣高峰 9:31", "#ef6c00"),
    (188, "哭腔成型 11:47", "#c62828"),
    (263, "持续吸鼻子 13:02", "#b71c1c"),
    (300, "自述哭鼻子 13:39", "#6a1b9a"),
    (357, "\u201c你们都不安慰我\u201d 14:36", "#6a1b9a"),
]

y, sr = librosa.load(wav, sr=22050, mono=True)
t = np.arange(len(y)) / sr

fig, ax = plt.subplots(figsize=(24, 10))
fig.suptitle("哭鼻子片段情绪波形图·含歌曲与歌词（8:39-15:13）| 男猫女猫向前冲 2026-08-05", fontsize=17, y=0.985)

# 波形
ax.plot(t, y, color="#37474f", linewidth=0.4, alpha=0.85)
ax.set_xlim(0, 时长秒)
ax.set_ylim(-1.05, 0.85)
ax.set_ylabel("振幅")

# 情绪背景色
for 起, 止, 名, 色 in 阶段:
    ax.axvspan(起, 止, color=色, alpha=0.45, zorder=0)

# 歌曲分段条（y 顶部 0.55-0.85）
for 起, 止, 名, 色 in 歌曲:
    ax.axvspan(起, 止, ymin=0.66, ymax=0.98, color=色, alpha=0.85, zorder=1)
    if 止 - 起 >= 30:
        ax.text((起 + 止) / 2, 0.72, 名, ha="center", va="center", fontsize=11,
                color="#263238", fontweight="bold", zorder=3)

# 情绪阶段标签（y 中部 0.05-0.18）
for 起, 止, 名, 色 in 阶段:
    if 止 - 起 >= 20:
        ax.text((起 + 止) / 2, 0.12, 名, ha="center", va="center", fontsize=10,
                color="#455a64", fontweight="bold", zorder=3)

# 歌词标注（y 下方 -0.15~-0.55）
for 秒, 文本, 色 in 歌词标注:
    ax.axvline(秒, color=色, linestyle=":", linewidth=0.9, alpha=0.5, zorder=2)
    ax.text(秒, -0.45, 文本, ha="center", va="top", fontsize=8.5,
            color=色, fontweight="bold", zorder=3)

# 情绪事件标记
for 秒, 标签, 色 in 事件:
    ax.axvline(秒, color=色, linestyle="--", linewidth=1.3, alpha=0.9, zorder=4)
    ax.text(秒, 0.48, 标签, ha="center", va="bottom", fontsize=9.5,
            color=色, fontweight="bold", zorder=4)

# 图例
图例 = [mpatches.Patch(color="#c8e6c9", label="平静"),
        mpatches.Patch(color="#bbdefb", label="微伤感"),
        mpatches.Patch(color="#ffe0b2", label="含泪/哭腔"),
        mpatches.Patch(color="#ef9a9a", label="哭腔明显"),
        mpatches.Patch(color="#ef5350", label="决堤临界"),
        mpatches.Patch(color="#e1bee7", label="委屈/自嘲"),
        mpatches.Patch(color="#80deea", label="歌曲·《幻听》"),
        mpatches.Patch(color="#ffab91", label="歌曲·《素颜》"),
        mpatches.Patch(color="#b0bec5", label="互动/说话")]
ax.legend(handles=图例, loc="upper left", fontsize=9.5, framealpha=0.9, ncol=3)

刻度 = list(range(0, 时长秒 + 1, 30))
ax.set_xticks(刻度)
ax.set_xticklabels([f"{(起点秒 + x)//60}:{(起点秒 + x)%60:02d}" for x in 刻度], fontsize=10)
ax.set_xlabel("时间（绝对时间，X轴数值为距 8:39 的秒数）")
ax.grid(axis="x", linestyle=":", alpha=0.35)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(输出, dpi=150, bbox_inches="tight")
print("已生成:", 输出)
