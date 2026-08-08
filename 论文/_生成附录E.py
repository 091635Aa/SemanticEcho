# -*- coding: utf-8 -*-
"""从实验 JSON 提取逐条明细，生成《附录E_实验逐条明细与生成样例.md》"""
import json, glob, os

输出路径 = r"i:\Desktop\语义回响\论文\附录E_实验逐条明细与生成样例.md"
数据目录 = r"i:\Desktop\语义回响\实验数据\多模型对照"

lines = []
lines.append("# 附录 E · 实验逐条明细与生成样例")
lines.append("")
lines.append("> 本附录从原始 JSON 输出逐条提取每一轮、每一条提示词的指标与生成文本预览，确保实验内容可逐条复核。文本预览截断至 40 字符。")
lines.append("")

def 短(t, n=40):
    t = (t or "").replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")

def 处理配置(f, 标题):
    try:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:
        return
    汇总 = d.get("汇总", {})  # 轮明细在 "汇总" 键（"汇总_全部模式" 为精简版）
    模式s = [m for m in ("裸", "回响") if m in 汇总]
    if not 模式s:
        return
    lines.append(f"### {标题}")
    lines.append("")
    lines.append(f"- 模型：{d.get('模型')}｜量化：{d.get('量化')}｜runs：{d.get('runs')}｜hidden_dim：{d.get('hidden_dim')}｜推荐参数：{d.get('推荐参数', {})}")
    lines.append("")
    for 模式 in 模式s:
        lines.append(f"**{模式} 模式**（生成文本预览，截断 40 字符）")
        lines.append("")
        lines.append("| run | 维度 | 熵 | 重复率 | 命中 | 生成文本预览 |")
        lines.append("|---|---|---|---|---|---|")
        轮明细 = 汇总[模式].get("轮明细", [])
        for 轮 in 轮明细:
            for 条 in 轮.get("每条", []):
                lines.append(
                    f"| {轮.get('run')} | {条.get('维度','')} "
                    f"| {条.get('平均熵',0):.3f} | {条.get('重复率',0):.3f} "
                    f"| {条.get('情感命中率',0):.3f} | {短(条.get('文本',''))} |")
        lines.append("")

# 1. 多模型对照 19 配置（每个配置取最新文件）
配置文件 = {}
for f in glob.glob(os.path.join(数据目录, "*_全部_*.json")):
    base = os.path.basename(f)
    key = base.split("_全部_")[0]  # 模型_量化
    配置文件[key] = f  # glob 排序取最后一个（最新）
lines.append("## E.1 多模型对照逐条明细（19 配置）")
lines.append("")
for i, key in enumerate(sorted(配置文件), 1):
    处理配置(配置文件[key], f"E.1.{i} {key}")

# 2. Qwen3 通用注入 6 配置
q3 = sorted(glob.glob(os.path.join(数据目录, "Qwen3通用注入", "*_全部_*.json")))
lines.append("## E.2 Qwen3 通用注入重测逐条明细（6 配置）")
lines.append("")
for f in q3:
    base = os.path.basename(f).replace("_全部_", " ")
    处理配置(f, f"E.2 {base.split('.json')[0]}")

with open(输出路径, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("已生成:", 输出路径, "| 行数:", len(lines))
