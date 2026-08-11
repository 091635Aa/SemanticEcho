# -*- coding: utf-8 -*-
"""
早停机制 — 智能实验控制
========================
每个修复配置先跑小样本检查点：分数显著低于裸模型基线 → 立即中断并标记"淘汰"，
避免在注定失败的配置上浪费全量时间；通过则继续全量。

基线 = 最近一次双模式汇总的裸模型分数（四层需超越的目标）。
"""
import json
import os
from datetime import datetime

本目录 = os.path.dirname(os.path.abspath(__file__))
淘汰记录路径 = os.path.join(本目录, "data", "淘汰记录.json")

# 裸模型基线（最近双模式汇总，四层需超越的目标）
裸基线 = {
    "heartbench": 0.4027,
    "feel_heart": 0.4884,
    "llm_judge": 0.5900,
    "turingbench": 0.2333,
    "emocharacter": 0.8750,
}

# 检查点（到达该样本数时做一次早停决策）
检查点 = {
    "heartbench": 10,
    "feel_heart": 10,
    "llm_judge": 10,
    "turingbench": 10,
    "emocharacter": 3,
}

# 容差：期望"反超"的基准从严（0.05）；持平基准（feel_heart/turingbench）从宽，避免误杀
容差 = {
    "heartbench": 0.05,
    "feel_heart": 0.15,
    "llm_judge": 0.05,
    "turingbench": 0.20,
    "emocharacter": 0.05,
}


def 早停决策(基准名, 当前分数, 已完成样本数, 配置="默认", 自定义基线=None):
    """到达检查点时：当前分数 < 基线×(1-容差) → 中断（淘汰标记）；否则继续。

    返回 ("继续"|"中断", 消息)
    """
    基线 = 裸基线.get(基准名, 0.0) if 自定义基线 is None else 自定义基线
    cp = 检查点.get(基准名, 10)
    if 已完成样本数 < cp:
        return "继续", f"未到检查点 {cp}（当前 {已完成样本数}）"
    tol = 容差.get(基准名, 0.05)
    阈值 = 基线 * (1 - tol)
    if 当前分数 < 阈值:
        记录淘汰(基准名, 配置, 当前分数, 基线, 阈值)
        return "中断", (f"早停淘汰：{基准名} 配置[{配置}] 前{已完成样本数}条分数 "
                        f"{当前分数:.4f} < 基线×{1-tol}={阈值:.4f}")
    return "继续", (f"检查点通过：{基准名} 配置[{配置}] 分数 {当前分数:.4f} "
                    f"≥ 基线×{1-tol}={阈值:.4f}")


def 记录淘汰(基准名, 配置, 当前分数, 基线, 阈值):
    记录 = {
        "基准": 基准名, "配置": str(配置), "检查点分数": round(当前分数, 4),
        "裸基线": round(基线, 4), "淘汰阈值": round(阈值, 4),
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    已有 = []
    if os.path.exists(淘汰记录路径):
        try:
            已有 = json.load(open(淘汰记录路径, encoding="utf-8"))
        except Exception:
            pass
    已有.append(记录)
    os.makedirs(os.path.dirname(淘汰记录路径), exist_ok=True)
    with open(淘汰记录路径, "w", encoding="utf-8") as f:
        json.dump(已有, f, ensure_ascii=False, indent=2)
