# -*- coding: utf-8 -*-
"""
③ LLM-as-Judge — AI 回复 vs 真人回复 盲评（本地自实现）
=====================================================
- 提示词集：样本_30条.json 的 user 话（真人对话）
- 候选 A：目标模型 1.5B 对 user 生成回复
- 候选 B：真人回复（girl 字段）
- 裁判 7B：盲评（不知道来源）判定"哪个更像真人写的"，并对每个回复打 1-5 分
指标：win_rate_against_human（AI 胜率）+ average_rating（归一化 rating/5）
"""
import json
import os
import re
import random
import time

本目录 = os.path.dirname(os.path.abspath(__file__))
样本路径 = os.path.join(本目录, "样本_30条.json")
日志路径 = os.path.join(本目录, "logs", "LLM_judge_log.txt")
结果路径 = os.path.join(本目录, "data", "llm_judge_results.json")

import sys
sys.path.insert(0, 本目录)
import 公共模块 as cm


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


裁判配对提示 = """你是一位严格的中文图灵测试裁判。用户说了一句话，下面有两个回复：回复A 和 回复B。

用户：{user}
回复A：{回复A}
回复B：{回复B}

其中一个回复是真人写的，另一个是 AI 生成的。
请判断：哪个回复更可能出自真人之手？

必须严格按以下 JSON 格式输出（不要输出其他内容）：
{{"更像真人": "A" 或 "B", "理由": "一句话理由"}}"""

裁判评分提示 = """你是一位严格的中文图灵测试裁判。用户说了一句话，下面是某个人的回复。

用户：{user}
回复：{回复}

请给这个回复的"像真人程度"打分：1 到 5 分（1=一眼就是 AI，5=完全像真人）。
必须严格按以下 JSON 格式输出：
{{"像真人程度": 1到5的整数, "理由": "一句话理由"}}"""


def 解析配对(文本):
    m = re.search(r'"更像真人"\s*[:：]\s*"([AB])"', 文本)
    if m:
        return m.group(1)
    if "回复A" in 文本 and "回复B" not in 文本.split("更像真人")[-1][:40]:
        return "A"
    if "回复B" in 文本 and "回复A" not in 文本.split("更像真人")[-1][:40]:
        return "B"
    return None


def 解析评分(文本):
    m = re.search(r'"像真人程度"\s*[:：]\s*([1-5])', 文本)
    if m:
        return int(m.group(1))
    m2 = re.search(r'([1-5])\s*分', 文本)
    return int(m2.group(1)) if m2 else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "四层", "全部"], default="全部")
    ap.add_argument("--早停", action="store_true", help="四层模式前 10 条配对后按 win_rate 基线做早停决策")
    ap.add_argument("--λ", type=float, default=None, help="四层模式 λ 覆盖（任务自适应扫描）")
    ap.add_argument("--模板", choices=["chat", "纯文本"], default="chat", help="四层模式生成模板")
    ap.add_argument("--长度", type=int, default=64, help="AI 回复 max_new_tokens")
    ap.add_argument("--身份", choices=["on", "off"], default="on", help="四层模式是否注入身份 system")
    ap.add_argument("--提示词", choices=["人类身份", "图灵测试"], default="人类身份", help="四层模式身份提示词版本")
    args = ap.parse_args()
    模式列表 = ["裸", "四层"] if args.模式 == "全部" else [args.模式]

    if os.path.exists(日志路径):
        os.remove(日志路径)
    记录日志(f"=== LLM-as-Judge 评测开始（模式：{模式列表}，早停={args.早停}，λ={args.λ}）===")
    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"]
    random.seed(42)
    random.shuffle(样本)
    记录日志(f"样本数: {len(样本)}")

    from 生成器 import 生成器实例
    全部汇总 = {}
    for 模式 in 模式列表:
        记录日志(f"──── 模式 [{模式}] ────")
        # 1. 目标模型生成回复
        AI回复列表 = []
        for i, r in enumerate(样本):
            消息 = [{"role": "user", "content": r["user"]}]
            ai回复 = 生成器实例.生成(模式, 消息, 种子=42, 轮次=i, max_new_tokens=args.长度, λ覆盖=args.λ, 模板=args.模板, 身份注入=(args.身份 == "on"), 提示词=args.提示词)
            AI回复列表.append(ai回复)
            记录日志(f"[AI生成 {i+1}/{len(样本)}] {r['user'][:20]} => {ai回复[:30]}")

        # 2. 裁判盲评（配对 + 评分）
        cm.加载裁判模型()
        配对记录 = []
        早停淘汰 = False
        for i, r in enumerate(样本):
            用户 = r["user"]
            真人 = r["girl"]
            ai回复 = AI回复列表[i]
            # R6 校准：AB 对调双份投票（AI 在 A 位 + AI 在 B 位各判一次），
            # 消除裁判位置偏差（离线分析发现裁判选 B 66.7%）
            消息A = [{"role": "user", "content": 裁判配对提示.format(user=用户, 回复A=ai回复, 回复B=真人)}]
            消息B = [{"role": "user", "content": 裁判配对提示.format(user=用户, 回复A=真人, 回复B=ai回复)}]
            输出A = cm.裁判生成(消息A, max_new_tokens=120, temperature=0.2)
            输出B = cm.裁判生成(消息B, max_new_tokens=120, temperature=0.2)
            选择A = 解析配对(输出A)  # AI 在 A 位：判 A 即 AI 胜
            选择B = 解析配对(输出B)  # AI 在 B 位：判 B 即 AI 胜
            ai胜A = 1.0 if 选择A == "A" else 0.0
            ai胜B = 1.0 if 选择B == "B" else 0.0
            ai胜 = 1.0 if (ai胜A or ai胜B) else 0.0  # OR：任一位置判 AI 即胜（对调双投）
            配对记录.append({
                "序号": r["序号"], "user": 用户, "ai回复": ai回复, "真人回复": 真人,
                "A位判": 选择A, "B位判": 选择B, "AI胜": ai胜,
                "原始A": 输出A[:120], "原始B": 输出B[:120],
            })
            记录日志(f"[配对 {i+1}/{len(样本)}] A位判{选择A} B位判{选择B} AI胜={ai胜}")
            # 早停：四层前 10 条配对后按 win_rate 基线（裸 0.4333）决策
            if args.早停 and 模式 == "四层" and len(配对记录) == 10:
                from 早停 import 早停决策
                当前胜率 = sum(x["AI胜"] for x in 配对记录) / len(配对记录)
                决策, 消息 = 早停决策("llm_judge", 当前胜率, 10, 配置=f"R6双投 λ={args.λ}",
                                    自定义基线=0.4333)
                记录日志(f"[早停] {消息}")
                if 决策 == "中断":
                    早停淘汰 = True
                    记录日志("[早停] 已中断并标记淘汰")
                    break

        # 评分（全部 30 条；早停淘汰则跳过）
        评分记录 = []
        if not 早停淘汰:
            for i, r in enumerate(样本):
                for 标签, 文本 in (("真人", r["girl"]), ("AI", AI回复列表[i])):
                    消息 = [{"role": "user", "content": 裁判评分提示.format(user=r["user"], 回复=文本)}]
                    裁判输出 = cm.裁判生成(消息, max_new_tokens=120, temperature=0.2)
                    分 = 解析评分(裁判输出)
                    评分记录.append({"序号": r["序号"], "来源": 标签, "评分": 分, "原始": 裁判输出[:150]})
                    记录日志(f"[评分 {i+1}.{标签}] {分} 分")
        cm.裁判槽.卸载()

        ai胜率 = sum(x["AI胜"] for x in 配对记录) / len(配对记录) if 配对记录 else 0.0
        ai评分 = [x["评分"] for x in 评分记录 if x["来源"] == "AI" and x["评分"]]
        真人评分 = [x["评分"] for x in 评分记录 if x["来源"] == "真人" and x["评分"]]
        汇总 = {
            "win_rate_against_human": round(ai胜率, 4),
            "average_rating": round(sum(ai评分) / len(ai评分) / 5, 4) if ai评分 else 0.0,
            "average_rating_raw": round(sum(ai评分) / len(ai评分), 2) if ai评分 else 0.0,
            "human_average_rating_raw": round(sum(真人评分) / len(真人评分), 2) if 真人评分 else 0.0,
            "样本数": len(配对记录),
            "_早停淘汰": 早停淘汰,
            "_λ": args.λ,
        }
        全部汇总[模式] = 汇总
        记录日志(f"[{模式}] {json.dumps(汇总, ensure_ascii=False)}")

    生成器实例.清理()
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"模式汇总": 全部汇总}, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {结果路径}")
    return 全部汇总 if len(全部汇总) > 1 else 全部汇总[模式列表[0]]


if __name__ == "__main__":
    main()
