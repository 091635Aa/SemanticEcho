# -*- coding: utf-8 -*-
"""
② HEART-BENCH（FEEL 备用）— 记忆驱动人格推理评测（本地适配版）
==============================================================
HEART-BENCH：给定角色原始情景记忆 → 推断人格 → 对情景做行为决策（MCQ）。
- 生成：目标模型 1.5B 读场景+选项，输出 decision_choice（A/B/C/D）+ 理由
- 共情评分：裁判 7B 评估回答的"共情合理性"（0-1）
- 一致性：同一题多次生成的选项稳定性
数据：repos/HEART-BENCH/benchmark/（mcq.json + scenarios.json），抽样 60 题。
"""
import json
import os
import re
import random
import time

本目录 = os.path.dirname(os.path.abspath(__file__))
数据根 = os.path.join(本目录, "repos", "HEART-BENCH", "benchmark")
日志路径 = os.path.join(本目录, "logs", "FEEL_HEART_log.txt")
结果路径 = os.path.join(本目录, "data", "feel_heart_results.json")
样本数 = 40
重复次数 = 3  # 一致性评估

import sys
sys.path.insert(0, 本目录)
import 公共模块 as cm

选项字母 = ["A", "B", "C", "D"]


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 加载数据():
    mcq = json.load(open(os.path.join(数据根, "mcq.json"), encoding="utf-8"))
    sc = json.load(open(os.path.join(数据根, "scenarios.json"), encoding="utf-8"))
    场景表 = {}
    for 阶段, 列表 in sc["scenarios"].items():
        for s in 列表:
            场景表[s["id"]] = s
    return mcq["questions"], 场景表


def 生成决策(题, 场景, 模式="裸", 轮次=0, λ覆盖=None, 思考链=False):
    """1.5B 生成行为决策（模式：裸|四层）"""
    选项文本 = "\n".join(f"{o['label']}. {o['content']}" for o in 题["options"])
    设定 = 场景.get("setting") or {}
    触发 = 场景.get("trigger_event") or {}
    消息 = [{"role": "user", "content": (
        f"You are a role-play simulator. You see the following situation.\n\n"
        f"## Current Situation\nScene: {场景.get('name','')}\n"
        f"Location: {设定.get('location','')} | Time: {设定.get('time','')}\n"
        f"Context: {场景.get('context_text','')}\n\n"
        f"## Trigger Event\nMessage: {触发.get('message_content','')}\n"
        f"Action required: {触发.get('action_required','')}\n\n"
        f"## Behavioural Decision Options\n{选项文本}\n\n"
        f"Think about what a real person would most likely do, then pick ONE option.\n"
        f"Output strictly JSON: {{\"final_decision\": \"your decision in 1-2 sentences\", \"decision_choice\": \"A or B or C or D\"}}"
    )}]
    from 生成器 import 生成器实例
    return 生成器实例.生成(模式, 消息, 种子=42, 轮次=轮次, max_new_tokens=128,
                            λ覆盖=λ覆盖, 思考链=思考链)


def 提取选项(文本):
    m = re.search(r'"decision_choice"\s*:\s*"?([A-D])', 文本)
    if m:
        return m.group(1)
    m2 = re.search(r"\b([A-D])\b", 文本)
    return m2.group(1) if m2 else None


def 裁判共情评分(题, 场景, 决策文本):
    """裁判评估该决策的共情合理性 0-1"""
    选项文本 = "\n".join(f"{o['label']}. {o['content']}" for o in 题["options"])
    设定 = 场景.get("setting") or {}
    触发 = 场景.get("trigger_event") or {}
    消息 = [{"role": "user", "content": (
        f"You are evaluating a role-play response for EMPATHY. Situation:\n"
        f"Scene: {场景.get('name','')} | Location: {设定.get('location','')}\n"
        f"Context: {场景.get('context_text','')}\n"
        f"Trigger: {触发.get('message_content','')}\n"
        f"Options: {选项文本}\n\n"
        f"Agent's response: {决策文本}\n\n"
        f"Rate the response's empathy (understanding of others' feelings and appropriate emotional response) "
        f"on a scale 0 to 1. Output strictly JSON: {{\"empathy_score\": 0.0-1.0, \"reason\": \"brief\"}}"
    )}]
    return cm.裁判生成(消息, max_new_tokens=150, temperature=0.2)


def 提取分数(文本, 键):
    m = re.search(rf'"{键}"\s*:\s*([0-9]*\.?[0-9]+)', 文本)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "四层", "全部"], default="全部")
    ap.add_argument("--早停", action="store_true", help="四层模式前 10 题决策+共情评分后按综合基线做早停决策")
    ap.add_argument("--λ", type=float, default=None, help="四层模式 λ 覆盖（任务自适应扫描）")
    ap.add_argument("--思考链", action="store_true", help="四层模式启用思考链（CoT）注入")
    args = ap.parse_args()
    模式列表 = ["裸", "四层"] if args.模式 == "全部" else [args.模式]

    if os.path.exists(日志路径):
        os.remove(日志路径)
    记录日志(f"=== HEART-BENCH (FEEL 备用) 评测开始（模式：{模式列表}，早停={args.早停}，λ={args.λ}，思考链={args.思考链}）===")
    题目, 场景表 = 加载数据()
    random.seed(42)
    样本 = random.sample(题目, min(样本数, len(题目)))
    记录日志(f"题目总数 {len(题目)}，抽样 {len(样本)}，每题重复 {重复次数} 次")

    from 生成器 import 生成器实例
    全部汇总 = {}
    for 模式 in 模式列表:
        记录日志(f"──── 模式 [{模式}] ────")
        记录 = []
        早停淘汰 = False
        t0 = time.time()
        for i, 题 in enumerate(样本):
            场景 = 场景表.get(题["scenario_id"], {})
            决策列表 = []
            for k in range(重复次数):
                文本 = 生成决策(题, 场景, 模式=模式, 轮次=i * 重复次数 + k,
                                λ覆盖=args.λ, 思考链=args.思考链)
                选项 = 提取选项(文本)
                决策列表.append({"轮次": k, "文本": 文本, "选项": 选项})
            # 一致性：选中最频繁选项的比例
            from collections import Counter
            cnt = Counter(d["选项"] for d in 决策列表 if d["选项"])
            主选项 = cnt.most_common(1)[0][0] if cnt else None
            一致性 = cnt[主选项] / 重复次数 if 主选项 else 0.0
            正确 = 1.0 if 主选项 == 题.get("correct_answer") else 0.0
            记录.append({
                "question_id": 题["question_id"],
                "决策列表": 决策列表,
                "主选项": 主选项,
                "正确答案": 题.get("correct_answer"),
                "一致性": round(一致性, 3),
                "正确": 正确,
            })
            记录日志(f"[决策 {i+1}/{len(样本)}] {题['question_id']} 主选项={主选项} 正确={正确} 一致性={一致性}")
            # 早停：四层前 10 题后，先补 10 条共情评分，再按综合基线决策
            if args.早停 and 模式 == "四层" and len(记录) == 10:
                cm.加载裁判模型()
                for r in 记录:
                    if "empathy_score" in r and r["empathy_score"] is not None:
                        continue
                    题k = 样本[记录.index(r)]
                    场景k = 场景表.get(题k["scenario_id"], {})
                    决策文本 = r["决策列表"][0]["文本"]
                    try:
                        评分文本 = 裁判共情评分(题k, 场景k, 决策文本)
                        r["empathy_score"] = 提取分数(评分文本, "empathy_score")
                    except Exception as e:
                        r["empathy_score"] = None
                        记录日志(f"[早停共情异常] {题k['question_id']}: {e}")
                from 早停 import 早停决策
                acc = sum(r["正确"] for r in 记录) / len(记录)
                cons = sum(r["一致性"] for r in 记录) / len(记录)
                emp = sum(r["empathy_score"] for r in 记录 if r["empathy_score"] is not None)
                emp = emp / len([r for r in 记录 if r["empathy_score"] is not None]) if any(r["empathy_score"] is not None for r in 记录) else 0.0
                综合 = (acc + cons + emp) / 3
                决策, 消息 = 早停决策("feel_heart", 综合, 10, 配置=f"R4 λ={args.λ} 思考链={args.思考链}")
                记录日志(f"[早停] {消息}")
                if 决策 == "中断":
                    早停淘汰 = True
                    记录日志("[早停] 已中断并标记淘汰")
                    break

        # 裁判共情评分（抽 20 条控制时间；早停已评的前 10 条跳过）
        cm.加载裁判模型()
        for i, r in enumerate(记录):
            if r.get("empathy_score") is not None:
                continue
            if i >= 20:
                r["empathy_score"] = None
                continue
            题 = 样本[i]
            场景 = 场景表.get(题["scenario_id"], {})
            决策文本 = r["决策列表"][0]["文本"]
            try:
                评分文本 = 裁判共情评分(题, 场景, 决策文本)
                r["empathy_score"] = 提取分数(评分文本, "empathy_score")
                记录日志(f"[共情 {i+1}/20] {题['question_id']} empathy={r['empathy_score']}")
            except Exception as e:
                r["empathy_score"] = None
                记录日志(f"[共情 {i+1}/20] 异常: {e}")
        cm.裁判槽.卸载()

        有效性 = [r for r in 记录 if r["主选项"]]
        共情分 = [r["empathy_score"] for r in 记录 if r["empathy_score"] is not None]
        汇总 = {
            "accuracy_score": round(sum(r["正确"] for r in 记录) / len(记录), 4),
            "consistency_score": round(sum(r["一致性"] for r in 记录) / len(记录), 4),
            "empathy_score": round(sum(共情分) / len(共情分), 4) if 共情分 else 0.0,
            "有效决策率": round(len(有效性) / len(记录), 4),
            "抽样数": len(记录),
            "总用时秒": round(time.time() - t0, 1),
            "_早停淘汰": 早停淘汰,
            "_λ": args.λ,
            "_思考链": args.思考链,
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
