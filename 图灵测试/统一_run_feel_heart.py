# -*- coding: utf-8 -*-
"""
② HEART-BENCH（FEEL 备用）— 记忆驱动人格推理评测（本地适配版）
==============================================================
HEART-BENCH：给定角色原始情景记忆 → 推断人格 → 对情景做行为决策（MCQ）。
- 生成：目标模型 1.5B 读场景+选项，输出 decision_choice（A/B/C/D）+ 理由
- 共情评分：裁判 7B 评估回答的"共情合理性"（0-1）
- 一致性：同一题多次生成的选项稳定性
数据：repos/HEART-BENCH/benchmark/（mcq.json + scenarios.json），抽样 60 题。

两阶段架构（生成 1.5B 与裁判 4bit 7B 分离，避免内存不足崩溃）：
  python 统一_run_feel_heart.py --仅生成 缓存.json
  python 统一_run_feel_heart.py --仅裁判 缓存.json
"""
import json
import os
import re
import random
import time

本目录 = os.path.dirname(os.path.abspath(__file__))
数据根 = os.path.join(本目录, "repos", "HEART-BENCH", "benchmark")
日志路径 = "i:\\Desktop\\\u8bed\u4e49\u56de\u54cd\\\u56fe\u7075\u6d4b\u8bd5\\\u7edf\u4e00\u57fa\u51c6\\FEEL_HEART_log.txt"
结果路径 = "i:\\Desktop\\\u8bed\u4e49\u56de\u54cd\\\u56fe\u7075\u6d4b\u8bd5\\\u7edf\u4e00\u57fa\u51c6\\feel_heart_results_2026.json"
样本数 = 40
重复次数 = 3  # 一致性评估

import sys
sys.path.insert(0, 本目录)
import 公共模块 as cm

# ===== 统一 7 模式补丁（种子 2026） =====
七模式列表 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合"]
统一目录 = os.path.join(本目录, "统一基准")
os.makedirs(统一目录, exist_ok=True)
# ===== /统一 7 模式补丁 =====


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


def 生成决策(题, 场景, 模式="裸", 轮次=0, λ覆盖=None, 思考链=False, 种子基数=42):
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
    from 统一生成器 import 生成器实例
    return 生成器实例.生成(模式, 消息, 种子=种子基数, 轮次=轮次, max_new_tokens=128,
                            λ覆盖=λ覆盖, 思考链=思考链)


def 提取选项(文本):
    m = re.search(r'"decision_choice"\s*:\s*"?([A-D])', 文本)
    if m:
        return m.group(1)
    m2 = re.search(r"\b([A-D])\b", 文本)
    return m2.group(1) if m2 else None


def 构造共情上下文(题, 场景, 决策文本):
    选项文本 = "\n".join(f"{o['label']}. {o['content']}" for o in 题["options"])
    设定 = 场景.get("setting") or {}
    触发 = 场景.get("trigger_event") or {}
    return (
        f"Scene: {场景.get('name','')} | Location: {设定.get('location','')}\n"
        f"Context: {场景.get('context_text','')}\n"
        f"Trigger: {触发.get('message_content','')}\n"
        f"Options: {选项文本}\n\n"
        f"Agent's response: {决策文本}")


def 裁判共情评分(题, 场景, 决策文本):
    """裁判评估该决策的共情合理性 0-1"""
    context = 构造共情上下文(题, 场景, 决策文本)
    消息 = [{"role": "user", "content": (
        f"You are evaluating a role-play response for EMPATHY. Situation:\n{context}\n\n"
        f"Rate the response's empathy (understanding of others' feelings and appropriate emotional response) "
        f"on a scale 0 to 1. Output strictly JSON: {{\"empathy_score\": 0.0-1.0, \"reason\": \"brief\"}}"
    )}]
    return cm.裁判生成(消息, max_new_tokens=150, temperature=0.2)


def 提取分数(文本, 键):
    m = re.search(rf'"{键}"\s*:\s*([0-9]*\.?[0-9]+)', 文本)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


# ============================================================
# 阶段1：生成决策缓存（1.5B，进程随后退出）
# ============================================================
def 生成决策缓存(模式列表, runs, seed_base, 缓存路径, λ=None, 思考链=False):
    from collections import Counter
    from 统一生成器 import 生成器实例
    random.seed(seed_base)
    题目, 场景表 = 加载数据()
    样本 = random.sample(题目, min(样本数, len(题目)))
    缓存 = {"seed_base": seed_base, "runs": runs, "模式记录": {}}
    for 模式 in 模式列表:
        记录日志(f"──── 生成 [{模式}] ────")
        for run_idx in range(runs):
            seed_offset = seed_base + run_idx * 100
            记录日志(f"  [run {run_idx+1}/{runs}] seed_offset={seed_offset}")
            记录 = []
            for i, 题 in enumerate(样本):
                场景 = 场景表.get(题["scenario_id"], {})
                决策列表 = []
                for k in range(重复次数):
                    文本 = 生成决策(题, 场景, 模式=模式,
                                    轮次=run_idx * 1000 + i * 重复次数 + k,
                                    λ覆盖=λ, 思考链=思考链,
                                    种子基数=seed_offset)
                    选项 = 提取选项(文本)
                    决策列表.append({"轮次": k, "文本": 文本, "选项": 选项})
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
            缓存["模式记录"].setdefault(模式, {})[str(run_idx)] = 记录
        生成器实例.清理()
    with open(缓存路径, "w", encoding="utf-8") as f:
        json.dump(缓存, f, ensure_ascii=False, indent=2)
    记录日志(f"生成缓存已保存 -> {缓存路径}")


# ============================================================
# 阶段2：独立裁判（干净进程，子进程加载 4bit 7B）
# ============================================================
def 裁判汇总(模式列表, runs, 缓存):
    import subprocess
    random.seed(缓存["seed_base"])
    题目, 场景表 = 加载数据()
    样本 = random.sample(题目, min(样本数, len(题目)))
    裁判子进程 = os.path.join(本目录, "裁判子进程.py")
    全部汇总 = {}
    多次运行明细 = {}
    for 模式 in 模式列表:
        记录日志(f"──── 裁判 [{模式}] ────")
        多次运行明细[模式] = []
        for run_idx in range(runs):
            记录 = 缓存["模式记录"][模式][str(run_idx)]
            记录日志(f"  [run {run_idx+1}/{runs}]")
            # 共情评分（子进程批处理，前 20 条）
            共情请求 = []
            共情索引 = []
            for i, r in enumerate(记录):
                if i >= 20:
                    break
                题 = 样本[i]
                场景 = 场景表.get(题["scenario_id"], {})
                决策文本 = r["决策列表"][0]["文本"]
                共情请求.append({"context": 构造共情上下文(题, 场景, 决策文本)})
                共情索引.append(i)
            if 共情请求:
                任务路径 = os.path.join(统一目录, f"_fh_{模式.replace(' ','')}_emp.json")
                输出路径 = os.path.join(统一目录, f"_fh_{模式.replace(' ','')}_emp_out.json")
                with open(任务路径, "w", encoding="utf-8") as f:
                    json.dump({"裁判": "feel", "请求": 共情请求}, f, ensure_ascii=False)
                subprocess.run([sys.executable, 裁判子进程, 任务路径, 输出路径], check=True)
                with open(输出路径, encoding="utf-8") as f:
                    共情结果 = json.load(f)
                for idx, cr in zip(共情索引, 共情结果):
                    记录[idx]["empathy_score"] = cr.get("共情合理性")
                    记录日志(f"[共情 {idx+1}/20] {样本[idx]['question_id']} empathy={记录[idx]['empathy_score']}")

            有效性 = [r for r in 记录 if r["主选项"]]
            共情分 = [r.get("empathy_score") for r in 记录 if r.get("empathy_score") is not None]
            run_summary = {
                "run_idx": run_idx,
                "accuracy_score": round(sum(r["正确"] for r in 记录) / len(记录), 4),
                "consistency_score": round(sum(r["一致性"] for r in 记录) / len(记录), 4),
                "empathy_score": round(sum(共情分) / len(共情分), 4) if 共情分 else 0.0,
                "有效决策率": round(len(有效性) / len(记录), 4),
                "抽样数": len(记录),
            }
            多次运行明细[模式].append({"run_summary": run_summary, "记录": 记录})
            记录日志(f"[run {run_idx+1}] {json.dumps(run_summary, ensure_ascii=False)}")

        acc_list = [d["run_summary"]["accuracy_score"] for d in 多次运行明细[模式]]
        cons_list = [d["run_summary"]["consistency_score"] for d in 多次运行明细[模式]]
        emp_list = [d["run_summary"]["empathy_score"] for d in 多次运行明细[模式]]
        有效_list = [d["run_summary"]["有效决策率"] for d in 多次运行明细[模式]]
        汇总 = {
            "accuracy_score": round(sum(acc_list) / len(acc_list), 4),
            "accuracy_std": round((sum((x - sum(acc_list)/len(acc_list))**2 for x in acc_list) / len(acc_list)) ** 0.5, 4) if len(acc_list) > 1 else 0.0,
            "consistency_score": round(sum(cons_list) / len(cons_list), 4),
            "consistency_std": round((sum((x - sum(cons_list)/len(cons_list))**2 for x in cons_list) / len(cons_list)) ** 0.5, 4) if len(cons_list) > 1 else 0.0,
            "empathy_score": round(sum(emp_list) / len(emp_list), 4),
            "empathy_std": round((sum((x - sum(emp_list)/len(emp_list))**2 for x in emp_list) / len(emp_list)) ** 0.5, 4) if len(emp_list) > 1 else 0.0,
            "有效决策率": round(sum(有效_list) / len(有效_list), 4),
            "有效决策率_std": round((sum((x - sum(有效_list)/len(有效_list))**2 for x in 有效_list) / len(有效_list)) ** 0.5, 4) if len(有效_list) > 1 else 0.0,
            "抽样数": len(样本),
            "_runs": runs,
            "_多次运行明细": [d["run_summary"] for d in 多次运行明细[模式]],
        }
        全部汇总[模式] = 汇总
        记录日志(f"[{模式}] {json.dumps(汇总, ensure_ascii=False)}")

    # 增量合并保存
    已有汇总 = {}
    if os.path.exists(结果路径):
        try:
            with open(结果路径, encoding="utf-8") as f:
                已有汇总 = json.load(f).get("模式汇总", {})
        except Exception:
            pass
    已有汇总.update(全部汇总)
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"模式汇总": 已有汇总, "_runs": runs, "_多次运行明细": 多次运行明细}, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {结果路径}")
    return 全部汇总


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", nargs="+", choices=["全部", "裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合"], default=["全部"])
    ap.add_argument("--早停", action="store_true", help="v2 已改为两阶段，早停仅作兼容保留")
    ap.add_argument("--λ", type=float, default=None, help="四层模式 λ 覆盖（任务自适应扫描）")
    ap.add_argument("--思考链", action="store_true", help="四层模式启用思考链（CoT）注入")
    ap.add_argument("--runs", type=int, default=1, help="多次测试轮数（>=1）")
    ap.add_argument("--seed_base", type=int, default=42, help="随机种子基数，每次 run 递增")
    ap.add_argument("--仅生成", type=str, default=None, help="只生成决策缓存到指定 JSON 后退出")
    ap.add_argument("--仅裁判", type=str, default=None, help="只裁判（读缓存 JSON），进程干净加载")
    args = ap.parse_args()
    模式列表 = 七模式列表 if "全部" in args.模式 else args.模式
    runs = max(1, args.runs)

    if os.path.exists(日志路径):
        os.remove(日志路径)
    记录日志(f"=== HEART-BENCH (FEEL 备用) 评测开始（模式：{模式列表}，runs={runs}）===")
    题目, 场景表 = 加载数据()
    random.seed(args.seed_base)
    样本 = random.sample(题目, min(样本数, len(题目)))
    记录日志(f"题目总数 {len(题目)}，抽样 {len(样本)}，每题重复 {重复次数} 次")

    默认缓存 = os.path.join(统一目录, "feel_heart_cache_2026.json")

    if args.仅生成:
        生成决策缓存(模式列表, runs, args.seed_base, args.仅生成, λ=args.λ, 思考链=args.思考链)
        return
    if args.仅裁判:
        with open(args.仅裁判, encoding="utf-8") as f:
            缓存 = json.load(f)
        裁判汇总(模式列表, runs, 缓存)
        return

    print("请分两步执行（先 --仅生成 再 --仅裁判），以彻底释放 1.5B 内存：")
    print(f"  python 统一_run_feel_heart.py --仅生成 {默认缓存}")
    print(f"  python 统一_run_feel_heart.py --仅裁判 {默认缓存}")


if __name__ == "__main__":
    main()
