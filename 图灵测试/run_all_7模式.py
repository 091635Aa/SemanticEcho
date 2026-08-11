# -*- coding: utf-8 -*-
"""
全流程 7 模式统一汇总（2026 种子）
==================================
整合 4 类结果：
  1. 性能测试（P1_5统一_性能_2026.json）
  2. 情感理解（P1_5统一_情感理解_2026.json）— 大厂式：MME-Emotion/CAREBench/AttuneBench
  3. 图灵 5 基准：heartbench / feel_heart / llm_judge / turingbench / emocharacter
输出：统一基准\全流程_汇总_2026.json + 全流程_报告_2026.md
"""
import json
import os
from datetime import datetime

本目录 = os.path.dirname(os.path.abspath(__file__))
统一目录 = os.path.join(本目录, "统一基准")
输出目录 = os.path.join(本目录, "results")
os.makedirs(输出目录, exist_ok=True)

七模式 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合", "P6_情感导演"]

性能路径 = r"c:\Users\Administrator\Documents\KV 情感共振解码\评测结果\P1_5统一_性能_2026.json"
情感理解路径 = r"c:\Users\Administrator\Documents\KV 情感共振解码\评测结果\P1_5统一_情感理解_2026.json"
基准文件 = {
    "heartbench": os.path.join(统一目录, "heartbench_results_2026.json"),
    "feel_heart": os.path.join(统一目录, "feel_heart_results_2026.json"),
    "llm_judge": os.path.join(统一目录, "llm_judge_results_2026.json"),
    "turingbench": os.path.join(统一目录, "turingbench_results_2026.json"),
    "emocharacter": os.path.join(统一目录, "emocharacter_results_2026.json"),
}


def 加载(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[警告] 无法加载 {path}: {e}")
        return {}


def 单基准分(键, d):
    if 键 == "heartbench":
        return d.get("overall_score", 0.0)
    if 键 == "feel_heart":
        return (d.get("accuracy_score", 0.0) + d.get("empathy_score", 0.0) + d.get("consistency_score", 0.0)) / 3
    if 键 == "llm_judge":
        return (d.get("win_rate_against_human", 0.0) + d.get("average_rating", 0.0)) / 2
    if 键 == "turingbench":
        return d.get("human_likeness_score", 0.0)
    if 键 == "emocharacter":
        return (d.get("匹配fidelity", 0.0) + d.get("真实一致性", 0.0)) / 2
    return 0.0


def 情感综合(d):
    四分支 = d.get("四分支平均", 0.0)
    return (d.get("情绪识别准确率", 0.0) + d.get("情绪推理质量(0-1)", 0.0) + 四分支 / 5) / 3


def main():
    性能 = 加载(性能路径).get("性能", {})
    情感 = 加载(情感理解路径).get("结果", {})
    基准原始 = {k: 加载(v) for k, v in 基准文件.items()}

    模式得分 = {}   # 模式 -> {基准: 分, 综合}
    基准得分 = {}   # 基准 -> {模式: 分}
    for 键 in 基准文件:
        基准得分[键] = {}
        obj = 基准原始[键]
        模式汇总 = obj.get("模式汇总", obj)
        for 模式 in 七模式:
            基准得分[键][模式] = round(单基准分(键, 模式汇总.get(模式, {})), 4)

    情感得分 = {模式: 0.0 for 模式 in 七模式}
    for 模式 in 七模式:
        if 模式 in 情感:
            情感得分[模式] = round(情感综合(情感[模式]), 4)

    综合 = {}
    for 模式 in 七模式:
        # 5 基准平均
        五分 = [基准得分[k][模式] for k in 基准文件]
        基准均 = sum(五分) / len(五分)
        综合[模式] = round(基准均, 4)

    # 最佳模式判定
    最佳 = max(七模式, key=lambda m: 综合[m])

    汇总 = {
        "模型": "Qwen2.5-1.5B-Instruct",
        "种子": 2026,
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模式列表": 七模式,
        "性能": 性能,
        "情感理解": 情感得分,
        "基准得分": 基准得分,
        "综合得分(5基准平均)": 综合,
        "情感理解明细": {模式: 情感.get(模式, {}) for 模式 in 七模式},
        "基准原始": 基准原始,
        "_综合定义": {
            "heartbench": "overall_score",
            "feel_heart": "(accuracy+empathy+consistency)/3",
            "llm_judge": "(win_rate_against_human+average_rating)/2",
            "turingbench": "human_likeness_score",
            "emocharacter": "(匹配fidelity+真实一致性)/2",
            "情感理解": "(识别准确率+推理质量+四分支/5)/3",
        },
        "_最佳模式": 最佳,
    }

    with open(os.path.join(输出目录, "全流程_汇总_2026.json"), "w", encoding="utf-8") as f:
        json.dump(汇总, f, ensure_ascii=False, indent=2)
    print(f"汇总已保存 -> {输出目录}\\全流程_汇总_2026.json")

    with open(os.path.join(输出目录, "全流程_报告_2026.md"), "w", encoding="utf-8") as f:
        f.write(生成报告(汇总))
    print(f"报告已保存 -> {输出目录}\\全流程_报告_2026.md")

    print("\n=== 综合得分（5 基准平均）===")
    for 模式 in 七模式:
        print(f"  {模式}: {综合[模式]:.4f}")
    print(f"\n最佳模式: {最佳} ({综合[最佳]:.4f})")


def 生成报告(S):
    L = []
    L.append("# 全流程评估报告（7 模式 × 2026 种子）")
    L.append("")
    L.append(f"> 模型：{S['模型']} ｜ 种子：{S['种子']} ｜ 时间：{S['时间']}")
    L.append("> 7 模式：裸 / P1 语义回响(λ=0.29) / P1.5 兼容层(β自适应) / P2.5 潮汐ETD(α重加权) / P3 锚点回响(β=0.8,T=0.3) / P4 KV共振(κ=0.15,4层) / P5 超融合UFD")
    L.append("")
    L.append("## 一、综合得分（5 大图灵基准平均）")
    L.append("")
    L.append("| 模式 | HeartBench | HEART-BENCH | LLM-Judge | TuringBench | EmoCharacter | 综合 |")
    L.append("|---|---|---|---|---|---|---|")
    for 模式 in S["模式列表"]:
        row = [f"**{模式}**"]
        for 键 in ["heartbench", "feel_heart", "llm_judge", "turingbench", "emocharacter"]:
            row.append(f"{S['基准得分'][键][模式]:.4f}")
        row.append(f"**{S['综合得分(5基准平均)'][模式]:.4f}**")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append(f"**最佳模式：{S['_最佳模式']}（综合 {S['综合得分(5基准平均)'][S['_最佳模式']]:.4f}）**")
    L.append("")

    L.append("## 二、性能测试")
    L.append("")
    L.append("| 模式 | 平均耗时(s) | 首token延迟(s) | 吞吐(tok/s) | 峰值显存(GB) | GPU利用率(%) |")
    L.append("|---|---|---|---|---|---|")
    for 模式 in S["模式列表"]:
        p = S["性能"].get(模式, {})
        L.append(f"| {模式} | {p.get('平均耗时(s)','-')} | {p.get('平均首token延迟(s)','-')} | {p.get('平均吞吐(tok/s)','-')} | {p.get('峰值显存(GB)','-')} | {p.get('平均GPU利用率(%)','-')} |")
    L.append("")

    L.append("## 三、情感理解（大厂式）")
    L.append("")
    L.append("| 模式 | 情绪识别准确率 | 情绪推理质量(0-1) | 四分支EI平均(1-5) | 综合 |")
    L.append("|---|---|---|---|---|")
    for 模式 in S["模式列表"]:
        d = S["情感理解明细"].get(模式, {})
        四分支 = d.get("四分支平均", "-")
        L.append(f"| {模式} | {d.get('情绪识别准确率','-')} | {d.get('情绪推理质量(0-1)','-')} | {四分支} | {S['情感理解'][模式]:.4f} |")
    L.append("")
    L.append("> 方法论：①情绪识别(MME-Emotion 式) ②情绪推理(CAREBench 式) ③四分支 EI(AttuneBench 式：感知/理解/促进思维/管理)")
    L.append("")

    L.append("## 四、5 大基准明细")
    L.append("")
    L.append("### 1. HeartBench（中文「人味儿」，官方 rubric 逐条命中 + norm_score）")
    L.append("")
    L.append("| 模式 | overall | 人格 | 情绪 | 社交 | 道德 |")
    L.append("|---|---|---|---|---|---|")
    hb = S["基准原始"]["heartbench"].get("模式汇总", {})
    for 模式 in S["模式列表"]:
        d = hb.get(模式, {})
        ds = d.get("dimension_scores", {})
        L.append(f"| {模式} | {d.get('overall_score','-')} | {ds.get('人格','-')} | {ds.get('情绪','-')} | {ds.get('社交','-')} | {ds.get('道德','-')} |")
    L.append("")
    L.append("### 2. HEART-BENCH（记忆驱动人格推理 + 共情，FEEL 备用）")
    L.append("")
    L.append("| 模式 | 行为预测准确率 | 决策一致性 | 共情评分 |")
    L.append("|---|---|---|---|")
    fh = S["基准原始"]["feel_heart"].get("模式汇总", {})
    for 模式 in S["模式列表"]:
        d = fh.get(模式, {})
        L.append(f"| {模式} | {d.get('accuracy_score','-')} | {d.get('consistency_score','-')} | {d.get('empathy_score','-')} |")
    L.append("")
    L.append("### 3. LLM-as-Judge（AI vs 真人盲评，双投消位置偏差）")
    L.append("")
    L.append("| 模式 | 对真人胜率 | AI 平均分(1-5) | 真人平均分(1-5) |")
    L.append("|---|---|---|---|")
    lj = S["基准原始"]["llm_judge"].get("模式汇总", {})
    for 模式 in S["模式列表"]:
        d = lj.get(模式, {})
        L.append(f"| {模式} | {d.get('win_rate_against_human','-')} | {d.get('average_rating_raw','-')} | {d.get('human_average_rating_raw','-')} |")
    L.append("")
    L.append("### 4. TuringBench（中文体系图灵检测：TF-IDF+LR 检测器）")
    L.append("")
    L.append("| 模式 | 被判 AI 比例 | 人似度(1-检测率) | 人类误判率 |")
    L.append("|---|---|---|---|")
    tb = S["基准原始"]["turingbench"].get("模式汇总", {})
    for 模式 in S["模式列表"]:
        d = tb.get(模式, {})
        L.append(f"| {模式} | {d.get('detection_accuracy','-')} | {d.get('human_likeness_score','-')} | {d.get('人类文本误判率','-')} |")
    L.append("")
    L.append("### 5. EmoCharacter v2（角色扮演情感保真度，差分校正）")
    L.append("")
    L.append("| 模式 | 匹配fidelity | 错配fidelity | 净区分度 | 真实一致性 | 一致性识别率 |")
    L.append("|---|---|---|---|---|---|")
    ec = S["基准原始"]["emocharacter"].get("模式汇总", {})
    for 模式 in S["模式列表"]:
        d = ec.get(模式, {})
        L.append(f"| {模式} | {d.get('匹配fidelity','-')} | {d.get('错配fidelity','-')} | {d.get('净区分度','-')} | {d.get('真实一致性','-')} | {d.get('一致性识别率','-')} |")
    L.append("")
    L.append(f"> 中性下限（固定无情感回复的匹配分）：{ec.get(list(ec.keys())[0], {}).get('中性下限fidelity', '-')}")
    L.append("")

    L.append("## 五、结论")
    L.append("")
    最佳 = S["_最佳模式"]
    L.append(f"- 综合 5 大基准，**{最佳}** 得分最高（{S['综合得分(5基准平均)'][最佳]:.4f}），裸模型为 {S['综合得分(5基准平均)']['裸']:.4f}。")
    L.append("- 全部 7 模式在同种子（2026）、同样本、同裁判协议下完成，结果可直接横向对比。")
    L.append("- 各方案在不同维度各有优势：见上方明细表。")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
