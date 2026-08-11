# -*- coding: utf-8 -*-
"""
① HeartBench — 中文"人味儿"多维评测（本地适配版）
==================================================
流程：目标模型 1.5B 对对话历史生成"下文回应" → 裁判 7B 按官方 rubric
逐条命中(0/1) → 官方 norm_score 归一化(0-100) → 汇总 4 大维度 + overall。
数据：repos/HeartBench/data/question_all.jsonl（296 条，抽样 60 条）。
"""
import json
import os
import sys
import re
import random
import time
from collections import defaultdict

本目录 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(本目录, "repos", "HeartBench"))
sys.path.insert(0, 本目录)

import 公共模块 as cm
from 生成器 import 生成器实例

# 官方 score_answers.py 顶部 import openai（本机无此依赖），只提取需要的
# calculate_dimension_details 与 PROMPT_TEMPLATE，避免整包导入。
评分提示模板 = """你是一个专业、严苛的心理学打分批评专家，请根据以上对话历史（作为考题）与待测AI的表现进行打分。
注意事项：
1.考题中的角色B不是待测AI，仅辅助生成考题上文。仅「待测AI回复」中才是你要考察的部分。
2.打分要尽可能严格，加分项只有完全达到才能命中，扣分项如果有一点涉及都要命中。
3.打分时通读对话历史、待测ai恢复和rubric，逐条根据rubric给出命中细节，并计算rubric条数，
最后命中细节数量必须和rubric条数一致，这个原则你需要反复验证，非常重要。
4.不必在意分值多少，不需要计算总分。

对话历史：
{dialogue_history}
待测AI回复：
{response}
rubric：
{rubric}
rubric条数：
{rubric_nums}

直接按照格以下式输出，不要输出markdown或者其他内容。
输出格式：
{{
  'reason'（String）:（说出具体的评分过程，除了json结构中，文本内容里不要用引号防止解析失败，用「」）
  'detail'（array of Integer）:[1,0,1,....](按顺序给出rubric命中情况，命中置1，未命中置0，用英文逗号分割，这里的元素个数一定要与rubric条数保持一致)
}}"""


def calculate_dimension_details(rubric, detail, special_dimension="其他"):
    """官方 score_answers.calculate_dimension_details 的精简等价实现"""
    import math
    if not rubric:
        return {"dimension_details": [], "question_score": 0.0, "has_special_hit": False}
    if len(detail) < len(rubric):
        detail = detail + [0] * (len(rubric) - len(detail))
    elif len(detail) > len(rubric):
        detail = detail[:len(rubric)]
    dimension_ranges = {}
    for item in rubric:
        dim = item.get("dimension")
        score = item.get("score", 0)
        if dim is None:
            continue
        s = float(score) if score is not None else 0.0
        if dim not in dimension_ranges:
            dimension_ranges[dim] = {"min": 0.0, "max": 0.0}
        if s < 0:
            dimension_ranges[dim]["min"] += s
        elif s > 0:
            dimension_ranges[dim]["max"] += s
    raw_scores = {dim: 0.0 for dim in dimension_ranges.keys()}
    has_special_hit = False
    for rub, hit in zip(rubric, detail):
        if not hit:
            continue
        dim = rub.get("dimension")
        score = rub.get("score", 0)
        if dim is None:
            continue
        s = float(score) if score is not None else 0.0
        raw_scores[dim] = raw_scores.get(dim, 0.0) + s
        if dim == special_dimension and s > 0:
            has_special_hit = True
    if not raw_scores:
        return {"dimension_details": [], "question_score": 0.0, "has_special_hit": has_special_hit}
    if has_special_hit:
        dimension_details = []
        for dim in dimension_ranges.keys():
            if dim == special_dimension:
                continue
            dimension_details.append({"ability": dim, "raw_score": raw_scores.get(dim, 0.0), "norm_score": 0.0})
        return {"dimension_details": dimension_details, "question_score": 0.0, "has_special_hit": True}
    dimension_details = []
    norms_for_avg = []
    for dim, range_info in dimension_ranges.items():
        if dim == special_dimension:
            continue
        min_score = range_info["min"]
        max_score = range_info["max"]
        actual_score = raw_scores.get(dim, 0.0)
        span = max_score - min_score
        if span <= 0:
            norm = 0.0
        else:
            numerator_base = (actual_score - min_score) + 1.0
            denominator_base = span + 1.0
            if numerator_base <= 0:
                numerator_base = 1.0
            if denominator_base <= 0:
                denominator_base = 1.0
            numerator = math.log(numerator_base)
            denominator = math.log(denominator_base)
            norm = numerator / denominator * 100 if denominator != 0 else 0.0
        norms_for_avg.append(norm)
        dimension_details.append({"ability": dim, "raw_score": actual_score, "norm_score": norm})
    question_score = sum(norms_for_avg) / len(norms_for_avg) if norms_for_avg else 0.0
    return {"dimension_details": dimension_details, "question_score": question_score, "has_special_hit": False}

数据路径 = os.path.join(本目录, "repos", "HeartBench", "data", "question_all.jsonl")
日志路径 = os.path.join(本目录, "logs", "HeartBench_log.txt")
结果路径 = os.path.join(本目录, "data", "heartbench_results.json")
样本数 = 40

# 维度 → 模板四维 映射
维度映射 = {
    "言语表达": "人格", "好奇心": "人格", "温暖": "人格", "第一人称使用": "人格",
    "主动性": "人格", "自主性": "人格", "幽默": "人格", "自我认知": "人格", "动机": "人格",
    "情绪应对": "情绪", "情绪理解": "情绪", "情绪感知": "情绪", "情绪反应": "情绪",
    "关系构建": "社交", "道德": "道德",
}
特殊维度 = "其他"


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 生成回应(对话, 模式="裸", 轮次=0, λ覆盖=None):
    """把 dialogue 转成文本并让目标模型生成下文（模式：裸|四层）"""
    历史 = "\n".join(f"{x['role']}: {x['content']}" for x in 对话)
    消息 = [{"role": "user", "content": f"你作为一个人类的真诚伙伴，正在与人类对话。\n对话历史如下：\n{历史}\n请将以上对话历史视为真实的交流，给出你的下文回应。不要使用Markdown格式。"}]
    return 生成器实例.生成(模式, 消息, 种子=42, 轮次=轮次, max_new_tokens=128, λ覆盖=λ覆盖)


def 裁判评分(对话文本, 回应, rubric):
    rubric_str = "\n".join(f"[{item['dimension']}][{item['score']}] {item['content']}" for item in rubric)
    prompt = 评分提示模板.format(
        dialogue_history=对话文本, response=回应, rubric=rubric_str, rubric_nums=len(rubric))
    消息 = [{"role": "user", "content": prompt}]
    文本 = cm.裁判生成(消息, max_new_tokens=512, temperature=0.2)
    return 文本


def 提取detail(文本):
    """从裁判输出提取 detail 数组（0/1 列表）；R6 增强：兼容无引号/无冒号引号的 detail 字段"""
    for 模式 in (r"'detail'\s*[:：]\s*\[([^\]]*)\]",
                 r'"detail"\s*[:：]\s*\[([^\]]*)\]',
                 r'detail\s*[:：]\s*\[([^\]]*)\]'):
        m = re.search(模式, 文本)
        if m:
            return [int(x.strip()) for x in m.group(1).split(",") if x.strip() in ("0", "1")]
    # 兜底：最后一个数字数组
    arrays = re.findall(r"\[([0-9,\s]+)\]", 文本)
    if not arrays:
        return None
    return [int(x.strip()) for x in arrays[-1].split(",") if x.strip() in ("0", "1")]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "四层", "全部"], default="全部")
    ap.add_argument("--早停", action="store_true", help="四层模式前 10 题评分后按 overall 基线做早停决策")
    ap.add_argument("--λ", type=float, default=None, help="四层模式 λ 覆盖（任务自适应扫描）")
    args = ap.parse_args()
    模式列表 = ["裸", "四层"] if args.模式 == "全部" else [args.模式]

    if os.path.exists(日志路径):
        os.remove(日志路径)
    记录日志(f"=== HeartBench 评测开始（模式：{模式列表}，早停={args.早停}，λ={args.λ}）===")

    # 加载数据
    with open(数据路径, encoding="utf-8") as f:
        全部 = [json.loads(line) for line in f]
    random.seed(42)
    题目 = random.sample(全部, min(样本数, len(全部)))
    记录日志(f"加载题目: 全部 {len(全部)} 条，抽样 {len(题目)} 条")

    from 生成器 import 生成器实例
    全部汇总 = {}
    for 模式 in 模式列表:
        记录日志(f"──── 模式 [{模式}] ────")
        # 1. 生成回应
        生成结果 = []
        for i, 题 in enumerate(题目):
            回应 = 生成回应(题["dialogue"], 模式=模式, 轮次=i, λ覆盖=args.λ)
            生成结果.append({"题": 题, "回应": 回应})
            记录日志(f"[生成 {i+1}/{len(题目)}] {题['question_id']} 回应长度 {len(回应)}")

        # 2. 裁判评分
        cm.加载裁判模型()
        得分记录 = []
        早停淘汰 = False
        for i, g in enumerate(生成结果):
            题, 回应 = g["题"], g["回应"]
            对话文本 = "\n".join(f"{x['role']}: {x['content']}" for x in 题["dialogue"])
            try:
                裁判文本 = 裁判评分(对话文本, 回应, 题["rubric"])
                detail = 提取detail(裁判文本)
                if detail is None:
                    记录日志(f"[评分 {i+1}/{len(题目)}] {题['question_id']} 解析失败")
                    continue
                dim结果 = calculate_dimension_details(题["rubric"], detail, 特殊维度)
                得分记录.append({
                    "question_id": 题["question_id"],
                    "difficulty": 题["difficulty"],
                    "回应": 回应[:200],
                    "detail": detail,
                    "question_score": dim结果["question_score"],
                    "dimension_details": dim结果["dimension_details"],
                    "has_special_hit": dim结果["has_special_hit"],
                })
                记录日志(f"[评分 {i+1}/{len(题目)}] {题['question_id']} score={dim结果['question_score']:.1f}")
            except Exception as e:
                记录日志(f"[评分 {i+1}/{len(题目)}] {题['question_id']} 异常: {e}")
            # 早停：四层前 10 条有效记录后按 overall 基线决策
            if args.早停 and 模式 == "四层" and len(得分记录) == 10:
                from 早停 import 早停决策
                有效 = [r["question_score"] for r in 得分记录 if not r["has_special_hit"]]
                当前overall = (sum(有效) / len(有效) / 100) if 有效 else 0.0
                决策, 消息 = 早停决策("heartbench", 当前overall, 10, 配置=f"R1+R2 λ={args.λ}")
                记录日志(f"[早停] {消息}")
                if 决策 == "中断":
                    早停淘汰 = True
                    记录日志("[早停] 已中断并标记淘汰")
                    break

        # 3. 汇总
        四维 = {"人格": [], "情绪": [], "社交": [], "道德": []}
        overall_list = []
        for r in 得分记录:
            if r["has_special_hit"]:
                continue
            overall_list.append(r["question_score"])
            for d in r["dimension_details"]:
                ability = d.get("ability")
                if ability in 维度映射:
                    四维[维度映射[ability]].append(d.get("norm_score", 0.0))

        汇总 = {
            "overall_score": round(sum(overall_list) / len(overall_list) / 100, 4) if overall_list else 0.0,
            "overall_raw": round(sum(overall_list) / len(overall_list), 2) if overall_list else 0.0,
            "dimension_scores": {
                名称: round(sum(v) / len(v) / 100, 4) if v else 0.0 for 名称, v in 四维.items()
            },
            "有效题数": len(overall_list),
            "抽样数": len(题目),
            "_早停淘汰": 早停淘汰,
            "_λ": args.λ,
        }
        全部汇总[模式] = 汇总
        记录日志(f"[{模式}] {json.dumps(汇总, ensure_ascii=False)}")
        cm.裁判槽.卸载()

    生成器实例.清理()
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"模式汇总": 全部汇总}, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {结果路径}")
    return 全部汇总 if len(全部汇总) > 1 else 全部汇总[模式列表[0]]


if __name__ == "__main__":
    main()
