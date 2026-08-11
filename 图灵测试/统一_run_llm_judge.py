# -*- coding: utf-8 -*-
"""
③ LLM-as-Judge — AI 回复 vs 真人回复 盲评（本地自实现）
=====================================================
- 提示词集：样本_30条.json 的 user 话（真人对话）
- 候选 A：目标模型 1.5B 对 user 生成回复
- 候选 B：真人回复（girl 字段）
- 裁判 7B：盲评（不知道来源）判定"哪个更像真人写的"，并对每个回复打 1-5 分
指标：win_rate_against_human（AI 胜率）+ average_rating（归一化 rating/5）

两阶段架构（生成 1.5B 与裁判 4bit 7B 分离，避免内存不足崩溃）：
  python 统一_run_llm_judge.py --仅生成 缓存.json
  python 统一_run_llm_judge.py --仅裁判 缓存.json
"""
import json
import os
import re
import random
import time

本目录 = os.path.dirname(os.path.abspath(__file__))
样本路径 = os.path.join(本目录, "样本_30条.json")
日志路径 = "i:\\Desktop\\\u8bed\u4e49\u56de\u54cd\\\u56fe\u7075\u6d4b\u8bd5\\\u7edf\u4e00\u57fa\u51c6\\LLM_judge_log.txt"
结果路径 = "i:\\Desktop\\\u8bed\u4e49\u56de\u54cd\\\u56fe\u7075\u6d4b\u8bd5\\\u7edf\u4e00\u57fa\u51c6\\llm_judge_results_2026.json"

import sys
sys.path.insert(0, 本目录)
import 公共模块 as cm

# ===== 统一 7 模式补丁（种子 2026） =====
七模式列表 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合"]
统一目录 = os.path.join(本目录, "统一基准")
os.makedirs(统一目录, exist_ok=True)
# ===== /统一 7 模式补丁 =====



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


# ============================================================
# 阶段1：生成 AI 回复缓存（1.5B，进程随后退出）
# ============================================================
def 生成回复缓存(模式列表, runs, seed_base, 长度, 模板, 身份注入, 提示词, 缓存路径, λ=None):
    from 统一生成器 import 生成器实例
    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"]
    记录日志(f"样本数: {len(样本)}")
    缓存 = {"seed_base": seed_base, "runs": runs, "样本": 样本, "AI回复": {}}
    for 模式 in 模式列表:
        记录日志(f"──── 生成 [{模式}] ────")
        for run_idx in range(runs):
            seed_offset = seed_base + run_idx * 100
            random.seed(seed_offset)
            random.shuffle(样本)
            记录日志(f"  [run {run_idx+1}/{runs}] seed_offset={seed_offset}")
            AI回复列表 = []
            for i, r in enumerate(样本):
                消息 = [{"role": "user", "content": r["user"]}]
                ai回复 = 生成器实例.生成(模式, 消息, 种子=seed_offset, 轮次=i, max_new_tokens=长度, λ覆盖=λ, 模板=模板, 身份注入=身份注入, 提示词=提示词)
                AI回复列表.append(ai回复)
                记录日志(f"[AI生成 {i+1}/{len(样本)}] {r['user'][:20]} => {ai回复[:30]}")
            缓存["AI回复"].setdefault(模式, {})[str(run_idx)] = AI回复列表
        生成器实例.清理()
    with open(缓存路径, "w", encoding="utf-8") as f:
        json.dump(缓存, f, ensure_ascii=False, indent=2)
    记录日志(f"生成缓存已保存 -> {缓存路径}")


# ============================================================
# 阶段2：独立裁判（干净进程，子进程加载 4bit 7B）
# ============================================================
def 裁判汇总(模式列表, runs, 缓存):
    import subprocess
    样本 = 缓存["样本"]
    裁判子进程 = os.path.join(本目录, "裁判子进程.py")
    全部汇总 = {}
    多次运行明细 = {}
    for 模式 in 模式列表:
        记录日志(f"──── 裁判 [{模式}] ────")
        多次运行明细[模式] = []
        for run_idx in range(runs):
            AI回复列表 = 缓存["AI回复"][模式][str(run_idx)]
            记录日志(f"  [run {run_idx+1}/{runs}]")
            # 2a. 配对（双投）
            请求列表 = []
            for i, r in enumerate(样本):
                请求列表.append({"user": r["user"], "AI": AI回复列表[i], "真人": r["girl"], "AI在前": True})
                请求列表.append({"user": r["user"], "AI": AI回复列表[i], "真人": r["girl"], "AI在前": False})
            任务路径 = os.path.join(统一目录, f"_lj_{模式.replace(' ','')}_pair.json")
            输出路径 = os.path.join(统一目录, f"_lj_{模式.replace(' ','')}_pair_out.json")
            with open(任务路径, "w", encoding="utf-8") as f:
                json.dump({"裁判": "llm_judge_配对", "请求": 请求列表}, f, ensure_ascii=False)
            subprocess.run([sys.executable, 裁判子进程, 任务路径, 输出路径], check=True)
            with open(输出路径, encoding="utf-8") as f:
                配对输出 = json.load(f)
            配对记录 = []
            for i, r in enumerate(样本):
                ai回复 = AI回复列表[i]
                crA = 配对输出[i * 2]
                crB = 配对输出[i * 2 + 1]
                选择A = crA.get("选择")
                选择B = crB.get("选择")
                ai胜A = 1.0 if 选择A == "A" else 0.0
                ai胜B = 1.0 if 选择B == "B" else 0.0
                ai胜 = 1.0 if (ai胜A or ai胜B) else 0.0
                配对记录.append({
                    "序号": r["序号"], "user": r["user"], "ai回复": ai回复, "真人回复": r["girl"],
                    "A位判": 选择A, "B位判": 选择B, "AI胜": ai胜,
                    "原始A": crA.get("原始", ""), "原始B": crB.get("原始", ""),
                })
                记录日志(f"[配对 {i+1}/{len(样本)}] A位判{选择A} B位判{选择B} AI胜={ai胜}")

            # 评分
            评分记录 = []
            评分请求 = []
            for i, r in enumerate(样本):
                for 标签, 文本 in (("真人", r["girl"]), ("AI", AI回复列表[i])):
                    评分请求.append({"user": r["user"], "回复": 文本, "标签": 标签, "序号": r["序号"]})
            评分任务 = os.path.join(统一目录, f"_lj_{模式.replace(' ','')}_score.json")
            评分输出 = os.path.join(统一目录, f"_lj_{模式.replace(' ','')}_score_out.json")
            with open(评分任务, "w", encoding="utf-8") as f:
                json.dump({"裁判": "llm_judge_评分", "请求": 评分请求}, f, ensure_ascii=False)
            subprocess.run([sys.executable, 裁判子进程, 评分任务, 评分输出], check=True)
            with open(评分输出, encoding="utf-8") as f:
                评分结果 = json.load(f)
            for (序号, 标签, 文本), cr in zip(
                    [(q["序号"], q["标签"], q["回复"]) for q in 评分请求], 评分结果):
                分 = cr.get("评分")
                评分记录.append({"序号": 序号, "来源": 标签, "评分": 分, "原始": cr.get("原始", "")})
                记录日志(f"[评分 {序号}.{标签}] {分} 分")

            ai胜率 = sum(x["AI胜"] for x in 配对记录) / len(配对记录) if 配对记录 else 0.0
            ai评分 = [x["评分"] for x in 评分记录 if x["来源"] == "AI" and x["评分"]]
            真人评分 = [x["评分"] for x in 评分记录 if x["来源"] == "真人" and x["评分"]]
            run_summary = {
                "win_rate_against_human": round(ai胜率, 4),
                "average_rating": round(sum(ai评分) / len(ai评分) / 5, 4) if ai评分 else 0.0,
                "average_rating_raw": round(sum(ai评分) / len(ai评分), 2) if ai评分 else 0.0,
                "human_average_rating_raw": round(sum(真人评分) / len(真人评分), 2) if 真人评分 else 0.0,
                "样本数": len(配对记录),
            }
            多次运行明细[模式].append({
                "run_summary": run_summary,
                "配对记录": 配对记录,
                "评分记录": 评分记录,
            })
            记录日志(f"[run {run_idx+1}] {json.dumps(run_summary, ensure_ascii=False)}")

        # 汇总多次运行结果：取均值与标准差
        win_rate_list = [d["run_summary"]["win_rate_against_human"] for d in 多次运行明细[模式]]
        avg_rating_list = [d["run_summary"]["average_rating"] for d in 多次运行明细[模式]]
        avg_rating_raw_list = [d["run_summary"]["average_rating_raw"] for d in 多次运行明细[模式]]
        human_rating_raw_list = [d["run_summary"]["human_average_rating_raw"] for d in 多次运行明细[模式]]
        汇总 = {
            "win_rate_against_human": round(sum(win_rate_list) / len(win_rate_list), 4),
            "win_rate_std": round((sum((x - sum(win_rate_list)/len(win_rate_list))**2 for x in win_rate_list) / len(win_rate_list)) ** 0.5, 4) if len(win_rate_list) > 1 else 0.0,
            "average_rating": round(sum(avg_rating_list) / len(avg_rating_list), 4),
            "average_rating_std": round((sum((x - sum(avg_rating_list)/len(avg_rating_list))**2 for x in avg_rating_list) / len(avg_rating_list)) ** 0.5, 4) if len(avg_rating_list) > 1 else 0.0,
            "average_rating_raw": round(sum(avg_rating_raw_list) / len(avg_rating_raw_list), 2),
            "average_rating_raw_std": round((sum((x - sum(avg_rating_raw_list)/len(avg_rating_raw_list))**2 for x in avg_rating_raw_list) / len(avg_rating_raw_list)) ** 0.5, 4) if len(avg_rating_raw_list) > 1 else 0.0,
            "human_average_rating_raw": round(sum(human_rating_raw_list) / len(human_rating_raw_list), 2),
            "human_average_rating_raw_std": round((sum((x - sum(human_rating_raw_list)/len(human_rating_raw_list))**2 for x in human_rating_raw_list) / len(human_rating_raw_list)) ** 0.5, 4) if len(human_rating_raw_list) > 1 else 0.0,
            "样本数": len(多次运行明细[模式][0]["配对记录"]),
            "_runs": runs,
            "_多次运行明细": [d["run_summary"] for d in 多次运行明细[模式]],
        }
        全部汇总[模式] = 汇总
        记录日志(f"[{模式} 汇总] {json.dumps(汇总, ensure_ascii=False)}")

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
    ap.add_argument("--早停", action="store_true", help="已改为两阶段，早停仅作兼容保留")
    ap.add_argument("--λ", type=float, default=None, help="四层模式 λ 覆盖（任务自适应扫描）")
    ap.add_argument("--模板", choices=["chat", "纯文本"], default="chat", help="四层模式生成模板")
    ap.add_argument("--长度", type=int, default=64, help="AI 回复 max_new_tokens")
    ap.add_argument("--身份", choices=["on", "off"], default="on", help="四层模式是否注入身份 system")
    ap.add_argument("--提示词", choices=["人类身份", "图灵测试"], default="人类身份", help="四层模式身份提示词版本")
    ap.add_argument("--runs", type=int, default=1, help="多次测试轮数（>=1）")
    ap.add_argument("--seed_base", type=int, default=42, help="随机种子基数，每次 run 递增")
    ap.add_argument("--仅生成", type=str, default=None, help="只生成 AI 回复缓存到指定 JSON 后退出")
    ap.add_argument("--仅裁判", type=str, default=None, help="只裁判（读缓存 JSON），进程干净加载")
    args = ap.parse_args()
    模式列表 = 七模式列表 if "全部" in args.模式 else args.模式
    runs = max(1, args.runs)

    if os.path.exists(日志路径):
        os.remove(日志路径)
    记录日志(f"=== LLM-as-Judge 评测开始（模式：{模式列表}，runs={runs}）===")

    默认缓存 = os.path.join(统一目录, "llm_judge_cache_2026.json")

    if args.仅生成:
        生成回复缓存(模式列表, runs, args.seed_base, args.长度, args.模板,
                     (args.身份 == "on"), args.提示词, args.仅生成, λ=args.λ)
        return
    if args.仅裁判:
        with open(args.仅裁判, encoding="utf-8") as f:
            缓存 = json.load(f)
        裁判汇总(模式列表, runs, 缓存)
        return

    print("请分两步执行（先 --仅生成 再 --仅裁判），以彻底释放 1.5B 内存：")
    print(f"  python 统一_run_llm_judge.py --仅生成 {默认缓存}")
    print(f"  python 统一_run_llm_judge.py --仅裁判 {默认缓存}")


if __name__ == "__main__":
    main()
