# -*- coding: utf-8 -*-
"""
④ TuringBench — 大规模图灵测试基准（本地适配版 · 中文体系）
================================================================
TuringBench 官方数据集为英文，而语义回响整个架构（cnsenti 中文情感词库、
中文提示词、中文语料）建立在中文之上——直接用英文数据集测会与体系脱节。

因此本基准按 TuringBench 思想在中文上本地构建检测器（GPTZero/TuringBench
均为"训练检测器区分 AI/人类文本"）：
  1. 人类语料：chinese-adorable 高情商对话数据集 girl 回复（真人中文）
  2. AI 语料：目标模型 1.5B 对相同 user 中文输入生成回复
  3. 检测器：TF-IDF(1-2gram) + 逻辑回归，在 人类 vs AI 上训练（留出测试集）
  4. detection_accuracy = 1.5B 新生成文本被判为 AI 的比例（越低越像人）
     human_likeness = 1 - detection_accuracy

数据完全中文，检测器在目标模型自身输出与真人之间做区分，符合
"AI 文本能否与人类中文写作区分开"的图灵测试语义。
"""
import json
import os
import random
import time

本目录 = os.path.dirname(os.path.abspath(__file__))
数据集路径 = r"c:\Users\Administrator\.cache\huggingface\hub\datasets--sunorme--chinese-adorable-high-emotional-intelligence-chat\snapshots\15f8a4895c7529c16cd8b43bccc95abf4f8b7c6b\chinese-adorable-high-emotional-intelligence-chat.json"
日志路径 = os.path.join(本目录, "logs", "TuringBench_log.txt")
结果路径 = os.path.join(本目录, "data", "turingbench_results.json")

import sys
sys.path.insert(0, 本目录)
import 公共模块 as cm

训练对数 = 60
测试对数 = 30


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 加载对话():
    with open(数据集路径, encoding="utf-8") as f:
        data = json.load(f)
    有效 = [
        d for d in data
        if isinstance(d, dict) and d.get("user") and d.get("girl")
        and 2 <= len(d["user"]) <= 80 and 2 <= len(d["girl"]) <= 200
    ]
    return 有效


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "四层", "全部"], default="全部")
    ap.add_argument("--runs", type=int, default=1, help="多次测试轮数（>=1）")
    ap.add_argument("--seed_base", type=int, default=42, help="随机种子基数，每次 run 递增")
    args = ap.parse_args()
    模式列表 = ["裸", "四层"] if args.模式 == "全部" else [args.模式]
    runs = max(1, args.runs)

    if os.path.exists(日志路径):
        os.remove(日志路径)
    记录日志(f"=== TuringBench 评测开始（中文体系检测器，模式：{模式列表}，runs={runs}）===")

    对话 = 加载对话()
    记录日志(f"对话总数 {len(对话)} | 训练 {训练对数} 对 | 测试 {测试对数} 对")

    from 生成器 import 生成器实例
    全部汇总 = {}
    多次运行明细 = {}
    for 模式 in 模式列表:
        记录日志(f"──── 模式 [{模式}] ────")
        多次运行明细[模式] = []
        for run_idx in range(runs):
            seed_offset = args.seed_base + run_idx * 100
            random.seed(seed_offset)
            random.shuffle(对话)
            训练对话 = 对话[:训练对数]
            测试对话 = 对话[训练对数:训练对数 + 测试对数]
            记录日志(f"  [run {run_idx+1}/{runs}] seed_offset={seed_offset}")

            AI训练文本, AI测试文本 = [], []
            for i, d in enumerate(训练对话):
                文本 = 生成器实例.生成(模式, [{"role": "user", "content": d["user"]}], 种子=seed_offset, 轮次=i, max_new_tokens=64, 模板="纯文本")
                AI训练文本.append(文本)
                记录日志(f"[AI训练 {i+1}/{len(训练对话)}] {d['user'][:16]} => {文本[:30]}")
            for i, d in enumerate(测试对话):
                文本 = 生成器实例.生成(模式, [{"role": "user", "content": d["user"]}], 种子=seed_offset, 轮次=训练对数 + i, max_new_tokens=64, 模板="纯文本")
                AI测试文本.append(文本)
                记录日志(f"[AI测试 {i+1}/{len(测试对话)}] {d['user'][:16]} => {文本[:30]}")

            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.metrics import accuracy_score

            X人类 = [d["girl"] for d in 训练对话]
            XAI = AI训练文本
            X训练 = X人类 + XAI
            y训练 = [1] * len(X人类) + [0] * len(XAI)

            X人类测试 = [d["girl"] for d in 测试对话]
            XAI测试 = AI测试文本
            X测试 = X人类测试 + XAI测试
            y测试 = [1] * len(X人类测试) + [0] * len(XAI测试)

            检测器 = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True),
                LogisticRegression(max_iter=1000),
            )
            t0 = time.time()
            检测器.fit(X训练, y训练)
            训练acc = accuracy_score(y训练, 检测器.predict(X训练))
            测试acc = accuracy_score(y测试, 检测器.predict(X测试))
            记录日志(f"[{模式}][run {run_idx+1}] 检测器训练完成 {time.time()-t0:.1f}s | 训练准确率 {训练acc:.3f} | 留出测试准确率 {测试acc:.3f}")

            判定明细 = []
            AI判AI = 0
            for t in XAI测试:
                预测 = 检测器.predict([t])[0]
                是AI = (预测 == 0)
                AI判AI += int(是AI)
                判定明细.append({"文本": t[:120], "预测": "AI" if 是AI else "人类"})
                记录日志(f"[检测] {t[:40]}... -> {'AI' if 是AI else '人类'}")

            真人判AI = sum(1 for t in X人类测试 if 检测器.predict([t])[0] == 0)

            n_ai = len(XAI测试)
            n_hu = len(X人类测试)
            run_summary = {
                "run_idx": run_idx,
                "seed_offset": seed_offset,
                "detection_accuracy": round(AI判AI / n_ai, 4) if n_ai else 0.0,
                "human_likeness_score": round((n_ai - AI判AI) / n_ai, 4) if n_ai else 0.0,
                "人类文本误判率": round(真人判AI / n_hu, 4) if n_hu else 0.0,
                "检测器训练准确率": round(训练acc, 4),
                "检测器留出测试准确率": round(测试acc, 4),
                "AI文本数": n_ai,
                "真人文本数": n_hu,
            }
            多次运行明细[模式].append(run_summary)
            记录日志(f"[run {run_idx+1}] {json.dumps(run_summary, ensure_ascii=False)}")

        det_acc_list = [d["detection_accuracy"] for d in 多次运行明细[模式]]
        human_like_list = [d["human_likeness_score"] for d in 多次运行明细[模式]]
        误判_list = [d["人类文本误判率"] for d in 多次运行明细[模式]]
        训练acc_list = [d["检测器训练准确率"] for d in 多次运行明细[模式]]
        测试acc_list = [d["检测器留出测试准确率"] for d in 多次运行明细[模式]]

        def _std(lst):
            n = len(lst)
            if n <= 1:
                return 0.0
            mean = sum(lst) / n
            return round((sum((x - mean) ** 2 for x in lst) / n) ** 0.5, 4)

        汇总 = {
            "detection_accuracy": round(sum(det_acc_list) / len(det_acc_list), 4),
            "detection_accuracy_std": _std(det_acc_list),
            "human_likeness_score": round(sum(human_like_list) / len(human_like_list), 4),
            "human_likeness_score_std": _std(human_like_list),
            "人类文本误判率": round(sum(误判_list) / len(误判_list), 4),
            "人类文本误判率_std": _std(误判_list),
            "检测器训练准确率": round(sum(训练acc_list) / len(训练acc_list), 4),
            "检测器训练准确率_std": _std(训练acc_list),
            "检测器留出测试准确率": round(sum(测试acc_list) / len(测试acc_list), 4),
            "检测器留出测试准确率_std": _std(测试acc_list),
            "AI文本数": n_ai,
            "真人文本数": n_hu,
            "方法": "中文体系：TF-IDF(1-2gram)+LR，真人(girl) vs 1.5B 生成回复，留出测试",
            "_runs": runs,
            "_多次运行明细": 多次运行明细[模式],
        }
        全部汇总[模式] = 汇总
        记录日志(f"[{模式} 汇总] {json.dumps(汇总, ensure_ascii=False)}")

    生成器实例.清理()
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"模式汇总": 全部汇总, "_runs": runs, "_多次运行明细": 多次运行明细}, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {结果路径}")
    return 全部汇总 if len(全部汇总) > 1 else 全部汇总[模式列表[0]]


if __name__ == "__main__":
    main()
