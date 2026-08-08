# -*- coding: utf-8 -*-
"""
三组同种子对比：裸 vs 全面向量纠正 vs 思考链纠正（创新方案）
============================================================
- 模型：Qwen3-4B（fp16，通用注入 λ=0.0588 γ=0.0845 τ=0.0592）
- prompt：chat template 渲染（enable_thinking，触发 Qwen3 预训练思考链 <think>...</think>）
- 种子：42（每组同种子，直接对比）
- 指标：平均熵 / 重复率 / 情感命中率
- 三组：
  裸         = 运行基线（纯 generate）
  全面纠正   = echo_common.运行回响（全程池质心注入，现有方案）
  思考链纠正 = 思考链中断注入器（</think> 硬中断 → 定格总体向量 → 正文固定注入，创新方案）
"""
import os
import sys
import json
import time
import torch
from datetime import datetime

回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)
本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

from echo_common import (加载模型, 运行基线, 运行回响, 创建情感过滤器,
                         计算语义熵, 计算重复率, 测试提示词, 清理显存)
from semantic_echo.回响池 import 语义回响池
from 思考链注入器 import 思考链中断注入器

模型路径 = r"l:\模型空间\Qwen3-4B"
量化 = None  # fp16
λ, γ, τ = 0.0588, 0.0845, 0.0592  # 通用注入值（Qwen3≥4B×0.6 × fp16×1.0）
种子基数 = 42
RUNS = 2  # Qwen3 思考链较长，512 token 下耗时翻倍，用 2 轮平衡
输出目录 = r"i:\Desktop\语义回响\实验数据\多模型对照\思考链中断"
MAX_NEW = 512  # 需容纳 思考链(100-300 token) + 正文注入


def 解析参数():
    import argparse
    ap = argparse.ArgumentParser(description="思考链中断注入验证（两级策略第二步）")
    ap.add_argument("模式列表", nargs="*", default=None,
                    help="要跑的模式：裸 / 全面纠正 / 思考链纠正（默认全部）")
    ap.add_argument("--模型", default=模型路径, help="模型路径")
    ap.add_argument("--λ", type=float, default=λ, help="注入强度 λ")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--max_new", type=int, default=MAX_NEW)
    ap.add_argument("--标签", default="", help="输出文件标签")
    return ap.parse_args()


def 渲染提示(tokenizer, 提示):
    """chat template 渲染（enable_thinking 触发 Qwen3 思考链）"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": 提示}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )


def 展开提示词():
    return [{"维度": 维度, "文本": 文本}
            for 维度, 列表 in 测试提示词.items() for 文本 in 列表]


def 汇总均值(值列表):
    if not 值列表:
        return 0.0, 0.0
    均值 = sum(值列表) / len(值列表)
    方差 = sum((x - 均值) ** 2 for x in 值列表) / len(值列表)
    return round(均值, 4), round(方差 ** 0.5, 4)


def 跑裸(model, tokenizer, 提示集, runs, 过滤器=None):
    轮列表 = []
    for run in range(runs):
        base = 种子基数 + run * 100
        for i, 条 in enumerate(提示集):
            torch.manual_seed(base + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(base + i)
            r = 运行基线(model, tokenizer, 渲染提示(tokenizer, 条["文本"]), max_new_tokens=MAX_NEW)
            轮列表.append({"维度": 条["维度"], "熵": r["平均熵"], "重": r["重复率"], "命中": 0.0})
    return 轮列表


def 跑全面(model, tokenizer, 提示集, runs, 过滤器):
    轮列表 = []
    for run in range(runs):
        base = 种子基数 + run * 100
        for i, 条 in enumerate(提示集):
            torch.manual_seed(base + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(base + i)
            r = 运行回响(model, tokenizer, 渲染提示(tokenizer, 条["文本"]), lam=λ, gamma=γ,
                         情感过滤器实例=过滤器, 保留策略="衰减", 滑动窗口=3,
                         max_new_tokens=MAX_NEW, repetition_penalty=1.05)
            池统计 = r.get("池统计") or {}
            命中 = max(0.0, min(1.0, 池统计.get("情感命中率", 0.0)))
            轮列表.append({"维度": 条["维度"], "熵": r["平均熵"], "重": r["重复率"], "命中": 命中})
    return 轮列表


def 跑思考链(model, tokenizer, 提示集, runs, 过滤器):
    轮列表 = []
    # 注入器只创建一次（投影矩阵仅分配一次），每条提示词前重置
    pool = 语义回响池(hidden_dim=model.config.hidden_size, decay_gamma=γ,
                     保留策略="衰减", 滑动窗口大小=3)
    injector = 思考链中断注入器(model, pool, tokenizer, lambda_strength=λ,
                               思考结束token文本="</think>", 思考长度上限=256,
                               情感过滤器实例=过滤器)
    for run in range(runs):
        base = 种子基数 + run * 100
        for i, 条 in enumerate(提示集):
            torch.manual_seed(base + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(base + i)
            injector.重置()
            prompt = 渲染提示(tokenizer, 条["文本"])
            输入ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
            熵列表 = []

            def cb(步, logits):
                熵列表.append(计算语义熵(logits))

            输出ids = injector.生成(
                输入ids, max_new_tokens=MAX_NEW, temperature=1.0, top_p=0.9, top_k=50,
                repetition_penalty=1.05, logits_callback=cb, tokenizer=tokenizer)
            pre_len = 输入ids.shape[1]
            生成ids = 输出ids[0][pre_len:]
            文本 = tokenizer.decode(生成ids, skip_special_tokens=False)
            轮列表.append({
                "维度": 条["维度"], "熵": sum(熵列表) / len(熵列表) if 熵列表 else 0.0,
                "重": 计算重复率(生成ids.tolist()),
                "命中": max(0.0, min(1.0, pool.情感命中率)),
                "思考步数": injector.阶段日志["思考步数"],
                "总体范数": injector.阶段日志["总体范数"],
                "含思考标记": "<think>" in 文本 or "</think>" in 文本,
                "文本预览": 文本[:80].replace("\n", " "),
            })
    del injector
    return 轮列表


def 主流程(args):
    global λ, RUNS, MAX_NEW, 模型路径
    模型路径, λ, RUNS, MAX_NEW = args.模型, args.λ, args.runs, args.max_new
    os.makedirs(输出目录, exist_ok=True)
    print(f"=== 思考链中断验证 | {os.path.basename(模型路径)} | 种子{种子基数} | runs={RUNS} | λ={λ} | max_new={MAX_NEW} ===", flush=True)
    model, tokenizer = 加载模型(模型路径, 量化=量化)
    print(f"  hidden_dim={model.config.hidden_size}", flush=True)
    过滤器 = 创建情感过滤器()
    提示集 = 展开提示词()
    print(f"  提示词 {len(提示集)} 条 × {RUNS} 轮", flush=True)

    记录 = {"模型": os.path.basename(模型路径), "量化": "fp16", "种子": 种子基数,
            "runs": RUNS, "λ": λ, "γ": γ, "τ": τ,
            "prompt": "chat template (enable_thinking)",
            "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    mode_list = args.模式列表 or ["裸", "全面纠正", "思考链纠正"]
    for 模式 in mode_list:
        print(f"  ── [{模式}] ──", flush=True)
        t0 = time.time()
        函数 = {"裸": 跑裸, "全面纠正": 跑全面, "思考链纠正": 跑思考链}[模式]
        轮列表 = 函数(model, tokenizer, 提示集, RUNS, 过滤器)
        熵, 熵std = 汇总均值([x["熵"] for x in 轮列表])
        重, 重std = 汇总均值([x["重"] for x in 轮列表])
        命, 命std = 汇总均值([x["命中"] for x in 轮列表])
        思考步数列表 = [x.get("思考步数") for x in 轮列表 if x.get("思考步数") is not None]
        含标记数 = sum(1 for x in 轮列表 if x.get("含思考标记"))
        print(f"  [{模式}] 熵={熵}(std{熵std}) 重={重}(std{重std}) 命中={命}(std{命std})"
              f" 耗时{time.time()-t0:.0f}s", flush=True)
        if 思考步数列表:
            print(f"   思考链：触发{含标记数}/{len(轮列表)}条，平均思考{sum(思考步数列表)/len(思考步数列表):.0f}步", flush=True)
        记录[模式] = {"平均熵": 熵, "熵std": 熵std, "重复率": 重, "重std": 重std,
                     "情感命中率": 命, "命std": 命std, "每条": 轮列表}

    标签 = args.标签 or f"{os.path.basename(模型路径)}_λ{λ}"
    输出 = os.path.join(输出目录, f"对比_{标签}_{datetime.now().strftime('%H%M%S')}.json")
    with open(输出, "w", encoding="utf-8") as f:
        json.dump(记录, f, ensure_ascii=False, indent=2)
    print(f"  [输出] -> {输出}", flush=True)
    清理显存()
    print("=== 完成 ===")


if __name__ == "__main__":
    主流程(解析参数())
