# -*- coding: utf-8 -*-
"""
全链路验证 · 未注册模型测试：DeepSeek-R1-Distill-Qwen-7B（4bit）
============================================================
两级策略第二步验证：模型名未命中架构族注册表 → 走"兜底"路径：
  ① 保守 λ 回响（通用注入默认值，观察是否坍缩）
  ② 思考链中断方案（R1-Distill 自带 <think> 思考链）兜底
模式（同种子 42）：
  裸         = 纯生成（基线）
  回响       = 运行回响（通用注入参数）
  思考链纠正 = 思考链中断注入器（兜底）
指标：平均熵 / 重复率 / 情感命中率 / 思考步数 / 总体范数
"""
import os
import sys
import json
import math
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
from semantic_echo.采样处理器 import 回响注入器 as _回响注入器
from 思考链注入器 import 思考链中断注入器


# ── GPU 投影补丁：DeepSeek vocab≈152K → 投影矩阵 2.18GB，CPU 分配会 OOM ──
def _GPU初始化投影(self, seed: int) -> None:
    """重写父类：投影矩阵直接在 GPU 分配（与思考链注入器一致）"""
    rng = torch.Generator(device=self.device)
    rng.manual_seed(seed)
    scale = math.sqrt(2.0 / self.hidden_dim)
    self.投影矩阵 = torch.randn(
        self.hidden_dim, self.vocab_size,
        generator=rng, dtype=torch.float32, device=self.device) * scale
    self.投影矩阵.requires_grad_(False)


_回响注入器._初始化投影 = _GPU初始化投影
print("[全链路验证] 已应用 GPU 投影分配补丁（回响注入器）")

模型路径 = r"l:\模型空间\DeepSeek-R1-Distill-Qwen-7B"
量化 = "4bit"  # DeepSeek 7B fp16 14GB 会 OOM，必须 4bit
# 未注册模型兜底参数：架构族因子未命中 → Qwen2.5/通用 1.0 × 4bit 0.75
λ, γ, τ = 0.045, 0.12, 0.05
种子基数 = 42
RUNS = 2
输出目录 = r"i:\Desktop\语义回响\实验数据\全链路验证"
MAX_NEW = 512
思考上限 = 512


def 解析参数():
    import argparse
    ap = argparse.ArgumentParser(description="全链路验证：未注册模型 DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("模式列表", nargs="*", default=None)
    ap.add_argument("--模型", default=模型路径)
    ap.add_argument("--λ", type=float, default=λ)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--max_new", type=int, default=MAX_NEW)
    ap.add_argument("--标签", default="")
    return ap.parse_args()


def 渲染提示(tokenizer, 提示):
    """DeepSeek-R1 系列：apply_chat_template 默认（不支持 enable_thinking 参数）"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": 提示}],
        tokenize=False, add_generation_prompt=True,
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


def 跑模式(model, tokenizer, 提示集, runs, 过滤器, 模式, 思考上限=512):
    """按模式执行；思考链组的注入器函数级创建一次，循环内重置复用（防显存累积）"""
    轮列表 = []
    pool = None
    injector = None
    if 模式 == "思考链纠正":
        pool = 语义回响池(hidden_dim=model.config.hidden_size, decay_gamma=γ,
                         保留策略="衰减", 滑动窗口大小=3)
        injector = 思考链中断注入器(model, pool, tokenizer, lambda_strength=λ,
                                   思考结束token文本="</think>", 思考长度上限=思考上限,
                                   情感过滤器实例=过滤器)
    try:
        for run in range(runs):
            base = 种子基数 + run * 100
            for i, 条 in enumerate(提示集):
                torch.manual_seed(base + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(base + i)
                prompt = 渲染提示(tokenizer, 条["文本"])
                输入ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids

                if 模式 == "裸":
                    r = 运行基线(model, tokenizer, prompt, max_new_tokens=MAX_NEW)
                    轮列表.append({"维度": 条["维度"], "熵": r["平均熵"], "重": r["重复率"], "命中": 0.0})
                elif 模式 == "回响":
                    r = 运行回响(model, tokenizer, prompt, lam=λ, gamma=γ,
                                 情感过滤器实例=过滤器, 保留策略="衰减", 滑动窗口=3,
                                 max_new_tokens=MAX_NEW, repetition_penalty=1.05)
                    池统计 = r.get("池统计") or {}
                    命中 = max(0.0, min(1.0, 池统计.get("情感命中率", 0.0)))
                    轮列表.append({"维度": 条["维度"], "熵": r["平均熵"], "重": r["重复率"], "命中": 命中})
                elif 模式 == "思考链纠正":
                    injector.重置()
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
    finally:
        if injector is not None:
            del injector
    return 轮列表


def 主流程(args):
    global λ, RUNS, MAX_NEW, 模型路径, 思考上限
    模型路径, λ, RUNS, MAX_NEW = args.模型, args.λ, args.runs, args.max_new
    os.makedirs(输出目录, exist_ok=True)
    print(f"=== 全链路验证·未注册模型 | {os.path.basename(模型路径)} | 4bit | 种子{种子基数} | runs={RUNS} | 兜底λ={λ} ===", flush=True)
    model, tokenizer = 加载模型(模型路径, 量化=量化)
    print(f"  hidden_dim={model.config.hidden_size} 参数={model.num_parameters()/1e9:.1f}B", flush=True)
    过滤器 = 创建情感过滤器()
    提示集 = 展开提示词()
    print(f"  提示词 {len(提示集)} 条 × {RUNS} 轮", flush=True)

    记录 = {"模型": os.path.basename(模型路径), "量化": "4bit", "注册状态": "未注册(兜底)",
            "种子": 种子基数, "runs": RUNS, "λ": λ, "γ": γ, "τ": τ,
            "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    # 裸模式已完成（熵 0.7858 重 0.0175），重跑只补 回响 与 思考链纠正
    mode_list = args.模式列表 or ["回响", "思考链纠正"]
    for 模式 in mode_list:
        print(f"  ── [{模式}] ──", flush=True)
        t0 = time.time()
        轮列表 = 跑模式(model, tokenizer, 提示集, RUNS, 过滤器, 模式, 思考上限)
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

    标签 = args.标签 or f"DeepSeek_{os.path.basename(模型路径)}"
    输出 = os.path.join(输出目录, f"全链路_{标签}_{datetime.now().strftime('%H%M%S')}.json")
    with open(输出, "w", encoding="utf-8") as f:
        json.dump(记录, f, ensure_ascii=False, indent=2)
    print(f"  [输出] -> {输出}", flush=True)
    清理显存()
    print("=== 完成 ===")


if __name__ == "__main__":
    主流程(解析参数())
