# -*- coding: utf-8 -*-
"""
融合测试 v2：LoRA 外挂 + 思考链中断注入（验证能否解决外挂语义偏差）
================================================================
背景：外挂（LoRA）+ 回响（全面纠正）在 3B 上退化（熵 -23%，过注入）。
假设：思考链中断方案（思考阶段只捕获不注入 → 总体向量定格 → 正文注入）
      能把思考阶段与外挂叠加的噪声隔离，缓解外挂语义偏差。

v2 修复：
  1) 裸组在 PeftModel 挂载【之前】运行（v1 裸组误带 LoRA，裸=外挂）；
  2) LoRA 适配器改用 enable_thinking=True 训练版（保留 Qwen3 思考链习惯，
     v1 适配器使模型 3 步结束思考且池范数=0，思考链方案失效）；
  3) 思考长度上限调至 512。

模型：Qwen3-4B（fp16）+ 情感 LoRA（emotion_qwen3_4B_think）
四组同种子（42）对比：
  裸             = 纯生成（enable_thinking，无 LoRA）
  LoRA外挂       = PeftModel 纯生成（无回响）
  LoRA+全面纠正  = PeftModel + 运行回响（现有方案，对照"外挂+回响退化"）
  LoRA+思考链    = PeftModel + 思考链中断注入器（新方案）
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

from peft import PeftModel
from echo_common import (加载模型, 运行基线, 运行回响, 创建情感过滤器,
                         计算语义熵, 计算重复率, 测试提示词, 清理显存)
from semantic_echo.回响池 import 语义回响池
from 思考链注入器 import 思考链中断注入器

模型路径 = r"l:\模型空间\Qwen3-4B"
LoRA路径 = r"f:\lora外挂\lora_adapters\emotion_qwen3_4B_think"
量化 = None  # fp16
λ, γ, τ = 0.059, 0.0845, 0.0592  # Qwen3-4B 通用注入值
种子基数 = 42
RUNS = 2
输出目录 = r"i:\Desktop\语义回响\实验数据\多模型对照\思考链中断"
MAX_NEW = 512
思考上限 = 512


def 解析参数():
    import argparse
    ap = argparse.ArgumentParser(description="LoRA 外挂 + 思考链中断注入融合测试 v2")
    ap.add_argument("模式列表", nargs="*", default=None)
    ap.add_argument("--模型", default=模型路径)
    ap.add_argument("--lora", default=LoRA路径)
    ap.add_argument("--λ", type=float, default=λ)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--max_new", type=int, default=MAX_NEW)
    ap.add_argument("--思考上限", type=int, default=思考上限)
    ap.add_argument("--标签", default="")
    return ap.parse_args()


def 渲染提示(tokenizer, 提示):
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


def 跑模式(model, tokenizer, 提示集, runs, 过滤器, 模式, 思考上限=512):
    """按模式执行；model 可为 PeftModel（LoRA 已挂载）
    v3：LoRA思考链 的注入器在函数级创建一次，循环内 重置() 复用，
    避免每次创建投影矩阵(1.45GB)与注册 hook 累积导致 CUDA OOM。"""
    轮列表 = []
    pool = None
    injector = None
    if 模式 == "LoRA思考链":
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

                if 模式 == "裸" or 模式 == "LoRA外挂":
                    r = 运行基线(model, tokenizer, prompt, max_new_tokens=MAX_NEW)
                    轮列表.append({"维度": 条["维度"], "熵": r["平均熵"], "重": r["重复率"], "命中": 0.0})
                elif 模式 == "LoRA全面":
                    r = 运行回响(model, tokenizer, prompt, lam=λ, gamma=γ,
                                 情感过滤器实例=过滤器, 保留策略="衰减", 滑动窗口=3,
                                 max_new_tokens=MAX_NEW, repetition_penalty=1.05)
                    池统计 = r.get("池统计") or {}
                    命中 = max(0.0, min(1.0, 池统计.get("情感命中率", 0.0)))
                    轮列表.append({"维度": 条["维度"], "熵": r["平均熵"], "重": r["重复率"], "命中": 命中})
                elif 模式 == "LoRA思考链":
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
                    # 诊断：打印前 12 个生成 token（判断思考链是否异常短/是否为 KV 断裂）
                    if i == 0 and run == 0:
                        print(f"    [诊断] 前12token: {tokenizer.decode(生成ids[:12], skip_special_tokens=False)!r}", flush=True)
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
    global λ, RUNS, MAX_NEW, 模型路径, LoRA路径, 思考上限
    模型路径, LoRA路径, λ, RUNS, MAX_NEW = args.模型, args.lora, args.λ, args.runs, args.max_new
    思考上限 = args.思考上限
    os.makedirs(输出目录, exist_ok=True)
    print(f"=== 融合测试v2 | {os.path.basename(模型路径)} + {os.path.basename(LoRA路径)} | 种子{种子基数} | runs={RUNS} | λ={λ} | max_new={MAX_NEW} ===", flush=True)
    model, tokenizer = 加载模型(模型路径, 量化=量化)
    print(f"  hidden_dim={model.config.hidden_size}", flush=True)
    过滤器 = 创建情感过滤器()
    提示集 = 展开提示词()
    print(f"  提示词 {len(提示集)} 条 × {RUNS} 轮", flush=True)

    记录 = {"模型": os.path.basename(模型路径), "LoRA": os.path.basename(LoRA路径),
            "量化": "fp16", "种子": 种子基数, "runs": RUNS, "λ": λ, "γ": γ, "τ": τ,
            "思考上限": 思考上限, "prompt": "chat template (enable_thinking)",
            "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    # v3：裸与 LoRA外挂 已在 v2 拿到有效数据，默认只重跑缺失两组
    mode_list = args.模式列表 or ["LoRA全面", "LoRA思考链"]
    # 先跑真裸（此时 model 未挂 LoRA）
    if "裸" in mode_list:
        print(f"  ── [裸]（无 LoRA）──", flush=True)
        t0 = time.time()
        轮列表 = 跑模式(model, tokenizer, 提示集, RUNS, 过滤器, "裸")
        熵, 熵std = 汇总均值([x["熵"] for x in 轮列表])
        重, 重std = 汇总均值([x["重"] for x in 轮列表])
        print(f"  [裸] 熵={熵}(std{熵std}) 重={重}(std{重std}) 耗时{time.time()-t0:.0f}s", flush=True)
        记录["裸"] = {"平均熵": 熵, "熵std": 熵std, "重复率": 重, "重std": 重std,
                     "情感命中率": 0.0, "每条": 轮列表}
    # 挂载 LoRA 后跑其余组
    model = PeftModel.from_pretrained(model, LoRA路径)
    model.eval()
    print(f"  LoRA 已挂载: {LoRA路径}", flush=True)
    for 模式 in [m for m in mode_list if m != "裸"]:
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

    标签 = args.标签 or f"LoRA融合v2_{os.path.basename(模型路径)}_{os.path.basename(LoRA路径)}"
    输出 = os.path.join(输出目录, f"融合_{标签}_{datetime.now().strftime('%H%M%S')}.json")
    with open(输出, "w", encoding="utf-8") as f:
        json.dump(记录, f, ensure_ascii=False, indent=2)
    print(f"  [输出] -> {输出}", flush=True)
    清理显存()
    print("=== 完成 ===")


if __name__ == "__main__":
    主流程(解析参数())
