# -*- coding: utf-8 -*-
"""
一体化测试运行器 — 多模型 × 量化 × 模式 对照（通用架构二期）
============================================================
- 复用 echo_common：加载模型（fp16/4bit）、裸基座生成、语义回响生成
- 标准提示词集：开心/悲伤/愤怒/中性/复杂混合 各3条（15条）
- 默认 runs=5 轮；若指标波动大（重复率 std>0.08 或熵 std>0.5）自动叠加至 10 轮
- 统一 JSON 记录（全参数），同时支持 JSONL / CSV 格式输出
- 不删除任何文件；输出目录自动创建

用法：
    f:\\打标\\.venv\\Scripts\\python.exe experiments\\一体化测试运行器.py ^
        --模型 l:\\模型空间\\Qwen2.5-0.5B-Instruct --量化 fp16 --模式 全部
"""
import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime

# ── 路径注入 ──
回响工程根 = r"i:\Desktop\语义回响"
for p in (回响工程根,):
    if p not in sys.path:
        sys.path.insert(0, p)
agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from echo_common import (加载模型, 运行基线, 运行回响, 创建情感过滤器,
                         计算语义熵, 计算重复率, 测试提示词, 清理显存)


def 本地加载模型(模型路径, 量化=None, trust_remote_code=True):
    """本地加载（可选关闭 trust_remote_code，规避部分模型远程代码与新版 transformers 不兼容）"""
    kwargs = {"device_map": "cuda:0", "low_cpu_mem_usage": True,
              "trust_remote_code": trust_remote_code}
    if 量化 == "4bit":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.float16)
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(模型路径, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=trust_remote_code)
    return model, tokenizer

# ── λ/γ/τ 推荐（扫描表 + 公式，跨模型通用）──
扫描表 = {896: (0.50, 0.05, 0.10), 1536: (0.08, 0.07, 0.09),
          2048: (0.10, 0.08, 0.06), 3584: (0.06, 0.12, 0.05)}
基准 = 896


def 公式λ(hidden_dim):
    if hidden_dim >= 2048:
        return 0.28 * (基准 / hidden_dim)
    return 0.5 * (基准 / hidden_dim) ** 1.5


def 推荐参数(hidden_dim):
    hidden_dim = int(hidden_dim)
    if hidden_dim in 扫描表:
        λ, γ, τ = 扫描表[hidden_dim]
        return {"λ": float(λ), "γ": float(γ), "τ": float(τ), "来源": "扫描表"}
    return {"λ": round(公式λ(hidden_dim), 4),
            "γ": round(0.05 * (hidden_dim / 基准) ** 0.5, 4),
            "τ": round(0.10 * (基准 / hidden_dim) ** 0.5, 4),
            "来源": "公式"}


# ── 架构族因子（20260806 多模型对照实验实测：Qwen3 系列对回响注入极度敏感）──
def 架构族因子(模型名, hidden_dim):
    """返回 (因子, 族名)。基础 λ × 因子 = 该模型可用的注入强度
    按模型名参数量分段（不能用 hidden_dim：Qwen3-1.7B 与 Qwen3-4B 的 hidden
    均 >1536，但敏感性差异大）。"""
    if "Qwen3" in 模型名:
        # 实测：0.6B(λ0.41→坍缩)、1.7B(λ0.10→坍缩)、4B(λ0.098→有效但偏强)
        if any(k in 模型名 for k in ("0.6", "1.7", "1.5")):
            return (0.3, "Qwen3≤1.7B")
        return (0.6, "Qwen3≥4B")
    if "gemma" in 模型名.lower():
        return (0.7, "Gemma")
    if "SmolLM" in 模型名:
        return (0.5, "SmolLM")
    if "Phi" in 模型名:
        return (0.8, "Phi")
    return (1.0, "Qwen2.5/通用")


def 通用注入参数(模型名, hidden_dim, 量化):
    """通用最大化激活注入值：λ = 基础值(hidden_dim) × 架构族因子 × 量化因子
    γ/τ 沿用基础推荐（池衰减与情感筛选，与架构敏感性弱相关）。"""
    hidden_dim = int(hidden_dim)
    if hidden_dim in 扫描表:
        λ基础, γ基础, τ基础 = 扫描表[hidden_dim]
        来源基础 = "扫描表"
    else:
        λ基础, γ基础, τ基础 = 公式λ(hidden_dim), 0.05 * (hidden_dim / 基准) ** 0.5, 0.10 * (基准 / hidden_dim) ** 0.5
        来源基础 = "公式"
    族因子, 族名 = 架构族因子(模型名, hidden_dim)
    量化因子 = 0.75 if 量化 == "4bit" else 1.0
    λ = λ基础 * 族因子 * 量化因子
    return {"λ": round(λ, 4), "γ": round(float(γ基础), 4), "τ": round(float(τ基础), 4),
            "来源": f"通用架构({族名}×{族因子}, {量化}×{量化因子}, 基础{来源基础})"}


# ── 标准提示词集（维度 → 列表）──
def 展开提示词():
    列表 = []
    for 维度, 提示列表 in 测试提示词.items():
        for 文本 in 提示列表:
            列表.append({"维度": 维度, "文本": 文本})
    return 列表


class 结果记录:
    """单条结果：指标 + 全参数"""

    def __init__(self, 维度, 文本, 平均熵, 重复率, 步数, 情感命中率, 耗时, λ, γ, τ, 来源):
        self.维度 = 维度
        self.文本 = 文本
        self.平均熵 = 平均熵
        self.重复率 = 重复率
        self.步数 = 步数
        self.情感命中率 = 情感命中率
        self.耗时 = 耗时
        self.λ = λ
        self.γ = γ
        self.τ = τ
        self.来源 = 来源

    def to_dict(self):
        return vars(self)


def 跑单条(model, tokenizer, 提示, 模式, λ, γ, τ, 过滤器, max_new_tokens=128):
    t0 = time.time()
    if 模式 == "裸":
        r = 运行基线(model, tokenizer, 提示, max_new_tokens=max_new_tokens)
        命中 = 0.0
        λ_用, γ_用, τ_用 = 0.0, 0.0, 0.0
        来源 = "裸基座"
    else:
        r = 运行回响(model, tokenizer, 提示, lam=λ, gamma=γ,
                     情感过滤器实例=过滤器, 保留策略="衰减", 滑动窗口=3,
                     max_new_tokens=max_new_tokens, repetition_penalty=1.0)
        池统计 = r.get("池统计") or {}
        命中 = max(0.0, min(1.0, 池统计.get("情感命中率", 0.0)))
        λ_用, γ_用, τ_用 = λ, γ, τ
        来源 = "语义回响"
    耗时 = round(time.time() - t0, 2)
    return 结果记录(
        维度="", 文本=r.get("文本", ""), 平均熵=r.get("平均熵", 0.0),
        重复率=r.get("重复率", 0.0), 步数=r.get("步数", 0),
        情感命中率=命中, 耗时=耗时, λ=λ_用, γ=γ_用, τ=τ_用, 来源=来源)


def 均值标准差(值列表):
    if not 值列表:
        return 0.0, 0.0
    均值 = sum(值列表) / len(值列表)
    方差 = sum((x - 均值) ** 2 for x in 值列表) / len(值列表)
    return round(均值, 4), round(方差 ** 0.5, 4)


def 跑配置(model, tokenizer, 提示集, 模式, λ, γ, τ, 过滤器, runs, 上限runs):
    """跑 N 轮（每轮对全部提示词生成一遍），汇总；波动大则自动叠加轮数"""
    轮汇总 = []
    for run_idx in range(runs):
        seed_base = 42 + run_idx * 100
        条列表 = []
        for i, 条 in enumerate(提示集):
            torch.manual_seed(seed_base + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed_base + i)
            记录 = 跑单条(model, tokenizer, 条["文本"], 模式, λ, γ, τ, 过滤器)
            记录.维度 = 条["维度"]
            条列表.append(记录)
        # 本轮均值
        熵 = [x.平均熵 for x in 条列表]
        重 = [x.重复率 for x in 条列表]
        命 = [x.情感命中率 for x in 条列表]
        时 = [x.耗时 for x in 条列表]
        轮汇总.append({
            "run": run_idx + 1, "平均熵": sum(熵) / len(熵),
            "重复率": sum(重) / len(重), "情感命中率": sum(命) / len(命),
            "耗时": sum(时), "每条": [x.to_dict() for x in 条列表],
        })
        print(f"    [run {run_idx+1}/{runs}] 熵={轮汇总[-1]['平均熵']:.4f} "
              f"重复率={轮汇总[-1]['重复率']:.4f} 命中率={轮汇总[-1]['情感命中率']:.4f}", flush=True)
        # 自动叠加：波动大且未达上限
        if runs < 上限runs and run_idx == runs - 1:
            last = 轮汇总[-1]
            熵集 = [x["平均熵"] for x in 轮汇总]
            重集 = [x["重复率"] for x in 轮汇总]
            熵std = (sum((x - sum(熵集)/len(熵集))**2 for x in 熵集)/len(熵集)) ** 0.5
            重std = (sum((x - sum(重集)/len(重集))**2 for x in 重集)/len(重集)) ** 0.5
            if 重std > 0.08 or 熵std > 0.5:
                print(f"    [叠加] 波动较大(重std={重std:.3f},熵std={熵std:.3f})，"
                      f"从 {runs} 轮叠加至 {上限runs} 轮", flush=True)
                runs = 上限runs
    # 汇总
    汇总 = {"_runs": runs}
    for 键 in ("平均熵", "重复率", "情感命中率"):
        均值, std = 均值标准差([x[键] for x in 轮汇总])
        汇总[键] = 均值
        汇总[键 + "_std"] = std
    汇总["平均耗时"] = round(sum(x["耗时"] for x in 轮汇总) / len(轮汇总), 2)
    汇总["轮明细"] = 轮汇总
    return 汇总


def 写输出(记录, 输出目录, 模型名, 量化, 模式, 格式列表):
    os.makedirs(输出目录, exist_ok=True)
    时间戳 = datetime.now().strftime("%Y%m%d_%H%M%S")
    基础名 = f"{模型名}_{量化}_{模式}_{时间戳}"
    json路径 = os.path.join(输出目录, 基础名 + ".json")
    with open(json路径, "w", encoding="utf-8") as f:
        json.dump(记录, f, ensure_ascii=False, indent=2)
    print(f"  [输出] JSON -> {json路径}")
    if "jsonl" in 格式列表:
        jsonl路径 = os.path.join(输出目录, 基础名 + ".jsonl")
        with open(jsonl路径, "w", encoding="utf-8") as f:
            for run in 记录.get("汇总", {}).get("轮明细", []):
                for 条 in run["每条"]:
                    f.write(json.dumps(条, ensure_ascii=False) + "\n")
        print(f"  [输出] JSONL -> {jsonl路径}")
    if "csv" in 格式列表:
        csv路径 = os.path.join(输出目录, 基础名 + ".csv")
        with open(csv路径, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "维度", "平均熵", "重复率", "情感命中率", "步数", "耗时", "λ", "γ", "τ", "来源", "文本"])
            for run in 记录.get("汇总", {}).get("轮明细", []):
                for 条 in run["每条"]:
                    w.writerow([run["run"], 条["维度"], 条["平均熵"], 条["重复率"],
                                条["情感命中率"], 条["步数"], 条["耗时"], 条["λ"],
                                条["γ"], 条["τ"], 条["来源"], 条["文本"].replace("\n", " ")[:120]])
        print(f"  [输出] CSV -> {csv路径}")
    return json路径


def 主流程(args):
    输出目录 = args.输出 or r"i:\Desktop\语义回响\实验数据\多模型对照"
    量化值 = "4bit" if args.量化 == "4bit" else None
    模型名 = os.path.basename(os.path.normpath(args.模型))
    print(f"=== {模型名} | 量化={args.量化} | 模式={args.模式} | runs={args.runs}(上限{args.上限runs}) ===")

    if args.无远程代码:
        model, tokenizer = 本地加载模型(args.模型, 量化=量化值, trust_remote_code=False)
    else:
        model, tokenizer = 加载模型(args.模型, 量化=量化值)
    hidden_dim = int(model.config.hidden_size)
    print(f"  hidden_dim={hidden_dim} 参数={model.num_parameters()/1e6:.0f}M")
    if args.通用注入:
        参数 = 通用注入参数(模型名, hidden_dim, args.量化)
    else:
        参数 = 推荐参数(hidden_dim)
    print(f"  推荐 λ={参数['λ']} γ={参数['γ']} τ={参数['τ']}（{参数['来源']}）")
    过滤器 = 创建情感过滤器()
    提示集 = 展开提示词()
    print(f"  提示词 {len(提示集)} 条（5维度×3）")

    记录 = {
        "模型": 模型名, "模型路径": args.模型, "hidden_dim": hidden_dim,
        "量化": args.量化, "模式": args.模式, "runs": args.runs, "上限runs": args.上限runs,
        "seed_base": 42, "推荐参数": 参数, "提示词数": len(提示集),
        "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    模式列表 = ["裸", "回响"] if args.模式 == "全部" else [args.模式]
    汇总区 = {}
    for 模式 in 模式列表:
        print(f"  ── 模式 [{模式}] ──")
        汇总 = 跑配置(model, tokenizer, 提示集, 模式, 参数["λ"], 参数["γ"], 参数["τ"],
                    过滤器, args.runs, args.上限runs)
        汇总区[模式] = 汇总
        print(f"  [{模式}] 熵={汇总['平均熵']:.4f}(std{汇总['平均熵_std']:.4f}) "
              f"重复率={汇总['重复率']:.4f}(std{汇总['重复率_std']:.4f}) "
              f"命中率={汇总['情感命中率']:.4f} 耗时={汇总['平均耗时']}s", flush=True)
    记录["汇总"] = 汇总区
    记录["汇总_全部模式"] = {m: {k: v for k, v in 汇总区[m].items() if k != "轮明细"} for m in 汇总区}

    写输出(记录, 输出目录, 模型名, args.量化, args.模式, args.格式)
    清理显存()
    print(f"=== {模型名} 完成 ===")
    return 记录


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="一体化多模型对照测试运行器")
    ap.add_argument("--模型", required=True, help="模型路径")
    ap.add_argument("--量化", choices=["fp16", "4bit"], default="fp16")
    ap.add_argument("--模式", choices=["裸", "回响", "全部"], default="全部")
    ap.add_argument("--runs", type=int, default=5, help="默认轮数")
    ap.add_argument("--上限runs", type=int, default=10, help="自动叠加上限轮数")
    ap.add_argument("--格式", nargs="+", choices=["json", "jsonl", "csv"], default=["json"])
    ap.add_argument("--输出", default=None, help="输出目录（默认 实验数据\\多模型对照）")
    ap.add_argument("--无远程代码", action="store_true", help="关闭 trust_remote_code（规避远程模型代码兼容问题）")
    ap.add_argument("--通用注入", action="store_true", help="用通用最大化激活注入值（架构族因子×量化因子）替代扫描表/公式")
    args = ap.parse_args()
    主流程(args)
