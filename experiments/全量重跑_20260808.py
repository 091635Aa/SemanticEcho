# -*- coding: utf-8 -*-
"""
全量重跑调度器（20260808 数据复验）
===================================
- 10 模型 × {fp16, 4bit} = 19 配置全部强制重跑（不跳过任何已测数据）
- Qwen3 系列（0.6B/1.7B/4B）额外用 --通用注入 参数再跑一遍（两套对照）
- 输出到新目录 i:\Desktop\语义回响\实验数据\多模型对照_重跑\（旧数据保留）
- 串行逐个调用 一体化测试运行器（GPU 单卡）
- 完成后生成《多模型对照汇总表_重跑.md》
"""
import os
import sys
import json
import glob
import time
import subprocess

回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

本目录 = os.path.dirname(os.path.abspath(__file__))
运行器 = os.path.join(本目录, "一体化测试运行器.py")
输出目录 = r"i:\Desktop\语义回响\实验数据\多模型对照_重跑"
通用注入目录 = os.path.join(输出目录, "Qwen3通用注入重跑")
PYTHON = r"f:\打标\.venv\Scripts\python.exe"

模型空间_本地 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
模型空间_L = r"l:\模型空间"

# (路径, 量化列表)
配置 = [
    (os.path.join(模型空间_本地, "Qwen2.5-1.5B-Instruct"), ["fp16", "4bit"]),
    (os.path.join(模型空间_本地, "Qwen2.5-3B-Instruct"), ["fp16", "4bit"]),
    (os.path.join(模型空间_本地, "Qwen2.5-7B-Instruct"), ["4bit"]),   # fp16 14GB 易 OOM
    (os.path.join(模型空间_本地, "Qwen3-1.7B-Instruct"), ["fp16", "4bit"]),
    (os.path.join(模型空间_L, "Qwen2.5-0.5B-Instruct"), ["fp16", "4bit"]),
    (os.path.join(模型空间_L, "Qwen3-0.6B"), ["fp16", "4bit"]),
    (os.path.join(模型空间_L, "SmolLM2-1.7B-Instruct"), ["fp16", "4bit"]),
    (os.path.join(模型空间_L, "gemma-2-2b-it"), ["fp16", "4bit"]),
    (os.path.join(模型空间_L, "Phi-3.5-mini-instruct"), ["fp16", "4bit"]),
    (os.path.join(模型空间_L, "Qwen3-4B"), ["fp16", "4bit"]),
]

TARGET_RUNS = 5


def 构建命令(路径, 量化, 输出, 通用注入=False):
    模型名 = os.path.basename(os.path.normpath(路径))
    命令 = [PYTHON, 运行器, "--模型", 路径, "--量化", 量化, "--模式", "全部",
            "--runs", str(TARGET_RUNS), "--上限runs", "10",
            "--格式", "json", "jsonl", "csv", "--输出", 输出]
    if 通用注入:
        命令.append("--通用注入")
    if "Phi-3.5" in 模型名:
        # Phi-3.5 远程 modeling_phi3.py 与新版 transformers 不兼容，关闭远程代码
        命令.append("--无远程代码")
    return 命令


def 汇总表(目录, 输出md):
    os.makedirs(目录, exist_ok=True)
    行 = []
    for f in sorted(glob.glob(os.path.join(目录, "*_全部_*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            m = data.get("汇总_全部模式", {})
            裸 = m.get("裸", {})
            回 = m.get("回响", {})
            行.append({
                "模型": data.get("模型"), "量化": data.get("量化"),
                "runs": m.get("_runs", ""),
                "裸熵": 裸.get("平均熵"), "裸重": 裸.get("重复率"),
                "回熵": 回.get("平均熵"), "回重": 回.get("重复率"),
                "回命中": 回.get("情感命中率"),
                "回熵std": 回.get("平均熵_std"), "回重std": 回.get("重复率_std"),
                "推荐": data.get("推荐参数", {}),
            })
        except Exception:
            continue
    行.sort(key=lambda x: (str(x["模型"]), str(x["量化"])))
    with open(输出md, "w", encoding="utf-8") as f:
        f.write("# 多模型对照汇总表（重跑）\n\n")
        f.write(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} | 标准提示词集（5维度×3）| runs={TARGET_RUNS}（波动大自动叠加至10）\n\n")
        f.write("| 模型 | 量化 | runs | 裸熵 | 裸重 | 回熵 | 回重 | 回命中 | 回熵std | 回重std | λ/γ/τ |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in 行:
            参数 = r["推荐"]
            λγτ = f"{参数.get('λ')}/{参数.get('γ')}/{参数.get('τ')}"
            f.write(f"| {r['模型']} | {r['量化']} | {r['runs']} | {r['裸熵']} | {r['裸重']} "
                    f"| {r['回熵']} | {r['回重']} | {r['回命中']} | {r['回熵std']} | {r['回重std']} | {λγτ} |\n")
    print(f"[汇总] -> {输出md}")


def 主流程():
    os.makedirs(输出目录, exist_ok=True)
    os.makedirs(通用注入目录, exist_ok=True)

    # 组装任务
    任务 = []  # (路径, 量化, 输出目录, 是否通用注入)
    for 路径, 量化列表 in 配置:
        if not os.path.isdir(路径):
            print(f"[跳过] {路径} 目录不存在")
            continue
        for 量化 in 量化列表:
            任务.append((路径, 量化, 输出目录, False))
            if "Qwen3" in os.path.basename(os.path.normpath(路径)):
                # Qwen3 系列额外跑一套通用注入参数
                任务.append((路径, 量化, 通用注入目录, True))

    print(f"共 {len(任务)} 个任务")
    for i, (路径, 量化, 输出, 通用注入) in enumerate(任务, 1):
        模型名 = os.path.basename(os.path.normpath(路径))
        标记 = " [通用注入]" if 通用注入 else ""
        print(f"\n{'='*60}\n[{i}/{len(任务)}] {模型名} [{量化}]{标记}", flush=True)
        命令 = 构建命令(路径, 量化, 输出, 通用注入)
        ret = subprocess.call(命令)
        print(f"[{i}/{len(任务)}] {模型名} [{量化}]{标记} 退出码={ret}", flush=True)

    汇总表(输出目录, os.path.join(输出目录, "多模型对照汇总表_重跑.md"))
    汇总表(通用注入目录, os.path.join(通用注入目录, "Qwen3通用注入汇总表_重跑.md"))
    print("\n全量重跑完成")


if __name__ == "__main__":
    主流程()
