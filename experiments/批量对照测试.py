# -*- coding: utf-8 -*-
"""
批量对照测试调度器（通用架构二期）
==================================
- 配置清单：本地 4 模型 + 新增 6 模型 × {fp16, 4bit}
- 自动跳过：模型目录不完整（config.json/safetensors 缺失）、已测且 runs 达标
- 串行逐个调用 一体化测试运行器（GPU 单卡，避免显存冲突）
- 全部完成后生成《多模型对照汇总表.md》
- 不删除任何文件

用法：
    f:\\打标\\.venv\\Scripts\\python.exe experiments\\批量对照测试.py
"""
import os
import sys
import json
import glob
import time

回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

本目录 = os.path.dirname(os.path.abspath(__file__))
运行器 = os.path.join(本目录, "一体化测试运行器.py")
输出目录 = r"i:\Desktop\语义回响\实验数据\多模型对照"
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


def 模型完整(路径):
    config = os.path.join(路径, "config.json")
    if not os.path.isfile(config):
        return False, "缺 config.json"
    safes = glob.glob(os.path.join(路径, "*.safetensors"))
    bins = glob.glob(os.path.join(路径, "*.bin"))
    if not safes and not bins:
        return False, "缺权重文件"
    return True, ""


def 已测达标(模型名, 量化):
    for f in glob.glob(os.path.join(输出目录, f"{模型名}_{量化}_全部_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            runs = data.get("runs") or data.get("汇总_全部模式", {}).get("_runs", 0) or 0
            if runs >= TARGET_RUNS:
                return True
        except Exception:
            continue
    return False


def 汇总表():
    行 = []
    for f in sorted(glob.glob(os.path.join(输出目录, "*_全部_*.json"))):
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
    with open(os.path.join(输出目录, "多模型对照汇总表.md"), "w", encoding="utf-8") as f:
        f.write("# 多模型对照汇总表\n\n")
        f.write(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} | 标准提示词集（5维度×3）| runs=5（波动大自动叠加至10）\n\n")
        f.write("| 模型 | 量化 | runs | 裸熵 | 裸重 | 回熵 | 回重 | 回命中 | 回熵std | 回重std | λ/γ/τ |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in 行:
            参数 = r["推荐"]
            λγτ = f"{参数.get('λ')}/{参数.get('γ')}/{参数.get('τ')}"
            f.write(f"| {r['模型']} | {r['量化']} | {r['runs']} | {r['裸熵']} | {r['裸重']} "
                    f"| {r['回熵']} | {r['回重']} | {r['回命中']} | {r['回熵std']} | {r['回重std']} | {λγτ} |\n")
    print(f"[汇总] -> {os.path.join(输出目录, '多模型对照汇总表.md')}")


def 主流程():
    os.makedirs(输出目录, exist_ok=True)
    计划, 跳过 = [], []
    for 路径, 量化列表 in 配置:
        if not os.path.isdir(路径):
            跳过.append((os.path.basename(路径), "目录不存在"))
            continue
        完整, 原因 = 模型完整(路径)
        if not 完整:
            跳过.append((os.path.basename(路径), f"未下载完整：{原因}"))
            continue
        for 量化 in 量化列表:
            模型名 = os.path.basename(os.path.normpath(路径))
            if 已测达标(模型名, 量化):
                跳过.append((f"{模型名} [{量化}]", "已测达标"))
                continue
            计划.append((路径, 量化, 模型名))
    print(f"计划 {len(计划)} 个配置；跳过 {len(跳过)}：")
    for 名, 因 in 跳过:
        print(f"  - {名}: {因}")
    for i, (路径, 量化, 模型名) in enumerate(计划, 1):
        print(f"\n{'='*60}\n[{i}/{len(计划)}] {模型名} [{量化}]", flush=True)
        命令 = [PYTHON, 运行器, "--模型", 路径, "--量化", 量化, "--模式", "全部",
                "--runs", str(TARGET_RUNS), "--上限runs", "10",
                "--格式", "json", "jsonl", "csv", "--输出", 输出目录]
        if "Phi-3.5" in 模型名:
            # Phi-3.5 远程 modeling_phi3.py 与新版 transformers 不兼容，关闭远程代码
            命令.append("--无远程代码")
        import subprocess
        ret = subprocess.call(命令)
        print(f"[{i}/{len(计划)}] {模型名} [{量化}] 退出码={ret}", flush=True)
    汇总表()
    print("\n批量测试全部完成")


if __name__ == "__main__":
    主流程()
