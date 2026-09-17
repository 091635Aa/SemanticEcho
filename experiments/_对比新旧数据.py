# -*- coding: utf-8 -*-
"""新旧实验数据对比报告（20260808 全量重跑复验）"""
import json
import glob
import os

旧目录 = r"i:\Desktop\语义回响\实验数据\多模型对照"
新目录 = r"i:\Desktop\语义回响\实验数据\多模型对照_重跑"
旧注入目录 = os.path.join(旧目录, "Qwen3通用注入")
新注入目录 = os.path.join(新目录, "Qwen3通用注入重跑")


def 取最新(目录, 模型, 量化):
    """取指定模型+量化最新一个 *_全部_*.json"""
    files = sorted(glob.glob(os.path.join(目录, f"{模型}_{量化}_全部_*.json")))
    if not files:
        return None
    return json.load(open(files[-1], encoding="utf-8"))


def 指标(d):
    m = d.get("汇总_全部模式", {})
    裸, 回 = m.get("裸", {}), m.get("回响", {})
    return {
        "裸熵": 裸.get("平均熵"), "裸重": 裸.get("重复率"),
        "回熵": 回.get("平均熵"), "回重": 回.get("重复率"),
        "回命中": 回.get("情感命中率"),
        "runs": m.get("_runs"),
    }


def 对比(名称, 旧目录, 新目录, 模型, 量化):
    d旧 = 取最新(旧目录, 模型, 量化)
    d新 = 取最新(新目录, 模型, 量化)
    if d旧 is None or d新 is None:
        print(f"  [缺失] {名称}: 旧={d旧 is not None} 新={d新 is not None}")
        return None
    a, b = 指标(d旧), 指标(d新)
    行 = {"模型": f"{模型}[{量化}]"}
    maxdiff = 0.0
    for k in ("裸熵", "裸重", "回熵", "回重", "回命中"):
        if a[k] is None or b[k] is None:
            diff = None
        else:
            diff = abs(a[k] - b[k])
            if diff is not None:
                maxdiff = max(maxdiff, diff)
        行[k + "旧"] = a[k]
        行[k + "新"] = b[k]
        行["Δ" + k] = diff
    行["maxΔ"] = round(maxdiff, 4)
    return 行


def 主流程():
    配置 = [
        ("Qwen2.5-1.5B-Instruct", "fp16"), ("Qwen2.5-1.5B-Instruct", "4bit"),
        ("Qwen2.5-3B-Instruct", "fp16"), ("Qwen2.5-3B-Instruct", "4bit"),
        ("Qwen2.5-7B-Instruct", "4bit"),
        ("Qwen3-1.7B-Instruct", "fp16"), ("Qwen3-1.7B-Instruct", "4bit"),
        ("Qwen2.5-0.5B-Instruct", "fp16"), ("Qwen2.5-0.5B-Instruct", "4bit"),
        ("Qwen3-0.6B", "fp16"), ("Qwen3-0.6B", "4bit"),
        ("SmolLM2-1.7B-Instruct", "fp16"), ("SmolLM2-1.7B-Instruct", "4bit"),
        ("gemma-2-2b-it", "fp16"), ("gemma-2-2b-it", "4bit"),
        ("Phi-3.5-mini-instruct", "fp16"), ("Phi-3.5-mini-instruct", "4bit"),
        ("Qwen3-4B", "fp16"), ("Qwen3-4B", "4bit"),
    ]
    注入配置 = [("Qwen3-0.6B", "fp16"), ("Qwen3-0.6B", "4bit"),
                ("Qwen3-1.7B-Instruct", "fp16"), ("Qwen3-1.7B-Instruct", "4bit"),
                ("Qwen3-4B", "fp16"), ("Qwen3-4B", "4bit")]

    print("=" * 100)
    print("一、标准扫描表/公式参数（19 配置）新旧对比")
    print("=" * 100)
    行s = []
    for 模型, 量化 in 配置:
        r = 对比("", 旧目录, 新目录, 模型, 量化)
        if r:
            行s.append(r)
    print(f"\n{'模型':<28}{'裸熵旧':<8}{'裸熵新':<8}{'Δ裸熵':<8}{'回熵旧':<8}{'回熵新':<8}{'Δ回熵':<8}{'回重旧':<8}{'回重新':<8}{'Δ回重':<8}{'命中旧':<8}{'命中新':<8}{'maxΔ':<8}")
    print("-" * 100)
    for r in 行s:
        print(f"{r['模型']:<28}{r['裸熵旧']:<8}{r['裸熵新']:<8}{r['Δ裸熵']:<8}{r['回熵旧']:<8}{r['回熵新']:<8}{r['Δ回熵']:<8}{r['回重旧']:<8}{r['回重新']:<8}{r['Δ回重']:<8}{r['回命中旧']:<8}{r['回命中新']:<8}{r['maxΔ']:<8}")
    print(f"\n19 配置中 maxΔ 均值: {sum(r['maxΔ'] for r in 行s)/len(行s):.4f}")
    超差 = [r for r in 行s if r["maxΔ"] > 0.05]
    print(f"超过 0.05 的配置数: {len(超差)}")
    for r in 超差:
        print(f"  ⚠️ {r['模型']}: {r}")

    print("\n" + "=" * 100)
    print("二、Qwen3 通用注入参数（6 配置）新旧对比")
    print("=" * 100)
    行i = []
    for 模型, 量化 in 注入配置:
        r = 对比("", 旧注入目录, 新注入目录, 模型, 量化)
        if r:
            行i.append(r)
    print(f"\n{'模型':<28}{'裸熵旧':<8}{'裸熵新':<8}{'回熵旧':<8}{'回熵新':<8}{'Δ回熵':<8}{'回重旧':<8}{'回重新':<8}{'Δ回重':<8}{'命中旧':<8}{'命中新':<8}{'maxΔ':<8}")
    print("-" * 100)
    for r in 行i:
        print(f"{r['模型']:<28}{r['裸熵旧']:<8}{r['裸熵新']:<8}{r['回熵旧']:<8}{r['回熵新']:<8}{r['Δ回熵']:<8}{r['回重旧']:<8}{r['回重新']:<8}{r['Δ回重']:<8}{r['回命中旧']:<8}{r['回命中新']:<8}{r['maxΔ']:<8}")
    print(f"\n6 配置中 maxΔ 均值: {sum(r['maxΔ'] for r in 行i)/len(行i):.4f}")
    超差i = [r for r in 行i if r["maxΔ"] > 0.05]
    print(f"超过 0.05 的配置数: {len(超差i)}")
    for r in 超差i:
        print(f"  ⚠️ {r['模型']}: {r}")


if __name__ == "__main__":
    主流程()
