# -*- coding: utf-8 -*-
"""提取实验数据 JSON 的关键汇总字段（论文附录A素材）"""
import json, glob, os

def 摘要(path, keys=("平均熵", "重复率", "情感命中率")):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        out = {}
        for k, v in d.items():
            if isinstance(v, dict) and any(x in v for x in ("平均熵", "重复率", "轮明细")):
                row = {x: v.get(x) for x in keys}
                stds = {x + "_std": v.get(x + "_std") for x in keys if v.get(x + "_std") is not None}
                out[k] = {**row, **stds}
        return out
    except Exception as e:
        return {"ERR": str(e)}

# 1. E7-E13
for 名 in ["实验结果汇总_第二轮", "实验结果汇总_保留策略"]:
    p = rf"i:\Desktop\语义回响\实验数据\{名}.json"
    if os.path.exists(p):
        print("=" * 20, 名)
        print(json.dumps(摘要(p), ensure_ascii=False, indent=1)[:1500])

# 2. 思考链中断 4 组
print("=" * 20, "思考链中断")
for f in sorted(glob.glob(r"i:\Desktop\语义回响\实验数据\多模型对照\思考链中断\*.json")):
    try:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        print("-", os.path.basename(f))
        print("  模型:", d.get("模型"), "| λ:", d.get("λ"), "| runs:", d.get("runs"))
        for 模式 in ("裸", "全面纠正", "思考链纠正", "LoRA外挂", "LoRA全面", "LoRA思考链"):
            if 模式 in d:
                v = d[模式]
                print(f"  [{模式}] 熵={v.get('平均熵')} 重={v.get('重复率')} 命中={v.get('情感命中率')}")
    except Exception as e:
        print("-", os.path.basename(f), "ERR", e)
