# -*- coding: utf-8 -*-
import json, glob, os

目录 = r"i:\Desktop\语义回响\实验数据\多模型对照"
f = sorted(glob.glob(os.path.join(目录, "Qwen2.5-0.5B-Instruct_fp16_全部_*.json")))[-1]
d = json.load(open(f, encoding="utf-8"))
print("顶层键:", list(d.keys()))
汇总 = d.get("汇总_全部模式") or {}
print("汇总键:", list(汇总.keys()))
回响 = 汇总.get("回响", {})
print("回响键:", list(回响.keys()))
轮 = 回响.get("轮明细", [])
print("轮数:", len(轮))
if 轮:
    print("轮键:", list(轮[0].keys()))
    每条 = 轮[0].get("每条", [])
    print("每条数:", len(每条))
    if 每条:
        print("条键:", list(每条[0].keys()))
        print("样例:", {k: (str(v)[:60]) for k, v in 每条[0].items()})
