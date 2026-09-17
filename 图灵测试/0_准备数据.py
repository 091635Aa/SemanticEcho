# -*- coding: utf-8 -*-
"""
图灵测试 · 数据准备
===================
从 中文高情商对话数据集 抽取 N 条 (user, girl) 真人对话样本，
作为图灵测试的"人类样本"；后续各模型对同一 user 输入生成回复作为"AI 样本"。
"""
import json
import random
import os

数据集路径 = r"c:\Users\Administrator\.cache\huggingface\hub\datasets--sunorme--chinese-adorable-high-emotional-intelligence-chat\snapshots\15f8a4895c7529c16cd8b43bccc95abf4f8b7c6b\chinese-adorable-high-emotional-intelligence-chat.json"
输出路径 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "样本_30条.json")

随机种子 = 42
样本数 = 30

with open(数据集路径, encoding="utf-8") as f:
    data = json.load(f)

# 过滤有效条目：user 与 girl 均非空且不过短
有效 = [
    d for d in data
    if isinstance(d, dict)
    and d.get("user") and d.get("girl")
    and 2 <= len(d["user"]) <= 80
    and 2 <= len(d["girl"]) <= 200
]
print(f"数据集中有效对话: {len(有效)} 条")

random.seed(随机种子)
样本 = random.sample(有效, min(样本数, len(有效)))

记录 = []
for i, d in enumerate(样本, 1):
    记录.append({
        "序号": i,
        "user": d["user"].strip(),
        "girl": d["girl"].strip(),
    })

with open(输出路径, "w", encoding="utf-8") as f:
    json.dump({"总条数": len(记录), "样本": 记录}, f, ensure_ascii=False, indent=2)

print(f"已保存 {len(记录)} 条样本 -> {输出路径}")
for r in 记录[:5]:
    print(f"  [{r['序号']}] {r['user']}  =>  {r['girl']}")
