# -*- coding: utf-8 -*-
"""仅重新生成 P6_情感导演 模式的 LLM-Judge 回复（替换缓存中 AI 腔版本）"""
import os, sys, json
本目录 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 本目录)
from 统一生成器 import 生成器实例

缓存路径 = os.path.join(本目录, "统一基准", "llm_judge_cache_2026.json")
with open(缓存路径, encoding="utf-8") as f:
    缓存 = json.load(f)

样本 = 缓存["样本"]
seed_base = 缓存["seed_base"]
新回复 = []
for i, r in enumerate(样本):
    消息 = [{"role": "user", "content": r["user"]}]
    文本 = 生成器实例.生成("P6_情感导演", 消息, 种子=seed_base, 轮次=i, max_new_tokens=64)
    新回复.append(文本)
    print(f"[P6 {i+1}/30] {r['user'][:16]} => {文本[:40]}", flush=True)

缓存["AI回复"]["P6_情感导演"] = {"0": 新回复}
with open(缓存路径, "w", encoding="utf-8") as f:
    json.dump(缓存, f, ensure_ascii=False, indent=2)
print("P6 回复已更新并保存")

生成器实例.清理()
