# -*- coding: utf-8 -*-
import json
d = json.load(open(r'i:\Desktop\语义回响\图灵测试\统一基准\llm_judge_cache_2026.json', encoding='utf-8'))
a = d['AI回复'].get('P6_情感导演', {}).get('0', [])
print(f"P6 生成条数: {len(a)}")
for i, s in enumerate(d['样本'][:10]):
    print(f"#{i+1} {s['user'][:16]} => {a[i][:60]}")
print("...")
# 统计 AI 腔
AI词 = ["AI", "助手", "模型", "语言模型", "作为", "无法", "提供", "回答", "请问", "帮助"]
for i, s in enumerate(d['样本']):
    t = a[i]
    hits = [w for w in AI词 if w in t]
    if hits:
        print(f"  AI腔#{i+1}: {hits} | {t[:40]}")
