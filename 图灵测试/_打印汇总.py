# -*- coding: utf-8 -*-
"""打印泛化测试汇总表"""
import json
路径 = r"i:\Desktop\语义回响\图灵测试\统一基准\泛化测试_2026.json"
数据 = json.load(open(路径, encoding="utf-8"))["模式汇总"]
模式列 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响",
          "P4_KV共振", "P5_超融合", "P6_情感导演"]
print("模型 | " + " | ".join(模式列))
print("-" * 160)
for 模型, 模式s in 数据.items():
    行 = []
    for m in 模式列:
        行.append(str(模式s.get(m, {}).get("win_rate_against_human", "-")))
    print(f"{模型} | " + " | ".join(行))
# 最佳模式
print()
for 模型, 模式s in 数据.items():
    最佳 = max(模式s.items(), key=lambda kv: kv[1].get("win_rate_against_human", 0))
    print(f"{模型}: 最佳 {最佳[0]} ({最佳[1]['win_rate_against_human']}) vs 裸 ({模式s.get('裸',{}).get('win_rate_against_human','-')})")
