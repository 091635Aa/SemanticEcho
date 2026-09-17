import json
import time
import sys
import os

本目录 = r"i:\Desktop\语义回响\图灵测试"
sys.path.insert(0, 本目录)
import run_emocharacter

print("开始 EmoCharacter 多次测试（runs=5）...", flush=True)
start = time.time()
result = run_emocharacter.main()
elapsed = time.time() - start
print(f"测试完成，用时 {elapsed:.1f} 秒", flush=True)

# 读取结果
with open(os.path.join(本目录, "data", "emocharacter_results.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

print("\n=== 多次测试汇总 ===", flush=True)
for 模式 in ["裸", "四层"]:
    汇总 = data["模式汇总"][模式]
    print(f"\n[{模式}]", flush=True)
    print(f"  情感保真度: {汇总['fidelity_score']} (std={汇总.get('fidelity_std', 0)})", flush=True)
    print(f"  跨轮一致性: {汇总['consistency_across_turns']} (std={汇总.get('consistency_std', 0)})", flush=True)
    if "_多次运行明细" in 汇总:
        for d in 汇总["_多次运行明细"]:
            print(f"    run {d['run_idx']+1}: fidelity={d['fidelity_score']}, consistency={d['consistency_across_turns']}", flush=True)
