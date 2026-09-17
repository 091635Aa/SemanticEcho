"""Compute per-dimension semantic entropy averages from experimental data."""
import json
import os
from typing import Dict, List

DATA_DIR = r"d:\Desktop\语义回响\实验数据"

def compute_per_dimension(filepath: str) -> Dict[str, float]:
    """Compute per-dimension average semantic entropy from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    per_dim: Dict[str, List[float]] = {}
    for entry in data.get("数据", []):
        dim = entry.get("维度", "unknown")
        results = entry.get("重复结果", [])
        entropies = []
        for res in results:
            entropy_list = res.get("熵列表", [])
            valid = [e for e in entropy_list if e is not None and e > 0]
            entropies.extend(valid)
        if entropies:
            avg = sum(entropies) / len(entropies)
        else:
            avg = 0.0
        if dim not in per_dim:
            per_dim[dim] = []
        per_dim[dim].append(avg)
    
    result = {}
    for dim, values in per_dim.items():
        result[dim] = sum(values) / len(values) if values else 0.0
    return result

# Files to process
files = {
    "E1": "E1.json",
    "E3": "E3.json",
    "E4": "E4.json",
    "E5": "E5.json",
    "E7": "E7.json",
    "E8": "E8.json",
    "E9": "E9.json",
    "E10": "E10.json",
}

print("=" * 80)
print("Per-dimension average semantic entropy")
print("=" * 80)
print(f"{'Exp':<8} {'开心':<12} {'悲伤':<12} {'愤怒':<12} {'恐惧':<12} {'惊讶':<12} {'中性':<12} {'复杂混合':<12}")
print("-" * 80)

for exp_name, filename in files.items():
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        dim_data = compute_per_dimension(filepath)
        print(f"{exp_name:<8}", end="")
        for d in ["开心", "悲伤", "愤怒", "恐惧", "惊讶", "中性", "复杂混合"]:
            val = dim_data.get(d, 0.0)
            print(f"{val:<12.4f}", end="")
        print()

print("\n" + "=" * 80)
print("Summary statistics from JSON summaries")
print("=" * 80)

# Round 1
with open(os.path.join(DATA_DIR, "实验结果汇总.json"), 'r', encoding='utf-8') as f:
    r1 = json.load(f)
    
print("\nRound 1:")
for exp, info in r1.get("按配置统计", {}).items():
    print(f"  {exp}: {info['描述']}, 平均语义熵={info['平均语义熵']:.4f}, 用时={info.get('总用时(秒)', 'N/A')}s")

# Round 2
with open(os.path.join(DATA_DIR, "实验结果汇总_第二轮.json"), 'r', encoding='utf-8') as f:
    r2 = json.load(f)

print("\nRound 2:")
for exp, info in r2.get("按配置统计", {}).items():
    print(f"  {exp}: {info['描述']}, 平均语义熵={info['平均语义熵']:.4f}, 情感命中率={info.get('平均情感命中率', 'N/A')}, 用时={info.get('总用时(秒)', 'N/A')}s")

print("\n\nDetailed per-dimension for LaTeX table:")
print("-" * 80)
# Print LaTeX table format
print("实验 & 快乐 & 悲伤 & 愤怒 & 恐惧 & 惊讶 \\\\")
print("\\midrule")
for exp_name, filename in files.items():
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        dim_data = compute_per_dimension(filepath)
        vals = [dim_data.get(d, 0.0) for d in ["开心", "悲伤", "愤怒", "恐惧", "惊讶"]]
        print(f"{exp_name} & {vals[0]:.2f} & {vals[1]:.2f} & {vals[2]:.2f} & {vals[3]:.2f} & {vals[4]:.2f} \\\\")
