# -*- coding: utf-8 -*-
"""检查 多模型对照 目录下所有配置 JSON 的完整性与指标"""
import json
import glob
import os

目录 = r"i:\Desktop\语义回响\实验数据\多模型对照"
files = sorted(glob.glob(os.path.join(目录, "*_全部_*.json")))
print(f"共 {len(files)} 个配置 JSON\n")
问题 = 0
for f in files:
    try:
        d = json.load(open(f, encoding="utf-8"))
        汇总 = d.get("汇总_全部模式", {})
        runs = 汇总.get("_runs", d.get("runs"))
        裸 = 汇总.get("裸", {})
        回 = 汇总.get("回响", {})
        # 完整性检查
        missing = []
        for 模式, m in (("裸", 裸), ("回响", 回)):
            for k in ("平均熵", "重复率", "情感命中率"):
                if k not in m:
                    missing.append(f"{模式}.{k}")
        if runs is None:
            missing.append("_runs")
        if not missing:
            print(f"{os.path.basename(f)[:58]:<60} runs={runs} 裸熵={裸.get('平均熵')} 回熵={回.get('平均熵')} 回重={回.get('重复率')} 回命={回.get('情感命中率')}")
        else:
            问题 += 1
            print(f"[缺失] {os.path.basename(f)}: {missing}")
    except Exception as e:
        问题 += 1
        print(f"[错误] {os.path.basename(f)}: {e}")
print(f"\n发现问题数: {问题}")
