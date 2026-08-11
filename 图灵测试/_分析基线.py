# -*- coding: utf-8 -*-
"""提取 P1~P5 基线分项与综合分（5 基准平均），供 P6 对比"""
import json, os

统一 = r"i:\Desktop\语义回响\图灵测试\统一基准"
def 加载(p):
    with open(os.path.join(统一, p), encoding="utf-8") as f:
        return json.load(f)

hb = 加载("heartbench_results_2026.json")["模式汇总"]
fh = 加载("feel_heart_results_2026.json")["模式汇总"]
lj = 加载("llm_judge_results_2026.json")["模式汇总"]
tb = 加载("turingbench_results_2026.json")["模式汇总"]
ec = 加载("emocharacter_results_2026.json")["模式汇总"]

模式s = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合", "P6_情感导演"]
print(f"{'模式':<10}{'HeartBench':>10}{'FEEL':>8}{'LLM-Judge':>10}{'Turing':>8}{'EmoChar':>9}{'综合':>8}")
for m in 模式s:
    h = hb.get(m, {}).get("overall_score", 0.0)
    f_ = (fh.get(m, {}).get("accuracy_score", 0.0) +
          fh.get(m, {}).get("empathy_score", 0.0) +
          fh.get(m, {}).get("consistency_score", 0.0)) / 3
    l = (lj.get(m, {}).get("win_rate_against_human", 0.0) +
         lj.get(m, {}).get("average_rating", 0.0)) / 2
    t = tb.get(m, {}).get("human_likeness_score", 0.0)
    e = (ec.get(m, {}).get("匹配fidelity", 0.0) +
         ec.get(m, {}).get("真实一致性", 0.0)) / 2
    综 = (h + f_ + l + t + e) / 5
    print(f"{m:<10}{h:>10.4f}{f_:>8.4f}{l:>10.4f}{t:>8.4f}{e:>9.4f}{综:>8.4f}")
    if m == "P5_超融合":
        print(f"    [P5 细节] FEEL acc={fh['P5_超融合'].get('accuracy_score')} emp={fh['P5_超融合'].get('empathy_score')} con={fh['P5_超融合'].get('consistency_score')}")
        print(f"    [P5 细节] LLM win={lj['P5_超融合'].get('win_rate_against_human')} rating={lj['P5_超融合'].get('average_rating_raw')}")
