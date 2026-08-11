# -*- coding: utf-8 -*-
"""从 5 大基准原脚本生成 7 模式统一版本（种子 2026）

关键：生成脚本放在原目录（保证 repos/样本 相对路径正确），
输出结果重定向到 统一基准 目录（绝对路径字符串）。
"""
import os
import json

本目录 = os.path.dirname(os.path.abspath(__file__))
统一目录 = os.path.join(本目录, "统一基准")
os.makedirs(统一目录, exist_ok=True)
统一目录字面量 = json.dumps(统一目录)

七模式 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合"]
七模式码 = json.dumps(七模式, ensure_ascii=False)
七模式choices = '", "'.join(七模式)

替换规则 = [
    ("from 生成器 import 生成器实例", "from 统一生成器 import 生成器实例", "生成器入口"),
    ('模式列表 = ["裸", "四层"] if args.模式 == "全部" else [args.模式]',
     '模式列表 = 七模式列表 if args.模式 == "全部" else [args.模式]', "模式列表"),
    ('choices=["裸", "四层", "全部"]', f'choices=["全部", "{七模式choices}"]', "choices"),
]

# 每个基准的输出文件重定向（绝对路径字符串）
结果重定向 = {
    "run_emocharacter.py": {
        'os.path.join(本目录, "data", "emocharacter_results.json")':
            json.dumps(os.path.join(统一目录, "emocharacter_results_2026.json")),
        'os.path.join(本目录, "logs", "EmoCharacter_log.txt")':
            json.dumps(os.path.join(统一目录, "EmoCharacter_log.txt")),
        '会话 = 角色["角色"] if 模式 == "四层" else None': '会话 = None',
    },
    "run_heartbench.py": {
        'os.path.join(本目录, "data", "heartbench_results.json")':
            json.dumps(os.path.join(统一目录, "heartbench_results_2026.json")),
        'os.path.join(本目录, "logs", "HeartBench_log.txt")':
            json.dumps(os.path.join(统一目录, "HeartBench_log.txt")),
    },
    "run_llm_judge.py": {
        'os.path.join(本目录, "data", "llm_judge_results.json")':
            json.dumps(os.path.join(统一目录, "llm_judge_results_2026.json")),
        'os.path.join(本目录, "logs", "LLM_judge_log.txt")':
            json.dumps(os.path.join(统一目录, "LLM_judge_log.txt")),
    },
    "run_turingbench.py": {
        'os.path.join(本目录, "data", "turingbench_results.json")':
            json.dumps(os.path.join(统一目录, "turingbench_results_2026.json")),
        'os.path.join(本目录, "logs", "TuringBench_log.txt")':
            json.dumps(os.path.join(统一目录, "TuringBench_log.txt")),
    },
    "run_feel_heart.py": {
        'os.path.join(本目录, "data", "feel_heart_results.json")':
            json.dumps(os.path.join(统一目录, "feel_heart_results_2026.json")),
        'os.path.join(本目录, "logs", "FEEL_HEART_log.txt")':
            json.dumps(os.path.join(统一目录, "FEEL_HEART_log.txt")),
    },
}


def 补丁(源路径, 目标路径, 额外替换):
    with open(源路径, encoding="utf-8") as f:
        代码 = f.read()
    未命中 = []
    for 旧, 新, 说明 in 替换规则:
        if 旧 in 代码:
            代码 = 代码.replace(旧, 新)
        else:
            未命中.append(说明)
    for 旧, 新 in 额外替换.items():
        if 旧 in 代码:
            代码 = 代码.replace(旧, 新)
        else:
            未命中.append(旧[:40])
    头部补丁 = (
        f"\n# ===== 统一 7 模式补丁（种子 2026） =====\n"
        f"七模式列表 = {七模式码}\n"
        f"# ===== /统一 7 模式补丁 =====\n"
    )
    代码 = 代码.replace('import 公共模块 as cm', 'import 公共模块 as cm\n' + 头部补丁, 1) \
        if 'import 公共模块 as cm' in 代码 else 头部补丁 + 代码
    with open(目标路径, "w", encoding="utf-8") as f:
        f.write(代码)
    return 未命中


if __name__ == "__main__":
    for 脚本, 额外 in 结果重定向.items():
        源 = os.path.join(本目录, 脚本)
        目标 = os.path.join(本目录, "统一_" + 脚本)
        if not os.path.exists(源):
            print(f"⚠ 缺失 {源}")
            continue
        未命中 = 补丁(源, 目标, 额外)
        print(f"✓ {脚本} → {目标} {'（未命中：'+str(未命中)+'）' if 未命中 else ''}")
