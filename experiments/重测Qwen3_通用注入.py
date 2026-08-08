# -*- coding: utf-8 -*-
"""
Qwen3 全系重测（通用最大化激活注入值）
======================================
背景：Qwen3 系列（0.6B/1.7B）在 Qwen2.5 扫描表 λ 下坍缩（重复率 0.86~0.92）。
本脚本用"通用注入值"（λ = 基础值 × 架构族因子 Qwen3×0.3/0.6 × 量化因子 4bit×0.75）重测，
验证通用激活方法能否把 Qwen3 从坍缩拉回有效区间。
输出到独立目录 Qwen3通用注入\\，与旧结果对比。
"""
import os
import sys
import subprocess

回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

本目录 = os.path.dirname(os.path.abspath(__file__))
运行器 = os.path.join(本目录, "一体化测试运行器.py")
输出目录 = r"i:\Desktop\语义回响\实验数据\多模型对照\Qwen3通用注入"
PYTHON = r"f:\打标\.venv\Scripts\python.exe"
模型空间_L = r"l:\模型空间"

# Qwen3 全系
配置 = [
    (os.path.join(模型空间_L, "Qwen3-0.6B"), ["fp16", "4bit"]),
    (os.path.join(模型空间_L, "Qwen3-1.7B-Instruct"), ["fp16", "4bit"]),
    (os.path.join(模型空间_L, "Qwen3-4B"), ["fp16", "4bit"]),
]

# 注意：Qwen3-1.7B-Instruct 在 模型空间_本地（c:\...\模型空间\Qwen3-1.7B-Instruct）
本地空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
配置[1] = (os.path.join(本地空间, "Qwen3-1.7B-Instruct"), ["fp16", "4bit"])


def 主流程():
    os.makedirs(输出目录, exist_ok=True)
    for i, (路径, 量化列表) in enumerate(配置, 1):
        if not os.path.isdir(路径):
            print(f"跳过：{路径} 不存在", flush=True)
            continue
        for 量化 in 量化列表:
            print(f"\n{'='*60}\n[{i}] {os.path.basename(路径)} [{量化}] 通用注入", flush=True)
            命令 = [PYTHON, 运行器, "--模型", 路径, "--量化", 量化, "--模式", "全部",
                    "--runs", "5", "--上限runs", "10",
                    "--格式", "json", "jsonl", "csv", "--输出", 输出目录,
                    "--通用注入"]
            ret = subprocess.call(命令)
            print(f"  退出码={ret}", flush=True)
    print("\nQwen3 通用注入重测完成")


if __name__ == "__main__":
    主流程()
