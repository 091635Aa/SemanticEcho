"""
快速运行 E1 基线实验

仅对前 2 个情感维度（共 3 条提示词/维度 = 6 条提示词）运行基线生成，
重复次数设为 1 以快速出结果。
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# 切换到项目目录
os.chdir(Path(__file__).parent)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from 实验运行器 import (
    实验配置,
    实验运行器,
    # 汇总统计器,   # 本脚本暂不使用
    测试提示词,
)


def main() -> None:
    """E1 基线实验主入口"""
    # ── 加载模型 ──
    本地路径 = os.path.join(os.path.dirname(__file__), "本地模型")
    if os.path.exists(本地路径):
        model_path = 本地路径
        print(f"从本地加载模型: {model_path}")
    else:
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"从 HuggingFace 加载模型: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print(f"模型加载成功，参数: {model.num_parameters() / 1e6:.1f}M")

    # ── 创建实验运行器 ──
    print("\n创建实验运行器...")
    runner = 实验运行器(model, tokenizer, 输出目录="./实验数据")
    print(f"输出目录: {runner.输出目录}")

    # ── E1 配置：Baseline (top_p=0.9) ──
    配置 = 实验配置(
        实验编号="E1",
        条件描述="Baseline (top_p=0.9)",
        lambda_strength=None,
        decay_gamma=None,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        max_new_tokens=128,
        重复次数=1,  # 快速验证，只跑 1 次
    )

    # ── 运行实验 ──
    # 只取前 2 个情感维度（共 6 条提示词）快速验证
    所有结果: list[dict] = []
    for 维度, 提示词列表 in list(测试提示词.items())[:2]:
        print(f"\n{'='*50}")
        print(f"维度: {维度} ({len(提示词列表)} 条提示词)")
        print(f"{'='*50}")
        for 提示词 in 提示词列表:
            print(f"\n--- 提示词: 「{提示词}」 ---")
            try:
                结果 = runner.运行单次生成(提示词, 配置, 是回响=False)
                eval_obj = 结果["评估器"]
                # 使用实际 API：平均熵 是 property，熵序列 是 property
                平均熵 = eval_obj.平均熵
                熵列表 = eval_obj.熵序列
                生成文本 = 结果["文本"]

                所有结果.append({
                    "维度": 维度,
                    "提示词": 提示词,
                    "文本": 生成文本,
                    "平均熵": 平均熵,
                    "熵列表": 熵列表,
                })
                # 截断过长输出
                preview = 生成文本[:60].replace("\n", " ")
                print(f"  输出: {preview}...")
                print(f"  平均熵: {平均熵:.4f}")

            except Exception as e:
                print(f"  [错误] {type(e).__name__}: {e}")

    # ── 保存 E1 结果 ──
    输出 = {
        "实验编号": "E1",
        "条件": "Baseline (top_p=0.9)",
        "温度": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 128,
        "重复次数": 1,
        "时间": datetime.now().isoformat(),
        "结果": 所有结果,
    }

    输出路径 = os.path.join(".", "实验数据", "E1_基线结果.json")
    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump(输出, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"E1 基线实验完成！")
    print(f"共运行 {len(所有结果)} 条提示词")
    print(f"结果保存至: {输出路径}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
