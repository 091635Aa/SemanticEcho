# -*- coding: utf-8 -*-
"""
快速开始（Quick Start）—— 体验"锚点回响 Anchor Echo"
=====================================================
小白友好版：加载一个小模型，构建情感锚点（6 维质心 + 稠密余弦打分表），
用「锚点解码器」生成一条带情感方向的回复，并打印 熵/重复率/情感命中率 指标
与只读校验（sum/data_ptr 断言权重零修改）。

环境：Python 3.10+，`pip install torch transformers cnsenti`（见 README.md）
用法：python 快速开始.py
"""
import os
import sys

# 仓库根目录（本文件所在位置）与 核心模块目录
本仓库 = os.path.dirname(os.path.abspath(__file__))
核心目录 = os.path.join(本仓库, "锚点回响核心")
for _p in (本仓库, 核心目录):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 修改这里：你的模型路径（Qwen2.5-1.5B-Instruct 可从 HuggingFace / 魔搭下载）──
模型路径 = r"你的模型路径/Qwen2.5-1.5B-Instruct"
# 例：模型路径 = r"C:\models\Qwen2.5-1.5B-Instruct"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, 自动适配
from 锚点解码器 import 锚点解码器

用户消息 = "我今天真的好累，感觉什么都做不好。"


def 裸生成(model, tokenizer, 提示):
    """裸模式：模型原样采样，不做任何增强（对照用）"""
    输入 = tokenizer(提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            输入.input_ids, max_new_tokens=64,
            temperature=1.0, top_p=0.9, top_k=50,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, 输入.input_ids.shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("=" * 60)
    print("快速开始：锚点回响（Anchor Echo, P4）")
    print("=" * 60)

    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{设备}（RTX 显卡体验最佳）")

    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径,
        torch_dtype=torch.float16 if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    模型.eval()

    # ── ① 构建锚点：6 维情感质心 + 预计算打分表 S ∈ R^{V×K}（只读，不写权重）──
    print("\n[1/3] 构建情感锚点（6 维 × 50 种子词，含流行语：破防/泪目/内耗/扎心）...")
    锚点库实例 = 锚点库(模型, 分词器).构建()
    只读基线 = 锚点库实例.记录只读基线()
    print(f"      锚点维度：{锚点库实例.维度名()}")
    print(f"      只读基线：sum={只读基线['sum']}, data_ptr={只读基线['data_ptr']}")

    # ── ② 目标决策器：VAD→v_target + 自动适配 β（cnsenti 缺失时内置简易 VAD 兜底）──
    print("\n[2/3] 初始化目标决策器（VAD→锚点映射，β 自动适配）...")
    决策器 = 目标决策器(锚点库=锚点库实例)
    β建议 = 自动适配(模型, "fp16")["β"]
    print(f"      自动适配 β = {β建议}（Qwen2.5-1.5B → 0.8）")

    # ── ③ 锚点解码器：稠密打分注入 + 生成循环 + 在线退化兜底 ──
    print("\n[3/3] 锚点解码器生成（β=0.8, T_anchor=0.3, K=6）...")
    解码器 = 锚点解码器(模型, 分词器, 锚点库实例, 决策器,
                     β=β建议, T_anchor=0.3, 稀疏阈值=0.0)

    提示 = 分词器.apply_chat_template(
        [{"role": "user", "content": 用户消息}],
        tokenize=False, add_generation_prompt=True)
    输入 = 分词器(提示, return_tensors="pt").to(设备)

    with torch.no_grad():
        生成ids, 统计 = 解码器.生成(
            输入.input_ids, max_new_tokens=64, 用户文本=用户消息,
            eos_token_id=分词器.eos_token_id)

    锚点回复 = 分词器.decode(生成ids[0, 输入.input_ids.shape[1]:], skip_special_tokens=True).strip()
    裸回复 = 裸生成(模型, 分词器, 提示)

    print(f"\n用户：{用户消息}")
    print(f"\n[裸模式]   {裸回复}")
    print(f"[锚点模式] {锚点回复}")

    print("\n" + "-" * 60)
    print(f"指标：熵={统计['平均熵']:.3f}  重复率={统计['重复率']}  "
          f"情感命中率={统计['情感命中率']:.4f}  β={统计['β']}")
    只读结果 = 锚点库实例.验证只读(只读基线)
    print(f"只读校验：sum 一致={只读结果['sum一致']}  data_ptr 一致={只读结果['指针一致']}  "
          f"→ 权重零修改（sum_before={只读结果['sum_before']} = sum_after={只读结果['sum_after']}）")
    print("=" * 60)
    print("对比两者：锚点模式的回复通常更贴合情绪、更短、更像真人聊天。")
    print("三级降级：锚点解码器(..., 接口='logprobs') 或 接口='提示'（见 接口降级.py）")
    print("三通道叠加：用 混合锚点器.py（锚点β0.8 + 回响λ0.08 + 潮汐倍率6）")
