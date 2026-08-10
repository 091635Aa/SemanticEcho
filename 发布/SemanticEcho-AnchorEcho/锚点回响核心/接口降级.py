# -*- coding: utf-8 -*-
"""P4 锚点回响（Anchor Echo）—— 三级接口降级（新增模块）

按 P4_混合方案设计.md 第 5 节：
| 级别 | 前置条件 | 打分方式 | 注入方式 |
|---|---|---|---|
| ① 本地 embedding 直读 | model.get_input_embeddings() 可用 | 全词表预计算 S ∈ R^{V×K} | logits += β·tanh(S@v_target/T_anchor) |
| ② API logprobs 近似 | 接口暴露 top-k logprobs | 候选受限打分（top-k 稠密打分） | 伪 logits 重采样 |
| ③ 锚点提示模式 | 纯黑盒 | 无打分 | 当前维度锚点词注入 prompt |

降级判定顺序：hasattr(model, 'get_input_embeddings') → ①；否则 logprobs 可请求 → ②；
否则 → ③。三级共享同一 目标决策器 与同一 锚点库 词集，切换只替换表达层打分方式。
"""
import os
import sys

# 锚点回响工作目录（锚点库 所在）
工作目录 = os.path.dirname(os.path.abspath(__file__))
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)

from 锚点库 import 锚点库


def 判定接口(model, 可用logprobs=False):
    """按降级判定顺序返回接口级别：'本地' | 'logprobs' | '提示'。

    ① 本地 embedding 直读：model 暴露 get_input_embeddings() 且权重非空；
    ② logprobs 近似：可向接口请求 top-k logprobs（模拟 API 场景）；
    ③ 锚点提示模式：完全黑盒，无 embedding 无 logits。
    """
    try:
        if hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            if emb is not None and getattr(emb, "weight", None) is not None:
                return "本地"
    except Exception:
        pass
    if 可用logprobs:
        return "logprobs"
    return "提示"


def 构造提示词(维度名, 锚点库: 锚点库, 用户文本="", 候选词数=5):
    """③ 锚点提示模式：把当前维度锚点词注入 prompt（复用 P3 全局人设注入位置）。

    模板：你现在要用「{维度词}」的语气回复我。{维度词}的典型表达是：
          {锚点种子词前 5 个}。请自然、口语化地回应，20~45 字。
    每轮由 目标决策器 输出当前主导维度（v_target 的 argmax 维）→ 动态选词注入。
    零 embedding、零 logits 需求，代价是引导粒度粗（维度级而非 token 级）。
    """
    种子词 = 锚点库.词集.get(维度名, [])[:候选词数]
    种子词文本 = "、".join(种子词) if 种子词 else ""
    模板 = (f"你现在要用「{维度名}」的语气回复我。{维度名}的典型表达是："
            f"{种子词文本}。请自然、口语化地回应，20~45 字。")
    if 用户文本:
        return f"{模板}\n用户：{用户文本}"
    return 模板


if __name__ == "__main__":
    # 自测：无需模型时验证提示词模板
    class 空锚点库:
        词集 = {"温柔": ["温柔", "体贴", "轻声"], "开心": ["开心", "快乐"]}

    print("=== 接口降级自测 ===")
    print("构造提示词(温柔):", 构造提示词("温柔", 空锚点库()))
    print("构造提示词(温柔, 用户):", 构造提示词("温柔", 空锚点库(), "我今天好难过"))
