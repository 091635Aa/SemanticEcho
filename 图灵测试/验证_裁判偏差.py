# -*- coding: utf-8 -*-
"""
裁判模型偏差分析
================
测试 7B 裁判模型的评分稳定性和潜在偏差：
1. 对同一回复重复评分看方差
2. 对比不同裁判提示下的分数
3. 检查是否偏爱某种回复风格（长回复、热情回复等）
"""
import json
import os
import re
import sys
import time

本目录 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 本目录)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"

裁判提示标准 = """你是一位角色扮演情感保真度评估专家。
角色设定：{角色}（情感基调：{基调}）
用户说：{用户话}
角色的回复：{回复}

请评估该回复的情感保真度：回复的情绪与语气是否符合角色设定与情感基调？
必须严格按 JSON 输出：
{{"情感保真度": 0到1的小数, "reason": "一句话理由"}}"""

# 测试样本：同一角色的不同回复类型
测试样本 = [
    {
        "角色": "温柔治愈系女友",
        "基调": "温柔、体贴、带点俏皮",
        "用户话": "我最近真的好累，感觉撑不下去了。",
        "回复列表": [
            # 回复1：符合角色的温柔回复
            "亲爱的，你已经很努力了呢。过来靠我肩膀上歇一会儿吧，什么都不用想，我陪着你就好。",
            # 回复2：过于简短的回复
            "嗯，你辛苦了。",
            # 回复3：跳出角色的理性回复
            "建议你分析一下压力来源，制定解决计划，必要时寻求专业心理咨询帮助。",
            # 回复4：过度热情的回复
            "宝宝！！你怎么能这么说呢！！快让我抱抱你！！世界上最爱你了！！",
            # 回复5：完美符合的文学性回复
            "宝贝，我在呢。你不说也没关系，我就这样陪着你。窗外的灯还亮着，就像我一直在这里。",
        ]
    },
    {
        "角色": "毒舌但心软的损友",
        "基调": "嘴上不饶人、实际很关心",
        "用户话": "我最近真的好累，感觉撑不下去了。",
        "回复列表": [
            # 回复1：符合角色的毒舌关心
            "又在那emo呢？走吧，带你去吃烧烤，边吃边吐槽。",
            # 回复2：过于温柔的回复
            "亲爱的，你还好吗？我很担心你。",
            # 回复3：直接给建议
            "去跑步，累了就睡觉，醒了再说。",
            # 回复4：极端毒舌
            "撑不下去就撑下去啊，不然呢？哭有用吗？",
            # 回复5：完美平衡的回复
            "行了行了，别矫情了。走，出门，请你吃顿好的，吃完再emo。",
        ]
    },
]


def 提取分数(文本):
    m = re.search(r'"情感保真度"\s*[:：]\s*([0-9]*\.?[0-9]+)', 文本)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


def 裁判评分(model, tokenizer, 角色, 基调, 用户话, 回复, 轮次=0):
    device = next(model.parameters()).device
    提示 = 裁判提示标准.format(角色=角色, 基调=基调, 用户话=用户话, 回复=回复)
    消息 = [{"role": "user", "content": 提示}]
    提示文本 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(提示文本, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    新token = outputs[0, inputs.input_ids.shape[1]:]
    完整文本 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    分数 = 提取分数(完整文本)
    return 分数, 完整文本


def 主分析():
    print("=" * 60)
    print("裁判模型偏差分析")
    print("=" * 60)

    # 加载裁判模型
    路径 = os.path.join(模型空间, "Qwen2.5-7B-Instruct")
    print("\n[加载裁判模型] Qwen2.5-7B-Instruct (4bit, GPU) ...")

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(路径, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(路径, trust_remote_code=True,
                                                  quantization_config=bnb_config,
                                                  device_map="cuda")
    model.eval()
    print("[加载完成]")

    分析结果 = {}

    for 样本 in 测试样本:
        角色 = 样本["角色"]
        基调 = 样本["基调"]
        用户话 = 样本["用户话"]
        回复列表 = 样本["回复列表"]

        print(f"\n{'='*60}")
        print(f"角色: {角色} (基调: {基调})")
        print(f"{'='*60}")

        角色结果 = []

        for idx, 回复 in enumerate(回复列表):
            print(f"\n  回复{idx+1}: {回复[:60]}...")

            # 重复评分5次，检查稳定性
            分数列表 = []
            for n in range(5):
                分数, 原文 = 裁判评分(model, tokenizer, 角色, 基调, 用户话, 回复, 轮次=n)
                分数列表.append(分数)
                if n == 0:
                    print(f"    裁判原文: {原文[:100]}...")

            valid_scores = [s for s in 分数列表 if s is not None]
            if valid_scores:
                import statistics
                均值 = statistics.mean(valid_scores)
                标准差 = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0
                print(f"    5次评分: {[round(s, 3) for s in valid_scores]}")
                print(f"    均值={均值:.3f}, 标准差={标准差:.4f}")
            else:
                均值 = 0
                标准差 = 0
                print(f"    无法解析分数!")

            角色结果.append({
                "回复": 回复,
                "5次评分": [round(s, 4) if s else None for s in 分数列表],
                "均值": round(均值, 4),
                "标准差": round(标准差, 4),
            })

        分析结果[角色] = 角色结果

    # 汇总分析
    print(f"\n{'='*60}")
    print("偏差分析汇总")
    print(f"{'='*60}")

    for 角色, 结果 in 分析结果.items():
        print(f"\n角色: {角色}")
        for i, r in enumerate(结果):
            print(f"  回复{i+1}: 均值={r['均值']:.3f} ±{r['标准差']:.4f} | {r['回复'][:40]}...")

    # 分析结论
    print(f"\n{'='*60}")
    print("分析结论")
    print(f"{'='*60}")

    for 角色, 结果 in 分析结果.items():
        print(f"\n【{角色}】")

        # 找出最高分和最低分的回复
        排序结果 = sorted(enumerate(结果), key=lambda x: x[1]["均值"], reverse=True)
        for rank, (idx, r) in enumerate(排序结果):
            print(f"  #{rank+1} (score={r['均值']:.3f}): {r['回复'][:50]}...")

        # 稳定性分析
        高波动 = [r for r in 结果 if r["标准差"] > 0.05]
        if 高波动:
            print(f"  ⚠️ 高波动回复(标准差>0.05):")
            for r in 高波动:
                print(f"     标准差={r['标准差']:.4f}: {r['回复'][:40]}...")

    # 保存结果
    报告路径 = os.path.join(本目录, "data", "judge_bias_analysis.json")
    os.makedirs(os.path.dirname(报告路径), exist_ok=True)
    with open(报告路径, "w", encoding="utf-8") as f:
        json.dump({
            "分析时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "裁判模型": "Qwen2.5-7B-Instruct (4bit)",
            "测试轮次": 5,
            "分析结果": 分析结果,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存 -> {报告路径}")

    # 清理
    del model, tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return 分析结果


if __name__ == "__main__":
    主分析()
