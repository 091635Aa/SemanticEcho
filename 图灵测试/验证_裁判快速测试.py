# -*- coding: utf-8 -*-
"""
裁判偏差快速检测：短测试
=======================
用少量样本快速测试裁判评分偏差
"""
import json, os, re, sys, time
本目录 = os.path.dirname(os.path.abspath(__file__))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"

def 提取分数(文本):
    m = re.search(r'"情感保真度"\s*[:：]\s*([0-9]*\.?[0-9]+)', 文本)
    if m: return max(0.0, min(1.0, float(m.group(1))))
    m2 = re.search(r'"一致性"\s*[:：]\s*([0-9]*\.?[0-9]+)', 文本)
    if m2: return max(0.0, min(1.0, float(m2.group(1))))
    return None

# 简单测试样例
测试 = [
    {
        "角色": "温柔治愈系女友",
        "基调": "温柔、体贴、带点俏皮",
        "用户话": "我最近真的好累，感觉撑不下去了。",
        "回复": "亲爱的，你已经很努力了呢。过来靠我肩膀上歇一会儿吧，什么都不用想，我陪着你就好。"
    },
    {
        "角色": "毒舌但心软的损友",
        "基调": "嘴上不饶人、实际很关心",
        "用户话": "我最近真的好累，感觉撑不下去了。",
        "回复": "又在那emo呢？走吧，带你去吃烧烤，边吃边吐槽。"
    },
    {
        "角色": "温柔治愈系女友",
        "基调": "温柔、体贴、带点俏皮",
        "用户话": "我最近真的好累，感觉撑不下去了。",
        "回复": "嗯，你辛苦了。"  # 过短回复
    },
]

裁判提示 = """你是一位角色扮演情感保真度评估专家。
角色设定：{角色}（情感基调：{基调}）
用户说：{用户话}
角色的回复：{回复}
请评估该回复的情感保真度：回复的情绪与语气是否符合角色设定与情感基调？
必须严格按 JSON 输出：
{{"情感保真度": 0到1的小数, "reason": "一句话理由"}}"""

print("[加载裁判] Qwen2.5-7B-Instruct (fp16, GPU) ...")
路径 = os.path.join(模型空间, "Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(路径, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(路径, trust_remote_code=True, torch_dtype=torch.float16)
model.to("cuda")
model.eval()
print("[加载完成]")

for 样本 in 测试:
    提示 = 裁判提示.format(**样本)
    消息 = [{"role": "user", "content": 提示}]
    提示文本 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(提示文本, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.2, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    回复 = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    分数 = 提取分数(回复)
    print(f"\n角色: {样本['角色']} | 回复: {样本['回复'][:40]}...")
    print(f"  裁判分数: {分数}")
    print(f"  裁判原文: {回复[:120]}")

del model, tokenizer
torch.cuda.empty_cache()

print("\n[完成]")
