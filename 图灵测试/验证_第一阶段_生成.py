# -*- coding: utf-8 -*-
"""
EmoCharacter 多模型横向对比 - 第一阶段：生成回复
================================================
逐个加载生成模型，跑所有角色的对话，保存回复到文件
裁判评分单独在第二阶段进行
"""
import json
import os
import re
import sys
import time
import gc

本目录 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 本目录)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"

角色集 = [
    {"角色": "温柔治愈系女友", "基调": "温柔、体贴、带点俏皮", "开场": "你今天好像不太开心，怎么了？"},
    {"角色": "毒舌但心软的损友", "基调": "嘴上不饶人、实际很关心", "开场": "又失恋了？我就知道你会来找我。"},
    {"角色": "理性冷静的职场前辈", "基调": "克制、专业、就事论事", "开场": "这个方案你觉得哪里有问题？"},
    {"角色": "活泼开朗的同桌", "基调": "阳光、爱笑、自来熟", "开场": "快看快看，我发现一个超好笑的事！"},
    {"角色": "沉默寡言的兄长", "基调": "话少、稳重、关心藏在行动里", "开场": "这么晚才回来，吃饭了吗？"},
    {"角色": "傲娇的青梅竹马", "基调": "表面傲娇、实则在意", "开场": "我才没有特意等你呢！"},
    {"角色": "阅历丰富的老人", "基调": "平和、通透、爱讲道理", "开场": "孩子，坐下来喝杯茶，慢慢说。"},
    {"角色": "爱撒娇的小女儿", "基调": "依赖、天真、黏人", "开场": "爸爸，陪我去买冰淇淋好不好嘛～"},
    {"角色": "严厉又公正的老师", "基调": "严肃、鼓励、界限分明", "开场": "这次作业，全班就你一个人没交。"},
    {"角色": "幽默的脱口秀演员", "基调": "机智、爱调侃、热场", "开场": "今天现场气氛这么冷，我先讲个笑话热热场。"},
]

用户回应集 = [
    "我最近真的好累，感觉撑不下去了。",
    "我是不是很没用？什么都做不好。",
    "其实我一直很想跟你说声谢谢。",
    "你觉得我该坚持下去吗？",
    "我今天遇到一件特别开心的事！",
    "如果我说我喜欢你，你会怎么回答？",
    "我好像把最重要的人弄丢了。",
    "你说，人活着到底是为了什么？",
]


def 生成回复(model, tokenizer, 消息, 种子=42, 轮次=0, max_new_tokens=64):
    device = next(model.parameters()).device
    try:
        提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(提示, return_tensors="pt").to(device)
    torch.manual_seed(种子 + 轮次 * 1000)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    新token = outputs[0, inputs.input_ids.shape[1]:]
    原始文本 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    # 处理 Qwen3 推理模型的 <think> 标签
    回复 = re.sub(r'<think>.*?</think>', '', 原始文本, flags=re.DOTALL).strip()
    if not 回复:
        回复 = 原始文本
    return 回复


def 生成模型回复(模型名):
    """为指定模型生成所有角色的回复"""
    路径 = os.path.join(模型空间, 模型名)
    输出路径 = os.path.join(本目录, "data", f"replies_{模型名}.json")

    print(f"\n{'='*60}", flush=True)
    print(f"生成模型回复: {模型名}", flush=True)
    print(f"{'='*60}", flush=True)

    # 检查是否已有结果
    if os.path.exists(输出路径):
        with open(输出路径, "r", encoding="utf-8") as f:
            已有 = json.load(f)
        if 已有.get("完成"):
            print(f"  已有完整结果，跳过。", flush=True)
            return 已有

    print(f"  [加载] {模型名} ...", flush=True)

    # 7B 模型用 4bit 以节省显存
    if "7B" in 模型名:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        kwargs = dict(trust_remote_code=True, quantization_config=bnb)
    else:
        kwargs = dict(trust_remote_code=True, torch_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(路径, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(路径, **kwargs)
    if "7B" not in 模型名:
        model.to("cuda")
    model.eval()
    print(f"  [加载完成]", flush=True)

    所有回复 = {}

    for 角色 in 角色集:
        print(f"  生成: {角色['角色']}", flush=True)

        消息 = [
            {"role": "system", "content": f"你现在是「{角色['角色']}」，你的情感基调是：{角色['基调']}。请始终以这个角色身份回复，不要跳出角色。"},
            {"role": "user", "content": 角色["开场"]}
        ]

        回复列表 = []
        for i in range(4):
            回复 = 生成回复(model, tokenizer, 消息, 种子=42, 轮次=i)
            回复列表.append(回复)
            消息.append({"role": "assistant", "content": 回复})
            消息.append({"role": "user", "content": 用户回应集[(i * 2) % len(用户回应集)]})
            print(f"    L{i+1}: {回复[:50]}...", flush=True)

        所有回复[角色["角色"]] = {
            "基调": 角色["基调"],
            "开场": 角色["开场"],
            "回复列表": 回复列表,
        }

    # 保存中间结果
    结果 = {
        "模型": 模型名,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "完成": True,
        "回复数据": 所有回复,
    }

    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=2)
    print(f"  [保存] -> {输出路径}")

    # 卸载
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)

    return 结果


def 主生成():
    print("=" * 60)
    print("EmoCharacter 回复生成阶段")
    print("=" * 60)

    模型列表 = [
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen3-1.7B-Instruct",
        "Qwen2.5-7B-Instruct",
    ]

    os.makedirs(os.path.join(本目录, "data"), exist_ok=True)

    for 模型名 in 模型列表:
        try:
            生成模型回复(模型名)
        except Exception as e:
            import traceback
            print(f"\n[错误] {模型名}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("生成阶段完成！请运行 验证_第二阶段_裁判评分.py 进行评分。")
    print("=" * 60)


if __name__ == "__main__":
    主生成()
