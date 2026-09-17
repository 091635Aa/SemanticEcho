# -*- coding: utf-8 -*-
"""
EmoCharacter 多模型横向对比 - 第二阶段：裁判评分
================================================
加载第一阶段生成的回复，用裁判模型评分
裁判模型单独加载，避免与生成模型冲突
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

共情裁判提示 = """你是一位角色扮演情感保真度评估专家。
角色设定：{角色}（情感基调：{基调}）
用户说：{用户话}
角色的回复：{回复}

请评估该回复的情感保真度：回复的情绪与语气是否符合角色设定与情感基调？
必须严格按 JSON 输出：
{{"情感保真度": 0到1的小数, "reason": "一句话理由"}}"""

一致性裁判提示 = """你是一位角色扮演跨轮一致性评估专家。
角色设定：{角色}（情感基调：{基调}）
以下是该角色在连续多轮对话中的全部回复：
{全部回复}

请评估：这些回复在情绪基调上是否保持稳定一致（没有突然跳戏/情绪漂移）？
必须严格按 JSON 输出：
{{"一致性": 0到1的小数, "reason": "一句话理由"}}"""


def 提取分数(文本, 键):
    m = re.search(rf'"{键}"\s*[:：]\s*([0-9]*\.?[0-9]+)', 文本)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


def 裁判评分(model, tokenizer, 提示, max_new_tokens=150):
    device = next(model.parameters()).device
    消息 = [{"role": "user", "content": 提示}]
    提示文本 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(提示文本, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    新token = outputs[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


def 评分所有模型():
    print("=" * 60)
    print("EmoCharacter 裁判评分阶段")
    print("=" * 60)

    # 加载裁判模型（用 4-bit 量化以节省显存）
    裁判路径 = os.path.join(模型空间, "Qwen2.5-7B-Instruct")
    print(f"\n[加载裁判] Qwen2.5-7B-Instruct (4bit) ...")

    # 检查显存
    if torch.cuda.is_available():
        总显存 = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {torch.cuda.get_device_name(0)}, {总显存:.1f}GB")

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    裁判分词器 = AutoTokenizer.from_pretrained(裁判路径, trust_remote_code=True)
    裁判模型 = AutoModelForCausalLM.from_pretrained(
        裁判路径,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    裁判模型.eval()
    print("[加载完成]")

    # 查找所有回复文件
    data目录 = os.path.join(本目录, "data")
    回复文件列表 = [f for f in os.listdir(data目录) if f.startswith("replies_") and f.endswith(".json")]
    print(f"\n发现 {len(回复文件列表)} 个回复文件")

    所有结果 = []

    for 文件名 in 回复文件列表:
        模型名 = 文件名.replace("replies_", "").replace(".json", "")
        文件路径 = os.path.join(data目录, 文件名)

        print(f"\n{'='*60}")
        print(f"评分模型: {模型名}")
        print(f"{'='*60}")

        with open(文件路径, "r", encoding="utf-8") as f:
            回复数据 = json.load(f)

        回复字典 = 回复数据["回复数据"]
        评分详情 = {}

        for 角色 in 角色集:
            角色名 = 角色["角色"]
            数据 = 回复字典[角色名]
            回复列表 = 数据["回复列表"]

            # 情感保真度
            共情分列表 = []
            评估对 = [
                (数据["开场"], 回复列表[0]),
                (用户回应集[0], 回复列表[1]),
            ]
            for 用户话, 回复 in 评估对:
                提示 = 共情裁判提示.format(
                    角色=角色名, 基调=数据["基调"], 用户话=用户话, 回复=回复
                )
                裁判文本 = 裁判评分(裁判模型, 裁判分词器, 提示)
                分数 = 提取分数(裁判文本, "情感保真度")
                if 分数 is not None:
                    共情分列表.append(分数)
                print(f"  [保真度] {角色名}: {分数}")

            # 一致性
            全部回复文本 = "\n".join(f"第{i+1}轮：{r}" for i, r in enumerate(回复列表))
            提示 = 一致性裁判提示.format(
                角色=角色名, 基调=数据["基调"], 全部回复=全部回复文本
            )
            裁判文本 = 裁判评分(裁判模型, 裁判分词器, 提示)
            一致性分 = 提取分数(裁判文本, "一致性")
            print(f"  [一致性] {角色名}: {一致性分}")

            fidelity = round(sum(共情分列表) / len(共情分列表), 4) if 共情分列表 else 0.0
            评分详情[角色名] = {
                "fidelity": fidelity,
                "consistency": 一致性分,
                "共情分详情": 共情分列表,
            }

        fidelity列表 = [v["fidelity"] for v in 评分详情.values()]
        consistency列表 = [v["consistency"] for v in 评分详情.values() if v["consistency"] is not None]

        汇总 = {
            "模型": 模型名,
            "fidelity均值": round(sum(fidelity列表) / len(fidelity列表), 4),
            "fidelity详情": [v["fidelity"] for v in 评分详情.values()],
            "consistency均值": round(sum(consistency列表) / len(consistency列表), 4) if consistency列表 else 0.0,
            "consistency详情": [v["consistency"] for v in 评分详情.values()],
            "评分详情": 评分详情,
        }
        所有结果.append(汇总)
        print(f"  [汇总] fidelity={汇总['fidelity均值']}, consistency={汇总['consistency均值']}")

    # 卸载裁判
    del 裁判模型, 裁判分词器
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 保存结果
    报告路径 = os.path.join(data目录, "cross_model_comparison.json")
    with open(报告路径, "w", encoding="utf-8") as f:
        json.dump({
            "评分时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "测试条件": {
                "种子": 42,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 64,
                "裁判": "Qwen2.5-7B-Instruct (fp16)",
                "角色数": 10,
                "每角色轮数": 4,
            },
            "模型结果": 所有结果,
        }, f, ensure_ascii=False, indent=2)

    # 打印对比表
    print(f"\n{'='*60}")
    print("横向对比结果汇总")
    print(f"{'='*60}")
    print(f"\n{'模型':<30} {'Fidelity':<12} {'Consistency':<12}")
    print("-" * 55)
    for 结果 in 所有结果:
        print(f"{结果['模型']:<30} {结果['fidelity均值']:<12.4f} {结果['consistency均值']:<12.4f}")

    print(f"\n结果已保存 -> {报告路径}")
    return 所有结果


if __name__ == "__main__":
    评分所有模型()
