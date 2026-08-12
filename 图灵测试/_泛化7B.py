# -*- coding: utf-8 -*-
"""7B 专用：Qwen2.5-7B-Instruct 仅跑 P6 情感导演（省显存：不构建 P1 回响/P2.5 等）
输出 12 条 P6 回复 -> 统一基准\\泛化回复_Qwen2.5-7B-Instruct_P6.json
"""
import os, sys, json, gc, traceback

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)
KV核心 = r"c:\Users\Administrator\Documents\KV 情感共振解码\核心"
if KV核心 not in sys.path:
    sys.path.insert(0, KV核心)
ETD目录 = r"h:\情感潮汐解码（Emotion Tidal Decoding, ETD）"
if ETD目录 not in sys.path:
    sys.path.insert(0, ETD目录)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 情感导演解码器 import 情感导演解码器

统一目录 = os.path.join(本目录, "统一基准")
样本路径 = os.path.join(本目录, "样本_30条.json")
输出路径 = os.path.join(统一目录, "泛化回复_Qwen2.5-7B-Instruct_P6.json")
模型名 = "Qwen2.5-7B-Instruct"
模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-7B-Instruct"

P6人格注入提示 = (
    "你是真实的人类，正在和朋友聊天。你的回复要简短自然、口语化，像微信消息，"
    "通常一到两句话，可以有情绪和小俏皮，但绝不要提及你是AI、模型、助手或机器人，"
    "也不要使用「您」这样的敬语。"
)

def main():
    print(f"[7B-P6] 加载 {模型名} ...", flush=True)
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = None
    print("[7B-P6] 7B fp16 需要 ~15GB 显存且 RAM 峰值高，直接使用 4bit 量化加载 ...", flush=True)
    加载参数 = dict(quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True),
        device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)
    模型 = AutoModelForCausalLM.from_pretrained(模型路径, **加载参数)
    模型.eval()
    torch.cuda.empty_cache()
    print(f"[7B-P6] 模型加载完成 {next(模型.parameters()).dtype}", flush=True)

    锚库 = 锚点库(模型, 分词器)
    锚库.构建()
    目标决策 = 目标决策器(感知器=None, 潮汐决策器=None, 锚点库=锚库)
    P6 = 情感导演解码器(
        模型, 分词器, 锚库, 目标决策,
        开启DMR=True, 开启KV调制=True, 开启V调制=True,
        开启锚点偏置=True, 开启DSA=True,
        α基=0.18, κ基=0.20, κ_v基=0.12, β基=0.6,
        AI腔抑制强度=2.0, 口语化强度=0.6,
        任务自适应=True, 进度调度=True, 在线纠正=True, 句子停止=True)
    # 释放锚点库大矩阵
    锚库.W_e = None
    锚库._有效权重 = None
    锚库.权重 = None
    torch.cuda.empty_cache()
    print("[7B-P6] 解码器就绪", flush=True)

    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"][:12]

    结果 = []
    for i, r in enumerate(样本):
        种子值 = 2026 + i
        torch.manual_seed(种子值)
        torch.cuda.manual_seed(种子值)
        消息 = [{"role": "user", "content": r["user"]}]
        用户内容 = r["user"]
        _跳过 = ("JSON", "decision_choice", "final_decision", "Output strictly",
                "rubric", "```", "dialogue_history", "对话历史", "真诚伙伴",
                "下文回应", "考题", "请将以上", "设定", "场景")
        if not any(_w in 用户内容 for _w in _跳过):
            消息 = [{"role": "system", "content": P6人格注入提示}] + 消息
        try:
            提示 = 分词器.apply_chat_template(
                消息, tokenize=False, add_generation_prompt=True)
        except Exception:
            提示 = 分词器.apply_chat_template(
                [{"role": "user", "content": P6人格注入提示 + "\n" + 用户内容}],
                tokenize=False, add_generation_prompt=True)
        inputs = 分词器(提示, return_tensors="pt").to(模型.device)
        try:
            out, _ = P6.生成(
                inputs.input_ids, max_new_tokens=64,
                eos_token_id=分词器.eos_token_id,
                tokenizer=分词器, 用户文本=用户内容,
                temperature=1.0, top_p=0.9, top_k=50,
                repetition_penalty=1.05)
            新 = out[0, inputs.input_ids.shape[1]:]
            文本 = 分词器.decode(新, skip_special_tokens=True).strip()
        except Exception as e:
            文本 = f"[生成失败:{e}]"
        if 文本.startswith("<think>"):
            文本 = "（思考）" + 文本
        结果.append(文本)
        print(f"[7B-P6 {i+1}/12] {用户内容[:12]} => {文本[:40]}", flush=True)

    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump({"P6_情感导演": 结果}, f, ensure_ascii=False, indent=2)
    print(f"[7B-P6] 完成 -> {输出路径}", flush=True)

    # 清理
    try:
        P6._移除钩子()
    except Exception:
        pass
    del 模型
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
