# -*- coding: utf-8 -*-
"""
训练「情感外挂 Qwen3-4B（带思考链）」LoRA 适配器
================================================
与 train_emotion_lora.py 的唯一差异：chat template 渲染时 enable_thinking=True，
使 LoRA 保留 Qwen3 的预训练思考链习惯（<think>...</think>）。
这样挂载 LoRA 后，思考链中断注入器的思考阶段依然有效（思考 token 正常产出、情感向量可捕获）。

输出: f:\lora外挂\lora_adapters\emotion_qwen3_4B_think
"""
import os
import sys
import json
import argparse
from pathlib import Path

项目根 = Path(r"f:\lora外挂")
sys.path.insert(0, str(项目根))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

数据集路径 = 项目根 / "data" / "emotion_dataset.jsonl"
输出目录 = 项目根 / "lora_adapters" / "emotion_qwen3_4B_think"


def 构建数据集(tokenizer):
    from datasets import Dataset
    原始 = []
    with open(数据集路径, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                原始.append(json.loads(line))

    def 格式化(d):
        messages = [
            {"role": "user", "content": d["instruction"]},
            {"role": "assistant", "content": d["response"]},
        ]
        # 关键差异：enable_thinking=True，让 LoRA 学到"先思考再表达"的习惯
        文本 = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=True)
        return {"text": 文本}

    ds = Dataset.from_list([格式化(d) for d in 原始])

    def tokenize(example):
        out = tokenizer(example["text"], truncation=True, max_length=384, padding=False)
        out["labels"] = out["input_ids"].copy()
        return out

    return ds.map(tokenize, remove_columns=["text"])


def 主():
    global 输出目录
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=r"l:\模型空间\Qwen3-4B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--quant", choices=["fp16", "4bit"], default="4bit")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.out:
        输出目录 = Path(args.out)

    print(f"[加载基座模型] {args.model} 模式={args.quant} enable_thinking=True")
    kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.float16,
        )
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    配置 = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, 配置)
    model.print_trainable_parameters()

    数据集 = 构建数据集(tokenizer)
    print(f"训练数据: {len(数据集)} 条")
    训练参数 = TrainingArguments(
        output_dir=str(项目根 / "training_scripts" / "checkpoints_emotion_think"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        fp16=True,
        dataloader_pin_memory=False,
    )
    训练器 = Trainer(
        model=model,
        args=训练参数,
        train_dataset=数据集,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    print("[开始训练]")
    训练器.train()
    输出目录.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(输出目录))
    tokenizer.save_pretrained(str(输出目录))
    print(f"LoRA 适配器已保存: {输出目录}")


if __name__ == "__main__":
    主()
