"""验证模型加载"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型和分词器
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 方式1: 如果有本地模型
import os
本地路径 = os.path.join(os.path.dirname(__file__), "本地模型")
if os.path.exists(本地路径) and any(os.listdir(本地路径)):
    print(f"从本地加载: {本地路径}")
    tokenizer = AutoTokenizer.from_pretrained(本地路径, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        本地路径, 
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )
else:
    print(f"从 HuggingFace 加载: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )

# 测试生成
prompt = "你好"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(f"输入: {prompt}")
print(f"输出: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
print("模型加载成功！")
print(f"hidden_size: {model.config.hidden_size}")
print(f"vocab_size: {model.config.vocab_size}")
print(f"设备: {model.device}")
