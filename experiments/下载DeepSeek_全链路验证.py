# -*- coding: utf-8 -*-
"""下载未注册模型 DeepSeek-R1-Distill-Qwen-7B（全链路验证用）"""
from modelscope import snapshot_download

目标 = r"l:\模型空间\DeepSeek-R1-Distill-Qwen-7B"
print("[下载] deepseek-ai/DeepSeek-R1-Distill-Qwen-7B ->", 目标, flush=True)
model_dir = snapshot_download(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    local_dir=目标,
)
print("下载完成:", model_dir, flush=True)
