# -*- coding: utf-8 -*-
"""
批量下载多模型（通用架构二期）→ l:\\模型空间
============================================
- 优先魔搭(modelscope)下载（国内快），失败自动回退 hf-mirror
- 单个失败仅记录继续，不中断整体
- 不删除任何文件；已存在的模型目录自动跳过（快照增量续传）
用法：
    f:\\打标\\.venv\\Scripts\\python.exe scripts\\下载多模型.py
"""
import os
import time
import sys

模型空间 = r"l:\模型空间"
os.makedirs(模型空间, exist_ok=True)

# (魔搭ID, hf-mirror ID, 本地子目录, 说明)
任务 = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct", "中文0.5B"),
    ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", "Qwen3-0.6B", "中文0.6B"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2-1.7B-Instruct", "英文1.7B(替代gated的Llama-3.2-1B)"),
    ("google/gemma-2-2b-it", "google/gemma-2-2b-it", "gemma-2-2b-it", "多语言2B"),
    ("microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3.5-mini-instruct", "Phi-3.5-mini-instruct", "英文3.8B"),
    ("Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen3-4B", "中英4B"),
]


def 从魔搭下载(模型ID, 本地):
    from modelscope import snapshot_download
    return snapshot_download(模型ID, local_dir=本地, max_workers=8)


def 从hf镜像下载(模型ID, 本地):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import snapshot_download
    return snapshot_download(模型ID, local_dir=本地, resume_download=True)


def 已有完整(本地):
    """存在 config.json 视为已下载过（快照可能缺文件，仍走增量）"""
    return os.path.isfile(os.path.join(本地, "config.json"))


def 主流程():
    成功, 失败 = [], []
    for 魔搭ID, hfID, 子目录, 说明 in 任务:
        本地 = os.path.join(模型空间, 子目录)
        print(f"\n{'='*60}\n[下载] {说明} {魔搭ID}", flush=True)
        t0 = time.time()
        途径 = ""
        try:
            if 已有完整(本地):
                print(f"  已存在 {本地}，跳过（如需补齐请删除目录后重跑）", flush=True)
                成功.append(子目录)
                continue
            从魔搭下载(魔搭ID, 本地)
            途径 = "魔搭"
        except Exception as e1:
            print(f"  魔搭失败: {e1}", flush=True)
            try:
                从hf镜像下载(hfID, 本地)
                途径 = "hf-mirror"
            except Exception as e2:
                print(f"  hf-mirror 也失败: {e2}", flush=True)
                失败.append((子目录, str(e2)))
                continue
        成功.append(子目录)
        print(f"  完成 {子目录} 途径={途径} 耗时{time.time()-t0:.0f}s", flush=True)

    print(f"\n{'='*60}\n下载汇总：成功 {len(成功)} | 失败 {len(失败)}")
    if 失败:
        for 名, 错 in 失败:
            print(f"  ✗ {名}: {错}")


if __name__ == "__main__":
    主流程()
