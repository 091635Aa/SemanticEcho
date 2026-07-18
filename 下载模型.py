"""
模型下载脚本 — 从 HuggingFace 镜像下载 Qwen2.5-0.5B-Instruct
"""

import os
import sys


def 下载模型() -> str | None:
    """使用镜像下载 Qwen2.5-0.5B-Instruct"""
    # 设置镜像环境变量
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import snapshot_download

    模型路径 = os.path.join(os.path.dirname(__file__), "本地模型")
    os.makedirs(模型路径, exist_ok=True)

    print(f"正在从 hf-mirror.com 下载 Qwen/Qwen2.5-0.5B-Instruct 到 {模型路径}")
    print("模型大小约 0.5B 参数，fp16 约 1GB，请耐心等待...")

    try:
        snapshot_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            local_dir=模型路径,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"模型下载完成！保存在 {模型路径}")
        return 模型路径
    except Exception as e:
        print(f"下载失败: {e}")
        print("尝试使用 transformers 的 from_pretrained 方法...")
        return None


def 下载模型transformers() -> str | None:
    """备选方案：使用 transformers 下载"""
    模型路径 = os.path.join(os.path.dirname(__file__), "本地模型")
    os.makedirs(模型路径, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("使用 transformers 下载模型...")
    try:
        模型 = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            cache_dir=模型路径,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            cache_dir=模型路径,
            trust_remote_code=True,
        )
        print("模型和分词器下载成功！")
        return 模型路径
    except Exception as e:
        print(f"下载失败: {e}")
        return None


if __name__ == "__main__":
    # 先尝试镜像下载
    路径 = 下载模型()
    if 路径 is None:
        # 再尝试 transformers 方式
        路径 = 下载模型transformers()
    if 路径:
        print(f"模型已保存至: {路径}")
    else:
        print("所有下载方式均失败，请检查网络连接")
        sys.exit(1)
