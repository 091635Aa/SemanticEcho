"""
语义回响 (Semantic Echo) — 命令行工具

安装后可用:
    semantic-echo check <model_name>  检查模型兼容性
    semantic-echo demo <model_name>   快速演示回响效果
    semantic-echo info                显示包信息
    semantic-echo list                列出已知兼容模型

也可直接运行:
    python -m semantic_echo.cli check Qwen/Qwen2.5-0.5B-Instruct
"""

import sys
import argparse
from typing import NoReturn


def 命令_检查(model_name: str) -> None:
    """检查指定模型的兼容性。"""
    try:
        from semantic_echo.check_compatibility import check_model_compatibility
    except ImportError as e:
        print(f"[错误] 导入失败: {e}")
        print("请确保已安装 semantic-echo 及其依赖。")
        sys.exit(1)

    report = check_model_compatibility(model_name)
    print(report.summary())


def 命令_演示(model_name: str) -> None:
    """运行一次快速演示，展示回响效果。"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from semantic_echo.回响池 import 语义回响池
        from semantic_echo.采样处理器 import 回响注入器
        from semantic_echo.情感过滤器 import 情感过滤器
        from semantic_echo.回响评估器 import 计算语义熵, 逐Token评估器
    except ImportError as e:
        print(f"[错误] 导入失败: {e}")
        print("请确保已安装 semantic-echo、torch 和 transformers。")
        sys.exit(1)

    # 先检查兼容性
    from semantic_echo.check_compatibility import check_model_compatibility
    report = check_model_compatibility(model_name)
    print(report.summary())
    if not report.is_compatible:
        print("[终止] 模型不兼容，无法进行演示。")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/3] 加载模型: {model_name} (设备: {device})")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"[错误] 模型加载失败: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"[2/3] 初始化语义回响组件")
    hidden_dim = model.config.hidden_size
    回响池 = 语义回响池(
        hidden_dim=hidden_dim,
        max_pool_size=1024,
        decay_gamma=0.05,
    )
    注入器 = 回响注入器(
        model=model,
        echo_pool=回响池,
        lambda_strength=0.5,
    )

    prompt = "请告诉我一件让你开心的事情。"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    print(f"[3/3] 生成回响增强文本...")
    print(f"\n  提示词: {prompt}")
    print(f"  {'─' * 40}")

    评估器 = 逐Token评估器(model.config.vocab_size)

    def logits回调(step: int, logits: torch.Tensor) -> None:
        评估器.记录(step, logits)

    with torch.no_grad():
        full_ids = 注入器.生成(
            input_ids=inputs.input_ids,
            max_new_tokens=128,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            logits_callback=logits回调,
        )

    generated_ids = full_ids[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\n  生成文本:")
    print(f"  {generated_text}")
    print(f"  {'─' * 40}")
    print(f"  语义熵: {评估器.平均熵:.4f}")
    print(f"  回响池大小: {回响池.大小}")
    print(f"  演示完成！")


def 命令_信息() -> None:
    """显示包版本和作者信息。"""
    try:
        from semantic_echo import __version__, __author__, __license__
    except ImportError:
        __version__ = "?"
        __author__ = "?"
        __license__ = "?"

    print(f"""
╔══════════════════════════════════════════════╗
║          语义回响 (Semantic Echo)           ║
║   通过回收被丢弃Token嵌入增强语言模型表达   ║
╠══════════════════════════════════════════════╣
║  版本: {__version__:<34}║
║  作者: {__author__:<34}║
║  联系: DYPUBG2025@QQ.COM                    ║
╠══════════════════════════════════════════════╣
║  许可证: 保留所有权利                       ║
║  任何人可基于学术目的自由复现               ║
╚══════════════════════════════════════════════╝

论文: https://github.com/091635Aa/SemanticEcho
""")


def 命令_列表() -> None:
    """列出已知兼容模型。"""
    from semantic_echo.check_compatibility import get_compatible_models

    models = get_compatible_models()
    if not models:
        print("暂无已测试兼容的模型记录。")
        return

    print("\n  已知兼容模型（已通过实际实验验证）:\n")
    for m in models:
        print(f"    ✅ {m['model_name']}")
        print(f"       {m['description']}")
    print()


def main() -> None:
    """CLI 主入口。"""
    parser = argparse.ArgumentParser(
        description="语义回响 (Semantic Echo) — 命令行工具",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # check
    check_parser = subparsers.add_parser(
        "check", help="检查 HuggingFace 模型是否兼容语义回响框架"
    )
    check_parser.add_argument("model_name", type=str, help="模型名称（如 Qwen/Qwen2.5-0.5B-Instruct）")

    # demo
    demo_parser = subparsers.add_parser(
        "demo", help="运行一次快速演示，展示回响增强效果"
    )
    demo_parser.add_argument("model_name", type=str, help="模型名称")

    # info
    subparsers.add_parser("info", help="显示包信息")

    # list
    subparsers.add_parser("list", help="列出已知兼容模型")

    args = parser.parse_args()

    if args.command == "check":
        命令_检查(args.model_name)
    elif args.command == "demo":
        命令_演示(args.model_name)
    elif args.command == "info":
        命令_信息()
    elif args.command == "list":
        命令_列表()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
