"""
语义回响 (Semantic Echo) — 一键启动器

用法：
    python run_demo.py             启动 Web 交互式演示平台
    python run_demo.py check       检查模型兼容性
    python run_demo.py info        显示版本信息
"""

import sys
import os

# 将项目根目录加入 sys.path
项目根目录 = os.path.dirname(os.path.abspath(__file__))
if 项目根目录 not in sys.path:
    sys.path.insert(0, 项目根目录)


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] in ("check", "info", "list"):
        from semantic_echo import cli
        cli.main()
    else:
        print("""
╔══════════════════════════════════════════════╗
║       语义回响 (Semantic Echo)              ║
║   通过回收被丢弃Token嵌入增强语言模型表达   ║
║                                            ║
║  正在启动 Web 演示平台...                   ║
║  请稍后，浏览器将自动打开                  ║
║  地址: http://localhost:7860                ║
╚══════════════════════════════════════════════╝
""")
        from semantic_echo.demo_app import main as demo_main
        demo_main()


if __name__ == "__main__":
    main()
