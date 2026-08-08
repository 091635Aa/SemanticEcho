# -*- coding: utf-8 -*-
"""最终报告 Markdown → PDF 转换（markdown → HTML → Edge 无头打印）
用法: python 转PDF.py <输入.md> <输出.pdf> [--title 标题]
"""
import sys, time, argparse, subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import markdown

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page {
  size: A4;
  margin: 2cm 2.2cm 2cm 2.2cm;
  @bottom-center { content: "语义回响 · 超级智能体陪伴一体化推理框架 | 第 " counter(page) " / " counter(pages) " 页";
                   font-size: 8pt; color: #888; }
}
html, body { font-family: "Microsoft YaHei", "SimSun", sans-serif; font-size: 10.5pt;
             line-height: 1.65; color: #222; margin: 0; padding: 0; }
h1 { font-size: 16pt; color: #1a3c6e; text-align: center; margin: 0 0 6pt 0; }
h2 { font-size: 13pt; color: #1a3c6e; border-bottom: 1pt solid #c8d4e4;
     padding-bottom: 2pt; margin-top: 18pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #2a5488; margin-top: 12pt; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; }
th { background: #eef3f9; border: 0.5pt solid #9db2c8; padding: 3pt 5pt; text-align: center; }
td { border: 0.5pt solid #b7c6d6; padding: 3pt 5pt; }
tr { page-break-inside: avoid; }
blockquote { color: #555; border-left: 3pt solid #9db2c8; padding-left: 8pt; margin: 6pt 0; }
code { font-family: Consolas, monospace; background: #f4f4f4; font-size: 8.5pt; padding: 0 2pt; }
pre { background: #f4f4f4; border: 0.5pt solid #ddd; padding: 6pt; font-size: 8.5pt;
      white-space: pre-wrap; }
hr { border: 0; border-top: 1pt solid #ccc; margin: 12pt 0; }
p { margin: 6pt 0; }
ul, ol { margin: 6pt 0; padding-left: 22pt; }
li { margin: 2pt 0; }
"""

def 主():
    parser = argparse.ArgumentParser()
    parser.add_argument("输入", help="输入 .md 文件")
    parser.add_argument("输出", help="输出 .pdf 文件")
    args = parser.parse_args()

    md文本 = Path(args.输入).read_text(encoding="utf-8")
    body_html = markdown.markdown(md文本, extensions=["tables", "fenced_code", "sane_lists"])
    标题 = md文本.splitlines()[0].lstrip("# ").strip()

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{标题}</title>
<style>{CSS}</style></head>
<body>{body_html}</body></html>"""

    with TemporaryDirectory(prefix="md2pdf_", ignore_cleanup_errors=True) as 临时:
        临时 = Path(临时)
        html文件 = 临时 / "doc.html"
        html文件.write_text(html, encoding="utf-8")
        pdf文件 = Path(args.输出)
        pdf文件.parent.mkdir(parents=True, exist_ok=True)

        命令 = [
            EDGE,
            "--headless", "--disable-gpu", "--no-pdf-header-footer",
            f"--user-data-dir={临时 / 'profile'}",
            f"--print-to-pdf={pdf文件}",
            html文件.as_uri(),
        ]
        proc = subprocess.Popen(命令, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            if pdf文件.exists() and pdf文件.stat().st_size > 0:
                break
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        else:
            proc.kill()
            print("超时：Edge 未在 60s 内完成"); sys.exit(1)
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        time.sleep(1)

        if not pdf文件.exists() or pdf文件.stat().st_size == 0:
            print("错误：未生成 PDF"); sys.exit(1)

    print(f"PDF 已生成: {pdf文件} ({pdf文件.stat().st_size/1024:.0f} KB)")

if __name__ == "__main__":
    主()
