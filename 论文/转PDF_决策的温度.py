# -*- coding: utf-8 -*-
"""决策的温度 · 底层记忆化 AI 扮演五层架构 完整版论文 Markdown → PDF
（图片 base64 内嵌 + 封面顶图 + 模块分隔线 + 公式代码块样式）
用法: python 转PDF_决策的温度.py
"""
import base64, html, re, sys, time, subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import markdown

MD_PATH = Path(r"i:\Desktop\语义回响\论文\决策的温度_底层记忆化AI扮演架构_完整版.md")
PDF_PATH = Path(r"i:\Desktop\语义回响\论文\决策的温度_底层记忆化AI扮演架构_完整版.pdf")
# 附录文件（按序拼接在正文 <!-- APPENDIX --> 标记处）
附录文件 = [
    Path(r"i:\Desktop\语义回响\论文\附录A_实验数据全集.md"),
    Path(r"i:\Desktop\语义回响\论文\附录B_完整代码.md"),
    Path(r"i:\Desktop\语义回响\论文\附录C_图灵测试与情感实证.md"),
    Path(r"i:\Desktop\语义回响\论文\附录D_开发过程与错误修复.md"),
]
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not Path(EDGE).exists():
    EDGE = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 1.9cm 2.2cm 1.9cm 2.2cm;
  @bottom-center { content: "决策的温度 · 底层记忆化 AI 扮演五层通用架构 | 第 " counter(page) " / " counter(pages) " 页";
                   font-size: 8pt; color: #888; } }
html, body { font-family: "Microsoft YaHei", "SimSun", sans-serif; font-size: 10.5pt;
             line-height: 1.68; color: #222; margin: 0; padding: 0; }
/* ── 论文总标题 ── */
h1.paper-title { font-size: 18pt; color: #0f2a52; text-align: center;
                 margin: 0 0 4pt 0; line-height: 1.4; }
/* ── 模块一级标题（带底色分隔条，直观标示模块） ── */
h1 { font-size: 14.5pt; color: #0f2a52; text-align: center;
     background: linear-gradient(90deg, #eef3f9, #dfe9f5, #eef3f9);
     border-left: 5pt solid #1a5fb4; border-right: 5pt solid #1a5fb4;
     padding: 6pt 0; border-radius: 4pt; margin-top: 22pt;
     page-break-before: auto; page-break-after: avoid; }
h2 { font-size: 12.5pt; color: #1a3c6e; border-bottom: 1pt solid #c8d4e4;
     padding-bottom: 2pt; margin-top: 16pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #2a5488; margin-top: 12pt; page-break-after: avoid; }
h4 { font-size: 10.8pt; color: #2a5488; margin-top: 10pt; page-break-after: avoid; }
/* ── 模块分隔线（---- 上方）：渐变分隔条，直观标示模块切换 ── */
/* ── 模块分隔线（----）：渐变分隔条，避免孤立在页尾/页首 ── */
hr { border: 0; height: 3pt; margin: 8pt 0 12pt 0;
     background: linear-gradient(90deg, transparent, #9db2c8 18%, #1a5fb4 50%, #9db2c8 82%, transparent);
     border-radius: 2pt; page-break-after: avoid; }
/* ── 封面顶图：全宽通栏 ── */
img.cover { width: 100%; max-width: 100%; border: none; border-radius: 5pt;
            margin: 6pt auto 10pt auto; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 8.4pt; }
th { background: #eef3f9; border: 0.5pt solid #9db2c8; padding: 3pt 5pt; text-align: center; }
td { border: 0.5pt solid #b7c6d6; padding: 3pt 5pt; }
tr { page-break-inside: avoid; }
thead { display: table-header-group; }  /* 表格跨页时重复表头 */
blockquote { color: #555; border-left: 3pt solid #9db2c8; padding-left: 8pt; margin: 6pt 0; }
code { font-family: Consolas, monospace; background: #f4f4f4; font-size: 8.5pt; padding: 0 2pt; }
pre { background: #f6f8fa; border: 0.5pt solid #d0d7de; border-radius: 4pt; padding: 7pt 9pt;
      font-size: 8.8pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: transparent; padding: 0; }
img { max-width: 100%; height: auto; display: block; margin: 10pt auto; border: 0.5pt solid #d0d7de;
      border-radius: 4pt; page-break-inside: avoid; }
p { margin: 6pt 0; }
ul, ol { margin: 6pt 0; padding-left: 22pt; }
li { margin: 2pt 0; }
strong { color: #0f3460; }
/* 公式排版 */
mjx-container[display="true"] { display: block; text-align: center; margin: 10pt 0; }
mjx-container { font-size: 1.02em; }
"""

MATHJAX_HEAD = """
<script>
window.MathJax = {
  tex: {
    macros: { bm: '\\\\boldsymbol' },
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  },
  chtml: { mtextInheritFont: true, scale: 1 }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
"""

def 嵌入图片(md文本: str, 基准目录: Path) -> str:
    def 替换(m):
        alt, 路径 = m.group(1), m.group(2)
        if 路径.startswith(("http://", "https://", "data:")):
            return m.group(0)
        绝对 = (基准目录 / 路径).resolve()
        try:
            data = 绝对.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            ext = 绝对.suffix.lstrip(".").lower() or "png"
            return f'<img alt="{alt}" src="data:image/{ext};base64,{b64}">'
        except Exception as e:
            print(f"[警告] 图片加载失败 {路径}: {e}")
            return m.group(0)
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", 替换, md文本)


def 保护公式(md文本: str):
    """提取 LaTeX 公式为占位符，避免 markdown 破坏 _/* 等符号"""
    公式列表 = []
    def 收集块(m):
        公式列表.append("$$" + m.group(1) + "$$")
        return f"@@MJX-{len(公式列表)-1}@@"
    def 收集行内(m):
        公式列表.append("$" + m.group(1) + "$")
        return f"@@MJX-{len(公式列表)-1}@@"
    md文本 = re.sub(r"\$\$(.+?)\$\$", 收集块, md文本, flags=re.S)
    md文本 = re.sub(r"\$([^$\n]+?)\$", 收集行内, md文本)
    return md文本, 公式列表


def 还原公式(body_html: str, 公式列表: list) -> str:
    for i, 公式 in enumerate(公式列表):
        body_html = body_html.replace(
            f"@@MJX-{i}@@", html.escape(公式, quote=False))
    return body_html


def 主():
    md文本 = MD_PATH.read_text(encoding="utf-8")
    # 拼接附录：将 <!-- APPENDIX --> 标记替换为附录 A/B/C 内容
    附录 = "\n\n".join(p.read_text(encoding="utf-8") for p in 附录文件 if p.exists())
    if 附录:
        md文本 = md文本.replace("<!-- APPENDIX -->", "\n\n---\n\n" + 附录)
    # 剥离残留的 HTML 注释标记（PART/FUSION 等）
    md文本 = re.sub(r"<!--.*?-->", "", md文本, flags=re.S)
    md文本 = 嵌入图片(md文本, MD_PATH.parent)
    md文本, 公式列表 = 保护公式(md文本)
    body_html = markdown.markdown(
        md文本, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    body_html = 还原公式(body_html, 公式列表)
    # 封面图兜底：attr_list 对图片兼容性不稳，直接给第一个 img 加 cover 类（全宽通栏）
    if "class=\"cover\"" not in body_html:
        body_html = re.sub(r"<img ", "<img class=\"cover\" ", body_html, count=1)
    标题 = md文本.splitlines()[0].lstrip("# ").strip()
    等待脚本 = """
<script>
  if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(function(){ document.title = "MATHJAX_DONE"; });
  }
</script>
"""
    html文本 = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{标题}</title>
{MATHJAX_HEAD}
<style>{CSS}</style></head>
<body>{body_html}{等待脚本}</body></html>"""

    with TemporaryDirectory(prefix="md2pdf_", dir=str(PDF_PATH.parent), ignore_cleanup_errors=True) as 临时:
        临时 = Path(临时)
        html文件 = 临时 / "doc.html"
        html文件.write_text(html文本, encoding="utf-8")
        # 保留一份渲染预览 HTML（便于验证公式/图表，可删除）
        try:
            Path(MD_PATH.parent / "_决策的温度_渲染预览.html").write_text(html文本, encoding="utf-8")
        except Exception:
            pass
        PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        输出PDF = 临时 / "out.pdf"
        命令 = [EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                "--virtual-time-budget=25000",
                f"--user-data-dir={临时 / 'profile'}",
                f"--print-to-pdf={输出PDF}", html文件.as_uri()]
        proc = subprocess.Popen(命令, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(180):
            if 输出PDF.exists() and 输出PDF.stat().st_size > 0:
                break
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        else:
            proc.kill(); print("超时"); sys.exit(1)
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        time.sleep(1)
        if not 输出PDF.exists() or 输出PDF.stat().st_size == 0:
            print("错误：未生成 PDF"); sys.exit(1)
        # 尝试替换目标；被占用则输出 v2
        try:
            if PDF_PATH.exists():
                PDF_PATH.unlink()
            输出PDF.replace(PDF_PATH)
            最终 = PDF_PATH
        except Exception as e:
            最终 = PDF_PATH.with_name(PDF_PATH.stem + "_v2.pdf")
            输出PDF.replace(最终)
            print(f"[提示] 目标被占用，已输出到 {最终.name}")
    print(f"PDF 已生成: {最终} ({最终.stat().st_size/1024:.0f} KB)")

if __name__ == "__main__":
    主()
