# -*- coding: utf-8 -*-
"""验证完整版论文 PDF 页数与内容"""
from pypdf import PdfReader

p = r"i:\Desktop\语义回响\论文\决策的温度_底层记忆化AI扮演架构_完整版.pdf"
r = PdfReader(p)
n = len(r.pages)
print("总页数:", n)
# 关键章节所在页
关键词 = ["模块零", "模块五 · 第 0 层", "模块十 · 实验结果", "附录 A", "附录 B", "附录 C", "附录 D",
          "LoRA 外挂 + 思考链中断融合测试", "参考文献"]
for kw in 关键词:
    for i in range(n):
        t = r.pages[i].extract_text() or ""
        if kw in t:
            print(f"  [{kw}] -> 第 {i+1} 页")
            break
# 每页文本量粗估（确认无空白页）
空白 = sum(1 for i in range(n) if not (r.pages[i].extract_text() or "").strip())
print("空白页数:", 空白)
