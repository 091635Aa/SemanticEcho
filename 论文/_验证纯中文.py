# -*- coding: utf-8 -*-
"""验证纯中文版：英文段落/哭鼻子内容是否清除"""
from pypdf import PdfReader

r = PdfReader(r"i:\Desktop\语义回响\论文\决策的温度_底层记忆化AI扮演架构_完整版.pdf")
全文 = "".join((p.extract_text() or "") for p in r.pages)
print("总页数:", len(r.pages))

# 应已删除的英文段落
for kw in ["Abstract", "Keywords", "The Temperature of Decisions", "bottom-up memorization",
           "Large language models have approached"]:
    print(f"  英文[{kw}]:", "残留!" if kw in 全文 else "已清除")
# 哭鼻子内容
for kw in ["哭鼻子", "男猫女猫", "钟言", "同桌的你"]:
    print(f"  污染[{kw}]:", "残留!" if kw in 全文 else "已清除")
# 纯中文封面检查（第一页应无 DECISION 英文）
p1 = r.pages[0].extract_text() or ""
print("  封面英文[DECISION]:", "残留!" if "DECISION" in p1 else "已清除")
print("  封面图片数:", len(r.pages[0].images))
