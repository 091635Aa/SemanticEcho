# -*- coding: utf-8 -*-
"""读取思考链中断对比结果"""
import json, glob

d = r"i:\Desktop\语义回响\实验数据\多模型对照\思考链中断"
for f in sorted(glob.glob(d + r"\*.json")):
    j = json.load(open(f, encoding="utf-8"))
    print(f"文件: {f.split(chr(92))[-1]}")
    for 模式 in ("裸", "全面纠正", "思考链纠正"):
        if 模式 in j:
            m = j[模式]
            print(f"  [{模式}] 熵={m['平均熵']}(std{m['熵std']}) 重={m['重复率']} 命中={m['情感命中率']}")
    if "思考链纠正" in j:
        lst = j["思考链纠正"]["每条"]
        print(f"  思考链明细：总条数={len(lst)}")
        print(f"  含思考标记: {sum(1 for x in lst if x.get('含思考标记'))}/{len(lst)}")
        print(f"  中断条数(思考步数>0): {sum(1 for x in lst if x.get('思考步数',0)>0)}")
        print(f"  平均思考步数(仅中断条): "
              f"{sum(x['思考步数'] for x in lst if x.get('思考步数'))/max(1,sum(1 for x in lst if x.get('思考步数'))):.0f}")
        print(f"  总体范数均值: {sum(x.get('总体范数') or 0 for x in lst)/max(1,len(lst)):.1f}")
        print("  样例(前6条):")
        for x in lst[:6]:
            print(f"    [{x['维度']}] 思考{x.get('思考步数')}步 范数{x.get('总体范数')} 标记{x.get('含思考标记')} 熵{x['熵']:.3f} 重{x['重']:.3f} | {x.get('文本预览','')[:50]}")
