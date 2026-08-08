# -*- coding: utf-8 -*-
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
d = json.load(open(r'i:\Desktop\语义回响\实验数据\哭鼻子分析\整段_全面压榨分析.json', encoding='utf-8'))
ev = d['事件']
print('总事件:', len(ev))
seen = set(); 关键 = []
for e in ev:
    k = (e['时间'], e['描述'])
    if k in seen:
        continue
    seen.add(k)
    关键.append(e)
print('去重后:', len(关键))
for e in 关键:
    print(f"{e['时间']} | {e['情绪']} | {e['描述']} | {e['强度']}")
