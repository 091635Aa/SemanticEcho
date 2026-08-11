# -*- coding: utf-8 -*-
"""R1/R2 链路冒烟测试：四层（chat 模板 + 回响 + 跨轮池）vs 裸"""
import sys
sys.path.insert(0, r"i:\Desktop\语义回响\图灵测试")

from 生成器 import 生成器实例

消息 = [{"role": "system", "content": "你现在是「温柔治愈系女友」，你的情感基调是：温柔、体贴、带点俏皮。请始终以这个角色身份回复，不要跳出角色。"},
        {"role": "user", "content": "你今天好像不太开心，怎么了？"}]

for 轮 in range(2):
    print(f"—— 第{轮}轮 ——")
    print("四层:", 生成器实例.生成("四层", 消息, 种子=42, 轮次=轮, max_new_tokens=64, 会话="测试"))
    消息.append({"role": "assistant", "content": "（上一轮回复）"})
    消息.append({"role": "user", "content": "我最近真的好累，感觉撑不下去了。"})

生成器实例.清理()
print("冒烟测试完成")
