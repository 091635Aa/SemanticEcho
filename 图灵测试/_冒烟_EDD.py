# -*- coding: utf-8 -*-
"""P6 EDD 冒烟测试：多任务类型生成质量检查"""
import os, sys, gc
本目录 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 本目录)
from 统一生成器 import 生成器实例

测试 = [
    ("情感倾诉", "有时候我会想，努力到底有什么意义。"),
    ("情感倾诉", "我真的好累，感觉撑不下去了。"),
    ("角色扮演", "你是我的女朋友，我说我今天工作特别不顺心。"),
    ("知识决策", "我想学习编程，应该从哪个语言开始学？"),
    ("闲聊", "这饮料味道不错。"),
    ("角色扮演", "请你扮演一个毒舌但心软的损友，我失恋了。"),
]

for 类型, 话 in 测试:
    try:
        消息 = [{"role": "user", "content": 话}]
        文本 = 生成器实例.生成("P6_情感导演", 消息, 种子=2026, 轮次=0, max_new_tokens=128)
        print(f"\n[{类型}] {话[:24]}")
        print(f"  → {文本[:100]}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[{类型}] 失败: {e}")
    finally:
        gc.collect()

生成器实例.清理()
print("\n=== 冒烟测试完成 ===")
