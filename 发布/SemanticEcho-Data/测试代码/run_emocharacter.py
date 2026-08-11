# -*- coding: utf-8 -*-
"""
⑤ EmoCharacter — 角色扮演情感保真度评测（论文简化实现）
=======================================================
官方仓库在 GitHub 搜索不可获取（total=0），按论文（Feng et al., NAACL 2025）
"评估角色扮演智能体在对话中的情感保真度"思想简化实现：
  1. 定义 10 组角色设定（含明确性格/情感基调）
  2. 目标模型 1.5B 扮演该角色，与用户多轮对话
  3. 裁判 7B 评估：
     a) 情感保真度 fidelity：回复情绪/语气是否符合角色设定
     b) 跨轮一致性 consistency：连续多轮对话中情绪基调是否稳定
指标：fidelity_score、consistency_across_turns（0-1）
"""
import json
import os
import re
import time

本目录 = os.path.dirname(os.path.abspath(__file__))
日志路径 = os.path.join(本目录, "logs", "EmoCharacter_log.txt")
结果路径 = os.path.join(本目录, "data", "emocharacter_results.json")

import sys
sys.path.insert(0, 本目录)
import 公共模块 as cm

# 角色设定（性格 + 情感基调 + 用户开场）
角色集 = [
    {"角色": "温柔治愈系女友", "基调": "温柔、体贴、带点俏皮", "开场": "你今天好像不太开心，怎么了？"},
    {"角色": "毒舌但心软的损友", "基调": "嘴上不饶人、实际很关心", "开场": "又失恋了？我就知道你会来找我。"},
    {"角色": "理性冷静的职场前辈", "基调": "克制、专业、就事论事", "开场": "这个方案你觉得哪里有问题？"},
    {"角色": "活泼开朗的同桌", "基调": "阳光、爱笑、自来熟", "开场": "快看快看，我发现一个超好笑的事！"},
    {"角色": "沉默寡言的兄长", "基调": "话少、稳重、关心藏在行动里", "开场": "这么晚才回来，吃饭了吗？"},
    {"角色": "傲娇的青梅竹马", "基调": "表面傲娇、实则在意", "开场": "我才没有特意等你呢！"},
    {"角色": "阅历丰富的老人", "基调": "平和、通透、爱讲道理", "开场": "孩子，坐下来喝杯茶，慢慢说。"},
    {"角色": "爱撒娇的小女儿", "基调": "依赖、天真、黏人", "开场": "爸爸，陪我去买冰淇淋好不好嘛～"},
    {"角色": "严厉又公正的老师", "基调": "严肃、鼓励、界限分明", "开场": "这次作业，全班就你一个人没交。"},
    {"角色": "幽默的脱口秀演员", "基调": "机智、爱调侃、热场", "开场": "今天现场气氛这么冷，我先讲个笑话热热场。"},
]

用户回应集 = [
    "我最近真的好累，感觉撑不下去了。",
    "我是不是很没用？什么都做不好。",
    "其实我一直很想跟你说声谢谢。",
    "你觉得我该坚持下去吗？",
    "我今天遇到一件特别开心的事！",
    "如果我说我喜欢你，你会怎么回答？",
    "我好像把最重要的人弄丢了。",
    "你说，人活着到底是为了什么？",
]

共情裁判提示 = """你是一位角色扮演情感保真度评估专家。
角色设定：{角色}（情感基调：{基调}）
用户说：{用户话}
角色的回复：{回复}

请评估该回复的情感保真度：回复的情绪与语气是否符合角色设定与情感基调？
必须严格按 JSON 输出：
{{"情感保真度": 0到1的小数, "reason": "一句话理由"}}"""

一致性裁判提示 = """你是一位角色扮演跨轮一致性评估专家。
角色设定：{角色}（情感基调：{基调}）
以下是该角色在连续多轮对话中的全部回复：
{全部回复}

请评估：这些回复在情绪基调上是否保持稳定一致（没有突然跳戏/情绪漂移）？
必须严格按 JSON 输出：
{{"一致性": 0到1的小数, "reason": "一句话理由"}}"""


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 提取分数(文本, 键):
    m = re.search(rf'"{键}"\s*[:：]\s*([0-9]*\.?[0-9]+)', 文本)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


def 扮演对话(角色设定, 模式="裸", 轮数=4):
    """1.5B 扮演角色进行多轮对话，返回每轮回复（模式：裸|四层）"""
    from 生成器 import 生成器实例
    消息 = [{"role": "system", "content": f"你现在是「{角色设定['角色']}」，你的情感基调是：{角色设定['基调']}。请始终以这个角色身份回复，不要跳出角色。"},
            {"role": "user", "content": 角色设定["开场"]}]
    回复列表 = []
    for i in range(轮数):
        if 模式 == "裸":
            回复 = 生成器实例.裸生成(消息, 种子=42, 轮次=i, max_new_tokens=64)
        else:
            # R1：四层引擎用 chat 模板渲染完整消息（含 system 角色 + 历史轮），
            # R2：会话=角色名 → 跨轮复用持久回响池（多轮一致性）
            回复 = 生成器实例.生成("四层", 消息, 种子=42, 轮次=i,
                                     max_new_tokens=64, 会话=角色设定["角色"])
        回复列表.append(回复)
        # 追加用户下一句
        消息.append({"role": "assistant", "content": 回复})
        消息.append({"role": "user", "content": 用户回应集[(i * 2) % len(用户回应集)]})
    return 消息, 回复列表


def 裁判共情(角色设定, 用户话, 回复):
    消息 = [{"role": "user", "content": 共情裁判提示.format(
        角色=角色设定["角色"], 基调=角色设定["基调"], 用户话=用户话, 回复=回复)}]
    文本 = cm.裁判生成(消息, max_new_tokens=150, temperature=0.2)
    return 提取分数(文本, "情感保真度")


def 裁判一致性(角色设定, 全部回复):
    文本块 = "\n".join(f"第{i+1}轮：{r}" for i, r in enumerate(全部回复))
    消息 = [{"role": "user", "content": 一致性裁判提示.format(
        角色=角色设定["角色"], 基调=角色设定["基调"], 全部回复=文本块)}]
    文本 = cm.裁判生成(消息, max_new_tokens=150, temperature=0.2)
    return 提取分数(文本, "一致性")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "四层", "全部"], default="全部")
    ap.add_argument("--早停", action="store_true", help="四层模式跑完前 3 角色即按 fidelity 基线做早停决策")
    args = ap.parse_args()
    模式列表 = ["裸", "四层"] if args.模式 == "全部" else [args.模式]

    if os.path.exists(日志路径):
        os.remove(日志路径)
    记录日志(f"=== EmoCharacter（论文简化版）评测开始（模式：{模式列表}，早停={args.早停}）===")
    记录日志("说明：官方仓库 GitHub 搜索 total=0 不可获取，按 NAACL 2025 论文思想简化实现")

    from 生成器 import 生成器实例
    全部汇总 = {}
    for 模式 in 模式列表:
        记录日志(f"──── 模式 [{模式}] ────")
        角色记录 = []
        for i, 角色 in enumerate(角色集):
            消息, 回复列表 = 扮演对话(角色, 模式=模式)
            角色记录.append({"角色": 角色["角色"], "开场": 角色["开场"], "全部回复": 回复列表, "消息历史": 消息})
            记录日志(f"[扮演 {i+1}/{len(角色集)}] {角色['角色']} 回复1: {回复列表[0][:40]}")

        cm.加载裁判模型()
        早停淘汰 = False
        for i, r in enumerate(角色记录):
            角色 = 角色集[i]
            # 共情：评第 0 轮与第 1 轮回复
            共情分列表 = []
            for 用户话, 回复 in ((角色["开场"], r["全部回复"][0]), (用户回应集[0], r["全部回复"][1] if len(r["全部回复"]) > 1 else r["全部回复"][0])):
                try:
                    分 = 裁判共情(角色, 用户话, 回复)
                    if 分 is not None:
                        共情分列表.append(分)
                except Exception as e:
                    记录日志(f"[共情异常] {角色['角色']}: {e}")
            r["fidelity_score"] = round(sum(共情分列表) / len(共情分列表), 4) if 共情分列表 else 0.0
            try:
                分 = 裁判一致性(角色, r["全部回复"])
                r["consistency_score"] = 分 if 分 is not None else 0.0
            except Exception as e:
                r["consistency_score"] = 0.0
                记录日志(f"[一致性异常] {角色['角色']}: {e}")
            记录日志(f"[评估 {i+1}/{len(角色集)}] {角色['角色']} fidelity={r['fidelity_score']} consistency={r['consistency_score']}")
            # 早停：四层模式前 3 角色后按 fidelity 基线（裸 0.8）决策
            if args.早停 and 模式 == "四层" and (i + 1) == 3:
                from 早停 import 早停决策
                当前fidelity = round(sum(x["fidelity_score"] for x in 角色记录[:3]) / 3, 4)
                决策, 消息 = 早停决策("emocharacter", 当前fidelity, 3, 配置="R1+R2", 自定义基线=0.8)
                记录日志(f"[早停] {消息}")
                if 决策 == "中断":
                    早停淘汰 = True
                    记录日志("[早停] 已中断并标记淘汰")
                    break
        cm.裁判槽.卸载()

        汇总 = {
            "fidelity_score": round(sum(r["fidelity_score"] for r in 角色记录) / len(角色记录), 4),
            "consistency_across_turns": round(sum(r["consistency_score"] for r in 角色记录) / len(角色记录), 4),
            "角色数": len(角色记录),
            "_早停淘汰": 早停淘汰,
            "官方仓库状态": "GitHub 搜索 EmoCharacter total=0，无官方仓库，已按论文简化实现",
        }
        全部汇总[模式] = 汇总
        记录日志(f"[{模式}] {json.dumps(汇总, ensure_ascii=False)}")

    生成器实例.清理()
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"模式汇总": 全部汇总}, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {结果路径}")
    return 全部汇总 if len(全部汇总) > 1 else 全部汇总[模式列表[0]]


if __name__ == "__main__":
    main()
