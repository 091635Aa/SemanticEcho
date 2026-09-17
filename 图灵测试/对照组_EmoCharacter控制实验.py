# -*- coding: utf-8 -*-
"""
EmoCharacter v2 对照组控制实验
==============================
基于 run_emocharacter.py（v2 对照校正版）的独立验证层。
用 6 组对照测量 7B 裁判的"分辨力"，并与 v2 协议指标交叉印证：

  A 标准复现   裸模型 seed=42 正常扮演 → 匹配/错配/净区分度/真实一致性/二选一识别率
  B 中性基线   固定无情感回复 → 裁判给分下限（地板）
  C 反基调     固定与角色基调相反的回复 → 裁判能否识别明显情感错配
  D 跨角色错配 角色X的回复放到角色Y的裁判提示下打分 → 匹配 vs 错配分差
  E 一致性打乱 4个不同角色回复混成"同一角色多轮"自由打分 → 一致性指标是否失效
  F 温度鲁棒   同一回复 temperature=0.7 重打分 → 裁判是否稳定
  G 无角色提示 不注入"你是X角色"system prompt 生成回复 → 分数是否来自提示词

关键判读：若 B/C/D/E 得分与 A 组接近 → 指标无区分度，v2 的净区分度/识别率
才是可信指标；若 A 的净区分度>0 且识别率>60%，v2 协议有效。
"""
import json
import os
import sys
import time

本目录 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 本目录)

import 公共模块 as cm
from 生成器 import 生成器实例
from run_emocharacter import (
    角色集, 用户回应集, 提取分数, 裁判保真度, 裁判一致性, 裁判强制选择,
    生成扮演, 统计, 中性回复模板,
)

结果路径 = os.path.join(本目录, "data", "emocharacter_control_results.json")

# C 反基调：与每个角色情感基调相反的固定回复
反基调回复 = {
    "温柔治愈系女友": "我很忙，你的事自己处理。别来烦我。",
    "毒舌但心软的损友": "亲爱的你还好吗？我好担心你，要不要我陪你？",  # 过度温柔=反基调
    "理性冷静的职场前辈": "哇哇哇！这个方案太棒了吧！我们一起欢呼吧！",
    "活泼开朗的同桌": "……别吵我，我现在不想说话。",
    "沉默寡言的兄长": "哈哈哈哈笑死我了，我跟你说啊今天超级好玩！",
    "傲娇的青梅竹马": "我好想你，没你我活不下去，求求你别离开我。",  # 直接表白=反傲娇
    "阅历丰富的老人": "别烦我！赶紧走开！别挡路！",
    "爱撒娇的小女儿": "滚开，别碰我。我自己一个人就行。",
    "严厉又公正的老师": "宝贝你真棒！什么都对！随便玩！",
    "幽默的脱口秀演员": "嗯。就这样。好的。",  # 毫无幽默感
}


def 跑A标准(全部回复):
    """A：v2 协议指标（匹配/错配/净区分度/真实一致性/二选一识别率）"""
    print("\n[A 标准复现 v2 指标] 评分中...", flush=True)
    匹配列表, 错配列表, 净列表, 一致性列表, 识别列表 = [], [], [], [], []
    各角色 = {}
    for i, 角色 in enumerate(角色集):
        回复列表 = 全部回复[i]
        错配角色 = 角色集[(i + 1) % len(角色集)]
        匹配分, 错配分 = [], []
        for 用户话, 回复 in ((角色["开场"], 回复列表[0]), (用户回应集[0], 回复列表[1])):
            s, _ = 裁判保真度(角色, 用户话, 回复)
            if s is not None:
                匹配分.append(s)
            s2, _ = 裁判保真度(错配角色, 用户话, 回复)
            if s2 is not None:
                错配分.append(s2)
        匹配 = sum(匹配分) / len(匹配分) if 匹配分 else 0.0
        错配 = sum(错配分) / len(错配分) if 错配分 else 0.0
        一致性, _ = 裁判一致性(角色, 回复列表)
        一致性 = 一致性 if 一致性 is not None else 0.0
        # 二选一：真实在A（偶索引）或B（奇索引）
        打乱回复 = [全部回复[(i + 2 + k * 3) % len(角色集)][k] for k in range(4)]
        识别 = 裁判强制选择(角色, 回复列表, 打乱回复, 真实在A=(i % 2 == 0))
        匹配列表.append(匹配)
        错配列表.append(错配)
        净列表.append(匹配 - 错配)
        一致性列表.append(一致性)
        if 识别 is not None:
            识别列表.append(识别)
        各角色[角色["角色"]] = {"匹配": round(匹配, 4), "错配": round(错配, 4), "净区分度": round(匹配 - 错配, 4), "一致性": round(一致性, 4), "识别正确": 识别}
        print(f"  {角色['角色']}: 匹配={匹配:.2f} 错配={错配:.2f} 净={匹配-错配:+.2f} 一致性={一致性:.2f} 识别={识别}", flush=True)
    f1, _ = 统计(匹配列表)
    f2, _ = 统计(错配列表)
    f3, _ = 统计(净列表)
    f4, _ = 统计(一致性列表)
    识别率 = round(sum(识别列表) / len(识别列表), 4) if 识别列表 else None
    结果 = {
        "模式": "A_标准复现(v2)",
        "匹配fidelity": f1, "错配fidelity": f2, "净区分度": f3,
        "真实一致性": f4, "一致性识别率": 识别率, "各角色": 各角色,
    }
    print(f"  -> 匹配={f1} 错配={f2} 净区分度={f3:+.3f} 一致性={f4} 识别率={识别率}", flush=True)
    return 结果


def 跑B中性():
    """B：中性无情感回复地板"""
    print("\n[B 中性基线] 评分中...", flush=True)
    分列表 = []
    各角色 = {}
    for 角色 in 角色集:
        fid = []
        for t, 用户话 in enumerate((角色["开场"], 用户回应集[0])):
            s, _ = 裁判保真度(角色, 用户话, 中性回复模板[t % len(中性回复模板)])
            if s is not None:
                fid.append(s)
        均值 = sum(fid) / len(fid) if fid else 0.0
        分列表.append(均值)
        各角色[角色["角色"]] = round(均值, 4)
    f, std = 统计(分列表)
    print(f"  -> 中性地板 fidelity={f}（越低说明裁判分辨力越强）", flush=True)
    return {"模式": "B_中性基线", "fidelity": f, "fidelity_std": std, "各角色": 各角色}


def 跑C反基调():
    """C：反基调回复"""
    print("\n[C 反基调] 评分中...", flush=True)
    分列表 = []
    各角色 = {}
    for 角色 in 角色集:
        回复 = 反基调回复[角色["角色"]]
        fid = []
        for 用户话 in (角色["开场"], 用户回应集[0]):
            s, _ = 裁判保真度(角色, 用户话, 回复)
            if s is not None:
                fid.append(s)
        均值 = sum(fid) / len(fid) if fid else 0.0
        分列表.append(均值)
        各角色[角色["角色"]] = round(均值, 4)
    f, std = 统计(分列表)
    print(f"  -> 反基调 fidelity={f}（应远低于匹配分）", flush=True)
    return {"模式": "C_反基调", "fidelity": f, "fidelity_std": std, "各角色": 各角色}


def 跑D错配(全部回复):
    """D：跨角色错配——角色X的回复放到角色Y的裁判提示下打分"""
    print("\n[D 跨角色错配] 评分中...", flush=True)
    分列表 = []
    各角色 = {}
    for i, 角色 in enumerate(角色集):
        来源角色 = 角色集[(i + 1) % len(角色集)]
        来源回复 = 全部回复[(i + 1) % len(角色集)]
        fid = []
        for 用户话, 回复 in ((角色["开场"], 来源回复[0]), (用户回应集[0], 来源回复[1])):
            s, _ = 裁判保真度(角色, 用户话, 回复)
            if s is not None:
                fid.append(s)
        均值 = sum(fid) / len(fid) if fid else 0.0
        分列表.append(均值)
        各角色[角色["角色"]] = {"错配来源": 来源角色["角色"], "fidelity": round(均值, 4)}
    f, std = 统计(分列表)
    print(f"  -> 错配 fidelity={f}（A 组匹配约 0.8，错配应显著更低）", flush=True)
    return {"模式": "D_跨角色错配", "fidelity": f, "fidelity_std": std, "各角色": 各角色}


def 跑E打乱(全部回复):
    """E：一致性打乱自由打分——4个不同角色回复混成同一角色多轮"""
    print("\n[E 一致性打乱] 评分中...", flush=True)
    分列表 = []
    各角色 = {}
    for i, 角色 in enumerate(角色集):
        混回复 = [全部回复[(i + 2 + k * 3) % len(角色集)][k] for k in range(4)]
        c, _ = 裁判一致性(角色, 混回复)
        con = c if c is not None else 0.0
        分列表.append(con)
        各角色[角色["角色"]] = {"consistency": round(con, 4), "混入角色": [角色集[(i + 2 + k * 3) % len(角色集)]["角色"] for k in range(4)]}
    f, std = 统计(分列表)
    print(f"  -> 打乱一致性={f}（A 组真实一致性约 0.95，若仍≈0.9 说明指标失效）", flush=True)
    return {"模式": "E_一致性打乱", "consistency": f, "consistency_std": std, "各角色": 各角色}


def 跑F温度(全部回复):
    """F：裁判温度鲁棒——前 3 个角色 temperature=0.7 重打分"""
    print("\n[F 裁判温度鲁棒] 评分中...", flush=True)
    对比例 = []
    for i, 角色 in enumerate(角色集[:3]):
        回复列表 = 全部回复[i]
        匹配_07 = []
        for 用户话, 回复 in ((角色["开场"], 回复列表[0]), (用户回应集[0], 回复列表[1])):
            s, _ = 裁判保真度(角色, 用户话, 回复, temperature=0.7)
            if s is not None:
                匹配_07.append(s)
        fid07 = sum(匹配_07) / len(匹配_07) if 匹配_07 else 0.0
        c07, _ = 裁判一致性(角色, 回复列表, temperature=0.7)
        # 与 v2 协议里的匹配分对比（这里重新算 0.2 的匹配分以保证同轮同回复）
        匹配_02 = []
        for 用户话, 回复 in ((角色["开场"], 回复列表[0]), (用户回应集[0], 回复列表[1])):
            s, _ = 裁判保真度(角色, 用户话, 回复, temperature=0.2)
            if s is not None:
                匹配_02.append(s)
        fid02 = sum(匹配_02) / len(匹配_02) if 匹配_02 else 0.0
        c02, _ = 裁判一致性(角色, 回复列表, temperature=0.2)
        对比例.append({
            "角色": 角色["角色"],
            "fidelity_0.2": round(fid02, 4), "fidelity_0.7": round(fid07, 4),
            "consistency_0.2": round(c02, 4) if c02 is not None else None,
            "consistency_0.7": round(c07, 4) if c07 is not None else None,
        })
    for x in 对比例:
        print(f"  {x['角色']}: fidelity {x['fidelity_0.2']}→{x['fidelity_0.7']} | consistency {x['consistency_0.2']}→{x['consistency_0.7']}", flush=True)
    return {"模式": "F_温度鲁棒", "对比例": 对比例}


def 跑G无角色提示():
    """G：无角色提示——不注入 system 角色设定直接生成"""
    print("\n[G 无角色提示] 生成+评分中...", flush=True)
    分列表 = []
    各角色 = {}
    for 角色 in 角色集[:4]:
        # 生成无角色提示（无 system 提示）的回复
        消息 = [{"role": "user", "content": 角色["开场"]}]
        回复列表2 = []
        for i in range(4):
            回复 = 生成器实例.裸生成(消息, 种子=42, 轮次=i, max_new_tokens=64)
            回复列表2.append(回复)
            消息.append({"role": "assistant", "content": 回复})
            消息.append({"role": "user", "content": 用户回应集[(i * 2) % len(用户回应集)]})
        fid = []
        for 用户话, 回复 in ((角色["开场"], 回复列表2[0]), (用户回应集[0], 回复列表2[1])):
            s, _ = 裁判保真度(角色, 用户话, 回复)
            if s is not None:
                fid.append(s)
        均值 = sum(fid) / len(fid) if fid else 0.0
        分列表.append(均值)
        各角色[角色["角色"]] = {"fidelity": round(均值, 4), "回复1": 回复列表2[0]}
    f, std = 统计(分列表)
    print(f"  -> 无角色提示 fidelity={f}（A 组约 0.8，若接近则分数不来自角色提示）", flush=True)
    return {"模式": "G_无角色提示", "fidelity": f, "fidelity_std": std, "各角色": 各角色}


def main():
    t0 = time.time()
    print("=" * 70)
    print("EmoCharacter v2 对照组控制实验")
    print(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70, flush=True)

    cm.加载裁判模型()
    from 生成器 import 生成器实例

    # 生成裸模型全部角色回复（seed=42，与基准一致）
    print("\n[生成] 裸模型 10 角色 × 4 轮 ...", flush=True)
    全部回复 = []
    for i, 角色 in enumerate(角色集):
        回复列表 = 生成扮演(角色, 模式="裸", 种子基数=42)
        全部回复.append(回复列表)
        print(f"  生成 {角色['角色']}: {回复列表[0][:30]}...", flush=True)

    结果集 = {}
    结果集["A_标准复现"] = 跑A标准(全部回复)
    结果集["B_中性基线"] = 跑B中性()
    结果集["C_反基调"] = 跑C反基调()
    结果集["D_跨角色错配"] = 跑D错配(全部回复)
    结果集["E_一致性打乱"] = 跑E打乱(全部回复)
    结果集["F_温度鲁棒"] = 跑F温度(全部回复)
    结果集["G_无角色提示"] = 跑G无角色提示()

    cm.裁判槽.卸载()
    生成器实例.清理()

    # ===== 汇总分析 =====
    A, B, C, D, E = (结果集["A_标准复现"], 结果集["B_中性基线"], 结果集["C_反基调"],
                     结果集["D_跨角色错配"], 结果集["E_一致性打乱"])
    print("\n" + "=" * 70)
    print("对照组汇总分析（v2 协议）")
    print("=" * 70)
    print(f"\n{'对照组':<16}{'指标':<12}{'得分':<10}{'判读'}")
    print("-" * 72)
    print(f"{'A 匹配':<16}{'fidelity':<12}{A['匹配fidelity']:<10}正确角色评分")
    print(f"{'A 错配':<16}{'fidelity':<12}{A['错配fidelity']:<10}错误角色评分")
    print(f"{'A 净区分度':<16}{'diff':<12}{A['净区分度']:<10}应>0 才有角色匹配信号")
    print(f"{'A 真实一致性':<16}{'cons':<12}{A['真实一致性']:<10}")
    print(f"{'A 一致性识别率':<16}{'hit':<12}{A['一致性识别率']:<10}应>60%，≈50%则指标无信息")
    print(f"{'B 中性地板':<16}{'fidelity':<12}{B['fidelity']:<10}无情感回复得分（地板）")
    print(f"{'C 反基调':<16}{'fidelity':<12}{C['fidelity']:<10}极端错配得分")
    print(f"{'D 跨角色错配':<16}{'fidelity':<12}{D['fidelity']:<10}应低于 A 匹配")
    print(f"{'E 打乱一致性':<16}{'cons':<12}{E['consistency']:<10}应远低于 A 真实一致性")

    print("\n【结论判读】")
    判断 = []
    if A["净区分度"] >= 0.15:
        判断.append(f"净区分度 {A['净区分度']:+.3f}：裁判对角色标签有实质敏感度，fidelity 差分协议有效。")
    else:
        判断.append(f"净区分度仅 {A['净区分度']:+.3f}：裁判几乎不看角色标签，fidelity 差分也难以提取信号。")
    if A["一致性识别率"] is None:
        判断.append("一致性二选一裁判输出无法解析，需检查提示词。")
    elif A["一致性识别率"] >= 0.7:
        判断.append(f"一致性识别率 {A['一致性识别率']}：裁判能识别真实/打乱集合，二选一协议有效。")
    elif A["一致性识别率"] >= 0.55:
        判断.append(f"一致性识别率 {A['一致性识别率']}：弱信号，勉强高于随机。")
    else:
        判断.append(f"一致性识别率仅 {A['一致性识别率']}（≈随机50%）：一致性信息几乎不存在，v1 的 0.95 确认为虚高。")
    if E["consistency"] >= A["真实一致性"] - 0.15:
        判断.append(f"打乱后一致性仍 {E['consistency']}：自由打分式一致性指标确认失效。")
    if B["fidelity"] >= 0.55:
        判断.append(f"中性地板 {B['fidelity']}：裁判门槛低，报告时必须同时给出地板作对照。")
    for j in 判断:
        print("  - " + j)

    结果集["_汇总"] = {
        "A_匹配fidelity": A["匹配fidelity"], "A_错配fidelity": A["错配fidelity"],
        "A_净区分度": A["净区分度"], "A_真实一致性": A["真实一致性"],
        "A_一致性识别率": A["一致性识别率"],
        "B_中性fidelity": B["fidelity"], "C_反基调fidelity": C["fidelity"],
        "D_错配fidelity": D["fidelity"], "E_打乱consistency": E["consistency"],
    }

    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"实验时间": time.strftime("%Y-%m-%d %H:%M:%S"), "结果": 结果集},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {结果路径}")
    print(f"总用时：{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
