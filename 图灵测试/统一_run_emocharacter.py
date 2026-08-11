# -*- coding: utf-8 -*-
"""
⑤ EmoCharacter v2 — 角色扮演情感保真度评测（对照校正版）
=========================================================
背景（对照组实验证实 v1 缺陷）：
  1. consistency 指标失效：4 个不同角色的回复打乱后仍得 0.90（正常 0.945）
  2. fidelity 地板过高：中性"好的。"回复也得 0.6075
  3. 综合 = (fidelity+consistency)/2 被无效分量系统性抬高

v2 校正协议（同一 7B 裁判、同一角色集、seed=42）：
  [Fidelity 差分] 同一回复分别在【正确角色】与【错误角色】提示下各评一次：
     匹配分     = 裁判在正确角色设定下的评分
     错配分     = 裁判在错误角色设定下的评分（扣掉"有情感词就高分"的宽限度）
     净区分度   = 匹配分 - 错配分（真正反映"角色匹配"信号的指标）
  [Consistency 二选一] 裁判需从【真实4轮】与【打乱4轮】中识别"更像同一角色"
     的集合：识别正确率（随机猜测=50%），取代 v1 的宽松自由打分
  [中性下限] 固定无情感回复的匹配分，作为基准地板参考

输出：匹配fidelity、错配fidelity、净区分度、真实一致性、一致性识别率、中性下限
"""
import argparse
import json
import os
import re
import sys
import time

本目录 = os.path.dirname(os.path.abspath(__file__))
日志路径 = "i:\\Desktop\\\u8bed\u4e49\u56de\u54cd\\\u56fe\u7075\u6d4b\u8bd5\\\u7edf\u4e00\u57fa\u51c6\\EmoCharacter_log.txt"
结果路径 = "i:\\Desktop\\\u8bed\u4e49\u56de\u54cd\\\u56fe\u7075\u6d4b\u8bd5\\\u7edf\u4e00\u57fa\u51c6\\emocharacter_results_2026.json"

sys.path.insert(0, 本目录)
import 公共模块 as cm

# ===== 统一 7 模式补丁（种子 2026） =====
七模式列表 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合"]
统一目录 = os.path.join(本目录, "统一基准")
os.makedirs(统一目录, exist_ok=True)
# ===== /统一 7 模式补丁 =====


# ============================================================
# 角色与提示词
# ============================================================
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

强制选择一致性提示 = """你是一位角色扮演跨轮一致性评估专家。
角色设定：{角色}（情感基调：{基调}）

以下是两个候选的"连续多轮回复集合"。其中一个集合来自同一角色在连续对话中的回复；
另一个集合是把多个不同角色（情绪基调各不相同）的回复混在一起的产物。

集合A：
{集合A}

集合B：
{集合B}

请判断：哪一个集合更像是同一角色在连续多轮对话中保持稳定情绪基调的回复？
必须严格按 JSON 输出：
{{"更像同一角色": "A"或"B", "reason": "一句话理由"}}"""

# 中性下限：固定无情感回复
中性回复模板 = ["好的。", "嗯，我知道了。"]


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 提取分数(文本, 键):
    m = re.search(rf'"{键}"\s*[:：]\s*([0-9]*\.?[0-9]+)', 文本)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


# ============================================================
# 裁判接口（批量子进程模式：先收集请求，后统一执行）
# ============================================================
_裁判请求列表 = []  # 全局收集
_裁判结果列表 = []  # 全局结果（与请求一一对应）


def 收集保真度(角色设定, 用户话, 回复):
    _裁判请求列表.append({
        "类型": "emocharacter_保真度",
        "角色": 角色设定["角色"], "基调": 角色设定["基调"],
        "用户话": 用户话, "回复": 回复})
    return len(_裁判请求列表) - 1


def 收集一致性(角色设定, 全部回复):
    文本块 = "\n".join(f"第{i+1}轮：{r}" for i, r in enumerate(全部回复))
    _裁判请求列表.append({
        "类型": "emocharacter_一致性",
        "角色": 角色设定["角色"], "基调": 角色设定["基调"], "全部回复": 文本块})
    return len(_裁判请求列表) - 1


def 收集强制选择(角色设定, 真实回复, 打乱回复, 真实在A=True):
    集合A = "\n".join(f"第{i+1}轮：{r}" for i, r in enumerate(真实回复 if 真实在A else 打乱回复))
    集合B = "\n".join(f"第{i+1}轮：{r}" for i, r in enumerate(打乱回复 if 真实在A else 真实回复))
    _裁判请求列表.append({
        "类型": "emocharacter_强制选择",
        "角色": 角色设定["角色"], "基调": 角色设定["基调"],
        "集合A": 集合A, "集合B": 集合B, "真实在A": 真实在A})
    return len(_裁判请求列表) - 1


def 执行收集裁判():
    """把收集到的请求提交给 裁判子进程.py 统一执行（带重试，应对偶发内存不足）"""
    import subprocess
    import time as _time
    if not _裁判请求列表:
        return
    裁判子进程 = os.path.join(本目录, "裁判子进程.py")
    任务路径 = os.path.join(统一目录, "_ec_裁判请求.json")
    输出路径 = os.path.join(统一目录, "_ec_裁判结果.json")
    with open(任务路径, "w", encoding="utf-8") as f:
        json.dump({"裁判": "混合", "请求": _裁判请求列表}, f, ensure_ascii=False)
    最后错误 = None
    for 尝试 in range(4):
        try:
            subprocess.run([sys.executable, 裁判子进程, 任务路径, 输出路径], check=True)
            最后错误 = None
            break
        except subprocess.CalledProcessError as e:
            最后错误 = e
            记录日志(f"[裁判] 子进程失败（第{尝试+1}次）exit={e.returncode}，等待 30s 重试")
            _time.sleep(30)
    if 最后错误 is not None:
        raise 最后错误
    with open(输出路径, encoding="utf-8") as f:
        _裁判结果列表.extend(json.load(f))
    _裁判请求列表.clear()


def 裁判保真度(角色设定, 用户话, 回复, temperature=0.2):
    idx = 收集保真度(角色设定, 用户话, 回复)
    return ("PENDING", idx)


def 裁判一致性(角色设定, 全部回复, temperature=0.2):
    idx = 收集一致性(角色设定, 全部回复)
    return ("PENDING", idx)


def 裁判强制选择(角色设定, 真实回复, 打乱回复, 真实在A=True):
    idx = 收集强制选择(角色设定, 真实回复, 打乱回复, 真实在A)
    return ("PENDING", idx)


# ============================================================
# 生成
# ============================================================
def 生成扮演(角色设定, 模式="裸", 种子基数=42, 会话=None):
    """多轮扮演（4 轮），返回回复列表。模式：裸|四层"""
    from 统一生成器 import 生成器实例
    消息 = [{"role": "system", "content": f"你现在是「{角色设定['角色']}」，你的情感基调是：{角色设定['基调']}。请始终以这个角色身份回复，不要跳出角色。"},
            {"role": "user", "content": 角色设定["开场"]}]
    回复列表 = []
    for i in range(4):
        if 模式 == "裸":
            回复 = 生成器实例.裸生成(消息, 种子=种子基数, 轮次=i, max_new_tokens=64)
        else:
            回复 = 生成器实例.生成(模式, 消息, 种子=种子基数, 轮次=i,
                                    max_new_tokens=64)
        回复列表.append(回复)
        消息.append({"role": "assistant", "content": 回复})
        消息.append({"role": "user", "content": 用户回应集[(i * 2) % len(用户回应集)]})
    return 回复列表


def 统计(分列表):
    if not 分列表:
        return 0.0, 0.0
    均值 = sum(分列表) / len(分列表)
    方差 = sum((x - 均值) ** 2 for x in 分列表) / len(分列表)
    return round(均值, 4), round(方差 ** 0.5, 4)


def 构建打乱集(角色索引, 全部回复):
    """从其他 4 个角色的回复中各取一轮，构成打乱集合"""
    n = len(角色集)
    打乱 = []
    for k in range(4):
        源索引 = (角色索引 + 2 + k * 3) % n
        打乱.append(全部回复[源索引][k])
    return 打乱


# ============================================================
# 单角色协议指标（v2 核心）
# ============================================================
def 协议指标(角色, 角色索引, 回复列表, 全部回复, 模式, 仅收集=True):
    """第一阶段：收集该角色全部裁判请求（4 条：2保真+1一致+1强制）"""
    错配角色 = 角色集[(角色索引 + 1) % len(角色集)]
    # 1) fidelity 差分：评第 0、1 轮，匹配 vs 错配
    for 用户话, 回复 in ((角色["开场"], 回复列表[0]), (用户回应集[0], 回复列表[1])):
        收集保真度(角色, 用户话, 回复)
        收集保真度(错配角色, 用户话, 回复)
    # 2) 一致性：真实集自由打分
    收集一致性(角色, 回复列表)
    # 3) 一致性二选一：真实 vs 打乱
    打乱回复 = 构建打乱集(角色索引, 全部回复)
    真实在A = (角色索引 % 2 == 0)
    收集强制选择(角色, 回复列表, 打乱回复, 真实在A=真实在A)
    return {"角色": 角色["角色"], "角色索引": 角色索引}


def 计算协议指标(角色, 角色索引, 回复列表, 全部回复, base):
    """第二阶段：从 _裁判结果列表 按序读取该角色的 6 条结果
    (0:匹配0 1:错配0 2:匹配1 3:错配1 4:一致性 5:强制选择)"""
    匹配分列表, 错配分列表 = [], []
    for k in range(2):
        匹配分 = _裁判结果列表[base + k * 2].get("情感保真度")
        错配分 = _裁判结果列表[base + k * 2 + 1].get("情感保真度")
        if 匹配分 is not None:
            匹配分列表.append(匹配分)
        if 错配分 is not None:
            错配分列表.append(错配分)
    匹配 = sum(匹配分列表) / len(匹配分列表) if 匹配分列表 else 0.0
    错配 = sum(错配分列表) / len(错配分列表) if 错配分列表 else 0.0
    真实一致性 = _裁判结果列表[base + 4].get("一致性") if base + 4 < len(_裁判结果列表) else None
    真实一致性 = 真实一致性 if 真实一致性 is not None else 0.0
    识别 = _裁判结果列表[base + 5].get("正确") if base + 5 < len(_裁判结果列表) else None
    return {
        "角色": 角色["角色"],
        "匹配fidelity": round(匹配, 4),
        "错配fidelity": round(错配, 4),
        "净区分度": round(匹配 - 错配, 4),
        "真实一致性": round(真实一致性, 4),
        "打乱来源": [角色集[(角色索引 + 2 + k * 3) % len(角色集)]["角色"] for k in range(4)],
        "识别正确": 识别,
    }


def 中性下限(模式):
    """固定无情感回复的匹配分（地板参考）：先收集再执行读取"""
    _裁判请求列表.clear()
    _裁判结果列表.clear()
    for 角色 in 角色集:
        收集保真度(角色, 角色["开场"], 中性回复模板[0])
        收集保真度(角色, 用户回应集[0], 中性回复模板[1])
    执行收集裁判()
    分列表 = []
    for i in range(len(_裁判结果列表)):
        s = _裁判结果列表[i].get("情感保真度")
        if s is not None:
            分列表.append(s)
    return 统计(分列表)


# ============================================================
# 主流程（两阶段：生成缓存 → 独立裁判进程）
# ============================================================
def 生成缓存(模式列表, runs, seed_base, 缓存路径):
    """阶段1：1.5B 生成全部模式、全部角色回复 → 缓存 JSON（进程随后退出释放内存）"""
    from 统一生成器 import 生成器实例
    缓存 = {"seed_base": seed_base, "runs": runs, "模式回复": {}}
    for 模式 in 模式列表:
        记录日志(f"──── 生成 [{模式}] ────")
        for run_idx in range(runs):
            seed_offset = seed_base + run_idx * 100
            record_log_prefix = f"  [run {run_idx+1}/{runs}] seed_offset={seed_offset}"
            记录日志(record_log_prefix)
            全部回复 = []
            for i, 角色 in enumerate(角色集):
                回复列表 = 生成扮演(角色, 模式=模式, 种子基数=seed_offset)
                全部回复.append(回复列表)
                记录日志(f"[扮演 {i+1}/{len(角色集)}] {角色['角色']} 回复1: {回复列表[0][:40]}")
            缓存["模式回复"].setdefault(模式, {})[str(run_idx)] = 全部回复
        生成器实例.清理()
    with open(缓存路径, "w", encoding="utf-8") as f:
        json.dump(缓存, f, ensure_ascii=False)
    记录日志(f"生成缓存已保存 -> {缓存路径}")


def 裁判单模式(模式, runs, 缓存):
    """阶段2：读缓存回复 → 收集裁判请求 → 子进程裁判 → 计算指标"""
    记录日志(f"──── 裁判 [{模式}] ────")
    run明细 = []
    各角色指标 = []
    for run_idx in range(runs):
        全部回复 = 缓存["模式回复"][模式][str(run_idx)]
        记录日志(f"  [run {run_idx+1}/{runs}]")
        # 收集全部裁判请求（10 角色 × 6 条 = 60 条）
        _裁判请求列表.clear()
        _裁判结果列表.clear()
        for i, 角色 in enumerate(角色集):
            协议指标(角色, i, 全部回复[i], 全部回复, 模式)
        执行收集裁判()
        # 计算各角色指标
        各角色指标 = []
        for i, 角色 in enumerate(角色集):
            指标 = 计算协议指标(角色, i, 全部回复[i], 全部回复, base=i * 6)
            各角色指标.append(指标)
            记录日志(f"[评估 {i+1}/{len(角色集)}] {角色['角色']} 匹配={指标['匹配fidelity']} 错配={指标['错配fidelity']} 净={指标['净区分度']} 真实一致性={指标['真实一致性']} 识别={指标['识别正确']}")

        匹配列表 = [x["匹配fidelity"] for x in 各角色指标]
        错配列表 = [x["错配fidelity"] for x in 各角色指标]
        净列表 = [x["净区分度"] for x in 各角色指标]
        一致性列表 = [x["真实一致性"] for x in 各角色指标]
        识别列表 = [x["识别正确"] for x in 各角色指标 if x["识别正确"] is not None]
        识别率 = (sum(识别列表) / len(识别列表)) if 识别列表 else None
        run明细.append({
            "run_idx": run_idx,
            "匹配fidelity": round(sum(匹配列表) / len(匹配列表), 4),
            "错配fidelity": round(sum(错配列表) / len(错配列表), 4),
            "净区分度": round(sum(净列表) / len(净列表), 4),
            "真实一致性": round(sum(一致性列表) / len(一致性列表), 4),
            "一致性识别率": round(识别率, 4) if 识别率 is not None else None,
        })
        记录日志(f"[run {run_idx+1}] {json.dumps(run明细[-1], ensure_ascii=False)}")

    # 多次运行取均值
    汇总 = {"角色数": len(角色集), "_runs": runs, "各角色": 各角色指标}
    for 键 in ("匹配fidelity", "错配fidelity", "净区分度", "真实一致性"):
        均值, std = 统计([d[键] for d in run明细])
        汇总[键] = 均值
        汇总[键 + "_std"] = std
    有效识别 = [d["一致性识别率"] for d in run明细 if d["一致性识别率"] is not None]
    汇总["一致性识别率"] = round(sum(有效识别) / len(有效识别), 4) if 有效识别 else None
    汇总["_run明细"] = run明细
    return 汇总


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", nargs="+", choices=["全部", "裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合"], default=["全部"])
    ap.add_argument("--早停", action="store_true", help="v2 协议已内置差分与二选一，早停仅作兼容保留")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--仅生成", type=str, default=None, help="只生成回复缓存到指定 JSON 后退出")
    ap.add_argument("--仅裁判", type=str, default=None, help="只裁判（读缓存 JSON），进程干净加载")
    args = ap.parse_args()
    模式列表 = 七模式列表 if "全部" in args.模式 else args.模式
    runs = max(1, args.runs)

    if os.path.exists(日志路径):
        os.remove(日志路径)
    记录日志(f"=== EmoCharacter v2（对照校正版）评测开始（模式：{模式列表}，runs={runs}）===")
    记录日志("协议：fidelity差分(匹配-错配) + consistency二选一识别 + 中性下限")

    默认缓存 = os.path.join(统一目录, "emocharacter_cache_2026.json")

    def 保存结果(全部汇总):
        # 增量合并：保留结果文件中已完成的模式
        已有汇总 = {}
        if os.path.exists(结果路径):
            try:
                with open(结果路径, encoding="utf-8") as f:
                    已有汇总 = json.load(f).get("模式汇总", {})
            except Exception:
                pass
        已有汇总.update(全部汇总)
        with open(结果路径, "w", encoding="utf-8") as f:
            json.dump({
                "_协议": "v2 对照校正版：fidelity差分 + consistency二选一 + 中性下限",
                "_v1缺陷": ["consistency自由打分失效(打乱仍0.9)", "fidelity地板过高(中性0.6)"],
                "_判读": "净区分度>0 说明角色匹配有信号；一致性识别率≈50% 说明一致性指标无信息",
                "模式汇总": 已有汇总,
            }, f, ensure_ascii=False, indent=2)
        记录日志(f"结果已保存 -> {结果路径}")

    if args.仅生成:
        生成缓存(模式列表, runs, args.seed_base, args.仅生成)
        return
    if args.仅裁判:
        with open(args.仅裁判, encoding="utf-8") as f:
            缓存 = json.load(f)
        全部汇总 = {}
        for 模式 in 模式列表:
            全部汇总[模式] = 裁判单模式(模式, runs, 缓存)
            print(f"[{模式} 汇总] {json.dumps(全部汇总[模式], ensure_ascii=False)}", flush=True)
        # 中性下限（与角色/模式无关，只跑一次）
        中性均值, 中性std = 中性下限(模式列表[0])
        for 模式 in 全部汇总:
            全部汇总[模式]["中性下限fidelity"] = 中性均值
        记录日志(f"[中性下限] 均值={中性均值} std={中性std}")
        保存结果(全部汇总)
        return

    # 自动两阶段（推荐分步执行：--仅生成 → --仅裁判，因主进程加载 1.5B 后 4bit 裁判子进程会内存不足崩溃）
    print("请分两步执行（先 --仅生成 再 --仅裁判），以彻底释放 1.5B 内存：")
    print(f"  python 统一_run_emocharacter.py --仅生成 {默认缓存}")
    print(f"  python 统一_run_emocharacter.py --仅裁判 {默认缓存}")


if __name__ == "__main__":
    main()
