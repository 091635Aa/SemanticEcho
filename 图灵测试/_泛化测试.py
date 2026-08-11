# -*- coding: utf-8 -*-
"""多模型泛化测试：EDD 作为零权重插件在不同模型上的效果
==================================================
对每个目标模型（Qwen2.5-3B / Qwen3-1.7B）：
  1. 生成 裸 vs P6_情感导演 在 30 条 LLM-Judge 样本上的回复
  2. 复用 7B 裁判子进程 做配对+评分
  3. 输出 win_rate / rating 对比（EDD 相对裸的正向提升即泛化证据）

用法: python _泛化测试.py [模型名...]
"""
import os
import sys
import json
import re
import subprocess
import gc

本目录 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 本目录)
KV核心 = r"c:\Users\Administrator\Documents\KV 情感共振解码\核心"
if KV核心 not in sys.path:
    sys.path.insert(0, KV核心)
ETD目录 = r"h:\情感潮汐解码（Emotion Tidal Decoding, ETD）"
if ETD目录 not in sys.path:
    sys.path.insert(0, ETD目录)
统一目录 = os.path.join(本目录, "统一基准")
样本路径 = os.path.join(本目录, "样本_30条.json")
模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
输出路径 = os.path.join(统一目录, "泛化测试_2026.json")

目标模型 = sys.argv[1:] if len(sys.argv) > 1 else ["Qwen2.5-3B-Instruct", "Qwen3-1.7B-Instruct"]


def 构建生成器(模型名):
    """参数化 统一生成器：指定模型名，仅注册 裸 + P6_情感导演"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from 锚点库 import 锚点库
    from 目标决策器 import 目标决策器
    from 超融合解码器 import 超融合解码器
    from 情感共振解码器 import 情感共振解码器
    from 情感导演解码器 import 情感导演解码器

    class _G:
        def __init__(self):
            self._模型 = None
            self._分词器 = None
            self._解码器 = {}
            self._锚库 = None
            self._设备 = "cuda" if torch.cuda.is_available() else "cpu"

        def _加载(self):
            print(f"[泛化] 加载 {模型名} ...")
            路径 = os.path.join(模型空间, 模型名)
            self._分词器 = AutoTokenizer.from_pretrained(路径, trust_remote_code=True)
            self._模型 = AutoModelForCausalLM.from_pretrained(
                路径, dtype=torch.float16, trust_remote_code=True,
                low_cpu_mem_usage=True).to(self._设备)
            self._模型.eval()
            torch.cuda.empty_cache()
            self._锚库 = 锚点库(self._模型, self._分词器)
            self._锚库.构建()
            感知器, 潮汐决策 = None, None
            try:
                from 潮汐感知器 import 潮汐感知器
                from 潮汐决策器 import 潮汐决策器
                感知器 = 潮汐感知器()
                潮汐决策 = 潮汐决策器(感知器)
            except Exception as e:
                print(f"  [泛化] 潮汐不可用: {e}")
            目标决策 = 目标决策器(感知器=感知器, 潮汐决策器=潮汐决策, 锚点库=self._锚库)
            self._解码器["P6_情感导演"] = 情感导演解码器(
                self._模型, self._分词器, self._锚库, 目标决策,
                开启DMR=True, 开启KV调制=True, 开启V调制=True,
                开启锚点偏置=True, 开启DSA=True,
                α基=0.18, κ基=0.20, κ_v基=0.12, β基=0.6,
                AI腔抑制强度=2.0, 口语化强度=0.6,
                任务自适应=True, 进度调度=True, 在线纠正=True,
                句子停止=True)
            self._锚库.W_e = None
            self._锚库._有效权重 = None
            self._锚库.权重 = None
            torch.cuda.empty_cache()
            print(f"  [泛化] 就绪")

        def 生成(self, 模式, 消息, 种子=42, 轮次=0, max_new_tokens=64):
            self._加载()
            种子值 = 种子 + 轮次
            torch.manual_seed(种子值)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(种子值)
            if 模式 == "P6_情感导演":
                if not any(x["role"] == "system" for x in 消息):
                    消息 = [{"role": "system",
                             "content": ("你是真实的人类，正在和朋友聊天。你的回复要简短自然、口语化，"
                                         "像微信消息，通常一到两句话，可以有情绪和小俏皮，但绝不要提及你"
                                         "是AI、模型、助手或机器人，也不要使用「您」这样的敬语。")}] + 消息
            提示 = self._分词器.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
            inputs = self._分词器(提示, return_tensors="pt").to(self._设备)
            if 模式 == "裸":
                out = self._模型.generate(
                    inputs.input_ids, max_new_tokens=max_new_tokens,
                    pad_token_id=self._分词器.eos_token_id,
                    temperature=1.0, top_p=0.9, top_k=50,
                    repetition_penalty=1.05, do_sample=True)
                新 = out[0, inputs.input_ids.shape[1]:]
                return self._分词器.decode(新, skip_special_tokens=True).strip()
            解码 = self._解码器["P6_情感导演"]
            out, _ = 解码.生成(
                inputs.input_ids, max_new_tokens=max_new_tokens,
                eos_token_id=self._分词器.eos_token_id,
                用户文本=消息[-1]["content"], temperature=1.0, top_p=0.9,
                top_k=50, repetition_penalty=1.05)
            新 = out[0, inputs.input_ids.shape[1]:]
            return self._分词器.decode(新, skip_special_tokens=True).strip()

        def 清理(self):
            import gc
            for d in self._解码器.values():
                try:
                    d._移除钩子()
                except Exception:
                    pass
            for d in self._解码器.values():
                try:
                    d.model = None
                except Exception:
                    pass
            self._解码器 = {}
            self._模型 = None
            self._分词器 = None
            gc.collect()
            torch.cuda.empty_cache()

    return _G()


def 跑裁判(裁判类型, 请求列表):
    """调用 裁判子进程.py 并返回结果列表"""
    任务路径 = os.path.join(统一目录, "_泛化_任务.json")
    输出路径 = os.path.join(统一目录, "_泛化_输出.json")
    with open(任务路径, "w", encoding="utf-8") as f:
        json.dump({"裁判": 裁判类型, "请求": 请求列表}, f, ensure_ascii=False)
    r = subprocess.run([sys.executable, os.path.join(本目录, "裁判子进程.py"),
                        任务路径, 输出路径], check=True)
    with open(输出路径, encoding="utf-8") as f:
        return json.load(f)


def 评估(模型名):
    """对单个模型：生成裸+P6 → 裁判 → 返回指标"""
    print(f"\n========== 模型 {模型名} ==========")
    生成器 = 构建生成器(模型名)
    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"]
    seed_base = 2026
    回复 = {}
    for 模式 in ("裸", "P6_情感导演"):
        列表 = []
        for i, r in enumerate(样本):
            消息 = [{"role": "user", "content": r["user"]}]
            try:
                文本 = 生成器.生成(模式, 消息, 种子=seed_base, 轮次=i, max_new_tokens=64)
            except Exception as e:
                文本 = f"[生成失败:{e}]"
                print(f"  [{模式} {i+1}/30] 失败: {e}", flush=True)
            # Qwen3 等输出 <think> 思考块 → 截断只取正文（EDD 已抑制，裸模式保留思考块以体现对比）
            if 文本.startswith("<think>"):
                文本 = "（思考）" + 文本
            列表.append(文本)
            print(f"  [{模式} {i+1}/30] {r['user'][:14]} => {文本[:36]}", flush=True)
        回复[模式] = 列表
    生成器.清理()

    汇总 = {}
    for 模式 in ("裸", "P6_情感导演"):
        # 配对（双投）
        配对请求 = []
        for i, r in enumerate(样本):
            for AI在前 in (True, False):
                配对请求.append({"user": r["user"], "AI": 回复[模式][i],
                                  "真人": r["girl"], "AI在前": AI在前})
        配对结果 = 跑裁判("llm_judge_配对", 配对请求)
        ai胜 = 0
        for i, r in enumerate(样本):
            # 双投中任一次判 AI 胜 → 该条算 AI 胜
            a = 配对结果[i * 2]
            b = 配对结果[i * 2 + 1]
            if (a.get("AI胜") or b.get("AI胜")):
                ai胜 += 1
        # 评分
        评分请求 = []
        for i, r in enumerate(样本):
            评分请求.append({"user": r["user"], "回复": 回复[模式][i]})
        评分结果 = 跑裁判("llm_judge_评分", 评分请求)
        ai评分 = []
        for cr in 评分结果:
            s = cr.get("评分")
            if s:
                ai评分.append(s)
        win = round(ai胜 / len(样本), 4)
        rating = round(sum(ai评分) / len(ai评分) / 5, 4) if ai评分 else 0.0
        汇总[模式] = {"win_rate_against_human": win, "average_rating": rating,
                     "average_rating_raw": round(sum(ai评分) / len(ai评分), 2) if ai评分 else 0.0}
        print(f"  [{模型名} {模式}] win={win} rating={rating}")
    return 汇总


def main():
    全部 = {}
    for 模型名 in 目标模型:
        全部[模型名] = 评估(模型名)
    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump({"目标模型": 目标模型, "样本数": 30, "模式汇总": 全部,
                   "说明": "P6_情感导演 vs 裸 在各模型上的 LLM-Judge（win_rate 与 rating）"},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {输出路径}")
    print("\n=== 泛化汇总 ===")
    for 模型名, d in 全部.items():
        for 模式 in ("裸", "P6_情感导演"):
            s = d[模式]
            print(f"  {模型名} {模式}: win={s['win_rate_against_human']} rating={s['average_rating_raw']}")


if __name__ == "__main__":
    main()
