# -*- coding: utf-8 -*-
"""多模型泛化测试 v3：P1~P6 全模式 × 任意模型

用法:
  python _泛化测试全模式.py <模型名> --仅生成 [缓存路径]        # 阶段1：生成回复缓存
  python _泛化测试全模式.py <模型名> --仅裁判 [缓存路径] [模式]  # 阶段2：LLM-Judge 裁判

模式列表：裸 / P1_语义回响 / P1.5_兼容层 / P2.5_潮汐 / P3_锚点回响 / P4_KV共振 / P5_超融合 / P6_情感导演
断点：按 模型×模式 缓存，已生成/已裁判的跳过。
结果：统一基准\\泛化测试_2026.json（增量合并，key=模型名 -> {模式: {win_rate, rating}}）
"""
import os
import sys
import json
import gc
import subprocess
import traceback
from datetime import datetime

本目录 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 本目录)
KV核心 = r"c:\Users\Administrator\Documents\KV 情感共振解码\核心"
if KV核心 not in sys.path:
    sys.path.insert(0, KV核心)
ETD目录 = r"h:\情感潮汐解码（Emotion Tidal Decoding, ETD）"
if ETD目录 not in sys.path:
    sys.path.insert(0, ETD目录)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 锚点解码器 import 锚点解码器
from 情感共振解码器 import 情感共振解码器
from 超融合解码器 import 超融合解码器
from 情感导演解码器 import 情感导演解码器
from 统一生成器 import _GPU回响注入器
from semantic_echo.回响池 import 语义回响池
from semantic_echo.情感过滤器 import 情感过滤器

统一目录 = os.path.join(本目录, "统一基准")
os.makedirs(统一目录, exist_ok=True)
样本路径 = os.path.join(本目录, "样本_30条.json")
输出路径 = os.path.join(统一目录, "泛化测试_2026.json")

模型空间A = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
模型空间B = r"l:\模型空间"
全模式列表 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合", "P6_情感导演"]

P6人格注入提示 = (
    "你是真实的人类，正在和朋友聊天。你的回复要简短自然、口语化，像微信消息，"
    "通常一到两句话，可以有情绪和小俏皮，但绝不要提及你是AI、模型、助手或机器人，"
    "也不要使用「您」这样的敬语。"
)


def 定位模型路径(模型名):
    for 空间 in (模型空间A, 模型空间B):
        p = os.path.join(空间, 模型名)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            return p
    raise FileNotFoundError(f"模型 {模型名} 未找到（已检查 {模型空间A} 与 {模型空间B}）")


class 泛化生成器:
    def __init__(self, 模型名):
        self.模型名 = 模型名
        self._模型 = None
        self._分词器 = None
        self._解码器 = {}
        self._回响 = None
        self._锚库 = None
        self._设备 = "cuda" if torch.cuda.is_available() else "cpu"

    def _加载(self):
        if self._模型 is not None:
            return
        路径 = 定位模型路径(self.模型名)
        print(f"[泛化] 加载 {self.模型名} fp16 ...", flush=True)
        self._分词器 = AutoTokenizer.from_pretrained(路径, trust_remote_code=True)
        self._模型 = AutoModelForCausalLM.from_pretrained(
            路径, torch_dtype=torch.float16, trust_remote_code=True,
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
            print(f"  [泛化] 潮汐感知器不可用: {e}")
        目标决策 = 目标决策器(感知器=感知器, 潮汐决策器=潮汐决策, 锚点库=self._锚库)

        # P1 回响注入器
        池 = 语义回响池(int(self._模型.config.hidden_size))
        过滤 = None
        try:
            过滤 = 情感过滤器()
            过滤.加载词库()
        except Exception:
            pass
        self._回响 = _GPU回响注入器(self._模型, 池, lambda_strength=0.29,
                                   情感过滤器实例=过滤)

        # 各方案解码器（与 统一生成器.py 完全一致）
        self._解码器["P1.5_兼容层"] = 锚点解码器(self._模型, self._分词器, self._锚库, 目标决策,
                                          β=None, 句子停止=True)
        self._解码器["P3_锚点回响"] = 锚点解码器(self._模型, self._分词器, self._锚库, 目标决策,
                                           β=0.8, T_anchor=0.3, 句子停止=True)
        self._解码器["P4_KV共振"] = 情感共振解码器(self._模型, self._分词器, self._锚库, 目标决策,
                                              开启KV调制=True, 开启V调制=False, 开启DSA=True,
                                              κ基=0.15, 情感阈值=0.08, 调制层数=4, 句子停止=True)
        self._解码器["P5_超融合"] = 超融合解码器(self._模型, self._分词器, self._锚库, 目标决策,
                                            开启DSA=True, 开启DMR=True, 开启锚点偏置=False,
                                            α基=0.15, α倍率=1.0, T_emo=0.5, 句子停止=True)
        self._解码器["P6_情感导演"] = 情感导演解码器(self._模型, self._分词器, self._锚库, 目标决策,
                                               开启DMR=True, 开启KV调制=True, 开启V调制=True,
                                               开启锚点偏置=True, 开启DSA=True,
                                               α基=0.18, κ基=0.20, κ_v基=0.12, β基=0.6,
                                               AI腔抑制强度=2.0, 口语化强度=0.6,
                                               任务自适应=True, 进度调度=True, 在线纠正=True,
                                               句子停止=True)
        try:
            from 潮汐解码器 import 潮汐解码器
            if 感知器 is not None:
                self._解码器["P2.5_潮汐"] = 潮汐解码器(self._模型, self._分词器, 感知器, 潮汐决策)
        except Exception as e:
            print(f"  [泛化] P2.5 潮汐不可用: {e}")

        self._锚库.W_e = None
        self._锚库._有效权重 = None
        self._锚库.权重 = None
        torch.cuda.empty_cache()
        print(f"  [泛化] 就绪，可用模式：{list(self._解码器.keys())}", flush=True)

    def 生成(self, 模式, 消息, 种子=42, 轮次=0, max_new_tokens=64):
        self._加载()
        种子值 = 种子 + 轮次
        torch.manual_seed(种子值)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子值)

        if 模式 == "P6_情感导演" and not any(x["role"] == "system" for x in 消息):
            用户内容 = " ".join(x["content"] for x in 消息 if x["role"] == "user")
            _跳过 = ("JSON", "decision_choice", "final_decision", "Output strictly",
                    "rubric", "```", "dialogue_history", "对话历史", "真诚伙伴",
                    "下文回应", "考题", "请将以上", "设定", "场景")
            if not any(_w in 用户内容 for _w in _跳过):
                消息 = [{"role": "system", "content": P6人格注入提示}] + 消息

        提示 = self._分词器.apply_chat_template(
            消息, tokenize=False, add_generation_prompt=True)
        if 模式 != "裸":
            # Qwen3 思考块污染：注入模式从提示层面禁用 <think>（裸保持默认作基线）
            try:
                提示 = self._分词器.apply_chat_template(
                    消息, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except Exception:
                pass
        inputs = self._分词器(提示, return_tensors="pt").to(self._设备)

        if 模式 == "裸":
            out = self._模型.generate(
                inputs.input_ids, max_new_tokens=max_new_tokens,
                pad_token_id=self._分词器.eos_token_id,
                temperature=1.0, top_p=0.9, top_k=50,
                repetition_penalty=1.05, do_sample=True)
            新 = out[0, inputs.input_ids.shape[1]:]
            return self._分词器.decode(新, skip_special_tokens=True).strip()

        if 模式 == "P1_语义回响":
            out = self._回响.生成(inputs.input_ids, max_new_tokens=max_new_tokens,
                                eos_token_id=self._分词器.eos_token_id,
                                tokenizer=self._分词器,
                                temperature=1.0, top_p=0.9, top_k=50,
                                repetition_penalty=1.05)
            新 = out[0, inputs.input_ids.shape[1]:]
            return self._分词器.decode(新, skip_special_tokens=True).strip()

        解码 = self._解码器.get(模式)
        if 解码 is None:
            raise ValueError(f"未知模式：{模式}")
        if 模式 == "P2.5_潮汐":
            out = 解码.生成(inputs.input_ids, max_new_tokens=max_new_tokens,
                           eos_token_id=self._分词器.eos_token_id,
                           tokenizer=self._分词器, 用户文本=消息[-1]["content"],
                           temperature=1.0, top_p=0.9, top_k=50,
                           repetition_penalty=1.05)
        else:
            out, _ = 解码.生成(inputs.input_ids, max_new_tokens=max_new_tokens,
                             eos_token_id=self._分词器.eos_token_id,
                             tokenizer=self._分词器, 用户文本=消息[-1]["content"],
                             temperature=1.0, top_p=0.9, top_k=50,
                             repetition_penalty=1.05)
        新 = out[0, inputs.input_ids.shape[1]:]
        return self._分词器.decode(新, skip_special_tokens=True).strip()

    def 清理(self):
        for d in list(self._解码器.values()) + ([self._回响] if self._回响 else []):
            try:
                d._移除钩子()
            except Exception:
                pass
        for d in list(self._解码器.values()):
            try:
                d.model = None
            except Exception:
                pass
        self._解码器 = {}
        self._回响 = None
        self._模型 = None
        self._分词器 = None
        self._锚库 = None
        gc.collect()
        torch.cuda.empty_cache()


def 跑裁判(裁判类型, 请求列表):
    任务路径 = os.path.join(统一目录, "_泛化全_任务.json")
    输出路径 = os.path.join(统一目录, "_泛化全_输出.json")
    with open(任务路径, "w", encoding="utf-8") as f:
        json.dump({"裁判": 裁判类型, "请求": 请求列表}, f, ensure_ascii=False)
    subprocess.run([sys.executable, os.path.join(本目录, "裁判子进程.py"),
                    任务路径, 输出路径], check=True)
    with open(输出路径, encoding="utf-8") as f:
        return json.load(f)


def 阶段生成(模型名, 模式们, 样本数, 缓存路径):
    回复 = {}
    if os.path.exists(缓存路径):
        with open(缓存路径, encoding="utf-8") as f:
            回复 = json.load(f)
    生成器 = None
    try:
        生成器 = 泛化生成器(模型名)
        for 模式 in 模式们:
            if 模式 in 回复 and len(回复[模式]) >= 样本数:
                print(f"  [{模式}] 已生成，跳过", flush=True)
                continue
            列表 = []
            with open(样本路径, encoding="utf-8") as f:
                样本 = json.load(f)["样本"][:样本数]
            for i, r in enumerate(样本):
                try:
                    消息 = [{"role": "user", "content": r["user"]}]
                    文本 = 生成器.生成(模式, 消息, 种子=2026, 轮次=i, max_new_tokens=64)
                except Exception as e:
                    文本 = f"[生成失败:{e}]"
                    print(f"  [{模式} {i+1}/{样本数}] 失败: {e}", flush=True)
                if 文本.startswith("<think>"):
                    文本 = "（思考）" + 文本
                列表.append(文本)
                print(f"  [{模式} {i+1}/{样本数}] {r['user'][:14]} => {文本[:34]}", flush=True)
            with open(缓存路径, "w", encoding="utf-8") as f:
                json.dump({**回复, 模式: 列表}, f, ensure_ascii=False, indent=2)
            回复[模式] = 列表
            print(f"  [{模式}] 完成 {len(列表)} 条", flush=True)
    finally:
        if 生成器 is not None:
            生成器.清理()
    print(f"[阶段1完成] 缓存 -> {缓存路径}", flush=True)


def 阶段裁判(模型名, 模式们, 样本数, 缓存路径):
    回复 = json.load(open(缓存路径, encoding="utf-8"))
    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"][:样本数]

    # 读取已有结果（增量合并）
    已有 = {}
    if os.path.exists(输出路径):
        try:
            已有 = json.load(open(输出路径, encoding="utf-8")).get("模式汇总", {})
        except Exception:
            pass
    模型结果 = 已有.get(模型名, {})

    for 模式 in 模式们:
        if 模式 not in 回复 or len(回复[模式]) < 样本数:
            print(f"  [{模式}] 缓存不足，跳过", flush=True)
            continue
        if 模式 in 模型结果:
            print(f"  [{模式}] 已有结果，跳过", flush=True)
            continue
        print(f"  [{模式}] 裁判中 ...", flush=True)
        配对请求 = []
        for i in range(样本数):
            for AI在前 in (True, False):
                配对请求.append({"user": 样本[i]["user"], "AI": 回复[模式][i],
                                  "真人": 样本[i]["girl"], "AI在前": AI在前})
        配对结果 = 跑裁判("llm_judge_配对", 配对请求)
        ai胜 = sum(1 for i in range(样本数)
                   if 配对结果[i * 2].get("AI胜") or 配对结果[i * 2 + 1].get("AI胜"))
        评分请求 = [{"user": 样本[i]["user"], "回复": 回复[模式][i]} for i in range(样本数)]
        评分结果 = 跑裁判("llm_judge_评分", 评分请求)
        ai评分 = [c.get("评分") for c in 评分结果 if c.get("评分")]
        win = round(ai胜 / 样本数, 4)
        rating = round(sum(ai评分) / len(ai评分) / 5, 4) if ai评分 else 0.0
        模型结果[模式] = {
            "win_rate_against_human": win,
            "average_rating": rating,
            "average_rating_raw": round(sum(ai评分) / len(ai评分), 2) if ai评分 else 0.0,
        }
        print(f"  [{模型名} {模式}] win={win} rating={rating} (raw={模型结果[模式]['average_rating_raw']})", flush=True)

    已有[模型名] = 模型结果
    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump({"模式汇总": 已有, "样本数": 样本数,
                   "说明": "各模型 × 各模式 的 LLM-Judge（win_rate 与 rating，真人 girl 基线）"},
                  f, ensure_ascii=False, indent=2)
    print(f"[阶段2完成] 结果 -> {输出路径}", flush=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    模型名 = sys.argv[1]
    # 解析模式与阶段
    if "--仅生成" in sys.argv:
        idx = sys.argv.index("--仅生成")
        缓存路径 = sys.argv[idx + 1] if len(sys.argv) > idx + 1 and not sys.argv[idx + 1].startswith("--") else os.path.join(统一目录, f"泛化回复_{模型名}_全模式.json")
        模式们 = [a for a in sys.argv[idx + 2:] if a in 全模式列表]
        模式们 = 模式们 or 全模式列表
        阶段生成(模型名, 模式们, 12, 缓存路径)
    elif "--仅裁判" in sys.argv:
        idx = sys.argv.index("--仅裁判")
        缓存路径 = sys.argv[idx + 1] if len(sys.argv) > idx + 1 and not sys.argv[idx + 1].startswith("--") else os.path.join(统一目录, f"泛化回复_{模型名}_全模式.json")
        模式们 = [a for a in sys.argv[idx + 2:] if a in 全模式列表]
        模式们 = 模式们 or 全模式列表
        阶段裁判(模型名, 模式们, 12, 缓存路径)
    else:
        缓存路径 = os.path.join(统一目录, f"泛化回复_{模型名}_全模式.json")
        模式们 = [a for a in sys.argv[2:] if a in 全模式列表] or 全模式列表
        print(f"===== 泛化全模式 | {模型名} | 模式 {模式们} =====", flush=True)
        阶段生成(模型名, 模式们, 12, 缓存路径)
        print("\n[进入裁判阶段]", flush=True)
        阶段裁判(模型名, 模式们, 12, 缓存路径)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
