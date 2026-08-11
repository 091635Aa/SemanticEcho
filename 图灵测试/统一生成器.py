# -*- coding: utf-8 -*-
"""统一生成器（7 模式）—— 接口兼容 生成器.py，供 5 大基准脚本复用

模式：裸 / P1_语义回响 / P1.5_兼容层 / P2.5_潮汐 / P3_锚点回响 / P4_KV共振 / P5_超融合
统一协议：统一种子、temperature=1.0、top_p=0.9、top_k=50、repetition_penalty=1.05。

用法与 生成器.py 一致：
    from 统一生成器 import 生成器实例
    文本 = 生成器实例.生成(模式, 消息, 种子=42, 轮次=0, max_new_tokens=128, 模板="chat")
"""
import os
import sys
import math
import torch

# ── 路径注入 ──
本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)
KV核心 = r"c:\Users\Administrator\Documents\KV 情感共振解码\核心"
if KV核心 not in sys.path:
    sys.path.insert(0, KV核心)
回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)
ETD目录 = r"h:\情感潮汐解码（Emotion Tidal Decoding, ETD）"
if ETD目录 not in sys.path:
    sys.path.insert(0, ETD目录)

from transformers import AutoModelForCausalLM, AutoTokenizer

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 锚点解码器 import 锚点解码器, 计算熵
from 超融合解码器 import 超融合解码器
from 情感共振解码器 import 情感共振解码器
from semantic_echo.回响池 import 语义回响池
from semantic_echo.采样处理器 import 回响注入器
from semantic_echo.情感过滤器 import 情感过滤器

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
目标模型名 = "Qwen2.5-1.5B-Instruct"
模型路径 = os.path.join(模型空间, 目标模型名)

模式列表 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合"]
采样参数 = dict(temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05)


class _GPU回响注入器(回响注入器):
    """GPU 直分配投影矩阵（父类 CPU 分配 933MB 且每步 to(device) 拖慢）"""

    def _初始化投影(self, seed: int) -> None:
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        scale = math.sqrt(2.0 / self.hidden_dim)
        self.投影矩阵 = torch.randn(
            self.hidden_dim, self.vocab_size,
            generator=rng, dtype=torch.float32, device=self.device,
        ) * scale
        self.投影矩阵.requires_grad_(False)


class 统一生成器:
    """7 模式生成器：懒加载 1.5B 基座 + 全部方案解码器"""

    def __init__(self):
        self._模型 = None
        self._分词器 = None
        self._解码器 = {}
        self._回响 = None
        self._锚库 = None
        self._设备 = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 加载 ──
    def _加载(self):
        if self._模型 is not None:
            return
        print(f"[统一生成器] 加载 {目标模型名} fp16 ...")
        self._分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
        self._模型 = AutoModelForCausalLM.from_pretrained(
            模型路径, torch_dtype=torch.float16, trust_remote_code=True).to(self._设备)
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
            print(f"[统一生成器] 潮汐感知器不可用：{e}")
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

        # 各方案解码器
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
        try:
            from 潮汐解码器 import 潮汐解码器
            if 感知器 is not None:
                self._解码器["P2.5_潮汐"] = 潮汐解码器(self._模型, self._分词器, 感知器, 潮汐决策)
        except Exception as e:
            print(f"[统一生成器] P2.5 潮汐不可用：{e}")

        # 释放锚点库大矩阵
        self._锚库.W_e = None
        self._锚库._有效权重 = None
        self._锚库.权重 = None
        torch.cuda.empty_cache()
        print(f"[统一生成器] 就绪，方案：{list(self._解码器.keys())}")

    # ── 生成入口（兼容 生成器.py 接口）──
    def 生成(self, 模式, 消息, 种子=42, 轮次=0, max_new_tokens=128,
             λ覆盖=None, 思考链=False, 模板="chat", 身份注入=True, 提示词="人类身份",
             **kwargs):
        """统一生成：模式 ∈ 7 模式列表；模板 ∈ chat|纯文本"""
        self._加载()
        种子值 = 种子 + 轮次
        torch.manual_seed(种子值)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子值)

        # 纯文本模板（TuringBench 用）：身份前缀 + 文本拼接
        if 模板 == "纯文本":
            提示 = "\n\n".join(x["content"] for x in 消息)
        else:
            提示 = self._分词器.apply_chat_template(
                消息, tokenize=False, add_generation_prompt=True)
        inputs = self._分词器(提示, return_tensors="pt").to(self._设备)

        if 模式 == "裸":
            out = self._模型.generate(
                inputs.input_ids, max_new_tokens=max_new_tokens,
                pad_token_id=self._分词器.eos_token_id, **采样参数, do_sample=True)
            新 = out[0, inputs.input_ids.shape[1]:]
            return self._分词器.decode(新, skip_special_tokens=True).strip()

        if 模式 == "P1_语义回响":
            out = self._回响.生成(inputs.input_ids, max_new_tokens=max_new_tokens,
                                eos_token_id=self._分词器.eos_token_id,
                                tokenizer=self._分词器, **采样参数)
            新 = out[0, inputs.input_ids.shape[1]:]
            return self._分词器.decode(新, skip_special_tokens=True).strip()

        解码 = self._解码器.get(模式)
        if 解码 is None:
            raise ValueError(f"未知模式：{模式}")
        if 模式 == "P2.5_潮汐":
            out = 解码.生成(inputs.input_ids, max_new_tokens=max_new_tokens,
                           eos_token_id=self._分词器.eos_token_id,
                           tokenizer=self._分词器, 用户文本=消息[-1]["content"], **采样参数)
        else:
            out, _ = 解码.生成(inputs.input_ids, max_new_tokens=max_new_tokens,
                             eos_token_id=self._分词器.eos_token_id, 用户文本=消息[-1]["content"],
                             **采样参数)
        新 = out[0, inputs.input_ids.shape[1]:]
        return self._分词器.decode(新, skip_special_tokens=True).strip()

    def 裸生成(self, 消息, 种子=42, 轮次=0, max_new_tokens=128, **kwargs):
        return self.生成("裸", 消息, 种子=种子, 轮次=轮次, max_new_tokens=max_new_tokens)

    def 清理(self):
        import gc
        for d in self._解码器.values():
            try:
                d._移除钩子()
            except Exception:
                pass
        try:
            self._回响._移除钩子()
        except Exception:
            pass
        # 断开所有 model 引用（含解码器内部 self.model、锚点库 self.model）
        for d in self._解码器.values():
            try:
                d.model = None
            except Exception:
                pass
            try:
                d.锚点库 = None
            except Exception:
                pass
            try:
                d.目标决策器 = None
            except Exception:
                pass
        try:
            self._回响.model = None
        except Exception:
            pass
        if self._锚库 is not None:
            try:
                self._锚库.model = None
                self._锚库.tokenizer = None
            except Exception:
                pass
        self._解码器 = {}
        self._回响 = None
        self._锚库 = None
        self._模型 = None
        self._分词器 = None
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


# 全局单例（与 生成器.py 的 生成器实例 同名，方便基准脚本替换 import）
生成器实例 = 统一生成器()
