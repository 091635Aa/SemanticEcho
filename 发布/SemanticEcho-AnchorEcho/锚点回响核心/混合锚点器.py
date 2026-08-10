# -*- coding: utf-8 -*-
"""P4 锚点回响（Anchor Echo）—— 混合锚点器（新增模块）

按 P4_混合方案设计.md 第 3 节：三通道正交叠加（均作用于 logits 加法，可独立开关）：

  A 锚点回响  β·tanh(S@v_target/T_anchor)     嵌入空间低秩稠密打分（核心）
  B 回响      λ·(池质心@投影矩阵)              表示空间情感底色（复用 P1 回响注入器）
  C 潮汐      α·引导倍率·极性词表偏置          概率空间方向密度（复用 P3 极性 token 表逻辑）

目标层完全复用：本器持有 目标决策器（内部复用 潮汐决策器），三通道共享同一个
决策输出（v_target / α / 目标V），保证叠加方向一致。

生成循环与 锚点解码器 相同但叠加三通道偏置；在线退化兜底作用于全部通道
（β×0.5、λ×0.5、α×0.5，连续 3 次 → 全部归零 = 纯裸采样）。
"""
import math
import os
import re
import sys
from collections import Counter
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F

# 路径注入：锚点回响工作目录 + 语义回响工程根（回响注入器/回响池）
工作目录 = os.path.dirname(os.path.abspath(__file__))
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)
回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 锚点解码器 import 锚点解码器, 计算熵, 计算重复率

# 通道 B 依赖（P1 回响注入器）；import 失败时自动关闭 B 通道（不影响 A/C）
_回响可用 = True
_回响错误 = ""
try:
    from semantic_echo.回响池 import 语义回响池
    from semantic_echo.采样处理器 import 回响注入器
except Exception as _e:  # noqa: BLE001 —— 降级：关闭回响通道
    _回响可用 = False
    _回响错误 = str(_e)
    语义回响池 = None
    回响注入器 = None


class _GPU回响注入器(回响注入器):
    """GPU 直分配投影矩阵（父类 CPU 分配对 151936 大词表占 933MB 内存，改 GPU 分配）"""

    def _初始化投影(self, seed: int) -> None:
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        scale = math.sqrt(2.0 / self.hidden_dim)
        self.投影矩阵 = torch.randn(
            self.hidden_dim, self.vocab_size,
            generator=rng, dtype=torch.float32, device=self.device,
        ) * scale
        self.投影矩阵.requires_grad_(False)


class 混合锚点器:
    """锚点 × 回响 × 潮汐 三通道正交叠加解码器"""

    def __init__(
        self,
        model,
        tokenizer,
        锚点库: 锚点库,
        目标决策器: 目标决策器,
        锚点β: float = 0.8,
        锚点T: float = 0.3,
        回响λ: float = 0.08,
        潮汐倍率: float = 12.0,
        开启A: bool = True,
        开启B: bool = True,
        开启C: bool = True,
        温度: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        退化窗口: int = 40,
        兜底阈值: float = 0.6,
        句分隔符: str = r"[。！？!?；;\n～…~]",
        密度基: float = 0.06,
        密度增益: float = 0.10,
        密度上限: float = 0.25,
        最短字数: int = 12,
        最大字数: int = 90,
        最长句数: int = 2,
        句子停止: bool = True,
        最小长度: int = 0,
    ):
        """三通道参数：
        锚点β / 锚点T：通道 A（锚点回响）强度与温度（0 = 关闭锚点）；
        回响λ：通道 B（回响）强度（0 = 关闭回响）；
        潮汐倍率：通道 C（潮汐）引导倍率 k = α × 倍率（0 = 关闭潮汐）；
        开启A/开启B/开启C：三通道独立开关（默认全开）。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.锚点库 = 锚点库
        self.目标决策器 = 目标决策器
        self.锚点β = 锚点β
        self.锚点T = 锚点T
        self.回响λ = 回响λ
        self.潮汐倍率 = 潮汐倍率
        self.开启A = bool(开启A)
        self.开启B = bool(开启B and _回响可用)
        self.开启C = bool(开启C)
        self.温度 = 温度
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.退化窗口 = 退化窗口
        self.兜底阈值 = 兜底阈值
        self.句分隔符 = re.compile(句分隔符)
        self.密度基 = 密度基
        self.密度增益 = 密度增益
        self.密度上限 = 密度上限
        self.最短字数 = 最短字数
        self.最大字数 = 最大字数
        self.最长句数 = 最长句数
        self.句子停止 = 句子停止
        self.最小长度 = 最小长度

        self.device = model.device
        self.vocab_size = int(model.config.vocab_size)
        self._回响错误 = _回响错误

        # ── 通道 A：锚点解码器（复用其 注入偏置/打分表/目标状态）──
        self.锚点 = 锚点解码器(
            model, tokenizer, 锚点库, 目标决策器,
            β=锚点β, T_anchor=锚点T, 接口="本地",
            退化窗口=退化窗口, 兜底阈值=兜底阈值,
            句分隔符=句分隔符, 密度基=密度基, 密度增益=密度增益, 密度上限=密度上限,
            句子停止=False, 最短字数=最短字数, 最大字数=最大字数,
            最长句数=最长句数, 最小长度=最小长度)

        # ── 通道 B：回响注入器（P1 钩子/池/投影）──
        self.回响 = None
        if self.开启B:
            try:
                池 = 语义回响池(int(model.config.hidden_size))
                self.回响 = _GPU回响注入器(model, 池, lambda_strength=回响λ)
            except Exception as e:  # noqa: BLE001
                self.开启B = False
                self._回响错误 = str(e)

        # ── 通道 C：极性 token 表（复用 P3 极性引导逻辑）──
        self._极性token表: Dict[int, float] = self._构建极性token表()

        # ── 目标状态（共享决策器输出）──
        self.当前目标 = None
        self.当前α = 0.0
        self.当前目标V = 0.0
        self.密度目标 = self.密度基
        self.手动锚点β = 锚点β is not None

        # ── 句状态 / 兜底 ──
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表: List[int] = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0

    # ──────────────────────────────────────────────
    # 通道 C 极性 token 表
    # ──────────────────────────────────────────────

    def _构建极性token表(self) -> Dict[int, float]:
        """极性情感 token 表：token_id → 极性（+1 正面 / -1 负面）。

        来源：锚点库词集（温柔/开心/平静 → 正；难过/愤怒/害怕 → 负）
        + cnsenti 词库（感知器._正面词/_负面词，若可用）。
        """
        表: Dict[int, float] = {}
        正维 = {"温柔", "开心", "平静"}
        for 维, 词列表 in self.锚点库.词集.items():
            极性 = 1.0 if 维 in 正维 else -1.0
            for 词 in 词列表:
                ids = self.tokenizer.encode(词, add_special_tokens=False)
                if len(ids) == 1:
                    表.setdefault(ids[0], 极性)
        try:
            感知器 = self.目标决策器.感知器
            for 词 in getattr(感知器, "_正面词", set()):
                ids = self.tokenizer.encode(词, add_special_tokens=False)
                if len(ids) == 1:
                    表.setdefault(ids[0], 1.0)
            for 词 in getattr(感知器, "_负面词", set()):
                ids = self.tokenizer.encode(词, add_special_tokens=False)
                if len(ids) == 1:
                    表.setdefault(ids[0], -1.0)
        except Exception:  # noqa: BLE001 —— 词库缺失不影响主流程
            pass
        return 表

    # ──────────────────────────────────────────────
    # 目标更新（三通道共享同一 目标决策器 输出）
    # ──────────────────────────────────────────────

    def 更新目标(self, 用户文本: str = "", 思考链文本: str = "", 指令: str = "") -> object:
        目标 = self.目标决策器.计算目标(
            用户当前=用户文本 or None, 思考链文本=思考链文本, 指令=指令)
        self.当前目标 = 目标
        # 通道 A：v_target / β
        self.锚点.v_target = np.asarray(目标.v_target, dtype=np.float32)
        if not self.手动锚点β:
            self.锚点.β = 目标.β
        else:
            self.锚点.β = self.锚点β
        # 通道 C：α / 目标V（与锚点通道同源）
        self.当前α = float(目标.决策日志.get("潮汐α", 0.15) or 0.15)
        self.当前目标V = float(目标.决策日志.get("目标V", 0.0) or 0.0)
        self.密度目标 = 目标.情感词密度目标
        return 目标

    # ──────────────────────────────────────────────
    # 三通道注入
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 注入偏置(self, logits: torch.Tensor) -> torch.Tensor:
        """三通道正交叠加：A 锚点 + B 回响 + C 潮汐"""
        if self.开启A:
            logits = self.锚点.注入偏置(logits)
        if self.开启B and self.回响 is not None:
            logits = self.回响.注入偏置(logits)
        if self.开启C:
            logits = self._潮汐引导(logits)
        return logits

    @torch.no_grad()
    def _潮汐引导(self, logits: torch.Tensor) -> torch.Tensor:
        """通道 C：极性定向引导（复用 P3 混合注入器._情感引导 逻辑）"""
        if not self._极性token表 or self.当前α <= 0:
            return logits
        if abs(self.当前目标V) < 0.03:
            return logits
        强度系数 = self.当前α * self.潮汐倍率
        密度系数 = 1.0
        if self._句情感词数 > 0:
            L = max(len(self._当前句文本), 1)
            if self._句情感词数 / L > self.密度目标:
                密度系数 = 0.3
        目标方向 = 1.0 if self.当前目标V > 0 else -1.0
        for tid, 极性 in self._极性token表.items():
            if tid >= self.vocab_size:
                continue
            if 极性 * 目标方向 > 0:
                logits[0, tid] += 强度系数 * 密度系数
            else:
                logits[0, tid] -= 强度系数 * 0.3 * 密度系数
        return logits

    # ──────────────────────────────────────────────
    # 生成循环
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 生成(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        logits_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        tokenizer=None,
        轮次回调: Optional[Callable[[int, object], None]] = None,
        用户文本: str = "",
        思考链文本: str = "",
        指令: str = "",
    ):
        """生成循环 = 前向 → 三通道注入 → 采样 → 句状态/密度/兜底。

        返回 (token_ids, 统计字典{平均熵, 重复率, 情感命中率, β, T_anchor,
        v_target, 触发兜底次数})（与 锚点解码器 同构）。
        """
        temperature = self.温度 if temperature is None else temperature
        top_p = self.top_p if top_p is None else top_p
        top_k = self.top_k if top_k is None else top_k
        repetition_penalty = self.repetition_penalty if repetition_penalty is None else repetition_penalty
        tokenizer = tokenizer or self.tokenizer
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id

        self.更新目标(用户文本, 思考链文本, 指令)

        # 状态重置（含三通道强度基准）
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表 = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0
        if self.开启B and self.回响 is not None:
            self.回响.pool.清空()
            self.回响.lambda_strength = self.回响λ

        past_key_values = None
        已生成 = input_ids.clone()
        已生成token集合: Set[int] = set()
        熵列表: List[float] = []

        for 步 in range(max_new_tokens):
            模型输入 = 已生成[:, -1:] if past_key_values is not None else 已生成
            outputs = self.model(模型输入, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            if repetition_penalty != 1.0:
                for tid in 已生成token集合:
                    logits[0, tid] /= repetition_penalty

            # ── 三通道注入 ──
            logits = self.注入偏置(logits)

            if logits_callback is not None:
                logits_callback(步, logits)

            熵列表.append(计算熵(logits))

            # 最小长度：压制 EOS
            if self.最小长度 > 0 and self._生成文本 and len(self._生成文本) < self.最小长度:
                if eos_token_id is not None and eos_token_id < self.vocab_size:
                    logits[0, eos_token_id] = float('-inf')

            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < top_k_values[:, -1].unsqueeze(-1)] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            下一个token = torch.multinomial(probs, num_samples=1)

            已生成 = torch.cat([已生成, 下一个token], dim=-1)
            已生成token集合.add(下一个token.item())
            self._已生成token列表.append(下一个token.item())

            # 句状态 / 兜底（作用于三通道强度）
            self._更新句状态(下一个token.item())
            if len(self._已生成token列表) >= 8 and len(self._已生成token列表) % self.退化窗口 == 0:
                self._兜底监测()

            # 通道 B：捕获回响（情感 token 入池）
            if self.开启B and self.回响 is not None:
                self.回响.捕获回响(logits, tokenizer=tokenizer)
                self.回响.pool.推进()

            if 轮次回调 is not None:
                轮次回调(步, self)
            if 下一个token.item() == eos_token_id:
                break
            if self.句子停止 and self._句子停止():
                break

        统计 = {
            "平均熵": round(sum(熵列表) / len(熵列表), 4) if 熵列表 else 0.0,
            "重复率": 计算重复率(self._已生成token列表),
            "情感命中率": self.锚点._情感命中率(self._已生成token列表),
            "β": round(self.锚点.β, 4),
            "T_anchor": self.锚点T,
            "v_target": None if self.锚点.v_target is None
            else [round(float(x), 4) for x in self.锚点.v_target],
            "触发兜底次数": self.触发兜底次数,
        }
        return 已生成, 统计

    # ──────────────────────────────────────────────
    # 句状态 / 兜底
    # ──────────────────────────────────────────────

    def _更新句状态(self, token_id: int) -> None:
        文本 = self.tokenizer.decode([token_id], skip_special_tokens=True)
        if not 文本:
            return
        if token_id in self._极性token表:
            极性 = self._极性token表[token_id]
            if (极性 > 0 and self.当前目标V > 0) or (极性 < 0 and self.当前目标V < 0):
                self._句情感词数 += 1
        self._生成文本 += 文本
        if self.句分隔符.search(文本):
            self._当前句文本 = ""
            self._句情感词数 = 0
            self._句子数 += 1
            self._兜底监测()
        else:
            self._当前句文本 += 文本

    def _兜底监测(self) -> None:
        """在线退化兜底：重复率 > 阈值 或 熵 < 0.6 → 三通道强度各 ×0.5；
        连续 3 次 → 全部归零（纯裸采样）。"""
        最近 = self._已生成token列表[-self.退化窗口:]
        if len(最近) < 8:
            return
        重复率 = 1.0 - len(set(tuple(最近[i:i + 2]) for i in range(len(最近) - 1))) / max(len(最近) - 1, 1)
        熵 = self._经验熵(最近)
        if 重复率 > self.兜底阈值 or 熵 < 0.6:
            self.触发兜底次数 += 1
            self._兜底计数 += 1
            if self._兜底计数 >= 3:
                self.锚点.β = 0.0
                self.当前α = 0.0
                if self.回响 is not None:
                    self.回响.lambda_strength = 0.0
            else:
                self.锚点.β *= 0.5
                self.当前α *= 0.5
                if self.回响 is not None:
                    self.回响.lambda_strength *= 0.5
        else:
            self._兜底计数 = 0

    @staticmethod
    def _经验熵(token列表: List[int]) -> float:
        if not token列表:
            return 0.0
        n = len(token列表)
        c = Counter(token列表)
        return float(-sum((v / n) * math.log(v / n) for v in c.values()))

    def _句子停止(self) -> bool:
        if not self._生成文本:
            return False
        if len(self._生成文本) >= self.最短字数 and self._句子数 >= self.最长句数:
            return True
        if self._句子数 >= self.最长句数 + 1:
            return True
        if len(self._生成文本) >= self.最大字数:
            return True
        return False

    def 重置(self) -> None:
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表 = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0
        self.当前目标 = None
        self.当前α = 0.0

    def __repr__(self):
        return (f"混合锚点器(A={self.开启A}(β={self.锚点β}), "
                f"B={self.开启B}(λ={self.回响λ}), C={self.开启C}(倍率={self.潮汐倍率}), "
                f"极性token={len(self._极性token表)})")
