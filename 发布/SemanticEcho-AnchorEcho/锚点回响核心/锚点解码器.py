# -*- coding: utf-8 -*-
"""P4 锚点回响（Anchor Echo）—— 锚点解码器（新增模块，核心表达层）

按 P4_混合方案设计.md 第 2.4 节，独立实现（不用 forward hook），接口签名对齐
回响注入器.生成()/注入偏置()（参数名、返回结构尽量一致），评测脚本可无痛接入。

核心公式：
  预计算打分表  S[w, k] = cos(W_e[w], e_k)          # S ∈ R^{V×K}，fp16
  极性内积      a(w)     = S[w] · v_target
  主注入公式    logits[w] += β · tanh(a(w) / T_anchor)   # tanh 有界防坍缩
  线性简化式    logits[w] += β · a(w)                    # T_anchor→∞ 等价

在线退化兜底：每 退化窗口(40) 个 token 检查语义熵（经验熵 < 0.6）与 2-gram
重复率（> 兜底阈值 0.6）→ β×0.5；连续 3 次 → β=0（锚点通道静默，纯裸采样）。
密度控制：句内目标方向情感词密度 > 密度目标 → 注入幅度 ×0.3（防情感词堆砌）。

三级接口降级（配合 接口降级.py）：
  '本地'     全词表稠密打分（默认，S 查表）；
  'logprobs' 仅对 top-k（默认 100）候选做稠密打分并加偏置（模拟 API top-k logprobs）；
  '提示'     无任何模型内部访问，锚点词注入 prompt（生成时自动包裹用户文本）。

生成() 返回 (token_ids, 统计字典{平均熵, 重复率, 情感命中率, β, T_anchor,
v_target, 触发兜底次数})，与 P3 评测脚本期望的输出结构尽量一致。
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

# 锚点回响工作目录（锚点库 / 目标决策器 / 接口降级 所在）
工作目录 = os.path.dirname(os.path.abspath(__file__))
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, 自动适配
from 接口降级 import 判定接口, 构造提示词


# ══════════════════════════════════════════════════
# 指标辅助（与 P3/回响评估器 同口径）
# ══════════════════════════════════════════════════

def 计算熵(logits: torch.Tensor) -> float:
    """单步 logits 的 softmax 语义熵（自然对数底）。

    注：fp16 下 clamp_min(1e-12) 会下溢为 0 → 0·log0=nan，故先转 fp32。"""
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    return float(-(probs * log_probs).sum().item())


def 经验熵(token列表: List[int]) -> float:
    """已生成文本的 token 级经验分布熵（H = -Σ p·ln p）；空/单 token 返回 0"""
    if not token列表:
        return 0.0
    n = len(token列表)
    c = Counter(token列表)
    return float(-sum((v / n) * math.log(v / n) for v in c.values()))


def 计算重复率(token列表: List[int], 阶数: int = 2) -> float:
    """n-gram 重复率 = 1 - 唯一 n-gram 数 / 总 n-gram 数"""
    if len(token列表) < 阶数 + 1:
        return 0.0
    ngrams = [tuple(token列表[i:i + 阶数]) for i in range(len(token列表) - 阶数 + 1)]
    return round(1.0 - len(set(ngrams)) / max(len(ngrams), 1), 4)


# ══════════════════════════════════════════════════
# 锚点解码器
# ══════════════════════════════════════════════════

class 锚点解码器:
    """锚点回响解码器：稠密打分注入 + 自回归生成循环 + 在线退化兜底。

    接口签名对齐 回响注入器：注入偏置(logits) → logits；生成(input_ids, ...) → (ids, 统计)。
    """

    def __init__(
        self,
        model,
        tokenizer,
        锚点库: 锚点库,
        目标决策器: 目标决策器,
        β: Optional[float] = 0.8,
        T_anchor: float = 0.3,
        稀疏阈值: float = 0.0,
        延迟注入步数: int = 0,
        打分表缓存路径: Optional[str] = None,
        温度: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        退化窗口: int = 40,
        兜底阈值: float = 0.6,
        接口: str = "本地",
        topk候选: int = 100,
        量化类型: str = "fp16",
        角色扰动幅度: float = 0.0,
        句分隔符: str = r"[。！？!?；;\n～…~]",
        密度基: float = 0.06,
        密度增益: float = 0.10,
        密度上限: float = 0.25,
        目标长度: int = 34,
        最短字数: int = 12,
        最大字数: int = 90,
        最长句数: int = 2,
        句子停止: bool = True,
        最小长度: int = 0,
    ):
        """参数说明：
        β：锚点注入强度（手动指定，如 0.8；None → 自动适配：自动适配(model, 量化类型)）；
        T_anchor：tanh 内积温度（放大弱信号、饱和强信号）；
        稀疏阈值：稀疏注入阈值（默认 0.0 保持向后兼容；>0 时只对
        tanh(S[w]·v_target/T_anchor) > 稀疏阈值 的 token 加偏置，其余 token 偏置=0，
        非情感 token 不动、情感 token 抬升 → 熵保持修复，Task5）；
        延迟注入步数：>0 时生成前 N 步不注入（先让分布定型，过注入兜底）；
        打分表缓存路径：预计算打分表缓存文件（默认 锚点表.pt；换词集/改 K 时
        传独立路径，避免覆盖默认缓存）；
        温度/top_p/top_k/repetition_penalty：采样默认值（生成() 可逐次覆盖）；
        退化窗口/兜底阈值：在线退化兜底参数（熵<0.6 或 2-gram 重复率>阈值 → β×0.5）；
        角色扰动幅度（SubTask 6b·角色差异随机性）：>0 时对 v_target 按 角色名 做
        确定性轻微扰动（crc32 种子 + 高斯噪声，同一角色固定方向、跨角色各异），
        缓解跨角色回复模板化（如"别担心/太好了"开头雷同）；默认 0.0 关闭（向后兼容）；
        接口：'本地' | 'logprobs' | '提示'（三级降级，None → 判定接口(model)）；
        topk候选：logprobs 模式候选受限打分的 top-k 数；
        句分隔符/密度*/目标长度/最短字数/最大字数/最长句数/句子停止/最小长度：
        句级密度控制与长度收尾（复用 P3 结构）。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.锚点库 = 锚点库
        self.目标决策器 = 目标决策器
        self.T_anchor = T_anchor
        self.稀疏阈值 = 稀疏阈值
        self.延迟注入步数 = 延迟注入步数
        self.温度 = 温度
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.退化窗口 = 退化窗口
        self.兜底阈值 = 兜底阈值
        self.角色扰动幅度 = 角色扰动幅度
        self.topk候选 = topk候选
        self.句分隔符 = re.compile(句分隔符)
        self.密度基 = 密度基
        self.密度增益 = 密度增益
        self.密度上限 = 密度上限
        self.目标长度 = 目标长度
        self.最短字数 = 最短字数
        self.最大字数 = 最大字数
        self.最长句数 = 最长句数
        self.句子停止 = 句子停止
        self.最小长度 = 最小长度

        self.device = model.device
        self.vocab_size = int(model.config.vocab_size)

        # ── β 来源：手动指定 or 自动适配 ──
        self.手动β = β is not None
        self.β基 = β if β is not None else 自动适配(model, 量化类型)["β"]
        self.β = self.β基

        # ── 接口级别（三级降级）──
        self.接口 = 接口 or 判定接口(model)

        # ── 预计算打分表 S ∈ R^{V×K}（fp16），生成期零 hook 开销 ──
        # 小修（Task9 接口降级验证）：提示模式（③ 纯黑盒）按设计应「零 embedding、
        # 零 logits 需求」，故跳过打分表构建（不读 embedding），延迟到本地/logprobs
        # 模式真正注入前再构建（见 _注入偏置稠密/_注入偏置logprobs 开头兜底）。
        self.打分表缓存路径 = 打分表缓存路径
        if self.接口 == "提示":
            self.打分表 = None
        else:
            self.打分表 = 锚点库.预计算打分表(缓存路径=打分表缓存路径)

        # ── 目标状态 ──
        self.v_target: Optional[np.ndarray] = None
        self.密度目标 = self.密度基

        # ── 句状态 / 兜底状态 ──
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表: List[int] = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0

        # ── 情感命中率统计（cnsenti 优先，锚点库词集兜底）──
        self._情感token集: Set[int] = self._构建情感token集()

    # ──────────────────────────────────────────────
    # 情感命中率词集
    # ──────────────────────────────────────────────

    def _构建情感token集(self) -> Set[int]:
        """单 token 情感词 id 集合：锚点库词集 + cnsenti 词库（若有）"""
        集: Set[int] = set()
        for 维, 词列表 in self.锚点库.词集.items():
            for 词 in 词列表:
                ids = self.tokenizer.encode(词, add_special_tokens=False)
                if len(ids) == 1:
                    集.add(ids[0])
        try:
            感知器 = self.目标决策器.感知器
            for 词 in getattr(感知器, "_正面词", set()) | getattr(感知器, "_负面词", set()):
                ids = self.tokenizer.encode(词, add_special_tokens=False)
                if len(ids) == 1:
                    集.add(ids[0])
        except Exception:  # noqa: BLE001 —— 词库缺失不影响主流程
            pass
        return 集

    # ──────────────────────────────────────────────
    # 目标更新
    # ──────────────────────────────────────────────

    def 更新目标(self, 用户文本: str = "", 思考链文本: str = "", 指令: str = "",
                角色=None, 轮次: int = 0) -> object:
        """复用 感知器.测量/追加轨迹 + 目标决策器.计算目标 → 设置 v_target/β/密度目标

        角色（SubTask 6b 新增·角色感知）：透传给 目标决策器.计算目标()，
        命中角色锚点基调表时 v_target = 0.7·v_角色 + 0.3·v_用户。"""
        try:
            目标 = self.目标决策器.计算目标(
                用户当前=用户文本 or None, 思考链文本=思考链文本, 指令=指令,
                角色=角色, 轮次=轮次)
            self.v_target = np.asarray(目标.v_target, dtype=np.float32)
            self.密度目标 = 目标.情感词密度目标
            if not self.手动β:  # 自动模式：采用决策器自适应 β
                self.β = 目标.β
            return 目标
        except Exception as e:  # noqa: BLE001 —— 决策失败时静默降级（无目标 → 不注入）
            print(f"[锚点解码器] 目标计算失败，本次不注入：{e}")
            self.v_target = None
            return None

    @staticmethod
    def _角色扰动(v_target, 角色):
        """按角色轻微扰动 v_target（SubTask 6b·角色差异随机性，确定性可复现）。

        crc32(角色名) 作种子生成 K 维高斯噪声，v_target ← normalize(v + 幅度·噪声)。
        同一角色每轮扰动方向固定（不破坏跨轮一致性）；不同角色方向各异
        （缓解"别担心/太好了"等跨角色模板化开头）。返回 fp32 单位向量。
        """
        import zlib
        种子 = zlib.crc32(str(角色).encode("utf-8"))
        rng = np.random.RandomState(种子)
        噪声 = rng.randn(len(v_target)).astype(np.float32)
        v = np.asarray(v_target, dtype=np.float32) + 0.06 * 噪声
        范数 = float(np.linalg.norm(v)) + 1e-9
        return (v / 范数).astype(np.float32)

    def 主导维度(self) -> str:
        """当前 v_target 的 argmax 维度名（提示模式选词用；无目标默认 温柔）"""
        名 = self.锚点库.维度名()
        if self.v_target is not None and len(self.v_target) == len(名):
            return 名[int(np.argmax(self.v_target))]
        return "温柔"

    def 构造提示词(self, 用户文本: str = "") -> str:
        """③ 锚点提示模式：把当前主导维度的锚点词注入 prompt"""
        return 构造提示词(self.主导维度(), self.锚点库, 用户文本)

    # ──────────────────────────────────────────────
    # 注入偏置（接口对齐 回响注入器.注入偏置）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 注入偏置(self, logits: torch.Tensor) -> torch.Tensor:
        """logits(1,V) += β·tanh(S@v_target / T_anchor)；无目标/β=0/提示接口 → 原样返回"""
        if self.v_target is None or self.β <= 0:
            return logits
        if self.接口 == "提示":
            return logits  # 提示模式无内部访问，不注入 logits
        if self.接口 == "logprobs":
            return self._注入偏置logprobs(logits)
        return self._注入偏置稠密(logits)

    @torch.no_grad()
    def _注入偏置稠密(self, logits: torch.Tensor) -> torch.Tensor:
        """全词表稠密注入（主路径）：b(w) = β·密度系数·tanh(S[w]·v_target / T_anchor)。

        稀疏注入（Task5 熵保持修复）：稀疏阈值>0 时只对
        tanh(S[w]·v_target/T_anchor) > 稀疏阈值 的 token 加偏置，其余 token 偏置=0
        （非情感 token 不动、情感 token 抬升 → 分布收窄大幅减小、命中率保持）；
        稀疏阈值=0.0 为向后兼容默认（负向 token 不再被负偏置压制，同样是熵保持
        的一部分）。"""
        密度系数 = self._密度系数()
        if self.打分表 is None:  # 延迟构建兜底（小修：见 __init__ 打分表注释）
            self.打分表 = self.锚点库.预计算打分表(缓存路径=self.打分表缓存路径)
        v = torch.as_tensor(self.v_target, dtype=self.打分表.dtype, device=self.打分表.device)
        a = self.打分表 @ v                       # (V,) = S @ v_target
        tanh值 = torch.tanh(a / self.T_anchor)
        if self.稀疏阈值 > 0:
            tanh值 = tanh值.where(tanh值 > self.稀疏阈值, torch.zeros_like(tanh值))
        偏置 = (self.β * 密度系数) * tanh值
        return logits + 偏置.unsqueeze(0)

    @torch.no_grad()
    def _注入偏置logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        """② logprobs 近似：只对 top-k 候选做稠密打分并加偏置（模拟 API 只能拿到
        top-k logprobs 的场景；k = topk候选，默认 100）。"""
        密度系数 = self._密度系数()
        if self.打分表 is None:  # 延迟构建兜底（小修：见 __init__ 打分表注释）
            self.打分表 = self.锚点库.预计算打分表(缓存路径=self.打分表缓存路径)
        k = min(self.topk候选, self.vocab_size)
        topk值, topk索引 = torch.topk(logits, k=k, dim=-1)      # (1,k)
        cand = self.打分表[topk索引[0]]                           # (k,K)
        v = torch.as_tensor(self.v_target, dtype=self.打分表.dtype, device=self.打分表.device)
        a = cand @ v                                              # (k,)
        偏置 = (self.β * 密度系数) * torch.tanh(a / self.T_anchor)
        out = logits.clone()
        out[0, topk索引[0]] = topk值[0] + 偏置                    # 候选重打分，其余保持
        return out

    def _密度系数(self) -> float:
        """句内目标方向情感词密度 > 密度目标 → 注入幅度 ×0.3（防情感词堆砌）"""
        if self._句情感词数 <= 0:
            return 1.0
        L = max(len(self._当前句文本), 1)
        if self._句情感词数 / L > self.密度目标:
            return 0.3
        return 1.0

    # ──────────────────────────────────────────────
    # 生成循环（接口对齐 回响注入器.生成，返回 (token_ids, 统计字典)）
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
        句子停止: Optional[bool] = None,
        角色=None,
        轮次: int = 0,
    ):
        """自回归生成：前向 → 注入偏置(锚点) → 采样 → 句状态/密度/兜底。

        用户文本/思考链文本/指令：情感目标来源（目标决策器）；
        角色/轮次（SubTask 6b 新增·角色感知）：透传给 目标决策器.计算目标()，
            命中角色锚点基调表时 v_target = 0.7·v_角色 + 0.3·v_用户；
        句子停止：None → 用构造时的 self.句子停止。

        返回 (token_ids, 统计字典{平均熵, 重复率, 情感命中率, β, T_anchor,
        v_target, 触发兜底次数})。
        """
        temperature = self.温度 if temperature is None else temperature
        top_p = self.top_p if top_p is None else top_p
        top_k = self.top_k if top_k is None else top_k
        repetition_penalty = self.repetition_penalty if repetition_penalty is None else repetition_penalty
        句子停止 = self.句子停止 if 句子停止 is None else 句子停止
        tokenizer = tokenizer or self.tokenizer
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id

        # ── 目标更新（含 β 重置）──
        self.β = self.β基
        self.更新目标(用户文本, 思考链文本, 指令, 角色=角色, 轮次=轮次)

        # ── 角色差异随机性（SubTask 6b）：按角色轻微扰动 v_target（确定性）──
        if 角色 and self.角色扰动幅度 > 0 and self.v_target is not None:
            self.v_target = self._角色扰动(self.v_target, 角色)

        # ── 提示模式：把当前维度锚点词注入 prompt（零模型内部访问）──
        if self.接口 == "提示" and 用户文本:
            input_ids = tokenizer(self.构造提示词(用户文本),
                                  return_tensors="pt").input_ids.to(input_ids.device)

        # ── 状态重置 ──
        past_key_values = None
        已生成 = input_ids.clone()
        已生成token集合: Set[int] = set()
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表 = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0
        熵列表: List[float] = []

        for 步 in range(max_new_tokens):
            # ── 前向传播 ──
            模型输入 = 已生成[:, -1:] if past_key_values is not None else 已生成
            outputs = self.model(模型输入, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]                    # (1, vocab_size)
            past_key_values = outputs.past_key_values

            # ── 重复惩罚 ──
            if repetition_penalty != 1.0:
                for tid in 已生成token集合:
                    logits[0, tid] /= repetition_penalty

            # ── (1) 锚点注入（延迟注入步数>0 时前 N 步不注入，先让分布定型）──
            if self.延迟注入步数 <= 0 or 步 >= self.延迟注入步数:
                logits = self.注入偏置(logits)

            if logits_callback is not None:
                logits_callback(步, logits)

            熵列表.append(计算熵(logits))

            # ── 最小长度：不足则压制 EOS，强制续写 ──
            if self.最小长度 > 0 and self._生成文本 and len(self._生成文本) < self.最小长度:
                if eos_token_id is not None and eos_token_id < self.vocab_size:
                    logits[0, eos_token_id] = float('-inf')

            # ── 温度缩放 ──
            logits = logits / temperature

            # ── Top-p 过滤 ──
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # ── Top-k 过滤 ──
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')

            # ── 采样 ──
            probs = F.softmax(logits, dim=-1)
            下一个token = torch.multinomial(probs, num_samples=1)

            已生成 = torch.cat([已生成, 下一个token], dim=-1)
            已生成token集合.add(下一个token.item())
            self._已生成token列表.append(下一个token.item())

            # ── 句级状态 / 密度 / 兜底 ──
            self._更新句状态(下一个token.item())
            # 每 退化窗口 个 token 强制检查一次退化
            if len(self._已生成token列表) >= 8 and len(self._已生成token列表) % self.退化窗口 == 0:
                self._兜底监测()

            if 轮次回调 is not None:
                轮次回调(步, self)
            if 下一个token.item() == eos_token_id:
                break
            if 句子停止 and self._句子停止():
                break

        统计 = {
            "平均熵": round(sum(熵列表) / len(熵列表), 4) if 熵列表 else 0.0,
            "重复率": 计算重复率(self._已生成token列表),
            "情感命中率": self._情感命中率(self._已生成token列表),
            "β": round(self.β, 4),
            "T_anchor": self.T_anchor,
            "稀疏阈值": self.稀疏阈值,
            "延迟注入步数": self.延迟注入步数,
            "v_target": None if self.v_target is None
            else [round(float(x), 4) for x in self.v_target],
            "触发兜底次数": self.触发兜底次数,
        }
        return 已生成, 统计

    # ──────────────────────────────────────────────
    # 句状态 / 兜底 / 指标
    # ──────────────────────────────────────────────

    def _更新句状态(self, token_id: int) -> None:
        文本 = self.tokenizer.decode([token_id], skip_special_tokens=True)
        if not 文本:
            return
        if token_id in self._情感token集:
            self._句情感词数 += 1
        self._生成文本 += 文本
        if self.句分隔符.search(文本):
            self._当前句文本 = ""
            self._句情感词数 = 0
            self._句子数 += 1
            self._兜底监测()  # 句结束也检查
        else:
            self._当前句文本 += 文本

    def _兜底监测(self) -> None:
        """在线退化兜底：最近 退化窗口 个 token 的 2-gram 重复率 > 兜底阈值
        或经验熵 < 0.6 → β×0.5；连续 3 次 → β=0（锚点通道静默，纯裸采样）。"""
        最近 = self._已生成token列表[-self.退化窗口:]
        if len(最近) < 8:
            return
        重复率 = 1.0 - len(set(tuple(最近[i:i + 2]) for i in range(len(最近) - 1))) / max(len(最近) - 1, 1)
        熵 = 经验熵(最近)
        if 重复率 > self.兜底阈值 or 熵 < 0.6:
            self.触发兜底次数 += 1
            self._兜底计数 += 1
            if self._兜底计数 >= 3:
                self.β = 0.0
            else:
                self.β *= 0.5
        else:
            self._兜底计数 = 0

    def _句子停止(self) -> bool:
        """v6 句子边界硬停止（短回复更像真人，复用 P3 结构）"""
        if not self._生成文本:
            return False
        if len(self._生成文本) >= self.最短字数 and self._句子数 >= self.最长句数:
            return True
        if self._句子数 >= self.最长句数 + 1:
            return True
        if len(self._生成文本) >= self.最大字数:
            return True
        return False

    def _情感命中率(self, token列表: List[int]) -> float:
        """生成文本中情感词 token 命中比例（cnsenti 词库优先，锚点库词集兜底）"""
        if not token列表:
            return 0.0
        命中 = sum(1 for t in token列表 if t in self._情感token集)
        return round(命中 / len(token列表), 4)

    def 重置(self) -> None:
        """清空句状态/目标/β（打分表不重建）"""
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表 = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0
        self.v_target = None
        self.β = self.β基

    def __repr__(self):
        return (f"锚点解码器(接口={self.接口}, β={self.β:.3f}, T_anchor={self.T_anchor}, "
                f"稀疏阈值={self.稀疏阈值}, 延迟注入步数={self.延迟注入步数}, "
                f"情感token={len(self._情感token集)}, 触发兜底={self.触发兜底次数})")
