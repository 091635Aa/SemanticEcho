# -*- coding: utf-8 -*-
"""
创新方案：思考链中断注入器（语义回响 二期）
==========================================
【不修改任何现有源码】——semantic_echo/*、echo_common、推理框架 均不动。
继承 回响注入器 复用：钩子注册（最后N层平均捕获 hidden_state）、随机静态投影矩阵、回响池。
仅重写 生成 循环，实现「思考链中断 + 总体向量注入」：

核心思想（与"全面向量纠正"的区别）：
  全面向量纠正（现有）：全程注入池质心，池随时序指数衰减变化；
  本方案（思考链纠正）：先让模型"思考"（Qwen3 预训练输出 <think>...</think>），
    思考阶段【只捕获情感向量、不注入】；
    检测到思考结束标记 </think> 时【硬中断】生成循环，
    计算思考阶段全部向量的【总体向量】（加权质心）；
    正文阶段用这个【定格总体向量】持续注入 → "情感先在脑中想好，再表达"。

流程：
  思考阶段 ──捕获(情感筛选)→ 池 ──检测到 </think>──硬中断→ 定格总体向量 ──正文注入──→ 输出
"""
import math
import torch
import torch.nn.functional as F
from semantic_echo.采样处理器 import 回响注入器
from semantic_echo.回响池 import 语义回响池


class 思考链中断注入器(回响注入器):
    """思考链纠正：思考结束中断 → 总体向量 → 正文固定注入"""

    def __init__(
        self,
        model,
        echo_pool: 语义回响池,
        tokenizer,
        lambda_strength: float = 1.0,
        思考结束token文本: str = "</think>",
        思考长度上限: int = 256,
        uncertainty_threshold: float = 0.01,
        projection_seed: int = 42,
        last_n_layers: int = 4,
        情感过滤器实例=None,
    ):
        # 不传 思考标记对/阶段λ（本方案用 token 级硬中断，不走字符串软切换）
        super().__init__(
            model, echo_pool, lambda_strength=lambda_strength,
            uncertainty_threshold=uncertainty_threshold,
            projection_seed=projection_seed, last_n_layers=last_n_layers,
            情感过滤器实例=情感过滤器实例,
        )
        self.tokenizer = tokenizer
        self.思考结束token = tokenizer.convert_tokens_to_ids(思考结束token文本)
        self.思考长度上限 = 思考长度上限
        self.思考步数 = 0
        self.总体向量 = None  # 思考阶段定格向量（正文注入源）
        self.阶段日志 = {"思考步数": 0, "是否中断": False, "总体范数": None}

    def 重置(self) -> None:
        """复用注入器：清空池与阶段状态（投影矩阵不重建，避免显存累积）"""
        self.pool.清空()
        self.思考步数 = 0
        self.总体向量 = None
        self.当前阶段 = "思考"
        self.阶段日志 = {"思考步数": 0, "是否中断": False, "总体范数": None}

    def _初始化投影(self, seed: int) -> None:
        """重写：投影矩阵直接在 GPU 分配（本机 CPU RAM 不足，父类 CPU 分配会 OOM）"""
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        scale = math.sqrt(2.0 / self.hidden_dim)
        self.投影矩阵 = torch.randn(
            self.hidden_dim, self.vocab_size,
            generator=rng, dtype=torch.float32, device=self.device,
        ) * scale
        self.投影矩阵.requires_grad_(False)

    @torch.no_grad()
    def 生成(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id=None,
        logits_callback=None,
        tokenizer=None,
        轮次回调=None,
    ) -> torch.Tensor:
        """重写生成循环：思考捕获 → </think>硬中断 → 定格总体向量 → 正文固定注入"""
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id

        past_key_values = None
        已生成 = input_ids.clone()
        已生成token集合 = set()
        self.思考步数 = 0
        self.总体向量 = None
        self.当前阶段 = "思考"

        for 步 in range(max_new_tokens):
            模型输入 = 已生成[:, -1:] if past_key_values is not None else 已生成
            outputs = self.model(模型输入, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # ── 重复惩罚 ──
            if repetition_penalty != 1.0:
                for token_id in 已生成token集合:
                    logits[0, token_id] /= repetition_penalty

            if self.当前阶段 == "思考":
                # 思考阶段：只捕获（情感筛选入池），不注入
                self.捕获回响(logits, tokenizer=tokenizer or self.tokenizer)
            else:
                # 正文阶段：用定格总体向量固定注入
                if self.总体向量 is not None:
                    偏置 = self.总体向量.to(self.device) @ self.投影矩阵.to(self.device)
                    logits = logits + 偏置.unsqueeze(0) * self.lambda_strength

            if logits_callback is not None:
                logits_callback(步, logits)

            # ── 采样 ──
            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < top_k_values[:, -1].unsqueeze(-1)] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            下一个token = torch.multinomial(probs, num_samples=1)

            # ── 思考结束检测（token 级硬中断） ──
            if self.当前阶段 == "思考":
                self.思考步数 += 1
                if (self.思考结束token is not None
                        and 下一个token.item() == self.思考结束token) or self.思考步数 >= self.思考长度上限:
                    # 硬中断：定格思考阶段总体向量
                    self.总体向量 = self.pool.计算质心()
                    self.当前阶段 = "正文"
                    self.阶段日志["思考步数"] = self.思考步数
                    self.阶段日志["是否中断"] = True
                    self.阶段日志["总体范数"] = round(self.总体向量.norm().item(), 4)
                    self.pool.清空()  # 释放思考阶段向量，正文不再更新
                    print(f"  [思考链中断] 思考 {self.思考步数} 步 → 总体向量范数={self.阶段日志['总体范数']}", flush=True)

            已生成 = torch.cat([已生成, 下一个token], dim=-1)
            已生成token集合.add(下一个token.item())
            self.pool.推进()

            if 轮次回调 is not None:
                轮次回调(self.pool.当前步数, self.pool)
            if 下一个token.item() == eos_token_id:
                break

        return 已生成
