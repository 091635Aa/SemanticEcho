"""
采样处理器 — 语义回响的核心推理引擎

管理 hidden_state 钩子、随机静态投影、回响注入与捕获的完整生命周期。

论文对应：第 3.1–3.6 节
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Callable, List, Dict
from transformers import PreTrainedModel

from semantic_echo.回响池 import 语义回响池
from semantic_echo.情感过滤器 import 情感过滤器
from semantic_echo.翻译毒药 import 语义回响异常, 获取错误码


# ══════════════════════════════════════════════════
# 辅助：自动定位模型最后一层
# ══════════════════════════════════════════════════

def _定位最后一层(model: PreTrainedModel) -> torch.nn.Module:
    """
    根据模型架构自动定位最后一层 Transformer 层。

    Raises
    ------
    ValueError
        如果模型架构不属于已知模式
    """
    # LLaMA / Qwen / Mistral / DeepSeek 等
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[-1]

    # GPT-2 系列
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h[-1]

    # OPT / BLOOM
    if (
        hasattr(model, 'model')
        and hasattr(model.model, 'decoder')
        and hasattr(model.model.decoder, 'layers')
    ):
        return model.model.decoder.layers[-1]

    raise ValueError(
        f"无法自动定位模型 {type(model).__name__} 的最后一层。"
        f"支持：LLaMA/Qwen/Mistral/GPT-2/OPT/BLOOM"
    )


# ══════════════════════════════════════════════════
# 回响注入器
# ══════════════════════════════════════════════════

class 回响注入器:
    """
    语义回响注入器 — 核心推理增强模块。

    使用方式
    --------
    >>> pool = 语义回响池(hidden_dim=model.config.hidden_size)
    >>> injector = 回响注入器(model, pool, lambda_strength=1.0)
    >>> output_ids = injector.生成(input_ids, max_new_tokens=256)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        echo_pool: 语义回响池,
        lambda_strength: float = 1.0,
        uncertainty_threshold: float = 0.01,
        projection_seed: int = 42,
        last_n_layers: int = 4,
        情感过滤器实例: Optional[情感过滤器] = None,
        思考标记对: tuple[str, str] = ("", ""),
        思考阶段λ: Optional[float] = None,
        正文阶段λ: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        model : PreTrainedModel
            HuggingFace Transformers 兼容的预训练模型
        echo_pool : 语义回响池
            共享的回响池实例
        lambda_strength : float
            注入偏置的强度系数
        uncertainty_threshold : float
            不确定性权重低于此阈值时不入池（避免噪声积累）
        projection_seed : int
            随机投影矩阵的固定种子
        last_n_layers : int
            取最后 N 层的 hidden_state 平均作为"语义场"向量
        情感过滤器实例 : Optional[情感过滤器]
            情感筛选器实例，为 None 时跳过情感筛选（兼容旧行为）
        思考标记对 : tuple[str, str]
            思考阶段边界标记，如 ("<think>", "</think>")；为空时行为与升级前一致
        思考阶段λ : Optional[float]
            思考阶段的注入强度，为 None 时与 lambda_strength 相同
        正文阶段λ : float
            正文阶段的注入强度，为 0.0 时正文阶段不注入
        """
        self.model = model
        self.pool = echo_pool
        self.lambda_strength = lambda_strength
        self.uncertainty_threshold = uncertainty_threshold
        self.last_n_layers = last_n_layers

        # 阶段相关参数
        self.思考标记对 = 思考标记对
        self.思考阶段λ = 思考阶段λ
        self.正文阶段λ = 正文阶段λ

        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        self.device = model.device

        # 当前步捕获的 hidden_state（多层平均），在 forward hook 中更新
        self.当前hidden_state: Optional[torch.Tensor] = None

        # 阶段状态跟踪
        self.当前阶段: str = "思考"
        self.已解码文本: str = ""
        self.情感过滤器引用: Optional[情感过滤器] = 情感过滤器实例

        # 随机静态投影矩阵
        self._初始化投影(projection_seed)

        # 注册 forward hook
        self._钩子列表: List[torch.utils.hooks.RemovableHandle] = []
        self._注册钩子()

    # ──────────────────────────────────────────────
    # 投影矩阵初始化
    # ──────────────────────────────────────────────

    def _初始化投影(self, seed: int) -> None:
        """
        创建固定的随机投影矩阵：hidden_dim → vocab_size

        使用 Kaiming 均匀初始化缩放因子，保证投影后输出的
        方差 ≈ 输入方差，避免注入偏置过大或过小。
        """
        rng = torch.Generator()
        rng.manual_seed(seed)

        scale = math.sqrt(2.0 / self.hidden_dim)
        self.投影矩阵 = torch.randn(
            self.hidden_dim, self.vocab_size,
            generator=rng,
            dtype=torch.float32,
        ) * scale

        # 固定为常量，永不参与梯度计算
        self.投影矩阵.requires_grad_(False)

    # ──────────────────────────────────────────────
    # Forward Hook 注册
    # ──────────────────────────────────────────────

    def _注册钩子(self) -> None:
        """
        在模型最后 N 层注册 forward hook，捕获每步的 hidden_state。

        如果是标准 Decoder-only 架构（model.model.layers），
        则对最后 last_n_layers 层各注册一个钩子，取输出平均。

        否则退化为只捕获最后一层的输出。

        Raises
        ------
        语义回响异常
            当钩子注册失败时抛出
        """
        try:
            # 支持多层平均的标准架构
            if (
                hasattr(self.model, 'model')
                and hasattr(self.model.model, 'layers')
            ):
                所有层 = self.model.model.layers
                目标层数 = min(self.last_n_layers, len(所有层))
                起始索引 = len(所有层) - 目标层数
                self.目标层索引 = list(range(起始索引, len(所有层)))

                for idx in self.目标层索引:
                    handle = 所有层[idx].register_forward_hook(self._创建多层钩子(idx))
                    self._钩子列表.append(handle)
            else:
                # 退化为单层模式
                最后一层 = _定位最后一层(self.model)
                handle = 最后一层.register_forward_hook(self._单层钩子)
                self._钩子列表.append(handle)
        except Exception as e:
            raise 语义回响异常("钩子注册失败", str(e))

    def _单层钩子(
        self,
        module: torch.nn.Module,
        inputs: tuple,
        output: tuple,
    ) -> None:
        """单层钩子：直接从输出中提取最后一个位置的 hidden_state（1D 向量）"""
        if isinstance(output, tuple):
            hs = output[0][0, -1, :]  # squeeze batch dim: (1, seq_len, dim) -> (dim,)
        else:
            hs = output[0, -1, :]
        self.当前hidden_state = hs.detach().clone()

    def _创建多层钩子(self, layer_idx: int) -> Callable:
        """
        多层钩子工厂：为指定层创建钩子，收集所有目标层输出后取平均。

        Notes
        -----
        使用实例字典 _层输出缓存 暂存各层输出，
        当所有目标层都捕获完毕时计算平均并写入 self.当前hidden_state。
        """
        def hook(
            module: torch.nn.Module,
            inputs: tuple,
            output: tuple,
        ) -> None:
            if isinstance(output, tuple):
                hs = output[0][0, -1, :]  # (1, seq_len, dim) -> (dim,)
            else:
                hs = output[0, -1, :]

            if not hasattr(self, '_层输出缓存'):
                self._层输出缓存: Dict[int, torch.Tensor] = {}
            self._层输出缓存[layer_idx] = hs.detach()

            # 所有目标层都到位时，计算平均
            if len(self._层输出缓存) == len(self.目标层索引):
                向量列表 = [self._层输出缓存[i] for i in sorted(self.目标层索引)]
                self.当前hidden_state = torch.stack(向量列表).mean(dim=0)
                self._层输出缓存.clear()

        return hook

    def _移除钩子(self) -> None:
        """移除所有注册的 forward hook"""
        for handle in self._钩子列表:
            handle.remove()
        self._钩子列表.clear()

    # ──────────────────────────────────────────────
    # 阶段切换检测
    # ──────────────────────────────────────────────

    def _检测阶段切换(self, 新token_id: int, tokenizer) -> bool:
        """
        检测是否进入了新阶段（思考→正文）。

        通过检测已生成文本中是否出现了思考结束标记来判断。

        Parameters
        ----------
        新token_id : int
            当前步采样得到的新 token ID
        tokenizer : AutoTokenizer
            HuggingFace tokenizer，用于将 token_id 解码为文本

        Returns
        -------
        bool
            是否发生了阶段切换（思考→正文）
        """
        # 1. 将 token_id 解码后追加到已解码文本
        新文本 = tokenizer.decode([新token_id]) if tokenizer else ""
        self.已解码文本 += 新文本

        # 2. 判断阶段切换：思考阶段且标记对不为空且已遇到结束标记
        if (
            self.思考标记对
            and self.思考标记对 != ("", "")
            and self.当前阶段 == "思考"
            and self.思考标记对[1] in self.已解码文本
        ):
            self.当前阶段 = "正文"
            return True

        # 3. 正文阶段保持不变
        return False

    # ──────────────────────────────────────────────
    # 核心操作：注入 + 捕获
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 注入偏置(self, logits: torch.Tensor) -> torch.Tensor:
        """
        将回响池质心通过随机投影映射到 logits 空间，作为偏置注入。

        根据当前阶段选择不同的 λ 强度系数：
        - 思考阶段：使用 思考阶段λ（若不为 None）或 lambda_strength
        - 正文阶段：使用 正文阶段λ（为 0 时不注入）

        Parameters
        ----------
        logits : torch.Tensor
            shape=(1, vocab_size)，当前步的原始 logits

        Returns
        -------
        torch.Tensor
            注入偏置后的 logits，shape 不变
        """
        # 正文阶段且正文阶段λ=0时，不注入
        if self.当前阶段 == "正文" and self.正文阶段λ == 0.0:
            return logits

        if self.pool.是否为空:
            return logits

        质心 = self.pool.计算质心().to(self.device)

        # 根据阶段选择不同的 λ
        if self.当前阶段 == "思考" and self.思考阶段λ is not None:
            有效λ = self.思考阶段λ
        elif self.当前阶段 == "正文":
            有效λ = self.正文阶段λ
        else:
            有效λ = self.lambda_strength

        # 质心 @ 投影矩阵 → shape=(vocab_size,)
        偏置 = 质心 @ self.投影矩阵.to(self.device)
        偏置 = 偏置 * 有效λ

        return logits + 偏置.unsqueeze(0)

    @torch.no_grad()
    def 捕获回响(self, logits: torch.Tensor, tokenizer=None) -> None:
        """
        将当前 hidden_state 存入回响池（带情感筛选）。

        如果配置了情感过滤器，只存储命中情感词库的 token。

        Parameters
        ----------
        logits : torch.Tensor
            shape=(1, vocab_size)，注入偏置后的 logits
        tokenizer : AutoTokenizer, optional
            HuggingFace tokenizer，用于将 token_id 解码为文本进行情感筛选
        """
        if self.当前hidden_state is None:
            return

        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max().item()
        不确定性 = 1.0 - max_prob

        # 不确定性阈值筛选
        if 不确定性 <= self.uncertainty_threshold:
            return

        # 情感相关性筛选
        情感权重 = None
        if self.情感过滤器引用 is not None and tokenizer is not None:
            # 获取 top-1 token
            top_token_id = logits.argmax(dim=-1).item()
            top_token_text = tokenizer.decode([top_token_id])

            # 筛选
            命中 = self.情感过滤器引用.筛选([top_token_text], tokenizer)
            if not 命中:
                # 未命中情感词，不入池
                self.pool.自增检查()
                return

            情感权重 = 命中[0][1]  # 使用情感强度

        # 入池（带情感权重）
        self.pool.添加(self.当前hidden_state, 不确定性, 情感相关性=情感权重)

    # ──────────────────────────────────────────────
    # 自定义生成循环
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 生成(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        logits_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        tokenizer=None,  # 新增：用于情感筛选和阶段检测
        轮次回调: Optional[Callable[[int, object], None]] = None,
    ) -> torch.Tensor:
        """
        带语义回响的自回归生成循环。

        与 HuggingFace model.generate() 签名保持一致，
        每步执行：前向 → 注入 → 捕获 → 采样 → 推进

        Parameters
        ----------
        input_ids : torch.Tensor
            shape=(1, seq_len)，初始 prompt token ID 序列
        max_new_tokens : int
            最大新生成 token 数
        temperature : float
            采样温度，>1 更随机，<1 更确定
        top_p : float
            nucleus sampling 累积概率阈值
        top_k : int
            top-k 采样保留的候选数
        repetition_penalty : float
            重复惩罚系数（>1 抑制重复）
        eos_token_id : Optional[int]
            结束标记 ID，为 None 时从模型配置获取
        logits_callback : Optional[Callable[[int, torch.Tensor], None]]
            可选回调函数，每步生成后调用，传入 (步数, logits)，
            用于外部记录每一步的 logits（如实验记录）
        tokenizer : AutoTokenizer, optional
            HuggingFace tokenizer，用于情感筛选和思考阶段边界检测
        轮次回调 : Optional[Callable[[int, object], None]]
            可选回调函数，每步推进后调用，传入 (当前步数, self.pool)，
            用于外部控制轮次切换（如滑动窗口策略）

        Returns
        -------
        torch.Tensor
            shape=(1, total_len)，完整生成的 token ID 序列
        """
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id

        past_key_values: Optional[tuple] = None
        已生成 = input_ids.clone()
        已生成token集合: set = set()

        for 步 in range(max_new_tokens):
            # ── 前向传播 ──
            # 注：hook 会在 forward 中自动填充 self.当前hidden_state
            # 当使用 KV cache 时，只需传入最后一个新 token
            模型输入 = 已生成[:, -1:] if past_key_values is not None else 已生成
            outputs = self.model(
                模型输入,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            past_key_values = outputs.past_key_values

            # ── 重复惩罚 ──
            if repetition_penalty != 1.0:
                for token_id in 已生成token集合:
                    logits[0, token_id] /= repetition_penalty

            # ── (1) 注入：将回响池偏置加到 logits ──
            logits = self.注入偏置(logits)

            # ── 外部日志回调（用于实验记录） ──
            if logits_callback is not None:
                logits_callback(步, logits)

            # ── (2) 捕获：将当前 hidden_state 存入回响池 ──
            self.捕获回响(logits, tokenizer=tokenizer)

            # ── 温度缩放 ──
            logits = logits / temperature

            # ── Top-p (nucleus) 过滤 ──
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过 top_p 的 token（至少保留一个）
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove,
                )
                logits[indices_to_remove] = float('-inf')

            # ── Top-k 过滤 ──
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')

            # ── 采样 ──
            probs = F.softmax(logits, dim=-1)
            下一个token = torch.multinomial(probs, num_samples=1)

            # ── (3) 检测阶段切换（思考→正文） ──
            self._检测阶段切换(下一个token.item(), tokenizer)

            已生成 = torch.cat([已生成, 下一个token], dim=-1)
            已生成token集合.add(下一个token.item())

            # ── (4) 推进回响池步数 ──
            self.pool.推进()

            # ── (5) 外部轮次回调（用于滑动窗口策略） ──
            if 轮次回调 is not None:
                轮次回调(self.pool.当前步数, self.pool)

            if 下一个token.item() == eos_token_id:
                break

        return 已生成

    def __del__(self) -> None:
        """清理注册的 hook，防止内存泄漏"""
        if hasattr(self, '_钩子列表'):
            self._移除钩子()
