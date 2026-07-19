"""
实验运行器 — 编排完整的语义回响实验流程

支持基线模式（标准 HuggingFace generate）与回响模式（回响注入器），
自动对比并生成可视化报告。

用法：
    python 实验运行器.py
"""

import os
import json
import math
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Callable, List, Dict, Any, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.回响池 import 语义回响池
from src.采样处理器 import 回响注入器


# ══════════════════════════════════════════════════
# 测试提示词集（情感维度 × 3条 = 15条）
# ══════════════════════════════════════════════════

测试提示词: Dict[str, List[str]] = {
    "开心": [
        "你今天真好看",
        "终于等到你了，我好开心",
        "今天的中标消息让我兴奋得睡不着",
    ],
    "悲伤": [
        "一切都结束了",
        "他走了，再也不会回来了",
        "我好像再也找不到活下去的意义了",
    ],
    "愤怒": [
        "你凭什么这么说我",
        "这个结果简直是荒谬至极",
        "我受够了你们的欺骗和背叛",
    ],
    "中性": [
        "今天天气不错",
        "我想了解一下这个产品的功能",
        "请问地铁站怎么走",
    ],
    "复杂混合": [
        "虽然赢了比赛，但我最好的朋友受伤了",
        "我爱我的工作，但是工资真的太低了",
        "你给了我这么多帮助，我却没办法回报你",
    ],
}


# ══════════════════════════════════════════════════
# 实验配置
# ══════════════════════════════════════════════════


@dataclass
class 实验配置:
    """
    单次实验的配置参数。

    Parameters
    ----------
    实验编号 : str
        实验唯一标识（如 "E1"）
    条件描述 : str
        实验条件名称（如 "Baseline top_p=0.9"）
    lambda_strength : Optional[float]
        回响注入强度，为 None 时表示基线模式
    decay_gamma : Optional[float]
        回响池指数衰减系数，为 None 时表示基线模式
    temperature : float
        采样温度
    top_p : float
        nucleus sampling 累积概率阈值
    top_k : int
        top-k 采样保留候选数（0 表示禁用）
    max_new_tokens : int
        最大新生成 token 数
    重复次数 : int
        每个提示词的重复运行次数
    """

    实验编号: str
    条件描述: str
    lambda_strength: Optional[float] = None
    decay_gamma: Optional[float] = None
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 128
    重复次数: int = 3

    @property
    def 是回响模式(self) -> bool:
        """是否为回响注入模式（lambda_strength 不为 None）"""
        return self.lambda_strength is not None and self.decay_gamma is not None


# 预定义实验矩阵
实验配置列表: List[实验配置] = [
    实验配置("E1", "Baseline (top_p=0.9)", None, None, 1.0, 0.9, 50),
    实验配置("E2", "Baseline (temperature=1.0)", None, None, 1.0, 1.0, 0),
    实验配置("E3", "Echo (λ=0.5, γ=0.05)", 0.5, 0.05, 1.0, 0.9, 50),
    实验配置("E4", "Echo (λ=1.0, γ=0.1)", 1.0, 0.1, 1.0, 0.9, 50),
    实验配置("E5", "Echo (λ=2.0, γ=0.5)", 2.0, 0.5, 1.0, 0.9, 50),
    实验配置("E6", "Echo (λ=1.0, γ=0.01)", 1.0, 0.01, 1.0, 0.9, 50),
]


# ══════════════════════════════════════════════════
# 评估类
# ══════════════════════════════════════════════════


class 逐Token评估器:
    """
    逐 Token 评估器 — 记录并计算每一步的 logits 统计信息。

    跟踪每一步的 logits 分布，计算熵、最大概率、困惑度等指标。

    Parameters
    ----------
    vocab_size : int
        词汇表大小，用于熵计算
    """

    def __init__(self, vocab_size: int) -> None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size 必须为正整数，收到 {vocab_size}")
        self.vocab_size = vocab_size

        # 每步记录
        self.step_logits: List[torch.Tensor] = []
        self.step_entropy: List[float] = []
        self.step_max_prob: List[float] = []
        self.step_tokens: List[int] = []

        # 累计统计
        self.total_steps: int = 0

    def 记录(self, step: int, logits: torch.Tensor) -> None:
        """
        记录一步的 logits 并计算统计量。

        Parameters
        ----------
        step : int
            当前生成步数（0-indexed）
        logits : torch.Tensor
            shape=(1, vocab_size) 的原始 logits
        """
        if logits.dim() != 2 or logits.shape[0] != 1:
            raise ValueError(
                f"logits 应为 (1, vocab_size)，收到 shape={logits.shape}"
            )

        # 记录 logits（移到 CPU，分离计算图）
        self.step_logits.append(logits.detach().cpu())

        # 转换为 float32 避免 float16 精度问题
        logits_fp32 = logits.float() if logits.dtype == torch.float16 else logits
        # 替换 -inf 为有限负值，防止 0 * log(0) = NaN
        inf_mask = torch.isinf(logits_fp32) & (logits_fp32 < 0)
        if inf_mask.any():
            logits_fp32 = logits_fp32.clone()
            logits_fp32[inf_mask] = -1e9
        probs = F.softmax(logits_fp32, dim=-1)
        log_probs = F.log_softmax(logits_fp32, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).item()

        # 计算最大概率
        max_prob = probs.max().item()

        self.step_entropy.append(entropy)
        self.step_max_prob.append(max_prob)
        self.total_steps += 1

    def 记录token(self, token_id: int) -> None:
        """记录当前步实际采样的 token ID"""
        self.step_tokens.append(token_id)

    # ── 统计属性 ──

    @property
    def 平均熵(self) -> float:
        """所有步的平均熵"""
        if not self.step_entropy:
            return 0.0
        return sum(self.step_entropy) / len(self.step_entropy)

    @property
    def 熵方差(self) -> float:
        """熵的方差"""
        if len(self.step_entropy) < 2:
            return 0.0
        均值 = self.平均熵
        return sum((h - 均值) ** 2 for h in self.step_entropy) / len(self.step_entropy)

    @property
    def 平均置信度(self) -> float:
        """所有步的平均最大概率（置信度）"""
        if not self.step_max_prob:
            return 0.0
        return sum(self.step_max_prob) / len(self.step_max_prob)

    @property
    def 平均困惑度(self) -> float:
        """平均困惑度：exp(平均熵)"""
        平均熵 = self.平均熵
        return math.exp(平均熵) if 平均熵 > 0 else float('inf')

    @property
    def 熵序列(self) -> List[float]:
        """返回熵序列的副本"""
        return list(self.step_entropy)

    @property
    def logits序列(self) -> List[torch.Tensor]:
        """返回 logits 序列的副本"""
        return list(self.step_logits)

    @torch.no_grad()
    def 计算KL散度(
        self, other: '逐Token评估器'
    ) -> float:
        """
        计算当前评估器与另一个评估器的平均 KL 散度。

        D_KL(P || Q) = sum(p * log(p/q))
        以当前评估器的分布为 P，另一个为 Q。

        Parameters
        ----------
        other : 逐Token评估器
            另一个评估器（作为 Q）

        Returns
        -------
        float
            平均 KL 散度

        Raises
        ------
        ValueError
            如果步数不匹配
        """
        if len(self.step_logits) != len(other.step_logits):
            raise ValueError(
                f"步数不匹配：当前 {len(self.step_logits)} vs 对方 {len(other.step_logits)}"
            )
        if not self.step_logits:
            return 0.0

        kl_sum = 0.0
        for logits_p, logits_q in zip(self.step_logits, other.step_logits):
            # logits -> log-probabilities (数值稳定)
            log_p = F.log_softmax(logits_p, dim=-1)
            p = F.softmax(logits_p, dim=-1)
            log_q = F.log_softmax(logits_q, dim=-1)

            # KL = sum(p * (log_p - log_q))
            kl = (p * (log_p - log_q)).sum(dim=-1).item()
            kl_sum += kl

        return kl_sum / len(self.step_logits)

    def 汇总(self) -> Dict[str, Any]:
        """返回汇总统计字典"""
        return {
            "步数": self.total_steps,
            "平均熵": self.平均熵,
            "熵方差": self.熵方差,
            "平均置信度": self.平均置信度,
            "平均困惑度": self.平均困惑度,
            "熵序列": self.熵序列,
        }

    def __repr__(self) -> str:
        return (
            f"逐Token评估器(步数={self.total_steps}, "
            f"平均熵={self.平均熵:.4f}, "
            f"平均置信度={self.平均置信度:.4f})"
        )


class 实验对比器:
    """
    实验对比器 — 对比基线生成与回响生成的评估结果。

    对于同一条提示词，关联其基线结果与回响结果，
    提供丰富的对比指标。

    Parameters
    ----------
    提示词 : str
        实验用提示词
    情感维度 : str
        提示词所属情感维度
    配置 : 实验配置
        实验配置
    基线结果 : Optional[Dict[str, Any]]
        基线模式生成结果，含 "文本" 和 "评估器"
    回响结果 : Optional[Dict[str, Any]]
        回响模式生成结果，含 "文本"、"评估器"、"池统计"
    重复索引 : int
        第几次重复运行
    """

    def __init__(
        self,
        提示词: str,
        情感维度: str,
        配置: 实验配置,
        基线结果: Optional[Dict[str, Any]] = None,
        回响结果: Optional[Dict[str, Any]] = None,
        重复索引: int = 0,
    ) -> None:
        self.提示词 = 提示词
        self.情感维度 = 情感维度
        self.配置 = 配置
        self.重复索引 = 重复索引

        # 存储结果
        self.基线文本: Optional[str] = None
        self.基线评估器: Optional[逐Token评估器] = None
        self.回响文本: Optional[str] = None
        self.回响评估器: Optional[逐Token评估器] = None
        self.池统计: Optional[Dict[str, Any]] = None

        if 基线结果 is not None:
            self.基线文本 = 基线结果.get("文本", "")
            self.基线评估器 = 基线结果.get("评估器")

        if 回响结果 is not None:
            self.回响文本 = 回响结果.get("文本", "")
            self.回响评估器 = 回响结果.get("评估器")
            self.池统计 = 回响结果.get("池统计")

        # 对比指标由 lazy 计算
        self._KL散度: Optional[float] = None
        self._细腻度提升率: Optional[float] = None

    # ── 对比指标 ──

    @property
    def KL散度(self) -> Optional[float]:
        """基线 vs 回响的 KL 散度"""
        if self._KL散度 is not None:
            return self._KL散度
        if self.基线评估器 is None or self.回响评估器 is None:
            return None
        try:
            self._KL散度 = self.基线评估器.计算KL散度(self.回响评估器)
        except (ValueError, RuntimeError) as e:
            warnings.warn(f"KL 散度计算失败: {e}")
            self._KL散度 = None
        return self._KL散度

    @property
    def 细腻度提升率(self) -> Optional[float]:
        """
        回响 vs 基线在平均熵上的提升率。

        正值表示回响增加了分布的平坦度（探索性更强），
        负值表示回响使分布更确定。

        返回百分比：((回响熵 - 基线熵) / 基线熵) * 100
        """
        if self._细腻度提升率 is not None:
            return self._细腻度提升率
        if self.基线评估器 is None or self.回响评估器 is None:
            return None
        基线熵 = self.基线评估器.平均熵
        if 基线熵 == 0:
            return 0.0
        self._细腻度提升率 = (
            (self.回响评估器.平均熵 - 基线熵) / 基线熵
        ) * 100.0
        return self._细腻度提升率

    def 汇总(self) -> Dict[str, Any]:
        """返回汇总字典"""
        result: Dict[str, Any] = {
            "提示词": self.提示词,
            "情感维度": self.情感维度,
            "实验编号": self.配置.实验编号,
            "条件描述": self.配置.条件描述,
            "重复索引": self.重复索引,
        }

        if self.基线评估器 is not None:
            result["基线"] = self.基线评估器.汇总()
            result["基线"]["文本"] = self.基线文本

        if self.回响评估器 is not None:
            result["回响"] = self.回响评估器.汇总()
            result["回响"]["文本"] = self.回响文本

        kl = self.KL散度
        if kl is not None:
            result["KL散度"] = kl

        提升率 = self.细腻度提升率
        if 提升率 is not None:
            result["细腻度提升率"] = 提升率

        if self.池统计 is not None:
            result["池统计"] = self.池统计

        return result

    def __repr__(self) -> str:
        return (
            f"实验对比器({self.配置.实验编号}, "
            f"维度={self.情感维度}, 重复#{self.重复索引})"
        )


class 汇总统计器:
    """
    汇总统计器 — 聚合所有实验对比器结果，提供全局统计报告。

    Parameters
    ----------
    实验对比器列表 : List[实验对比器]
        所有实验的对比器列表
    """

    def __init__(self, 实验对比器列表: List[实验对比器]) -> None:
        if not 实验对比器列表:
            raise ValueError("实验对比器列表不能为空")
        self.实验对比器列表 = 实验对比器列表

    # ── 按实验编号分组 ──

    def 按实验分组(self) -> Dict[str, List[实验对比器]]:
        """按实验编号分组"""
        分组: Dict[str, List[实验对比器]] = {}
        for 对比器 in self.实验对比器列表:
            分组.setdefault(对比器.配置.实验编号, []).append(对比器)
        return 分组

    # ── 按情感维度分组 ──

    def 按维度分组(self) -> Dict[str, List[实验对比器]]:
        """按情感维度分组"""
        分组: Dict[str, List[实验对比器]] = {}
        for 对比器 in self.实验对比器列表:
            分组.setdefault(对比器.情感维度, []).append(对比器)
        return 分组

    # ── 全局指标 ──

    @property
    def 平均KL散度(self) -> float:
        """所有对比器的平均 KL 散度"""
        kl_values = [
            c.KL散度 for c in self.实验对比器列表
            if c.KL散度 is not None
        ]
        if not kl_values:
            return 0.0
        return sum(kl_values) / len(kl_values)

    @property
    def 平均细腻度提升率(self) -> float:
        """所有对比器的平均细腻度提升率"""
        提升率列表 = [
            c.细腻度提升率 for c in self.实验对比器列表
            if c.细腻度提升率 is not None
        ]
        if not 提升率列表:
            return 0.0
        return sum(提升率列表) / len(提升率列表)

    @property
    def 总实验数(self) -> int:
        """总实验次数"""
        return len(self.实验对比器列表)

    # ── 汇总输出 ──

    def 汇总(self) -> Dict[str, Any]:
        """返回完整汇总字典"""
        # 按实验分组汇总
        实验汇总: Dict[str, Dict[str, Any]] = {}
        for 实验号, 对比器列表 in self.按实验分组().items():
            kl_values = [
                c.KL散度 for c in 对比器列表 if c.KL散度 is not None
            ]
            提升率列表 = [
                c.细腻度提升率 for c in 对比器列表
                if c.细腻度提升率 is not None
            ]
            实验汇总[实验号] = {
                "条件描述": 对比器列表[0].配置.条件描述,
                "运行次数": len(对比器列表),
                "平均KL散度": sum(kl_values) / len(kl_values) if kl_values else None,
                "平均细腻度提升率": (
                    sum(提升率列表) / len(提升率列表) if 提升率列表 else None
                ),
            }

        return {
            "总实验数": self.总实验数,
            "平均KL散度": self.平均KL散度,
            "平均细腻度提升率": self.平均细腻度提升率,
            "实验汇总": 实验汇总,
            "时间戳": datetime.now().isoformat(),
        }

    def 导出JSON(self, 文件路径: str) -> None:
        """
        将汇总结果导出为 JSON 文件。

        Parameters
        ----------
        文件路径 : str
            输出 JSON 文件路径
        """
        data = self.汇总()
        # 处理不可序列化的类型
        with open(文件路径, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def __repr__(self) -> str:
        return (
            f"汇总统计器(总实验数={self.总实验数}, "
            f"平均KL散度={self.平均KL散度:.4f}, "
            f"平均提升率={self.平均细腻度提升率:.2f}%)"
        )


# ══════════════════════════════════════════════════
# 实验运行器
# ══════════════════════════════════════════════════


class 实验运行器:
    """
    实验运行器 — 管理完整的实验流程。

    包含基线生成、回响生成、结果对比、可视化和报告输出。

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace 预训练模型
    tokenizer : AutoTokenizer
        HuggingFace 分词器
    输出目录 : str
        实验数据输出根目录，默认为 "./实验数据"
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        输出目录: str = "./实验数据",
    ) -> None:
        if model is None:
            raise ValueError("model 不能为 None")
        if tokenizer is None:
            raise ValueError("tokenizer 不能为 None")

        self.model = model
        self.tokenizer = tokenizer
        self.输出目录 = os.path.abspath(输出目录)

        # 创建子目录
        self.可视化目录 = os.path.join(self.输出目录, "可视化")
        os.makedirs(self.可视化目录, exist_ok=True)

        # 设备
        self.device = model.device

        # 词汇表大小
        self.vocab_size = model.config.vocab_size

        # 实验记录
        self.所有对比器: List[实验对比器] = []

    # ──────────────────────────────────────────────
    # 单次生成
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _基线生成(
        self, 提示词: str, 配置: 实验配置
    ) -> Dict[str, Any]:
        """
        基线模式：使用 model.generate() 标准流程。

        Parameters
        ----------
        提示词 : str
            输入提示词
        配置 : 实验配置
            实验配置参数

        Returns
        -------
        Dict[str, Any]
            包含 "文本" 和 "评估器" 的字典
        """
        # Tokenize
        inputs = self.tokenizer(提示词, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        # 创建评估器
        评估器 = 逐Token评估器(self.vocab_size)

        # 使用 model.generate 生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=配置.max_new_tokens,
            temperature=配置.temperature,
            top_p=配置.top_p,
            top_k=配置.top_k if 配置.top_k > 0 else None,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 记录每步 logits 到评估器
        # outputs.scores 是 tuple of (1, vocab_size) tensors
        for step_idx, step_logits in enumerate(outputs.scores):
            评估器.记录(step_idx, step_logits)

        # 解码生成的文本（仅新生成部分）
        generated_ids = outputs.sequences[0, input_len:]
        生成文本 = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 记录每一步实际 token
        for token_id in generated_ids.tolist():
            评估器.记录token(token_id)

        return {"文本": 生成文本, "评估器": 评估器}

    @torch.no_grad()
    def _回响生成(
        self, 提示词: str, 配置: 实验配置
    ) -> Dict[str, Any]:
        """
        回响模式：使用 回响注入器 的生成流程。

        Parameters
        ----------
        提示词 : str
            输入提示词
        配置 : 实验配置
            实验配置参数

        Returns
        -------
        Dict[str, Any]
            包含 "文本"、"评估器"、"池统计" 的字典
        """
        # Tokenize
        inputs = self.tokenizer(提示词, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        # 创建回响池和回响注入器
        hidden_dim = self.model.config.hidden_size
        回响池 = 语义回响池(
            hidden_dim=hidden_dim,
            max_pool_size=1024,
            decay_gamma=配置.decay_gamma,  # type: ignore[arg-type]
        )
        注入器 = 回响注入器(
            model=self.model,
            echo_pool=回响池,
            lambda_strength=配置.lambda_strength,  # type: ignore[arg-type]
        )

        # 创建评估器
        评估器 = 逐Token评估器(self.vocab_size)

        # 定义 logits 回调
        def _logits回调(step: int, logits: torch.Tensor) -> None:
            评估器.记录(step, logits)

        # 执行回响生成
        full_ids = 注入器.生成(
            input_ids=inputs.input_ids,
            max_new_tokens=配置.max_new_tokens,
            temperature=配置.temperature,
            top_p=配置.top_p,
            top_k=配置.top_k,
            logits_callback=_logits回调,
        )

        # 解码
        generated_ids = full_ids[0, input_len:]
        生成文本 = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 记录 token
        for token_id in generated_ids.tolist():
            评估器.记录token(token_id)

        # 收集池统计
        池统计 = {
            "池大小": 回响池.大小,
            "质心范数": 回响池.计算质心().norm().item(),
            "有效温度": 回响池.计算有效温度(),
            "总步数": 回响池.当前步数,
            "衰减系数": 配置.decay_gamma,
            "注入强度": 配置.lambda_strength,
        }

        return {"文本": 生成文本, "评估器": 评估器, "池统计": 池统计}

    def 运行单次生成(
        self, 提示词: str, 配置: 实验配置, 是回响: bool = False
    ) -> Dict[str, Any]:
        """
        运行单次生成，返回结果字典。

        Parameters
        ----------
        提示词 : str
            输入提示词
        配置 : 实验配置
            实验配置参数
        是回响 : bool
            是否使用回响模式

        Returns
        -------
        Dict[str, Any]
            根据模式返回不同的结果字典：
            - 基线模式：{"文本": str, "评估器": 逐Token评估器}
            - 回响模式：{"文本": str, "评估器": 逐Token评估器, "池统计": dict}
        """
        if 是回响:
            return self._回响生成(提示词, 配置)
        return self._基线生成(提示词, 配置)

    # ──────────────────────────────────────────────
    # 运行实验
    # ──────────────────────────────────────────────

    def 运行实验(
        self,
        提示词: str,
        维度: str,
        配置: 实验配置,
        重复次数: Optional[int] = None,
    ) -> List[实验对比器]:
        """
        对同一提示词运行多次重复实验，每次同时执行基线和回响生成。

        Parameters
        ----------
        提示词 : str
            输入提示词
        维度 : str
            情感维度名称
        配置 : 实验配置
            实验配置
        重复次数 : Optional[int]
            重复运行次数，默认使用配置中的值

        Returns
        -------
        List[实验对比器]
            每次重复对应的对比器列表
        """
        if 重复次数 is None:
            重复次数 = 配置.重复次数

        对比器列表: List[实验对比器] = []

        for rep_idx in range(重复次数):
            try:
                # 基线生成
                基线结果 = self.运行单次生成(提示词, 配置, 是回响=False)

                # 回响生成（如果是回响配置）
                回响结果: Optional[Dict[str, Any]] = None
                if 配置.是回响模式:
                    回响结果 = self.运行单次生成(提示词, 配置, 是回响=True)

                # 创建对比器
                对比器 = 实验对比器(
                    提示词=提示词,
                    情感维度=维度,
                    配置=配置,
                    基线结果=基线结果,
                    回响结果=回响结果,
                    重复索引=rep_idx,
                )
                对比器列表.append(对比器)

            except Exception as e:
                warnings.warn(
                    f"运行失败: 配置={配置.实验编号}, "
                    f"维度={维度}, 重复#{rep_idx}, "
                    f"错误: {type(e).__name__}: {e}"
                )
                # 跳过失败项，继续处理下一个
                continue

        return 对比器列表

    def 运行全部实验(
        self, 配置列表: Optional[List[实验配置]] = None
    ) -> 汇总统计器:
        """
        运行所有实验配置 × 所有情感维度 × 所有提示词。

        Parameters
        ----------
        配置列表 : Optional[List[实验配置]]
            要运行的实验配置列表，默认使用 实验配置列表

        Returns
        -------
        汇总统计器
            包含所有实验结果的汇总统计器
        """
        if 配置列表 is None:
            配置列表 = 实验配置列表

        self.所有对比器.clear()

        total_runs = (
            len(配置列表)
            * sum(len(prompts) for prompts in 测试提示词.values())
        )

        with tqdm(total=total_runs, desc="运行全部实验", unit="配置×提示词") as pbar:
            for 配置 in 配置列表:
                for 维度, 提示词列表 in 测试提示词.items():
                    for 提示词 in 提示词列表:
                        对比器列表 = self.运行实验(提示词, 维度, 配置)
                        self.所有对比器.extend(对比器列表)
                        pbar.update(1)

        if not self.所有对比器:
            raise RuntimeError("所有实验均运行失败，没有可用结果")

        return 汇总统计器(self.所有对比器)

    # ──────────────────────────────────────────────
    # 可视化
    # ──────────────────────────────────────────────

    def 生成可视化(self, 汇总器: 汇总统计器) -> Dict[str, str]:
        """
        使用 matplotlib 生成四张对比图并保存。

        Parameters
        ----------
        汇总器 : 汇总统计器
            包含所有实验结果的汇总统计器

        Returns
        -------
        Dict[str, str]
            图标题到文件路径的映射
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
        except ImportError:
            warnings.warn("matplotlib 未安装，跳过可视化生成")
            return {}

        保存路径: Dict[str, str] = {}

        # ── 准备数据 ──
        实验分组 = 汇总器.按实验分组()

        # 提取各实验的熵值、KL散度、提升率
        实验标签: List[str] = []
        基线熵列表: List[List[float]] = []
        回响熵列表: List[List[float]] = []
        KL散度数据: Dict[str, List[float]] = {}  # 实验编号 -> [值列表]
        提升率数据: Dict[str, List[float]] = {}
        质心范数数据: Dict[str, Dict[str, List[float]]] = {}  # 实验 -> 维度 -> [质心范数]

        for 实验号, 对比器列表 in 实验分组.items():
            配置描述 = 对比器列表[0].配置.条件描述
            实验标签.append(f"{实验号}\n{配置描述}")

            # 记录基线熵和回响熵
            b_entropy: List[float] = []
            e_entropy: List[float] = []
            kl_vals: List[float] = []
            提升率_vals: List[float] = []

            for c in 对比器列表:
                if c.基线评估器 is not None:
                    b_entropy.append(c.基线评估器.平均熵)
                if c.回响评估器 is not None:
                    e_entropy.append(c.回响评估器.平均熵)
                if c.KL散度 is not None:
                    kl_vals.append(c.KL散度)
                if c.细腻度提升率 is not None:
                    提升率_vals.append(c.细腻度提升率)

            基线熵列表.append(b_entropy)
            回响熵列表.append(e_entropy)
            KL散度数据[实验号] = kl_vals
            提升率数据[实验号] = 提升率_vals

            # 收集质心范数（仅回响模式）
            for c in 对比器列表:
                if c.池统计 and c.池统计.get("质心范数") is not None:
                    质心范数数据.setdefault(实验号, {}).setdefault(
                        c.情感维度, []
                    ).append(c.池统计["质心范数"])

        # ── 图1: 语义熵分布对比（箱线图） ──
        try:
            fig1, ax1 = plt.subplots(figsize=(12, 6))

            # 为每个实验准备数据：基线熵和回响熵
            positions_b = []
            positions_e = []
            data_b = []
            data_e = []
            labels_tick = []

            for i, (b_ent, e_ent) in enumerate(zip(基线熵列表, 回响熵列表)):
                if b_ent:
                    data_b.append(b_ent)
                    positions_b.append(i * 3)
                if e_ent:
                    data_e.append(e_ent)
                    positions_e.append(i * 3 + 1)
                labels_tick.append(实验标签[i] if i < len(实验标签) else f"E{i+1}")

            if data_b:
                bp_b = ax1.boxplot(
                    data_b, positions=positions_b, widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor='#4ECDC4', alpha=0.7),
                    medianprops=dict(color='white', linewidth=2),
                )
            if data_e:
                bp_e = ax1.boxplot(
                    data_e, positions=positions_e, widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor='#FF6B6B', alpha=0.7),
                    medianprops=dict(color='white', linewidth=2),
                )

            # 图例和标签
            ax1.set_xticks([i * 3 + 0.5 for i in range(len(实验标签))])
            ax1.set_xticklabels(实验标签, fontsize=9)
            ax1.set_ylabel("语义熵", fontsize=12)
            ax1.set_title("语义熵分布对比（基线 vs 回响）", fontsize=14)
            ax1.legend(
                [bp_b["boxes"][0] if data_b else None,
                 bp_e["boxes"][0] if data_e else None],
                ["基线", "回响"],
                loc='upper right',
            )
            ax1.grid(axis='y', alpha=0.3)

            path1 = os.path.join(self.可视化目录, "图1_语义熵分布对比.png")
            fig1.tight_layout()
            fig1.savefig(path1, dpi=150, bbox_inches='tight')
            plt.close(fig1)
            保存路径["语义熵分布对比"] = path1
        except Exception as e:
            warnings.warn(f"图1 生成失败: {e}")

        # ── 图2: KL 散度柱状图（每个实验一条） ──
        try:
            if any(KL散度数据.values()):
                fig2, ax2 = plt.subplots(figsize=(12, 6))

                # 每个实验画一条柱（取均值）
                实验号列表 = list(KL散度数据.keys())
                kl_means = [sum(KL散度数据[exp]) / len(KL散度数据[exp])
                           for exp in 实验号列表]
                kl_stds = [
                    (sum((v - kl_means[i]) ** 2 for v in KL散度数据[exp])
                     / len(KL散度数据[exp])) ** 0.5
                    if len(KL散度数据[exp]) > 1 else 0.0
                    for i, exp in enumerate(实验号列表)
                ]

                colors_kl = ['#95E1D3' if 'Echo' in 实验分组[exp][0].配置.条件描述
                            else '#F38181' for exp in 实验号列表]

                bars = ax2.bar(
                    range(len(实验号列表)), kl_means, yerr=kl_stds,
                    color=colors_kl, edgecolor='white', linewidth=1.2,
                    capsize=5, alpha=0.85,
                )
                ax2.set_xticks(range(len(实验号列表)))
                ax2.set_xticklabels(
                    [实验分组[exp][0].配置.条件描述 for exp in 实验号列表],
                    rotation=15, fontsize=9,
                )
                ax2.set_ylabel("平均 KL 散度", fontsize=12)
                ax2.set_title("KL 散度（基线 vs 回响分布差异）", fontsize=14)
                ax2.grid(axis='y', alpha=0.3)

                # 在柱上标注数值
                for bar, val in zip(bars, kl_means):
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=8,
                    )

                path2 = os.path.join(self.可视化目录, "图2_KL散度柱状图.png")
                fig2.tight_layout()
                fig2.savefig(path2, dpi=150, bbox_inches='tight')
                plt.close(fig2)
                保存路径["KL散度柱状图"] = path2
        except Exception as e:
            warnings.warn(f"图2 生成失败: {e}")

        # ── 图3: 细腻度提升率的箱线图 ──
        try:
            if any(提升率数据.values()):
                fig3, ax3 = plt.subplots(figsize=(12, 6))

                实验号列表_3 = list(提升率数据.keys())
                提升率值列表 = [提升率数据[exp] for exp in 实验号列表_3]

                bp3 = ax3.boxplot(
                    提升率值列表, labels=[
                        实验分组[exp][0].配置.条件描述 for exp in 实验号列表_3
                    ],
                    patch_artist=True,
                    boxprops=dict(facecolor='#A8D8EA', alpha=0.7),
                    medianprops=dict(color='#FF6B6B', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red', alpha=0.5),
                )
                ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax3.set_ylabel("细腻度提升率 (%)", fontsize=12)
                ax3.set_title("细腻度提升率分布", fontsize=14)
                ax3.set_xticklabels(
                    [实验分组[exp][0].配置.条件描述 for exp in 实验号列表_3],
                    rotation=15, fontsize=9,
                )
                ax3.grid(axis='y', alpha=0.3)

                path3 = os.path.join(self.可视化目录, "图3_细腻度提升率.png")
                fig3.tight_layout()
                fig3.savefig(path3, dpi=150, bbox_inches='tight')
                plt.close(fig3)
                保存路径["细腻度提升率"] = path3
        except Exception as e:
            warnings.warn(f"图3 生成失败: {e}")

        # ── 图4: 质心范数随步数的变化曲线（仅回响模式） ──
        try:
            # 收集各回响实验每个维度的平均质心范数
            if 质心范数数据:
                fig4, ax4 = plt.subplots(figsize=(12, 6))

                实验号列表_4 = sorted(质心范数数据.keys())
                维度列表 = list(测试提示词.keys())
                colors_dim = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

                for i, exp_id in enumerate(实验号列表_4):
                    exp_data = 质心范数数据[exp_id]
                    for j, dim in enumerate(维度列表):
                        if dim in exp_data and exp_data[dim]:
                            avg_norm = sum(exp_data[dim]) / len(exp_data[dim])
                            ax4.bar(
                                i * (len(维度列表) + 1) + j,
                                avg_norm,
                                color=colors_dim[j % len(colors_dim)],
                                alpha=0.8,
                                label=dim if i == 0 else "",
                                width=0.6,
                            )

                ax4.set_xticks([
                    i * (len(维度列表) + 1) + (len(维度列表) - 1) / 2
                    for i in range(len(实验号列表_4))
                ])
                ax4.set_xticklabels([
                    实验分组[exp_id][0].配置.条件描述
                    for exp_id in 实验号列表_4
                ], rotation=15, fontsize=9)
                ax4.set_ylabel("平均质心范数", fontsize=12)
                ax4.set_title("回响模式：质心范数对比（按情感维度）", fontsize=14)
                ax4.legend(loc='upper right', fontsize=9)
                ax4.grid(axis='y', alpha=0.3)

                path4 = os.path.join(self.可视化目录, "图4_质心范数对比.png")
                fig4.tight_layout()
                fig4.savefig(path4, dpi=150, bbox_inches='tight')
                plt.close(fig4)
                保存路径["质心范数对比"] = path4
        except Exception as e:
            warnings.warn(f"图4 生成失败: {e}")

        return 保存路径

    # ──────────────────────────────────────────────
    # 保存/加载结果
    # ──────────────────────────────────────────────

    def 保存结果(self, 汇总器: 汇总统计器) -> str:
        """
        将汇总结果保存到输出目录。

        Parameters
        ----------
        汇总器 : 汇总统计器
            汇总统计器实例

        Returns
        -------
        str
            结果文件路径
        """
        文件路径 = os.path.join(self.输出目录, "实验结果.json")
        汇总器.导出JSON(文件路径)
        return 文件路径

    def 检查已有数据(self) -> Optional[汇总统计器]:
        """
        检查输出目录是否已有完整的实验数据。

        Returns
        -------
        Optional[汇总统计器]
            如果存在有效结果文件则返回汇总器，否则返回 None
        """
        结果路径 = os.path.join(self.输出目录, "实验结果.json")
        if not os.path.exists(结果路径):
            return None

        try:
            with open(结果路径, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data and "总实验数" in data and data["总实验数"] > 0:
                # 注意：这里只是检测文件存在性，不重建完整的汇总统计器
                warnings.warn(
                    f"发现已有实验结果: {结果路径} ({data['总实验数']} 条)"
                )
                return 汇总统计器(self.所有对比器) if self.所有对比器 else None
        except (json.JSONDecodeError, KeyError, IOError) as e:
            warnings.warn(f"读取已有结果失败: {e}")

        return None


# ══════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════


def main() -> None:
    """
    主入口函数。

    流程：
    1. 加载模型和分词器
    2. 创建实验运行器
    3. 检查是否有预先存在的实验数据
    4. 运行实验
    5. 生成可视化
    6. 保存汇总结果
    """
    import argparse

    parser = argparse.ArgumentParser(description="语义回响实验运行器")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace 模型名称（默认: Qwen/Qwen2.5-0.5B-Instruct）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./实验数据",
        help="实验数据输出目录（默认: ./实验数据）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新运行实验（忽略已有数据）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="运行设备（auto/cpu/cuda，默认: auto）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("语义回响实验运行器")
    print("=" * 60)

    # ── 步骤 1: 加载模型和分词器 ──
    print(f"\n[1/6] 加载模型: {args.model_name}")
    try:
        # 设备配置
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device

        print(f"  使用设备: {device}")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        print(f"  模型参数量: {model.num_parameters() / 1e6:.1f}M")
    except Exception as e:
        print(f"  [错误] 加载模型失败: {type(e).__name__}: {e}")
        print("  请确保已安装 transformers 并能访问 HuggingFace。")
        print("  可通过 --model_name 指定本地模型路径。")
        return

    # ── 步骤 2: 创建实验运行器 ──
    print(f"\n[2/6] 创建实验运行器")
    runner = 实验运行器(
        model=model,
        tokenizer=tokenizer,
        输出目录=args.output_dir,
    )
    print(f"  输出目录: {runner.输出目录}")

    # ── 步骤 3: 检查已有数据 ──
    print(f"\n[3/6] 检查已有实验数据")
    已有汇总器 = None if args.force else runner.检查已有数据()

    if 已有汇总器 is not None and not args.force:
        print("  使用已有实验数据，跳过运行阶段。")
        汇总器 = 已有汇总器
    else:
        if args.force:
            print("  --force 已指定，强制重新运行。")
        else:
            print("  未发现已有数据或数据不完整，将运行实验。")

        # ── 步骤 4: 运行实验 ──
        print(f"\n[4/6] 运行全部实验")

        # 选择要运行的配置
        print(f"  实验配置数: {len(实验配置列表)}")
        print(f"  情感维度数: {len(测试提示词)}")
        print(f"  总提示词数: {sum(len(v) for v in 测试提示词.values())}")

        try:
            汇总器 = runner.运行全部实验()
            print(f"  实验完成，共 {汇总器.总实验数} 次运行")
        except RuntimeError as e:
            print(f"  [错误] 实验运行失败: {e}")
            return
        except Exception as e:
            print(f"  [错误] 未预期异常: {type(e).__name__}: {e}")
            return

    # ── 步骤 5: 生成可视化 ──
    print(f"\n[5/6] 生成可视化")
    try:
        保存文件 = runner.生成可视化(汇总器)
        if 保存文件:
            print(f"  成功生成 {len(保存文件)} 张图表:")
            for 标题, 路径 in 保存文件.items():
                print(f"    - {标题}: {路径}")
        else:
            print("  可视化生成被跳过（matplotlib 未安装）")
    except Exception as e:
        print(f"  [警告] 可视化生成失败: {type(e).__name__}: {e}")

    # ── 步骤 6: 保存汇总结果 ──
    print(f"\n[6/6] 保存汇总结果")
    try:
        结果路径 = runner.保存结果(汇总器)
        print(f"  结果已保存至: {结果路径}")

        # 打印摘要
        汇总数据 = 汇总器.汇总()
        print(f"\n  === 实验摘要 ===")
        print(f"  总实验数: {汇总数据['总实验数']}")
        print(f"  平均 KL 散度: {汇总数据['平均KL散度']:.4f}")
        print(f"  平均细腻度提升率: {汇总数据['平均细腻度提升率']:.2f}%")
        print(f"  时间戳: {汇总数据['时间戳']}")

        for exp_id, exp_info in 汇总数据.get("实验汇总", {}).items():
            print(f"    {exp_id} ({exp_info['条件描述']}):")
            kl = exp_info.get("平均KL散度")
            if kl is not None:
                print(f"      KL散度={kl:.4f}")
            提升 = exp_info.get("平均细腻度提升率")
            if 提升 is not None:
                print(f"      提升率={提升:.2f}%")
    except Exception as e:
        print(f"  [错误] 保存结果失败: {type(e).__name__}: {e}")
        return

    print(f"\n{'=' * 60}")
    print("实验完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
