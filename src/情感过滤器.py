# 翻译毒药
#
# 情感过滤器
# 基于 cnsenti 情感分析库，从候选 token 中筛选情感词并计算情感相关性权重。
#
# 本文件是"翻译毒药"项目的一部分，用于在生成过程中注入情感约束，
# 使输出文本在保留语义的同时具备期望的情感色彩。

from typing import Optional

import torch
from cnsenti import Sentiment


class 情感过滤器:
    """
    情感过滤器

    集成 cnsenti 情感分析库，从候选 token 中筛选出情感相关的 token，
    并计算其情感相关性权重 alpha_i。

    Usage
    -----
    >>> filter = 情感过滤器()
    >>> filter.加载词库()
    >>> 命中列表 = filter.筛选(["开心", "的", "悲伤", "于是"], tokenizer=None)
    >>> # 返回: [("开心", 0.999), ("悲伤", 0.999)]
    """

    def __init__(self) -> None:
        # 情感分析器实例，在调用加载词库后初始化
        self._分析器: Optional[Sentiment] = None
        # 会话统计
        self._总检查数: int = 0
        self._命中数: int = 0

    def 加载词库(self) -> None:
        """
        初始化情感分析器。

        使用 cnsenti 的 Sentiment 类创建情感分析器实例。
        Sentiment 内部使用 jieba 分词和内置情感词典进行情感分析。

        Raises
        ------
        ImportError
            当 cnsenti 库未正确安装时抛出
        RuntimeError
            当情感分析器初始化失败时抛出
        """
        try:
            self._分析器 = Sentiment()
        except ImportError as e:
            raise ImportError(
                "cnsenti 库未正确安装，请运行: pip install cnsenti"
            ) from e
        except Exception as e:
            raise RuntimeError(f"情感分析器初始化失败: {e}") from e

    def 筛选(
        self,
        候选token列表: list[str],
        tokenizer,  # type: ignore[type-arg]
    ) -> list[tuple[str, float]]:
        """
        从候选 token 列表中筛选出情感词。

        使用 cnsenti 对每个 token 进行情感分析，若检测到积极或消极情感
        (pos + neg > 0)，则将其视为情感词并计算情感强度得分。

        Parameters
        ----------
        候选token列表 : list[str]
            解码后的候选 token 文本列表。
        tokenizer : AutoTokenizer or None
            用于将文本转回 token ID 的 tokenizer（当前暂未使用，
            保留以兼容未来扩展）。

        Returns
        -------
        list[tuple[str, float]]
            命中情感词的 (token文本, 情感强度得分) 列表。
            得分范围 [0, 1]，0 = 无情感，1 = 强情感。

        Raises
        ------
        RuntimeError
            当情感分析器未初始化时抛出（需先调用加载词库方法）。
        TypeError
            当候选token列表类型不正确时抛出。

        Notes
        -----
        情感强度计算公式: (pos + neg) / (pos + neg + 0.001)
        分母中的 0.001 为平滑项，用于避免零除。
        """
        if not isinstance(候选token列表, list):
            raise TypeError(
                f"候选token列表必须是 list 类型，收到: {type(候选token列表).__name__}"
            )
        if self._分析器 is None:
            raise RuntimeError("情感分析器未初始化，请先调用 加载词库() 方法")

        命中结果: list[tuple[str, float]] = []

        for token in 候选token列表:
            if not isinstance(token, str):
                continue

            self._总检查数 += 1

            try:
                # 使用 cnsenti 分析单个 token 的情感
                情感计数 = self._分析器.sentiment_count(token)
            except Exception as e:
                # 单个 token 分析失败时跳过，不中断整体流程
                continue

            # 从结果中提取积极和消极情感计数
            try:
                积极数 = int(情感计数.get("pos", 0))
                消极数 = int(情感计数.get("neg", 0))
            except (ValueError, TypeError, AttributeError):
                continue

            情感总和 = 积极数 + 消极数

            if 情感总和 > 0:
                self._命中数 += 1
                # 情感强度: (pos + neg) / (pos + neg + 0.001)
                情感强度得分 = 情感总和 / (情感总和 + 0.001)
                命中结果.append((token, 情感强度得分))

        return 命中结果

    def 计算权重(
        self,
        token文本: str,
        上下文隐藏状态: torch.Tensor,
    ) -> float:
        """
        计算给定 token 的情感相关性权重 alpha_i。

        使用 cnsenti 的情感得分作为权重基础。当前实现直接返回
        情感强度值作为 alpha_i，忽略上下文隐藏状态参数（保留以
        支持未来基于上下文的动态权重扩展）。

        Parameters
        ----------
        token文本 : str
            待计算权重的 token 文本。
        上下文隐藏状态 : torch.Tensor
            模型在某层的隐藏状态张量，形状通常为
            (batch_size, seq_len, hidden_dim) 或
            (hidden_dim,)。当前版本暂未使用此参数。

        Returns
        -------
        float
            情感相关性权重，范围 [0, 1]。
            0 = 无情感相关性，1 = 强情感相关性。

        Raises
        ------
        RuntimeError
            当情感分析器未初始化时抛出。
        ValueError
            当 token 文本为空时抛出。
        """
        if not token文本:
            raise ValueError("token 文本不能为空")
        if self._分析器 is None:
            raise RuntimeError("情感分析器未初始化，请先调用 加载词库() 方法")

        # 使用 cnsenti 分析情感
        try:
            情感计数 = self._分析器.sentiment_count(token文本)
        except Exception as e:
            return 0.0

        try:
            积极数 = int(情感计数.get("pos", 0))
            消极数 = int(情感计数.get("neg", 0))
        except (ValueError, TypeError, AttributeError):
            return 0.0

        情感总和 = 积极数 + 消极数

        # 直接使用情感强度值作为权重 alpha_i
        if 情感总和 > 0:
            return 情感总和 / (情感总和 + 0.001)

        return 0.0

    def 获取情感统计(self) -> dict[str, float]:
        """
        获取当前会话的情感统计信息。

        Returns
        -------
        dict[str, float]
            包含以下键的字典:
            - "总检查数": 总共检查的 token 数量
            - "命中数": 被识别为情感词的 token 数量
            - "命中率": 命中数 / 总检查数（若无检查则返回 0.0）
        """
        总检查数 = self._总检查数
        命中数 = self._命中数

        if 总检查数 > 0:
            命中率 = 命中数 / 总检查数
        else:
            命中率 = 0.0

        return {
            "总检查数": float(总检查数),
            "命中数": float(命中数),
            "命中率": 命中率,
        }
