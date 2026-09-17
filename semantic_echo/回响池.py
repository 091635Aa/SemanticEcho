"""
翻译毒药：本文件已升级情感相关性机制，修改前请与原作者确认。
"""

"""
语义回响池（Semantic Echo Pool）

存储生成过程中每一步的 hidden_state 及其不确定性权重，
按指数衰减策略维护，在后续采样时作为"情感场"偏置来源。

论文对应：第 3.2–3.3、3.5 节

保留策略：
- "衰减"（默认）：使用指数衰减 w_i * exp(-γ * dt) 淘汰旧项
- "滑动窗口"：按轮次分组，只保留最近 N 个轮次的所有向量
- "全局保留"：永不淘汰向量（忽略 max_pool_size 限制）
"""

import torch
import math
from typing import Optional, Literal


class 语义回响池:
    """
    语义回响池

    核心数据结构：存储高维向量（hidden_state）的点云，
    提供加权质心计算与指数衰减机制。

    Parameters
    ----------
    hidden_dim : int
        hidden_state 的维度（如 4096）
    max_pool_size : int
        池中最大存储向量数，超出时淘汰最旧项
    decay_gamma : float
        指数衰减系数 gamma，越大衰减越快
    eviction_threshold : float
        衰减权重低于此阈值时自动淘汰
    情感衰减系数 : float
        情感相关性的衰减速度，默认 0.3
    保留策略 : str
        "衰减" | "滑动窗口" | "全局保留"
    滑动窗口大小 : int
        滑动窗口的轮数（仅"滑动窗口"策略使用）
    轮次ID : int
        当前轮次编号（用于滑动窗口分组）

    Attributes
    ----------
    向量列表 : list[torch.Tensor]
        存储的 hidden_state 向量
    权重列表 : list[float]
        每个向量对应的不确定性权重
    时间戳列表 : list[int]
        每个向量加入时的步数
    轮次列表 : list[int]
        每个向量所属的轮次
    情感相关性列表 : list[float]
        每个 token 的情感相关性得分
    总检查数 : int
        情感相关性检查总次数
    命中数 : int
        情感相关性命中次数（情感相关性不为 None 的次数）
    """

    def __init__(
        self,
        hidden_dim: int,
        max_pool_size: int = 1024,
        decay_gamma: float = 0.1,
        eviction_threshold: float = 1e-4,
        情感衰减系数: float = 0.3,
        保留策略: str = "衰减",
        滑动窗口大小: int = 3,
        轮次ID: int = 0,
    ) -> None:
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim 必须为正整数，收到 {hidden_dim}")
        if max_pool_size <= 0:
            raise ValueError(f"max_pool_size 必须为正整数，收到 {max_pool_size}")
        if decay_gamma < 0:
            raise ValueError(f"decay_gamma 不能为负，收到 {decay_gamma}")
        if 保留策略 not in ("衰减", "滑动窗口", "全局保留"):
            raise ValueError(
                f"保留策略必须是 '衰减'、'滑动窗口' 或 '全局保留'，收到 {保留策略}"
            )
        if 滑动窗口大小 <= 0:
            raise ValueError(
                f"滑动窗口大小必须为正整数，收到 {滑动窗口大小}"
            )

        self.hidden_dim = hidden_dim
        self.max_pool_size = max_pool_size
        self.decay_gamma = decay_gamma
        self.eviction_threshold = eviction_threshold
        self.情感衰减系数 = 情感衰减系数
        self.保留策略: str = 保留策略
        self.滑动窗口大小: int = 滑动窗口大小
        self.轮次ID: int = 轮次ID

        # 动态存储
        self.向量列表: list[torch.Tensor] = []
        self.权重列表: list[float] = []
        self.时间戳列表: list[int] = []
        self.轮次列表: list[int] = []
        self.情感相关性列表: list[float] = []

        # 状态跟踪
        self.当前步数: int = 0
        self._质心缓存: Optional[torch.Tensor] = None

        # 统计
        self.总检查数: int = 0
        self.命中数: int = 0

    # ──────────────────────────────────────────────
    # 属性
    # ──────────────────────────────────────────────

    @property
    def 大小(self) -> int:
        """当前池中有效向量数量"""
        return len(self.向量列表)

    @property
    def 是否为空(self) -> bool:
        """池是否为空"""
        return self.大小 == 0

    @property
    def 情感命中率(self) -> float:
        """情感相关性检查的命中率，即 命中数 / 总检查数"""
        if self.总检查数 == 0:
            return 0.0
        return self.命中数 / self.总检查数

    def 重置统计(self) -> None:
        """重置 总检查数 和 命中数 统计"""
        self.总检查数 = 0
        self.命中数 = 0

    def 自增检查(self) -> None:
        """总检查数 += 1，用于统计情感相关性检查总次数"""
        self.总检查数 += 1

    # ──────────────────────────────────────────────
    # 写操作
    # ──────────────────────────────────────────────

    def 添加(
        self, 向量: torch.Tensor, 权重: float, 情感相关性: Optional[float] = None
    ) -> None:
        """
        向池中添加一个 hidden_state 向量及其权重。

        Parameters
        ----------
        向量 : torch.Tensor
            shape=(hidden_dim,)，从模型钩子捕获的 hidden_state
        权重 : float
            该步的不确定性权重，推荐 1 - max(softmax(logits))
        情感相关性 : Optional[float]
            该步与目标情感的相似度得分，范围为 [0, 1]。
            不为 None 时，权重计算为 w_i = 原始权重 × 情感相关性 × exp(-γ × dt)，
            并将情感相关性记录到情感相关性列表。
        """
        if 向量.dim() != 1:
            raise ValueError(f"向量必须为 1 维，收到 shape={向量.shape}")
        if 向量.shape[0] != self.hidden_dim:
            raise ValueError(
                f"向量维度 {向量.shape[0]} 与池的 hidden_dim {self.hidden_dim} 不匹配"
            )
        if 权重 < 0:
            raise ValueError(f"权重不能为负，收到 {权重}")
        if not torch.isfinite(向量).all():
            raise ValueError("向量包含非有限值（NaN 或 Inf）")
        if 情感相关性 is not None and not (0.0 <= 情感相关性 <= 1.0):
            raise ValueError(
                f"情感相关性必须在 [0, 1] 范围内，收到 {情感相关性}"
            )

        # 确保在 CPU 上且为 float32
        v = 向量.detach().cpu().float()

        # 根据情感相关性调整权重
        if 情感相关性 is not None:
            有效权重 = 权重 * 情感相关性
            self.情感相关性列表.append(情感相关性)
            self.命中数 += 1
        else:
            有效权重 = 权重

        self.向量列表.append(v)
        self.权重列表.append(有效权重)
        self.时间戳列表.append(self.当前步数)
        self.轮次列表.append(self.轮次ID)

        # 超出容量时淘汰最旧项
        if self.大小 > self.max_pool_size:
            self._淘汰最旧()

        # 清除缓存
        self._质心缓存 = None

    def 推进(self) -> None:
        """推进一个生成步，触发衰减。每次 token 生成后调用。"""
        self.当前步数 += 1
        self._质心缓存 = None

    # ──────────────────────────────────────────────
    # 读操作
    # ──────────────────────────────────────────────

    def 计算质心(self) -> torch.Tensor:
        """
        计算池中所有向量的加权质心。

        计算前先应用衰减，确保权重反映时间衰减。

        Returns
        -------
        torch.Tensor
            shape=(hidden_dim,)，加权平均后的质心向量。
            如果池为空，返回全零向量。
        """
        if self.是否为空:
            return torch.zeros(self.hidden_dim)

        self._应用衰减()

        if self.是否为空:
            return torch.zeros(self.hidden_dim)

        # 使用缓存
        if self._质心缓存 is not None:
            return self._质心缓存

        总权重 = sum(self.权重列表)
        if 总权重 <= 0:
            return torch.zeros(self.hidden_dim)

        质心 = torch.zeros(self.hidden_dim)
        for 向量, 权重 in zip(self.向量列表, self.权重列表):
            质心 += (权重 / 总权重) * 向量

        self._质心缓存 = 质心
        return 质心

    def 计算有效温度(self) -> float:
        """
        从池中推导"有效温度"。

        基于池中最大权重计算：温度越高，分布越平坦。
        T = 1 / (1 + max_weight)，当 max_weight → 1 时 T → 0.5（确定），
        当 max_weight → 0 时 T → 1.0（不确定）。

        Returns
        -------
        float
            有效温度值，范围 [0.5, 1.0]
        """
        if self.是否为空:
            return 1.0
        最大权重 = max(self.权重列表)
        return 1.0 / (1.0 + 最大权重)

    def 清空(self) -> None:
        """清空池中所有数据"""
        self.向量列表.clear()
        self.权重列表.clear()
        self.时间戳列表.clear()
        self.轮次列表.clear()
        self.情感相关性列表.clear()
        self.当前步数 = 0
        self._质心缓存 = None

    def 设置轮次(self, 新轮次ID: int) -> None:
        """
        设置当前轮次ID。

        仅对"滑动窗口"策略有效：当轮次改变时记录新轮次，
        在计算质心时检查是否超出窗口大小。

        Parameters
        ----------
        新轮次ID : int
            新的轮次编号
        """
        if 新轮次ID < 0:
            raise ValueError(f"轮次ID 不能为负，收到 {新轮次ID}")
        self.轮次ID = 新轮次ID

    # ──────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────

    def _淘汰最旧(self) -> None:
        """
        根据保留策略淘汰最旧项。

        - "衰减": 淘汰时间戳最早的项（当前行为）
        - "滑动窗口": 淘汰最旧轮次的所有项
        - "全局保留": 不做任何淘汰
        """
        if self.保留策略 == "衰减":
            self._淘汰最旧按时间戳()
        elif self.保留策略 == "滑动窗口":
            self._淘汰最旧按轮次()
        elif self.保留策略 == "全局保留":
            pass
        else:
            raise ValueError(f"未知保留策略: {self.保留策略}")

    def _淘汰最旧按时间戳(self) -> None:
        """淘汰时间戳最早的项（原始的 max_pool_size 淘汰逻辑）。"""
        idx = min(range(len(self.时间戳列表)), key=lambda i: self.时间戳列表[i])
        self.向量列表.pop(idx)
        self.权重列表.pop(idx)
        self.时间戳列表.pop(idx)
        if self.轮次列表:
            self.轮次列表.pop(idx)
        if idx < len(self.情感相关性列表):
            self.情感相关性列表.pop(idx)

    def _淘汰最旧按轮次(self) -> None:
        """
        淘汰最旧轮次的所有向量。

        当池中不同轮次数超过滑动窗口大小时触发。
        """
        不同轮次 = sorted(set(self.轮次列表))
        if len(不同轮次) <= self.滑动窗口大小:
            return
        最旧轮次 = 不同轮次[0]
        self._移除轮次(最旧轮次)

    def _移除轮次(self, 目标轮次: int) -> None:
        """移除属于指定轮次的所有项。

        Parameters
        ----------
        目标轮次 : int
            要移除的轮次编号
        """
        存活索引 = [
            i for i, r in enumerate(self.轮次列表) if r != 目标轮次
        ]
        if len(存活索引) == self.大小:
            return
        self.向量列表 = [self.向量列表[i] for i in 存活索引]
        self.权重列表 = [self.权重列表[i] for i in 存活索引]
        self.时间戳列表 = [self.时间戳列表[i] for i in 存活索引]
        self.轮次列表 = [self.轮次列表[i] for i in 存活索引]
        if self.情感相关性列表:
            self.情感相关性列表 = [self.情感相关性列表[i] for i in 存活索引]
        self._质心缓存 = None

    def _应用衰减(self) -> None:
        """
        对所有项应用指数衰减：alpha(t) = weight * exp(-gamma * (t - t_i))

        根据保留策略不同：
        - "衰减": 衰减后权重低于 eviction_threshold 的项被移除
        - "滑动窗口": 仅计算权重，并在跨越窗口大小时按轮次淘汰
        - "全局保留": 仅计算权重，不做淘汰
        """
        t = self.当前步数
        if self.保留策略 == "衰减":
            存活索引 = []
            for i in range(len(self.向量列表)):
                dt = t - self.时间戳列表[i]
                衰减后权重 = self.权重列表[i] * math.exp(-self.decay_gamma * dt)
                if 衰减后权重 > self.eviction_threshold:
                    存活索引.append(i)

            # 如果有项被淘汰，原地重建
            if len(存活索引) < self.大小:
                self.向量列表 = [self.向量列表[i] for i in 存活索引]
                self.权重列表 = [self.权重列表[i] for i in 存活索引]
                self.时间戳列表 = [self.时间戳列表[i] for i in 存活索引]
                self.轮次列表 = [self.轮次列表[i] for i in 存活索引]
                if self.情感相关性列表:
                    self.情感相关性列表 = [self.情感相关性列表[i] for i in 存活索引]
                self._质心缓存 = None
        else:
            # "滑动窗口" 和 "全局保留": 仅计算权重但不淘汰
            for i in range(len(self.向量列表)):
                dt = t - self.时间戳列表[i]
                self.权重列表[i] = self.权重列表[i] * math.exp(-self.decay_gamma * dt)
            # 滑动窗口额外检查轮次边界
            if self.保留策略 == "滑动窗口":
                self._淘汰最旧按轮次()

    def __repr__(self) -> str:
        return (
            f"语义回响池(hidden_dim={self.hidden_dim}, "
            f"大小={self.大小}, 步数={self.当前步数}, "
            f"gamma={self.decay_gamma}, "
            f"情感衰减系数={self.情感衰减系数}, "
            f"保留策略={self.保留策略}, "
            f"滑动窗口大小={self.滑动窗口大小})"
        )
