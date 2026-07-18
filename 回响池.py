"""
语义回响池（Semantic Echo Pool）

存储生成过程中每一步的 hidden_state 及其不确定性权重，
按指数衰减策略维护，在后续采样时作为"情感场"偏置来源。

论文对应：第 3.2–3.3、3.5 节
"""

import torch
import math
from typing import Optional


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

    Attributes
    ----------
    向量列表 : list[torch.Tensor]
        存储的 hidden_state 向量
    权重列表 : list[float]
        每个向量对应的不确定性权重
    时间戳列表 : list[int]
        每个向量加入时的步数
    """

    def __init__(
        self,
        hidden_dim: int,
        max_pool_size: int = 1024,
        decay_gamma: float = 0.1,
        eviction_threshold: float = 1e-4,
    ) -> None:
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim 必须为正整数，收到 {hidden_dim}")
        if max_pool_size <= 0:
            raise ValueError(f"max_pool_size 必须为正整数，收到 {max_pool_size}")
        if decay_gamma < 0:
            raise ValueError(f"decay_gamma 不能为负，收到 {decay_gamma}")

        self.hidden_dim = hidden_dim
        self.max_pool_size = max_pool_size
        self.decay_gamma = decay_gamma
        self.eviction_threshold = eviction_threshold

        # 动态存储
        self.向量列表: list[torch.Tensor] = []
        self.权重列表: list[float] = []
        self.时间戳列表: list[int] = []

        # 状态跟踪
        self.当前步数: int = 0
        self._质心缓存: Optional[torch.Tensor] = None

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

    # ──────────────────────────────────────────────
    # 写操作
    # ──────────────────────────────────────────────

    def 添加(self, 向量: torch.Tensor, 权重: float) -> None:
        """
        向池中添加一个 hidden_state 向量及其权重。

        Parameters
        ----------
        向量 : torch.Tensor
            shape=(hidden_dim,)，从模型钩子捕获的 hidden_state
        权重 : float
            该步的不确定性权重，推荐 1 - max(softmax(logits))
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

        # 确保在 CPU 上且为 float32
        v = 向量.detach().cpu().float()

        self.向量列表.append(v)
        self.权重列表.append(权重)
        self.时间戳列表.append(self.当前步数)

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
        self.当前步数 = 0
        self._质心缓存 = None

    # ──────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────

    def _淘汰最旧(self) -> None:
        """淘汰时间戳最早的项（超出 max_pool_size 时触发）"""
        idx = min(range(len(self.时间戳列表)), key=lambda i: self.时间戳列表[i])
        self.向量列表.pop(idx)
        self.权重列表.pop(idx)
        self.时间戳列表.pop(idx)

    def _应用衰减(self) -> None:
        """
        对所有项应用指数衰减：alpha(t) = weight * exp(-gamma * (t - t_i))

        衰减后权重低于 eviction_threshold 的项被移除。
        """
        t = self.当前步数
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
            self._质心缓存 = None

    def __repr__(self) -> str:
        return (
            f"语义回响池(hidden_dim={self.hidden_dim}, "
            f"大小={self.大小}, 步数={self.当前步数}, "
            f"gamma={self.decay_gamma})"
        )
