"""
回响评估器 — 语义回响实验的指标评估模块

提供语义熵、KL 散度等分布度量函数，以及逐 token 评估、
实验对比和汇总统计等高层评估工具。

论文对应：第 4 节（实验与分析）
"""

import json
import torch
import torch.nn.functional as F
import math
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════
# 常数
# ══════════════════════════════════════════════════

_EPS: float = 1e-12
"""KL 散度计算中用于数值稳定的极小量"""


# ══════════════════════════════════════════════════
# 1. 语义熵
# ══════════════════════════════════════════════════

def 计算语义熵(logits: torch.Tensor) -> float:
    """
    计算单 token 位置的语义熵。

    定义：H = -Σ P(w_i) ln P(w_i)，其中 P = softmax(logits)。

    Parameters
    ----------
    logits : torch.Tensor
        shape 可为 (vocab_size,) 或 (1, vocab_size)。
        模型在某 token 位置输出的原始 logits。

    Returns
    -------
    float
        语义熵值（自然对数底），标量。

    Raises
    ------
    TypeError
        如果输入不是 torch.Tensor。
    ValueError
        如果 logits 维度超出 2 维，或最后一个维度为空。

    Examples
    --------
    >>> logits = torch.tensor([0.0, 1.0, 0.0])
    >>> ent = 计算语义熵(logits)
    >>> isinstance(ent, float)
    True
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits 必须是 torch.Tensor，收到 {type(logits)}")

    logits_dim = logits.dim()
    if logits_dim not in (1, 2):
        raise ValueError(
            f"logits 维度必须为 1 或 2，收到 {logits_dim} 维"
        )

    # 展平为 1 维向量方便处理
    if logits_dim == 2:
        if logits.shape[0] != 1:
            raise ValueError(
                f"2 维 logits 的 batch 维度必须为 1，"
                f"收到 shape={tuple(logits.shape)}"
            )
        logits = logits.squeeze(0)

    if logits.shape[0] == 0:
        raise ValueError("logits 的 vocab_size 不能为 0")

    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + _EPS)
    entropy = -(probs * log_probs).sum().item()

    return entropy


# ══════════════════════════════════════════════════
# 2. KL 散度
# ══════════════════════════════════════════════════

def 计算KL散度(基线logits: torch.Tensor, 回响logits: torch.Tensor) -> float:
    """
    计算两个 logits 分布之间的 KL 散度 D_KL(P || Q)。

    P = softmax(基线logits)，Q = softmax(回响logits)。
    定义：D_KL(P || Q) = Σ P(i) * ln(P(i) / Q(i))。

    Parameters
    ----------
    基线logits : torch.Tensor
        基线（无回响注入）条件下的 logits。
        shape 可为 (vocab_size,) 或 (1, vocab_size)。
    回响logits : torch.Tensor
        回响注入后的 logits。
        shape 必须与基线 logits 相同。

    Returns
    -------
    float
        KL 散度值，非负标量。

    Raises
    ------
    TypeError
        如果任一输入不是 torch.Tensor。
    ValueError
        如果两个张量 shape 不一致，或维度不合法。

    Examples
    --------
    >>> baseline = torch.tensor([0.0, 2.0, 0.0])
    >>> echoed = torch.tensor([0.5, 1.0, 0.5])
    >>> kld = 计算KL散度(baseline, echoed)
    >>> kld >= 0.0
    True
    """
    if not isinstance(基线logits, torch.Tensor):
        raise TypeError(
            f"基线logits 必须是 torch.Tensor，收到 {type(基线logits)}"
        )
    if not isinstance(回响logits, torch.Tensor):
        raise TypeError(
            f"回响logits 必须是 torch.Tensor，收到 {type(回响logits)}"
        )

    if 基线logits.shape != 回响logits.shape:
        raise ValueError(
            f"两个 logits shape 必须一致，"
            f"收到 {tuple(基线logits.shape)} vs {tuple(回响logits.shape)}"
        )

    logits_dim = 基线logits.dim()
    if logits_dim not in (1, 2):
        raise ValueError(
            f"logits 维度必须为 1 或 2，收到 {logits_dim} 维"
        )

    # 统一处理为 1 维
    基线_p = 基线logits
    回响_p = 回响logits
    if logits_dim == 2:
        基线_p = 基线logits.squeeze(0)
        回响_p = 回响logits.squeeze(0)

    if 基线_p.shape[0] == 0:
        raise ValueError("logits 的 vocab_size 不能为 0")

    p = F.softmax(基线_p, dim=-1)
    q = F.softmax(回响_p, dim=-1)

    # clamp 确保分母不为零
    q = q.clamp(min=_EPS)
    log_ratio = torch.log(p / q + _EPS)

    kl_div = (p * log_ratio).sum().item()
    # 由于数值误差，KL 散度可能为极小的负值，钳制到 0
    return max(0.0, kl_div)


# ══════════════════════════════════════════════════
# 3. 逐 Token 评估器
# ══════════════════════════════════════════════════

class 逐Token评估器:
    """
    逐 Token 评估器 — 记录生成过程中每一步的分布信息。

    在生成循环中，每步调用 `记录步` 传入 token_id、logits 和熵值，
    评估器自动保存完整轨迹，用于后续的统计计算与导出。

    Attributes
    ----------
    token_id_list : List[int]
        每一步生成的 token ID 列表。
    logits_list : List[List[float]]
        每一步的 logits 分布（已转为 Python list 保存）。
    熵列表 : List[float]
        每一步的语义熵值。
    生成token列表 : List[str]
        每一步生成 token 的文本表示（可选）。
    """

    def __init__(self) -> None:
        self.token_id_list: List[int] = []
        self.logits_list: List[List[float]] = []
        self.熵列表: List[float] = []
        self.生成token列表: List[str] = []

    def 记录步(
        self,
        token_id: int,
        logits: torch.Tensor,
        熵: float,
        生成token: str = "",
    ) -> None:
        """
        记录一步生成的分布信息。

        Parameters
        ----------
        token_id : int
            当前步采样的 token ID。
        logits : torch.Tensor
            shape=(1, vocab_size) 或 (vocab_size,) 的 logits 分布。
        熵 : float
            计算出的语义熵值。
        生成token : str, optional
            当前 token 的文本表示（用于可读性），默认为空。

        Raises
        ------
        TypeError
            如果 logits 不是 torch.Tensor。
        ValueError
            如果 token_id 为负，或熵值为无穷/NaN。
        """
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"logits 必须是 torch.Tensor，收到 {type(logits)}"
            )
        if token_id < 0:
            raise ValueError(f"token_id 不能为负，收到 {token_id}")
        if math.isnan(熵) or math.isinf(熵):
            raise ValueError(
                f"熵值不能为 NaN 或 Inf，收到 {熵}"
            )

        self.token_id_list.append(token_id)
        self.logits_list.append(logits.detach().cpu().tolist())
        self.熵列表.append(熵)
        self.生成token列表.append(生成token)

    def 计算平均熵(self) -> float:
        """
        计算所有记录步的语义熵均值。

        Returns
        -------
        float
            平均熵值。如果未记录任何步，返回 0.0。
        """
        if not self.熵列表:
            return 0.0
        return sum(self.熵列表) / len(self.熵列表)

    def 计算熵列表(self) -> List[float]:
        """
        获取所有步的熵值列表。

        Returns
        -------
        List[float]
            按时间顺序排列的熵值列表。
        """
        return self.熵列表.copy()

    def 导出JSON(self) -> dict:
        """
        将当前评估器的所有记录导出为可序列化字典。

        Returns
        -------
        dict
            包含以下字段：
            - "token_ids": List[int]
            - "logits": List[List[float]]
            - "entropies": List[float]
            - "tokens": List[str]
            - "平均熵": float
        """
        return {
            "token_ids": self.token_id_list,
            "logits": self.logits_list,
            "entropies": self.熵列表,
            "tokens": self.生成token列表,
            "平均熵": self.计算平均熵(),
        }


# ══════════════════════════════════════════════════
# 4. 实验对比器
# ══════════════════════════════════════════════════

class 实验对比器:
    """
    实验对比器 — 管理基线和回响版本的对比评估。

    一次"对比"包含同一提示词在基线和回响注入两种条件下
    的逐 token 评估记录，以及回响池的最终状态统计。

    Attributes
    ----------
    提示词 : str
        本次实验的目标提示词。
    情感维度 : str
        对应的情感维度（如"开心"、"悲伤"等）。
    基线评估器 : Optional[逐Token评估器]
        基线（无回响）条件下的评估记录。
    回响评估器 : Optional[逐Token评估器]
        回响注入条件下的评估记录。
    回响池统计 : dict
        回响池在实验结束时的快照统计。
    """

    def __init__(self, 提示词: str, 情感维度: str) -> None:
        """
        Parameters
        ----------
        提示词 : str
            实验使用的提示词。
        情感维度 : str
            实验针对的情感维度标签。
        """
        self.提示词 = 提示词
        self.情感维度 = 情感维度
        self.基线评估器: Optional[逐Token评估器] = None
        self.回响评估器: Optional[逐Token评估器] = None
        self.回响池统计: dict = {}

    def 设置基线(self, 评估器: 逐Token评估器) -> None:
        """
        设置基线条件下的评估器。

        Parameters
        ----------
        评估器 : 逐Token评估器
            基线生成结果对应的评估器实例。

        Raises
        ------
        TypeError
            如果输入不是逐Token评估器实例。
        """
        if not isinstance(评估器, 逐Token评估器):
            raise TypeError(
                f"评估器必须是逐Token评估器实例，收到 {type(评估器)}"
            )
        self.基线评估器 = 评估器

    def 设置回响(
        self,
        评估器: 逐Token评估器,
        池统计: dict,
    ) -> None:
        """
        设置回响注入条件下的评估器和池统计。

        Parameters
        ----------
        评估器 : 逐Token评估器
            回响注入生成结果对应的评估器实例。
        池统计 : dict
            回响池的最终统计信息，应包含如下键：
            - "最终大小": int
            - "有效温度": float
            - "质心范数": float

        Raises
        ------
        TypeError
            如果评估器不是逐Token评估器实例。
        ValueError
            如果池统计字典中缺少必要字段。
        """
        if not isinstance(评估器, 逐Token评估器):
            raise TypeError(
                f"评估器必须是逐Token评估器实例，收到 {type(评估器)}"
            )

        必需字段 = {"最终大小", "有效温度", "质心范数"}
        缺失字段 = 必需字段 - set(池统计.keys())
        if 缺失字段:
            raise ValueError(
                f"池统计缺少必需字段：{缺失字段}"
            )

        self.回响评估器 = 评估器
        self.回响池统计 = 池统计

    def 计算整体统计(self) -> dict:
        """
        计算本次实验对比的整体统计指标。

        Returns
        -------
        dict
            包含基线/回响对比的完整统计字典，结构如下：

            .. code-block:: python

                {
                    "提示词": "...",
                    "情感维度": "开心",
                    "基线文本": "...",
                    "回响文本": "...",
                    "基线平均熵": 0.0,
                    "回响平均熵": 0.0,
                    "细腻度提升率(%)": 0.0,
                    "KL散度": 0.0,
                    "基线熵序列": [],
                    "回响熵序列": [],
                    "池统计": {"最终大小": 0, "有效温度": 0.0, "质心范数": 0.0}
                }
        """
        基线平均熵 = (
            self.基线评估器.计算平均熵()
            if self.基线评估器 is not None
            else 0.0
        )
        回响平均熵 = (
            self.回响评估器.计算平均熵()
            if self.回响评估器 is not None
            else 0.0
        )

        # 细腻度提升率：回响平均熵相对基线的变化百分比
        细腻度提升率 = 0.0
        if 基线平均熵 > _EPS:
            细腻度提升率 = (
                (回响平均熵 - 基线平均熵) / 基线平均熵 * 100.0
            )

        基线文本 = ""
        if self.基线评估器 is not None:
            基线文本 = "".join(self.基线评估器.生成token列表)
        回响文本 = ""
        if self.回响评估器 is not None:
            回响文本 = "".join(self.回响评估器.生成token列表)

        return {
            "提示词": self.提示词,
            "情感维度": self.情感维度,
            "基线文本": 基线文本,
            "回响文本": 回响文本,
            "基线平均熵": 基线平均熵,
            "回响平均熵": 回响平均熵,
            "细腻度提升率(%)": round(细腻度提升率, 4),
            "KL散度": 0.0,
            "基线熵序列": (
                self.基线评估器.计算熵列表()
                if self.基线评估器 is not None
                else []
            ),
            "回响熵序列": (
                self.回响评估器.计算熵列表()
                if self.回响评估器 is not None
                else []
            ),
            "池统计": self.回响池统计,
        }

    def 计算KL散度(self) -> float:
        """
        计算基线与回响各对应 token 位置的 KL 散度均值。

        如果两个评估器的步数不一致，则只取较短序列的前 N 步。
        如果其中一个评估器未设置，返回 0.0。

        Returns
        -------
        float
            各步 KL 散度的均值。
        """
        if self.基线评估器 is None or self.回响评估器 is None:
            return 0.0

        基线logits = self.基线评估器.logits_list
        回响logits = self.回响评估器.logits_list

        if not 基线logits or not 回响logits:
            return 0.0

        min_len = min(len(基线logits), len(回响logits))
        kl_values: List[float] = []

        for i in range(min_len):
            bl = torch.tensor(基线logits[i], dtype=torch.float32)
            el = torch.tensor(回响logits[i], dtype=torch.float32)
            kl = 计算KL散度(bl, el)
            kl_values.append(kl)

        return sum(kl_values) / len(kl_values) if kl_values else 0.0

    def 导出全部统计(self) -> List[dict]:
        """
        导出包含 KL 散度的完整统计列表（单元素列表，保持接口一致性）。

        Returns
        -------
        List[dict]
            包含一个整体统计字典的列表，字典中已写入 KL 散度字段。
        """
        统计 = self.计算整体统计()
        统计["KL散度"] = round(self.计算KL散度(), 6)
        return [统计]


# ══════════════════════════════════════════════════
# 5. 汇总统计器
# ══════════════════════════════════════════════════

class 汇总统计器:
    """
    汇总统计器 — 管理多次实验（多个提示词 × 多个情感维度）的汇总结果。

    支持逐条添加对比结果、计算整体汇总统计、以及 JSON 序列化
    与反序列化，方便实验结果的持久化存储和后续分析。

    Attributes
    ----------
    对比列表 : List[实验对比器]
        已添加的所有实验对比器实例。
    """

    def __init__(self) -> None:
        self.对比列表: List[实验对比器] = []

    def 添加对比(self, 对比器: 实验对比器) -> None:
        """
        添加一个实验对比器实例。

        Parameters
        ----------
        对比器 : 实验对比器
            已完成设置（基线和回响均已配置）的实验对比器。

        Raises
        ------
        TypeError
            如果输入不是实验对比器实例。
        """
        if not isinstance(对比器, 实验对比器):
            raise TypeError(
                f"对比器必须是实验对比器实例，收到 {type(对比器)}"
            )
        self.对比列表.append(对比器)

    def 计算整体汇总(self) -> dict:
        """
        计算所有实验对比的整体汇总统计。

        Returns
        -------
        dict
            包含实验配置和整体统计的字典：

            .. code-block:: python

                {
                    "实验配置": {
                        "实验数量": 5,
                        "情感维度列表": ["开心", "悲伤", "愤怒"],
                        "提示词列表": ["今天天气真好", ...]
                    },
                    "整体统计": {
                        "平均细腻度提升率(%)": 3.5,
                        "平均KL散度": 0.02,
                        "最大细腻度提升率(%)": 8.1,
                        "最小细腻度提升率(%)": -1.2,
                        "平均基线熵": 1.5,
                        "平均回响熵": 1.6,
                        "实验详情": [ ... ]
                    }
                }
        """
        实验数量 = len(self.对比列表)

        # 收集所有情感维度和提示词
        情感维度集合: set = set()
        提示词集合: set = set()
        所有统计: List[dict] = []

        for 对比器 in self.对比列表:
            情感维度集合.add(对比器.情感维度)
            提示词集合.add(对比器.提示词)
            统计 = 对比器.导出全部统计()
            所有统计.extend(统计)

        if 实验数量 == 0:
            return {
                "实验配置": {
                    "实验数量": 0,
                    "情感维度列表": [],
                    "提示词列表": [],
                },
                "整体统计": {
                    "平均细腻度提升率(%)": 0.0,
                    "平均KL散度": 0.0,
                    "最大细腻度提升率(%)": 0.0,
                    "最小细腻度提升率(%)": 0.0,
                    "平均基线熵": 0.0,
                    "平均回响熵": 0.0,
                    "实验详情": [],
                },
            }

        细腻度列表 = [s["细腻度提升率(%)"] for s in 所有统计]
        kl列表 = [s["KL散度"] for s in 所有统计]
        基线熵列表 = [s["基线平均熵"] for s in 所有统计]
        回响熵列表 = [s["回响平均熵"] for s in 所有统计]

        return {
            "实验配置": {
                "实验数量": 实验数量,
                "情感维度列表": sorted(情感维度集合),
                "提示词列表": sorted(提示词集合),
            },
            "整体统计": {
                "平均细腻度提升率(%)": round(
                    sum(细腻度列表) / len(细腻度列表), 4
                ),
                "平均KL散度": round(
                    sum(kl列表) / len(kl列表), 6
                ),
                "最大细腻度提升率(%)": round(max(细腻度列表), 4),
                "最小细腻度提升率(%)": round(min(细腻度列表), 4),
                "平均基线熵": round(
                    sum(基线熵列表) / len(基线熵列表), 4
                ),
                "平均回响熵": round(
                    sum(回响熵列表) / len(回响熵列表), 4
                ),
                "实验详情": 所有统计,
            },
        }

    def 导出JSON(self, 文件路径: str) -> None:
        """
        将汇总结果保存为 JSON 文件。

        Parameters
        ----------
        文件路径 : str
            目标文件路径（建议以 .json 结尾）。

        Raises
        ------
        TypeError
            如果文件路径不是字符串。
        OSError
            如果文件写入失败（如路径不存在或无权限）。
        """
        if not isinstance(文件路径, str):
            raise TypeError(
                f"文件路径必须是字符串，收到 {type(文件路径)}"
            )

        data = self.计算整体汇总()

        try:
            with open(文件路径, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            raise OSError(
                f"无法写入文件 {文件路径}：{e}"
            ) from e

    def 从JSON加载(self, 文件路径: str) -> dict:
        """
        从 JSON 文件加载汇总统计结果。

        Parameters
        ----------
        文件路径 : str
            已有的 JSON 文件路径。

        Returns
        -------
        dict
            反序列化后的汇总统计字典。

        Raises
        ------
        TypeError
            如果文件路径不是字符串。
        FileNotFoundError
            如果文件不存在。
        json.JSONDecodeError
            如果文件内容不是合法的 JSON。
        """
        if not isinstance(文件路径, str):
            raise TypeError(
                f"文件路径必须是字符串，收到 {type(文件路径)}"
            )

        try:
            with open(文件路径, "r", encoding="utf-8") as f:
                data: dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到：{文件路径}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"JSON 解析失败：{e.msg}", e.doc, e.pos
            )
        except OSError as e:
            raise OSError(
                f"无法读取文件 {文件路径}：{e}"
            ) from e

        return data
