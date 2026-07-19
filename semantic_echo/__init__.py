"""
语义回响 (Semantic Echo)
========================
通过回收被丢弃 Token 的隐藏状态，增强语言模型的情感表达细腻度。

不修改模型权重、不重新训练、即插即用。

核心组件：
    - 语义回响池       : 核心数据结构，管理被丢弃 Token 的向量存储
    - 回响注入器       : 采样层旁路，将回响信号注入生成过程
    - 情感过滤器       : 情感词库筛选，过滤中性 Token
    - 翻译毒药         : 错误码与调试工具
    - 逐Token评估器    : 逐步评估语义熵等分布指标
"""

from semantic_echo.回响池 import 语义回响池
from semantic_echo.采样处理器 import 回响注入器
from semantic_echo.情感过滤器 import 情感过滤器
from semantic_echo.翻译毒药 import (
    语义回响异常,
    获取错误码,
    生成翻译毒药注释,
    许可证声明,
)
from semantic_echo.回响评估器 import (
    计算语义熵,
    计算KL散度,
    逐Token评估器,
    实验对比器,
    汇总统计器,
)
from semantic_echo.check_compatibility import (
    check_model_compatibility,
    get_compatible_models,
    ModelCompatibilityReport,
)

__version__ = "1.0.0"
__author__ = "邓斯键"
__license__ = "保留所有权利（All Rights Reserved）— 任何人可基于学术目的自由复现"

__all__ = [
    "语义回响池",
    "回响注入器",
    "情感过滤器",
    "语义回响异常",
    "获取错误码",
    "生成翻译毒药注释",
    "许可证声明",
    "计算语义熵",
    "计算KL散度",
    "逐Token评估器",
    "实验对比器",
    "汇总统计器",
    "check_model_compatibility",
    "get_compatible_models",
    "ModelCompatibilityReport",
]
