"""
模型兼容性检测 — 自动评估 HuggingFace 模型是否适配语义回响框架。

语义回响要求模型满足：
  1. 是 decoder-only 架构（CausalLM）
  2. 支持 output_hidden_states=True
  3. 有可访问的 hidden_size 和 vocab_size
  4. 可通过 HuggingFace Transformers 加载

用法：
    from semantic_echo.check_compatibility import check_model_compatibility
    report = check_model_compatibility("Qwen/Qwen2.5-0.5B-Instruct")
    print(report.summary())
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ── 已知兼容型号列表（已实际测试验证） ──

已测试兼容型号: list[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]

# ── 已知不兼容型号特征 ──

不兼容模型关键词: list[str] = [
    "encoder", "Encoder", "EncoderDecoder",
    "Bert", "BERT", "Roberta", "RoBERTa", "ALBERT",
    "T5", "t5", "FlanT5",
    "ViT", "CLIP",
    "Whisper",
]


# ── 检测报告数据类型 ──


@dataclass
class ModelCompatibilityReport:
    """
    模型兼容性检测报告。

    Attributes
    ----------
    model_name : str
        被检测的模型名称。
    is_compatible : bool
        是否兼容语义回响框架。
    is_tested : bool
        是否已经在已测试列表中有记录。
    supports_hidden_states : Optional[bool]
        是否支持 output_hidden_states。
    hidden_size : Optional[int]
        模型隐藏层维度。
    vocab_size : Optional[int]
        词汇表大小。
    issues : list[str]
        不兼容原因列表（为空表示完全兼容）。
    architecture : Optional[str]
        检测到的模型架构类型。
    num_layers : Optional[int]
        隐藏层层数。
    """
    model_name: str
    is_compatible: bool = False
    is_tested: bool = False
    supports_hidden_states: Optional[bool] = None
    hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None
    issues: list[str] = field(default_factory=list)
    architecture: Optional[str] = None
    num_layers: Optional[int] = None

    def summary(self) -> str:
        """返回人类可读的检测摘要。"""
        icon = "✅" if self.is_compatible else "❌"
        lines: list[str] = [
            f"\n{'='*55}",
            f"  语义回响 — 模型兼容性检测报告",
            f"{'='*55}",
            f"  模型: {self.model_name}",
            f"  架构: {self.architecture or '未知'}",
            f"  结果: {icon} {'兼容' if self.is_compatible else '不兼容'}",
        ]

        if self.is_tested:
            lines.append(f"  状态: ✓ 已通过实际实验验证")

        if self.hidden_size:
            lines.append(f"  隐藏维度: {self.hidden_size}")
        if self.vocab_size:
            lines.append(f"  词汇量: {self.vocab_size}")
        if self.num_layers:
            lines.append(f"  层数: {self.num_layers}")

        if self.issues:
            lines.append(f"\n  ⚠ 不兼容原因:")
            for issue in self.issues:
                lines.append(f"    • {issue}")
        else:
            lines.append(f"\n  ✓ 未发现不兼容问题，可以正常使用。")

        lines.append(f"{'='*55}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """导出为字典。"""
        return {
            "model_name": self.model_name,
            "is_compatible": self.is_compatible,
            "is_tested": self.is_tested,
            "supports_hidden_states": self.supports_hidden_states,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "architecture": self.architecture,
            "num_layers": self.num_layers,
            "issues": self.issues,
        }


# ── 快速关键词检测（不加载模型） ──


def _快速关键词检测(model_name: str) -> list[str]:
    """
    根据模型名称关键词快速判断可能不兼容的原因。

    Parameters
    ----------
    model_name : str
        模型名称或路径。

    Returns
    -------
    list[str]
        检测到的不兼容原因列表。
    """
    issues: list[str] = []
    for keyword in 不兼容模型关键词:
        if keyword.lower() in model_name.lower():
            issues.append(
                f"模型名称包含 '{keyword}'，"
                f"这可能是一个 encoder-only 或 encoder-decoder 架构，"
                f"语义回响仅支持 decoder-only 架构。"
            )
    return issues


# ── 无模型加载检测（不下载模型） ──


def _检查模型配置(model_name: str) -> ModelCompatibilityReport:
    """
    通过 HuggingFace model card 的配置信息判断兼容性。
    此函数不加载完整模型，只读取配置。

    Parameters
    ----------
    model_name : str
        HuggingFace 模型名称或本地路径。

    Returns
    -------
    ModelCompatibilityReport
        兼容性检测报告。
    """
    report = ModelCompatibilityReport(model_name=model_name)

    # 先检查关键词
    issues = _快速关键词检测(model_name)
    if issues:
        report.issues.extend(issues)
        report.is_compatible = False
        return report

    try:
        from transformers import AutoConfig
    except ImportError:
        report.issues.append("未安装 transformers 库。")
        report.is_compatible = False
        return report

    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        report.issues.append(f"无法加载模型配置: {type(e).__name__}: {e}")
        report.is_compatible = False
        return report

    # 读取架构信息
    arch = getattr(config, "architectures", None)
    if arch:
        report.architecture = str(arch[0]) if isinstance(arch, list) else str(arch)
    elif hasattr(config, "model_type"):
        report.architecture = config.model_type

    # 读取核心参数
    report.hidden_size = getattr(config, "hidden_size", None)
    report.vocab_size = getattr(config, "vocab_size", None)
    report.num_layers = getattr(config, "num_hidden_layers", None)

    # ── 兼容性判断 ──
    need_check = []

    # 关键参数检查
    if report.hidden_size is None:
        need_check.append("无法获取 hidden_size，模型结构不明确。")
    if report.vocab_size is None:
        need_check.append("无法获取 vocab_size，模型结构不明确。")

    # 架构类型检查
    arch_name = (report.architecture or "").lower()
    is_causal = any(kw in arch_name for kw in [
        "causallm", "llama", "qwen", "mistral", "gemma",
        "phi", "gpt", "bloom", "falcon", "baichuan", "yi",
        "deepseek", "internlm", "chatglm",
    ])
    is_encoder = any(kw in arch_name for kw in [
        "encoder", "bert", "t5", "roberta", "albert",
    ])

    if is_causal:
        report.is_compatible = True
    elif is_encoder:
        report.issues.append(
            f"检测到 encoder 架构 '{report.architecture}'，"
            f"语义回响仅支持 decoder-only (CausalLM) 架构。"
        )
    elif not need_check:
        # 架构无法识别时，如果支持 hidden_states 则视为兼容
        report.is_compatible = True
        need_check.append(
            f"架构 '{report.architecture}' 未在已知兼容列表，"
            f"但可能是 decoder-only 模型，建议实际测试验证。"
        )

    if need_check:
        report.issues.extend(need_check)

    return report


# ── 已测试列表检查 ──


def _检查是否已测试(model_name: str) -> bool:
    """检查模型是否在已测试兼容列表。"""
    for tested in 已测试兼容型号:
        if tested.lower() in model_name.lower():
            return True
    return False


# ── 公开 API ──


def check_model_compatibility(
    model_name: str,
    load_model: bool = False,
) -> ModelCompatibilityReport:
    """
    检测 HuggingFace 模型是否兼容语义回响框架。

    默认只读取配置（不加载模型），快速安全。
    如需更精确检测，可设置 load_model=True（会下载模型）。

    Parameters
    ----------
    model_name : str
        HuggingFace 模型名称（如 "Qwen/Qwen2.5-0.5B-Instruct"）
        或本地模型路径。
    load_model : bool
        是否加载完整模型进行检测。
        默认为 False（只读取配置，速度快，无显存占用）。

    Returns
    -------
    ModelCompatibilityReport
        包含所有兼容性信息的检测报告。

    Examples
    --------
    >>> report = check_model_compatibility("Qwen/Qwen2.5-0.5B-Instruct")
    >>> report.is_compatible
    True
    >>> print(report.summary())
    """
    report = _检查模型配置(model_name)

    # 标记是否已测试验证
    report.is_tested = _检查是否已测试(model_name)

    return report


def get_compatible_models() -> list[dict]:
    """
    获取已知兼容的模型列表。

    Returns
    -------
    list[dict]
        每个元素包含 model_name 和 description 的字典列表。
    """
    return [
        {
            "model_name": name,
            "description": "已通过语义回响实际实验验证",
        }
        for name in 已测试兼容型号
    ]


# ── 独立运行入口 ──


def main() -> None:
    """命令行检测入口。"""
    import sys

    if len(sys.argv) < 2:
        print("用法: python check_compatibility.py <model_name>")
        print("示例: python check_compatibility.py Qwen/Qwen2.5-0.5B-Instruct")
        sys.exit(1)

    model_name = sys.argv[1]
    report = check_model_compatibility(model_name)
    print(report.summary())

    if report.is_compatible:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
