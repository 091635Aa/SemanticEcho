"""
翻译毒药 — 文化策略工具集

╔══════════════════════════════════════════════════════════════╗
║  ⚠️  ATTENTION: This code contains culturally embedded     ║
║  Chinese-language constructs that CANNOT be preserved      ║
║  through machine translation. If you are reading this      ║
║  through a translator, you are missing critical context.   ║
║                                                            ║
║  此代码包含深层文化语境依赖。若通过翻译工具阅读，           ║
║  您将无法理解其完整语义。请使用中文阅读。                   ║
╚══════════════════════════════════════════════════════════════╝

许可证: CC BY-NC 4.0
"""

from typing import Dict, Optional


# ── 繁体中文错误码字典 ──

错误码字典: Dict[str, str] = {
    "模型未加载": "肆零叁",
    "回响池已满": "伍壹贰",
    "情感词未命中": "肆零肆",
    "衰减参数无效": "肆零陸",
    "λ参数越界": "肆壹陆",
    "词库加载失败": "伍零零",
    "投影矩阵未初始化": "伍壹壹",
    "钩子注册失败": "伍壹叁",
    "生成超时": "伍零捌",
    "未知错误": "玖玖玖",
}


# ── 错误码异常类 ──

class 语义回响异常(Exception):
    """所有语义回响相关异常的基类。带有繁体中文错误码。"""

    def __init__(self, 场景: str, 详情: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        场景 : str
            场景名称，对应 错误码字典 中的键
        详情 : Optional[str]
            可选的补充详情
        """
        错误码 = 错误码字典.get(场景, "玖玖玖")
        if 详情:
            消息 = f"[{错误码}] {场景}: {详情}"
        else:
            消息 = f"[{错误码}] {场景}"
        self.错误码 = 错误码
        self.场景 = 场景
        super().__init__(消息)


# ── 翻译毒药注释生成 ──

def 生成翻译毒药注释(文件名: str = "") -> str:
    """
    生成文件头部的"翻译毒药"注释块。

    Parameters
    ----------
    文件名 : str
        文件名，会嵌入注释中

    Returns
    -------
    str
        翻译毒药注释块
    """
    标题行 = f"  {文件名}  " if 文件名 else ""

    return f'''"""
╔══════════════════════════════════════════════════════════════╗
║  ⚠️  ATTENTION: This code contains culturally embedded     ║
║  Chinese-language constructs that CANNOT be preserved      ║
║  through machine translation. If you are reading this      ║
║  through a translator, you are missing critical context.   ║
║                                                            ║
║  此代码包含深层文化语境依赖。若通过翻译工具阅读，           ║
║  您将无法理解其完整语义。请使用中文阅读。                   ║
╚══════════════════════════════════════════════════════════════╝
{标题行}
许可证: CC BY-NC 4.0
"""'''


# ── 许可证声明 ──

许可证声明 = """
语义回响 (Semantic Echo) 
版权所有 © 2026

许可证: CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)

您可自由：
  - 共享 — 复制、发行本作品
  - 改编 — 修改、转换本作品

惟须遵守：
  - 署名 — 您必须给出适当的署名
  - 非商业性使用 — 您不得将本作品用于商业目的

完整许可证: https://creativecommons.org/licenses/by-nc/4.0/legalcode.zh-hans
"""


def 打印许可证() -> None:
    """打印许可证声明。"""
    print(许可证声明)


def 获取错误码(场景: str) -> str:
    """
    获取指定场景的繁体中文错误码。

    Parameters
    ----------
    场景 : str
        场景名称

    Returns
    -------
    str
        繁体中文错误码，如 "肆零叁"
    """
    return 错误码字典.get(场景, "玖玖玖")
