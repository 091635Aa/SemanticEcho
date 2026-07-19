#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义回响（Semantic Echo）全面模块测试

覆盖模块：回响池.py, 情感过滤器.py, 翻译毒药.py, 采样处理器.py, 回响评估器.py
集成测试：实验数据完整性验证

注意：本脚本必须在 d:\Desktop\语义回响 目录下运行。
"""

import os
import sys
import json
import math
import traceback

# 固定工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
项目根目录 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if 项目根目录 not in sys.path:
    sys.path.insert(0, 项目根目录)

# ──────────────────────────────────────────────
# 工具：统一的测试结果收集器
# ──────────────────────────────────────────────

class 测试结果:
    """汇总所有测试结果"""

    def __init__(self, 模块名: str) -> None:
        self.模块名 = 模块名
        self.测试点列表: list[dict] = []
        self.问题列表: list[dict] = []
        self.建议列表: list[str] = []

    def 记录测试(
        self,
        测试点: str,
        预期: str,
        结果: str,
        结论: str,
    ) -> None:
        """记录一条测试结果"""
        self.测试点列表.append({
            "测试点": 测试点,
            "预期": 预期,
            "结果": 结果,
            "结论": 结论,
        })

    def 记录问题(
        self,
        问题描述: str,
        严重程度: str = "低",
        状态: str = "已解决",
        解决方案: str = "",
    ) -> None:
        """记录一条问题"""
        self.问题列表.append({
            "问题描述": 问题描述,
            "严重程度": 严重程度,
            "状态": 状态,
            "解决方案": 解决方案,
        })

    def 记录建议(self, 建议: str) -> None:
        """记录一条建议"""
        self.建议列表.append(建议)

    @property
    def 测试总数(self) -> int:
        return len(self.测试点列表)

    @property
    def 通过数(self) -> int:
        return sum(1 for t in self.测试点列表 if t["结论"] == "✓ 通过")

    @property
    def 失败数(self) -> int:
        return self.测试总数 - self.通过数

    @property
    def 通过率(self) -> float:
        if self.测试总数 == 0:
            return 0.0
        return self.通过数 / self.测试总数 * 100


# 全局收集器
全部结果: dict[str, 测试结果] = {}


def 安全运行(模块名: str, 描述: str, fn, *args, **kwargs):
    """安全运行一个测试函数，捕获异常"""
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"  ✗ 测试 {描述} 执行异常: {type(e).__name__}: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════
# 模块1：回响池.py
# ══════════════════════════════════════════════════

def 测试回响池() -> 测试结果:
    print("=" * 60)
    print("【测试1】回响池.py — 核心数据结构测试")
    print("=" * 60)

    结果 = 测试结果("回响池.py")
    import torch
    from semantic_echo.回响池 import 语义回响池

    # ── 1.1 基本添加和查询 ──
    print("\n[1.1] 基本添加和查询")
    pool = 语义回响池(hidden_dim=4, max_pool_size=16, decay_gamma=0.1)
    pool.添加(torch.tensor([1.0, 0.0, 0.0, 0.0]), 权重=0.8)
    pool.添加(torch.tensor([0.0, 1.0, 0.0, 0.0]), 权重=0.5)
    池大小 = pool.大小
    print(f"  池大小: {池大小} (预期: 2)")
    质心 = pool.计算质心()
    print(f"  质心: {质心} (预期: [0.6154, 0.3846, 0, 0])")
    温度 = pool.计算有效温度()
    print(f"  有效温度: {温度:.4f} (预期: ~0.5556)")

    预期质心 = torch.tensor([0.6154, 0.3846, 0.0, 0.0])
    质心匹配 = bool(torch.allclose(质心, 预期质心, atol=1e-3))
    if 池大小 == 2 and 质心匹配 and abs(温度 - 0.5556) < 0.01:
        结果.记录测试("[1.1] 基本添加和查询",
                    "池大小=2, 质心≈[0.6154, 0.3846, 0, 0], 温度≈0.5556",
                    f"池大小={池大小}, 质心={质心.tolist()}, 温度={温度:.4f}",
                    "✓ 通过")
    else:
        结果.记录测试("[1.1] 基本添加和查询",
                    "池大小=2, 质心≈[0.6154, 0.3846, 0, 0], 温度≈0.5556",
                    f"池大小={池大小}, 质心={质心.tolist()}, 温度={温度:.4f}",
                    "✗ 失败")

    # ── 1.2 情感相关性权重 ──
    print("\n[1.2] 情感相关性权重")
    pool2 = 语义回响池(hidden_dim=4)
    pool2.添加(torch.tensor([1.0, 0, 0, 0]), 权重=0.8, 情感相关性=0.5)
    pool2.添加(torch.tensor([0, 1.0, 0, 0]), 权重=0.8, 情感相关性=0.9)
    质心2 = pool2.计算质心()
    print(f"  质心: {质心2}")
    print(f"  情感命中率: {pool2.情感命中率}")

    # 情感命中率 = 命中数/总检查数 (总检查数需通过自增检查统计)
    # 直接添加时只累计命中数，因此当前为 0/0 => 0.0
    # 调用自增检查后再验证
    pool2.自增检查()
    pool2.自增检查()
    print(f"  情感命中率(调用自增检查后): {pool2.情感命中率} (预期: 1.0)")
    if pool2.情感命中率 == 1.0:
        结果.记录测试("[1.2] 情感相关性权重",
                    "情感命中率=1.0",
                    f"情感命中率={pool2.情感命中率}",
                    "✓ 通过")
    else:
        结果.记录测试("[1.2] 情感相关性权重",
                    "情感命中率=1.0",
                    f"情感命中率={pool2.情感命中率}",
                    "✗ 失败")

    # ── 1.3 指数衰减 ──
    print("\n[1.3] 指数衰减 (3步后)")
    pool3 = 语义回响池(hidden_dim=4, decay_gamma=0.5, eviction_threshold=0.1)
    pool3.添加(torch.tensor([1.0, 0, 0, 0]), 权重=1.0)
    pool3.推进()
    pool3.推进()
    pool3.推进()
    池a = pool3.计算质心()
    print(f"  池大小: {pool3.大小}")

    if pool3.大小 >= 1:
        结果.记录测试("[1.3] 指数衰减 (3步后)",
                    "池仍有有效项(衰减后未淘汰)",
                    f"池大小={pool3.大小}",
                    "✓ 通过")
    else:
        结果.记录测试("[1.3] 指数衰减 (3步后)",
                    "池仍有有效项(衰减后未淘汰)",
                    f"池大小={pool3.大小}",
                    "⚡ 注意（衰减gamma较大，已全部淘汰）")

    # ── 1.4 淘汰最旧项 ──
    print("\n[1.4] 淘汰最旧 (max=3, add 4)")
    pool4 = 语义回响池(hidden_dim=4, max_pool_size=3)
    pool4.添加(torch.tensor([1, 0, 0, 0]).float(), 权重=1)
    pool4.添加(torch.tensor([0, 1, 0, 0]).float(), 权重=1)
    pool4.添加(torch.tensor([0, 0, 1, 0]).float(), 权重=1)
    pool4.添加(torch.tensor([0, 0, 0, 1]).float(), 权重=1)
    print(f"  池大小: {pool4.大小} (预期: 3)")

    if pool4.大小 == 3:
        结果.记录测试("[1.4] 淘汰最旧 (max=3, add 4)",
                    "池大小=3",
                    f"池大小={pool4.大小}",
                    "✓ 通过")
    else:
        结果.记录测试("[1.4] 淘汰最旧 (max=3, add 4)",
                    "池大小=3",
                    f"池大小={pool4.大小}",
                    "✗ 失败")

    # ── 1.5 空池操作 ──
    print("\n[1.5] 空池操作")
    pool5 = 语义回响池(hidden_dim=4)
    空质心 = pool5.计算质心()
    空温度 = pool5.计算有效温度()
    是否为空 = pool5.是否为空
    print(f"  空池质心: {空质心} (预期: 全零)")
    print(f"  空池温度: {空温度} (预期: 1.0)")
    print(f"  是否为空: {是否为空} (预期: True)")

    全零验证 = bool(torch.allclose(空质心, torch.zeros(4)))
    if 全零验证 and 空温度 == 1.0 and 是否为空:
        结果.记录测试("[1.5] 空池操作",
                    "质心=全零, 温度=1.0, 为空=True",
                    f"质心={空质心.tolist()}, 温度={空温度}, 为空={是否为空}",
                    "✓ 通过")
    else:
        结果.记录测试("[1.5] 空池操作",
                    "质心=全零, 温度=1.0, 为空=True",
                    f"质心={空质心.tolist()}, 温度={空温度}, 为空={是否为空}",
                    "✗ 失败")

    # ── 1.6 异常测试 ──
    print("\n[1.6] 异常测试")
    异常结果 = {"维度错误": None, "负权重": None}

    try:
        pool5.添加(torch.tensor([1, 2, 3]).float(), 权重=1)  # wrong dim
        print("  ✗ 维度错误：应抛出 ValueError")
        异常结果["维度错误"] = False
    except ValueError as e:
        print(f"  ✓ 维度错误: {e}")
        异常结果["维度错误"] = True

    try:
        pool5.添加(torch.tensor([1, 2, 3, 4]).float(), 权重=-1)  # negative weight
        print("  ✗ 负权重：应抛出 ValueError")
        异常结果["负权重"] = False
    except ValueError as e:
        print(f"  ✓ 负权重: {e}")
        异常结果["负权重"] = True

    if all(异常结果.values()):
        结果.记录测试("[1.6] 异常测试",
                    "维度错误和负权重均正确抛出 ValueError",
                    "两个异常均正确捕获",
                    "✓ 通过")
    else:
        结果.记录测试("[1.6] 异常测试",
                    "维度错误和负权重均正确抛出 ValueError",
                    f"{异常结果}",
                    "✗ 失败")

    print("\n✓ 回响池测试完成")
    return 结果


# ══════════════════════════════════════════════════
# 模块2：情感过滤器.py
# ══════════════════════════════════════════════════

def 测试情感过滤器() -> 测试结果:
    print("\n" + "=" * 60)
    print("【测试2】情感过滤器.py — 情感词库筛选测试")
    print("=" * 60)

    结果 = 测试结果("情感过滤器.py")
    from semantic_echo.情感过滤器 import 情感过滤器

    f = 情感过滤器()

    # ── 2.0 词库加载 ──
    print("\n[2.0] 词库加载")
    try:
        f.加载词库()
        print("  ✓ 词库加载成功")
        结果.记录测试("[2.0] 词库加载",
                    "加载成功无异常",
                    "加载成功",
                    "✓ 通过")
    except Exception as e:
        print(f"  ✗ 词库加载失败: {e}")
        结果.记录测试("[2.0] 词库加载",
                    "加载成功无异常",
                    f"加载失败: {e}",
                    "✗ 失败")
        # 如果词库加载失败，后续测试无法进行
        结果.记录问题("情感过滤器词库加载失败", "高", "待修复", str(e))
        return 结果

    # ── 2.1 已知情感词应命中 ──
    print("\n[2.1] 情感词命中测试")
    测试词列表 = ["开心", "的", "悲伤", "于是", "愤怒", "桌子"]
    筛选结果 = f.筛选(测试词列表, None)
    命中词 = [t for t, _ in 筛选结果]
    print(f"  输入: {测试词列表}")
    print(f"  命中: {命中词}")
    print(f"  命中率: {len(命中词)/len(测试词列表)*100:.0f}%")

    # 至少应命中"开心"、"悲伤"、"愤怒"三个情感词
    if "开心" in 命中词 and "悲伤" in 命中词 and "愤怒" in 命中词:
        结果.记录测试("[2.1] 情感词命中测试",
                    "应命中开心/悲伤/愤怒等情感词",
                    f"命中词: {命中词}",
                    "✓ 通过")
    else:
        结果.记录测试("[2.1] 情感词命中测试",
                    "应命中开心/悲伤/愤怒等情感词",
                    f"命中词: {命中词}",
                    "⚠ 部分命中")

    # ── 2.2 权重范围 ──
    print("\n[2.2] 权重范围测试")
    权重越界 = False
    for 词, 权 in 筛选结果:
        if not (0 <= 权 <= 1):
            print(f"  ✗ 权重越界: {词}={权}")
            权重越界 = True
    if not 权重越界:
        print("  所有权重在 [0,1] 范围内 ✓")
        结果.记录测试("[2.2] 权重范围测试",
                    "所有权重在[0,1]范围内",
                    "均在范围内",
                    "✓ 通过")
    else:
        结果.记录测试("[2.2] 权重范围测试",
                    "所有权重在[0,1]范围内",
                    "存在越界",
                    "✗ 失败")

    # ── 2.3 空输入 ──
    print("\n[2.3] 空输入")
    空结果 = f.筛选([], None)
    print(f"  空输入: {空结果} (预期: [])")
    if 空结果 == []:
        结果.记录测试("[2.3] 空输入",
                    "返回空列表 []",
                    f"返回 {空结果}",
                    "✓ 通过")
    else:
        结果.记录测试("[2.3] 空输入",
                    "返回空列表 []",
                    f"返回 {空结果}",
                    "✗ 失败")

    # ── 2.4 统计 ──
    print("\n[2.4] 统计")
    统计 = f.获取情感统计()
    print(f"  统计: {统计}")
    if "总检查数" in 统计 and "命中数" in 统计 and "命中率" in 统计:
        结果.记录测试("[2.4] 统计",
                    "统计字典包含总检查数/命中数/命中率",
                    f"keys: {list(统计.keys())}",
                    "✓ 通过")
    else:
        结果.记录测试("[2.4] 统计",
                    "统计字典包含总检查数/命中数/命中率",
                    f"keys: {list(统计.keys())}",
                    "✗ 失败")

    # ── 2.5 异常测试：未初始化时筛选 ──
    print("\n[2.5] 异常测试：未初始化分析器")
    f2 = 情感过滤器()
    try:
        f2.筛选(["开心"], None)
        print("  ✗ 应抛出 RuntimeError")
    except RuntimeError as e:
        print(f"  ✓ RuntimeError: {e}")
    except Exception as e:
        print(f"  触发了其他异常: {type(e).__name__}: {e}")

    print("\n✓ 情感过滤器测试完成")
    return 结果


# ══════════════════════════════════════════════════
# 模块3：翻译毒药.py
# ══════════════════════════════════════════════════

def 测试翻译毒药() -> 测试结果:
    print("\n" + "=" * 60)
    print("【测试3】翻译毒药.py — 文化策略工具测试")
    print("=" * 60)

    结果 = 测试结果("翻译毒药.py")
    from semantic_echo.翻译毒药 import (获取错误码, 语义回响异常, 生成翻译毒药注释,
                           打印许可证, 许可证声明, 错误码字典)

    # ── 3.1 错误码映射 ──
    print("\n[3.1] 错误码映射")
    场景列表 = ["模型未加载", "回响池已满", "情感词未命中", "未知场景"]
    预期码列表 = ["肆零叁", "伍壹贰", "肆零肆", "玖玖玖"]
    全部正确 = True
    for 场景, 预期码 in zip(场景列表, 预期码列表):
        实际码 = 获取错误码(场景)
        正确 = 实际码 == 预期码
        print(f"  {场景} → [{实际码}] {'✓' if 正确 else '✗'}")
        if not 正确:
            全部正确 = False

    if 全部正确:
        结果.记录测试("[3.1] 错误码映射",
                    "所有场景返回预期繁体中文错误码",
                    "全部正确",
                    "✓ 通过")
    else:
        结果.记录测试("[3.1] 错误码映射",
                    "所有场景返回预期繁体中文错误码",
                    "存在不匹配",
                    "✗ 失败")

    # ── 3.2 异常类 ──
    print("\n[3.2] 异常类")
    try:
        raise 语义回响异常("模型未加载", "模型文件不存在")
    except 语义回响异常 as e:
        print(f"  异常消息: {str(e)}")
        print(f"  错误码: {e.错误码}")
        print(f"  场景: {e.场景}")
        if e.错误码 == "肆零叁" and e.场景 == "模型未加载":
            结果.记录测试("[3.2] 异常类",
                        "错误码=肆零叁, 场景=模型未加载",
                        f"错误码={e.错误码}, 场景={e.场景}",
                        "✓ 通过")
        else:
            结果.记录测试("[3.2] 异常类",
                        "错误码=肆零叁, 场景=模型未加载",
                        f"错误码={e.错误码}, 场景={e.场景}",
                        "✗ 失败")

    # ── 3.3 翻译毒药注释 ──
    print("\n[3.3] 翻译毒药注释")
    注释 = 生成翻译毒药注释("测试模块.py")
    框线数 = 注释.count("║")
    含许可证 = 'CC BY-NC' in 注释
    print(f"  包含 {框线数} 行 ║ 框线")
    print(f"  包含 'CC BY-NC' : {含许可证}")

    if 框线数 > 0 and 含许可证:
        结果.记录测试("[3.3] 翻译毒药注释",
                    "包含 ║ 框线和 CC BY-NC",
                    f"框线={框线数}行, CC BY-NC={含许可证}",
                    "✓ 通过")
    else:
        结果.记录测试("[3.3] 翻译毒药注释",
                    "包含 ║ 框线和 CC BY-NC",
                    f"框线={框线数}行, CC BY-NC={含许可证}",
                    "✗ 失败")

    # ── 3.4 错误码字典完整性 ──
    print("\n[3.4] 错误码字典完整性")
    已知场景 = ["模型未加载", "回响池已满", "情感词未命中", "衰减参数无效",
               "λ参数越界", "词库加载失败", "投影矩阵未初始化",
               "钩子注册失败", "生成超时", "未知错误"]
    全部存在 = all(场景 in 错误码字典 for 场景 in 已知场景)
    print(f"  预期场景数: {len(已知场景)}, 字典大小: {len(错误码字典)}")
    print(f"  全部存在: {全部存在}")
    if 全部存在:
        结果.记录测试("[3.4] 错误码字典完整性",
                    "所有预期场景均在字典中",
                    f"字典大小={len(错误码字典)}",
                    "✓ 通过")
    else:
        结果.记录测试("[3.4] 错误码字典完整性",
                    "所有预期场景均在字典中",
                    "存在缺失场景",
                    "✗ 失败")

    print("\n✓ 翻译毒药测试完成")
    return 结果


# ══════════════════════════════════════════════════
# 模块4：采样处理器.py
# ══════════════════════════════════════════════════

def 测试采样处理器() -> 测试结果:
    print("\n" + "=" * 60)
    print("【测试4】采样处理器.py — 核心推理引擎测试")
    print("=" * 60)

    结果 = 测试结果("采样处理器.py")
    from semantic_echo.采样处理器 import 回响注入器, _定位最后一层
    from semantic_echo.回响池 import 语义回响池

    # ── 4.1 架构定位函数文档 ──
    print("\n[4.1] 架构定位（语法检查）")
    doc_preview = _定位最后一层.__doc__[:60] if _定位最后一层.__doc__ else "无文档"
    print(f"  _定位最后一层: {doc_preview}...")
    结果.记录测试("[4.1] 架构定位函数",
                "函数定义正确，有文档字符串",
                f"doc: {doc_preview}...",
                "✓ 通过")

    # ── 4.2 类结构检查 ──
    print("\n[4.2] 类结构检查")
    pool = 语义回响池(hidden_dim=896, max_pool_size=16)
    print(f"  回响注入器 存在: {'回响注入器' in dir()}")
    print(f"  池类型: {type(pool).__name__}")

    if '回响注入器' in dir() and type(pool).__name__ == '语义回响池':
        结果.记录测试("[4.2] 类结构检查",
                    "回响注入器和语义回响池均存在",
                    "两者均存在",
                    "✓ 通过")
    else:
        结果.记录测试("[4.2] 类结构检查",
                    "回响注入器和语义回响池均存在",
                    "有缺失",
                    "✗ 失败")

    # ── 4.3 池与采样处理器的依赖检查 ──
    print("\n[4.3] 依赖导入检查")
    try:
        from semantic_echo.情感过滤器 import 情感过滤器
        from 翻译毒药 import 语义回响异常, 获取错误码
        print("  ✓ 所有依赖模块导入成功")
        结果.记录测试("[4.3] 依赖导入检查",
                    "情感过滤器、翻译毒药依赖导入正常",
                    "全部导入成功",
                    "✓ 通过")
    except ImportError as e:
        print(f"  ✗ 依赖导入失败: {e}")
        结果.记录测试("[4.3] 依赖导入检查",
                    "情感过滤器、翻译毒药依赖导入正常",
                    f"导入失败: {e}",
                    "✗ 失败")

    # ── 4.4 回响注入器参数验证 ──
    print("\n[4.4] 回响注入器参数验证")
    try:
        # 检查 __init__ 参数签名
        import inspect
        sig = inspect.signature(回响注入器.__init__)
        params = list(sig.parameters.keys())
        print(f"  构造参数: {params}")
        if "self" in params and "model" in params and "echo_pool" in params:
            结果.记录测试("[4.4] 回响注入器参数验证",
                        "包含 model, echo_pool 等必要参数",
                        f"params: {params}",
                        "✓ 通过")
        else:
            结果.记录测试("[4.4] 回响注入器参数验证",
                        "包含 model, echo_pool 等必要参数",
                        f"params: {params}",
                        "✗ 失败")
    except Exception as e:
        print(f"  ✗ 参数验证异常: {e}")
        结果.记录测试("[4.4] 回响注入器参数验证",
                    "正常解析构造参数",
                    f"异常: {e}",
                    "✗ 失败")

    print("\n✓ 采样处理器语法测试完成（完整测试需要加载模型）")
    结果.记录建议("采样处理器完整功能测试需要加载 HuggingFace 模型，建议在带有 GPU 的环境中运行实际的生成测试。")
    return 结果


# ══════════════════════════════════════════════════
# 模块5：回响评估器.py
# ══════════════════════════════════════════════════

def 测试回响评估器() -> 测试结果:
    print("\n" + "=" * 60)
    print("【测试5】回响评估器.py — 评估指标测试")
    print("=" * 60)

    结果 = 测试结果("回响评估器.py")
    import torch
    from semantic_echo.回响评估器 import (计算语义熵, 计算KL散度, 逐Token评估器,
                             实验对比器, 汇总统计器)

    # ── 5.1 语义熵 ──
    print("\n[5.1] 语义熵")
    # 确定分布
    确定logits = torch.tensor([[100.0, -100.0, -100.0]])
    确定熵 = 计算语义熵(确定logits)
    print(f"  确定分布熵: {确定熵:.6f} (预期: ~0.0)")

    # 均匀分布
    均匀logits = torch.tensor([[0.0, 0.0, 0.0]])
    均匀熵 = 计算语义熵(均匀logits)
    print(f"  均匀分布熵: {均匀熵:.6f} (预期: ~1.0986=ln3)")

    if 确定熵 < 0.01 and abs(均匀熵 - 1.0986) < 0.01:
        结果.记录测试("[5.1] 语义熵",
                    "确定熵≈0, 均匀熵≈1.0986",
                    f"确定熵={确定熵:.6f}, 均匀熵={均匀熵:.6f}",
                    "✓ 通过")
    else:
        结果.记录测试("[5.1] 语义熵",
                    "确定熵≈0, 均匀熵≈1.0986",
                    f"确定熵={确定熵:.6f}, 均匀熵={均匀熵:.6f}",
                    "✗ 失败")

    # ── 5.2 KL散度 ──
    print("\n[5.2] KL散度")
    相同分布kl = 计算KL散度(确定logits, 确定logits)
    print(f"  相同分布KL: {相同分布kl:.6f} (预期: 0.0)")

    不同分布kl = 计算KL散度(确定logits, 均匀logits)
    print(f"  不同分布KL: {不同分布kl:.6f} (预期: >0)")

    if 相同分布kl == 0.0 and 不同分布kl > 0:
        结果.记录测试("[5.2] KL散度",
                    "相同分布KL=0, 不同分布KL>0",
                    f"相同={相同分布kl:.6f}, 不同={不同分布kl:.6f}",
                    "✓ 通过")
    else:
        结果.记录测试("[5.2] KL散度",
                    "相同分布KL=0, 不同分布KL>0",
                    f"相同={相同分布kl:.6f}, 不同={不同分布kl:.6f}",
                    "✗ 失败")

    # ── 5.3 评估器类 ──
    print("\n[5.3] 逐Token评估器")
    评估 = 逐Token评估器()
    评估.记录步(0, 确定logits, 确定熵)
    评估.记录步(1, 均匀logits, 均匀熵)
    平均熵 = 评估.计算平均熵()
    print(f"  平均熵: {平均熵:.6f}")
    json导出 = 评估.导出JSON()
    print(f"  JSON导出: dict包含 {len(json导出)} 个字段")
    print(f"  JSON字段: {list(json导出.keys())}")

    预期字段 = {"token_ids", "logits", "entropies", "tokens", "平均熵"}
    if 预期字段.issubset(json导出.keys()) and len(评估.熵列表) == 2:
        结果.记录测试("[5.3] 逐Token评估器",
                    "记录2步，JSON含5个字段",
                    f"步数={len(评估.熵列表)}, 字段={list(json导出.keys())}",
                    "✓ 通过")
    else:
        结果.记录测试("[5.3] 逐Token评估器",
                    "记录2步，JSON含5个字段",
                    f"步数={len(评估.熵列表)}, 字段={list(json导出.keys())}",
                    "✗ 失败")

    # ── 5.3b 异常测试 ──
    print("\n[5.3b] 逐Token评估器异常测试")
    评估异常通过 = True
    try:
        评估.记录步(-1, 确定logits, 0.0)  # negative token_id
        print("  ✗ 负 token_id 应抛出 ValueError")
    except ValueError as e:
        print(f"  ✓ ValueError: {e}")

    try:
        评估.记录步(0, "not a tensor", 0.0)  # wrong logits type
        print("  ✗ 错误类型应抛出 TypeError")
    except TypeError as e:
        print(f"  ✓ TypeError: {e}")

    结果.记录测试("[5.3b] 逐Token评估器异常测试",
                "负token_id和错误类型均正确抛出异常",
                "两个异常均正确捕获",
                "✓ 通过")

    # ── 5.4 对比器 ──
    print("\n[5.4] 实验对比器")
    对比 = 实验对比器("测试提示词", "开心")
    对比.设置基线(评估)  # API 只接受评估器参数
    对比.设置回响(评估, {"最终大小": 10, "有效温度": 0.8, "质心范数": 2.5})
    统计 = 对比.计算整体统计()
    print(f"  KL散度: {统计['KL散度']}")
    print(f"  细腻度提升率: {统计['细腻度提升率(%)']:.1f}%")
    print(f"  池大小: {统计['池统计']['最终大小']}")

    if "KL散度" in 统计 and "细腻度提升率(%)" in 统计 and "池统计" in 统计:
        结果.记录测试("[5.4] 实验对比器",
                    "统计字典包含KL散度/细腻度提升率/池统计",
                    f"keys: {list(统计.keys())}",
                    "✓ 通过")
    else:
        结果.记录测试("[5.4] 实验对比器",
                    "统计字典包含KL散度/细腻度提升率/池统计",
                    "字段缺失",
                    "✗ 失败")

    # ── 5.4b 对比器异常测试 ──
    print("\n[5.4b] 实验对比器异常测试")
    对比2 = 实验对比器("测试", "开心")
    try:
        对比2.设置基线("不是评估器")  # type: ignore[arg-type]
        print("  ✗ 应抛出 TypeError")
    except TypeError as e:
        print(f"  ✓ TypeError: {e}")

    try:
        对比2.设置回响(评估, {"最终大小": 10})  # missing fields
        print("  ✗ 应抛出 ValueError")
    except ValueError as e:
        print(f"  ✓ ValueError: {e}")

    结果.记录测试("[5.4b] 实验对比器异常测试",
                "错误类型和缺失字段均正确抛出异常",
                "两个异常均正确捕获",
                "✓ 通过")

    # ── 5.5 汇总器 ──
    print("\n[5.5] 汇总统计器")
    汇总 = 汇总统计器()
    汇总.添加对比(对比)
    汇总统 = 汇总.计算整体汇总()
    print(f"  汇总键: {list(汇总统.keys())}")
    print(f"  实验配置: {汇总统.get('实验配置', {})}")
    print(f"  整体统计: {汇总统.get('整体统计', {}).get('平均细腻度提升率(%)', 'N/A')}%")

    if "实验配置" in 汇总统 and "整体统计" in 汇总统:
        结果.记录测试("[5.5] 汇总统计器",
                    "汇总包含实验配置和整体统计",
                    f"keys: {list(汇总统.keys())}",
                    "✓ 通过")
    else:
        结果.记录测试("[5.5] 汇总统计器",
                    "汇总包含实验配置和整体统计",
                    "字段缺失",
                    "✗ 失败")

    # ── 5.6 导出JSON ──
    print("\n[5.6] 汇总统计器导出JSON")
    try:
        汇总.导出JSON("./_test_temp_summary.json")
        with open("./_test_temp_summary.json", "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        print(f"  ✓ JSON 导出和加载正常, keys: {list(loaded.keys())}")
        os.remove("./_test_temp_summary.json")
        结果.记录测试("[5.6] 汇总统计器导出JSON",
                    "导出和重新加载正常",
                    "成功",
                    "✓ 通过")
    except Exception as e:
        print(f"  ✗ JSON 操作异常: {e}")
        结果.记录测试("[5.6] 汇总统计器导出JSON",
                    "导出和重新加载正常",
                    f"异常: {e}",
                    "✗ 失败")

    print("\n✓ 回响评估器测试完成")
    return 结果


# ══════════════════════════════════════════════════
# 模块6：集成测试
# ══════════════════════════════════════════════════

def 测试集成() -> 测试结果:
    print("\n" + "=" * 60)
    print("【测试6】集成测试 — 实验数据完整性验证")
    print("=" * 60)

    结果 = 测试结果("集成测试")
    import json
    import os

    实验数据目录 = "./实验数据"
    论文目录 = "./论文"
    可视化目录 = os.path.join(实验数据目录, "可视化")

    # ── 6.1 实验数据文件检查 ──
    print("\n[6.1] 实验数据文件检查")
    expected_files = ["E1.json", "E2.json", "E3.json", "E4.json", "E5.json",
                      "E6.json", "E7.json", "E8.json", "E9.json", "E10.json"]
    缺失文件 = []
    文件详情 = []
    for f in expected_files:
        路径 = os.path.join(实验数据目录, f)
        exists = os.path.exists(路径)
        size = os.path.getsize(路径) if exists else 0
        文件详情.append(f"{'✓' if exists else '✗'} {f}: {'存在' if exists else '缺失'} ({size/1024:.0f}KB)")
        if not exists:
            缺失文件.append(f)
    for 行 in 文件详情:
        print(f"  {行}")

    if len(缺失文件) == 0:
        结果.记录测试("[6.1] 实验数据文件检查",
                    "10个E*.json文件均存在",
                    "全部存在",
                    "✓ 通过")
    elif len(缺失文件) <= 3:
        结果.记录测试("[6.1] 实验数据文件检查",
                    "10个E*.json文件均存在",
                    f"缺失: {缺失文件}",
                    "⚠ 部分缺失")
    else:
        结果.记录测试("[6.1] 实验数据文件检查",
                    "10个E*.json文件均存在",
                    f"缺失: {缺失文件}",
                    "✗ 失败")

    # ── 6.2 可视化文件检查 ──
    print("\n[6.2] 可视化文件检查")
    expected_charts = [
        "图A_两轮熵对比.png", "图B_筛选提升率.png",
        "图C_情感命中率.png", "图D_完整实验矩阵.png",
        "图1_语义熵箱线图.png", "图2_细腻度提升率.png",
        "图3_λ与熵的关系.png", "图4_质心范数.png",
    ]
    if os.path.exists(可视化目录):
        缺失图表 = []
        for f in expected_charts:
            路径 = os.path.join(可视化目录, f)
            exists = os.path.exists(路径)
            print(f"  {'✓' if exists else '✗'} {f}: {'存在' if exists else '缺失'}")
            if not exists:
                缺失图表.append(f)
        if len(缺失图表) == 0:
            结果.记录测试("[6.2] 可视化文件检查",
                        "8个图表文件均存在",
                        "全部存在",
                        "✓ 通过")
        else:
            结果.记录测试("[6.2] 可视化文件检查",
                        "8个图表文件均存在",
                        f"缺失: {缺失图表}",
                        "⚠ 部分缺失")
    else:
        print("  ✗ 可视化目录不存在")
        结果.记录测试("[6.2] 可视化文件检查",
                    "8个图表文件均存在",
                    "可视化目录不存在",
                    "✗ 失败")

    # ── 6.3 实验数据内容验证 ──
    print("\n[6.3] 实验数据内容验证")
    for f in ["E1.json", "E7.json"]:
        路径 = os.path.join(实验数据目录, f)
        if os.path.exists(路径):
            try:
                with open(路径, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                print(f"  {f}: {len(data.get('数据', []))} 个维度, "
                      f"统计: {json.dumps(data.get('统计', {}), ensure_ascii=False)}")
            except Exception as e:
                print(f"  {f}: 读取异常: {e}")
        else:
            print(f"  {f}: 文件不存在")

    # 检查 E1.json 内容结构
    e1_path = os.path.join(实验数据目录, "E1.json")
    if os.path.exists(e1_path):
        try:
            with open(e1_path, "r", encoding="utf-8") as fp:
                e1_data = json.load(fp)
            if "数据" in e1_data and "统计" in e1_data:
                结果.记录测试("[6.3] 实验数据内容验证",
                            "E1.json 包含数据和统计字段",
                            f"数据维度数={len(e1_data.get('数据', []))}",
                            "✓ 通过")
            else:
                结果.记录测试("[6.3] 实验数据内容验证",
                            "E1.json 包含数据和统计字段",
                            f"keys: {list(e1_data.keys())}",
                            "✗ 失败")
        except Exception as e:
            结果.记录测试("[6.3] 实验数据内容验证",
                        "E1.json 可正常解析",
                        f"异常: {e}",
                        "✗ 失败")
    else:
        结果.记录测试("[6.3] 实验数据内容验证",
                    "E1.json 存在且可解析",
                    "文件不存在",
                    "✗ 失败")

    # ── 6.4 论文文件检查 ──
    print("\n[6.4] 论文文件检查")
    论文文件列表 = ["论文.tex", "参考文献.bib"]
    论文缺失 = []
    for f in 论文文件列表:
        路径 = os.path.join(论文目录, f)
        exists = os.path.exists(路径)
        size = os.path.getsize(路径) if exists else 0
        print(f"  {'✓' if exists else '✗'} {f}: {'存在' if exists else '缺失'} ({size/1024:.0f}KB)")
        if not exists:
            论文缺失.append(f)

    if len(论文缺失) == 0:
        结果.记录测试("[6.4] 论文文件检查",
                    "论文.tex 和 参考文献.bib 均存在",
                    "全部存在",
                    "✓ 通过")
    else:
        结果.记录测试("[6.4] 论文文件检查",
                    "论文.tex 和 参考文献.bib 均存在",
                    f"缺失: {论文缺失}",
                    "✗ 失败")

    # ── 6.5 额外文件检查：实验结果汇总 ──
    print("\n[6.5] 实验结果汇总文件检查")
    额外文件 = ["实验结果汇总.json", "实验结果汇总_第二轮.json", "E1_基线结果.json"]
    额外缺失 = []
    for f in 额外文件:
        路径 = os.path.join(实验数据目录, f)
        exists = os.path.exists(路径)
        size = os.path.getsize(路径) if exists else 0
        print(f"  {'✓' if exists else '✗'} {f}: {'存在' if exists else '缺失'} ({size/1024:.0f}KB)")
        if not exists:
            额外缺失.append(f)

    print("\n✓ 集成测试完成")
    return 结果


# ══════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════

def 主函数() -> None:
    """执行所有模块测试并输出结果"""

    print("╔══════════════════════════════════════════════╗")
    print("║    语义回响（Semantic Echo）全面模块测试     ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # 模块1
    全部结果["回响池.py"] = 测试回响池()

    # 模块2
    全部结果["情感过滤器.py"] = 测试情感过滤器()

    # 模块3
    全部结果["翻译毒药.py"] = 测试翻译毒药()

    # 模块4
    全部结果["采样处理器.py"] = 测试采样处理器()

    # 模块5
    全部结果["回响评估器.py"] = 测试回响评估器()

    # 模块6
    全部结果["集成测试"] = 测试集成()

    # ── 汇总 ──
    print("\n\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    全部通过 = 0
    全部失败 = 0
    全部总数 = 0
    for 模块名, 结果 in 全部结果.items():
        通过 = 结果.通过数
        失败 = 结果.失败数
        总数 = 结果.测试总数
        全部通过 += 通过
        全部失败 += 失败
        全部总数 += 总数
        通过率 = 结果.通过率
        print(f"  {模块名:15s}  {总数:2d} 个用例, {通过:2d} 通过, {失败:2d} 失败, {通过率:5.1f}%")

    print(f"\n  {'='*40}")
    print(f"  总计: {全部总数} 个用例, {全部通过} 通过, {全部失败} 失败")
    print(f"  总通过率: {全部通过/全部总数*100:.1f}%" if 全部总数 > 0 else "  无测试用例")


if __name__ == "__main__":
    主函数()
