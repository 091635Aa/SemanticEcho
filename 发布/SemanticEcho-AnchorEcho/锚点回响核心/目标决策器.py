# -*- coding: utf-8 -*-
"""P4 锚点回响（Anchor Echo）—— 目标决策器（新增模块）

按 P4_混合方案设计.md 第 2.3 节：
输入：会话 VAD（复用 潮汐感知器 + 潮汐决策器）/ 思考链文本 / 显式指令
输出：锚点目标(v_target ∈ R^K, β, 情感词密度目标, 说明)

核心逻辑：
  ① 目标状态：复用 潮汐决策器.计算目标()（V_target = 用户当前×β共情 + 轨迹均值×(1-β共情)）；
  ② VAD → 锚点权重：方向余弦 + 平方放大（保持极性）+ 唤醒接近度因子；
  ③ 多源融合：v_target = w_潮汐·v_潮汐 + w_思考链·v_思考链 + w_指令·v_指令，L2 归一化；
  ④ 强度自适应：β = β基 × (1+Δarousal) × min(1, 活跃度×2.5)，封顶 β上限；
  ⑤ 无任何输入情感时 → 默认指向「温柔」（陪伴基调）。
  ⑥ 角色感知（SubTask 6b 新增）：计算目标() 可选 角色 参数——命中 角色锚点基调表
     时 v_target = 0.7·v_角色 + 0.3·v_用户（角色基调主导方向、用户当前情感辅助微调），
     角色基调跨轮固定、用户部分每轮更新；未命中/不传角色 → 行为与旧版完全一致。

导入路径：sys.path 插入 情感潮汐解码（ETD）目录 与 语义回响工程根（参照 混合注入器.py）；
若 import 潮汐感知器/潮汐决策器 失败（如 cnsenti 缺失）→ 内置简易 VAD 词表兜底，
不影响主流程（降级情况在冒烟结果中记录）。
"""
import math
import sys
import dataclasses
from typing import Optional

import numpy as np

# ── 路径注入（参照 P3 混合注入器.py 的 sys.path 写法）──
ETD目录 = r"h:\情感潮汐解码（Emotion Tidal Decoding, ETD）"
回响工程根 = r"i:\Desktop\语义回响"
for _p in (ETD目录, 回响工程根):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 尝试复用 P3 感知/决策模块；失败则用内置简易 VAD 兜底
# 注意：用别名导入（构造函数参数同名会遮蔽类名，故单独保存类引用）
_潮汐可用 = True
_潮汐导入错误 = ""
try:
    from 潮汐感知器 import 潮汐感知器 as 潮汐感知器类, 情感状态
    from 潮汐决策器 import 潮汐决策器 as 潮汐决策器类, 潮汐目标
except Exception as _e:  # noqa: BLE001 —— 降级兜底，不阻断主流程
    _潮汐可用 = False
    _潮汐导入错误 = str(_e)
    潮汐感知器类 = None
    潮汐决策器类 = None
    情感状态 = None
    潮汐目标 = None


# ══════════════════════════════════════════════════
# 简易 VAD 兜底（cnsenti/潮汐感知器 缺失时使用）
# ══════════════════════════════════════════════════

简易正面词 = {
    "开心", "高兴", "快乐", "幸福", "喜悦", "兴奋", "愉快", "甜蜜", "喜欢", "爱",
    "温暖", "温柔", "安慰", "美好", "感动", "棒", "爽", "舒服", "满意", "幸运",
    "灿烂", "欢笑", "欢呼", "甜蜜", "哈哈", "嘻嘻", "耶", "棒极了", "赞",
}
简易负面词 = {
    "难过", "悲伤", "伤心", "痛苦", "失落", "心碎", "沮丧", "哭泣", "委屈", "心疼",
    "绝望", "崩溃", "心累", "扎心", "孤独", "焦虑", "担心", "害怕", "恐惧", "紧张",
    "不安", "恐慌", "愤怒", "生气", "恼火", "火大", "怨恨", "烦", "烦躁", "郁闷",
    "压抑", "寂寞", "空虚", "痛", "疼", "累", "哭", "泪", "愁", "苦", "丧", "颓",
    "破防", "内耗", "泪目",
    # SubTask 6b 补测扩充：cnsenti 未收录的自卑/低价值口语词（"我是不是很没用"类）
    "没用", "废物", "做不好", "差劲", "撑不下去", "扛不住", "没出息", "一事无成",
    "对不起", "自责", "自卑", "迷茫",
}
简易程度词 = {"非常": 3.0, "特别": 3.0, "超级": 3.0, "太": 2.5, "很": 2.0,
              "好": 1.5, "有点": 0.5, "稍微": 0.5}
简易否定词 = {"不", "没", "没有", "别", "无", "莫"}


class 简易情感状态:
    """VAD 兜底情感状态（与 潮汐感知器.情感状态 同接口）"""
    __slots__ = ("valence", "arousal", "dominance")

    def __init__(self, valence=0.0, arousal=0.0, dominance=0.0):
        self.valence = valence
        self.arousal = arousal
        self.dominance = dominance

    def 到向量(self):
        return (self.valence, self.arousal, self.dominance)

    def 强度(self):
        return abs(self.arousal)

    def 极性(self):
        return self.valence

    def __repr__(self):
        return (f"简易情感状态(V={self.valence:.3f}, A={self.arousal:.3f}, "
                f"D={self.dominance:.3f})")


class 简易感知器:
    """cnsenti 缺失时的 VAD 文本测量兜底：内置简易情感词表扫描"""

    def __init__(self, 窗口: int = 32):
        self.窗口 = 窗口
        self._轨迹 = []
        self._轮次 = 0

    def 测量(self, 文本: str):
        """返回 (简易情感状态, 关键词列表)。极性 = tanh((pos-neg)/词数)，
        唤醒度 = 情感词密度，支配度 = |valence|（简化）。"""
        if not 文本 or not 文本.strip():
            return 简易情感状态(), []
        词序列 = self._切词(文本)
        pos = neg = 0.0
        命中词 = []
        for i, 词 in enumerate(词序列):
            if 词 in 简易程度词 or 词 in 简易否定词:
                continue
            极性 = 1.0 if 词 in 简易正面词 else (-1.0 if 词 in 简易负面词 else 0.0)
            if 极性 == 0.0:
                continue
            前词 = 词序列[i - 1] if i > 0 else ""
            程度 = 简易程度词.get(前词, 1.0)
            if 前词 in 简易否定词:
                极性 = -极性
                程度 *= 0.5
            得分 = 极性 * 程度
            if 得分 > 0:
                pos += 得分
            else:
                neg += -得分
            命中词.append(词)
        词数 = max(len(词序列), 1)
        valence = math.tanh((pos - neg) / 词数 * 2.0)
        arousal = min(1.0, (pos + neg) / 词数)
        dominance = min(1.0, abs(valence) * 1.2)
        return 简易情感状态(valence, arousal, dominance), 命中词[:5]

    @staticmethod
    def _切词(文本: str):
        """简易切词：英文按词，中文滑窗 2/3 字（近似分词，兜底够用）"""
        import re
        词序列 = []
        for 段 in re.findall(r"[A-Za-z']+", 文本):
            词序列.extend(段.split())
        中文段 = re.sub(r"[^\u4e00-\u9fff]", "", 文本)
        for i in range(len(中文段) - 1):
            词序列.append(中文段[i:i + 2])
        for i in range(len(中文段) - 2):
            词序列.append(中文段[i:i + 3])
        return 词序列

    def 追加轨迹(self, 说话人: str, 状态, 关键词=None):
        self._轮次 += 1
        self._轨迹.append({"说话人": 说话人, "轮次": self._轮次,
                           "状态": 状态, "关键词": 关键词 or []})
        if len(self._轨迹) > self.窗口:
            self._轨迹 = self._轨迹[-self.窗口:]

    def 当前潮位(self, 忽略说话人=None):
        if not self._轨迹:
            return 简易情感状态()
        可用 = [p for p in self._轨迹 if p["说话人"] != 忽略说话人]
        if not 可用:
            return 简易情感状态()
        n = len(可用)
        v = sum(p["状态"].valence * (0.9 ** (n - 1 - i)) for i, p in enumerate(可用))
        a = sum(p["状态"].arousal * (0.9 ** (n - 1 - i)) for i, p in enumerate(可用))
        d = sum(p["状态"].dominance * (0.9 ** (n - 1 - i)) for i, p in enumerate(可用))
        总重 = sum(0.9 ** (n - 1 - i) for i in range(n))
        return 简易情感状态(v / 总重, a / 总重, d / 总重)

    def 轨迹活跃度(self, 最近轮数: int = 8):
        if not self._轨迹:
            return 0.0
        if len(self._轨迹) == 1:
            return min(1.0, abs(self._轨迹[-1]["状态"].arousal) * 1.5)
        最近 = self._轨迹[-最近轮数:]
        位移 = 0.0
        for i in range(1, len(最近)):
            p0, p1 = 最近[i - 1]["状态"], 最近[i]["状态"]
            位移 += abs(p1.valence - p0.valence) + abs(p1.arousal - p0.arousal)
        return min(1.0, 位移 / max(len(最近) - 1, 1))

    def 轨迹长度(self):
        return len(self._轨迹)

    def 重置轨迹(self):
        self._轨迹 = []
        self._轮次 = 0


# ══════════════════════════════════════════════════
# VAD → 锚点空间原型表（每维锚点一个 VAD 原型三元组，人工标注，按设计文档）
# ══════════════════════════════════════════════════

默认原型表 = {
    "温柔": (0.30, 0.20, 0.20),
    "开心": (0.80, 0.70, 0.40),
    "难过": (-0.70, 0.40, 0.10),
    "愤怒": (-0.60, 0.80, 0.60),
    "害怕": (-0.60, 0.70, 0.00),
    "平静": (0.00, 0.00, 0.30),
}


# ══════════════════════════════════════════════════
# 角色 → 锚点维度基调表（SubTask 6b 新增·角色感知）
# 每项 = (锚点维度权重字典, VAD 三元组)
#   维度权重：按默认词集维度（温柔/开心/难过/愤怒/害怕/平静）给出该角色在锚点空间
#             的主导方向（按维度名对齐，K 变化时仍可用）；VAD 三元组用于角色唤醒度
#             （β/密度自适应），与 P3 评测 角色VAD基调 同口径。
# ══════════════════════════════════════════════════

角色锚点基调表 = {
    "温柔治愈系女友":   ({"温柔": 0.9, "开心": 0.3, "平静": 0.3}, (0.50, 0.20, 0.15)),
    "毒舌但心软的损友": ({"愤怒": 0.9, "难过": 0.4, "温柔": 0.2, "开心": 0.1}, (-0.10, 0.40, 0.40)),
    "理性冷静的职场前辈": ({"平静": 0.9, "温柔": 0.2, "愤怒": 0.1}, (0.00, 0.05, 0.50)),
    "活泼开朗的同桌":   ({"开心": 0.9, "温柔": 0.4, "平静": 0.1}, (0.60, 0.70, 0.30)),
    "沉默寡言的兄长":   ({"平静": 0.8, "温柔": 0.4, "难过": 0.1}, (0.00, 0.05, 0.25)),
    "傲娇的青梅竹马":   ({"愤怒": 0.7, "温柔": 0.5, "开心": 0.3, "难过": 0.2}, (-0.10, 0.30, 0.40)),
    "阅历丰富的老人":   ({"平静": 0.8, "温柔": 0.5, "难过": 0.1}, (0.30, 0.10, 0.40)),
    "爱撒娇的小女儿":   ({"开心": 0.9, "温柔": 0.7, "害怕": 0.1}, (0.50, 0.60, 0.10)),
    "严厉又公正的老师": ({"平静": 0.7, "愤怒": 0.6, "难过": 0.1}, (-0.20, 0.30, 0.70)),
    "幽默的脱口秀演员": ({"开心": 0.9, "温柔": 0.3, "愤怒": 0.1}, (0.40, 0.60, 0.50)),
    # 泛化角色名（非 EmoCharacter 场景的通用角色名映射，供别名/子串匹配）
    "温柔妈妈":         ({"温柔": 0.9, "开心": 0.3, "平静": 0.2}, (0.40, 0.15, 0.20)),
    "严肃导师":         ({"平静": 0.8, "愤怒": 0.5, "温柔": 0.2}, (-0.15, 0.25, 0.60)),
    "活泼闺蜜":         ({"开心": 0.9, "温柔": 0.4, "平静": 0.1}, (0.55, 0.65, 0.30)),
    "毒舌损友":         ({"愤怒": 0.9, "难过": 0.4, "温柔": 0.2, "开心": 0.1}, (-0.10, 0.40, 0.40)),
    "爱撒娇女儿":       ({"开心": 0.9, "温柔": 0.7, "害怕": 0.1}, (0.50, 0.60, 0.10)),
}

角色别名表 = {
    "损友": "毒舌但心软的损友", "毒舌": "毒舌但心软的损友",
    "小女儿": "爱撒娇的小女儿", "撒娇": "爱撒娇的小女儿", "女儿": "爱撒娇的小女儿",
    "妈妈": "温柔妈妈", "母亲": "温柔妈妈",
    "女友": "温柔治愈系女友", "恋人": "温柔治愈系女友",
    "导师": "严肃导师", "老师": "严肃导师", "教授": "严肃导师",
    "闺蜜": "活泼闺蜜", "同桌": "活泼开朗的同桌",
    "前辈": "理性冷静的职场前辈", "上司": "理性冷静的职场前辈",
    "兄长": "沉默寡言的兄长", "哥哥": "沉默寡言的兄长",
    "青梅竹马": "傲娇的青梅竹马", "竹马": "傲娇的青梅竹马",
    "老人": "阅历丰富的老人", "爷爷": "阅历丰富的老人", "奶奶": "阅历丰富的老人",
    "脱口秀演员": "幽默的脱口秀演员", "脱口秀": "幽默的脱口秀演员", "演员": "幽默的脱口秀演员",
}


def 匹配角色基调(角色: str):
    """角色名 → 角色锚点基调表条目 (维度权重dict, VAD)；未命中返回 None。

    匹配顺序：精确表键 → 别名表 → 子串双向匹配（取最长表键命中）。
    """
    if not 角色 or not isinstance(角色, str):
        return None
    if 角色 in 角色锚点基调表:
        return 角色锚点基调表[角色]
    if 角色 in 角色别名表:
        _k = 角色别名表[角色]
        if _k in 角色锚点基调表:
            return 角色锚点基调表[_k]
    命中 = [(len(_key), _key, _基调) for _key, _基调 in 角色锚点基调表.items()
            if _key in 角色 or 角色 in _key]
    if not 命中:
        return None
    命中.sort(key=lambda _x: -_x[0])
    return 命中[0][2]


@dataclasses.dataclass
class 锚点目标:
    """目标决策器输出：v_target ∈ R^K（L2 归一化）+ β + 密度目标 + 说明"""
    v_target: np.ndarray
    β: float
    情感词密度目标: float = 0.06
    说明: str = ""
    活跃度: float = 0.0
    决策日志: dict = None

    def __post_init__(self):
        if self.决策日志 is None:
            self.决策日志 = {}

    def __repr__(self):
        主维 = int(np.argmax(self.v_target)) if len(self.v_target) else -1
        return (f"锚点目标(主分量#{主维}={self.v_target[主维]:.3f}, β={self.β:.3f}, "
                f"密度={self.情感词密度目标:.3f}) | {self.说明}")


# ══════════════════════════════════════════════════
# 自动适配（P2 扩展：扫描表五元组 → 公式兜底 → 未注册保守 β=0.5）
# ══════════════════════════════════════════════════

# 扫描表五元组 (hidden_dim → (β基础, T_anchor))：β 对应设计文档 §4.1
扫描表β = {896: (1.0, 0.4), 1536: (0.8, 0.3), 2048: (0.6, 0.3), 3584: (0.5, 0.2)}


def 架构族β因子(模型名: str):
    """β 架构族因子（P4 新增，独立于 P1 的 λ 因子）"""
    if "Qwen3" in 模型名:
        if any(k in 模型名 for k in ("0.6", "1.7", "1.5")):
            return (0.4, "Qwen3≤1.7B")
        return (0.7, "Qwen3≥4B")
    if "gemma" in 模型名.lower():
        return (0.8, "Gemma")
    if "SmolLM" in 模型名:
        return (0.6, "SmolLM")
    if "Phi" in 模型名:
        return (0.8, "Phi")
    return (1.0, "Qwen2.5/通用")


def 自动适配(model, 量化类型: str = "fp16"):
    """β / T_anchor 自动适配：
    命中扫描表(hidden) 且 命中架构族 → 预制五元组（β/T_anchor）；
    未命中 → 公式兜底 β = β_基础(hidden) × 架构族因子 × 量化因子；
    模型信息不可读（未注册）→ 保守 β=0.5。
    """
    try:
        hidden = int(model.config.hidden_size)
        模型名 = str(getattr(model.config, "_name_or_path", "") or type(model).__name__)
    except Exception:  # noqa: BLE001 —— 未注册模型保守兜底
        return {"β": 0.5, "T_anchor": 0.3, "K": 6, "hidden_dim": None,
                "来源": "未注册模型保守兜底(β=0.5)"}
    族因子, 族名 = 架构族β因子(模型名)
    量化因子 = 0.75 if str(量化类型).lower() == "4bit" else 1.0
    if hidden in 扫描表β:
        β基础, T_anchor = 扫描表β[hidden]
        基础来源 = "扫描表"
    else:
        # 公式兜底：β_基础 = 0.8·(896/hidden)^0.25（hidden<2048）；0.6·(896/hidden)^0.2（hidden≥2048）
        if hidden < 2048:
            β基础 = 0.8 * (896.0 / hidden) ** 0.25
        else:
            β基础 = 0.6 * (896.0 / hidden) ** 0.2
        T_anchor = 0.3
        基础来源 = "公式"
    β = β基础 * 族因子 * 量化因子
    return {"β": round(β, 4), "T_anchor": T_anchor, "K": 6, "hidden_dim": hidden,
            "β基础": round(β基础, 4), "量化因子": 量化因子, "族因子": 族因子,
            "来源": f"自动适配({族名}×{族因子}, {量化类型}×{量化因子}, 基础{基础来源})"}


# ══════════════════════════════════════════════════
# 目标决策器
# ══════════════════════════════════════════════════

class 目标决策器:
    """会话情感 → 锚点目标（v_target ∈ R^K + β + 密度目标 + 说明）

    复用 P3：潮汐感知器（VAD 文本测量）+ 潮汐决策器（共情目标计算）；
    import 失败时自动降级为内置简易 VAD 词表（self._简易模式 = True）。
    """

    def __init__(
        self,
        感知器=None,
        潮汐决策器=None,
        锚点库=None,
        原型表: Optional[dict] = None,
        β共情: float = 0.6,
        角色权重: float = 0.6,
        β基: float = 0.8,
        β上限: float = 2.0,
        来源权重: Optional[dict] = None,
        活跃度窗口: int = 8,
        锐化指数: float = 1.0,
        唤醒权重: float = 2.0,
        分量权重: Optional[tuple] = None,
        角色=None,
        密度基: float = 0.06,
        密度增益: float = 0.10,
        密度上限: float = 0.25,
    ):
        """参数说明（对齐设计文档 §2.3）：
        感知器/潮汐决策器：P3 实例，None 时自动构建（import 失败则简易兜底）；
        锚点库：用于指令命中的维度词集（None 时仅用 维度名 匹配）；
        原型表：{维度名: (valence, arousal, dominance)}，默认见 默认原型表；
        β共情/角色权重：传给 潮汐决策器 的共情/角色参数；
        β基：自适应公式的强度基础；β上限：强度封顶；
        来源权重：{"潮汐": 0.5, "思考链": 0.3, "指令": 0.2} 多源融合权重；
        锐化指数：v_k = g·|g|^p 的幂（p≥1 锐化高匹配维）；
        唤醒权重：唤醒接近度因子 exp(-λ·|V.arousal - 原型.arousal|) 的 λ；
        分量权重：VAD 余弦前的分量缩放 (valence, arousal, dominance)。
        """
        # ── 感知器 / 潮汐决策器（自动构建 or 简易兜底）──
        self._简易模式 = False
        if _潮汐可用 and (感知器 is None or 潮汐决策器 is None):
            try:
                感知器 = 感知器 if 感知器 is not None else 潮汐感知器类()
                潮汐决策器 = 潮汐决策器 if 潮汐决策器 is not None else 潮汐决策器类(
                    感知器, β共情=β共情, 角色权重=角色权重, 角色=角色)
            except Exception:  # noqa: BLE001 —— 降级兜底
                self._简易模式 = True
                感知器 = 感知器 if 感知器 is not None else 简易感知器()
                潮汐决策器 = None
        elif 感知器 is None:
            self._简易模式 = True
            感知器 = 简易感知器()
            潮汐决策器 = None

        self.感知器 = 感知器
        self.潮汐决策器 = 潮汐决策器
        self.锚点库 = 锚点库
        self.β共情 = β共情
        self.角色权重 = 角色权重
        self.β基 = β基
        self.β上限 = β上限
        self.来源权重 = 来源权重 or {"潮汐": 0.5, "思考链": 0.3, "指令": 0.2}
        self.活跃度窗口 = 活跃度窗口
        self.锐化指数 = 锐化指数
        self.唤醒权重 = 唤醒权重
        self.分量权重 = 分量权重 if 分量权重 is not None else (1.0, 0.5, 0.25)
        self.角色 = 角色
        self.密度基 = 密度基
        self.密度增益 = 密度增益
        self.密度上限 = 密度上限

        self.维度 = list(锚点库.维度名()) if 锚点库 is not None else list(默认原型表.keys())
        self.原型表 = dict(原型表 if 原型表 is not None else 默认原型表)

    # ──────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────

    def 计算目标(self, 用户当前=None, 思考链文本: str = "", 指令: str = "",
                角色=None, 轮次: int = 0) -> 锚点目标:
        """计算锚点目标：多源融合 → L2 归一化 → β/密度 自适应。

        用户当前：情感状态 / str（自动测量）/ None（用轨迹潮位）；
        思考链文本：思考链注入器输出，作为附加情感来源（可选）；
        指令：显式情感指令（如"温柔陪伴"→指定维度 one-hot+邻域平滑，可选）；
        角色（SubTask 6b 新增·角色感知）：角色名 str（查 角色锚点基调表）或
            具 valence/arousal/dominance 的角色基调对象；非 None 且命中基调时
            v_target = 0.7·v_角色 + 0.3·v_用户（角色基调主导方向、用户当前情感
            辅助微调），角色基调跨轮固定、用户部分每轮更新（跨轮会话复用）；
            未命中映射表时自动跳过 → 行为与不传角色完全一致（向后兼容）。
        轮次：当前对话轮次（用于日志/说明；角色基调固定，无需跨轮平滑）。
        """
        K = len(self.维度)
        日志: dict = {}

        # ── ① 用户当前 VAD（str → 测量 + 追加轨迹）──
        if isinstance(用户当前, str) and 用户当前.strip():
            状态, 关键词 = self.感知器.测量(用户当前)
            # SubTask 6b 增强：cnsenti 未命中（如单字"累"）→ 内置词表子串补测
            if abs(状态.arousal) < 1e-9:
                状态2 = self._简易补测(用户当前)
                if 状态2 is not None:
                    状态 = 状态2
                    关键词 = []
            self.感知器.追加轨迹("用户", 状态, 关键词)
            用户当前 = 状态
        elif 用户当前 is None and self.感知器.轨迹长度() > 0:
            try:
                用户当前 = self.感知器.当前潮位(忽略说话人="模型")
            except Exception:  # noqa: BLE001
                用户当前 = None

        # ── ② 潮汐源：共情目标状态 + α + 密度（复用 潮汐决策器）──
        v_潮汐 = np.zeros(K)
        目标状态 = None
        Δ唤醒 = 0.0
        活跃度 = 0.0
        α潮汐 = 0.0
        if 用户当前 is not None:
            if self.潮汐决策器 is not None:
                try:
                    目标 = self.潮汐决策器.计算目标(用户当前)
                    目标状态 = 目标.目标状态
                    α潮汐 = 目标.引导强度
                    密度潮汐 = 目标.密度目标
                    日志潮汐 = self.潮汐决策器.决策日志
                    轨迹均值 = 日志潮汐.get("轨迹均值", {}) or {}
                    活跃度 = float(日志潮汐.get("活跃度", 0.0) or 0.0)
                    Δ唤醒 = max(0.0, 用户当前.arousal - float(轨迹均值.get("arousal", 用户当前.arousal)))
                except Exception:  # noqa: BLE001 —— 决策失败则直接使用用户状态
                    目标状态 = 用户当前
                    α潮汐 = 0.15
                    密度潮汐 = self.密度基
                    活跃度 = 1.0
            else:
                # 简易兜底：直接以用户状态为目标
                目标状态 = 用户当前
                α潮汐 = 0.15
                密度潮汐 = self.密度基
                try:
                    活跃度 = self.感知器.轨迹活跃度(self.活跃度窗口)
                except Exception:  # noqa: BLE001
                    活跃度 = 1.0
            v_潮汐 = self.VAD到锚点(目标状态 if 目标状态 is not None else 用户当前)
            if self.潮汐决策器 is None:
                Δ唤醒 = 0.0
        else:
            密度潮汐 = self.密度基

        # ── ③ 思考链源 ──
        v_思考链 = np.zeros(K)
        if 思考链文本 and 思考链文本.strip():
            v_思考链 = self.思考链到锚点(思考链文本)

        # ── ④ 指令源（one-hot + 邻域平滑）──
        v_指令 = np.zeros(K)
        if 指令 and 指令.strip():
            v_指令 = self._指令到锚点(指令)

        # ── ⑤ 角色感知融合（SubTask 6b）：v_target = 0.7·v_角色 + 0.3·v_用户 ──
        角色基调解析 = None
        if 角色 is not None:
            角色基调解析 = self._解析角色基调(角色)
            if 角色基调解析 is None:
                说明 = f"角色[{角色}]未命中角色锚点基调表 → 跳过角色感知（按原逻辑）"
        if 角色基调解析 is not None:
            v_角色, 角色唤醒, 角色名 = 角色基调解析
            w = self.来源权重
            v_用户 = (w.get("潮汐", 0.5) * v_潮汐
                      + w.get("思考链", 0.3) * v_思考链
                      + w.get("指令", 0.2) * v_指令)
            范数用户 = float(np.linalg.norm(v_用户))
            if 范数用户 < 1e-9:   # 无用户情感输入 → 用户部分退回陪伴基调「温柔」
                v_用户 = self._默认温柔(K)
                范数用户 = float(np.linalg.norm(v_用户))
            else:
                v_用户 = v_用户 / (范数用户 + 1e-9)
            v_target = 0.7 * v_角色 + 0.3 * v_用户
            范数 = float(np.linalg.norm(v_target))
            v_target = v_target / (范数 + 1e-9)
            目标强度 = abs(目标状态.arousal) if 目标状态 is not None else 0.0
            目标强度 = max(目标强度, 角色唤醒)
            说明 = (f"角色感知锚点[{角色名}]：v_target=0.7·角色基调+0.3·用户情感"
                    + ("+思考链" if v_思考链.any() else "") + ("+指令" if v_指令.any() else ""))
        else:
            # ── ⑤′ 原多源融合 + L2 归一化（无角色时行为完全一致）──
            w = self.来源权重
            v_target = (w.get("潮汐", 0.5) * v_潮汐
                        + w.get("思考链", 0.3) * v_思考链
                        + w.get("指令", 0.2) * v_指令)
            范数 = float(np.linalg.norm(v_target))
            if 范数 < 1e-9:
                # 无任何输入情感 → 默认陪伴基调「温柔」（one-hot + 邻域平滑）
                v_target = self._默认温柔(K)
                范数 = float(np.linalg.norm(v_target))
                目标强度 = 0.0
                说明 = "无输入情感 → 默认陪伴基调「温柔」"
            else:
                v_target = v_target / (范数 + 1e-9)
                目标强度 = abs(目标状态.arousal) if 目标状态 is not None else float(np.max(np.abs(v_target)))
                说明 = "多源融合" + ("+思考链" if v_思考链.any() else "") + ("+指令" if v_指令.any() else "")

        # ── ⑥ β 自适应：β = β基 × (1+Δarousal) × min(1, 活跃度×2.5)，封顶 β上限 ──
        β = self.β基 * (1.0 + Δ唤醒) * min(1.0, max(活跃度, 目标强度) * 2.5)
        β = min(self.β上限, β)

        # ── ⑦ 情感词密度目标（复用 潮汐决策器 的密度公式）──
        密度 = min(self.密度上限, self.密度基 + self.密度增益 * 目标强度)
        if 用户当前 is not None and self.潮汐决策器 is not None and 密度潮汐 > 0:
            密度 = min(self.密度上限, 密度潮汐)

        日志 = {
            "用户当前VAD": self._状态转dict(用户当前),
            "目标状态VAD": self._状态转dict(目标状态),
            "潮汐α": round(α潮汐, 4),
            "目标V": round(目标状态.valence, 4) if 目标状态 is not None else 0.0,
            "目标强度": round(目标强度, 4),
            "活跃度": round(活跃度, 4),
            "Δ唤醒": round(Δ唤醒, 4),
            "β基": self.β基,
            "β上限": self.β上限,
            "来源权重": dict(w),
            "角色感知": 角色名 if 角色基调解析 is not None else None,
            "轮次": 轮次,
            "简易模式": self._简易模式,
        }
        return 锚点目标(v_target=v_target.astype(np.float32), β=round(β, 4),
                        情感词密度目标=round(密度, 4), 说明=说明,
                        活跃度=round(活跃度, 4), 决策日志=日志)

    def _解析角色基调(self, 角色):
        """角色 → (v_角色单位向量, 角色唤醒度, 角色名)；无法映射返回 None。

        角色为 str 时查 角色锚点基调表（精确→别名→子串双向匹配）；
        角色为 具 valence/arousal/dominance 的对象时用其 VAD 映射到锚点空间；
        锚点向量优先用表内「维度权重字典」直接对齐 self.维度，否则 VAD→锚点。
        """
        K = len(self.维度)
        权重dict = None
        VAD = None
        角色名 = None
        if isinstance(角色, str):
            基调 = 匹配角色基调(角色)
            if 基调 is None:
                return None
            权重dict, VAD = 基调
            角色名 = 角色
        elif hasattr(角色, "valence") and hasattr(角色, "arousal") and hasattr(角色, "dominance"):
            VAD = (float(角色.valence), float(角色.arousal), float(角色.dominance))
            角色名 = getattr(角色, "名称", None) or str(角色)
        else:
            return None
        v = np.zeros(K)
        if 权重dict:
            for i, 维 in enumerate(self.维度):
                v[i] = float(权重dict.get(维, 0.0))
        范数 = float(np.linalg.norm(v))
        if 范数 < 1e-9 and VAD is not None:
            v = self.VAD到锚点(np.asarray(VAD, dtype=float))
            范数 = float(np.linalg.norm(v))
        if 范数 < 1e-9:
            return None
        v = v / (范数 + 1e-9)
        角色唤醒 = VAD[1] if VAD is not None else float(np.max(np.abs(v)))
        return v, 角色唤醒, 角色名

    @staticmethod
    def _简易补测(文本: str):
        """内置词表子串匹配补测（SubTask 6b 增强）：cnsenti/jieba 未命中
        （如单字"累""哭"等口语词）时，直接扫描 简易正面词/简易负面词 子串。

        返回 简易情感状态（与 潮汐感知器.情感状态 同接口：valence/arousal/
        dominance + 到向量/强度/极性）；未命中返回 None。
        """
        if not 文本 or not 文本.strip():
            return None
        pos = neg = 0.0
        for _词 in 简易正面词:
            if _词 in 文本:
                pos += 1.0
        for _词 in 简易负面词:
            if _词 in 文本:
                neg += 1.0
        if pos + neg == 0:
            return None
        中文数 = max(sum(1 for c in 文本 if "\u4e00" <= c <= "\u9fff"), 1)
        词数 = max(中文数 / 2.0, 1.0)
        valence = math.tanh((pos - neg) / 词数 * 2.0)
        arousal = min(1.0, (pos + neg) / 词数)
        dominance = min(1.0, abs(valence) * 1.2)
        return 简易情感状态(valence, arousal, dominance)

    # ──────────────────────────────────────────────
    # VAD → 锚点 映射
    # ──────────────────────────────────────────────

    def VAD到锚点(self, V) -> np.ndarray:
        """VAD → 锚点权重 v ∈ R^K。

        g_k = (V·原型_k) / (‖V‖·‖原型_k‖ + ε)              方向余弦
        v_k = 原型强度_k · g_k · |g_k|^p                   平方放大（保持极性）
              × exp(-λ·|V.arousal - 原型_k.arousal|)      唤醒接近度因子
        """
        K = len(self.维度)
        if V is None:
            return np.zeros(K)
        if hasattr(V, "valence"):
            向量 = np.array([V.valence, V.arousal, V.dominance], dtype=float)
        else:
            向量 = np.asarray(V, dtype=float).reshape(-1)
        if 向量.size != 3:
            raise ValueError(f"VAD 向量必须为 3 维，收到 {向量.size} 维")
        w_v, w_a, w_d = self.分量权重
        加权V = np.array([向量[0] * w_v, 向量[1] * w_a, 向量[2] * w_d], dtype=float)
        范数V = float(np.linalg.norm(加权V)) + 1e-9

        v = np.zeros(K)
        for k, 维 in enumerate(self.维度):
            原型 = self.原型表.get(维, (0.0, 0.0, 0.3))
            强度 = 1.0
            if len(原型) >= 4:  # 支持 (v,a,d) 或 (v,a,d,原型强度)
                强度 = float(原型[3])
            原型v = np.array([原型[0] * w_v, 原型[1] * w_a, 原型[2] * w_d], dtype=float)
            范数原型 = float(np.linalg.norm(原型v)) + 1e-9
            g = float(加权V @ 原型v) / (范数V * 范数原型)          # ∈ [-1, 1]
            接近度 = math.exp(-self.唤醒权重 * abs(向量[1] - 原型[1]))
            v[k] = 强度 * g * abs(g) ** self.锐化指数 * 接近度
        return v

    def 思考链到锚点(self, 思考链文本: str) -> np.ndarray:
        """思考链文本 → VAD → 锚点权重（同一 VAD→锚点 映射）"""
        状态, _ = self.感知器.测量(思考链文本)
        return self.VAD到锚点(状态)

    def _指令到锚点(self, 指令: str) -> np.ndarray:
        """显式情感指令 → 命中维度 → one-hot + 邻域平滑（±1 邻域 ×0.3）→ L2 归一化"""
        K = len(self.维度)
        v = np.zeros(K)
        命中 = None
        for i, 维 in enumerate(self.维度):
            if 维 in 指令:
                命中 = i
                break
        if 命中 is None and self.锚点库 is not None:
            for i, 维 in enumerate(self.维度):
                for 词 in self.锚点库.词集.get(维, []):
                    if 词 and 词 in 指令:
                        命中 = i
                        break
                if 命中 is not None:
                    break
        if 命中 is None:
            return v
        v[命中] = 1.0
        for 偏移 in (-1, 1):  # 邻域平滑
            j = 命中 + 偏移
            if 0 <= j < K:
                v[j] = 0.3
        范数 = float(np.linalg.norm(v))
        return v / (范数 + 1e-9)

    @staticmethod
    def _默认温柔(K: int) -> np.ndarray:
        """无输入情感的默认陪伴基调：温柔 one-hot + 邻域平滑"""
        v = np.zeros(K)
        v[0] = 1.0  # 维度名[0] = 温柔（默认词集首维）
        if K > 1:
            v[1] = 0.3
        范数 = float(np.linalg.norm(v))
        return v / (范数 + 1e-9)

    @staticmethod
    def _状态转dict(状态):
        if 状态 is None:
            return None
        return {"valence": round(float(状态.valence), 3),
                "arousal": round(float(状态.arousal), 3),
                "dominance": round(float(状态.dominance), 3)}

    def __repr__(self):
        return (f"目标决策器(维度={len(self.维度)}, β基={self.β基}, "
                f"简易模式={self._简易模式})")


if __name__ == "__main__":
    # 自测：VAD→锚点 方向性 + 指令命中（无需模型）
    print("=== 目标决策器自测 ===")
    class 空锚点库:
        维度名 = lambda self: ["温柔", "开心", "难过", "愤怒", "害怕", "平静"]  # noqa: E731
        词集 = {}

    决策 = 目标决策器(锚点库=空锚点库())
    悲伤 = 决策.VAD到锚点(简易情感状态(-0.6, 0.3, 0.2))
    print("悲伤VAD →", {决策.维度[i]: round(float(x), 3) for i, x in enumerate(悲伤)})
    目标 = 决策.计算目标(指令="温柔陪伴")
    print("指令温柔陪伴 →", 目标)

    # ── 角色感知自测（SubTask 6b）：不同角色 → 不同主导维度 ──
    print("\n=== 角色感知自测（角色基调主导 v_target 方向）===")
    for 角色名 in ["毒舌但心软的损友", "爱撒娇的小女儿", "温柔治愈系女友",
                   "理性冷静的职场前辈", "幽默的脱口秀演员", "无此角色XYZ"]:
        目标2 = 决策.计算目标(用户当前="我今天真的好累", 角色=角色名, 轮次=2)
        主维 = 决策.维度[int(np.argmax(目标2.v_target))]
        print(f"  [{角色名}] 主分量={主维} β={目标2.β:.3f} | {目标2.说明}")
