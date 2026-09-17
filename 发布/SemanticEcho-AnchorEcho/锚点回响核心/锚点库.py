# -*- coding: utf-8 -*-
"""P4 锚点回响（Anchor Echo）—— 锚点库模块（Task1 交付物，Task3 扩展）

核心灵感：把模型自身的 token-embedding 权重 W_e 当作只读接口（绝不写原权重，
也不影响推理），对情感锚点词集逐词取 token 嵌入均值、再按维度求均值构造
多维锚点质心 E ∈ R^{K×d}，从而对任意 token/词做稠密余弦相似度情感打分。

Task3 扩展（按 P4_混合方案设计.md）：
- 词集从每维 10 词扩至 50 词（纳入网络流行语：破防/泪目/内耗/扎心/心累/宠溺 等）；
- 新增 预计算打分表()：S ∈ R^{V×K}（fp16，V=vocab_size，K=6，约 1.8MB），
  支持保存/加载到 锚点表.pt（首次构建后直接加载，避免重复计算）；
- 新增 max_pooling 聚合选项：词向量 = 该词所有 token 向量逐元素 max
  （作为 得分() 的备选聚合方式，缓解多 token 中文词均值稀释）；
- 保持：只读接口、构建()/维度名()/得分(token_ids) 原接口不变（兼容 Task1 用法）。

两种构造模式（Task1 中对比方向分离度后选定主模式）：
- 原始：直接用 embedding 向量；
- 层归一化：先把每个 token 向量减均值、除标准差（层归一化），再取均值。

全部计算都基于 W_e 的 fp32 只读副本；原始权重张量仅被读取（可用 sum 与
data_ptr 断言其完全未被修改）。
"""
import os
import torch


# 内置默认 6 维锚点词集（每维 50 个中文种子词，含网络流行语）
默认词集 = {
    "温柔": [
        "温柔", "体贴", "呵护", "温暖", "轻声", "安慰", "心疼", "拥抱", "抚摸", "软软",
        "细腻", "关怀", "温存", "抚慰", "软语", "贴心", "暖心", "亲昵", "宠溺", "疼惜",
        "怜惜", "温软", "柔和", "安抚", "温情", "温馨", "柔软", "深情", "治愈", "和蔼",
        "亲切", "宽容", "包容", "耐心", "柔声", "暖暖", "软和", "蜜语", "甜言", "绵绵",
        "柔情", "温婉", "暖意", "温柔乡", "软绵绵", "轻抚", "柔顺", "顺毛", "轻声细语", "暖人心",
    ],
    "开心": [
        "开心", "高兴", "快乐", "愉快", "幸福", "喜悦", "兴奋", "灿烂", "欢笑", "甜蜜",
        "欢乐", "欣喜", "欢快", "雀跃", "愉悦", "满足", "得意", "畅快", "喜乐", "开怀",
        "欢喜", "美滋滋", "心花怒放", "眉开眼笑", "喜气洋洋", "活蹦乱跳", "喜滋滋", "甜滋滋",
        "乐开怀", "舒心", "哈哈大笑", "呵呵", "哈哈", "嘻嘻", "蹦蹦跳跳", "元气满满",
        "神采奕奕", "兴冲冲", "欢天喜地", "喜出望外", "如获至宝", "欣悦", "欢腾", "乐呵",
        "笑逐颜开", "喜上眉梢", "甜蜜蜜", "笑嘻嘻", "兴高采烈", "手舞足蹈",
    ],
    "难过": [
        "难过", "悲伤", "伤心", "痛苦", "失落", "心碎", "沮丧", "哭泣", "委屈", "心疼",
        "悲痛", "哀伤", "忧伤", "凄惨", "苦闷", "消沉", "颓丧", "绝望", "心酸", "伤感",
        "忧愁", "愁苦", "悲凉", "凄楚", "呜咽", "哽咽", "泪目", "崩溃", "心累", "扎心",
        "孤独", "落寞", "凄然", "黯然", "垂头丧气", "郁郁寡欢", "愁眉苦脸", "泪流满面",
        "泣不成声", "心如刀割", "撕心裂肺", "悲恸", "苦楚", "怅然", "寂寥", "空虚", "颓废",
        "内耗", "破防", "惆怅",
    ],
    "愤怒": [
        "愤怒", "生气", "恼火", "怨恨", "火大", "气死", "抓狂", "暴怒", "怒火", "憋屈",
        "气愤", "恼怒", "震怒", "狂怒", "愤懑", "怨气", "憎恨", "怀恨", "记恨", "翻脸",
        "发飙", "怒斥", "咆哮", "发火", "动怒", "盛怒", "雷霆", "恼羞成怒", "气急败坏",
        "火冒三丈", "怒不可遏", "咬牙切齿", "怒发冲冠", "暴躁", "易怒", "愤愤不平", "怒目",
        "火气", "怨念", "敌意", "痛恨", "不满", "窝火", "气恼", "气炸", "炸毛", "怒容",
        "怒气冲冲", "怒喊", "吼",
    ],
    "害怕": [
        "害怕", "恐惧", "担心", "焦虑", "不安", "恐慌", "紧张", "惊慌", "畏惧", "忐忑",
        "惧怕", "惊骇", "惊惶", "惶恐", "发怵", "心悸", "慌张", "忧心", "忧虑", "心神不宁",
        "坐立不安", "提心吊胆", "心惊胆战", "胆战心惊", "毛骨悚然", "不寒而栗", "惴惴不安",
        "惶恐不安", "胆怯", "怯懦", "畏缩", "发抖", "颤抖", "哆嗦", "心惊", "后怕", "忧惧",
        "忧心忡忡", "紧张兮兮", "如坐针毡", "诚惶诚恐", "心惊肉跳", "吓坏", "惊吓", "受惊",
        "悚然", "惊魂", "惊惧", "胆战", "慌张张",
    ],
    "平静": [
        "平静", "淡然", "平和", "冷静", "安稳", "坦然", "释然", "从容", "宁静", "恬淡",
        "安详", "沉稳", "沉着", "淡定", "心平气和", "气定神闲", "波澜不惊", "泰然自若",
        "处变不惊", "云淡风轻", "悠然", "闲适", "安闲", "恬静", "静谧", "安逸", "自在",
        "稳如泰山", "心如止水", "静心", "静默", "温和", "舒坦", "放松", "舒缓", "平缓",
        "安然", "从容不迫", "心静", "淡定自若", "安然自若", "闲庭信步", "不慌不忙", "稳当",
        "踏实", "安定", "静水流深", "淡泊", "平心静气", "随和",
    ],
}


class 锚点库:
    """用模型 token-embedding 权重（只读）构造情感锚点质心并做稠密情感打分。"""

    def __init__(self, model, tokenizer, 词集=None, 模式="原始", 打分模式="均值"):
        """model/tokenizer 为已加载的模型与分词器；词集为 {维度名: [种子词]}；
        模式取 '原始' 或 '层归一化'；
        打分模式取 '均值'（默认，token 向量均值后余弦）或 'max池化'
        （词向量 = 该词所有 token 向量逐元素 max，缓解弱语义词被均值稀释）。"""
        if 模式 not in ("原始", "层归一化"):
            raise ValueError(f"未知模式：{模式}，应为 '原始' 或 '层归一化'")
        if 打分模式 not in ("均值", "max池化"):
            raise ValueError(f"未知打分模式：{打分模式}，应为 '均值' 或 'max池化'")
        self.model = model
        self.tokenizer = tokenizer
        self.词集 = 词集 if 词集 is not None else 默认词集
        self.模式 = 模式
        self.打分模式 = 打分模式
        # 只读接口：仅保存对原始 embedding 权重的引用，绝不对它做任何写操作
        self.权重 = model.get_input_embeddings().weight   # 原始张量（只读引用，不动）
        self.W_e = self.权重.detach().float()            # fp32 只读副本，全部计算基于它
        self._有效权重 = None                             # 构建时按模式确定
        self.锚点矩阵 = None                              # (K, d) 每行已 L2 归一化
        self.打分表 = None                                # (V, K) fp16 预计算打分表
        self._词向量表 = {}                               # 单词向量缓存

    def 维度名(self):
        """返回锚点维度名列表。"""
        return list(self.词集.keys())

    def 构建(self):
        """计算 K 维情感锚点质心 E ∈ R^{K×d}（每个质心已 L2 归一化）。

        每维质心 = 该维度所有种子词的向量均值，再 L2 归一化；
        词向量 = 该词各 token 嵌入的均值（中文 BPE 多 token 词取均值），
        打分模式为 'max池化' 时词向量 = 该词各 token 向量逐元素 max。
        """
        if self.模式 == "层归一化":
            # 每个 token 向量减均值、除标准差（层归一化），生成新张量，不动原权重
            μ = self.W_e.mean(dim=-1, keepdim=True)
            σ = self.W_e.std(dim=-1, keepdim=True)
            self._有效权重 = (self.W_e - μ) / (σ + 1e-8)
        else:
            self._有效权重 = self.W_e
        锚点列表 = []
        for 维 in self.维度名():
            词向量列表 = []
            for 词 in self.词集[维]:
                v = self._词向量(词)
                if v is not None:
                    词向量列表.append(v)
            if not 词向量列表:
                raise ValueError(f"维度「{维}」无任何有效词向量")
            质心 = torch.stack(词向量列表).mean(dim=0)
            锚点列表.append(质心 / (质心.norm() + 1e-8))
        self.锚点矩阵 = torch.stack(锚点列表)             # (K, d)，每行单位向量
        return self.锚点矩阵

    def _词向量(self, 词):
        """单词的嵌入向量：均值模式取 token 向量均值；max池化模式取逐元素 max。
        无法编码返回 None。"""
        if 词 in self._词向量表:
            return self._词向量表[词]
        ids = self.tokenizer.encode(词, add_special_tokens=False)
        if not ids:
            return None
        ids = torch.tensor(ids, dtype=torch.long, device=self._有效权重.device)
        向量 = self._有效权重[ids]
        if self.打分模式 == "max池化":
            v = 向量.max(dim=0).values
        else:
            v = 向量.mean(dim=0)
        self._词向量表[词] = v
        return v

    def 词向量(self, 词):
        """返回单词的嵌入向量（fp32, d 维 numpy）；无法编码返回 None。"""
        v = self._词向量(词)
        return None if v is None else v.detach().cpu().numpy()

    def 得分(self, token_ids, 打分模式=None):
        """对一批 token id 计算稠密情感特征 f(w) ∈ R^{N×K}（每个 token 一行）。

        即每个 token 向量与各锚点的余弦相似度。输入可以是 list[int] / torch
        张量 / str（str 自动 encode）；输出 numpy (N, K)。

        Task3 扩展：打分模式='max池化' 且输入为 str 词时，按「该词各 token
        向量逐元素 max → 与各锚点余弦」计算，返回 (K,)（词级聚合，替代 token
        级均值，缓解弱语义词被均值稀释）。
        """
        if 打分模式 is None:
            打分模式 = self.打分模式
        if self.锚点矩阵 is None:
            raise RuntimeError("请先调用 构建()")
        if isinstance(token_ids, str) and 打分模式 == "max池化":
            v = self._词向量(token_ids)
            if v is None:
                raise ValueError(f"词「{token_ids}」无法编码")
            v = v / (v.norm() + 1e-8)
            A = self.锚点矩阵 / (self.锚点矩阵.norm(dim=-1, keepdim=True) + 1e-8)
            return (v @ A.T).detach().cpu().numpy()
        if isinstance(token_ids, str):
            token_ids = self.tokenizer.encode(token_ids, add_special_tokens=False)
        ids = torch.as_tensor(list(token_ids), dtype=torch.long,
                              device=self._有效权重.device)
        if ids.numel() == 0:
            raise ValueError("空输入，无法打分")
        V = self._有效权重[ids]
        V = V / (V.norm(dim=-1, keepdim=True) + 1e-8)
        A = self.锚点矩阵 / (self.锚点矩阵.norm(dim=-1, keepdim=True) + 1e-8)
        return (V @ A.T).detach().cpu().numpy()

    def 词得分(self, 词):
        """单词的稠密情感特征 f ∈ R^K：词的聚合向量（均值或 max 池化，按打分模式）
        与各锚点的余弦相似度。"""
        if self.锚点矩阵 is None:
            raise RuntimeError("请先调用 构建()")
        v = self._词向量(词)
        if v is None:
            return None
        v = v / (v.norm() + 1e-8)
        A = self.锚点矩阵 / (self.锚点矩阵.norm(dim=-1, keepdim=True) + 1e-8)
        return (v @ A.T).detach().cpu().numpy()

    def 预计算打分表(self, 缓存路径=None, 强制重算=False):
        """预计算全词表情感打分表 S ∈ R^{V×K}（fp16，V=vocab_size，K=6）。

        S[w, k] = cos(W_e[w], e_k)，即每个 token 向量与各锚点质心的余弦相似度。
        生成循环内仅需一次查表 + 向量乘（S @ v_target），零 hook 开销。

        支持缓存：默认保存/加载到 工作目录/锚点表.pt（max池化模式为 锚点表_max.pt），
        首次构建后直接加载避免重复计算；强制重算可传入 强制重算=True。

        内存：151936 × 6 × 2B ≈ 1.8MB（fp16），16GB 显存/内存完全无压力。
        """
        if self.锚点矩阵 is None:
            self.构建()
        if 缓存路径 is None:
            模式后缀 = "_max" if self.打分模式 == "max池化" else ""
            缓存路径 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    f"锚点表{模式后缀}.pt")
        if os.path.exists(缓存路径) and not 强制重算:
            try:
                S = torch.load(缓存路径, map_location="cpu", weights_only=True).to(
                    self._有效权重.device)
                if (S.shape == (self._有效权重.shape[0], len(self.维度名()))
                        and S.dtype == torch.float16 and torch.isfinite(S).all()):
                    self.打分表 = S
                    return S
            except Exception:
                pass  # 缓存损坏/形状不符 → 重算
        A = self.锚点矩阵 / (self.锚点矩阵.norm(dim=-1, keepdim=True) + 1e-8)
        W = self._有效权重
        Wn = W / (W.norm(dim=-1, keepdim=True) + 1e-8)
        S = (Wn @ A.T).half()                              # (V, K) fp16
        self.打分表 = S
        try:
            torch.save(S.detach().cpu(), 缓存路径)
        except Exception as e:
            print(f"[锚点库] 打分表缓存保存失败（不影响使用）：{e}")
        return S

    def 记录只读基线(self):
        """构建/打分前记录原始 embedding 权重 sum 与 data_ptr（只读证明基线）。"""
        return {"sum": self.权重.sum().item(), "data_ptr": self.权重.data_ptr()}

    def 验证只读(self, 基线=None):
        """验证只读：构建+打分前后原始权重 sum 与 data_ptr 完全一致。"""
        if 基线 is None:
            基线 = self.记录只读基线()
        s = self.权重.sum().item()
        p = self.权重.data_ptr()
        sum一致 = (s == 基线["sum"])
        指针一致 = (p == 基线["data_ptr"])
        assert sum一致 and 指针一致, "只读验证失败：embedding 权重被修改！"
        return {"sum一致": sum一致, "指针一致": 指针一致,
                "sum_before": 基线["sum"], "sum_after": s,
                "data_ptr_before": 基线["data_ptr"], "data_ptr_after": p}
