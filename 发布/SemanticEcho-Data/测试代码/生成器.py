# -*- coding: utf-8 -*-
"""
生成器 — 裸生成 vs 四层全开 双模式（同一提示词、同一随机种子）
==============================================================
- 裸生成：目标模型 Qwen2.5-1.5B-Instruct 直接 model.generate（无任何模块）
- 四层全开：推理框架.推理框架 四层协同 = 微调基座 + 语义回响(短期记忆)
            + RAG(中期记忆) + LoRA gentle_v2(长期记忆) + 动态策略B + 记忆注入

种子策略：生成前 torch.manual_seed(种子) + 按轮次微调，保证裸组可复现，
四层组内部随机采样受同一种子初始化影响，两组提示词完全一致。
"""
import os
import sys
import torch

本工程目录 = r"f:\最终工程架构"
if 本工程目录 not in sys.path:
    sys.path.insert(0, 本工程目录)

agent_echo目录 = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\agent_echo"
if agent_echo目录 not in sys.path:
    sys.path.insert(0, agent_echo目录)

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
目标模型路径 = os.path.join(模型空间, "Qwen2.5-1.5B-Instruct")
LoRA_1_5B = r"f:\lora外挂\lora_adapters\gentle_v2"

# 1.5B 扫描表最优参数（已验证：λ=0.29 坍缩，0.08 稳定）
推荐参数 = {"λ": 0.08, "γ": 0.07, "τ": 0.09}

# 四层开关：仅语义回响引擎（用户要求不做 RAG + LoRA；记忆注入会污染 prompt，一并关闭）
RAG开启 = False
LoRA开启 = False
动态策略 = "B"
记忆开启 = False  # 关闭超长期记忆注入，保证对照纯净（裸 vs 纯回响引擎）


class 生成器:
    """双模式生成器：同一实例同时持有裸模型与四层框架，避免重复加载"""

    def __init__(self, 模式="全部"):
        self.模式 = 模式
        self.设备 = "cuda" if torch.cuda.is_available() else "cpu"
        self._裸模型 = None
        self._裸分词器 = None
        self._框架 = None
        self._记忆 = None
        self._会话池 = {}  # 会话名 → 跨轮持久回响池（R2 多轮一致性）

    # ── 裸模型 ──
    def _加载裸模型(self):
        if self._裸模型 is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("[生成器] 加载裸模型 1.5B ...")
        self._裸分词器 = AutoTokenizer.from_pretrained(目标模型路径, trust_remote_code=True)
        self._裸模型 = AutoModelForCausalLM.from_pretrained(
            目标模型路径, torch_dtype=torch.float16 if self.设备 == "cuda" else torch.float32,
            trust_remote_code=True).to(self.设备)
        self._裸模型.eval()
        print(f"[生成器] 裸模型加载完成, {self._裸模型.num_parameters()/1e6:.0f}M")

    # ── 四层框架 ──
    def _加载框架(self):
        if self._框架 is not None:
            return
        from 推理框架 import 推理框架
        print("[生成器] 加载四层框架（基座+回响+RAG+LoRA+动态策略B）...")
        self._框架 = 推理框架(
            目标模型路径, 量化=None, rag=RAG开启,
            lora=LoRA_1_5B if LoRA开启 else None,
            动态策略=动态策略, 长上下文=False)
        # 覆盖为扫描表最优参数（推理框架会自动从扫描表取，此处显式确保）
        print(f"[生成器] 四层框架加载完成 | 推荐 λ={self._框架.λ基准} γ={self._框架.γ基准} τ={self._框架.τ基准}")

    # ── 记忆注入 ──
    def _记忆前缀(self, prompt):
        if not 记忆开启:
            return ""
        if self._记忆 is None:
            sys.path.insert(0, os.path.join(本工程目录, "API服务"))
            try:
                from 记忆 import 记忆
                self._记忆 = 记忆
            except Exception as e:
                print(f"[生成器] 记忆系统加载失败: {e}")
                self._记忆 = False
        if not self._记忆:
            return ""
        try:
            return self._记忆.构建前缀(prompt, top_k=3)
        except Exception as e:
            print(f"[生成器] 记忆检索失败: {e}")
            return ""

    # ── 生成入口 ──
    def 裸生成(self, 消息列表, 种子=42, 轮次=0, max_new_tokens=128):
        """裸生成：标准 model.generate"""
        self._加载裸模型()
        torch.manual_seed(种子 + 轮次)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子 + 轮次)
        # 聊天模板
        提示 = self._裸分词器.apply_chat_template(
            消息列表, tokenize=False, add_generation_prompt=True)
        inputs = self._裸分词器(提示, return_tensors="pt").to(self.设备)
        with torch.no_grad():
            out = self._裸模型.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                # 与四层回响引擎采样参数一致，保证对照唯一变量是"是否挂四层模块"
                temperature=1.0, top_p=0.9, top_k=50, do_sample=True,
                repetition_penalty=1.05,
                pad_token_id=self._裸分词器.eos_token_id,
            )
        新token = out[0, inputs.input_ids.shape[1]:]
        return self._裸分词器.decode(新token, skip_special_tokens=True).strip()

    def 四层生成(self, 消息列表, 种子=42, 轮次=0, max_new_tokens=128,
                会话=None, λ覆盖=None, 思考链=False, 模板="chat", 身份注入=True,
                提示词="人类身份"):
        """回响引擎生成（R1 修复）：
        - 模板="chat"：apply_chat_template 渲染完整消息（system+历史轮），
          适用于角色扮演（EmoCharacter）等需要严格角色位的场景。
        - 模板="纯文本"：身份前缀 + 纯文本拼接（不激活 Qwen chat 模板的
          AI 助手行为惯性），适用于"人似度"场景（LLM-Judge/TuringBench）。
        - 身份注入=False：不注入身份提示（保留 Qwen 默认 AI 助手身份），
          用于隔离"身份"变量的对照实验（LLM-Judge）。
        - 提示词="人类身份"：现有身份提示；提示词="图灵测试"：明确告知 AI
          正在参加图灵测试、要以人类方式回应（用户建议，LLM-Judge 验证用）。
        - 会话: 非 None 时跨轮复用持久回响池（R2 多轮一致性）
        - λ覆盖: 任务自适应 λ（R3）；思考链: 启用注入器思考阶段（R4）
        """
        self._加载框架()
        torch.manual_seed(种子 + 轮次)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子 + 轮次)
        人类身份文案 = ("请以真实人类的身份自然地回复对方：语气口语化、像日常聊天一样简短，"
                    "不要提及你是人工智能、助手或模型，不要说教，不要提供服务式回应。")
        图灵测试文案 = ("你在参加一场图灵测试，评委正在判断你的回答出自真人还是 AI。"
                    "请像真实人类一样自然地回复对方：语气口语化、简短，"
                    "不要暴露你是人工智能、助手或模型，不要说教，不要提供服务式回应。")
        当前提示 = 图灵测试文案 if 提示词 == "图灵测试" else 人类身份文案
        if 模板 == "纯文本":
            # 身份前缀 + 纯文本消息拼接。
            # 注意：不拼接"用户：/回复："格式标签——1.5B 会模仿该格式
            # 续写（输出"你：…""答：…"），被图灵检测器秒识别（实测 TuringBench 人似度 0.63→0.23）。
            if 身份注入 and (not 消息列表 or 消息列表[0].get("role") != "system"):
                前缀身份 = 当前提示
            else:
                前缀身份 = ""
            文本段 = [m["content"] for m in 消息列表]
            提示 = ("\n\n".join(x for x in [前缀身份] + 文本段 if x))
        else:
            # R1b/R1c：无 system 时注入身份提示，覆盖 Qwen chat 模板默认的
            # "AI 助手"身份（该身份泄漏会破坏 TuringBench 人似度等指标）。
            # 身份注入=False 时保留默认身份。
            if 身份注入 and (not 消息列表 or 消息列表[0].get("role") != "system"):
                消息列表 = ([{"role": "system", "content": 当前提示 + "不要使用正式书面语。"}]
                           + list(消息列表))
            提示 = self._框架.tokenizer.apply_chat_template(
                消息列表, tokenize=False, add_generation_prompt=True)
        前缀 = self._记忆前缀(提示)
        思考标记对 = ("思考：", "\n回答：") if 思考链 else None
        复用池 = self._会话池.get(会话) if 会话 else None
        结果 = self._框架.生成(
            提示, max_new_tokens=max_new_tokens, 前缀=前缀,
            λ覆盖=λ覆盖, 思考标记对=思考标记对, 复用池=复用池,
            repetition_penalty=1.05)  # 对齐裸模型采样参数
        if 会话:
            self._会话池[会话] = 结果.get("池")
        return 结果.get("文本", "")

    def 生成(self, 模式, 消息列表, 种子=42, 轮次=0, max_new_tokens=128,
             会话=None, λ覆盖=None, 思考链=False, 模板="chat", 身份注入=True, 提示词="人类身份"):
        """统一入口：模式 ∈ 裸|四层；四层支持 会话/λ覆盖/思考链/模板/身份注入/提示词"""
        if 模式 == "裸":
            return self.裸生成(消息列表, 种子, 轮次, max_new_tokens)
        return self.四层生成(消息列表, 种子, 轮次, max_new_tokens,
                             会话=会话, λ覆盖=λ覆盖, 思考链=思考链, 模板=模板,
                             身份注入=身份注入, 提示词=提示词)

    def 清理(self):
        for 名称, 对象 in (("裸模型", self._裸模型), ("框架", self._框架)):
            if 对象 is not None:
                try:
                    del 对象
                except Exception:
                    pass
        self._裸模型 = self._框架 = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 全局单例
生成器实例 = 生成器()
