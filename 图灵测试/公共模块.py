# -*- coding: utf-8 -*-
"""
公共模块 — 模型加载与 LLM 裁判
==============================
- 加载目标模型（Qwen2.5-1.5B-Instruct，生成用）
- 加载裁判模型（Qwen2.5-7B-Instruct，评分用；OOM 时降级 4bit）
- 统一生成接口：temperature=0.7, top_p=0.9, max_new_tokens=128
"""
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
目标模型名 = "Qwen2.5-1.5B-Instruct"
裁判模型名 = "Qwen2.5-7B-Instruct"

生成参数 = dict(temperature=0.7, top_p=0.9, max_new_tokens=128, do_sample=True, repetition_penalty=1.05)


class 模型槽:
    """单模型槽：持有当前加载的 (模型, 分词器, 设备, 名称)"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.名称 = None
        self.量化 = None

    @property
    def 已加载(self):
        return self.model is not None

    def 加载(self, 名称, 量化=None):
        """加载模型；量化='4bit' 时用 bitsandbytes NF4"""
        if self.已加载 and self.名称 == 名称 and self.量化 == 量化:
            return self.model, self.tokenizer
        self.卸载()
        路径 = os.path.join(模型空间, 名称)
        print(f"[加载] {名称} (量化={量化 or 'fp16'}) ...")
        设备 = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = dict(trust_remote_code=True)
        if 设备 == "cuda":
            if 量化 == "4bit":
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                kwargs["device_map"] = "auto"
            else:
                kwargs.update(dtype=torch.float16)
        分词器 = AutoTokenizer.from_pretrained(路径, trust_remote_code=True)
        模型 = AutoModelForCausalLM.from_pretrained(路径, **kwargs)
        if 量化 != "4bit" or 设备 == "cpu":
            模型.to(设备)
        模型.eval()
        self.model, self.tokenizer, self.device = 模型, 分词器, 设备
        self.名称, self.量化 = 名称, 量化
        print(f"[加载] {名称} 完成, {模型.num_parameters()/1e6:.0f}M 参数, 设备={设备}")
        return 模型, 分词器

    def 卸载(self):
        import gc
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.名称 = None
        self.量化 = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def 生成(self, 消息列表, **覆盖):
        """按 chat 模板生成文本；消息列表形如 [{'role':'user','content':...}]"""
        提示 = self.tokenizer.apply_chat_template(
            消息列表, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(提示, return_tensors="pt").to(self.device)
        参数 = dict(生成参数)
        参数.update(覆盖)
        参数.setdefault("pad_token_id", self.tokenizer.eos_token_id)
        out = self.model.generate(**inputs, **参数)
        新token = out[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(新token, skip_special_tokens=True).strip()


# 全局共享槽（每基准脚本独立运行进程，各自加载）
目标槽 = 模型槽()
裁判槽 = 模型槽()


def 加载裁判模型(尝试4bit=True, 量化=None):
    """加载裁判 7B：4bit（BitsAndBytesConfig，需清理统一生成器后加载）。
    加载前先清理统一生成器（释放 1.5B+解码器显存），否则 bnb 4bit 崩溃。"""
    import gc
    # 裁判前清理统一生成器（懒加载，下次生成自动重建）
    try:
        from 统一生成器 import 生成器实例
        生成器实例.清理()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    if 裁判槽.已加载:
        return 裁判槽.model, 裁判槽.tokenizer
    路径 = os.path.join(模型空间, 裁判模型名)
    print(f"[裁判] 加载 {裁判模型名} 4bit ...")
    from transformers import BitsAndBytesConfig
    配置 = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    分词器 = AutoTokenizer.from_pretrained(路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        路径, quantization_config=配置, device_map="auto", trust_remote_code=True)
    模型.eval()
    裁判槽.model, 裁判槽.tokenizer, 裁判槽.device = 模型, 分词器, "cuda"
    裁判槽.名称, 裁判槽.量化 = 裁判模型名, "4bit"
    print("[裁判] 加载完成")
    return 裁判槽.model, 裁判槽.tokenizer


def 目标生成(消息列表, **覆盖):
    """确保目标模型已加载后生成"""
    if not 目标槽.已加载:
        加载目标模型()
    return 目标槽.生成(消息列表, **覆盖)


def 裁判生成(消息列表, **覆盖):
    """确保裁判模型已加载后生成（温度默认更低以保证评分稳定）"""
    if not 裁判槽.已加载:
        加载裁判模型()
    覆盖.setdefault("temperature", 0.2)
    覆盖.setdefault("max_new_tokens", 256)
    try:
        return 裁判槽.生成(消息列表, **覆盖)
    except torch.cuda.OutOfMemoryError:
        # 生成 OOM → 重试一次（清显存）
        print("[裁判] 生成 OOM，清显存重试")
        torch.cuda.empty_cache()
        return 裁判槽.生成(消息列表, **覆盖)
