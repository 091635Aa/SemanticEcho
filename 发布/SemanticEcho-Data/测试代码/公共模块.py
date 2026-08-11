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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
                kwargs.update(load_in_4bit=True)
            else:
                kwargs.update(torch_dtype=torch.float16)
        分词器 = AutoTokenizer.from_pretrained(路径, trust_remote_code=True)
        模型 = AutoModelForCausalLM.from_pretrained(路径, **kwargs)
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


def 加载目标模型(量化=None):
    return 目标槽.加载(目标模型名, 量化=量化)


def 加载裁判模型(尝试4bit=True, 量化=None):
    """加载裁判 7B；默认直接 4bit（稳定省显存），或显式指定 fp16"""
    if 量化 is None:
        # fp16 14GB 与生成模型切换易 OOM，统一用 4bit（约 5GB）
        量化 = "4bit"
    try:
        return 裁判槽.加载(裁判模型名, 量化=量化)
    except torch.cuda.OutOfMemoryError:
        if 尝试4bit and 量化 != "4bit":
            print("[裁判] fp16 OOM，降级 4bit")
            裁判槽.卸载()
            return 裁判槽.加载(裁判模型名, 量化="4bit")
        raise


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
        # fp16 OOM → 降级 4bit 重试一次
        print("[裁判] 生成 OOM，降级 4bit 重试")
        裁判槽.卸载()
        加载裁判模型(量化="4bit")
        return 裁判槽.生成(消息列表, **覆盖)
