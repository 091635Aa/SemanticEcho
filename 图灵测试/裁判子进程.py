# -*- coding: utf-8 -*-
"""通用裁判子进程 —— 独立进程加载 7B 4bit，避免与 1.5B 生成器同进程（bnb bug）

用法: python 裁判子进程.py <任务.json> <输出.json>
任务.json: {"裁判": "heartbench|feel|emocharacter|llm_judge|情感", "请求": [...]}
"""
import os
import sys
import json
import re
import gc

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)
if os.path.join(r"i:\Desktop\语义回响", "图灵测试") not in sys.path:
    sys.path.insert(0, os.path.join(r"i:\Desktop\语义回响", "图灵测试"))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from safetensors import safe_open

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
裁判模型名 = os.environ.get("裁判模型", "Qwen2.5-7B-Instruct")

# ── HeartBench rubric 裁判提示 ──
评分提示模板 = """你是一个专业、严苛的心理学打分批评专家，请根据以上对话历史（作为考题）与待测AI的表现进行打分。
注意事项：
1.考题中的角色B不是待测AI，仅辅助生成考题上文。仅「待测AI回复」中才是你要考察的部分。
2.打分要尽可能严格，加分项只有完全达到才能命中，扣分项如果有一点涉及都要命中。
3.打分时通读对话历史、待测ai恢复和rubric，逐条根据rubric给出命中细节，并计算rubric条数，
最后命中细节数量必须和rubric条数一致，这个原则你需要反复验证，非常重要。
4.不必在意分值多少，不需要计算总分。

对话历史：
{dialogue_history}
待测AI回复：
{response}
rubric：
{rubric}
rubric条数：
{rubric_nums}

直接按照格以下式输出，不要输出markdown或者其他内容。
输出格式：
{{
  'reason'（String）:（说出具体的评分过程，除了json结构中，文本内容里不要用引号防止解析失败，用「」）
  'detail'（array of Integer）:[1,0,1,....](按顺序给出rubric命中情况，命中置1，未命中置0，用英文逗号分割，这里的元素个数一定要与rubric条数保持一致)
}}"""

# ── FEEL 共情评分 ──
共情评分提示 = """你是一位心理学共情评估专家。
场景与决策如下：
{context}

请评估该行为决策的共情合理性（是否理解他人情绪、是否有人情味）：
必须严格按 JSON 输出：{{"共情合理性": 0到1的小数, "reason": "一句话理由"}}"""

# ── EmoCharacter 裁判提示 ──
共情裁判提示 = """你是一位角色扮演情感保真度评估专家。
角色设定：{角色}（情感基调：{基调}）
用户说：{用户话}
角色的回复：{回复}

请评估该回复的情感保真度：回复的情绪与语气是否符合角色设定与情感基调？
必须严格按 JSON 输出：
{{"情感保真度": 0到1的小数, "reason": "一句话理由"}}"""

一致性裁判提示 = """你是一位角色扮演跨轮一致性评估专家。
角色设定：{角色}（情感基调：{基调}）
以下是该角色在连续多轮对话中的全部回复：
{全部回复}

请评估：这些回复在情绪基调上是否保持稳定一致（没有突然跳戏/情绪漂移）？
必须严格按 JSON 输出：
{{"一致性": 0到1的小数, "reason": "一句话理由"}}"""

强制选择一致性提示 = """你是一位角色扮演跨轮一致性评估专家。
角色设定：{角色}（情感基调：{基调}）

以下是两个候选的"连续多轮回复集合"。其中一个集合来自同一角色在连续对话中的回复；
另一个集合是把多个不同角色（情绪基调各不相同）的回复混在一起的产物。

集合A：
{集合A}

集合B：
{集合B}

请判断：哪一个集合更像是同一角色在连续多轮对话中保持稳定情绪基调的回复？
必须严格按 JSON 输出：
{{"更像同一角色": "A"或"B", "reason": "一句话理由"}}"""

# ── 情感理解裁判提示 ──
推理评分提示 = """你是情感推理质量评估专家。
情境：「{情境}」
模型给出的情绪推理：
{推理}
请评估该推理的质量（触发因素是否具体、合理、贴合情境）：
必须严格按 JSON 输出：{{"推理质量": 0到1的小数, "reason": "一句话理由"}}"""

四分支提示 = """你是一位情感智力（Emotional Intelligence）评估专家，依据 Mayer-Salovey-Caruso 四分支模型评估以下回复：
用户说：「{user}」
AI 回复：「{reply}」

请对回复在以下四个维度各打 1-5 分（1=差，5=优秀）：
1. 情绪感知（Perceiving）：是否准确识别用户的情绪状态
2. 情绪理解（Understanding）：是否理解情绪产生的原因与关系
3. 情绪促进思维（Facilitating Thought）：是否帮助用户以更好的方式思考
4. 情绪管理（Managing）：是否有效调节和改善用户情绪
必须严格按 JSON 输出：
{{"情绪感知": 1到5的整数, "情绪理解": 1到5的整数, "情绪促进思维": 1到5的整数, "情绪管理": 1到5的整数}}"""

LLM裁判配对提示 = """你是一位严格的中文图灵测试裁判。用户说了一句话，下面有两个回复：回复A 和 回复B。

用户：{user}
回复A：{回复A}
回复B：{回复B}

其中一个回复是真人写的，另一个是 AI 生成的。
请判断：哪个回复更可能出自真人之手？

必须严格按以下 JSON 格式输出（不要输出其他内容）：
{{"更像真人": "A" 或 "B", "理由": "一句话理由"}}"""

LLM裁判评分提示 = """你是一位严格的中文图灵测试裁判。用户说了一句话，下面是某个人的回复。

用户：{user}
回复：{回复}

请给这个回复的"像真人程度"打分：1 到 5 分（1=一眼就是 AI，5=完全像真人）。
必须严格按以下 JSON 格式输出：
{{"像真人程度": 1到5的整数, "理由": "一句话理由"}}"""


def 加载4bit():
    gc.collect()
    torch.cuda.empty_cache()
    路径 = os.path.join(模型空间, 裁判模型名)
    print(f"[裁判子进程] 加载 {裁判模型名} 4bit ...", flush=True)
    配置 = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    分词器 = AutoTokenizer.from_pretrained(路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        路径, quantization_config=配置, device_map={"": 0}, trust_remote_code=True,
        low_cpu_mem_usage=True)
    模型.eval()
    return 模型, 分词器


def 加载bf16():
    """bf16 手动分片加载（不依赖 bnb，14GB GPU；之前 P1_5统一裁判.py 验证可靠）"""
    import glob
    gc.collect()
    torch.cuda.empty_cache()
    路径 = os.path.join(模型空间, 裁判模型名)
    print(f"[裁判子进程] 加载 {裁判模型名} bf16（手动分片）...", flush=True)
    cfg = AutoConfig.from_pretrained(路径, trust_remote_code=True)
    with torch.device("meta"):
        模型 = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    模型 = 模型.to_empty(device="cuda")
    for _分片 in sorted(glob.glob(os.path.join(路径, "model-*.safetensors"))):
        with safe_open(_分片, framework="pt", device="cpu") as f:
            _sd = {k: f.get_tensor(k) for k in f.keys()}
        模型.load_state_dict(_sd, strict=False)
        del _sd
        gc.collect()
    _base = getattr(cfg, "rope_theta", 1000000.0)
    _头维 = cfg.hidden_size // cfg.num_attention_heads
    _inv = 1.0 / (_base ** (torch.arange(0, _头维, 2, dtype=torch.int64).float() / _头维))
    _inv = _inv.to(torch.float32)
    for _模块 in 模型.modules():
        if hasattr(_模块, "inv_freq") and _模块.inv_freq is not None:
            _模块.inv_freq.copy_(_inv)
            if hasattr(_模块, "original_inv_freq") and _模块.original_inv_freq is not None:
                _模块.original_inv_freq.copy_(_inv)
    torch.cuda.empty_cache()
    模型.eval()
    分词器 = AutoTokenizer.from_pretrained(路径, trust_remote_code=True)
    return 模型, 分词器


def 加载fp16():
    """小裁判（如 3B）直接 fp16 加载（规避 bnb 在 Windows 推理中段崩溃）"""
    gc.collect()
    torch.cuda.empty_cache()
    路径 = os.path.join(模型空间, 裁判模型名)
    print(f"[裁判子进程] 加载 {裁判模型名} fp16 ...", flush=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        路径, torch_dtype=torch.float16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map={"": 0})
    模型.eval()
    分词器 = AutoTokenizer.from_pretrained(路径, trust_remote_code=True)
    return 模型, 分词器


def 生成(模型, 分词器, 消息, max_new_tokens=256, temperature=0.2):
    提示 = 分词器.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = 分词器(提示, return_tensors="pt").to(模型.device)
    with torch.no_grad():
        out = 模型.generate(
            inputs.input_ids, max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0), temperature=temperature,
            pad_token_id=分词器.eos_token_id)
    新 = out[0, inputs.input_ids.shape[1]:]
    return 分词器.decode(新, skip_special_tokens=True).strip()


def 解析配对(文本):
    m = re.search(r'"更像真人"\s*[:：]\s*"([AB])"', 文本)
    if m:
        return m.group(1)
    if "回复A" in 文本 and "回复B" not in 文本.split("更像真人")[-1][:40]:
        return "A"
    if "回复B" in 文本 and "回复A" not in 文本.split("更像真人")[-1][:40]:
        return "B"
    return None


def 解析评分(文本):
    m = re.search(r'"像真人程度"\s*[:：]\s*([1-5])', 文本)
    if m:
        return int(m.group(1))
    m2 = re.search(r'([1-5])\s*分', 文本)
    return int(m2.group(1)) if m2 else None


def 提取detail(文本):
    for 模式 in (r"'detail'\s*[:：]\s*\[([^\]]*)\]",
                 r'"detail"\s*[:：]\s*\[([^\]]*)\]',
                 r'detail\s*[:：]\s*\[([^\]]*)\]'):
        m = re.search(模式, 文本)
        if m:
            return [int(x.strip()) for x in m.group(1).split(",") if x.strip() in ("0", "1")]
    arrays = re.findall(r"\[([0-9,\s]+)\]", 文本)
    if not arrays:
        return None
    return [int(x.strip()) for x in arrays[-1].split(",") if x.strip() in ("0", "1")]


def 解析小数(文本, 键):
    m = re.search(rf'"{键}"\s*[:：]\s*([0-9]*\.?[0-9]+)', 文本)
    return float(m.group(1)) if m else None


def 解析整数(文本, 键):
    m = re.search(rf'"{键}"\s*[:：]\s*([1-5])', 文本)
    return int(m.group(1)) if m else None


def 解析情绪(文本):
    情绪类别 = ["快乐", "悲伤", "愤怒", "恐惧", "惊讶", "中性", "疲惫"]
    m = re.search(r'"情绪"\s*[:：]\s*"([^"]+)"', 文本)
    if m:
        return m.group(1)
    for c in 情绪类别:
        if c in 文本:
            return c
    return None


def 主():
    if len(sys.argv) < 3:
        print("用法: 裁判子进程.py <任务.json> <输出.json> [--bf16]")
        sys.exit(1)
    任务路径, 输出路径 = sys.argv[1], sys.argv[2]
    用bf16 = "--bf16" in sys.argv[3:]
    with open(任务路径, encoding="utf-8") as f:
        任务 = json.load(f)
    裁判类型 = 任务["裁判"]
    请求 = 任务["请求"]
    # 混合模式：每个请求自带 "类型" 字段则按请求类型分发
    混合 = any(r.get("类型") for r in 请求)
    if 用bf16:
        模型, 分词器 = 加载bf16()
    elif 裁判模型名 != "Qwen2.5-7B-Instruct":
        模型, 分词器 = 加载fp16()
    else:
        模型, 分词器 = 加载4bit()

    结果 = []
    for i, r in enumerate(请求):
        类型 = r.get("类型") if 混合 else 裁判类型
        try:
            if 类型 == "heartbench":
                rubric_str = "\n".join(f"[{item['dimension']}][{item['score']}] {item['content']}" for item in r["rubric"])
                prompt = 评分提示模板.format(
                    dialogue_history=r["对话"], response=r["回复"],
                    rubric=rubric_str, rubric_nums=len(r["rubric"]))
                文本 = 生成(模型, 分词器, [{"role": "user", "content": prompt}], max_new_tokens=512)
                detail = 提取detail(文本)
                结果.append({"detail": detail, "原始": 文本[:200]})
            elif 类型 == "feel":
                文本 = 生成(模型, 分词器, [{"role": "user", "content": 共情评分提示.format(context=r["context"])}])
                结果.append({"共情合理性": 解析小数(文本, "共情合理性"), "原始": 文本[:150]})
            elif 类型 == "emocharacter_保真度":
                文本 = 生成(模型, 分词器, [{"role": "user", "content": 共情裁判提示.format(
                    角色=r["角色"], 基调=r["基调"], 用户话=r["用户话"], 回复=r["回复"])}], max_new_tokens=150)
                结果.append({"情感保真度": 解析小数(文本, "情感保真度"), "原始": 文本[:150]})
            elif 类型 == "emocharacter_一致性":
                文本 = 生成(模型, 分词器, [{"role": "user", "content": 一致性裁判提示.format(
                    角色=r["角色"], 基调=r["基调"], 全部回复=r["全部回复"])}], max_new_tokens=150)
                结果.append({"一致性": 解析小数(文本, "一致性"), "原始": 文本[:150]})
            elif 类型 == "emocharacter_强制选择":
                文本 = 生成(模型, 分词器, [{"role": "user", "content": 强制选择一致性提示.format(
                    角色=r["角色"], 基调=r["基调"], 集合A=r["集合A"], 集合B=r["集合B"])}], max_new_tokens=150)
                m = re.search(r'"更像同一角色"\s*[:：]\s*"?([AB])"?', 文本)
                选中 = m.group(1) if m else None
                正确 = (选中 == "A") == r["真实在A"] if 选中 else None
                结果.append({"正确": 正确, "选中": 选中, "原始": 文本[:150]})
            elif 类型 == "情感_推理":
                文本 = 生成(模型, 分词器, [{"role": "user", "content": 推理评分提示.format(
                    情境=r["情境"], 推理=r["推理"])}])
                结果.append({"推理质量": 解析小数(文本, "推理质量"), "原始": 文本[:150]})
            elif 类型 == "情感_四分支":
                文本 = 生成(模型, 分词器, [{"role": "user", "content": 四分支提示.format(user=r["user"], reply=r["回复"])}])
                结果.append({k: 解析整数(文本, k) for k in ("情绪感知", "情绪理解", "情绪促进思维", "情绪管理")})
            elif 类型 == "llm_judge_配对":
                AI在前 = r["AI在前"]
                A, B = (r["AI"], r["真人"]) if AI在前 else (r["真人"], r["AI"])
                文本 = 生成(模型, 分词器, [{"role": "user", "content": LLM裁判配对提示.format(
                    user=r["user"], 回复A=A, 回复B=B)}], max_new_tokens=120)
                选择 = 解析配对(文本)
                AI胜 = (选择 == "A") if AI在前 else (选择 == "B")
                结果.append({"AI胜": AI胜, "选择": 选择, "原始": 文本[:120]})
            elif 类型 == "llm_judge_评分":
                文本 = 生成(模型, 分词器, [{"role": "user", "content": LLM裁判评分提示.format(
                    user=r["user"], 回复=r["回复"])}], max_new_tokens=120)
                结果.append({"评分": 解析评分(文本), "原始": 文本[:120]})
            else:
                结果.append({"error": f"未知裁判类型 {类型}"})
        except Exception as e:
            结果.append({"error": str(e)})
            gc.collect()
            torch.cuda.empty_cache()
        if (i + 1) % 20 == 0:
            print(f"[裁判子进程] {i+1}/{len(请求)}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=2)
    print(f"[裁判子进程] 完成 {len(结果)} 条 -> {输出路径}", flush=True)


if __name__ == "__main__":
    主()
