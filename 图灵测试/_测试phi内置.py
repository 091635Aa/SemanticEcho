# -*- coding: utf-8 -*-
"""验证：Phi-3.5 用 transformers 内置 Phi3（去掉 auto_map）能否正常加载+生成"""
import os, json, traceback

原目录 = r"l:\模型空间\Phi-3.5-mini-instruct"
cfg路径 = os.path.join(原目录, "config.json")
备份路径 = os.path.join(原目录, "config.json.bak_phi")
已改 = False
try:
    with open(cfg路径, encoding="utf-8") as f:
        cfg = json.load(f)
    if not os.path.exists(备份路径):
        with open(备份路径, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    if "auto_map" in cfg:
        cfg.pop("auto_map", None)
        cfg.pop("_name_or_path", None)
        with open(cfg路径, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        已改 = True

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(原目录, trust_remote_code=False)
    模型 = AutoModelForCausalLM.from_pretrained(
        原目录, torch_dtype=torch.float16, trust_remote_code=False,
        low_cpu_mem_usage=True, attn_implementation="eager")
    模型.to("cuda")
    模型.eval()
    print("[OK] 加载成功（内置 Phi3 实现）", flush=True)

    提示 = tok.apply_chat_template(
        [{"role": "user", "content": "最近工作好累啊"}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(提示, return_tensors="pt").to("cuda")
    out = 模型.generate(
        inputs.input_ids, max_new_tokens=32,
        pad_token_id=tok.eos_token_id,
        temperature=1.0, top_p=0.9, top_k=50,
        repetition_penalty=1.05, do_sample=True)
    新 = out[0, inputs.input_ids.shape[1]:]
    print("[OK] 生成：", tok.decode(新, skip_special_tokens=True)[:60], flush=True)
except Exception:
    traceback.print_exc()
finally:
    if 已改 and os.path.exists(备份路径):
        with open(备份路径, encoding="utf-8") as f:
            cfg0 = json.load(f)
        with open(cfg路径, "w", encoding="utf-8") as f:
            json.dump(cfg0, f, ensure_ascii=False, indent=2)
        print("[恢复] config.json 已还原", flush=True)
