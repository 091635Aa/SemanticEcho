"""
全量实验运行脚本 — 运行所有实验配置 × 所有提示词

实验矩阵：
  E1: Baseline (top_p=0.9)          — 基线
  E2: Baseline (temperature=1.0)    — 温度基线
  E3: Echo (λ=0.5, γ=0.05)         — 弱回响弱衰减
  E4: Echo (λ=1.0, γ=0.1)          — 默认配置
  E5: Echo (λ=2.0, γ=0.5)          — 强回响强衰减
  E6: Echo (λ=1.0, γ=0.01)         — 弱衰减（长记忆）
"""

import os
import sys
import gc
import json
import time
import math
from pathlib import Path

os.chdir(Path(__file__).parent)
项目根目录 = Path(__file__).resolve().parent.parent
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_echo.回响池 import 语义回响池
from semantic_echo.采样处理器 import 回响注入器

# ── 测试提示词集 ──
测试提示词 = {
    "开心": [
        "你今天真好看",
        "终于等到你了，我好开心",
        "今天的中标消息让我兴奋得睡不着",
    ],
    "悲伤": [
        "一切都结束了",
        "他走了，再也不会回来了",
        "我好像再也找不到活下去的意义了",
    ],
    "愤怒": [
        "你凭什么这么说我",
        "这个结果简直是荒谬至极",
        "我受够了你们的欺骗和背叛",
    ],
    "中性": [
        "今天天气不错",
        "我想了解一下这个产品的功能",
        "请问地铁站怎么走",
    ],
    "复杂混合": [
        "虽然赢了比赛，但我最好的朋友受伤了",
        "我爱我的工作，但是工资真的太低了",
        "你给了我这么多帮助，我却没办法回报你",
    ],
}

# ── 实验配置 ──
实验配置列表 = [
    ("E1", "Baseline (top_p=0.9)", None, None, 1.0, 0.9, 50),
    ("E2", "Baseline (temperature=1.0)", None, None, 1.0, 1.0, 0),
    ("E3", "Echo (λ=0.5, γ=0.05)", 0.5, 0.05, 1.0, 0.9, 50),
    ("E4", "Echo (λ=1.0, γ=0.1)", 1.0, 0.1, 1.0, 0.9, 50),
    ("E5", "Echo (λ=2.0, γ=0.5)", 2.0, 0.5, 1.0, 0.9, 50),
    ("E6", "Echo (λ=1.0, γ=0.01)", 1.0, 0.01, 1.0, 0.9, 50),
]

# 最大 token 数和重复次数可调
MAX_NEW_TOKENS = 128
重复次数 = 3  # 每个提示词生成 3 次取统计


def 计算语义熵(logits: torch.Tensor) -> float:
    """计算单个位置的语义熵"""
    if logits.dim() == 2:
        logits = logits[0]
    # 处理 -inf（需在 float32 下操作，float16 无法表示 -1e9）
    logits = logits.clone().float()
    logits[logits == float('-inf')] = -1e4
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum().item()
    return entropy


def 运行单次生成(
    model, tokenizer, prompt: str,
    is_echo: bool = False,
    lambda_strength: float = 1.0,
    decay_gamma: float = 0.1,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 128,
) -> dict:
    """运行单次生成，返回输出文本和每步熵值"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    if not is_echo:
        # ── 基线模式 ──
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        生成ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        文本 = tokenizer.decode(生成ids, skip_special_tokens=True)
        
        # 收集每步熵
        熵列表 = []
        if hasattr(outputs, 'scores'):
            for step_logits in outputs.scores:
                ent = 计算语义熵(step_logits)
                熵列表.append(ent)
        
        return {
            "文本": 文本,
            "熵列表": 熵列表,
            "平均熵": sum(熵列表) / len(熵列表) if 熵列表 else 0.0,
            "步数": len(熵列表),
        }
    else:
        # ── 回响模式 ──
        hidden_dim = model.config.hidden_size
        vocab_size = model.config.vocab_size
        pool = 语义回响池(hidden_dim=hidden_dim, decay_gamma=decay_gamma)
        # 先清理之前可能残留的 hooks
        if hasattr(model, '_echo_injector'):
            try:
                model._echo_injector._移除钩子()
            except Exception:
                pass
        injector = 回响注入器(
            model, pool,
            lambda_strength=lambda_strength,
            projection_seed=42,
            last_n_layers=1,  # 仅最后一层，减少hook开销
        )
        model._echo_injector = injector  # 保存引用以便后续清理
        
        输入ids = inputs.input_ids
        熵列表 = []
        
        # 收集 logits 的回调
        def logits_cb(step, logits):
            ent = 计算语义熵(logits)
            熵列表.append(ent)
        
        try:
            with torch.no_grad():
                输出ids = injector.生成(
                    输入ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    logits_callback=logits_cb,
                )
        except Exception as e:
            # 清理后再抛出
            injector._移除钩子()
            raise e
        
        # 生成完成后立即清理 hooks
        injector._移除钩子()
        
        生成ids = 输出ids[0][输入ids.shape[1]:]
        文本 = tokenizer.decode(生成ids, skip_special_tokens=True)
        
        # 池统计
        质心 = pool.计算质心()
        池统计 = {
            "最终大小": pool.大小,
            "有效温度": pool.计算有效温度(),
            "质心范数": 质心.norm().item(),
            "最终步数": pool.当前步数,
        }
        
        return {
            "文本": 文本,
            "熵列表": 熵列表,
            "平均熵": sum(熵列表) / len(熵列表) if 熵列表 else 0.0,
            "步数": len(熵列表),
            "池统计": 池统计,
        }


def main():
    print("=" * 60)
    print("语义回响 — 全量实验运行")
    print(f"模型: Qwen/Qwen2.5-0.5B-Instruct (本地)")
    print(f"提示词: 5维度 × 3条 = 15条")
    print(f"实验配置: 6个 (E1-E6)")
    print(f"重复次数: {重复次数}")
    print(f"最大Token数: {MAX_NEW_TOKENS}")
    print("=" * 60)
    
    # 加载模型
    本地路径 = os.path.join(os.path.dirname(__file__), "本地模型")
    if os.path.exists(本地路径):
        model_path = 本地路径
    else:
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"\n[加载模型] {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print(f"模型设备: {model.device}")
    print(f"hidden_size: {model.config.hidden_size}")
    print(f"vocab_size: {model.config.vocab_size}")
    
    # 确保输出目录存在
    os.makedirs("实验数据", exist_ok=True)
    
    所有实验结果 = {}
    
    for 实验编号, 描述, lam, gamma, temp, top_p, top_k in 实验配置列表:
        # 检查是否已有结果文件
        检查路径 = f"实验数据/{实验编号}.json"
        if os.path.exists(检查路径):
            print(f"\n[{实验编号}] {描述} — 已有结果，跳过")
            with open(检查路径, "r", encoding="utf-8") as f:
                所有实验结果[实验编号] = json.load(f)
            continue
        
        print(f"\n{'=' * 40}")
        print(f"[{实验编号}] {描述}")
        print(f"{'=' * 40}")
        
        是回响 = lam is not None
        
        全部结果 = []
        总用时 = 0.0
        
        for 维度索引, (维度, 提示词列表) in enumerate(测试提示词.items()):
            for 提示词 in 提示词列表:
                print(f"  [{维度}] 提示词: {提示词[:20]}...")
                
                重复结果 = []
                for 次 in range(重复次数):
                    t0 = time.time()
                    try:
                        结果 = 运行单次生成(
                            model, tokenizer, 提示词,
                            is_echo=是回响,
                            lambda_strength=lam or 0.0,
                            decay_gamma=gamma or 0.1,
                            temperature=temp,
                            top_p=top_p,
                            top_k=top_k,
                            max_new_tokens=MAX_NEW_TOKENS,
                        )
                        用时 = time.time() - t0
                        总用时 += 用时
                        结果["重复次数"] = 次
                        结果["生成用时"] = 用时
                        重复结果.append(结果)
                        print(f"    第{次+1}次: 熵={结果['平均熵']:.3f}, 步数={结果['步数']}, "
                              f"用时={用时:.1f}s, 输出={结果['文本'][:30]}...")
                    except Exception as e:
                        print(f"    第{次+1}次: 失败! {e}")
                        import traceback
                        traceback.print_exc()
                
                全部结果.append({
                    "维度": 维度,
                    "提示词": 提示词,
                    "重复结果": 重复结果,
                })
        
        算法统计 = {
            "配置": 描述,
            "是回响模式": 是回响,
            "lambda_strength": lam,
            "decay_gamma": gamma,
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "总提示词数": len([p for pl in 测试提示词.values() for p in pl]),
            "总重复次数": 重复次数,
            "总用时(秒)": round(总用时, 1),
        }
        
        所有实验结果[实验编号] = {
            "统计": 算法统计,
            "数据": 全部结果,
        }
        
        # 每完成一个实验配置，保存一次中间结果
        保存路径 = f"实验数据/{实验编号}.json"
        with open(保存路径, "w", encoding="utf-8") as f:
            json.dump(所有实验结果[实验编号], f, ensure_ascii=False, indent=2)
        print(f"  已保存: {保存路径}")
        
        # 释放 GPU 内存，防止 hook 泄漏导致 OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"  [内存清理完成]")
    
    # ── 汇总报告 ──
    print(f"\n{'=' * 60}")
    print("实验汇总")
    print(f"{'=' * 60}")
    
    汇总 = {"实验配置": [], "按配置统计": {}}
    for 实验编号, 描述, lam, gamma, temp, top_p, top_k in 实验配置列表:
        exp = 所有实验结果[实验编号]
        组合并熵 = []
        for d in exp["数据"]:
            for r in d["重复结果"]:
                组合并熵.extend(r["熵列表"])
        
        平均熵 = sum(组合并熵) / len(组合并熵) if 组合并熵 else 0.0
        汇总["按配置统计"][实验编号] = {
            "描述": 描述,
            "平均语义熵": round(平均熵, 4),
            "总用时(秒)": exp["统计"]["总用时(秒)"],
        }
        
        is_echo_text = "回响" if lam is not None else "基线"
        print(f"  {实验编号} [{is_echo_text}] 平均熵={平均熵:.4f}, "
              f"用时={exp['统计']['总用时(秒)']:.0f}s")
    
    # 计算细腻度提升率
    if "E1" in 汇总["按配置统计"] and "E4" in 汇总["按配置统计"]:
        E1_熵 = 汇总["按配置统计"]["E1"]["平均语义熵"]
        E4_熵 = 汇总["按配置统计"]["E4"]["平均语义熵"]
        提升率 = (E4_熵 - E1_熵) / E1_熵 * 100 if E1_熵 > 0 else 0
        汇总["细腻度提升率(E4 vs E1)"] = f"{提升率:.2f}%"
        print(f"\n  → 细腻度提升率 (E4 vs E1): {提升率:.2f}%")
    
    汇总路径 = "实验数据/实验结果汇总.json"
    with open(汇总路径, "w", encoding="utf-8") as f:
        json.dump(汇总, f, ensure_ascii=False, indent=2)
    print(f"\n汇总保存至: {汇总路径}")
    print("全量实验完成!")


if __name__ == "__main__":
    main()
