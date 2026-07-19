"""
运行第二轮实验 — 情感词库筛选 + 思考阶段注入

实验矩阵：
  E7: Echo (λ=0.5, γ=0.05) + 情感筛选          — 验证情感词库筛选的效果
  E8: Echo (λ=1.0, γ=0.1) + 情感筛选           — 验证情感筛选能否缓解λ≥1.0的重复问题
  E9: Echo (λ=0.5, γ=0.05) + 情感筛选 + 思考阶段 — 完整方案
  E10: Echo (λ=1.0, γ=0.1) + 情感筛选 + 思考阶段  — 完整方案高强度
"""

import os
import gc
import json
import time
from pathlib import Path
from typing import Any, Optional

os.chdir(Path(__file__).parent)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from 回响池 import 语义回响池
from 采样处理器 import 回响注入器
from 情感过滤器 import 情感过滤器


# ── 测试提示词集（与第一轮相同） ──

测试提示词: dict[str, list[str]] = {
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


# ── 第二轮实验配置 ──
# (实验编号, 描述, lambda_strength, decay_gamma, use_emotion_filter, think_tag_pair)

第二轮配置: list[tuple[str, str, float, float, bool, Optional[tuple[str, str]]]] = [
    ("E7",  "Echo (λ=0.5, γ=0.05) + 情感筛选",       0.5,  0.05, True, None),
    ("E8",  "Echo (λ=1.0, γ=0.1) + 情感筛选",         1.0,  0.1,  True, None),
    ("E9",  "Echo (λ=0.5, γ=0.05) + 筛选 + 思考阶段", 0.5,  0.05, True, ("<think>", "</think>")),
    ("E10", "Echo (λ=1.0, γ=0.1) + 筛选 + 思考阶段",  1.0,  0.1,  True, ("<think>", "</think>")),
]

# 最大 token 数和重复次数（与第一轮一致）
MAX_NEW_TOKENS: int = 128
重复次数: int = 3


def 计算语义熵(logits: torch.Tensor) -> float:
    """计算单个位置的语义熵。

    Parameters
    ----------
    logits : torch.Tensor
        shape=(1, vocab_size) 或 (vocab_size,)，当前步的原始 logits

    Returns
    -------
    float
        语义熵值（非负），单位为 nats
    """
    if logits.dim() == 2:
        logits = logits[0]
    # 在 float32 下操作（float16 无法表示大负值）
    logits = logits.clone().float()
    logits[logits == float('-inf')] = -1e4
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy: float = -(probs * log_probs).sum().item()
    return entropy


def 运行单次生成_回响v2(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    lambda_strength: float = 1.0,
    decay_gamma: float = 0.1,
    情感过滤器实例: Optional[情感过滤器] = None,
    思考标记对: tuple[str, str] = ("", ""),
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """升级版回响模式生成（支持情感筛选 + 思考阶段注入）。

    Parameters
    ----------
    model : AutoModelForCausalLM
        HuggingFace 预训练模型
    tokenizer : AutoTokenizer
        配套的 tokenizer
    prompt : str
        输入提示文本
    lambda_strength : float
        回响注入强度系数 λ
    decay_gamma : float
        回响池指数衰减系数 γ
    情感过滤器实例 : Optional[情感过滤器]
        情感筛选器实例，为 None 时跳过情感筛选
    思考标记对 : tuple[str, str]
        思考阶段边界标记，如 ("<think>", "</think>")；为空时不启用阶段切换
    temperature : float
        采样温度
    top_p : float
        nucleus 采样累积概率阈值
    top_k : int
        保留的 top-k 候选数
    max_new_tokens : int
        最大新生成 token 数

    Returns
    -------
    dict[str, Any]
        包含 "文本", "熵列表", "平均熵", "步数", "池统计" 的字典
    """
    hidden_dim: int = model.config.hidden_size  # type: ignore[union-attr]
    pool = 语义回响池(hidden_dim=hidden_dim, decay_gamma=decay_gamma)

    # 清理之前可能残留的 hooks（防御性编程）
    if hasattr(model, '_echo_injector'):
        try:
            model._echo_injector._移除钩子()  # type: ignore[union-attr]
        except Exception:
            pass

    injector = 回响注入器(
        model, pool,
        lambda_strength=lambda_strength,
        projection_seed=42,
        last_n_layers=1,
        情感过滤器实例=情感过滤器实例,
        思考标记对=思考标记对,
        思考阶段λ=lambda_strength,
        正文阶段λ=0.0,
    )
    model._echo_injector = injector  # type: ignore[assignment]

    输入ids: torch.Tensor = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
    熵列表: list[float] = []

    def logits_cb(step: int, logits: torch.Tensor) -> None:
        ent = 计算语义熵(logits)
        熵列表.append(ent)

    try:
        输出ids: torch.Tensor = injector.生成(
            输入ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logits_callback=logits_cb,
            tokenizer=tokenizer,
        )
    finally:
        injector._移除钩子()

    生成ids: torch.Tensor = 输出ids[0][输入ids.shape[1]:]
    文本: str = tokenizer.decode(生成ids, skip_special_tokens=True)

    质心: torch.Tensor = pool.计算质心()
    池统计: dict[str, Any] = {
        "最终大小": pool.大小,
        "有效温度": pool.计算有效温度(),
        "质心范数": round(质心.norm().item(), 6),
        "情感命中率": pool.情感命中率,
    }

    return {
        "文本": 文本,
        "熵列表": 熵列表,
        "平均熵": sum(熵列表) / len(熵列表) if 熵列表 else 0.0,
        "步数": len(熵列表),
        "池统计": 池统计,
    }


def 构建失败结果(重复索引: int, 用时: float, 错误信息: str) -> dict[str, Any]:
    """构建失败时的占位结果，保持 JSON 结构完整。

    Parameters
    ----------
    重复索引 : int
        重复实验的索引
    用时 : float
        生成耗时（秒）
    错误信息 : str
        异常信息字符串

    Returns
    -------
    dict[str, Any]
        填充默认值的失败结果字典
    """
    return {
        "文本": "[失败]",
        "熵列表": [],
        "平均熵": 0.0,
        "步数": 0,
        "池统计": {
            "最终大小": 0,
            "有效温度": 1.0,
            "质心范数": 0.0,
            "情感命中率": 0.0,
        },
        "重复次数": 重复索引,
        "生成用时": 用时,
        "错误": 错误信息,
    }


def main() -> None:
    """主函数：加载模型 → 初始化情感过滤器 → 运行 E7-E10 → 保存汇总。"""
    print("=" * 60)
    print("语义回响 — 第二轮实验运行 (情感筛选 + 思考阶段注入)")
    print(f"模型: Qwen/Qwen2.5-0.5B-Instruct (本地)")
    print(f"提示词: 5维度 × 3条 = 15条")
    print(f"实验配置: 4个 (E7-E10)")
    print(f"重复次数: {重复次数}")
    print(f"最大Token数: {MAX_NEW_TOKENS}")
    print("=" * 60)

    # ── 加载模型（与第一轮使用相同的本地模型） ──
    本地路径: str = os.path.join(os.path.dirname(__file__), "本地模型")
    model_path: str = 本地路径 if os.path.exists(本地路径) else "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"\n[加载模型] {model_path}")
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print(f"  模型设备: {model.device}")
    print(f"  hidden_size: {model.config.hidden_size}")
    print(f"  vocab_size: {model.config.vocab_size}")

    # ── 初始化情感过滤器（所有实验共享同一实例） ──
    print("\n[初始化] 情感过滤器...")
    情感过滤: 情感过滤器 = 情感过滤器()
    情感过滤.加载词库()
    print("  情感过滤器加载完成 (cnsenti)")

    # 确保输出目录存在
    os.makedirs("实验数据", exist_ok=True)

    所有实验结果: dict[str, dict[str, Any]] = {}

    for 实验编号, 描述, lam, gamma, use_filter, think_pair in 第二轮配置:
        # 检查是否已有结果文件，存在则跳过
        检查路径: str = f"实验数据/{实验编号}.json"
        if os.path.exists(检查路径):
            print(f"\n[{实验编号}] {描述} — 已有结果，跳过")
            with open(检查路径, "r", encoding="utf-8") as f:
                所有实验结果[实验编号] = json.load(f)
            continue

        print(f"\n{'=' * 40}")
        print(f"[{实验编号}] {描述}")
        print(f"{'=' * 40}")

        是否使用思考阶段: bool = think_pair is not None
        思考阶段对: tuple[str, str] = think_pair if 是否使用思考阶段 else ("", "")

        全部结果: list[dict[str, Any]] = []
        总用时: float = 0.0

        for 维度, 提示词列表 in 测试提示词.items():
            for 提示词 in 提示词列表:
                print(f"  [{维度}] 提示词: {提示词[:20]}...")

                重复结果: list[dict[str, Any]] = []
                for 次 in range(重复次数):
                    t0: float = time.time()
                    try:
                        结果 = 运行单次生成_回响v2(
                            model, tokenizer, 提示词,
                            lambda_strength=lam,
                            decay_gamma=gamma,
                            情感过滤器实例=情感过滤 if use_filter else None,
                            思考标记对=思考阶段对,
                            temperature=1.0,
                            top_p=0.9,
                            top_k=50,
                            max_new_tokens=MAX_NEW_TOKENS,
                        )
                        耗时: float = time.time() - t0
                        总用时 += 耗时
                        结果["重复次数"] = 次
                        结果["生成用时"] = 耗时
                        重复结果.append(结果)
                        print(f"    第{次+1}次: 熵={结果['平均熵']:.3f}, "
                              f"命中率={结果['池统计']['情感命中率']:.3f}, "
                              f"步数={结果['步数']}, "
                              f"用时={耗时:.1f}s, "
                              f"输出={结果['文本'][:30]}...")
                    except Exception as e:
                        耗时 = time.time() - t0
                        import traceback
                        traceback.print_exc()
                        print(f"    第{次+1}次: 失败! {e}")
                        # 清理可能残留的 hook
                        if hasattr(model, '_echo_injector'):
                            try:
                                model._echo_injector._移除钩子()  # type: ignore[union-attr]
                            except Exception:
                                pass
                        # 填充空结果保持结构完整性
                        重复结果.append(
                            构建失败结果(次, 耗时, str(e))
                        )

                全部结果.append({
                    "维度": 维度,
                    "提示词": 提示词,
                    "重复结果": 重复结果,
                })

        算法统计: dict[str, Any] = {
            "配置": 描述,
            "lambda_strength": lam,
            "decay_gamma": gamma,
            "使用情感筛选": use_filter,
            "使用思考阶段": 是否使用思考阶段,
            "思考标记对": list(think_pair) if think_pair else None,
            "总提示词数": len([p for pl in 测试提示词.values() for p in pl]),
            "总重复次数": 重复次数,
            "总用时(秒)": round(总用时, 1),
        }

        所有实验结果[实验编号] = {
            "统计": 算法统计,
            "数据": 全部结果,
        }

        # 每完成一个实验配置，保存一次中间结果
        保存路径: str = f"实验数据/{实验编号}.json"
        with open(保存路径, "w", encoding="utf-8") as f:
            json.dump(所有实验结果[实验编号], f, ensure_ascii=False, indent=2)
        print(f"  已保存: {保存路径}")

        # 释放 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"  [内存清理完成]")

    # ── 生成汇总报告 ──
    print(f"\n{'=' * 60}")
    print("第二轮实验汇总")
    print(f"{'=' * 60}")

    # 加载第一轮汇总数据用于对比
    第一轮汇总路径: str = "实验数据/实验结果汇总.json"
    第一轮数据: dict[str, Any] = {}
    if os.path.exists(第一轮汇总路径):
        with open(第一轮汇总路径, "r", encoding="utf-8") as f:
            第一轮数据 = json.load(f).get("按配置统计", {})

    汇总: dict[str, Any] = {
        "实验说明": "第二轮实验：情感词库筛选 + 思考阶段注入",
        "实验配置": [],
        "按配置统计": {},
        "与第一轮对比": {},
    }

    for 实验编号, 描述, lam, gamma, use_filter, think_pair in 第二轮配置:
        if 实验编号 not in 所有实验结果:
            print(f"  {实验编号}: 无数据，跳过")
            continue

        exp = 所有实验结果[实验编号]
        组合并熵: list[float] = []
        组合并命中率: list[float] = []

        for d in exp["数据"]:
            for r in d["重复结果"]:
                if "错误" not in r:  # 跳过失败的结果
                    组合并熵.extend(r["熵列表"])
                    组合并命中率.append(r["池统计"]["情感命中率"])

        平均熵: float = sum(组合并熵) / len(组合并熵) if 组合并熵 else 0.0
        平均命中率: float = (
            sum(组合并命中率) / len(组合并命中率) if 组合并命中率 else 0.0
        )

        汇总["按配置统计"][实验编号] = {
            "描述": 描述,
            "平均语义熵": round(平均熵, 4),
            "平均情感命中率": round(平均命中率, 4),
            "总用时(秒)": exp["统计"]["总用时(秒)"],
        }

        print(f"\n  {实验编号} {描述}")
        print(f"    平均熵={平均熵:.4f}, 平均命中率={平均命中率:.4f}, "
              f"用时={exp['统计']['总用时(秒)']:.0f}s")

        # ── 与第一轮对应配置对比 ──

        _构建对比项(
            汇总, 第一轮数据, 实验编号, 平均熵,
            {
                "E7": ("E3", "λ=0.5, 加情感筛选"),
                "E8": ("E4", "λ=1.0, 加情感筛选"),
                "E9": ("E3", "λ=0.5, 加筛选+思考阶段"),
                "E10": ("E4", "λ=1.0, 加筛选+思考阶段"),
            },
        )

    汇总路径: str = "实验数据/实验结果汇总_第二轮.json"
    with open(汇总路径, "w", encoding="utf-8") as f:
        json.dump(汇总, f, ensure_ascii=False, indent=2)
    print(f"\n汇总保存至: {汇总路径}")
    print("第二轮实验完成!")


def _构建对比项(
    汇总: dict[str, Any],
    第一轮数据: dict[str, Any],
    实验编号: str,
    本轮平均熵: float,
    映射表: dict[str, tuple[str, str]],
) -> None:
    """将第二轮实验结果与第一轮对应项进行对比。

    Parameters
    ----------
    汇总 : dict
        汇总字典（会被就地修改）
    第一轮数据 : dict
        第一轮按配置统计数据
    实验编号 : str
        当前实验编号（如 E7）
    本轮平均熵 : float
        当前实验的平均语义熵
    映射表 : dict[str, tuple[str, str]]
        实验编号到 (第一轮编号, 描述) 的映射
    """
    if 实验编号 not in 映射表:
        return
    第一轮编号, 对比描述 = 映射表[实验编号]
    if 第一轮编号 not in 第一轮数据:
        return
    第一轮熵: float = 第一轮数据[第一轮编号]["平均语义熵"]
    变化率: float = (
        (本轮平均熵 - 第一轮熵) / 第一轮熵 * 100 if 第一轮熵 > 0 else 0.0
    )
    key = f"{实验编号} vs {第一轮编号} ({对比描述})"
    汇总["与第一轮对比"][key] = {
        f"{第一轮编号}_平均语义熵": 第一轮熵,
        f"{实验编号}_平均语义熵": 本轮平均熵,
        "变化率": f"{变化率:.2f}%",
    }
    print(f"    → vs {第一轮编号}: 熵变化 {变化率:.2f}%")


if __name__ == "__main__":
    main()
