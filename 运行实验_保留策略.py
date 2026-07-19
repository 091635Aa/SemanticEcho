"""
运行保留策略对比实验（E11-E13）

实验配置：
  E11: Echo (λ=0.5) + 情感筛选 + 滑动窗口(3轮)
  E12: Echo (λ=1.0) + 情感筛选 + 滑动窗口(3轮)
  E13: Echo (λ=0.5) + 情感筛选 + 全局保留

对比基线：E7 (λ=0.5 + 筛选+衰减), E8 (λ=1.0 + 筛选+衰减)
"""

import os
import gc
import json
import time
import sys
from pathlib import Path
from typing import Any, Optional, Callable

os.chdir(Path(__file__).parent)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from 回响池 import 语义回响池
from 采样处理器 import 回响注入器
from 情感过滤器 import 情感过滤器


# ── 测试提示词集（与第二轮实验相同） ──

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


# ── 实验配置 ──
# (实验编号, 描述, lambda_strength, decay_gamma, 保留策略, 滑动窗口大小)

实验配置: list[tuple[str, str, float, float, str, int]] = [
    ("E11", "Echo (λ=0.5) + 筛选 + 滑动窗口3轮", 0.5, 0.05, "滑动窗口", 3),
    ("E12", "Echo (λ=1.0) + 筛选 + 滑动窗口3轮", 1.0, 0.1, "滑动窗口", 3),
    ("E13", "Echo (λ=0.5) + 筛选 + 全局保留", 0.5, 0.05, "全局保留", 9999),
]

MAX_NEW_TOKENS: int = 128
重复次数: int = 3


# ── 快速验证模式 ──

快速验证: bool = "--quick" in sys.argv


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


def 创建轮次回调(pool: 语义回响池) -> Callable[[int, 语义回响池], None]:
    """创建轮次切换回调函数。

    每5个生成步切换一次轮次：轮次 = 步数 // 5

    Parameters
    ----------
    pool : 语义回响池
        回响池实例

    Returns
    -------
    Callable[[int, 语义回响池], None]
        轮次回调函数
    """
    def _回调(步数: int, _pool: 语义回响池) -> None:
        新轮次 = 步数 // 5
        if 新轮次 != _pool.轮次ID:
            _pool.设置轮次(新轮次)
    return _回调


def 运行单次生成_保留策略(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    lambda_strength: float = 1.0,
    decay_gamma: float = 0.1,
    情感过滤器实例: Optional[情感过滤器] = None,
    保留策略: str = "衰减",
    滑动窗口大小: int = 3,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """运行单次生成（支持保留策略对比实验）。

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
    保留策略 : str
        "衰减" | "滑动窗口" | "全局保留"
    滑动窗口大小 : int
        滑动窗口的轮数（仅"滑动窗口"策略使用）
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
    pool = 语义回响池(
        hidden_dim=hidden_dim,
        decay_gamma=decay_gamma,
        保留策略=保留策略,
        滑动窗口大小=滑动窗口大小,
    )

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
        思考标记对=("", ""),  # 不使用思考阶段
    )
    model._echo_injector = injector  # type: ignore[assignment]

    # 创建轮次回调（每5步切换一次轮次）
    轮次回调 = 创建轮次回调(pool)

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
            轮次回调=轮次回调,
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
        "保留策略": 保留策略,
        "滑动窗口大小": 滑动窗口大小,
        "总轮次数": len(set(pool.轮次列表)) if pool.轮次列表 else 0,
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
            "保留策略": "",
            "滑动窗口大小": 0,
            "总轮次数": 0,
        },
        "重复次数": 重复索引,
        "生成用时": 用时,
        "错误": 错误信息,
    }


def 加载基线数据(实验编号: str) -> Optional[float]:
    """从第二轮实验结果中加载基线平均熵。

    Parameters
    ----------
    实验编号 : str
        基线实验编号（如 "E7"）

    Returns
    -------
    Optional[float]
        基线平均熵值，不存在时返回 None
    """
    try:
        基线路径: str = f"实验数据/{实验编号}.json"
        if not os.path.exists(基线路径):
            return None
        with open(基线路径, "r", encoding="utf-8") as f:
            数据 = json.load(f)
        所有熵: list[float] = []
        for d in 数据.get("数据", []):
            for r in d.get("重复结果", []):
                if "错误" not in r:
                    所有熵.extend(r.get("熵列表", []))
        if 所有熵:
            return sum(所有熵) / len(所有熵)
        return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def main() -> None:
    """主函数：加载模型 → 初始化情感过滤器 → 运行 E11-E13 → 保存汇总。"""
    print("=" * 60)
    print("语义回响 — 保留策略对比实验运行 (E11-E13)")
    print(f"模型: Qwen/Qwen2.5-0.5B-Instruct (本地)")
    print(f"提示词: 5维度 × 3条 = 15条")
    if 快速验证:
        print(f"模式: 快速验证（仅前5条提示词 × 1次重复）")
    else:
        print(f"模式: 全量运行（15条提示词 × {重复次数}次重复）")
    print(f"实验配置: 3个 (E11-E13)")
    print(f"最大Token数: {MAX_NEW_TOKENS}")
    print("=" * 60)

    # ── 加载模型（与第二轮实验相同） ──
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

    for 实验编号, 描述, lam, gamma, 策略, 窗口大小 in 实验配置:
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

        全部结果: list[dict[str, Any]] = []
        总用时: float = 0.0

        # 快速验证模式：只取前5条提示词 × 1次重复
        提示词批次 = list(测试提示词.items())
        if 快速验证:
            提示词批次 = 提示词批次[:1]  # 只取第一个维度
            当前重复次数 = 1
        else:
            当前重复次数 = 重复次数

        for 维度, 提示词列表 in 提示词批次:
            for 提示词 in 提示词列表:
                if 快速验证:
                    # 快速验证只取第一个提示词
                    if 提示词 != 提示词列表[0]:
                        continue
                print(f"  [{维度}] 提示词: {提示词[:20]}...")

                重复结果: list[dict[str, Any]] = []
                for 次 in range(当前重复次数):
                    t0: float = time.time()
                    try:
                        结果 = 运行单次生成_保留策略(
                            model, tokenizer, 提示词,
                            lambda_strength=lam,
                            decay_gamma=gamma,
                            情感过滤器实例=情感过滤,
                            保留策略=策略,
                            滑动窗口大小=窗口大小,
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
                              f"轮次数={结果['池统计']['总轮次数']}, "
                              f"池大小={结果['池统计']['最终大小']}, "
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
            "保留策略": 策略,
            "滑动窗口大小": 窗口大小,
            "总提示词数": len([p for pl in 测试提示词.values() for p in pl]),
            "总重复次数": 当前重复次数,
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
    print("保留策略对比实验汇总")
    print(f"{'=' * 60}")

    # 加载基线数据（E7、E8）
    基线映射: dict[str, tuple[str, str]] = {
        "E11": ("E7", "λ=0.5 + 衰减"),
        "E12": ("E8", "λ=1.0 + 衰减"),
        "E13": ("E7", "λ=0.5 + 衰减"),
    }
    基线熵: dict[str, float] = {}
    for 基线编号, _ in set(基线映射.values()):
        熵值 = 加载基线数据(基线编号)
        if 熵值 is not None:
            基线熵[基线编号] = 熵值
            print(f"  加载基线 {基线编号}: 平均熵={熵值:.4f}")

    汇总: dict[str, Any] = {
        "实验说明": "第三轮实验：保留策略对比（滑动窗口 vs 全局保留 vs 衰减）",
        "实验配置": [],
        "按配置统计": {},
        "与基线对比": {},
    }

    for 实验编号, 描述, lam, gamma, 策略, 窗口大小 in 实验配置:
        if 实验编号 not in 所有实验结果:
            print(f"  {实验编号}: 无数据，跳过")
            continue

        exp = 所有实验结果[实验编号]
        组合并熵: list[float] = []
        组合并命中率: list[float] = []
        组合并轮次数: list[float] = []
        组合并池大小: list[float] = []

        for d in exp["数据"]:
            for r in d["重复结果"]:
                if "错误" not in r:  # 跳过失败的结果
                    组合并熵.extend(r["熵列表"])
                    组合并命中率.append(r["池统计"]["情感命中率"])
                    组合并轮次数.append(r["池统计"]["总轮次数"])
                    组合并池大小.append(r["池统计"]["最终大小"])

        平均熵: float = sum(组合并熵) / len(组合并熵) if 组合并熵 else 0.0
        平均命中率: float = (
            sum(组合并命中率) / len(组合并命中率) if 组合并命中率 else 0.0
        )
        平均轮次数: float = (
            sum(组合并轮次数) / len(组合并轮次数) if 组合并轮次数 else 0.0
        )
        平均池大小: float = (
            sum(组合并池大小) / len(组合并池大小) if 组合并池大小 else 0.0
        )

        汇总["按配置统计"][实验编号] = {
            "描述": 描述,
            "平均语义熵": round(平均熵, 4),
            "平均情感命中率": round(平均命中率, 4),
            "平均轮次数": round(平均轮次数, 2),
            "平均池大小": round(平均池大小, 2),
            "总用时(秒)": exp["统计"]["总用时(秒)"],
        }

        print(f"\n  {实验编号} {描述}")
        print(f"    平均熵={平均熵:.4f}, 平均命中率={平均命中率:.4f}, "
              f"平均轮次数={平均轮次数:.1f}, 平均池大小={平均池大小:.1f}, "
              f"用时={exp['统计']['总用时(秒)']:.0f}s")

        # ── 与基线对比 ──
        if 实验编号 in 基线映射:
            基线编号, 基线描述 = 基线映射[实验编号]
            if 基线编号 in 基线熵:
                基线_熵 = 基线熵[基线编号]
                变化率: float = (
                    (平均熵 - 基线_熵) / 基线_熵 * 100 if 基线_熵 > 0 else 0.0
                )
                key = f"{实验编号} vs {基线编号} ({基线描述})"
                汇总["与基线对比"][key] = {
                    f"{基线编号}_平均语义熵": 基线_熵,
                    f"{实验编号}_平均语义熵": 平均熵,
                    "变化率": f"{变化率:.2f}%",
                }
                print(f"    → vs {基线编号}: 熵变化 {变化率:.2f}%")

    汇总路径: str = "实验数据/实验结果汇总_保留策略.json"
    with open(汇总路径, "w", encoding="utf-8") as f:
        json.dump(汇总, f, ensure_ascii=False, indent=2)
    print(f"\n汇总保存至: {汇总路径}")

    # 生成文本汇总
    _生成文本汇总(汇总, 基线熵)
    print("实验完成!")


def _生成文本汇总(
    汇总: dict[str, Any],
    基线熵: dict[str, float],
) -> None:
    """生成简短的文本汇总文件。

    Parameters
    ----------
    汇总 : dict[str, Any]
        实验汇总数据
    基线熵 : dict[str, float]
        基线实验的平均熵
    """
    文本路径: str = "实验数据/实验结果_保留策略.txt"
    with open(文本路径, "w", encoding="utf-8") as f:
        f.write("=== 保留策略对比实验汇总 ===\n\n")

        # 基线
        for 基线编号, 熵值 in sorted(基线熵.items()):
            基线描述 = {
                "E7": "λ=0.5+衰减",
                "E8": "λ=1.0+衰减",
            }.get(基线编号, 基线编号)
            f.write(f"{基线编号} ({基线描述}): 平均熵={熵值:.4f}\n")

        # 新实验
        for 实验编号 in ["E11", "E12", "E13"]:
            if 实验编号 not in 汇总.get("按配置统计", {}):
                continue
            统计 = 汇总["按配置统计"][实验编号]
            平均熵 = 统计["平均语义熵"]
            描述 = 统计["描述"]
            f.write(f"\n{实验编号} ({描述}): 平均熵={平均熵:.4f}")

            # 与基线对比
            对比key = None
            for k in 汇总.get("与基线对比", {}):
                if k.startswith(实验编号):
                    对比key = k
                    break
            if 对比key:
                变化率 = 汇总["与基线对比"][对比key]["变化率"]
                f.write(f"  vs {对比key}: {变化率}")

            # 额外统计
            f.write(f"  平均轮次数={统计.get('平均轮次数', 'N/A')}")
            f.write(f"  平均池大小={统计.get('平均池大小', 'N/A')}")
            f.write("\n")

        f.write("\n")
        f.write("对比说明：\n")
        f.write("  滑动窗口3轮 仅保留最近 3 个轮次（约 15 步）的向量\n")
        f.write("  全局保留 保留所有轮次的所有向量\n")
        f.write("  衰减（基线）逐步遗忘旧向量\n")

    print(f"\n文本汇总保存至: {文本路径}")


if __name__ == "__main__":
    main()
