# 附录 B · 核心代码与注释

> 说明：完整工程代码位于 `experiments/` 与 `semantic_echo/` 目录。本附录只收录与本文创新点直接相关的两段核心代码，并附逐函数中文注释；其余模块以代码位置表说明。

---

## B.1 思考链中断注入器（experiments/思考链注入器.py · 核心创新）

**作用**：把 Qwen3 等模型的预训练思考链（`<think>…</think>`）改造成情感注入通道——思考阶段只捕获情感向量、不注入；检测到思考结束标记时硬中断，定格"总体向量"，正文阶段以固定偏置注入。继承 `回响注入器` 复用钩子/投影/池，未修改任何现有源码。

```python
class 思考链中断注入器(回响注入器):
    """思考链纠正：思考结束中断 → 总体向量 → 正文固定注入"""

    def 重置(self) -> None:
        """复用注入器：清空池与阶段状态。
        投影矩阵只创建一次，避免每次重建 1.45GB 导致显存 OOM。"""
        self.pool.清空()          # 清空回响池（短期记忆）
        self.思考步数 = 0          # 思考链已走步数
        self.总体向量 = None       # 思考阶段定格的情感向量
        self.当前阶段 = "思考"     # 阶段状态机：思考 → 正文

    def _初始化投影(self, seed: int) -> None:
        """创建随机静态投影矩阵（hidden_dim → vocab_size）。
        直接在 GPU 分配：本机 CPU RAM 不足，父类 CPU 分配会 OOM。
        scale 为 He 初始化缩放，保证投影后向量量级稳定。"""
        rng = torch.Generator(device=self.device)   # 用 GPU 上的随机数生成器
        rng.manual_seed(seed)                       # 固定种子，结果可复现
        scale = math.sqrt(2.0 / self.hidden_dim)
        self.投影矩阵 = torch.randn(
            self.hidden_dim, self.vocab_size,
            generator=rng, dtype=torch.float32, device=self.device) * scale
        self.投影矩阵.requires_grad_(False)          # 固定投影，不参与训练

    @torch.no_grad()
    def 生成(self, input_ids, max_new_tokens=256, temperature=1.0,
             top_p=0.9, top_k=50, repetition_penalty=1.0, ...):
        """重写生成循环：思考捕获 → 硬中断 → 总体向量定格 → 正文固定注入"""
        for 步 in range(max_new_tokens):
            # 单步前向：有缓存时只输入最后一个 token，否则输入全部
            outputs = self.model(已生成[:, -1:], past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]       # 取最后一个位置的 logits

            if self.当前阶段 == "思考":
                # 思考阶段：只捕获情感向量入池，【不注入】——不干扰逻辑链
                self.捕获回响(logits, tokenizer=self.tokenizer)
            else:
                # 正文阶段：用定格总体向量作固定偏置注入
                if self.总体向量 is not None:
                    偏置 = self.总体向量 @ self.投影矩阵   # hidden → vocab 空间
                    logits = logits + 偏置.unsqueeze(0) * self.lambda_strength

            下一个token = 采样(logits, temperature, top_p, top_k)  # 采样解码

            # ── 思考结束检测（token 级硬中断） ──
            if self.当前阶段 == "思考":
                self.思考步数 += 1
                if (下一个token.item() == self.思考结束token        # </think> id=151668
                        or self.思考步数 >= self.思考长度上限):
                    self.总体向量 = self.pool.计算质心()            # 定格思考阶段情绪
                    self.当前阶段 = "正文"                          # 切换到正文阶段
                    self.pool.清空()                                # 释放思考向量

            已生成 = torch.cat([已生成, 下一个token], dim=-1)       # 拼接已生成序列
```

**关键参数**：`思考结束token文本="</think>"`（Qwen3 中 id=151668）、`思考长度上限`（Qwen3 思考链常超 256 token）、`lambda_strength`（正文注入强度，通用注入值）。

---

## B.2 通用注入值公式（experiments/一体化测试运行器.py）

**作用**：跨模型自动推荐 λ/γ/τ，解决"同一 λ 在不同架构/规模上坍缩"问题。核心是**架构族因子**（按模型名分段，不能用 hidden_dim 判断——Qwen3-1.7B 与 4B 的 hidden 均 >1536 但敏感性差异大）。

```python
# 扫描表：四个基准 hidden_dim 的实测最优 (λ, γ, τ)
扫描表 = {896: (0.50, 0.05, 0.10), 1536: (0.08, 0.07, 0.09),
          2048: (0.10, 0.08, 0.06), 3584: (0.06, 0.12, 0.05)}
基准 = 896  # 参考维度

def 架构族因子(模型名, hidden_dim):
    """返回 (因子, 族名)。基础 λ × 因子 = 该模型可用注入强度。
    按模型名字符串分段（不能用 hidden_dim：Qwen3-1.7B 与 4B 的 hidden
    均 >1536，但敏感性差异大）。"""
    if "Qwen3" in 模型名:
        # 实测：0.6B(λ0.41→坍缩)、1.7B(λ0.10→坍缩)、4B(λ0.098→有效)
        if any(k in 模型名 for k in ("0.6", "1.7", "1.5")):
            return (0.3, "Qwen3≤1.7B")   # 小尺寸极度敏感，需大幅降 λ
        return (0.6, "Qwen3≥4B")
    if "gemma" in 模型名.lower():
        return (0.7, "Gemma")
    if "SmolLM" in 模型名:
        return (0.5, "SmolLM")
    if "Phi" in 模型名:
        return (0.8, "Phi")
    return (1.0, "Qwen2.5/通用")

def 通用注入参数(模型名, hidden_dim, 量化):
    """通用最大化激活注入值：λ = 基础值 × 架构族因子 × 量化因子"""
    hidden_dim = int(hidden_dim)
    if hidden_dim in 扫描表:
        λ基础, γ基础, τ基础 = 扫描表[hidden_dim]     # 命中扫描表直接用实测值
        来源基础 = "扫描表"
    else:
        # 公式兜底：λ 随 hidden_dim 增大而减小（注入剂量跨尺度归一化）
        λ基础, γ基础, τ基础 = (公式λ(hidden_dim),
                               0.05 * (hidden_dim / 基准) ** 0.5,
                               0.10 * (基准 / hidden_dim) ** 0.5)
        来源基础 = "公式"
    族因子, 族名 = 架构族因子(模型名, hidden_dim)   # 架构敏感性修正
    量化因子 = 0.75 if 量化 == "4bit" else 1.0      # 量化放大敏感度 → ×0.75
    λ = λ基础 * 族因子 * 量化因子
    return {"λ": round(λ, 4), "γ": round(float(γ基础), 4), "τ": round(float(τ基础), 4),
            "来源": f"通用架构({族名}×{族因子}, {量化}×{量化因子}, 基础{来源基础})"}
```

---

## B.3 其余模块代码位置一览

| 模块 | 文件 | 作用 |
|---|---|---|
| 回响池 | `semantic_echo/回响池.py` | 短期记忆：hidden_state 加权质心 + 衰减/滑动窗口/全局保留三种策略 |
| 回响注入器 | `semantic_echo/采样处理器.py` | 钩子注册、投影注入、思考/正文阶段切换 |
| 情感过滤器 | `semantic_echo/情感过滤器.py` | 基于 cnsenti 词库从候选 token 筛情感词、算情感强度 |
| 加载/基线/指标 | `agent_echo/echo_common.py` | 模型加载、裸生成、回响生成、熵/重复率/命中率 |
| 多模型对照 | `experiments/一体化测试运行器.py` | 模型×量化×模式对照，自动叠加轮数，JSON/JSONL/CSV 输出 |
| 批量调度 | `experiments/批量对照测试.py` | 15 配置调度 + 汇总表生成 |
| Qwen3 重测 | `experiments/重测Qwen3_通用注入.py` | Qwen3 全系通用注入重测 |
| LoRA 训练 | `f:\lora外挂\training_scripts\train_emotion_lora.py` | 情感适配器训练（r=8, α=16, q/k/v/o_proj） |
| 图灵测试 | `图灵测试/生成器.py` | 裸 vs 四层双模式生成，5 项基准 |
| 融合测试 | `experiments/LoRA思考链融合测试.py` | LoRA 外挂 × 思考链注入四组对比 |
| PDF 渲染 | `论文/转PDF_决策的温度.py` | md → HTML → Edge headless PDF |

---
