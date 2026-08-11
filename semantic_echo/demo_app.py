"""
语义回响 V2 — 交互式演示平台 (Gradio Web UI)

功能：
  1. 从 HuggingFace 加载任意兼容模型
  2. 基线 vs 语义回响 实时对比生成
  3. 语义熵、回响池状态、生成速度等指标可视化
  4. 支持本地推理 / 远程 API 两种模式

用法：
    semantic-echo demo-ui
    # 或
    python -m semantic_echo.demo_app
"""

import sys
import time
import json
import threading
import io
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

# ══════════════════════════════════════════════════
# 全局状态
# ══════════════════════════════════════════════════

_模型缓存: dict = {"model": None, "tokenizer": None, "name": None}
_日志缓存: list[dict] = []
_日志锁 = threading.Lock()


def _记录运行(entry: dict) -> None:
    """记录一次运行日志到内存缓存。"""
    with _日志锁:
        _日志缓存.append({
            **entry,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        if len(_日志缓存) > 200:
            _日志缓存.pop(0)


def _取日志() -> list[dict]:
    """获取运行日志列表。"""
    with _日志锁:
        return list(_日志缓存)


# ══════════════════════════════════════════════════
# 运行结果数据结构
# ══════════════════════════════════════════════════


@dataclass
class 运行结果:
    """单次推理的完整结果。"""
    模式: str  # "baseline" | "echo"
    生成文本: str = ""
    语义熵: float = 0.0
    耗时秒: float = 0.0
    回响池大小: int = 0
    有效温度: float = 1.0
    质心范数: float = 0.0
    情感命中率: float = 0.0


# ══════════════════════════════════════════════════
# 模型加载（支持本地 & API）
# ══════════════════════════════════════════════════


_兼容模型缓存: Optional[list[dict]] = None


def _扫描兼容模型() -> list[str]:
    """
    扫描已知兼容 + 本地缓存中可能的模型。

    Returns
    -------
    list[str]
        模型名称列表。
    """
    global _兼容模型缓存
    if _兼容模型缓存 is not None:
        return _兼容模型缓存

    models: list[str] = []

    # 1. 已测试验证的
    from semantic_echo.check_compatibility import get_compatible_models
    for m in get_compatible_models():
        models.append(m["model_name"])

    # 2. 扫描本地模型目录
    local_paths = [
        Path("./本地模型"),
        Path("./original"),
        Path("~/.cache/huggingface/hub").expanduser(),
    ]
    for p in local_paths:
        if p.exists() and p.is_dir():
            for sub in p.iterdir():
                if sub.is_dir() and not sub.name.startswith("."):
                    name = str(sub).replace("\\", "/")
                    if name not in models:
                        models.append(f"本地: {name}")

    _兼容模型缓存 = models
    return models


def _加载模型(
    模型名称: str,
    模式: str = "local",
    api_key: str = "",
    api_url: str = "",
) -> str:
    """
    加载模型到内存（仅 local 模式需要）。

    Parameters
    ----------
    模型名称 : str
        模型名称或路径。
    模式 : str
        "local" 本地推理 / "api" 远程 API。
    api_key : str
        API key（仅 api 模式需要）。
    api_url : str
        API 地址（仅 api 模式需要）。

    Returns
    -------
    str
        状态消息。
    """
    global _模型缓存

    if 模型名称 == _模型缓存["name"] and _模型缓存["model"] is not None:
        return f"✓ 模型已加载: {模型名称}"

    if 模式 == "api":
        _模型缓存 = {"model": None, "tokenizer": None, "name": 模型名称}
        return f"✓ API 模式已设置: {模型名称}"

    # 本地模式 → 加载模型
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 处理本地路径
        name = 模型名称
        if 模型名称.startswith("本地: "):
            name = 模型名称[4:]

        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        _模型缓存 = {
            "model": model,
            "tokenizer": tokenizer,
            "name": 模型名称,
        }
        return f"✓ 模型加载成功: {模型名称} (设备: {device})"
    except Exception as e:
        return f"✗ 加载失败: {type(e).__name__}: {e}"


def _卸载模型() -> str:
    """卸载当前模型释放显存。"""
    global _模型缓存
    _模型缓存 = {"model": None, "tokenizer": None, "name": None}
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "✓ 模型已卸载，显存已释放"


# ══════════════════════════════════════════════════
# 核心推理（基线 / 回响对比）
# ══════════════════════════════════════════════════


def _运行本地推理(
    prompt: str,
    lambda_val: float = 0.0,
    情感筛选: bool = False,
    思考分离: bool = False,
    max_tokens: int = 128,
) -> 运行结果:
    """
    使用本地模型运行一次推理。

    Parameters
    ----------
    prompt : str
        提示词。
    lambda_val : float
        回响强度。0.0 = 基线模式。
    情感筛选 : bool
        是否启用情感词库筛选。
    思考分离 : bool
        是否启用思考阶段分离注入。
    max_tokens : int
        最大生成 Token 数。

    Returns
    -------
    运行结果
    """
    import torch

    model = _模型缓存["model"]
    tokenizer = _模型缓存["tokenizer"]
    device = model.device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    mode = "echo" if lambda_val > 0 else "baseline"
    result = 运行结果(mode=mode)

    回响池 = None

    try:
        from semantic_echo.回响池 import 语义回响池
        from semantic_echo.采样处理器 import 回响注入器
        from semantic_echo.情感过滤器 import 情感过滤器
        from semantic_echo.回响评估器 import 计算语义熵, 逐Token评估器

        t_start = time.time()

        if mode == "baseline":
            # ── 基线模式: model.generate ──
            评估器 = 逐Token评估器(model.config.vocab_size)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=1.0,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            for step_idx, step_logits in enumerate(outputs.scores):
                评估器.记录(step_idx, step_logits)

            generated_ids = outputs.sequences[0, input_len:]
            result.生成文本 = tokenizer.decode(generated_ids, skip_special_tokens=True)
            result.语义熵 = 评估器.平均熵

        else:
            # ── 回响模式 ──
            hidden_dim = model.config.hidden_size
            gamma = 0.1 if lambda_val >= 1.0 else 0.05

            回响池 = 语义回响池(
                hidden_dim=hidden_dim,
                max_pool_size=1024,
                decay_gamma=gamma,
            )

            注入器 = 回响注入器(
                model=model,
                echo_pool=回响池,
                lambda_strength=lambda_val,
            )

            评估器 = 逐Token评估器(model.config.vocab_size)

            def _logits_cb(step: int, logits: torch.Tensor) -> None:
                评估器.记录(step, logits)

            with torch.no_grad():
                full_ids = 注入器.生成(
                    input_ids=inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=1.0,
                    top_p=0.9,
                    top_k=50,
                    logits_callback=_logits_cb,
                )

            generated_ids = full_ids[0, input_len:]
            result.生成文本 = tokenizer.decode(generated_ids, skip_special_tokens=True)
            result.语义熵 = 评估器.平均熵
            result.回响池大小 = 回响池.大小
            result.有效温度 = 回响池.计算有效温度()
            result.情感命中率 = 回响池.情感命中率

            # 质心范数
            try:
                质心 = 回响池.计算质心()
                result.质心范数 = 质心.norm().item()
            except Exception:
                result.质心范数 = 0.0

        result.耗时秒 = round(time.time() - t_start, 2)

    except Exception as e:
        result.生成文本 = f"[错误] {type(e).__name__}: {e}"
        result.语义熵 = -1

    return result


def _运行API推理(
    prompt: str,
    model_name: str,
    api_key: str,
    api_url: str,
    lambda_val: float = 0.0,
    max_tokens: int = 128,
) -> 运行结果:
    """
    使用远程 API 运行一次推理（回响模式仅限本地）。

    Parameters
    ----------
    prompt : str
        提示词。
    model_name : str
        模型名称。
    api_key : str
        API key。
    api_url : str
        API 地址。
    lambda_val : float
        回响强度（API 模式仅基线）。
    max_tokens : int
        最大生成 Token 数。

    Returns
    -------
    运行结果
    """
    import urllib.request
    import json as _json

    mode = "echo" if lambda_val > 0 else "baseline"
    result = 运行结果(mode=mode)

    t_start = time.time()

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 1.0,
        }

        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            body = _json.loads(resp.read().decode("utf-8"))

        result.生成文本 = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", str(body))
        )
        result.语义熵 = -1  # API 不暴露 logits

    except Exception as e:
        result.生成文本 = f"[API 错误] {type(e).__name__}: {e}"

    result.耗时秒 = round(time.time() - t_start, 2)
    return result


# ══════════════════════════════════════════════════
# 对比运行入口
# ══════════════════════════════════════════════════


def 运行对比(
    prompt: str,
    模型名称: str,
    lambda值: float,
    情感筛选: bool,
    思考分离: bool,
    max_tokens: int,
    推理模式: str,
    api_key: str,
    api_url: str,
) -> tuple[str, str, str, str, str]:
    """
    同时运行基线和回响模式，返回对比结果。

    Returns
    -------
    tuple[str, str, str, str, str]
        (状态消息, 基线文本, 回响文本, 指标HTML, 日志HTML)
    """
    # 检查模型是否加载
    if 推理模式 == "local" and _模型缓存["model"] is None:
        return "⚠ 请先点击「加载模型」按钮", "", "", "", ""

    # ── 基线 ──
    try:
        if 推理模式 == "api":
            基线结果 = _运行API推理(
                prompt, 模型名称, api_key, api_url,
                lambda_val=0.0, max_tokens=max_tokens,
            )
        else:
            基线结果 = _运行本地推理(
                prompt, lambda_val=0.0,
                max_tokens=max_tokens,
            )
    except Exception as e:
        return f"基线推理失败: {e}", "", "", "", ""

    # ── 回响 ──
    try:
        if 推理模式 == "api":
            回响结果 = _运行API推理(
                prompt, 模型名称, api_key, api_url,
                lambda_val=lambda值, max_tokens=max_tokens,
            )
        else:
            回响结果 = _运行本地推理(
                prompt, lambda_val=lambda值,
                情感筛选=情感筛选, 思考分离=思考分离,
                max_tokens=max_tokens,
            )
    except Exception as e:
        return f"回响推理失败: {e}", "", "", "", ""

    # ── 记录日志 ──
    _记录运行({
        "模型": 模型名称,
        "提示词": prompt[:50] + ("..." if len(prompt) > 50 else ""),
        "λ": lambda值,
        "基线熵": 基线结果.语义熵,
        "回响熵": 回响结果.语义熵,
        "基线耗时": 基线结果.耗时秒,
        "回响耗时": 回响结果.耗时秒,
        "池大小": 回响结果.回响池大小,
        "情感命中率": round(回响结果.情感命中率 * 100, 1),
    })

    # ── 指标展示 ──
    提升率 = ""
    if 基线结果.语义熵 > 0 and 回响结果.语义熵 > 0:
        delta = ((回响结果.语义熵 - 基线结果.语义熵) / 基线结果.语义熵) * 100
        提升率 = f"{delta:+.1f}%"

    指标_html = f"""
    <div style="display:flex; gap:20px; flex-wrap:wrap;">
      <div style="background:#f0f4ff; padding:12px 20px; border-radius:10px; min-width:140px;">
        <div style="font-size:12px; color:#666;">基线语义熵</div>
        <div style="font-size:24px; font-weight:bold;">{基线结果.语义熵:.4f}</div>
      </div>
      <div style="background:#fff0f0; padding:12px 20px; border-radius:10px; min-width:140px;">
        <div style="font-size:12px; color:#666;">回响语义熵</div>
        <div style="font-size:24px; font-weight:bold;">{回响结果.语义熵:.4f}</div>
      </div>
      <div style="background:{'#e8ffe8' if 提升率.startswith('+') else '#fff8e8'}; padding:12px 20px; border-radius:10px; min-width:140px;">
        <div style="font-size:12px; color:#666;">细腻度提升率</div>
        <div style="font-size:24px; font-weight:bold;">{提升率}</div>
      </div>
      <div style="background:#f5f5f5; padding:12px 20px; border-radius:10px; min-width:140px;">
        <div style="font-size:12px; color:#666;">回响池大小</div>
        <div style="font-size:24px; font-weight:bold;">{回响结果.回响池大小}</div>
      </div>
    </div>
    <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top:10px;">
      <div style="background:#f5f5f5; padding:8px 16px; border-radius:8px;">
        基线耗时: <b>{基线结果.耗时秒}s</b>
      </div>
      <div style="background:#f5f5f5; padding:8px 16px; border-radius:8px;">
        回响耗时: <b>{回响结果.耗时秒}s</b>
      </div>
      <div style="background:#f5f5f5; padding:8px 16px; border-radius:8px;">
        有效温度: <b>{回响结果.有效温度:.3f}</b>
      </div>
      <div style="background:#f5f5f5; padding:8px 16px; border-radius:8px;">
        情感命中率: <b>{回响结果.情感命中率*100:.1f}%</b>
      </div>
    </div>
    """

    # ── 日志展示 ──
    logs = _取日志()[-10:]  # 最近 10 条
    日志_html = "<table style='width:100%; border-collapse:collapse; font-size:13px;'>"
    日志_html += "<tr style='background:#eee;'><th>时间</th><th>模型</th><th>λ</th><th>基线熵</th><th>回响熵</th><th>提升率</th></tr>"
    for log in reversed(logs):
        r = ""
        if log["基线熵"] > 0 and log["回响熵"] > 0:
            dr = ((log["回响熵"] - log["基线熵"]) / log["基线熵"]) * 100
            r = f"{dr:+.1f}%"
        日志_html += f"<tr><td>{log['timestamp'][-8:]}</td><td>{log['模型'][:20]}</td><td>{log['λ']}</td><td>{log['基线熵']:.3f}</td><td>{log['回响熵']:.3f}</td><td>{r}</td></tr>"
    日志_html += "</table>"

    return "✓ 对比完成", 基线结果.生成文本, 回响结果.生成文本, 指标_html, 日志_html


# ══════════════════════════════════════════════════
# Gradio Web UI
# ══════════════════════════════════════════════════


def _创建界面() -> "gr.Blocks":
    """创建 Gradio 界面。"""
    import gradio as gr

    with gr.Blocks(
        title="语义回响 V2 — 交互式演示平台",
        theme=gr.themes.Soft(),
        css="""
        .对比框 { border: 2px solid #e0e0e0; border-radius: 12px; padding: 16px !important; }
        .基线区 { border-left: 4px solid #4ECDC4; }
        .回响区 { border-left: 4px solid #FF6B6B; }
        footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # 🧠 语义回响 V2 — 交互式演示平台

            **Semantic Echo: Enhancing LLM Expressiveness by Recycling Discarded Token Embeddings**

            在左侧选择/输入模型，点击「加载模型」后，输入提示词并点击「运行对比」，
            即可实时看到 **基线 vs 语义回响** 的生成效果差异。
            """
        )

        with gr.Row():
            # ── 左侧：控制面板 ──
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 配置")

                推理模式 = gr.Radio(
                    choices=["local (本地 GPU 推理)", "api (远程 API)"],
                    value="local (本地 GPU 推理)",
                    label="推理模式",
                )

                with gr.Group():
                    模型选择 = gr.Dropdown(
                        choices=_扫描兼容模型(),
                        value=_扫描兼容模型()[0] if _扫描兼容模型() else "",
                        label="选择模型",
                        allow_custom_value=True,
                    )
                    加载按钮 = gr.Button("📥 加载模型", variant="primary")
                    加载状态 = gr.Textbox(label="加载状态", interactive=False)

                with gr.Group(visible=False) as api配置区:
                    api_key输入 = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="sk-... 或 hf_...",
                    )
                    api_url输入 = gr.Textbox(
                        label="API 地址",
                        value="https://api.openai.com/v1/chat/completions",
                    )

                def on_mode_change(mode: str):
                    return gr.update(visible="api" in mode)

                推理模式.change(on_mode_change, 推理模式, api配置区)

                gr.Markdown("### 🎛️ 回响参数")
                λ滑块 = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.5, step=0.1,
                    label="λ (回响强度)",
                )
                情感筛选开关 = gr.Checkbox(value=True, label="情感词库筛选")
                思考分离开关 = gr.Checkbox(value=False, label="思考阶段分离注入")

                gr.Markdown("### 📝 输入")
                提示词框 = gr.Textbox(
                    value="请告诉我一件让你开心的事情。",
                    label="提示词",
                    lines=3,
                )
                max_token滑块 = gr.Slider(
                    minimum=16, maximum=256, value=128, step=16,
                    label="最大生成 Token 数",
                )

                运行按钮 = gr.Button("▶️ 运行对比", variant="primary", size="lg")

                # 模型加载
                def _on_load(model_name: str, mode: str, api_key: str, api_url: str):
                    if "api" in mode:
                        return _加载模型(model_name, mode="api", api_key=api_key, api_url=api_url)
                    return _加载模型(model_name, mode="local")
                加载按钮.click(
                    _on_load,
                    inputs=[模型选择, 推理模式, api_key输入, api_url输入],
                    outputs=加载状态,
                )

            # ── 右侧：结果展示 ──
            with gr.Column(scale=2):
                gr.Markdown("### 📊 对比结果")
                状态提示 = gr.Textbox(label="状态", interactive=False)

                with gr.Row():
                    with gr.Column():
                        基线输出 = gr.Textbox(
                            label="📗 基线 (Baseline)",
                            lines=8,
                            interactive=False,
                            elem_classes=["对比框", "基线区"],
                        )
                    with gr.Column():
                        回响输出 = gr.Textbox(
                            label="📕 语义回响 (Semantic Echo)",
                            lines=8,
                            interactive=False,
                            elem_classes=["对比框", "回响区"],
                        )

                指标展示 = gr.HTML(label="📈 实时指标")
                日志展示 = gr.HTML(label="📋 运行记录")

                # 运行
                def _on_run(prompt, model_name, lambda_val, sent_filter, think_sep, max_tok, mode, api_key, api_url):
                    if _模型缓存["model"] is None and "local" in mode:
                        return "⚠ 请先点击「加载模型」按钮", "", "", "", ""

                    return 运行对比(
                        prompt=prompt,
                        模型名称=model_name,
                        lambda值=lambda_val,
                        情感筛选=sent_filter,
                        思考分离=think_sep,
                        max_tokens=max_tok,
                        推理模式="local" if "local" in mode else "api",
                        api_key=api_key,
                        api_url=api_url,
                    )

                运行按钮.click(
                    _on_run,
                    inputs=[
                        提示词框, 模型选择, λ滑块,
                        情感筛选开关, 思考分离开关,
                        max_token滑块, 推理模式,
                        api_key输入, api_url输入,
                    ],
                    outputs=[状态提示, 基线输出, 回响输出, 指标展示, 日志展示],
                )

        # ── 底部：论文与版权 ──
        gr.Markdown(
            """
            ---
            <div style="text-align:center; color:#888; font-size:13px;">
              <b>语义回响 (Semantic Echo)</b> —
              通过回收被丢弃Token嵌入增强语言模型表达能力<br>
              作者: 邓同学 · 联系: DYPUBG2025@QQ.COM ·
              <a href="https://github.com/091635Aa/SemanticEcho" target="_blank">GitHub</a>
            </div>
            """
        )

    return demo


# ══════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════


def main() -> None:
    """启动 Gradio Web 演示平台。"""
    try:
        import gradio as gr
    except ImportError:
        print("请安装 gradio: pip install gradio")
        sys.exit(1)

    demo = _创建界面()
    print(f"""
╔══════════════════════════════════════════════╗
║       语义回响 V2 — 交互式演示平台          ║
║                                            ║
║  启动后请浏览器访问:                        ║
║    http://localhost:7860                    ║
║                                            ║
║  步骤:                                     ║
║    1. 选择/输入模型名称                     ║
║    2. 点击「加载模型」                      ║
║    3. 输入提示词                            ║
║    4. 点击「运行对比」查看效果              ║
╚══════════════════════════════════════════════╝
""")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
