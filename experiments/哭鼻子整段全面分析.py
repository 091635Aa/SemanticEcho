# -*- coding: utf-8 -*-
"""
三合一整段（8:39-15:13）· 全面压榨分析（max_tokens 8192）
============================================================
- 整段 394s 一次喂入 Qwen3-Omni（上下文 49152，KV q8_0 全力）
- 输出：超长全面分析 + 结构化情绪时间线事件表（供 Mermaid 流程图）
"""
import os, sys, json, time, base64, urllib.request, re

服务地址 = "http://127.0.0.1:8766"
wav路径 = r"i:\Desktop\语义回响\实验数据\哭鼻子分析\整段_0839_1513.wav"

系统提示词 = (
    "你是一位专业音频情感分析专家，精通声学信号分析与语音情感识别。本次是【全面压榨分析】："
    "请你把整段音频的每一个情绪细节都挖出来，最大限度输出（可以写很长很长）。\n"
    "输出格式：\n"
    "第一部分【全面分析】（自然语言，越长越好）：整体听感、哭腔演变全程、每个情绪阶段的声学证据、"
    "说话段与唱歌段的情绪差异、最终综合判断。\n"
    "第二部分【情绪时间线】（供画流程图，严格用如下格式，每行一条）：\n"
    "【事件】绝对时间|情绪状态|事件描述|哭腔强度\n"
    "其中：\n"
    "- 绝对时间：用 分:秒（如 8:56），基于音频从 8:39 开始计算\n"
    "- 情绪状态：用关键词（平静/微伤感/伤感/含泪/哽咽/哭腔明显/哭泣/委屈/自嘲/无奈/平静收尾 等）\n"
    "- 事件描述：一句话（如 首次吸鼻子、唱到'又想你到泪流'、13:39自述哭鼻子等）\n"
    "- 哭腔强度：无/轻微/明显/强烈\n"
    "请务必输出完整的第二部分，覆盖从 8:39 到 15:13 的完整情绪变化链。"
)

用户提示词 = (
    "请对这段【完整音频】（约394秒 = 源文件8:39~15:13，主播唱歌+互动+下播全程，含《幻听》《素颜》演唱、"
    "13:39'你们有听到刚刚我在唱歌的时候哭鼻子了吗'自述、14:36'你们都不安慰我'）进行全面压榨分析。\n"
    "已知背景：音频声学分析已确认哭腔真实存在（吸鼻子/抽泣/哽咽）。\n"
    "请你：\n"
    "1. 完整写出全面分析（情绪演变、哭腔证据、说话与唱歌差异、综合判断）；\n"
    "2. 输出【情绪时间线】事件表（严格按格式），覆盖全程每个情绪转折点。\n"
    "输出要尽量长、尽量详细。"
)

def 调用():
    with open(wav路径, "rb") as f:
        音频数据 = base64.b64encode(f.read()).decode("utf-8")
    消息 = [
        {"role": "system", "content": 系统提示词},
        {"role": "user", "content": [
            {"type": "text", "text": 用户提示词},
            {"type": "input_audio", "input_audio": {"data": 音频数据, "format": "wav"}},
        ]},
    ]
    请求体 = {"model": "qwen3-omni", "messages": 消息, "max_tokens": 8192,
              "temperature": 0.4, "top_p": 0.9, "stream": False}
    请求 = urllib.request.Request(
        服务地址 + "/v1/chat/completions",
        data=json.dumps(请求体).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    print("[全面分析] 发送整段394s请求（max_tokens=8192）...", flush=True)
    t0 = time.time()
    with urllib.request.urlopen(请求, timeout=2400) as 响应:
        数据 = json.loads(响应.read().decode("utf-8"))
    文本 = (数据.get("choices") or [{}])[0].get("message", {}).get("content", "")
    print(f"[全面分析] 完成，耗时 {round(time.time()-t0,1)}s，输出 {len(文本)} 字符", flush=True)
    return 文本

def 提取事件(文本):
    """解析【事件】行 → [(时间, 情绪, 描述, 强度)]"""
    事件 = []
    for 行 in 文本.splitlines():
        行 = 行.strip()
        if 行.startswith("【事件】"):
            内容 = 行.replace("【事件】", "").strip()
            部分 = [p.strip() for p in 内容.split("|")]
            if len(部分) >= 3:
                事件.append((部分[0], 部分[1], 部分[2], 部分[3] if len(部分) > 3 else ""))
    return 事件

if __name__ == "__main__":
    文本 = 调用()
    事件 = 提取事件(文本)
    记录 = {
        "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "模型": "Qwen3-Omni-30B (llama-server Q3_K_M, KV q8_0, ctx49152)",
        "音频": os.path.basename(wav路径),
        "事件数": len(事件),
        "事件": [{"时间": e[0], "情绪": e[1], "描述": e[2], "强度": e[3]} for e in 事件],
        "原始输出": 文本,
    }
    out = r"i:\Desktop\语义回响\实验数据\哭鼻子分析\整段_全面压榨分析.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(记录, f, ensure_ascii=False, indent=2)
    print(f"[输出] -> {out}（事件 {len(事件)} 条）")
