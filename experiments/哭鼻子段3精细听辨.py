# -*- coding: utf-8 -*-
"""
段3（13:00-15:13 哭鼻子发生段）· 精细分段听辨
================================================
重新喂入 Qwen3-Omni，要求：
- 以 10-15 秒为粒度逐段仔细听辨
- 对每个非语言事件（吸鼻子/抽泣/哽咽/破音/叹息）给出【精确到秒】的时间戳与声音特征
- 重点回答"哪一段哭腔最明显、哭腔如何一步步出现"
- 输出自然语言详细分段报告（不要求 JSON）
"""
import os, sys, json, time, base64, urllib.request

服务地址 = "http://127.0.0.1:8766"
wav路径 = r"i:\Desktop\语义回响\实验数据\哭鼻子分析\段3_1300_1513.wav"
已知背景 = ""
标题 = "段3（13:00-15:13）"

系统提示词 = (
    "你是一位音频鉴听专家，擅长捕捉极其微弱的哭腔声学痕迹。本次任务需要你【极其仔细地聆听】"
    "一段约133秒的音频，把每个声学细节都标出来。请用自然语言输出，不要JSON，不要代码块标记。\n"
    "听辨要求：\n"
    "1. 把整段按 10-15 秒为粒度切段，逐段描述；\n"
    "2. 对每一个非语言发声事件（吸鼻子、抽泣、哽咽、破音、颤音、叹息、笑声、清嗓子、呼吸加重），"
    "给出精确到秒的时间戳（段内第几秒到第几秒），并描述它的声音特征（是短促吸气声/鼻音重/声音断裂/气息不稳等）；\n"
    "3. 特别关注：哭腔是何时开始出现的？从正常的唱歌发声到含泪发声的转折点在哪一秒？\n"
    "4. 指出哪一段哭腔最明显、最清晰，哪一段可能只有很轻微的痕迹；\n"
    "5. 对每个时间点，说明你听到的证据（如：XX秒处有明显鼻吸气声，类似抽鼻子后吸气的'呲'声）。"
)

用户提示词 = (
    "请极其仔细地听辨这段音频（约133秒，主播唱《素颜》尾段 + 13:39自述哭鼻子 + 下播互动）。\n"
    "已知背景：主播在唱完歌后说'你们有听到刚刚我在唱歌的时候哭鼻子了吗'，确认她哭了。\n"
    "已知时间锚点：段内约 93秒 处是'你们有听到刚刚我在唱歌的时候哭鼻子了吗'这句话。\n"
    "请你以 10-15 秒为粒度逐段听辨，输出：\n"
    "【段X | 段内X:XX-X:XX】\n"
    "- 该段说了什么/唱了什么（简要）\n"
    "- 人声状态（正常/含泪/哽咽/沙哑等）\n"
    "- 非语言事件列表（精确到秒）：如 吸鼻子 00:05-00:06（特征描述）、抽泣 01:08-01:09 等\n"
    "- 哭腔痕迹评估（无/轻微/明显/强烈）\n"
    "最后输出一个【哭腔转折点汇总】：从哪一秒开始声音不再正常、哪一秒是哭腔最清晰的瞬间。\n"
    "请尽量详细，把每个可疑的声音都标出来。"
)

def 调用(wav, 用户提示词):
    with open(wav, "rb") as f:
        音频数据 = base64.b64encode(f.read()).decode("utf-8")
    消息 = [
        {"role": "system", "content": 系统提示词},
        {"role": "user", "content": [
            {"type": "text", "text": 用户提示词},
            {"type": "input_audio", "input_audio": {"data": 音频数据, "format": "wav"}},
        ]},
    ]
    请求体 = {"model": "qwen3-omni", "messages": 消息, "max_tokens": 4096,
              "temperature": 0.4, "top_p": 0.9, "stream": False}
    请求 = urllib.request.Request(
        服务地址 + "/v1/chat/completions",
        data=json.dumps(请求体).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    print(f"[听辨] 发送精细分段请求（{os.path.basename(wav)}）...", flush=True)
    t0 = time.time()
    with urllib.request.urlopen(请求, timeout=1800) as 响应:
        数据 = json.loads(响应.read().decode("utf-8"))
    文本 = (数据.get("choices") or [{}])[0].get("message", {}).get("content", "")
    print(f"[听辨] 完成，耗时 {round(time.time()-t0,1)}s，输出 {len(文本)} 字符", flush=True)
    return 文本

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="wav路径")
    ap.add_argument("--输出", required=True, help="输出md路径")
    ap.add_argument("--标题", required=True, help="报告标题，如 段1（8:39-11:00）")
    ap.add_argument("--时长秒", type=int, required=True, help="音频时长秒")
    ap.add_argument("--背景", default="", help="已知背景线索")
    args = ap.parse_args()

    用户提示词 = (
        f"请极其仔细地听辨这段音频（约{args.时长秒}秒）。\n"
        f"已知背景：{args.背景}\n"
        "请你以 10-15 秒为粒度逐段听辨，输出：\n"
        "【段X | 段内X:XX-X:XX】\n"
        "- 该段说了什么/唱了什么（简要）\n"
        "- 人声状态（正常/含泪/哽咽/沙哑等）\n"
        "- 非语言事件列表（精确到秒）：如 吸鼻子 00:05-00:06（特征描述）、抽泣 01:08-01:09 等\n"
        "- 哭腔痕迹评估（无/轻微/明显/强烈）\n"
        "最后输出一个【哭腔转折点汇总】：从哪一秒开始声音不再正常、哪一秒是哭腔最清晰的瞬间。\n"
        "请尽量详细，把每个可疑的声音都标出来。"
    )
    文本 = 调用(args.wav, 用户提示词)
    报告 = f"# {args.标题} 精细分段听辨报告\n\n"
    报告 += f"> 时间：{time.strftime('%Y-%m-%d %H:%M:%S')} | 模型：Qwen3-Omni-30B | 粒度：10-15秒/段 | 音频：{os.path.basename(args.wav)}（{args.时长秒}s）\n\n"
    报告 += 文本 + "\n"
    with open(args.输出, "w", encoding="utf-8") as f:
        f.write(报告)
    print(f"[输出] -> {args.输出}")
