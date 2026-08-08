# -*- coding: utf-8 -*-
"""
哭鼻子片段 · 最大化多模态分析（3S 级紧急任务）
================================================
调用已预加载的 Qwen3-Omni-30B 多模态服务（llama-server OpenAI 兼容接口），
对重点片段做超长思考链的详细情感分析：
- 声学线索扫描（音高颤抖/气息/响度/语速/停顿/非语言发声）
- 哭腔证据链（cry_type：抽泣/哽咽/含泪）
- 情感打标（VAD/主副情绪/强度/演变）
- 输出符合打标系统 Schema 的 JSON

用法：
    f:\\打标\\.venv\\Scripts\\python.exe experiments\\哭鼻子最大化分析.py ^
        --wav "i:\\Desktop\\语义回响\\实验数据\\哭鼻子分析\\段3_1300_1513.wav" ^
        --输出 "i:\\Desktop\\语义回响\\实验数据\\哭鼻子分析\\段3_重点_result.json" ^
        --重点
"""
import os
import sys
import json
import time
import base64
import argparse
import urllib.request

服务地址 = "http://127.0.0.1:8766"

# ── 最大化分析 · 系统提示词（超长思考链）──
系统提示词 = (
    "你是一位专业音频情感分析专家，精通声学信号分析与语音情感识别，尤其擅长捕捉"
    "微弱的哭腔、哽咽、抽泣等非语言情感信号。本次任务是【最大化深度分析】："
    "把思考链开到最长的详细程度，逐层推理，不遗漏任何声学与语义线索。\n"
    "【核心任务】判断该音频片段中主播（主角人声）是否有哭鼻子/哭腔迹象，以及整体的"
    "情感丰富度（情绪是否饱满、细腻、有层次）。\n"
    "【分析步骤（务必逐步在'思考'字段中详细呈现）】\n"
    "第1步 整体聆听：先说整体氛围与情绪基调；\n"
    "第2步 声学线索扫描：逐段检查音高（f0）是否有颤抖/上漂、气息是否不稳、"
    "响度起伏、语速快慢、停顿与呼吸声、是否有吸鼻子/抽泣/哽咽/破音/沙哑/颤音；\n"
    "第3步 语义线索：结合歌词与话语内容，判断情感触发点；\n"
    "第4步 哭腔证据链：明确给出 cry_type（抽泣/哽咽/含泪/无），并列出证据出现的"
    "具体时间点（秒）与声音特征；\n"
    "第5步 情感判定：主情绪、副情绪、valence 积极度、arousal 激活度、dominance 支配度、"
    "emotion_intensity 强度（1-5）、情感方向（积极/消极/中性/混合）；\n"
    "第6步 情感演变与丰富度：描述片段内情绪的起承转合，评估情感丰富度等级"
    "（低/中/高/极高）并给出理由；\n"
    "第7步 综合输出：按给定 Schema 输出完整 JSON。\n"
    "【要求】思考字段要极其详细（2000 字以上），把你的分析推理过程完整写出来："
    "你听到了什么、在哪里听到、如何判断、证据是什么、可能的误判与修正。\n"
    "只输出 JSON 对象，不要输出 JSON 以外的任何内容。"
)

# ── 用户提示词（含 Schema 引导 + 重点段关注哭腔）──
用户提示词模板 = (
    "请对这段音频（{时长秒} 秒）进行最大化深度情感分析。\n"
    "{重点提示}"
    "请按以下 Schema 输出严格合法的 JSON 对象：\n"
    "{{\n"
    "  \"思考\": \"超长思考链（2000字以上，逐层推理：整体聆听→声学线索扫描→语义线索→"
    "哭腔证据链→情感判定→情感演变与丰富度→综合结论）\",\n"
    "  \"内容描述\": \"50-100字一句话概括\",\n"
    "  \"主角声音\": \"一句话说明识别出的主角声音特征与依据（发声最长/最频繁/音色统一）\",\n"
    "  \"背景音乐情况\": \"一句话说明BGM存在情况\",\n"
    "  \"transcript\": \"完整文本转写\",\n"
    "  \"discrete_emotion_primary\": \"主情绪\",\n"
    "  \"discrete_emotion_secondary\": \"副情绪\",\n"
    "  \"valence\": 0.0~1.0,\n"
    "  \"arousal\": 0.0~1.0,\n"
    "  \"dominance\": 0.0~1.0,\n"
    "  \"emotion_intensity\": 1~5,\n"
    "  \"情感方向\": \"积极/消极/中性/混合\",\n"
    "  \"情感标签\": [\"至少5个\"],\n"
    "  \"情感演变\": \"片段内情感起伏过程\",\n"
    "  \"情感丰富度\": \"低/中/高/极高 + 理由\",\n"
    "  \"听众感受\": \"一句话\",\n"
    "  \"f0_mean_hz\": 0.0,\n"
    "  \"f0_std_hz\": 0.0,\n"
    "  \"f0_contour\": \"上升/下降/平缓/波浪\",\n"
    "  \"energy_rms_mean\": 0.0,\n"
    "  \"energy_dynamic_range\": 0.0,\n"
    "  \"speech_rate_syll_per_sec\": 0.0,\n"
    "  \"pause_ratio\": 0.0,\n"
    "  \"voice_quality\": \"清亮/沙哑/气声/鼻音/紧喉\",\n"
    "  \"prosodic_events\": [{{\"type\":\"重读/拖长音/顿挫/轻声/高亢/叹息/颤音\",\"start_sec\":0.0,\"end_sec\":0.0}}],\n"
    "  \"nonverbal_events\": [{{\"type\":\"吸鼻子/抽泣/哽咽/叹气/笑声/清嗓子\",\"start_sec\":0.0,\"end_sec\":0.0}}],\n"
    "  \"filled_pauses\": [\"嗯/啊/那个\"],\n"
    "  \"laugh_type\": \"大笑/偷笑/尬笑/轻笑/无\",\n"
    "  \"cry_type\": \"抽泣/哽咽/含泪/无\",\n"
    "  \"时间轴分段\": [{{\"段落序号\":1,\"开始秒\":0.0,\"结束秒\":0.0,\"段落类型\":\"连续话题\","
    "\"话题\":\"...\",\"transcript\":\"...\",\"话题标签\":[],\"情感标签\":[],\"情感方向\":\"...\","
    "\"声学打标\":{{\"f0_mean_hz\":0.0,\"f0_std_hz\":0.0,\"f0_contour\":\"...\","
    "\"energy_rms_mean\":0.0,\"speech_rate_syll_per_sec\":0.0,\"pause_ratio\":0.0,"
    "\"voice_quality\":\"...\",\"valence\":0.0,\"arousal\":0.0,\"dominance\":0.0,"
    "\"emotion_intensity\":0,\"prosodic_events\":[],\"nonverbal_events\":[],"
    "\"filled_pauses\":[],\"laugh_type\":\"...\",\"cry_type\":\"...\"}}}}],\n"
    "  \"时间轴事件\": [{{\"时间秒\":0.0,\"事件类型\":\"情感变化/风格转变/主题切换/高潮点/静默\",\"描述\":\"...\"}}],\n"
    "  \"置信度\": 0.0\n"
    "}}\n"
    "【时间轴分段要求】2-5 段，覆盖完整时间轴，每段的声学事件时间戳必须落在该段范围内。\n"
    "【重点】务必仔细判断 cry_type（抽泣/哽咽/含泪/无）：主播在 13:39 下播时自述"
    "\"你们有听到刚刚我在唱歌的时候哭鼻子了吗？\"——请在声学证据中寻找哭腔痕迹并交叉验证。\n"
)


def 调用分析(wav路径, 时长秒, 重点=False):
    with open(wav路径, "rb") as f:
        音频数据 = base64.b64encode(f.read()).decode("utf-8")
    重点提示 = ("【本段为哭鼻子重点段】请特别仔细捕捉：吸鼻子、抽泣、哽咽、气息不稳、"
              "音高颤抖、破音、鼻音加重等哭腔声学痕迹，并给时间戳。\n" if 重点 else "")
    用户提示词 = 用户提示词模板.format(时长秒=时长秒, 重点提示=重点提示)
    消息 = [
        {"role": "system", "content": 系统提示词},
        {"role": "user", "content": [
            {"type": "text", "text": 用户提示词},
            {"type": "input_audio", "input_audio": {"data": 音频数据, "format": "wav"}},
        ]},
    ]
    请求体 = {
        "model": "qwen3-omni",
        "messages": 消息,
        "max_tokens": 4096,
        "temperature": 0.5,
        "top_p": 0.9,
        "stream": False,
    }
    请求 = urllib.request.Request(
        服务地址 + "/v1/chat/completions",
        data=json.dumps(请求体).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    print(f"[分析] 发送请求（wav={os.path.basename(wav路径)} 时长={时长秒}s）...", flush=True)
    t0 = time.time()
    with urllib.request.urlopen(请求, timeout=900) as 响应:
        数据 = json.loads(响应.read().decode("utf-8"))
    文本 = (数据.get("choices") or [{}])[0].get("message", {}).get("content", "")
    耗时 = round(time.time() - t0, 1)
    print(f"[分析] 完成，耗时 {耗时}s，输出 {len(文本)} 字符", flush=True)
    return 文本, 耗时


def 主流程(args):
    文本, 耗时 = 调用分析(args.wav, args.时长秒, 重点=args.重点)
    记录 = {
        "wav": args.wav, "时长秒": args.时长秒, "重点段": bool(args.重点),
        "模型": "Qwen3-Omni-30B-A3B-Instruct (llama-server Q3_K_M)",
        "推理耗时秒": 耗时, "时间戳": time.strftime("%Y-%m-%d %H:%M:%S"),
        "原始输出": 文本,
    }
    # 尝试提取 JSON
    try:
        开始 = 文本.find("{")
        结束 = 文本.rfind("}")
        if 开始 != -1 and 结束 > 开始:
            记录["解析JSON"] = json.loads(文本[开始:结束 + 1])
    except Exception as e:
        记录["解析JSON错误"] = str(e)
    os.makedirs(os.path.dirname(args.输出), exist_ok=True)
    with open(args.输出, "w", encoding="utf-8") as f:
        json.dump(记录, f, ensure_ascii=False, indent=2)
    print(f"[输出] -> {args.输出}")
    return 记录


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="哭鼻子片段最大化分析")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--输出", required=True)
    ap.add_argument("--时长秒", type=float, required=True)
    ap.add_argument("--重点", action="store_true")
    args = ap.parse_args()
    主流程(args)
