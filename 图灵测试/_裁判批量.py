# -*- coding: utf-8 -*-
"""批量裁判 v2（分块自愈）：一次加载裁判模型，分块处理指定模型全部模式，逐块重试，最终汇总更新 json

用法: python _裁判批量.py <模型名> [裁判模型]
裁判模型默认 Qwen2.5-3B-Instruct
"""
import os, sys, json, subprocess, gc, time

本目录 = os.path.dirname(os.path.abspath(__file__))
统一目录 = os.path.join(本目录, "统一基准")
样本路径 = os.path.join(本目录, "样本_30条.json")
输出路径 = os.path.join(统一目录, "泛化测试_2026.json")

模型名 = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-4B"
裁判模型 = sys.argv[2] if len(sys.argv) > 2 else "Qwen2.5-3B-Instruct"
缓存路径 = os.path.join(统一目录, f"泛化回复_{模型名}_全模式.json")
全模式列表 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响",
             "P4_KV共振", "P5_超融合", "P6_情感导演"]
每块模式数 = 2  # 8 模式 → 4 块


def 清内存():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


回复 = json.load(open(缓存路径, encoding="utf-8"))
样本 = json.load(open(样本路径, encoding="utf-8"))["样本"][:12]

# 每模式请求：(配对x2, 评分x1) → 36 条
def 构建请求(模式们):
    请求, 映射 = [], []
    for 模式 in 模式们:
        for i in range(12):
            请求.append({"类型": "llm_judge_配对", "user": 样本[i]["user"],
                         "AI": 回复[模式][i], "真人": 样本[i]["girl"], "AI在前": True})
            映射.append((模式, i, "配对", 0))
            请求.append({"类型": "llm_judge_配对", "user": 样本[i]["user"],
                         "AI": 回复[模式][i], "真人": 样本[i]["girl"], "AI在前": False})
            映射.append((模式, i, "配对", 1))
            请求.append({"类型": "llm_judge_评分", "user": 样本[i]["user"],
                         "回复": 回复[模式][i]})
            映射.append((模式, i, "评分", 0))
    return 请求, 映射


def 跑块(请求):
    任务路径 = os.path.join(统一目录, "_批量_任务.json")
    输出路径_tmp = os.path.join(统一目录, "_批量_输出.json")
    json.dump({"裁判": "llm_judge_混合", "请求": 请求}, open(任务路径, "w", encoding="utf-8"),
              ensure_ascii=False)
    env = dict(os.environ, 裁判模型=裁判模型)
    最后错误 = None
    for 尝试 in range(5):
        try:
            subprocess.run([sys.executable, os.path.join(本目录, "裁判子进程.py"),
                            任务路径, 输出路径_tmp], env=env, check=True)
            最后错误 = None
            break
        except subprocess.CalledProcessError as e:
            最后错误 = e
            print(f"  [批量裁判] 块失败（第{尝试+1}/5 次）exit={e.returncode}，等待 30s 重试", flush=True)
            time.sleep(30)
            清内存()
    if 最后错误 is not None:
        raise 最后错误
    return json.load(open(输出路径_tmp, encoding="utf-8"))


汇总 = {}
可用模式 = [m for m in 全模式列表 if m in 回复 and len(回复[m]) >= 12]
print(f"[批量裁判] {模型名} 共 {len(可用模式)} 模式 / {len(可用模式)*36} 条请求，裁判 {裁判模型}，分 {max(1, len(可用模式)//每块模式数)} 块", flush=True)

块们 = [可用模式[k:k+每块模式数] for k in range(0, len(可用模式), 每块模式数)]
for 块号, 模式们 in enumerate(块们):
    清内存()
    print(f"[块 {块号+1}/{len(块们)}] 模式 {模式们} ...", flush=True)
    请求, 映射 = 构建请求(模式们)
    try:
        输出 = 跑块(请求)
    except Exception as e:
        print(f"[块 {块号+1}] 重试后仍失败：{e}", flush=True)
        continue
    assert len(输出) == len(映射), f"结果数 {len(输出)} != {len(映射)}"
    结果表 = {}
    for (模式, i, 类型, 序号), 结果 in zip(映射, 输出):
        if 模式 not in 结果表:
            结果表[模式] = {"ai胜": [False]*12, "评分": [None]*12}
        if 类型 == "配对":
            if 结果.get("AI胜"):
                结果表[模式]["ai胜"][i] = True
        else:
            结果表[模式]["评分"][i] = 结果.get("评分")
    for 模式 in 模式们:
        ai胜数 = sum(结果表[模式]["ai胜"])
        评分列表 = [s for s in 结果表[模式]["评分"] if s]
        win = round(ai胜数 / 12, 4)
        rating = round(sum(评分列表) / len(评分列表) / 5, 4) if 评分列表 else 0.0
        汇总[模式] = {
            "win_rate_against_human": win,
            "average_rating": rating,
            "average_rating_raw": round(sum(评分列表) / len(评分列表), 2) if 评分列表 else 0.0,
        }
        print(f"  [{模型名} {模式}] win={win} rating={rating} (raw={汇总[模式]['average_rating_raw']})", flush=True)
    # 每块后即时保存，避免全丢
    已有 = json.load(open(输出路径, encoding="utf-8"))
    已有["模式汇总"][模型名] = {**已有["模式汇总"].get(模型名, {}), **汇总}
    json.dump(已有, open(输出路径, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"[批量裁判] 完成 -> {输出路径}", flush=True)
