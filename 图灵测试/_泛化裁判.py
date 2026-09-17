# -*- coding: utf-8 -*-
"""泛化测试裁判阶段（独立进程，读检查点 → 7B 裁判 → 汇总）
用法: python _泛化裁判.py <模型名> [样本数]
"""
import os, sys, json, subprocess, time
本目录 = os.path.dirname(os.path.abspath(__file__))
统一目录 = os.path.join(本目录, "统一基准")
样本路径 = os.path.join(本目录, "样本_30条.json")
模型名 = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-3B-Instruct"
样本数 = int(sys.argv[2]) if len(sys.argv) > 2 else 12
检查点路径 = os.path.join(统一目录, f"泛化回复_{模型名}.json")
输出路径 = os.path.join(统一目录, "泛化测试_2026.json")

with open(样本路径, encoding="utf-8") as f:
    样本 = json.load(f)["样本"][:样本数]
with open(检查点路径, encoding="utf-8") as f:
    回复 = json.load(f)
print(f"[裁判] {模型名} 检查点: 裸 {len(回复.get('裸', []))} / P6 {len(回复.get('P6_情感导演', []))} 条")


def 跑裁判(裁判类型, 请求列表):
    """带重试：bnb 4bit 加载偶发段错误，等内存回落重试"""
    任务路径 = os.path.join(统一目录, "_泛化_任务.json")
    输出路径 = os.path.join(统一目录, "_泛化_输出.json")
    with open(任务路径, "w", encoding="utf-8") as f:
        json.dump({"裁判": 裁判类型, "请求": 请求列表}, f, ensure_ascii=False)
    最后错误 = None
    for 尝试 in range(5):
        try:
            subprocess.run([sys.executable, os.path.join(本目录, "裁判子进程.py"),
                            任务路径, 输出路径], check=True)
            with open(输出路径, encoding="utf-8") as f:
                return json.load(f)
        except subprocess.CalledProcessError as e:
            最后错误 = e
            print(f"[裁判] 子进程失败(第{尝试+1}次) exit={e.returncode}，等 20s 重试", flush=True)
            time.sleep(20)
    raise 最后错误


汇总 = {}
for 模式 in ("裸", "P6_情感导演"):
    配对请求 = []
    for i, r in enumerate(样本):
        for AI在前 in (True, False):
            配对请求.append({"user": r["user"], "AI": 回复[模式][i],
                              "真人": r["girl"], "AI在前": AI在前})
    print(f"[裁判] {模式} 配对 {len(配对请求)} 条 ...", flush=True)
    配对结果 = 跑裁判("llm_judge_配对", 配对请求)
    ai胜 = sum(1 for i in range(样本数)
               if 配对结果[i * 2].get("AI胜") or 配对结果[i * 2 + 1].get("AI胜"))
    评分请求 = [{"user": r["user"], "回复": 回复[模式][i]} for i, r in enumerate(样本)]
    print(f"[裁判] {模式} 评分 {len(评分请求)} 条 ...", flush=True)
    评分结果 = 跑裁判("llm_judge_评分", 评分请求)
    ai评分 = [c.get("评分") for c in 评分结果 if c.get("评分")]
    win = round(ai胜 / 样本数, 4)
    rating = round(sum(ai评分) / len(ai评分) / 5, 4) if ai评分 else 0.0
    汇总[模式] = {"win_rate_against_human": win, "average_rating": rating,
                 "average_rating_raw": round(sum(ai评分) / len(ai评分), 2) if ai评分 else 0.0}
    print(f"[{模型名} {模式}] win={win} rating={rating}", flush=True)

全部 = {}
if os.path.exists(输出路径):
    try:
        with open(输出路径, encoding="utf-8") as f:
            全部 = json.load(f).get("模式汇总", {})
    except Exception:
        pass
全部[模型名] = 汇总
with open(输出路径, "w", encoding="utf-8") as f:
    json.dump({"模式汇总": 全部, "样本数": 样本数,
               "说明": "各模型上 裸 vs P6_情感导演 的 LLM-Judge（win_rate 与 rating）"},
              f, ensure_ascii=False, indent=2)
print(f"\n结果已保存 -> {输出路径}")
print(f"[{模型名}] 裸: {汇总['裸']}")
print(f"[{模型名}] P6: {汇总['P6_情感导演']}")
