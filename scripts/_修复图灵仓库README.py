# -*- coding: utf-8 -*-
"""修复 TuringTest-Data 仓库描述乱码 + 添加英文 README（UTF-8 安全）"""
import base64
import os
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TOKEN = os.environ.get("GITHUB_TOKEN", "")
if not TOKEN:
    raise SystemExit("请先设置环境变量 GITHUB_TOKEN")
REPO = "091635Aa/TuringTest-Data"
API = f"https://api.github.com/repos/{REPO}"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}
session = requests.Session()
session.verify = False

# 1. 修复仓库描述（UTF-8 正确编码）
描述 = "语义回响（Semantic Echo）图灵测试完整数据集：TuringBench 中文体系图灵检测 + HeartBench/HEART-BENCH 情感基准 + 测试脚本与结果"
r = session.patch(f"{API}", headers=HEADERS, json={"description": 描述}, timeout=60)
if r.status_code == 200:
    print(f"[1/3] 仓库描述已修复: {r.json().get('description')}")
else:
    print(f"[1/3] 修复失败: {r.status_code} {r.text[:200]}")
    raise SystemExit(1)

# 2. 英文 README 内容
readme_en = r"""# TuringTest-Data

Complete Turing-Test dataset for the Semantic Echo project (Chinese Turing detection + emotion benchmarks).

> Main repo: [091635Aa/SemanticEcho](https://github.com/091635Aa/SemanticEcho)
> Showcase repo: [091635Aa/1.5B-Turing-Challenge](https://github.com/091635Aa/1.5B-Turing-Challenge)
> 中文版: [README.md](README.md)

## Directory Layout

```
├── data/
│   ├── turingbench.zip        # ⚠️ Oversized file (228MB) -> download from Release
│   ├── turingbench/           # Extracted TuringBench data
│   │   ├── TuringBench/
│   │   │   ├── AA/            # Human-written text (train.csv 135MB via Release; test/valid in repo)
│   │   │   └── TT_*/          # Machine text: GPT-1/GPT-2/GPT-3/Grover/XLNet/Transfo-XL/XLM etc.
│   ├── emocharacter_results.json      # EmoCharacter role-play emotional fidelity results
│   ├── feel_heart_results.json        # HEART-BENCH memory-driven personality inference results
│   ├── heartbench_results.json        # HeartBench Chinese "human-ness" benchmark results
│   ├── llm_judge_results.json         # LLM-as-Judge blind evaluation results
│   ├── turingbench_results.json       # TuringBench Chinese-system Turing detection results
│   ├── cross_model_comparison.json    # Cross-model comparison
│   ├── judge_bias_analysis.json       # Judge bias analysis
│   └── 淘汰记录.json                    # Eliminated model records
├── repos/                     # HeartBench / HEART-BENCH official benchmark repos
├── logs/                      # Benchmark run logs
├── results/                   # summary.json + report.md
├── 实验/                       # Phase results & experiment logs
├── 0_准备数据.py              # Data preparation script
├── run_*.py                   # Benchmark entry points
├── 公共模块.py / 生成器.py / 早停.py
└── 验证_*.py                  # Reproduction / judge-bias / cross-model scripts
```

## ⚠️ Oversized Files (>100MB, git limit)

These files exceed GitHub's 100MB per-file push limit, so they live in **Release v1.0.0**:

| File | Size | Release asset name | Description |
|---|---|---|---|
| `data/turingbench.zip` | 228MB | `turingbench.zip` | Official TuringBench data archive |
| `data/turingbench/TuringBench/AA/train.csv` | 135MB | `AA_train.csv` | Human-annotated training set |

**Download:** [Release v1.0.0](https://github.com/091635Aa/TuringTest-Data/releases/tag/v1.0.0)

Place them back to restore the full dataset:

```
turingbench.zip   ->  data/turingbench.zip
AA_train.csv      ->  data/turingbench/TuringBench/AA/train.csv
```

## Usage

1. Download the oversized files from Release and restore their paths
2. Run `0_准备数据.py` to prepare data
3. Run `run_turingbench.py` / `run_emocharacter.py` / `run_heartbench.py` / `run_feel_heart.py` / `run_llm_judge.py` to execute each benchmark
4. Run `run_all.py` to generate the summary report (`results/summary.json` + `results/报告.md`)
"""

# 检查 README.en.md 是否已存在（存在则需 sha 更新）
sha = None
r = session.get(f"{API}/contents/README.en.md", headers=HEADERS, timeout=60)
if r.status_code == 200:
    sha = r.json().get("sha")
    print(f"[2/3] README.en.md 已存在 (sha={sha[:7]}...)，将更新")
else:
    print("[2/3] README.en.md 不存在，将新建")

body = {
    "message": "add README.en.md: English version of TuringTest-Data docs",
    "content": base64.b64encode(readme_en.encode("utf-8")).decode(),
    "branch": "main",
}
if sha:
    body["sha"] = sha
r = session.put(f"{API}/contents/README.en.md", headers=HEADERS, json=body, timeout=120)
if r.status_code in (200, 201):
    print(f"[2/3] README.en.md 写入成功 -> commit {r.json().get('commit', {}).get('sha', '')[:7]}")
else:
    print(f"[2/3] 写入失败: {r.status_code} {r.text[:300]}")
    raise SystemExit(1)

# 3. 更新中文 README 顶部，加语言切换链接
r = session.get(f"{API}/contents/README.md", headers=HEADERS, timeout=60)
r.raise_for_status()
cur = r.json()
旧内容 = base64.b64decode(cur["content"]).decode("utf-8")
链接行 = "> English: [README.en.md](README.en.md)"
if "README.en.md" not in 旧内容:
    新内容 = 旧内容.replace(
        "> 展示仓库：[091635Aa/1.5B-Turing-Challenge](https://github.com/091635Aa/1.5B-Turing-Challenge)",
        "> 展示仓库：[091635Aa/1.5B-Turing-Challenge](https://github.com/091635Aa/1.5B-Turing-Challenge)\n"
        f"> English: [README.en.md](README.en.md)")
    body = {
        "message": "README: add English README link",
        "content": base64.b64encode(新内容.encode("utf-8")).decode(),
        "sha": cur["sha"],
        "branch": "main",
    }
    r = session.put(f"{API}/contents/README.md", headers=HEADERS, json=body, timeout=120)
    if r.status_code in (200, 201):
        print(f"[3/3] README.md 已加入英文链接 -> commit {r.json().get('commit', {}).get('sha', '')[:7]}")
    else:
        print(f"[3/3] README.md 更新失败: {r.status_code} {r.text[:300]}")
else:
    print("[3/3] README.md 已有英文链接，跳过")

print("\n===== 完成 =====")
print(f"仓库: https://github.com/{REPO}")
