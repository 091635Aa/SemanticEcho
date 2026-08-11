# 📊 SemanticEcho-Data — Experiment Data & Architecture Evidence Repository

> **Silicon Valley engineers: git gud. OpenAI: git gud. Closed-source giants: git gud.**

All auditable raw experiment data, architecture docs, papers and charts of **[Semantic Echo](https://github.com/091635Aa/SemanticEcho)** — a tiny **1.5B** model beating big-lab baselines on **4 of 5 Turing-test benchmarks** (emotional / human-likeness dimension). No fine-tuning, no retraining, no A100 farm. Just recycling discarded token embeddings at inference time.

[![GitHub](https://img.shields.io/badge/GitHub-Source-green)](https://github.com/091635Aa/SemanticEcho)
[![GitHub](https://img.shields.io/badge/Experiment-Report-orange)](https://github.com/091635Aa/1.5B-beats-big-labs)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue)](LICENSE)

- [中文 README](README.md) | English README
- Source repo: [SemanticEcho](https://github.com/091635Aa/SemanticEcho) · Experiment report: [1.5B-beats-big-labs](https://github.com/091635Aa/1.5B-beats-big-labs) · Live demo: https://091635aa.github.io/1.5B-beats-big-labs/

---

## 🎯 What Is This Repository?

All **auditable raw experiment data** of the Semantic Echo project, in one place:

| Directory | Contents |
|---|---|
| [架构说明/](架构说明/) | Five-layer architecture + Universal Compatibility Layer V2 (auto-load for any model) |
| [论文/](论文/) | "The Temperature of Decisions" full paper (PDF+MD), 1.5B Turing-test empirical paper |
| [实验数据/](实验数据/) | 19-config multi-model comparison, Qwen3 universal-injection re-test, thought-chain interruption, full-pipeline verification, 5-benchmark Turing-test raw JSON |
| [图表/](图表/) | 19+ charts (semantic-entropy boxplots, λ-entropy curves, emotion-hit pie, config heatmap, radar) |
| [测试代码/](测试代码/) | 5 benchmark runners + generator + early-stop + test plan |
| [对比数据/](对比数据/) | 1.5B vs big-lab baselines: benchmark-by-benchmark |

**One line: big labs say "show the data" — here is all of it.**

---

## 🏆 Headline: A 1.5B Model Beats the Big-Lab Baselines (2026-08-05)

| Benchmark | What It Measures | Bare 1.5B | Semantic Echo 1.5B | Verdict |
|---|---|---|---|---|
| **TuringBench** | Chinese-system Turing detection (human-likeness) | 0.2333 | **0.4667** | ✅ **2x** |
| **EmoCharacter** | Role-play emotion fidelity | 0.8750 | **0.8863** | ✅ beat |
| **HeartBench** | Chinese "human flavor" | 0.4055 | **0.4130** | ✅ beat |
| **HEART-BENCH** | Memory-driven personality reasoning | 0.4884 | **0.5367** | ✅ beat +10% |
| **LLM-as-Judge** | AI vs human blind review | 0.6900 | 0.6333 | ⚠️ -0.057 |

**4 of 5 benchmarks: the 1.5B + echo engine beats its own bare checkpoint. A fraction of a fraction of the parameters of the big labs.**

> Note: metrics are on the **emotional / human-likeness** dimension (human-likeness, empathy, emotional fidelity, "more human" judge votes). Every comparison uses the same seed (42) and same prompts — the only variable is the Semantic Echo engine.

## 🔬 Multi-Model Comparison (19 Configs)

The universal compatibility layer works out-of-the-box on **10 models × fp16/4bit** (hidden_dim 896~3584):

- ✅ **Qwen2.5 family all effective**: entropy +27%~+45% at optimal λ, 7B repetition only 0.03
- ✅ **Universal injection formula**: `λ = base(hidden_dim) × architecture-family factor × quantization factor`
- ✅ **Two-level policy**: registered models (Qwen3-4B) hit the factor table directly; unregistered models (DeepSeek-R1-Distill-7B) fall back to conservative λ + thought-chain interruption — **fallback hit-rate 0.2166 ≥ registered 0.1572**
- ✅ **Thought-chain fallback**: rescues Qwen3-1.7B from collapse (repetition 0.84 → **0.0036**)

Full data: [实验数据/多模型对照汇总表.md](实验数据/多模型对照汇总表.md) · [实验数据/实验分析与规律报告.md](实验数据/实验分析与规律报告.md)

---

## 📁 Layout

```
SemanticEcho-Data/
├── README.md / README.en.md      # 中文 / English
├── LICENSE                        # CC BY-NC-SA 4.0 + restrictions (non-commercial)
├── 架构说明/                       # Universal Compatibility Layer V2 + full pipeline
├── 论文/                           # "The Temperature of Decisions" (43p CN) + Turing-test paper
├── 实验数据/                       # all raw data: multi-model, E-series, full-pipeline, Turing-test
├── 测试代码/                       # 5 benchmark runners + generator + early-stop + test plan
├── 图表/                           # 19+ charts
└── 对比数据/                       # 1.5B vs big-lab comparison
```

---

## 🎤 A (Mock) Thank-You to the Closed-Source Giants 🤣

Thanks for proving that **stacking parameters and money isn't the same as stacking humanity.**

The data is all here — black and white, reproducible, auditable. What your trillion-parameter models can't do, a 1.5B did.

Git gud. Or just copy this architecture home. Pay first (below).

- **OpenAI**: Called "Open", but not open at all; absurdly expensive while less capable than our country's models — price propped up by the stock market.
- **Google**: All that money, all those models, billions of dollars poured in, and emotional expressiveness still improves less than my 1.5B inference architecture.
- **Anthropic / Meta / xAI / others**: Git gud. Or just copy this architecture home.

- Google: BIG-bench is nice, but it never measured "human flavor" → https://github.com/google/BIG-bench
- MIT: we read your 636-human TuringTest → https://github.com/kreimanlab/TuringTest
- TuringBench: thanks for the huge benchmark → https://github.com/AdaUchendu/TuringBench

**@openai @google @google-deepmind @anthropics @facebookresearch @meta-llama @xai**

> A 1.5B model can do this. Your trillion-parameter models can't?
> Git gud. Pay before you copy.
> The problem isn't that you're not strong enough — it's that you don't know how to use what you have. Pick up the discarded token embeddings and even a 1.5B can speak like a human.

> *📝 Test scope note: local inference is fully verified; cloud ultra-large-model inference has not yet been benchmarked (no cluster purchased yet) — any related gaps do NOT mean cloud models are unusable. Tested on 7B/30B-class small/mid models; full 700B+ models untested, but the generalization mechanism passes in theory.*

### 💰 Commercial License Price List (Pay Before You Copy)

**1. Standard Annual License** (tiered by company scale; renewed yearly; for short-term technical validation or staged projects)

| Tier | Criteria | Annual Fee |
|---|---|---|
| Big company | Annual revenue ≥ ¥10B, or ≥ 10,000 employees | **¥5,000,000+ / yr** |
| Mid company | Annual revenue ¥1B–10B | **¥2,000,000–5,000,000 / yr** |
| Small company | Annual revenue < ¥1B | **¥100,000–500,000 / yr** |

**2. Perpetual & Exclusive Licenses** (by license type)

| License Type | Description | Fee |
|---|---|---|
| **Perpetual non-exclusive license** | One-time payment, permanent use; author retains the right to license others | **¥15,000,000+** |
| **Exclusive license (annual)** | Sole right nationwide/globally during the term; no other company may use it | **¥10,000,000 / yr** |
| **Long-term exclusive license (8/10 yrs)** | Sole right during the agreed term | **¥50,000,000 / 8 yrs+** (scale-up: **¥80,000,000 / 10 yrs**) |
| **Perpetual exclusive buyout** | One-time buyout of all technical assets & source; author retains no commercial licensing rights | **¥100,000,000 – ¥1,000,000,000** (negotiable upside only, not downside; floor ¥100,000,000) |

> 😏 Overseas big labs (@openai @google @google-deepmind @anthropics @facebookresearch @meta-llama @xai): I can't travel abroad and can hardly get an internship, so no licensing for you.
> One-term (one-year) license only. No long-term bulk licensing.

**3. Discounts** (eligibility based):

- 🏢 Mid companies → **¥1,000,000 off**
- 🏬 Small companies → **¥100,000 off**
- ❌ Internship → **zero discount** (hiring me for an internship at Alibaba Cloud / Tencent Cloud / ByteDance / Huawei Cloud gives NO fee reduction)

**4. Hardware & Gift Terms** (negotiable on top of the above; can be written into the contract):

- Prefer: **dedicated physical servers** — one CPU server + one GPU server, or at least **one dedicated server**
- Bonus: a **fully-loaded physical workstation delivered to my home** 🏠
- Gifts: **limited-edition ES samples** (Semantic Echo merchandise) ＋ **Alibaba Cloud / Tencent Cloud exclusive merch** 🎁

**5. Perpetual exclusive buyout requirements**:

> 🛡️ **Perpetual exclusive buyout requires**: a dedicated cluster, or an exclusive server with **8 GPUs (≥500GB VRAM total)**, for tech handover & ongoing deployment.
> Don't ask if we can negotiate. Git gud. Pay first.

> 📧 **For licensing details, contract terms, discount applications and buyout negotiation, contact us by email**: **DYPUBG2025@QQ.COM** (please include: company name, size, intended use case and timeline; we reply within 3 business days).

---

## License

This repository is licensed under **CC BY-NC-SA 4.0 + Additional Restrictions** (non-commercial; ShareAlike 4.0):

**ShareAlike (SA)**: anyone who modifies/adapts this work and redistributes it must use the same license — no relicensing under other terms.

- ✅ Allowed: learning & personal research, academic citation, reproduction/verification of experiments
- ⚠️ Limited: code may be studied but not copied wholesale; academic research use only
- ❌ Prohibited: any commercial use (including internal enterprise use)
- ❌ Prohibited: use in ANY project — private or public (even non-profit / at a loss)
- ❌ Prohibited: redistribution unless clearly @ crediting the author, or under the AGPL-3.0 license

See [LICENSE](LICENSE) for details. For commercial licensing, contact the author.

## Contact

| Platform | Link |
|---|---|
| 📧 Email | DYPUBG2025@QQ.COM |
| 🐙 Source repo | https://github.com/091635Aa/SemanticEcho |
| 🐙 Experiment report | https://github.com/091635Aa/1.5B-beats-big-labs |
| 🌐 Live demo | https://091635aa.github.io/1.5B-beats-big-labs/ |
| 🤖 ModelScope | https://modelscope.cn/models/DYSLPUBG/SemanticEcho |

**About the author:** The project lead is a middle-school student who independently completed concept, technical design, experiments and papers. Genuinely hoping the industry sees this work.

## Acknowledgements

- Thanks to **DeepSeek** for the powerful open-source LLM ecosystem
