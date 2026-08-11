# 测试方案与复现说明

> 对应 1.5B 图灵测试实证的完整测试流程。测试代码与本目录下的原始结果数据一一对应，可完整复现。

## 1. 五个基准测什么

| 基准 | 脚本 | 测什么 | 判定方式 |
|---|---|---|---|
| **TuringBench** | run_turingbench.py | 中文体系图灵检测（真人 vs AI 生成） | TF-IDF(1-2gram)+LR 检测器，被判为 AI 的比例 → 人似度 = 1 − 检出率 |
| **EmoCharacter** | run_emocharacter.py | 角色扮演情感保真度 + 跨轮一致性 | 7B 裁判按 NAACL 2025 论文 rubric 评分 |
| **HeartBench** | run_heartbench.py | 中文"人味儿"多维评测（人格/情绪/社交/道德） | 7B 裁判按官方 rubric 逐条命中 + norm_score 归一化 |
| **HEART-BENCH** | run_feel_heart.py | 记忆驱动人格推理 MCQ 行为预测 | 决策准确率 + 共情评分 + 跨轮一致性 |
| **LLM-as-Judge** | run_llm_judge.py | AI 回复 vs 真人回复盲评 | 7B 裁判盲评"哪个更像真人"+ AB 对调双投消除位置偏差 |

## 2. 对照原则

- 所有对照 **同种子（42）、同提示词**，唯一变量 = 是否挂载语义回响引擎；
- 早停机制：指标波动超标时自动叠加轮数（见 早停.py）；
- 生成器（生成器.py）+ 公共模块（公共模块.py）统一封装模型加载与回响注入。

## 3. 复现命令

```bash
pip install -r requirements.txt   # transformers, torch, sklearn, jieba...

# 准备模型（本地路径，如 Qwen2.5-1.5B-Instruct），在 生成器.py 中配置路径后执行：

python run_turingbench.py --模式 全部
python run_heartbench.py --模式 全部
python run_feel_heart.py --模式 全部 --思考链
python run_emocharacter.py --模式 全部
python run_llm_judge.py --模式 全部 --λ 0.08 --身份 off
```

## 4. 实验记录

- [experiment_log_20260805.md](../实验数据/实验记录/experiment_log_20260805.md)：完整实验日志
- [repair_report.md](../实验数据/实验记录/repair_report.md)：修复报告（含失败实验与修复）
- [裁判校准报告.md](../实验数据/实验记录/裁判校准报告.md)：LLM-as-Judge 裁判偏差离线校准（R6）

## 5. 结果数据

原始结果 JSON 见 [实验数据/图灵测试结果/](../实验数据/图灵测试结果/)，与上方脚本一一对应。
