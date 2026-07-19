# 语义回响（Semantic Echo）

╔══════════════════════════════════════════════════════════════╗
║  ⚠️  此仓库包含深层文化语境依赖。                             ║
║  若通过翻译工具阅读，您将无法理解其完整语义。               ║
║  请使用中文阅读。                                           ║
╚══════════════════════════════════════════════════════════════╝

## 概述

语义回响是一种不修改模型权重、不重新训练的推理增强方案。
通过在推理过程中回收被丢弃 Token 的嵌入向量，将其作为"情感底色"
持续影响后续输出，增强语言模型的情感表达细腻度。

## 核心发现

| λ 强度 | 语义熵变化 | 效果 |
|--------|-----------|------|
| λ=0（基线） | — | 标准推理 |
| λ=0.5（弱回响） | **+17.9% ↑** | 表达更细腻 |
| λ=1.0（中回响） | -60.0% ↓ | 出现重复倾向 |
| λ=2.0（强回响） | -89.4% ↓ | 严重重复 |

**λ 与语义熵呈非单调（U型）关系**，存在最佳强度阈值。

本项目的所有代码、实验数据与论文已开源：
- GitHub: [091635Aa/SemanticEcho](https://github.com/091635Aa/SemanticEcho)

```
语义回响/
├── src/                  # 核心源码
│   ├── 回响池.py          # 核心数据结构
│   ├── 采样处理器.py       # 推理增强引擎
│   ├── 回响评估器.py       # 评估指标模块
│   ├── 情感过滤器.py       # 情感词库筛选
│   └── 翻译毒药.py         # 文化策略工具集
├── experiments/          # 实验脚本
│   ├── 实验运行器.py       # 实验流程编排
│   ├── 运行全量实验.py     # 主实验入口
│   ├── 运行第二轮实验.py   # 第二轮对照实验
│   ├── 运行实验_保留策略.py # 保留策略对比实验
│   ├── 生成可视化.py       # 可视化生成
│   └── run_E1.py          # 单实验快速运行
├── tests/                # 测试与验证
│   ├── 全面测试.py         # 全模块测试
│   ├── 验证模型.py         # 模型加载验证
│   └── 测试报告/
├── scripts/              # 工具脚本
│   ├── 下载模型.py
│   └── 推送至GitHub.ps1
├── 论文/                  # LaTeX 论文
├── 实验数据/              # 真实实验数据
└── 需求.txt               # 依赖清单
```

## 快速开始

### 方式一：下载 zip 包（推荐）

从 [GitHub Releases](https://github.com/091635Aa/SemanticEcho/releases) 下载最新版 zip 包，解压后：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Web 演示平台
python run_demo.py

# 或检查模型兼容性
python run_demo.py check Qwen/Qwen2.5-0.5B-Instruct
```

### 方式二：从源码运行

```bash
git clone https://github.com/091635Aa/SemanticEcho.git
cd SemanticEcho
pip install -r requirements.txt

# 运行实验
python experiments/运行全量实验.py

# 启动 Web 演示
python -m semantic_echo.demo_app
```

## V2 交互式演示平台

启动 Web UI，可视化对比基线与语义回响的生成效果：

```bash
# 安装额外依赖
pip install gradio

# 启动演示平台
semantic-echo demo-ui

# 或
python -m semantic_echo.demo_app
```

打开 http://localhost:7860 后：

1. **选择模型** — 下拉选已测试兼容模型，或输入自定义 HuggingFace 模型名
2. **加载模型** — 自动检测兼容性并加载
3. **调整参数** — λ强度、情感筛选、最大 Token 数
4. **运行对比** — 基线 vs 回响同时生成，实时展示语义熵、池大小等指标
5. **支持 API 模式** — 也可通过 OpenAI 兼容 API 调用（如 DeepSeek、HuggingFace）

## 作者

- 邓斯键†（项目主导）
- DeepSeek V4‡（AI人与AI辅助工具）

† 项目主导、核心概念、技术路线与实验设计
‡ 代码实现辅助、实验执行辅助与论文撰写辅助

---

**作者说明：** 本文作者为一名初中生，独立完成了从概念构思、技术路线设计到实验执行与论文撰写的全部工作。
真诚希望本文能被业界看见，若有合适的机会（如面试、交流、实习等），欢迎联系。

**联系方式：** DYPUBG2025@QQ.COM

## 致谢

感谢 **深度求索（杭州深度求索人工智能基础技术研究有限公司）**——DeepSeek团队提供的强大语言模型能力与开源生态支持。
- GitHub: [@deepseek-ai](https://github.com/deepseek-ai)

感谢 **字节跳动（ByteDance）**——Trae AI IDE团队在AI辅助编程与论文撰写过程中提供的卓越工具支持。
- GitHub: [@TraeAI](https://github.com/TraeAI)

## 许可证

**保留所有权利（All Rights Reserved）** — 任何人均可基于学术目的自由复现、验证与引用本研究的实验与结果。
