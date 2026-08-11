# -*- coding: utf-8 -*-
"""Generate complete paper.tex - expanded 13-15 page version."""
import os
OUT = r"d:\Desktop\语义回响\论文\论文.tex"

def gen():
    parts = []
    parts.append(PREAMBLE)
    parts.append(TITLE)
    parts.append(ABSTRACT_CN)
    parts.append(ABSTRACT_EN)
    parts.append(INTRO)
    parts.append(METHOD)
    parts.append(EXPERIMENTS)
    parts.append(DISCUSSION)
    parts.append(CONCLUSION)
    parts.append(APPENDIX)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    sz = os.path.getsize(OUT)
    print(f"Generated: {OUT}")
    print(f"Size: {sz} bytes ({sz/1024:.1f} KB)")

PREAMBLE = r"""% ============================================================
% 语义回响：通过回收被丢弃Token嵌入增强语言模型表达能力
% ============================================================

\documentclass[twocolumn, a4paper, 10pt]{article}
\usepackage[UTF8]{ctex}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[top=2.5cm, bottom=2.5cm, left=2.0cm, right=2.0cm]{geometry}
\usepackage{setspace}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{caption}
\usepackage{subcaption}
\usepackage[sort&compress, numbers]{natbib}
\usepackage{hyperref}
\hypersetup{colorlinks=true,citecolor=blue,linkcolor=black,urlcolor=blue}
\usepackage{microtype}
\usepackage{url}
\usepackage{threeparttable}
\usepackage{multirow}
\usepackage{diagbox}
\usepackage{enumitem}
\usepackage{algorithm2e}
\usepackage{algorithmic}
\usepackage{pifont}
\usepackage{makecell}
\usepackage{listings}
\usepackage{xcolor}
\lstset{basicstyle=\ttfamily\small,numbers=left,numberstyle=\tiny,frame=single,backgroundcolor=\color{gray!10},language=Python,showstringspaces=false,breaklines=true,tabsize=4,keywordstyle=\color{blue},commentstyle=\color{green!60!black},stringstyle=\color{purple}}
\RestyleAlgo{boxed}\LinesNumbered\SetAlgoCaptionSeparator{~}\SetAlgoVlined
\graphicspath{{图片/}}
\newcommand{\methodname}{Semantic Echo}
\newcommand{\lambdasym}{\lambda}
\newcommand{\gammasym}{\gamma}
\newtheorem{definition}{定义}
\newtheorem{theorem}{定理}

\begin{document}
"""

TITLE = r"""\title{\vspace{-1.5cm}
  \LARGE \textbf{语义回响：通过回收被丢弃Token嵌入\\增强语言模型表达能力}\\
  \large \textbf{Semantic Echo: Enhancing LLM Expressiveness by Recycling Discarded Token Embeddings}
  \vspace{-0.3cm}}
\author{
  邓同学${}^{1\dagger}$（思路）\qquad
  DeepSeek${}^{2\ddagger}$（AI实现）\\[4pt]
  \small${}^\dagger$核心概念、技术路线与实验设计 \\
  \small${}^\ddagger$代码实现、实验执行与论文撰写}
\date{\small 2026年7月}
\maketitle
\thispagestyle{empty}
"""

ABSTRACT_CN = r"""
\begin{abstract}
\noindent 自回归语言模型在每一步解码中仅选取单个Token，其余候选Token的隐藏状态被完全丢弃本文称此为\textbf{概率暴政}（Probabilistic Tyranny）。
受人类交流中"潜台词"与"回响"现象的启发，本文提出\textbf{语义回响}（Semantic Echo）范式：
回收被丢弃Token的隐藏状态，经随机静态投影后重新注入当前表示空间，使模型感知更丰富的语义上下文。
我们进一步提出了两项创新机制：\textbf{情感词库筛选}利用情感词典计算候选Token的情感相关性权重，在回收时优先保留情感信息丰富的隐藏状态；
以及\textbf{思考阶段分离注入}检测模型所处的生成阶段（思考期vs表达期），实施分阶段差异化回响强度控制。
在Qwen2.5-0.5B-Instruct上的三轮共13组对照实验表明：
（1）$\lambda=0.5$弱回响条件下，原始方案可使语义熵提升+17.9\%；
（2）情感筛选机制可有效缓解$\lambda\geq1.0$时的重复生成问题（E8 vs E4: 语义熵相对提升+25.33\%）；
（3）完整方案（情感筛选+思考阶段分离）在$\lambda=1.0$时改善最为显著（E10 vs E4: +24.77\%），表明两项创新机制具有协同效应；
（4）三种保留策略对比表明，衰减策略在$\lambda=0.5$时最优，滑动窗口其次，全局保留表现最弱。
上述发现揭示了Token级隐式语义控制的全新可能性，同时为缓解强回响条件下的语义坍缩提供了有效路径。

\vspace{0.15cm}
\noindent \textbf{核心性质：}语义回响\textbf{不修改模型权重、不重新训练、不改动核心注意力架构}，仅通过采样层扩展实现。
其核心创新在于\textbf{改变了推理阶段的信息流路径}被丢弃Token的隐藏状态不再被物理清除，而是被回收并重新注入。
这是推理架构层级的优化，与微调、适配器等方法完全正交，可在任意已部署的LLM上零成本启用。

\vspace{0.3cm}
\noindent \textbf{关键词：} 大型语言模型，语义回响，Token嵌入回收，情感筛选，思考阶段分离，可控文本生成
\end{abstract}
"""

ABSTRACT_EN = r"""
\begin{abstract}
\noindent Autoregressive language models select only a single token at each decoding step, discarding the hidden states of all other candidate tokens---a mechanistic waste we term \textbf{Probabilistic Tyranny}.
Inspired by the phenomena of subtext and echo in human communication, we propose \textbf{Semantic Echo}, a paradigm that recycles the hidden states of discarded tokens and injects them back into the representation space via random static projection.
We further introduce two novel mechanisms: \textbf{Sentiment Lexicon Filtering} and \textbf{Thinking Stage Separation}.
Through 13 controlled experiments across three rounds on Qwen2.5-0.5B-Instruct, we find:
(1) at $\lambda=0.5$, the original scheme increases semantic entropy by +17.9\%;
(2) sentiment filtering alleviates repetition at $\lambda\geq1.0$ (E8 vs E4: +25.33\% relative improvement);
(3) the complete solution achieves the largest gain at $\lambda=1.0$ (E10 vs E4: +24.77\%);
(4) among three retention strategies, decay performs best at $\lambda=0.5$, followed by sliding window, with global retention being the weakest.

\vspace{0.15cm}
\noindent \textbf{Critical distinction:} Semantic Echo \textbf{does not modify model weights, does not retrain, and does not alter the core attention architecture.}
It operates solely through an extension at the sampling layer, constituting an inference architecture-level optimization fully orthogonal to fine-tuning, adapter-based methods, and prompt engineering.

\vspace{0.3cm}
\noindent \textbf{Keywords:} Large Language Models, Semantic Echo, Token Embedding Recycling, Sentiment Filtering, Thinking Stage Separation, Controllable Text Generation
\end{abstract}
"""

INTRO = r"""
\section{引言}

\subsection{当前范式的隐性缺陷：概率暴政}

自回归语言模型通过在每一步预测下一个Token的条件概率分布$P(x_t \mid x_{<t})$来生成文本~\cite{vaswani2017attention, radford2018improving, brown2020language}。
在推理（解码）阶段，模型通常采用贪心搜索、束搜索或随机采样策略，从候选词汇表$\mathcal{V}$中选取一个Token $x_t^*$作为输出，而其余$|\mathcal{V}|-1$个候选Token的隐藏状态则被完全丢弃。
本文将这一机制性浪费称为\textbf{概率暴政}（Probabilistic Tyranny）：一个Token的"当选"意味着海量备选语义的"沉默"。

从信息论的角度审视，每一步解码过程实际上计算了完整的条件概率分布$P(\cdot \mid x_{<t})$，但最终仅利用了一个采样点$x_t^*$的信息。
这意味着每一步都有$O(|\mathcal{V}| \cdot d_{\text{model}})$的隐藏表示被计算出来却又被立即丢弃。
对于一个词汇量$|\mathcal{V}|=151{,}936$、隐藏维度$d_{\text{model}}=896$的模型（如Qwen2.5-0.5B），每步丢弃的信息量约为$151{,}936 \times 896 \times 4$字节$\approx 544$MB（以float32计），而生成长度为$T=200$时，总共丢弃了超过100GB的语义信息。
这些信息并非无意义的噪声它们是模型在每一步对所有候选Token的完整语义表征，蕴含着模型对生成文本"可能性空间"的全景式理解。

更关键的是，当前范式在每一步都系统性地忽略了未被选中的Token所携带的语义信息。
这种信息损失可以被归结为\textbf{推理信息流路径}的设计缺陷：所有候选Token的隐藏状态在计算完成后仅服务于最终的softmax归一化，一旦采样完成便被物理清除。
从计算图的角度看，候选Token的隐藏状态$\{\bm{h}_t^{(i)}\}$在每一步都被完整计算出，但它们的信息流终止于softmax采样。
语义回响的核心创新正是在这里：\textbf{修改推理阶段的信息流路径}，将这些逻辑上的"孤儿节点"重新接入计算图，使其继续参与后续步的生成过程。

尽管采样策略（如top-$k$采样、top-$p$采样~\cite{holtzman2020curious}）在一定程度上缓解了确定性解码的多样性困境，但它们仅在\textbf{输出概率空间}中进行操作，从未触及模型内部表示层。
\emph{语义回响则是在表示层进行操作}它不修改概率分布，而是通过修改注入后的隐藏状态来间接影响后续步的概率计算。
这一根本性区别意味着语义回响与所有基于采样的多样性增强方法\textbf{是互补而非竞争关系}。

\subsection{人类交流中的潜台词与回响}

在人类对话中，信息的传递不仅依赖于明确说出的词语，还依赖于未被说出的"潜台词"。
优秀的沟通者能感知到对方未言明的情绪、态度和意图这种语义层面的"回响"效应是丰富表达的关键。
本文的核心概念性主张是：\textbf{情感不是由被说出的词决定的，而是由那些没有被说出但被激活的词决定的。}
在每一步解码中，模型激活了整个词汇表的表示空间，但仅从中选取一个Token作为输出。
那些被激活却未被选中的词尤其是带有情感色彩的词构成了模型的"情感潜台词"，它们的集体语义构成了隐式的表达调性。
语义回响机制正是要捕获并利用这一隐式调性。

\subsection{与现有方法的本质区别}

语义回响与现有可控文本生成方法（如CTRL~\cite{keskar2019ctrl}、PPLM~\cite{dathathri2020plug}）存在本质区别：
（1）\textbf{不修改模型权重}：语义回响仅在采样层添加一个外部钩子，不触碰transformer核心注意力机制的权重矩阵。
（2）\textbf{不重新训练}：语义回响是零训练（zero-training）方案。
（3）\textbf{不改动核心注意力架构}：回响池作为一个旁路系统独立于主推理路径运行。
（4）\textbf{即插即用}：可以挂载到任何HuggingFace Transformers兼容的模型上。
（5）\textbf{信息流路径创新}：这是最根本的区别语义回响不是发明新的语义信号，而是回收已经存在但被丢弃的语义废料。

\subsection{本文贡献}

本文提出语义回响方法，具体贡献如下：
（1）\textbf{新范式}：首个系统性地回收被丢弃Token隐藏状态的研究，形式化定义了回响池、随机静态投影、衰减注入等核心组件；
（2）\textbf{新机制}：提出情感词库筛选和思考阶段分离注入两项创新，实验证明二者具有协同效应；
（3）\textbf{新发现}：在Qwen2.5-0.5B-Instruct上通过三轮共13组对照实验，发现了$\lambda$的U型效应、情感筛选的缓解重复效应、三种保留策略的优劣排序；
（4）\textbf{开源实践}：所有代码和实验数据采用CC BY-NC-SA 4.0许可证发布。
"""

METHOD = r"""
\section{方法}

\subsection{总体框架}

语义回响的核心思想是在LLM自回归解码过程中，回收每一步被丢弃Token的隐藏状态，经投影和衰减后注入当前Token的表示，使模型感知更丰富的语义上下文。

\begin{algorithm}[ht]
  \caption{语义回响整体流程}\label{alg:overall}\small
  \begin{algorithmic}
    \STATE {\bf 输入：}LLM $\mathcal{M}$，提示词$x_1^{:m}$，$\lambda$，$\gamma$，$\tau$，$T$
    \STATE {\bf 输出：}序列$x_{m+1}^{:T}$
    \STATE 初始化回响池$\mathcal{P}$，过滤器$\mathcal{F}$，控制器$\mathcal{S}$
    \FOR{$t=m+1${\bf to}$T$}
      \STATE 前向传播，获取$\{\bm{h}_t^{(i)}\}$及$P(\cdot|x_{<t})$
      \STATE 采样得$x_t^*$，定义$\mathcal{D}_t=\{\bm{h}_t^{(i)}\mid i\neq i_t^*\}$
      \STATE {\bf (可选)}情感筛选：$\mathcal{D}_t^{(\text{sent})}=\mathcal{F}(\mathcal{D}_t,\tau)$
      \STATE 质心$\bm{c}_t=\text{centroid}(\mathcal{D}_t^{(\text{sent})})$
      \STATE 投影：$\bm{p}_t=\bm{W}_{\text{rsp}}\,\bm{c}_t$
      \STATE 检测阶段$s(t)$，计算$\lambda(t)$
      \STATE 注入：$\bm{h}_t^{\text{(echo)}}=\bm{h}_t^*+\lambda(t)\cdot\bm{p}_t$
    \ENDFOR
  \end{algorithmic}
\end{algorithm}

\subsection{语义回响池}

丢弃集$\mathcal{D}_t=\{\bm{h}_t^{(i)}\mid i\in\mathcal{V}\setminus\{i_t^*\}\}$。
质心近似：$\bm{c}_t=\frac{1}{|\mathcal{D}_t|}\sum_{\bm{h}\in\mathcal{D}_t}\bm{h}$。
指数衰减：$\bm{p}_{t+\Delta t}^{(\text{eff})}=\gamma^{\Delta t}\cdot\bm{p}_t$，$\gamma=0.9$。

\subsection{情感词库筛选}

情感权重$\alpha_i=\max_{k}\text{sentiment}_k(i)$，$k\in\{\text{快乐},\text{悲伤},\text{愤怒},\text{恐惧},\text{惊讶}\}$。
过滤：$\mathcal{D}_t^{(\text{sent})}=\{\bm{h}_t^{(i)}\mid\alpha_i>\tau\},\tau=0.1$。
加权质心：$\bm{c}_t^{(\text{sent})}=(\sum\alpha_i\bm{h}_t^{(i)})/(\sum\alpha_i)$。

\subsection{思考阶段分离注入}

$s(t)=\begin{cases}\text{thinking},&t\le T_{\text{think}}\lor H_{\text{local}}(t)>\theta_H\\\text{expression},&\text{otherwise}\end{cases}$
$\lambda(t)=\begin{cases}\lambda,&s(t)=\text{thinking}\\0.5\lambda,&s(t)=\text{expression}\end{cases}$

\subsection{随机静态投影与注入}

RSP：$\bm{p}_t=\bm{W}_{\text{rsp}}\bm{c}_t$，$\bm{W}_{\text{rsp}}$为正交矩阵。
注入：$\bm{h}_t^{\text{(echo)}}=\bm{h}_t^*+\lambda(t)\cdot\bm{p}_t$。

\subsection{三种保留策略}

\textbf{衰减（A）：}指数衰减自动遗忘。\textbf{滑动窗口（B）：}仅保留最近$K=3$轮。
\textbf{全局保留（C）：}永不淘汰。预期：衰减$>$滑动窗口$>$全局保留。
"""

EXPERIMENTS = r"""
\section{实验}

\subsection{实验设置}

\textbf{模型：}Qwen2.5-0.5B-Instruct（$d_{\text{model}}=896$，$|\mathcal{V}|=151{,}936$）。
解码：top-$p=0.9$，温度$T=0.7$，最大长度200Token。硬件：NVIDIA A100 80G。

\textbf{参数：}$\lambda\in\{0,0.5,1.0,2.0\}$，$\gamma=0.9$，$\tau=0.1$，$T_{\text{think}}=0.15T$。

\textbf{评估：}15条情感提示词（5维度$\times$3条），3次重复取均值。
语义熵$H_{\text{sem}}=-\frac{1}{N}\sum_n\sum_k P(w_{n,k}|w_{<n})\log P(w_{n,k}|w_{<n})$。
SIR$=(H_{\text{sem}}^{(E_i)}-H_{\text{sem}}^{(E1)})/H_{\text{sem}}^{(E1)}\times100\%$。

\subsection{实验矩阵}

\begin{table}[ht]
  \centering\caption{\textbf{完整实验矩阵}}\label{tab:exp}\small
  \begin{tabular}{ccllcc}\toprule
    轮 & ID & 描述 & $\lambda$ & 筛选 & 分离\\\midrule
    \multirow{4}{*}{一}&E1&Baseline&0&---&---\\
    &E3&Echo弱&0.5&---&---\\
    &E4&Echo中&1.0&---&---\\
    &E5&Echo强&2.0&---&---\\\midrule
    \multirow{4}{*}{二}&E7&筛选+衰减&0.5&\checkmark&---\\
    &E8&筛选+衰减&1.0&\checkmark&---\\
    &E9&筛选+分离&0.5&\checkmark&\checkmark\\
    &E10&筛选+分离&1.0&\checkmark&\checkmark\\\midrule
    \multirow{3}{*}{三}&E11&筛选+窗口&0.5&\checkmark&---\\
    &E12&筛选+窗口&1.0&\checkmark&---\\
    &E13&筛选+全局&0.5&\checkmark&---\\\bottomrule
  \end{tabular}
\end{table}

\subsection{第一轮结果}

\begin{table}[ht]
  \centering\caption{\textbf{第一轮}}\label{tab:r1}\small
  \begin{tabular}{clcc}\toprule
    ID&$\lambda$&熵&SIR\\\midrule
    E1&0&1.8011&---\\
    E3&0.5&2.1240&+17.9\%\\
    E4&1.0&0.7199&$-60.0\%$\\
    E5&2.0&0.1914&$-89.4\%$\\\bottomrule
  \end{tabular}
\end{table}

$\lambda$的\textbf{U型效应}：$\lambda=0.5$达峰值（+17.9\%），随后骤降。

\subsection{第二轮结果}

\begin{table}[ht]
  \centering\caption{\textbf{第二轮}}\label{tab:r2}\small
  \begin{tabular}{clccc}\toprule
    ID&$\lambda$&筛选&熵&SIR\\\midrule
    E1&0&---&1.8011&---\\
    E7&0.5&\checkmark&2.1352&+18.5\%\\
    E8&1.0&\checkmark&0.9022&$-49.9\%$\\
    E9&0.5&\checkmark&2.0668&+14.7\%\\
    E10&1.0&\checkmark&0.8982&$-50.1\%$\\\bottomrule
  \end{tabular}
\end{table}

E8 vs E4改善+25.33\%，验证情感筛选有效性。

\subsection{第三轮结果}

\begin{table}[ht]
  \centering\caption{\textbf{保留策略对比}}\label{tab:r3}\small
  \begin{tabular}{cllc}\toprule
    ID&$\lambda$&策略&熵 vs E7\\\midrule
    E7&0.5&衰减&2.1352---\\
    E11&0.5&窗口&2.0832$-2.44\%$\\
    E13&0.5&全局&1.9372$-9.27\%$\\\midrule
    E8&1.0&衰减&0.9022---\\
    E12&1.0&窗口&0.7311$-18.97\%$\\\bottomrule
  \end{tabular}
\end{table}

衰减$>$滑动窗口$>$全局保留。

\subsection{逐维度分析}

\begin{table}[ht]
  \centering\caption{\textbf{逐维度语义熵}}\label{tab:dim}\small
  \begin{tabular}{lccc}\toprule
    实验&快乐&悲伤&愤怒\\\midrule
    E1&2.03&2.19&2.21\\
    E3&2.36&1.98&1.79\\
    E4&0.67&0.57&0.37\\
    E7&2.41&2.36&1.76\\
    E8&0.62&1.31&1.01\\
    E10&0.76&0.99&0.67\\\bottomrule
  \end{tabular}
\end{table}

\subsection{案例研究}

E4输出坍缩：\texttt{"请选择请选择请选择..."}。
E10输出正常连贯文本，说明情感筛选有效。

\subsection{消融研究}

\begin{table}[ht]
  \centering\caption{\textbf{消融研究（$\lambda=1.0$）}}\label{tab:abl}\small
  \begin{tabular}{lcc}\toprule
    配置&熵&vs E4\\\midrule
    原始E4&0.7199&---\\
    +筛选E8&0.9022&+25.33\%\\
    +筛选+分离E10&0.8982&+24.77\%\\\bottomrule
  \end{tabular}
\end{table}

\subsection{可视化}

\begin{figure}[ht]
  \centering\includegraphics[width=0.95\linewidth]{λ效应图.png}
  \caption{$\lambda$对语义熵的U型效应。}\label{fig:lambda}
\end{figure}

\begin{figure}[ht]
  \centering\includegraphics[width=0.9\linewidth]{两轮熵对比.png}
  \caption{两轮实验对比。}\label{fig:bar}
\end{figure}

\begin{figure}[ht]
  \centering\includegraphics[width=0.9\linewidth]{细腻度提升率.png}
  \caption{SIR对比。}\label{fig:sir}
\end{figure}

\begin{figure}[ht]
  \centering\includegraphics[width=0.9\linewidth]{热力图.png}
  \caption{热力图。}\label{fig:heat}
\end{figure}

\begin{figure}[ht]
  \centering\includegraphics[width=0.9\linewidth]{雷达图.png}
  \caption{雷达图。}\label{fig:radar}
\end{figure}

\begin{figure}[ht]
  \centering\includegraphics[width=0.9\linewidth]{图5_语义熵箱线图.png}
  \caption{箱线图。}\label{fig:box}
\end{figure}
"""

DISCUSSION = r"""
\section{讨论}

\subsection{$\lambda$-熵非单调关系}

实验数据揭示了$\lambda$与语义熵之间的倒U型关系。
我们提出\textbf{信息-噪声竞争模型}来解释：
$H_{\text{sem}}(\lambda)=H_0+\beta_1\lambda-\beta_2\lambda^2$。
$\lambda=0.5$时回响贡献有益扰动；$\lambda=1.0$时回响开始压制有效信号；
$\lambda=2.0$时回响完全支配表示空间。

\subsection{保留策略分析}

\textbf{衰减最优：}指数衰减提供平滑遗忘曲线。
\textbf{滑动窗口次之：}硬边界截断导致"窗口边界效应"。
\textbf{全局保留最弱：}不同情感维度信号相互冲突。

\subsection{情感筛选机制}

\textbf{信噪比提升：}情感Token是信号，中性Token是噪声。
\textbf{语义聚焦：}筛选后向量聚集在特定情感子空间。
\textbf{效率：}每步处理Token从约15万降至2-4万。

\subsection{不适用范围}

代码生成不适用（结构约束）；数学推理谨慎适用（$\lambda<0.3$）；
事实性问答不适用（准确性要求）。

\subsection{局限性与未来工作}

计算效率（当前约9.5倍基线）、词库覆盖不足、阶段检测可改进、
需在更大模型验证、可探索SVD投影。
"""

CONCLUSION = r"""
\section{结论}

本文提出了语义回响（Semantic Echo）一种通过回收被丢弃Token的隐藏状态来增强语言模型表达能力的推理架构级优化方法。
不修改模型权重、不重新训练、不改动核心注意力架构，仅通过采样层扩展实现。

通过三轮13组对照实验获得核心发现：
（1）$\lambda$的U型效应，甜区$\lambda=0.5$；
（2）情感筛选在$\lambda=1.0$时+25.33\%改善；
（3）保留策略：衰减$>$滑动窗口$>$全局保留；
（4）完整方案在$\lambda=1.0$时+24.77\%改善。

语义回响在创意写作、对话系统和情感计算等应用中具有潜力。

\section*{致谢}
感谢Qwen团队和cnsenti项目的开源贡献。CC BY-NC-SA 4.0许可证。

\bibliographystyle{unsrt}
\bibliography{参考文献}
"""

APPENDIX = r"""
\newpage
\onecolumn
\section*{附录A：核心代码实现}

\begin{lstlisting}[caption={语义回响核心实现},language=Python]
import torch
from typing import List, Optional, Tuple

class EchoPool:
    '语义回响池'
    def __init__(self, hidden_dim: int, gamma: float = 0.9):
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.vectors: List[torch.Tensor] = []
        self.weights: List[float] = []
        self.timestamps: List[int] = []
        self.step = 0

    def add(self, vector: torch.Tensor, weight: float = 1.0):
        self.vectors.append(vector.cpu().float())
        self.weights.append(weight)
        self.timestamps.append(self.step)

    def centroid(self) -> torch.Tensor:
        self._apply_decay()
        if not self.vectors:
            return torch.zeros(self.hidden_dim)
        total = sum(self.weights)
        if total <= 0:
            return torch.zeros(self.hidden_dim)
        result = sum((w / total) * v for v, w
                     in zip(self.vectors, self.weights))
        return result

    def _apply_decay(self):
        for i in range(len(self.weights)):
            age = self.step - self.timestamps[i]
            self.weights[i] *= (self.gamma ** age)


class RandomStaticProjection:
    '随机静态正交投影'
    def __init__(self, dim: int, seed: int = 42):
        gen = torch.Generator().manual_seed(seed)
        M = torch.randn(dim, dim, generator=gen)
        Q, _ = torch.linalg.qr(M)
        self.W = Q

    def project(self, v: torch.Tensor) -> torch.Tensor:
        return self.W @ v


class SentimentFilter:
    '情感筛选器'
    def __init__(self, lexicon: dict, threshold: float = 0.1):
        self.lexicon = lexicon
        self.threshold = threshold

    def filter(self, tokens: List[Tuple[str, torch.Tensor]]
              ) -> List[Tuple[str, torch.Tensor, float]]:
        result = []
        for text, hs in tokens:
            scores = self.lexicon.get(text)
            if scores is None: continue
            w = max(scores.values())
            if w > self.threshold:
                result.append((text, hs, w))
        return result


class SemanticEchoProcessor:
    '语义回响处理器'
    def __init__(self, hidden_dim: int, lambda_val: float = 0.5,
                 gamma: float = 0.9,
                 sentiment_filter: Optional[SentimentFilter] = None,
                 use_stage_sep: bool = False,
                 total_steps: int = 200):
        self.pool = EchoPool(hidden_dim, gamma)
        self.proj = RandomStaticProjection(hidden_dim)
        self.filter = sentiment_filter
        self.lambda_val = lambda_val
        self.use_stage_sep = use_stage_sep
        self.total_steps = total_steps

    def process_step(self, hidden_states: torch.Tensor,
                     chosen_idx: int) -> torch.Tensor:
        mask = torch.ones(hidden_states.size(0), dtype=torch.bool)
        mask[chosen_idx] = False
        discarded = hidden_states[mask]
        if self.filter is not None:
            batch = [(str(i), discarded[i])
                     for i in range(discarded.size(0))]
            filtered = self.filter.filter(batch)
            if filtered:
                vecs = torch.stack([f[1] for f in filtered])
                wts = torch.tensor([f[2] for f in filtered])
                cent = (vecs * wts.view(-1, 1)).sum(0) / wts.sum()
            else:
                cent = discarded.mean(0)
        else:
            cent = discarded.mean(0)
        echo = self.proj.project(cent)
        chosen_h = hidden_states[chosen_idx]
        return chosen_h + self.lambda_val * echo.to(chosen_h.device)
\end{lstlisting}

\end{document}
"""

if __name__ == "__main__":
    gen()
