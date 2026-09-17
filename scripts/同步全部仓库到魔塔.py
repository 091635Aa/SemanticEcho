# -*- coding: utf-8 -*-
"""
同步全部 GitHub 仓库到魔塔社区 (ModelScope)
=============================================
把以下本地 git 仓库镜像到魔塔独立仓库，保留各自 README（加魔塔 YAML 头）：
  1. 1.5B-Turing-Challenge  → DYSLPUBG/1.5B-Turing-Challenge
  2. SemanticEcho-Data      → DYSLPUBG/SemanticEcho-Data
  3. V3 最终生产架构         → DYSLPUBG/SemanticEcho-V3

用法：python 同步全部仓库到魔塔.py
"""
import os
import sys
import shutil
import tempfile
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 推送至魔塔 import MODELSCOPE_TOKEN, LICENSE

# ════════════════════════════════════════════════════════
# 仓库配置
# ════════════════════════════════════════════════════════
REPOS = [
    {
        "dir": r"i:\Desktop\语义回响\发布\1.5B-Turing-Challenge",
        "model": "DYSLPUBG/1.5B-Turing-Challenge",
        "cn": "1.5B图灵挑战（语义回响图灵测试实证）",
        "desc": "一个 1.5B 小模型挂载语义回响引擎，在 5 项图灵类基准的情感表达/人味维度 4 项反超大厂基准。含源码、测试脚本、日志、结果、论文。",
        "tags": "1.5B, 图灵测试, 语义回响, 情感增强, 推理架构, 情感表达, human-like, turing-test",
    },
    {
        "dir": r"i:\Desktop\语义回响\发布\SemanticEcho-Data",
        "model": "DYSLPUBG/SemanticEcho-Data",
        "cn": "语义回响实验数据与架构实证",
        "desc": "语义回响项目全部可审计原始实验数据：多模型对照19配置、Qwen3通用注入重测、图灵测试5基准原始JSON、19+图表、架构说明与论文。",
        "tags": "实验数据, 图灵测试, 语义回响, 情感增强, benchmark, dataset, 图表",
    },
    {
        "dir": r"f:\最终工程架构",
        "model": "DYSLPUBG/SemanticEcho-V3",
        "cn": "语义回响V3最终生产架构",
        "desc": "语义回响最终生产架构：全模型自适应参数匹配、λ步数衰减、FastAPI双协议服务、打标闭环。不训练不改权重，任意decoder-only模型输入hidden_dim即获最优注入参数。",
        "tags": "V3, 生产架构, 语义回响, 推理框架, FastAPI, 全模型自适应, 情感增强",
    },
]

EXCLUDE_SUFFIX = {".pyc", ".pyo"}
EXCLUDE_NAME = {"__pycache__", ".DS_Store"}
# 目录遍历时排除的顶层目录/文件
WALK_EXCLUDE = {".git", ".venv", "node_modules", "dist", "build"}


def git_tracked_files(repo_dir):
    """返回 git ls-files 的相对路径列表（禁止中文八进制转义）"""
    r = subprocess.run(
        ["git", "-C", repo_dir, "-c", "core.quotepath=false", "ls-files"],
        capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"git ls-files 失败: {r.stderr}")
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def walk_all_files(repo_dir):
    """目录遍历收集全部文件（git 无跟踪文件时兜底）"""
    out = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in WALK_EXCLUDE]
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, repo_dir)
            out.append(rel)
    return out


def build_model_card(desc, tags):
    """生成魔塔 YAML 头 + 仓库自身 README"""
    tag_lines = "\n".join(f"- {t.strip()}" for t in tags.split(",") if t.strip())
    yaml_head = f"""---
license: cc-by-nc-sa-4.0
tags:
{tag_lines}
- 中文
- chatbot
- text-generation
- inference
- transformer
- LLM
task: text-generation
pipeline_tag: text-generation
---

"""
    return yaml_head + desc + "\n"


def sync_one(cfg):
    print("=" * 60)
    print(f"  {cfg['model']}")
    print("=" * 60)
    repo_dir = cfg["dir"]
    model_name = cfg["model"]

    if not os.path.isdir(repo_dir):
        print(f"  ⚠️  目录不存在: {repo_dir}，跳过")
        return

    temp_dir = tempfile.mkdtemp(prefix="ms_sync_")
    try:
        files = git_tracked_files(repo_dir)
        source = "git"
        if len(files) == 0:
            files = walk_all_files(repo_dir)
            source = "目录遍历"
        print(f"  [1/4] 收集文件（{source}）: {len(files)} 个")

        copied = 0
        for rel in files:
            rel_n = rel.replace("/", os.sep)
            if os.path.basename(rel_n) in EXCLUDE_NAME:
                continue
            if os.path.splitext(rel_n)[1].lower() in EXCLUDE_SUFFIX:
                continue
            src = os.path.join(repo_dir, rel_n)
            dst = os.path.join(temp_dir, rel_n)
            if not os.path.exists(src):
                print(f"  ⚠️ 跳过不存在: {rel}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        print(f"  [2/4] 已拷贝 {copied} 个文件")

        # 用仓库自身 README.md 作为主体（已含嘲讽/报价/测试小注）
        readme_src = os.path.join(repo_dir, "README.md")
        body = ""
        if os.path.exists(readme_src):
            with open(readme_src, "r", encoding="utf-8") as f:
                body = f.read()
        card = build_model_card(body, cfg["tags"])
        readme_path = os.path.join(temp_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card)
        print("  [3/4] 魔塔 README.md（YAML头 + 仓库README）已生成")

        # 上传
        os.environ["MODELSCOPE_API_TOKEN"] = MODELSCOPE_TOKEN
        from modelscope.hub.api import HubApi
        api = HubApi(token=MODELSCOPE_TOKEN)
        try:
            api.get_model(model_id=model_name)
            print("  ℹ️  仓库已存在，将更新文件")
        except Exception:
            repo_url = api.create_model(
                model_id=model_name, visibility=5,
                license=LICENSE, chinese_name=cfg["cn"])
            print(f"  ✅ 创建新仓库: {repo_url}")

        print("  [4/4] 上传中...")
        api.upload_folder(
            repo_id=model_name,
            repo_type="model",
            folder_path=temp_dir,
            commit_message=f"同步 GitHub 仓库全量内容到魔塔社区：{cfg['cn']}",
        )
        print(f"  ✅ 同步成功: https://modelscope.cn/models/{model_name}")
    except Exception as e:
        print(f"  ❌ 同步失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    for cfg in REPOS:
        try:
            sync_one(cfg)
        except Exception as e:
            print(f"❌ {cfg['model']} 处理异常: {e}")
    print("\n全部仓库处理完毕。")


if __name__ == "__main__":
    main()
