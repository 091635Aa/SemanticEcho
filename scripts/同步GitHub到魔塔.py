# -*- coding: utf-8 -*-
"""
同步 GitHub 仓库内容到魔塔社区 (ModelScope)
============================================
把 GitHub 主仓库（SemanticEcho）跟踪的全部文件镜像到魔塔社区，
保留 ModelScope 专属模型卡片 README.md（YAML 头），其余文件与 GitHub 一致。

用法：python 同步GitHub到魔塔.py
"""
import os
import sys
import shutil
import tempfile
import subprocess

# 复用 推送至魔塔 的模型卡片与配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 推送至魔塔 import MODEL_CARD, MODELSCOPE_TOKEN, MODEL_NAME, MODEL_NAME_CN, LICENSE

# GitHub 主仓库本地目录
REPO_DIR = r"i:\Desktop\语义回响"

# 额外排除（即使被 git 跟踪也跳过，如超大临时文件）
EXCLUDE_SUFFIX = {".pyc", ".pyo"}
EXCLUDE_NAME = {"__pycache__", ".DS_Store"}


def git_tracked_files(repo_dir):
    """返回 git ls-files 的相对路径列表"""
    # core.quotepath=false：禁止 git 将中文路径转义为八进制（Windows 下默认会转义）
    r = subprocess.run(["git", "-C", repo_dir, "-c", "core.quotepath=false", "ls-files"],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"git ls-files 失败: {r.stderr}")
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def main():
    print("=" * 60)
    print("  GitHub (SemanticEcho) → 魔塔社区 全量同步")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp(prefix="semantic_echo_ms_sync_")
    try:
        # 1. 收集被跟踪文件
        files = git_tracked_files(REPO_DIR)
        print(f"\n[1/4] GitHub 跟踪文件: {len(files)} 个")

        # 2. 拷贝到临时目录（保留目录结构）
        print("[2/4] 拷贝文件到临时目录...")
        copied = 0
        for rel in files:
            rel = rel.replace("/", os.sep)
            if os.path.basename(rel) in EXCLUDE_NAME:
                continue
            if os.path.splitext(rel)[1].lower() in EXCLUDE_SUFFIX:
                continue
            src = os.path.join(REPO_DIR, rel)
            dst = os.path.join(temp_dir, rel)
            if not os.path.exists(src):
                print(f"  ⚠️ 跳过不存在: {rel}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        print(f"  ✅ 已拷贝 {copied} 个文件")

        # 3. 写入 ModelScope 专属模型卡片（覆盖 GitHub README.md）
        readme_path = os.path.join(temp_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(MODEL_CARD)
        print("[3/4] ModelScope 模型卡片 README.md 已写入")

        # 4. 上传
        print("[4/4] 上传到魔塔社区...")
        os.environ["MODELSCOPE_API_TOKEN"] = MODELSCOPE_TOKEN
        from modelscope.hub.api import HubApi

        api = HubApi(token=MODELSCOPE_TOKEN)
        try:
            api.get_model(model_id=MODEL_NAME)
            print("  ℹ️  模型仓库已存在，将更新文件")
        except Exception:
            repo_url = api.create_model(model_id=MODEL_NAME, visibility=5,
                                        license=LICENSE, chinese_name=MODEL_NAME_CN)
            print(f"  ✅ 创建新模型仓库: {repo_url}")

        result = api.upload_folder(
            repo_id=MODEL_NAME,
            repo_type="model",
            folder_path=temp_dir,
            commit_message="同步 GitHub 仓库全量内容（源码/脚本/实验/论文/文档）到魔塔社区",
        )
        print(f"  ✅ 同步成功!")
        print("=" * 60)
        print(f"  🔗 https://modelscope.cn/models/{MODEL_NAME}")
        print(f"  📂 临时目录: {temp_dir}")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 同步失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"   💡 备选: ms upload {MODEL_NAME} \"{temp_dir}\" --token {MODELSCOPE_TOKEN}")


if __name__ == "__main__":
    main()
