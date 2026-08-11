"""串行多次测试入口 - GPU 显存不足时串行运行所有基准"""
import subprocess
import sys
import os

本目录 = r"i:\Desktop\语义回响\图灵测试"
Python = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\.venv\Scripts\python.exe"

测试列表 = [
    ("run_heartbench.py", "--runs 3"),
    ("run_feel_heart.py", "--runs 3"),
    ("run_llm_judge.py", "--runs 3"),
    ("run_turingbench.py", "--runs 3"),
]

for 脚本, 参数 in 测试列表:
    cmd = f'"{Python}" "{os.path.join(本目录, 脚本)}" {参数}'
    print(f"\n{'='*60}")
    print(f"开始运行: {脚本} {参数}")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    result = subprocess.run(
        cmd, shell=True, cwd=本目录,
        timeout=600,
        capture_output=False
    )
    
    if result.returncode == 0:
        print(f"\n✅ {脚本} 完成")
    else:
        print(f"\n❌ {脚本} 失败 (code={result.returncode})")
    
    sys.stdout.flush()

print("\n" + "="*60)
print("全部串行测试完成")
print("="*60)
