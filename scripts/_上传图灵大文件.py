# -*- coding: utf-8 -*-
"""
TuringTest-Data 超大文件上传（Releases API，直接单请求上传）
==========================================================
GitHub Releases 资产上传：单文件最大 2GB。
直接用 POST 上传整个文件（Content-Type: application/octet-stream）。
- turingbench.zip (228MB) -> turingbench.zip
- AA/train.csv (135MB)    -> AA_train.csv
"""
import os
import requests
import urllib3
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TOKEN = os.environ.get("GITHUB_TOKEN", "")
if not TOKEN:
    raise SystemExit("请先设置环境变量 GITHUB_TOKEN（setx GITHUB_TOKEN 你的token）后再运行")
REPO = "091635Aa/TuringTest-Data"
API = f"https://api.github.com/repos/{REPO}"
UPLOAD_HOST = "https://uploads.github.com"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}

文件清单 = [
    (r"i:\Desktop\语义回响\图灵测试\data\turingbench.zip",
     "turingbench.zip", "data/turingbench.zip - TuringBench 官方数据压缩包（原始数据已解压至 data/turingbench/）"),
    (r"i:\Desktop\语义回响\图灵测试\data\turingbench\TuringBench\AA\train.csv",
     "AA_train.csv", "data/turingbench/TuringBench/AA/train.csv - TuringBench AA（人工标注）训练集，135MB 超 git 100MB 限制"),
]

session = requests.Session()
session.verify = False


def 创建或取release():
    r = session.get(f"{API}/releases", headers=HEADERS, timeout=60)
    r.raise_for_status()
    for rel in r.json():
        if rel.get("tag_name") == "v1.0.0":
            return rel
    r = session.post(f"{API}/releases", headers=HEADERS,
                     json={"tag_name": "v1.0.0", "name": "v1.0.0 超大文件数据包",
                           "body": "语义回响图灵测试超大文件（>100MB，git push 无法承载，单独走 Release）\n\n"
                                   "- turingbench.zip（TuringBench 原始压缩包）\n"
                                   "- AA_train.csv（人工标注训练集）"},
                     timeout=60)
    r.raise_for_status()
    return r.json()


def 直接上传(release_id, 本地路径, 资产名, 说明):
    大小 = os.path.getsize(本地路径)
    print(f"[上传] {资产名} ({大小/1024/1024:.1f}MB) ...", flush=True)
    # 检查是否已存在同名资产
    r = session.get(f"{API}/releases/{release_id}/assets", headers=HEADERS, timeout=60)
    r.raise_for_status()
    for a in r.json():
        if a["name"] == 资产名:
            print(f"  已存在资产 {资产名}，跳过")
            return True

    url = f"{UPLOAD_HOST}/repos/{REPO}/releases/{release_id}/assets?name={资产名}"
    上传头 = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/octet-stream",
        "Accept": "application/vnd.github+json",
    }
    with open(本地路径, "rb") as f:
        r = session.post(url, headers=上传头, data=f, timeout=900)
    if r.status_code in (200, 201):
        print(f"  ✅ {资产名} 上传完成 (id={r.json().get('id')})")
        return True
    print(f"  ❌ 失败: {r.status_code} {r.text[:300]}")
    return False


def main():
    print("=== TuringTest-Data 超大文件上传（Release 单请求）===")
    rel = 创建或取release()
    print(f"Release: {rel['tag_name']} (id={rel['id']})")
    ok = True
    for 本地, 资产名, 说明 in 文件清单:
        if not os.path.exists(本地):
            print(f"  [缺失] {本地}")
            continue
        r = 直接上传(rel["id"], 本地, 资产名, 说明)
        ok = ok and r
    print("\n===== 完成 =====")
    print(f"Release 页: https://github.com/{REPO}/releases/tag/{rel['tag_name']}")


if __name__ == "__main__":
    main()
