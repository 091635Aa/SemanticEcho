# -*- coding: utf-8 -*-
"""列出 TuringTest-Data 仓库完整文件树"""
import os
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
H = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
     "Accept": "application/vnd.github+json"}
API = "https://api.github.com/repos/091635Aa/TuringTest-Data"
s = requests.Session()
s.verify = False


def walk(path="", depth=0):
    r = s.get(f"{API}/contents/{path}", headers=H, timeout=60)
    for item in r.json():
        print("  " * depth + f"{item['type']:<4} {item['path']}")
        if item["type"] == "dir" and item["path"].count("/") < 3:
            walk(item["path"], depth + 1)


walk("")
