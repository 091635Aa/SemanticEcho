# -*- coding: utf-8 -*-
"""查询 DeepSeek-V4-Flash 在 ModelScope 的文件大小与 revisions"""
import requests, json

def 查文件(模型):
    url = f"https://modelscope.cn/api/v1/models/{模型}/repo/files"
    try:
        r = requests.get(url, params={"Revision": "master", "Recursive": True, "PageSize": 100}, timeout=20)
        d = r.json()
        files = d.get("Data", {}).get("Files", []) or d.get("data", {}).get("files", [])
        if not files:
            print(模型, "无文件(可能需登录/不存在):", str(d)[:200])
            return
        print(f"=== {模型} ===")
        总 = 0
        for f in files:
            名 = f.get("Path") or f.get("Name") or f.get("name")
            大小 = f.get("Size") or f.get("size") or 0
            if isinstance(名, str) and (名.endswith(".safetensors") or 名.endswith(".bin") or 名.endswith(".gguf")):
                总 += int(大小 or 0)
                print(f"  {名}  {int(大小 or 0)/1e9:.2f} GB")
        print(f"  权重总大小: {总/1e9:.2f} GB")
    except Exception as e:
        print(模型, "查询失败:", e)

for m in ["deepseek-ai/DeepSeek-V4-Flash"]:
    查文件(m)

# 查 revisions
try:
    r = requests.get("https://modelscope.cn/api/v1/models/deepseek-ai/DeepSeek-V4-Flash/revisions", timeout=20)
    print("revisions:", str(r.json())[:400])
except Exception as e:
    print("revisions 失败:", e)
