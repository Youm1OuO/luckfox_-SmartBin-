#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端云协同 - 云端 Mock 服务器（用于本地联调）

模拟"后台同学的服务"，提供两个端点（对应《不知道.md》后端配对机制）：

    POST /events/item         出入库事件 ITEM_IN / ITEM_OUT / ITEM_MOVED
    POST /events/label_scan   标签扫描（用户把标签朝镜头）

它做的事：
    1. 收到 label_scan → 把标签图存下来 + 模拟 OCR 读出品牌/保质期，
       写入一个"待配对标签"列表（pending_labels，5 分钟有效）。
    2. 收到 ITEM_IN → 从 pending_labels 里找"最近、未消费、未过期"的一条，
       配对成功就把标签信息挂到这件物品上（演示后端的时间窗配对逻辑）。
    3. ITEM_OUT / ITEM_MOVED → 仅记录。

真实部署时，后端用真的 OCR/VLM 替换 fake_ocr() 即可，端侧不用改。

用法:
    python3 mock_cloud_ocr.py            # 默认监听 0.0.0.0:8000
    python3 mock_cloud_ocr.py --port 8000 --host 0.0.0.0

依赖: 仅 Python3 标准库。
"""
import argparse
import base64
import json
import os
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "received")
os.makedirs(SAVE_DIR, exist_ok=True)

# 模拟"待配对标签"列表（真实后端是数据库表 pending_labels）
# 每条: { ts: 毫秒, ttl: 秒, label_data: {...}, consumed: bool }
PENDING_LABELS = []

MOCK_PRODUCTS = [
    ("纯牛奶",     "蒙牛",   "2026-06-15"),
    ("橙汁",       "汇源",   "2026-08-03"),
    ("酸奶",       "光明",   "2026-06-09"),
    ("乐事薯片",   "乐事",   "2026-09-12"),
    ("速冻水饺",   "三全",   "2026-07-20"),
    ("冷鲜鸡胸肉", "正大",   "2026-06-05"),
]


def fake_ocr():
    """模拟云端 OCR/VLM 读标签。真实环境换成调用真模型。"""
    name, brand, expiry = random.choice(MOCK_PRODUCTS)
    return {"name": name, "brand": brand, "expire_date": expiry}


def save_crop(tag, b64):
    if not b64:
        return ""
    try:
        jpg = base64.b64decode(b64)
        path = os.path.join(SAVE_DIR, f"{tag}_{int(time.time()*1000)}.jpg")
        with open(path, "wb") as f:
            f.write(jpg)
        return os.path.basename(path)
    except Exception as e:
        print(f"[MOCK] 图片解码/保存失败: {e}")
        return ""


def now_ms():
    return int(time.time() * 1000)


def match_pending_label():
    """后端时间窗配对：找最近、未消费、未过期的一条标签（后进先出）。"""
    t = now_ms()
    # 按时间倒序遍历（最近的优先）
    for entry in sorted(PENDING_LABELS, key=lambda e: e["ts"], reverse=True):
        if entry["consumed"]:
            continue
        if t - entry["ts"] > entry["ttl"] * 1000:
            continue  # 过期
        entry["consumed"] = True
        return entry["label_data"]
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as e:
            print(f"[MOCK] JSON 解析失败: {e}")
            self._reply(400, {"error": "bad json"})
            return

        path = self.path.rstrip("/")
        if path.endswith("/label_scan"):
            self.handle_label_scan(payload)
        elif path.endswith("/item"):
            self.handle_item(payload)
        else:
            print(f"[MOCK] 未知路径: {self.path}")
            self._reply(404, {"error": "unknown path"})

    # ----- 标签扫描：存图 + 模拟 OCR + 写入待配对列表 -----
    def handle_label_scan(self, payload):
        device = payload.get("device_id", "?")
        ttl = payload.get("ttl_seconds", 300)
        b64 = payload.get("label_image", "")
        saved = save_crop("label", b64)

        label_data = fake_ocr()
        PENDING_LABELS.append({
            "ts": now_ms(), "ttl": ttl,
            "label_data": label_data, "consumed": False,
        })
        print(f"[MOCK][LABEL_SCAN] {device} ttl={ttl}s crop={len(b64)}B "
              f"-> OCR={label_data}" + (f"  (saved {saved})" if saved else ""))
        self._reply(200, {"ok": True, "label_data": label_data})

    # ----- 出入库事件：ITEM_IN 时尝试配对标签 -----
    def handle_item(self, payload):
        device = payload.get("device_id", "?")
        event = payload.get("event_type", "?")
        data = payload.get("data", [])
        for it in data:
            tid = it.get("local_track_id", -1)
            cat = it.get("category", "?")
            conf = it.get("confidence", 0)
            bbox = it.get("bbox", [])
            saved = save_crop(f"item{tid}", it.get("crop_image", ""))

            paired = None
            if event == "ITEM_IN":
                paired = match_pending_label()  # 时间窗配对

            line = (f"[MOCK][{event}] {device} local_track_id={tid} "
                    f"category={cat} conf={conf} bbox(xywh)={bbox}")
            if saved:
                line += f"  (saved {saved})"
            if paired:
                line += f"  ★ 配对标签 -> {paired}"
            elif event == "ITEM_IN":
                line += "  (无可配对标签)"
            print(line)
        self._reply(200, {"ok": True})

    def _reply(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[MOCK] 云端 mock 服务已启动: http://{args.host}:{args.port}")
    print(f"[MOCK]   POST /events/item        出入库事件")
    print(f"[MOCK]   POST /events/label_scan  标签扫描")
    print(f"[MOCK] 裁剪图保存目录: {SAVE_DIR}")
    print("[MOCK] 等待端侧上传... (Ctrl+C 退出)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[MOCK] 退出")
        srv.shutdown()


if __name__ == "__main__":
    main()
