#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_整理数据集.py — 把拍好的照片整理成训练格式

双击运行即可，会自动：
  1. 找到 dataset_raw 里所有照片
  2. 随机打乱，分成训练集(80%)和验证集(20%)
  3. 生成 YOLO 训练需要的目录结构
"""

import os
import sys
import subprocess

# 找到 make_dataset.py 的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
make_dataset = os.path.join(script_dir, "tools", "make_dataset.py")

if not os.path.exists(make_dataset):
    print(f"[错误] 找不到 make_dataset.py")
    print(f"  期望位置: {make_dataset}")
    input("按回车键退出...")
    sys.exit(1)

# 找到 dataset_raw 目录
raw_dir = os.path.join(script_dir, "dataset_raw")
if not os.path.isdir(raw_dir):
    print(f"[错误] 找不到照片目录: {raw_dir}")
    print(f"  请先运行 '1_采集照片.bat' 拍照")
    input("按回车键退出...")
    sys.exit(1)

# 收集所有 batch 子目录
batch_dirs = []
for name in sorted(os.listdir(raw_dir)):
    full = os.path.join(raw_dir, name)
    if os.path.isdir(full) and name.startswith("batch"):
        batch_dirs.append(full)

if not batch_dirs:
    # 没有 batch 子目录，直接用 dataset_raw 本身
    # 检查里面有没有图片
    imgs = [f for f in os.listdir(raw_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if imgs:
        batch_dirs = [raw_dir]
    else:
        print(f"[错误] dataset_raw 里没有找到照片")
        print(f"  请先运行 '1_采集照片.bat' 拍照")
        input("按回车键退出...")
        sys.exit(1)

print("=" * 50)
print("  冰箱数据集整理工具")
print("=" * 50)
print()
print(f"  找到 {len(batch_dirs)} 批照片:")
for d in batch_dirs:
    count = len([f for f in os.listdir(d)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    print(f"    {d}  ({count} 张)")
print()

output_dir = os.path.join(script_dir, "dataset")

# 调用 make_dataset.py
cmd = [
    sys.executable, make_dataset,
    "--input"] + batch_dirs + [
    "--output", output_dir,
    "--split", "0.2"
]

print("  正在整理...")
print()
result = subprocess.run(cmd)

if result.returncode == 0:
    print()
    print("=" * 50)
    print("  整理完成！")
    print(f"  数据集在: {output_dir}")
    print("=" * 50)
    print()
    print("  下一步：运行 '3_打开标注工具.bat' 开始标注")
else:
    print()
    print("[错误] 整理失败，请检查上面的错误信息")

input()
