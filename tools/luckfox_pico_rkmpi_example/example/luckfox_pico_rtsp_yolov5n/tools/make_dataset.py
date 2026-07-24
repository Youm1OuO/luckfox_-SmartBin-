#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset.py — 把截图整理成 YOLO 训练数据集

用法:
    # 单个目录
    python3 make_dataset.py --input dataset_raw/

    # 多个目录（分多次采集，自动合并）
    python3 make_dataset.py --input batch1/ batch2/ batch3/

    # 指定输出目录和验证集比例
    python3 make_dataset.py --input dataset_raw/ --output my_dataset --split 0.15

功能:
    1. 从一个或多个目录中读取所有 .jpg 图片（支持分多次采集）
    2. 自动重命名（000001.jpg, 000002.jpg, ...），避免不同批次文件名冲突
    3. 随机打乱，按比例划分 train / val
    4. 生成 YOLO 标准目录结构:
         dataset/
         ├── images/
         │   ├── train/
         │   └── val/
         ├── labels/
         │   ├── train/   (空目录，等 LabelImg 标注后填入 .txt)
         │   └── val/
         ├── classes.txt  (类别列表，供 LabelImg 使用)
         └── data.yaml    (YOLO 训练配置)
    5. 打印 LabelImg 标注指引

依赖: 仅 Python3 标准库
"""

import argparse
import os
import random
import shutil


def read_labels(label_path):
    """从 labels_list.txt 读取类别列表"""
    labels = []
    if not os.path.exists(label_path):
        print(f"[WARN] 标签文件不存在: {label_path}")
        print("       将生成空的 classes.txt，请手动填写类别名")
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                labels.append(name)
    return labels


def collect_images(input_dirs):
    """从多个目录中收集所有图片（支持分多次采集）

    支持两种输入：
      - 图片目录: 直接包含 .jpg 的目录 → 收集里面的图片
      - YOLO 数据集目录: 包含 images/train/ 的目录 → 收集 images/ 下所有图片，
        同时把对应的 labels/ 也纳入合并
    """
    all_images = []  # [(目录, 文件名), ...]
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            print(f"[WARN] 目录不存在，跳过: {input_dir}")
            continue

        # 检测是否是已有的 YOLO 数据集目录（含 images/ 子目录）
        images_root = os.path.join(input_dir, "images")
        if os.path.isdir(images_root):
            # 这是已有的 YOLO 数据集 → 收集 images/ 下所有子目录的图片
            for sub in ("train", "val"):
                sub_dir = os.path.join(images_root, sub)
                if not os.path.isdir(sub_dir):
                    continue
                imgs = sorted([
                    f for f in os.listdir(sub_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ])
                for fname in imgs:
                    all_images.append((sub_dir, fname))
                if imgs:
                    print(f"  {sub_dir}/: {len(imgs)} 张 (已有YOLO数据集)")
        else:
            # 普通图片目录
            images = sorted([
                f for f in os.listdir(input_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ])
            for fname in images:
                all_images.append((input_dir, fname))
            if images:
                print(f"  {input_dir}/: {len(images)} 张")
    return all_images


def collect_labels(input_dirs):
    """从已有的 YOLO 数据集中收集标注文件（.txt）

    返回: { (原图片目录, 图片文件名不带后缀) : 标注文件路径 }
    这样在复制图片时，可以同时把对应的标注也复制过去，避免已标注的数据丢失。
    """
    label_map = {}  # key = (图片目录, 图片无后缀名), value = 标注文件路径
    for input_dir in input_dirs:
        labels_root = os.path.join(input_dir, "labels")
        images_root = os.path.join(input_dir, "images")
        if not os.path.isdir(labels_root) or not os.path.isdir(images_root):
            continue

        for sub in ("train", "val"):
            lbl_dir = os.path.join(labels_root, sub)
            img_dir = os.path.join(images_root, sub)
            if not os.path.isdir(lbl_dir):
                continue
            for fname in os.listdir(lbl_dir):
                if not fname.endswith(".txt"):
                    continue
                base = fname[:-4]  # 去掉 .txt
                # 检查对应的图片是否存在
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    img_path = os.path.join(img_dir, base + ext)
                    if os.path.exists(img_path):
                        label_map[(img_dir, base)] = os.path.join(lbl_dir, fname)
                        break
    return label_map


def main():
    ap = argparse.ArgumentParser(description="把截图整理成 YOLO 训练数据集")
    ap.add_argument("--input", nargs="+", default=["dataset_raw"],
                    help="截图目录，可指定多个（支持分多次采集）。"
                         "例: --input batch1/ batch2/ batch3/")
    ap.add_argument("--output", default="dataset",
                    help="输出目录（默认 dataset/）")
    ap.add_argument("--split", type=float, default=0.2,
                    help="验证集比例（默认 0.2 = 20%%）")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子，保证可复现（默认 42）")
    args = ap.parse_args()

    output_dir = args.output
    val_ratio = args.split

    # ---------------------------------------------------------------
    #  1. 收集图片 + 已有标注（支持多个目录，支持合并已有YOLO数据集）
    # ---------------------------------------------------------------
    print(f"扫描图片目录:")
    all_images = collect_images(args.input)
    existing_labels = collect_labels(args.input)

    if not all_images:
        print(f"[ERROR] 没有找到任何图片")
        print(f"  检查目录: {args.input}")
        return

    print(f"共找到 {len(all_images)} 张图片", end="")
    if existing_labels:
        print(f"，其中 {len(existing_labels)} 张已有标注（会自动保留）")
    else:
        print()

    # ---------------------------------------------------------------
    #  2. 随机划分 train / val
    # ---------------------------------------------------------------
    random.seed(args.seed)
    random.shuffle(all_images)

    val_count = max(1, int(len(all_images) * val_ratio))
    val_images = all_images[:val_count]
    train_images = all_images[val_count:]

    print(f"划分: train={len(train_images)}, val={len(val_images)}")

    # ---------------------------------------------------------------
    #  3. 创建目录结构 + 复制图片
    #  重命名为统一格式: 000001.jpg, 000002.jpg, ...
    #  这样即使来自不同目录，也不会文件名冲突
    # ---------------------------------------------------------------
    dirs = {}
    for split_name in ("train", "val"):
        dirs[f"img_{split_name}"] = os.path.join(output_dir, "images", split_name)
        dirs[f"lbl_{split_name}"] = os.path.join(output_dir, "labels", split_name)

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    def copy_images(img_list, split_name):
        dst = dirs[f"img_{split_name}"]
        lbl_dst = dirs[f"lbl_{split_name}"]
        copied = 0
        labels_copied = 0
        for i, (src_dir, fname) in enumerate(img_list):
            src = os.path.join(src_dir, fname)
            # 统一重命名，避免不同批次文件名冲突
            base, ext = os.path.splitext(fname)
            new_name = f"{i+1:06d}{ext}"
            shutil.copy2(src, os.path.join(dst, new_name))

            # 如果这张图片已有标注，也一起复制过去（重命名匹配新图片名）
            lbl_key = (src_dir, base)
            if lbl_key in existing_labels:
                new_lbl_name = f"{i+1:06d}.txt"
                shutil.copy2(existing_labels[lbl_key],
                             os.path.join(lbl_dst, new_lbl_name))
                labels_copied += 1
            copied += 1
        print(f"  已复制 {copied} 张图片到 {dst}"
              + (f"（其中 {labels_copied} 张带已有标注）" if labels_copied else ""))

    copy_images(train_images, "train")
    copy_images(val_images, "val")

    # ---------------------------------------------------------------
    #  4. 读取类别列表
    # ---------------------------------------------------------------
    # 尝试从 model/labels_list.txt 读取
    script_dir = os.path.dirname(os.path.abspath(__file__))
    label_path = os.path.join(script_dir, "..", "model", "labels_list.txt")
    labels = read_labels(label_path)

    # 如果没找到，尝试上级目录
    if not labels:
        label_path = os.path.join(script_dir, "model", "labels_list.txt")
        labels = read_labels(label_path)

    # 写 classes.txt（LabelImg 用）
    classes_file = os.path.join(output_dir, "classes.txt")
    with open(classes_file, "w", encoding="utf-8") as f:
        for name in labels:
            f.write(name + "\n")
    print(f"  已生成 {classes_file} ({len(labels)} 个类别)")

    # ---------------------------------------------------------------
    #  5. 生成 data.yaml（YOLO 训练配置）
    # ---------------------------------------------------------------
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# YOLOv5 训练配置 — 由 make_dataset.py 自动生成\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"\nnc: {len(labels)}\n")
        f.write(f"names: {labels}\n")
    print(f"  已生成 {yaml_path}")

    # ---------------------------------------------------------------
    #  6. 打印使用指引
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("  数据集已生成! 下一步用 LabelImg 标注:")
    print("=" * 60)
    print()
    print("  1. 安装 LabelImg:")
    print("     pip install labelImg")
    print()
    print("  2. 启动标注:")
    print(f"     labelImg {os.path.abspath(dirs['img_train'])} "
          f"{os.path.abspath(classes_file)}")
    print()
    print("  3. LabelImg 中的设置:")
    print("     - 点击左侧 'PascalVOC' 按钮切换为 'YOLO' 格式")
    print("     - 标注结果 (.txt) 会自动保存到 images/train 同级的 labels/train/")
    print(f"     - 如果保存路径不对，手动设置: Change Save Dir → {os.path.abspath(dirs['lbl_train'])}")
    print()
    print("  4. 标注完 train 后，再标 val:")
    print(f"     labelImg {os.path.abspath(dirs['img_val'])} "
          f"{os.path.abspath(classes_file)}")
    print()
    print("  5. 开始训练 YOLOv5:")
    print(f"     cd /path/to/yolov5")
    print(f"     python train.py --data {os.path.abspath(yaml_path)} --weights yolov5n.pt --epochs 100")
    print()
    print("  提示:")
    print("     - 每个类别至少标注 100-200 张，效果才好")
    print("     - 尽量覆盖不同角度、光照、遮挡情况")
    print("     - 标注框要紧贴物体边缘，不要留太大空隙")
    print("=" * 60)


if __name__ == "__main__":
    main()
