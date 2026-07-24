#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_dataset.py — 合并新旧数据集

用法：
    python3 tools/merge_dataset.py --old dataset_old --new dataset_new --output dataset

功能：
    1. 把旧数据集原样复制到输出目录
    2. 把新数据集的图片和标注重新编号（从旧数据最大编号+1 开始）
    3. 复制到输出目录，与旧数据合并

输出结构：
    dataset/
      images/
        train/
        val/
      labels/
        train/
        val/
"""

import argparse
import os
import shutil


def get_max_number(directory):
    """自动检测目录中图片的最大编号"""
    max_num = 0
    if not os.path.isdir(directory):
        return 0
    for f in os.listdir(directory):
        name = os.path.splitext(f)[0]
        if name.isdigit():
            max_num = max(max_num, int(name))
    return max_num


def copy_files(src_dir, dst_dir, extensions):
    """复制指定扩展名的文件，返回复制数量"""
    if not os.path.isdir(src_dir):
        return 0
    count = 0
    for f in os.listdir(src_dir):
        if os.path.splitext(f)[1].lower() in extensions:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description="合并新旧数据集")
    ap.add_argument("--old", required=True, help="旧数据集目录")
    ap.add_argument("--new", required=True, help="新数据集目录（make_dataset.py 生成的）")
    ap.add_argument("--offset", type=int, default=0,
                    help="编号偏移量（默认自动检测旧数据集最大编号）")
    ap.add_argument("--output", default="dataset", help="输出目录（默认 dataset）")
    args = ap.parse_args()

    old_dir = args.old
    new_dir = args.new
    output_dir = args.output

    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
    LBL_EXT = {".txt", ".json"}

    # 检查目录
    for d, name in [(old_dir, "旧数据集"), (new_dir, "新数据集")]:
        if not os.path.isdir(d):
            print(f"[错误] {name}目录不存在: {d}")
            return

    # 创建输出目录结构
    for split in ("train", "val"):
        for subdir in ("images", "labels"):
            os.makedirs(os.path.join(output_dir, subdir, split), exist_ok=True)

    # ---------------------------------------------------------------
    # 1. 复制旧数据
    # ---------------------------------------------------------------
    print("[1/3] 复制旧数据...")
    for split in ("train", "val"):
        for subdir, exts in [("images", IMG_EXT), ("labels", LBL_EXT)]:
            # 尝试两种目录结构：flat (images/train/) 和 Roboflow (train/images/)
            src = os.path.join(old_dir, subdir, split)
            if not os.path.isdir(src):
                src = os.path.join(old_dir, split, subdir)
            dst = os.path.join(output_dir, subdir, split)
            count = copy_files(src, dst, exts)
            print(f"  旧 {subdir}/{split}: {count} 个文件")

    # ---------------------------------------------------------------
    # 2. 检测偏移量
    # ---------------------------------------------------------------
    if args.offset > 0:
        offset = args.offset
    else:
        offset = get_max_number(os.path.join(output_dir, "images", "train"))
    print(f"\n[2/3] 编号偏移量: {offset}")

    # ---------------------------------------------------------------
    # 3. 复制新数据（重新编号）
    # ---------------------------------------------------------------
    print(f"\n[3/3] 复制新数据（编号从 {offset + 1} 开始）...")
    new_num = offset + 1

    for split in ("train", "val"):
        # 获取新数据的图片目录（尝试两种结构）
        new_img_dir = os.path.join(new_dir, "images", split)
        if not os.path.isdir(new_img_dir):
            new_img_dir = os.path.join(new_dir, split, "images")

        new_lbl_dir = os.path.join(new_dir, "labels", split)
        if not os.path.isdir(new_lbl_dir):
            new_lbl_dir = os.path.join(new_dir, split, "labels")

        if not os.path.isdir(new_img_dir):
            print(f"  新 {split}: 目录不存在，跳过")
            continue

        # 按文件名排序
        img_files = sorted([
            f for f in os.listdir(new_img_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXT
        ])

        count = 0
        for img_file in img_files:
            old_base = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]
            new_name = f"{new_num:06d}{ext}"

            # 复制图片
            shutil.copy2(
                os.path.join(new_img_dir, img_file),
                os.path.join(output_dir, "images", split, new_name)
            )

            # 复制对应的标注文件
            if os.path.isdir(new_lbl_dir):
                for lbl_ext in (".txt", ".json"):
                    old_lbl = os.path.join(new_lbl_dir, old_base + lbl_ext)
                    if os.path.exists(old_lbl):
                        new_lbl = os.path.join(output_dir, "labels", split,
                                               f"{new_num:06d}{lbl_ext}")
                        shutil.copy2(old_lbl, new_lbl)

            new_num += 1
            count += 1

        print(f"  新 {split}: {count} 张（编号 {offset + 1} ~ {new_num - 1}）")

    # 统计未标注的新图片
    unlabeled = []
    train_img_dir = os.path.join(output_dir, "images", "train")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    if os.path.isdir(train_img_dir):
        for f in sorted(os.listdir(train_img_dir)):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                base = os.path.splitext(f)[0]
                if not os.path.exists(os.path.join(train_lbl_dir, base + ".txt")):
                    unlabeled.append(f)

    print(f"\n完成！")
    if unlabeled:
        print(f"  还有 {len(unlabeled)} 张图片未标注（新图片）")
        print(f"  用 labelme 打开标注即可: {os.path.abspath(train_img_dir)}")


if __name__ == "__main__":
    main()
