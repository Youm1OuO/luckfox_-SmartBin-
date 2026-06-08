#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_labels.py — 把 labelme 的 JSON 标注转成 YOLO 格式

用法：
    python3 tools/convert_labels.py

它会自动：
  1. 把 dataset/images/train/ 和 dataset/images/val/ 里的 .json 文件
     移动到 dataset/labels/train/ 和 dataset/labels/val/
  2. 把 JSON 转成 YOLO 的 .txt 格式（class_id cx cy w h）
"""

import json
import os
import glob
import shutil

# 类别名 → ID 映射（必须和 labels_list.txt 一致）
CLASSES = [
    'apple', 'banana', 'orange', 'tomato', 'lemon', 'pear', 'grape',
    'strawberry', 'watermelon', 'cantaloupe', 'papaya', 'avocado',
    'cucumber', 'carrot', 'potato', 'onion', 'bell_pepper', 'leafy_green',
    'egg', 'meat_pack', 'fish_pack',
    'milk_box', 'milk_bottle', 'yogurt_cup', 'juice_bottle', 'soda_can',
    'water_bottle',
    'bagged_food', 'boxed_food', 'canned_food', 'jar_food', 'plastic_wrap',
    'fresh_box',
    'hand',
    'mushroom', 'pumpkin', 'garlic', 'ginger', 'radish', 'sweet_potato',
    'walnut', 'coriander', 'okra',
]


def convert_json_to_yolo(json_path, txt_path):
    """把一个 labelme JSON 文件转成 YOLO txt 格式"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data.get('imageWidth', 0)
    img_h = data.get('imageHeight', 0)
    if img_w == 0 or img_h == 0:
        print(f"  警告: {json_path} 缺少图片尺寸信息，跳过")
        return 0

    lines = []
    for shape in data.get('shapes', []):
        label = shape['label']
        if label not in CLASSES:
            print(f"  警告: 未知类别 '{label}'，跳过（请检查拼写）")
            continue

        cls_id = CLASSES.index(label)
        pts = shape['points']
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # 转成 YOLO 格式: class_id center_x center_y width height (归一化 0~1)
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    with open(txt_path, 'w', encoding='utf-8') as out:
        out.write('\n'.join(lines) + '\n' if lines else '')

    return len(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..')
    dataset_dir = os.path.join(project_dir, 'dataset')

    if not os.path.isdir(dataset_dir):
        print(f"[错误] 找不到 dataset 目录: {dataset_dir}")
        print("  请先运行: python3 tools/make_dataset.py --input dataset_raw --output dataset")
        return

    total_converted = 0
    total_moved = 0

    for split in ('train', 'val'):
        img_dir = os.path.join(dataset_dir, 'images', split)
        lbl_dir = os.path.join(dataset_dir, 'labels', split)

        if not os.path.isdir(img_dir):
            print(f"[跳过] 目录不存在: {img_dir}")
            continue

        os.makedirs(lbl_dir, exist_ok=True)

        # 1. 把 images 目录里的 .json 文件移到 labels 目录
        json_files = glob.glob(os.path.join(img_dir, '*.json'))
        if json_files:
            print(f"\n[{split}] 移动 {len(json_files)} 个 JSON 文件到 labels/{split}/")
            for jf in json_files:
                dest = os.path.join(lbl_dir, os.path.basename(jf))
                shutil.move(jf, dest)
                total_moved += 1

        # 2. 转换 labels 目录里的所有 JSON 为 YOLO txt
        json_in_lbl = glob.glob(os.path.join(lbl_dir, '*.json'))
        if json_in_lbl:
            print(f"[{split}] 转换 {len(json_in_lbl)} 个 JSON → YOLO txt")
            for jf in sorted(json_in_lbl):
                txt_path = jf.replace('.json', '.txt')
                count = convert_json_to_yolo(jf, txt_path)
                if count > 0:
                    total_converted += 1
                    print(f"  {os.path.basename(jf)} → {os.path.basename(txt_path)} ({count} 个框)")
        else:
            print(f"[{split}] 没有找到 JSON 标注文件")

    print(f"\n完成！移动了 {total_moved} 个 JSON，转换了 {total_converted} 个为 YOLO 格式")
    print(f"YOLO txt 文件在: dataset/labels/train/ 和 dataset/labels/val/")


if __name__ == '__main__':
    main()
