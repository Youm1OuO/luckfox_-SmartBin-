"""
统计 public_merged 数据集中各类别的图片数和标注框数。

用法:
    python fridge_project/scripts/stats_classes.py             # 统计全部类别
    python fridge_project/scripts/stats_classes.py 0 1 2 6 18  # 只统计指定类 ID
    python fridge_project/scripts/stats_classes.py --top 10     # 只显示框数最多的前 10 类
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 定位目录
HERE = Path(__file__).resolve().parent          # fridge_project/scripts/
PROJ = HERE.parent                              # fridge_project/
YOLO_ROOT = PROJ.parent                         # yolov5/
LABELS_DIR = PROJ / "datasets" / "public_merged" / "labels"
CLASSES_FILE = PROJ / "configs" / "classes.yaml"


def load_class_names() -> list[str]:
    """从 classes.yaml 读取类别名列表（顺序 = 类别 ID）。"""
    names = []
    with open(CLASSES_FILE, encoding="utf-8") as f:
        in_classes = False
        for line in f:
            if line.strip() == "classes:":
                in_classes = True
                continue
            if in_classes:
                if line.startswith("  - "):
                    # 格式: "  - apple          # 0   苹果"
                    name = line.strip().lstrip("- ").split("#")[0].strip().split()[0]
                    names.append(name)
                elif not line.startswith(" ") and line.strip():
                    break  # 到了下一个顶层 key
    return names


def count_dataset(class_names: list[str], target_ids: set[int] | None = None):
    """
    统计 train + val 的图片数和标注框数。
    返回 {class_id: {"name": str, "images": int, "boxes": int}}
    """
    box_counts: Counter = Counter()
    img_sets: dict[int, set[str]] = defaultdict(set)

    for split in ("train", "val"):
        label_dir = LABELS_DIR / split
        if not label_dir.is_dir():
            continue
        for txt_file in label_dir.glob("*.txt"):
            stem = txt_file.stem
            for line in txt_file.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if not parts:
                    continue
                cid = int(parts[0])
                if target_ids is not None and cid not in target_ids:
                    continue
                box_counts[cid] += 1
                img_sets[cid].add(stem)

    results = {}
    for cid in sorted(box_counts.keys()):
        results[cid] = {
            "name": class_names[cid] if cid < len(class_names) else f"class_{cid}",
            "images": len(img_sets[cid]),
            "boxes": box_counts[cid],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", type=int, default=None,
                        help="要统计的类 ID（不传则统计全部）")
    parser.add_argument("--top", type=int, default=0,
                        help="只显示框数最多的前 N 个类别")
    args = parser.parse_args()

    class_names = load_class_names()
    target_ids = set(args.ids) if args.ids else None

    results = count_dataset(class_names, target_ids)

    if args.top and not args.ids:
        # 按框数降序取前 N
        top_ids = sorted(results, key=lambda c: results[c]["boxes"], reverse=True)[:args.top]
        results = {c: results[c] for c in top_ids}

    # 输出
    print(f"{'类别':<16} {'ID':>4}  {'图片数':>8}  {'标注框数':>10}")
    print("-" * 46)

    total_imgs = 0
    total_boxes = 0
    for cid in sorted(results.keys()):
        r = results[cid]
        print(f"{r['name']:<16} {cid:>4}  {r['images']:>8}  {r['boxes']:>10}")
        total_imgs += r["images"]
        total_boxes += r["boxes"]

    print("-" * 46)
    print(f"{'合计':<16}       {total_imgs:>8}  {total_boxes:>10}")


if __name__ == "__main__":
    main()
