#!/bin/bash
# 统计 public_merged 数据集中各（或指定）类别的图片数和标注框数
#
# 用法:
#   bash stats_classes.sh              # 统计所有类别
#   bash stats_classes.sh 0 1 2 6 18  # 只统计指定类 ID

cd "$(dirname "$0")/.." || exit 1

LABELS="fridge_project/datasets/public_merged/labels"
CLASSES_FILE="fridge_project/configs/classes.yaml"

# 从 classes.yaml 提取类别名列表
mapfile -t ALL_NAMES < <(grep "^  - " "$CLASSES_FILE" | sed 's/^  - //' | awk '{print $1}')

# 如果命令行传了类 ID，只统计那些；否则统计全部
if [ $# -gt 0 ]; then
    IDS=("$@")
else
    IDS=()
    for i in "${!ALL_NAMES[@]}"; do
        IDS+=("$i")
    done
fi

printf "%-16s %4s  %8s  %10s\n" "类别" "ID" "图片数" "标注框数"
printf "%s\n" "----------------------------------------------"

total_imgs=0
total_boxes=0

for cid in "${IDS[@]}"; do
    name="${ALL_NAMES[$cid]:-unknown}"
    imgs=$(grep -rl "^${cid} " "$LABELS/train/" "$LABELS/val/" 2>/dev/null | wc -l)
    boxes=$(grep -rh "^${cid} " "$LABELS/train/" "$LABELS/val/" 2>/dev/null | wc -l)
    printf "%-16s %4d  %8d  %10d\n" "$name" "$cid" "$imgs" "$boxes"
    total_imgs=$((total_imgs + imgs))
    total_boxes=$((total_boxes + boxes))
done

printf "%s\n" "----------------------------------------------"
printf "%-16s      %8d  %10d\n" "合计" "$total_imgs" "$total_boxes"
