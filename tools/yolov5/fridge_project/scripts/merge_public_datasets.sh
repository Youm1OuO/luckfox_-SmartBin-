#!/bin/bash
# 一键合并 4 个公开数据集到 fridge_project/datasets/public_merged/
# Linux / WSL 版本(对应 .bat 的 shell 等价物)
#
# 使用:
#   cd /mnt/c/.../yolov5
#   bash fridge_project/scripts/merge_public_datasets.sh
# 或者给执行权限:
#   chmod +x fridge_project/scripts/merge_public_datasets.sh
#   fridge_project/scripts/merge_public_datasets.sh

set -e  # 任何一步出错就停

# 切到 yolov5 根目录(脚本所在目录的上两层)
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "开始合并公开数据集"
echo "当前工作目录: $(pwd)"
echo "============================================================"

python fridge_project/scripts/prepare_dataset.py \
    --source-yolo "datasets/FOOD-INGREDIENTS dataset.yolov5pytorch" \
    --source-yolo "datasets/Beverage Containers.yolov5pytorch" \
    --source-yolo "datasets/Hand Detection.yolov5pytorch" \
    --source-yolo "datasets/hand.yolov5pytorch" \
    --source-yolo "datasets/Vegetables.yolov5pytorch" \
    --source-yolo "datasets/Fruits and Vegetables.yolov5pytorch" \
    --source-yolo "datasets/fruit detection.yolov5pytorch" \
    --source-yolo "datasets/Cantaloupe Detection.yolov5pytorch" \
    --source-yolo "datasets/Watermelon.yolov5pytorch" \
    --source-yolo "datasets/grape.yolov5pytorch" \
    --source-yolo "datasets/Milk.yolov5pytorch" \
    --source-yolo "datasets/papaya.yolov5pytorch" \
    --output      "fridge_project/datasets/public_merged" \
    --val-ratio   0.1

echo ""
echo "============================================================"
echo "合并完成! 数据集位置:"
echo "  fridge_project/datasets/public_merged/"
echo "训练时使用 yaml:"
echo "  fridge_project/datasets/public_merged/public_merged.yaml"
echo "============================================================"
