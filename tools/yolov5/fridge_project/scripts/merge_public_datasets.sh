#!/bin/bash
# 一键合并所有公开数据集到 fridge_project/datasets/public_merged/
# Linux / WSL 版本(对应 .bat 的 shell 等价物)
#
# 使用:
#   cd /mnt/c/.../yolov5
#   bash fridge_project/scripts/merge_public_datasets.sh
# 或者给执行权限:
#   chmod +x fridge_project/scripts/merge_public_datasets.sh
#   fridge_project/scripts/merge_public_datasets.sh
#
# 注意:
#   每次合并前会先清空 public_merged/ 下的图片和标签目录,避免上一次合并的
#   train_000001.jpg 文件残留(这些文件是按"序号 + 切分顺序"重命名的,
#   保留旧文件会引发图片/标签错配)。yaml 文件本身会被新一轮直接覆盖,
#   不需要手动删。

set -e  # 任何一步出错就停

# 切到 yolov5 根目录(脚本所在目录的上两层)
cd "$(dirname "$0")/../.."

OUT_DIR="fridge_project/datasets/public_merged"

echo "============================================================"
echo "开始合并公开数据集"
echo "当前工作目录: $(pwd)"
echo "输出目录:     $OUT_DIR"
echo "============================================================"

# ---- 清理旧产物 ----
if [ -d "$OUT_DIR/images" ] || [ -d "$OUT_DIR/labels" ]; then
    echo "清理旧的合并产物 (images/ + labels/)..."
    rm -rf "$OUT_DIR/images" "$OUT_DIR/labels"
fi

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
    --source-yolo "datasets/detect can.yolov5pytorch" \
    --source-yolo "datasets/glass.yolov5pytorch" \
    --source-yolo "datasets/Plastic Bottle 2.0.yolov5pytorch" \
    --output      "$OUT_DIR" \
    --val-ratio   0.1

echo ""
echo "============================================================"
echo "合并完成! 数据集位置:"
echo "  $OUT_DIR/"
echo "训练时使用 yaml:"
echo "  $OUT_DIR/public_merged.yaml"
echo "============================================================"
