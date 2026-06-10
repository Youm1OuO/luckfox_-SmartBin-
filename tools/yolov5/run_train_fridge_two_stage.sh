#!/bin/bash
set -e

# Two-stage fine-tuning for the fridge demo dataset.
# Stage 1: freeze backbone, adapt detection head to the fridge camera/domain.
# Stage 2: unfreeze all layers, lightly polish the whole model on the same domain.

cd /home/muyou/Projects/workspace/luckfox_demo/tools/yolov5

DATA_YAML="/home/muyou/Projects/workspace/luckfox_demo/tools/luckfox_pico_rkmpi_example/example/luckfox_pico_rtsp_yolov5/dataset/data.yaml"
BASE_WEIGHTS="runs/train/hand_plus_v1/weights/best.pt"
IMGSZ=704
BATCH_SIZE=32
WORKERS=8

STAGE1_NAME="fridge_self_704_freeze10"
STAGE2_NAME="fridge_self_704_unfreeze"

echo "== Stage 1: freeze 10, train fridge head/domain =="
python fridge_project/scripts/train.py stage-c \
    --data "$DATA_YAML" \
    --weights "$BASE_WEIGHTS" \
    --epochs 800 \
    --batch-size "$BATCH_SIZE" \
    --imgsz "$IMGSZ" \
    --workers "$WORKERS" \
    --cache ram \
    --name "$STAGE1_NAME" \
    --freeze 10

echo "== Stage 2: unfreeze all layers, polish for demo fridge =="
python fridge_project/scripts/train.py stage-c \
    --data "$DATA_YAML" \
    --weights "runs/train/${STAGE1_NAME}/weights/best.pt" \
    --epochs 300 \
    --batch-size "$BATCH_SIZE" \
    --imgsz "$IMGSZ" \
    --workers "$WORKERS" \
    --cache ram \
    --name "$STAGE2_NAME" \
    --freeze 0

echo "Done."
echo "Stage 1 best: runs/train/${STAGE1_NAME}/weights/best.pt"
echo "Stage 2 best: runs/train/${STAGE2_NAME}/weights/best.pt"
