#!/bin/bash
python fridge_project/scripts/train.py stage-c \
    --data fridge_project/datasets/public_merged/public_merged.yaml \
    --weights runs/train/hand_plus_v1/weights/best.pt \
    --epochs 600 \
    --batch-size 32 \
    --imgsz 640 \
    --workers 8 \
    --cache ram \
    --name hand_plus_v2 \
    --freeze 0
