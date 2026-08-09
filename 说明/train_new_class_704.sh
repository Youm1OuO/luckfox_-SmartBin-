#!/usr/bin/env bash

#   要先运行这一条，生成自己的 data.yaml：
cd /home/muyou/Projects/workspace/luckfox_demo/tools/yolov5n

/home/muyou/miniconda3/envs/DL_env/bin/python fridge_project/scripts/build_data_yaml.py \
    --dataset-root /home/muyou/Projects/workspace/luckfox_demo/tools/luckfox_pico_rkmpi_example/example/luckfox_pico_rtsp_yolov5n/dataset \
    --output /home/muyou/Projects/workspace/luckfox_demo/tools/luckfox_pico_rkmpi_example/example/luckfox_pico_rtsp_yolov5n/dataset/data.yaml



# 然后运行这一条：
cd /home/muyou/Projects/workspace/luckfox_demo/tools/yolov5n

python fridge_project/scripts/train.py stage-c \
    --data /home/muyou/Projects/workspace/luckfox_demo/tools/luckfox_pico_rkmpi_example/example/luckfox_pico_rtsp_yolov5n/dataset/data.yaml \
    --weights runs/train/fridge_self_s1_704_unfreeze/weights/best.pt \
    --epochs 600 \
    --batch-size 32 \
    --imgsz 704 \
    --workers 8 \
    --cache ram \
    --name fridge_new_class49_704 \
    --freeze 0


# 训练时 stage-b 与 stage-c 的区别：
# 
#                      stage-b                             stage-c
#   ━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#    本来用途          用大量公开数据先打基础              用你自己的冰箱数据继续适应
#   ────────────────  ──────────────────────────────────  ──────────────────────────────
#    常见起点          官方 yolov5s.pt                     你之前训练好的 best.pt
#   ────────────────  ──────────────────────────────────  ──────────────────────────────
#    学习率            大一些，约 0.01                     小一些，约 0.001
#   ────────────────  ──────────────────────────────────  ──────────────────────────────
#    训练风格          学得更猛，适合重新学很多通用特征    学得更稳，尽量保留旧模型能力
#   ────────────────  ──────────────────────────────────  ──────────────────────────────
#    默认冻结          默认不冻结                          默认冻结前 10 层
#   ────────────────  ──────────────────────────────────  ──────────────────────────────
#    全层训练怎么做    默认就是全层                        加 --freeze 0 就是全层