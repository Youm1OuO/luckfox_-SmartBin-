#!/bin/bash
python fridge_project/scripts/train.py stage-c \
    --data fridge_project/datasets/public_merged/public_merged.yaml \
    --weights yolov5s.pt \
    --epochs 600 \
    --batch-size 32 \
    --imgsz 640 \
    --workers 8 \
    --cache ram \
    --name hand_plus_v1 \
    --freeze 0



python fridge_project/scripts/train.py stage-c \
    --data fridge_project/datasets/public_merged/public_merged.yaml \
    --weights yolov5s.pt \
    --epochs 600 \
    --batch-size 32 \
    --imgsz 640 \
    --workers 8 \
    --cache ram \
    --name hand_plus_v2 \
    --freeze 0

#   - stage-b：默认用 hyp.scratch-low.yaml（mosaic 1.0），适合从预训练权重开始训                                                                         
#   - stage-c：默认用 hyp_finetune.yaml（mosaic 0.8），适合微调自己的模型



python3 fridge_project/scripts/train.py stage-c \
    --data dataset/data.yaml \
    --weights runs/train/hand_plus_v1/weights/best.pt \
    --epochs 500 \
    --batch-size 32 \
    --img 704 \
    --freeze 10 \
    --name fridge_model

------------------------------------------------------------------------------------


cd /home/muyou/Projects/workspace/luckfox_demo/tools/yolov5

python fridge_project/scripts/train.py stage-c \
    --data /home/muyou/Projects/workspace/luckfox_demo/tools/luckfox_pico_rkmpi_example/example/luckfox_pico_rtsp_yolov5/dataset/data.yaml \
    --weights runs/train/hand_plus_v1/weights/best.pt \
    --epochs 500 \
    --batch-size 32 \
    --imgsz 704 \
    --workers 8 \
    --cache ram \
    --name fridge_self_704_v1 \
    --freeze 10