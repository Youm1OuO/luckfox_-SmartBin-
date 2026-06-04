cd /home/muyou/Projects/workspace/luckfox_demo/tools/yolov5
conda activate DL_env


python fridge_project/scripts/train.py stage-b \
    --data fridge_project/datasets/public_merged/public_merged.yaml \
    --epochs 100 \
    --batch-size 64 \
    --imgsz 640 \
    --workers 8 \
    --cache ram


python fridge_project/scripts/train.py stage-b \
    --data fridge_project/datasets/public_merged/public_merged.yaml \
    --epochs 100 \
    --batch-size 64 \      ← 翻倍,显存够
    --imgsz 640 \
    --workers 8 \          ← 加倍,CPU 还有空闲
    --cache ram            ← 关键!图全加载到内存,后续 epoch 不再读盘





python fridge_project/scripts/train.py stage-b \
    --data fridge_project/datasets/public_merged/public_merged.yaml \
    --epochs 200 \
    --batch-size 32 \
    --imgsz 640 \
    --workers 8 \
    --cache ram \
    --name stage_b_640



python fridge_project/scripts/train.py stage-b \
    --data fridge_project/datasets/public_merged/public_merged.yaml \
    --hyp  fridge_project/configs/hyp_stage_b.yaml \
    --weights runs/train/stage_b_640_aug/weights/best.pt \
    --freeze 0 \
    --epochs 300 \
    --batch-size 32 \
    --imgsz 640 \
    --workers 8 \
    --cache ram \
    --name stage_b_640_v1


python fridge_project/scripts/train.py stage-b \
    --data fridge_project/datasets/public_merged/public_merged.yaml \
    --hyp  fridge_project/configs/hyp_stage_b.yaml \
    --weights runs/train/stage_b_640/weights/best.pt \
    --freeze 0 \
    --epochs 200 \
    --batch-size 32 \
    --imgsz 640 \
    --workers 8 \
    --cache ram \
    --name stage_b_640_v3



python fridge_project/scripts/train.py stage-c \
    --data    fridge_project/datasets/public_merged/public_merged.yaml \
    --weights runs/train/stage_b_640_v3_b_clean/weights/best.pt \                                                               
    --epochs  600 \
    --batch-size 32 \
    --imgsz   640 \
    --workers 8 \
    --cache   ram \
    --name    hand_plus_v1 \
    --freeze  0
# -------------------------------------------------------------------------------


cd /home/muyou/Projects/workspace/luckfox_demo/tools/yolov5


# .pt → .ONNX
python export.py \
    --rknpu \
    --weight runs/train/stage_b3/weights/best.pt \
    --imgsz 320



python export.py \
    --rknpu \
    --weight runs/train/stage_b/weights/best.pt \   ← 权重文件路径
    --imgsz 320    ← 导出尺寸



python export.py \
    --rknpu \
    --weight runs/train/stage_b_640_v3_b_clean/weights/best.pt \
    --imgsz 640


python export.py \
    --rknpu \
    --weight runs/train/hand_plus_v1/weights/best.pt \
    --imgsz 640


---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------




│          阶段         │ freeze │                                              理由                                               │
├───────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 第一阶段（公开数据集）  │ 0      │ 模型已经学过 43 类，但手的数据是新的，需要全部层都更新                                          │
├───────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 第二阶段（冰箱照片）    │ 10     │ 冰箱图片量少，全部放开容易把之前学好的特征破坏掉；冻结 backbone，只训练分类头，保留已学好的特征 