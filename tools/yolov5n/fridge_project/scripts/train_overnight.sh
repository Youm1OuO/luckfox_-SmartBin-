#!/bin/bash
# =============================================================================
# 过夜对比训练 — 串行跑两组实验,睡前启动起床看结果
# =============================================================================
# 用法 (在 yolov5 根目录下):
#     bash fridge_project/scripts/train_overnight.sh
#
# 想后台跑(关掉终端也不停):
#     nohup bash fridge_project/scripts/train_overnight.sh > overnight.log 2>&1 &
#     disown
#
# 两组对照:
#   实验 A: 从 stage_b_640_aug 接力(你之前的产物)
#   实验 B: 从 yolov5s.pt 重训(业界干净起点)
# =============================================================================

# 切到 yolov5 根目录(脚本所在目录的上两层)
cd "$(dirname "$0")/../.." || exit 1

# 激活 conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_env

START_TS=$(date +%s)


# =============================================================================
# 实验 A — 从 stage_b_640_aug 接力
# =============================================================================
echo ""
echo "============================================================"
echo "  实验 A: 从 stage_b_640_aug 接力"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

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
    --name stage_b_640_v3_a_aug


# =============================================================================
# 实验 B — 从 yolov5s.pt 重训
# =============================================================================
echo ""
echo "============================================================"
echo "  实验 B: 从 yolov5s.pt 重训"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python fridge_project/scripts/train.py stage-b \
    --data fridge_project/datasets/public_merged/public_merged.yaml \
    --hyp  fridge_project/configs/hyp_stage_b.yaml \
    --weights yolov5s.pt \
    --freeze 0 \
    --epochs 300 \
    --batch-size 32 \
    --imgsz 640 \
    --workers 8 \
    --cache ram \
    --name stage_b_640_v3_b_clean


# =============================================================================
# 汇总 — 抽两次的最终 mAP 对比
# =============================================================================
END_TS=$(date +%s)
TOTAL_MIN=$(( (END_TS - START_TS) / 60 ))

echo ""
echo "============================================================"
echo "  汇总报告 (总耗时 ${TOTAL_MIN} 分钟)"
echo "============================================================"

# results.csv 列序: epoch, train/box_loss, train/obj_loss, train/cls_loss,
#                   precision, recall, mAP_0.5, mAP_0.5:0.95, val/...
report_metrics() {
    local name=$1
    local csv="runs/train/${name}/results.csv"
    if [ ! -f "$csv" ]; then
        printf "  %-30s  无 results.csv (训练失败?)\n" "$name"
        return
    fi
    awk -F',' -v n="$name" '
        NR>1 {
            gsub(/ /,"",$1); gsub(/ /,"",$7); gsub(/ /,"",$8);
            last_ep=$1; last_50=$7; last_5095=$8;
            if ($7+0 > best_50) { best_50=$7; best_ep=$1 }
        }
        END {
            printf "  %-30s  last(ep%s): mAP@.5=%s mAP@.5:.95=%s   best(ep%s): mAP@.5=%s\n",
                n, last_ep, last_50, last_5095, best_ep, best_50
        }
    ' "$csv"
}

report_metrics "stage_b_640_v3_a_aug"
report_metrics "stage_b_640_v3_b_clean"

echo ""
echo "权重文件:"
echo "  A: runs/train/stage_b_640_v3_a_aug/weights/best.pt"
echo "  B: runs/train/stage_b_640_v3_b_clean/weights/best.pt"
echo "训练曲线:"
echo "  A: runs/train/stage_b_640_v3_a_aug/results.png"
echo "  B: runs/train/stage_b_640_v3_b_clean/results.png"
echo "============================================================"
