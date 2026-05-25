# 冰箱食材识别 — 工程目录

本目录是比赛工程,**不修改** yolov5 仓库任何文件。
所有定制内容(类别定义、数据集脚本、训练脚本)集中在这里。

## 设计原则

1. **单一真源**:`configs/classes.yaml` 是类别定义的唯一来源。
   数据集合并、训练配置、推理后处理都从它读取。
2. **不污染原仓库**:训练通过 `subprocess` 调用上层 `train.py`,
   原仓库保持干净,可随时 `git pull` 同步上游。
3. **可复现**:每一阶段的产物(权重、日志)按阶段命名隔离。

## 三阶段训练流水线

```
COCO 预训练 (yolov5n.pt)              ← 官方权重,直接下载
        │
        ▼
[阶段 B] 公开数据集训练
        │  目标:让 backbone 学到"果蔬/包装/手"的视觉先验
        │  输入:多个公开数据集合并后的大数据(每类 200+ 张)
        │  产物:runs/train/stage_b/weights/best.pt
        ▼
[阶段 C] 自采冰箱数据微调
        │  目标:适配冰箱内部光照、视角、遮挡
        │  输入:自采数据 800-2000 张
        │  策略:冻结 backbone 前 10 层,小学习率
        │  产物:runs/train/stage_c/weights/best.pt  ← 最终部署模型
        ▼
导出 ONNX → RKNN(用 export.py)
```

## 使用流程

### Step 1 — 编辑类别表

打开 `configs/classes.yaml`,根据自己冰箱常见食材增删 `classes`
和 `aliases` 别名映射。

### Step 2 — 合并公开数据集

把下载好的数据集分别放好,然后运行(注意路径用绝对路径或相对工作区路径):

```cmd
python fridge_project\scripts\prepare_dataset.py ^
    --source-yolo  D:\datasets\roboflow_fridge ^
    --source-coco  D:\datasets\coco2017 ^
    --output       fridge_project\datasets\public_merged ^
    --val-ratio    0.1
```

支持的 `--source-*` 参数(可重复使用):
- `--source-yolo PATH`:已经是 YOLO 格式的目录(images/labels 子目录)
- `--source-coco PATH`:COCO JSON 标注格式
- `--source-voc  PATH`:Pascal VOC XML 标注格式

输出会得到 `images/{train,val}` + `labels/{train,val}` + 自动生成的 yaml。

### Step 3 — 阶段 B 训练(公开数据)

```cmd
python fridge_project\scripts\train.py stage-b ^
    --data fridge_project\datasets\public_merged\public_merged.yaml ^
    --epochs 100
```

### Step 4 — 自采数据准备

把 LabelImg / Roboflow 标注好的自采数据放成 YOLO 格式:

```
fridge_project/datasets/self_collected/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt
```

然后生成 yaml:

```cmd
python fridge_project\scripts\build_data_yaml.py ^
    --dataset-root fridge_project\datasets\self_collected ^
    --output       fridge_project\datasets\self_collected\self.yaml
```

### Step 5 — 阶段 C 微调

```cmd
python fridge_project\scripts\train.py stage-c ^
    --data    fridge_project\datasets\self_collected\self.yaml ^
    --weights runs\train\stage_b\weights\best.pt ^
    --epochs  50
```

### Step 6 — 导出 ONNX

```cmd
python export.py --weights runs\train\stage_c\weights\best.pt --include onnx --opset 12
```

## 关于粗粒度分类

`classes.yaml` 里的 `coarse_grained` 段定义了细类 → 粗类的映射。
这一映射**不参与 YOLO 训练**,而是在推理后处理:
检测出 `apple` 后查表得到粗类 `fruit_veg`,用于库存粗分类展示。
这样训练目标始终是细粒度,精度更高,粗粒度是"免费"的副产品。
