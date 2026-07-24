# PLAYBOOK — 冰箱食材识别项目操作手册

> 这份文档记录从"复制项目到 WSL"开始的所有操作步骤、命令、决策记录。
> 换到 WSL 后照着这份文档做就行。

---

## 0. 项目背景速览

- **目标**:基于 YOLOv5s 训练冰箱食材识别模型,部署到 LuckFox(Rockchip NPU)
- **赛题**:智能冰箱食材识别与管理(检测 + 分类 + 库存管理)
- **类别数**:43 类(详见 `fridge_project/configs/classes.yaml`)
- **训练策略**:两阶段
  - 阶段 B:在公开数据集上训练(给 backbone 学习视觉先验)
  - 阶段 C:在自采冰箱数据上微调(适配真实场景)
- **当前位置**:Windows 上的 yolov5 仓库已经准备好,即将整体复制到 WSL 训练

---

## 1. 项目结构

```
yolov5/                                  ← yolov5 官方仓库,我们没动一行原代码
├── train.py / detect.py / export.py     ← 官方训练/推理/导出脚本
├── models/yolov5s.yaml                  ← 我们用的模型结构(s 号)
├── yolov5s.pt                           ← 官方 COCO 预训练权重
├── datasets/                            ← 4 个 Roboflow 公开数据集解压后放这
│   ├── FOOD-INGREDIENTS dataset.yolov5pytorch/
│   ├── Beverage Containers.yolov5pytorch/
│   ├── Hand Detection.yolov5pytorch/
│   └── hand.yolov5pytorch/
└── fridge_project/                      ← 我们的全部定制内容
    ├── PLAYBOOK.md                      ← 本文档
    ├── README.md                        ← 工程说明
    ├── configs/
    │   ├── classes.yaml                 ← 单一真源:43 类 + 别名 + 粗分类
    │   └── hyp_finetune.yaml            ← 阶段 C 微调超参
    ├── scripts/
    │   ├── prepare_dataset.py           ← 数据集合并器(YOLO/COCO/VOC)
    │   ├── build_data_yaml.py           ← 自采数据 yaml 生成器
    │   ├── train.py                     ← 两阶段训练调度器
    │   ├── merge_public_datasets.bat    ← Windows 一键合并(WSL 不用)
    │   └── merge_public_datasets.sh     ← WSL/Linux 一键合并 ← 用这个
    └── datasets/                        ← 合并脚本的输出目录(自动创建)
        └── public_merged/               ← 跑完 merge 脚本会出现
```

---

## 2. 完整流程清单

按顺序做完这 8 步,就能拿到 `best.pt`:

```
[1] 复制 yolov5 整个文件夹到 WSL
[2] 在 WSL 用 Kiro 打开新位置
[3] 激活 conda 环境,确认依赖完整
[4] 跑数据集合并脚本
[5] 检查合并输出(类别分布、丢弃报告)
[6] 跑阶段 B 训练
[7] 训练完查看结果
[8] 把训练产物同步回 Windows
```

---

## 3. Step 1 — 复制项目到 WSL

### 3.1 打开 WSL 终端

Windows 任务栏搜 "WSL" 或 "Ubuntu",启动。

### 3.2 一行命令复制

```bash
cp -r /mnt/c/Users/Youm1OuO/Desktop/luckfox_demo/tools/yolov5 ~/yolov5_fridge
```

> 这会把整个目录原封不动复制到 WSL home,改名 `yolov5_fridge` 避免和你之前的项目重名。
> 时间预估:5–15 分钟(数据集占大头)。

### 3.3 验证

```bash
ls ~/yolov5_fridge
```

应该看到 `train.py`、`detect.py`、`fridge_project/`、`datasets/` 等。

---

## 4. Step 2 — 用 Kiro 打开

```bash
cd ~/yolov5_fridge
kiro .
# 如果 kiro 命令不存在,试 code .
```

Kiro 会启动并连接到 WSL,左下角会显示 "WSL: Ubuntu"(或类似)。
之后所有终端、Python、脚本运行都在 WSL 内。

---

## 5. Step 3 — 激活环境,验证依赖

### 5.1 激活你的 conda 环境

```bash
conda env list           # 看有哪些环境
conda activate <环境名>   # 激活之前训过 yolov5 的那个
```

### 5.2 验证关键依赖

```bash
python --version
python -c "import torch, cv2, PIL, yaml; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

期望输出:
- Python **3.8 或更高**(yolov5 要求)
- 形如 `torch: 2.1.0 cuda: True`

如果某个 import 报错,装一下:
```bash
pip install pillow opencv-python pyyaml
```

如果 torch 是 CPU 版本(`cuda: False`)但你有 N 卡:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

如果完全没装过 yolov5 依赖:
```bash
pip install -r requirements.txt
```

---

## 6. Step 4 — 跑合并脚本

### 6.1 一键运行

```bash
cd ~/yolov5_fridge
bash fridge_project/scripts/merge_public_datasets.sh
```

预计 5–15 分钟。完成后输出在 `fridge_project/datasets/public_merged/`。

### 6.2 命令做了什么

调用 `prepare_dataset.py`,做这些事:
1. 读 `configs/classes.yaml` 拿到 43 类 + 别名映射表
2. 扫 4 个数据集所有图片和标注
3. 把每个 bbox 的标签名通过 alias 翻译成统一类别 ID
4. 命中的 bbox 收下,未命中的丢弃(整张图至少有 1 个命中才被收入)
5. 按 9:1 切分 train/val,复制图片、写新标签
6. 生成 `public_merged.yaml` 给训练用

### 6.3 跑完检查输出

终端最后会打印:

**A. 每个数据集的合并统计**
```
[yolo] images: 4198/4198 kept | boxes: 6234/8901 kept (dropped 2667)
       top dropped labels: Akabare Khursani(420), Gundruk(312), ...
```

**B. 总图片数 + train/val 切分**
```
总计 12500 张 → train 11250, val 1250
```

**C. 类别分布**(43 行,每类多少 box)
```
  0  apple                    312
  1  banana                   245
  ...
 33  hand                     8654
 34  mushroom                 0     ← ⚠️ 注意:数量为 0 的类
```

### 6.4 检查清单

- [ ] 总图片数 > 8000(太少说明 alias 几乎全没命中)
- [ ] hand 类有 5000+ box(hand 数据集大头是它)
- [ ] 主要食材(apple, banana, tomato...)各有 100+ box
- [ ] 没有任何类别的数量是 0(0 表示 alias 写错了,需要排查)

---

## 7. Step 5 — 跑阶段 B 训练

### 7.1 命令

```bash
python fridge_project/scripts/train.py stage-b \
    --data fridge_project/datasets/public_merged/public_merged.yaml \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 640
```

### 7.2 参数说明

| 参数 | 含义 | 调整建议 |
|------|------|----------|
| `--data` | 数据集 yaml | 不动 |
| `--epochs 100` | 训练轮数 | 100 是经验值,数据多可以减到 80,数据少要 150+ |
| `--batch-size 32` | 批大小 | 8GB 显存通常 32 OK,报 OOM 降到 16 或 8 |
| `--imgsz 640` | 输入分辨率 | 640 训练精度高,部署用 320 |

### 7.3 启动后看到的关键信息

```
Transferred 343/355 items from yolov5s.pt    ← 迁移学习成功(355 个权重里 343 个加载)
Optimizer groups: 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
                  ↑ 343 = backbone + neck 的权重全部迁移
                  ↑ 12 个不迁移的 = 最后一层 Detect head(类别数从 80 改成 43,需重训)

Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
0/99     6.8G      0.084      0.052      0.028     156             640
1/99     ...
```

**全是正常,不是错误。** 如果看到 `Transferred 0/355` 才是有问题(权重没加载)。

### 7.4 时间预估

| 显卡 | 每 epoch | 100 epoch 总计 |
|------|----------|----------------|
| RTX 4090 | ~30s | ~1 小时 |
| RTX 3060 | ~2min | ~3.5 小时 |
| GTX 1660 | ~5min | ~8 小时 |
| CPU | 不要尝试 | 几天 |

### 7.5 训练中怎么看进度

每个 epoch 结束会打印 mAP@0.5、mAP@0.5:0.95、precision、recall。
也可以另开一个终端跑:

```bash
ls -la runs/train/stage_b/
# 看 weights/ 目录里 last.pt 在更新就说明训练在跑
```

### 7.6 中途想停

`Ctrl+C` 即可,产物保留。下次想接着跑用 `--resume`:

```bash
python fridge_project/scripts/train.py stage-b --data ... --resume
```

但通常重跑成本不高,直接重来更简单。

---

## 8. Step 6 — 训练完看结果

### 8.1 产物位置

```
runs/train/stage_b/
├── weights/
│   ├── best.pt          ← ★ 验证集 mAP 最高的权重(用这个)
│   └── last.pt          ← 最后一个 epoch 的权重
├── results.png          ← 训练曲线
├── confusion_matrix.png ← 混淆矩阵
├── PR_curve.png         ← PR 曲线
├── F1_curve.png
├── val_batch0_pred.jpg  ← 验证集预测可视化
└── train_batch0.jpg     ← 训练样本可视化
```

### 8.2 健康判断标准

打开 `results.png` 看曲线:

- **box_loss / obj_loss / cls_loss 都应该单调下降**(允许后期波动)
- **mAP@0.5 应该上升到 0.5+ 才算合格**(40 类的检测,这个数字算够用)
- **precision 和 recall 应该都在 0.6+**

如果 mAP 卡在 0.2 以下,可能问题:
- 数据集质量差(标签噪声大)
- 类别极度不平衡(hand 占了 80%,其他类没几个)
- 学习率太大,降到 0.005 重训

### 8.3 简单测试推理

挑一张冰箱图片测试:

```bash
python detect.py --weights runs/train/stage_b/weights/best.pt \
    --source data/images/bus.jpg \
    --imgsz 640 \
    --conf 0.25
```

(bus.jpg 只是 yolov5 自带的示例图;你也可以用任何冰箱图)

---

## 9. Step 7 — 同步回 Windows

训完之后,如果要让 Windows 那边的同学看到产物:

```bash
mkdir -p /mnt/c/Users/Youm1OuO/Desktop/luckfox_demo/tools/yolov5/runs
cp -r ~/yolov5_fridge/runs/* /mnt/c/Users/Youm1OuO/Desktop/luckfox_demo/tools/yolov5/runs/
```

---

## 10. 阶段 C(自采数据微调,以后做)

阶段 B 训完拿到 `best.pt` 之后,会自采几百张冰箱实拍图,然后:

```bash
# 1. 把自采数据按 YOLO 格式放好
fridge_project/datasets/self_collected/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt

# 2. 生成 yaml
python fridge_project/scripts/build_data_yaml.py \
    --dataset-root fridge_project/datasets/self_collected \
    --output       fridge_project/datasets/self_collected/self.yaml

# 3. 微调
python fridge_project/scripts/train.py stage-c \
    --data    fridge_project/datasets/self_collected/self.yaml \
    --weights runs/train/stage_b/weights/best.pt \
    --epochs  50
```

阶段 C 默认:
- 学习率 0.001(阶段 B 的 1/10)
- 冻结 backbone 前 10 层(整个 backbone)
- 数据增强降低 mosaic 强度,提高 flipud(冰箱顶视角)

---

## 11. 阶段 D(导出 ONNX → RKNN,以后做)

阶段 C 训完拿到最终 `best.pt`,导出:

```bash
python export.py --weights runs/train/stage_c/weights/best.pt \
    --include onnx --opset 12 --imgsz 320
```

> 注意:**部署用 320,不是 640**!训练时 640 提精度,部署时降到 320 求帧率。

得到 `runs/train/stage_c/weights/best.onnx`,这个文件交给 LuckFox 部署链路:
- 用 rknn-toolkit2(瑞芯微官方,**这步要 Docker**)转 .rknn
- 拷贝 .rknn 到 LuckFox 设备
- 加载推理

---

## 12. 重要决策记录(避免遗忘)

### 12.1 类别表为什么是 43 类

- 必备 17 类蔬果(包括牛油果、木瓜、哈密瓜等)
- 6 类肉蛋生鲜(包装/未包装)
- 6 类饮料乳品
- 6 类包装食品(袋装/盒装/罐头/玻璃罐/保鲜膜/保鲜盒)
- 1 类干扰(hand)
- 7 类后期补充(蘑菇/南瓜/大蒜/姜/白萝卜/红薯/核桃/香菜/秋葵)

总数 17+3+6+6+6+1+9 = 后续追加后是 **43**(细节见 classes.yaml)。

### 12.2 为什么不用 yolov5n

LuckFox NPU 0.5–1 TOPS,理论上 yolov5n 更适合。但用户已确认用 yolov5s,
原因可能是 yolov5n 在某些场景容量不够。

最坏情况:阶段 C 训完发现部署帧率太低,可以再训一份 yolov5n 版,
只需把 `train.py` 里的 `--cfg models/yolov5s.yaml` 改成 `models/yolov5n.yaml`。

### 12.3 为什么阶段 C 不传 --cfg

yolov5 的 train.py 有个特性:只给 `--weights` 不给 `--cfg`,
它会从权重文件读模型结构,然后**根据数据 yaml 的 nc 自动重建检测头**。
这正是迁移学习的标准姿势。

### 12.4 为什么 --freeze 10

yolov5s backbone 共 10 个 module(yolov5s.yaml 里 backbone 的 0~9 行)。
`--freeze 10` = 冻结整个 backbone,只训 head。
自采数据 < 2000 张时这能防过拟合。
如果欠拟合可以改 `--freeze 4`(只冻前 4 层)。

---

## 13. 常见问题排查

### 13.1 合并脚本报 "找不到 data.yaml"

数据集解压时套了两层目录。打开 `datasets/<某数据集>/`,
如果直接看到的是另一个文件夹而不是 `data.yaml`,把内层的内容剪切到外层。

### 13.2 训练报 OOM (out of memory)

降 batch-size:`--batch-size 16` 或 `--batch-size 8`。

### 13.3 训练报 "DataLoader worker exited unexpectedly"

WSL 下偶尔出现,加 `--workers 0` 关闭多进程加载即可。
速度会慢一点但稳定。

### 13.4 mAP 一直是 0

检查 `public_merged.yaml` 里 `nc:` 是不是 43,
和 `names:` 列表长度一致。

### 13.5 训练时 GPU 使用率为 0

Torch 是 CPU 版,看 5.2 节重装 GPU 版 PyTorch。

### 13.6 想看训练实时曲线

启动训练后另开终端:
```bash
tensorboard --logdir runs/train --port 6006
```
浏览器打开 http://localhost:6006

---

## 14. 参考命令速查

```bash
# 复制项目到 WSL
cp -r /mnt/c/Users/Youm1OuO/Desktop/luckfox_demo/tools/yolov5 ~/yolov5_fridge

# 进项目并打开 Kiro
cd ~/yolov5_fridge && kiro .

# 激活环境(改成你的环境名)
conda activate <env_name>

# 验证依赖
python -c "import torch, cv2, PIL, yaml; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# 合并数据集
bash fridge_project/scripts/merge_public_datasets.sh

# 阶段 B 训练
python fridge_project/scripts/train.py stage-b \
    --data fridge_project/datasets/public_merged/public_merged.yaml \
    --epochs 100 --batch-size 32 --imgsz 640

# 看训练产物
ls runs/train/stage_b/weights/

# 单图测试推理
python detect.py --weights runs/train/stage_b/weights/best.pt \
    --source data/images/bus.jpg --imgsz 640 --conf 0.25

# 同步产物回 Windows
mkdir -p /mnt/c/Users/Youm1OuO/Desktop/luckfox_demo/tools/yolov5/runs
cp -r ~/yolov5_fridge/runs/* /mnt/c/Users/Youm1OuO/Desktop/luckfox_demo/tools/yolov5/runs/
```
