# 端云协同 — 出入库事件上报 + 标签扫描

> 对应赛题附加分 (c) 端云协同 与 (e) 包装标签信息读取。
> 端侧只做"检测 → 裁剪 → 上传"，标签识别和"标签↔物品"配对都在后端完成。
> 严格对应仓库根目录《端侧返回数据格式.txt》与《不知道.md》。

## 为什么标签识别在后端而不在端侧

RV1106 只有约 0.5TOPS，连 YOLOv5s 都要 int8 量化压榨，跑不动 OCR/VLM。所以分工：

- **端侧(LuckFox)**：检测事件、拍照、裁剪、HTTP 上传。
- **后端(同学的服务)**：跑 OCR/VLM 读标签；用"同设备 + 时间窗"把标签配对到物品。

`tools/mock_cloud_ocr.py` 是后端服务的**临时替身**，方便后端没写好前先联调。后端写
好真服务后，端侧一行都不用改，把地址指过去即可。

## 两个独立上报端点

### 1) 出入库事件 → POST /events/item

session 检测到放入 / 取出 / 整理时各发一次：

```json
{
  "device_id": "luckfox",
  "timestamp": 1716466178000,
  "event_type": "ITEM_IN",
  "data": [{
    "local_track_id": 7,
    "category": "beverage_dairy",
    "confidence": 0.91,
    "bbox": [120, 45, 60, 60],
    "crop_image": "/9j/4AAQSk..."
  }]
}
```

- `event_type`：`ITEM_IN`(放入) / `ITEM_OUT`(取出) / `ITEM_MOVED`(挪位)。
- `local_track_id`：端侧库存稳定身份(item_id)，防抖去重 / 后端对号入座用。
- `bbox`：**[x, y, w, h]**（左上角 + 宽高），位置就是物品位置。
- ITEM_IN / ITEM_MOVED 物品还在画面 → 带 `crop_image`；ITEM_OUT 物已离场 →
  `crop_image` 为空字符串，但仍给出原位置 bbox。

### 2) 标签扫描 → POST /events/label_scan

用户把带标签那面朝镜头举着停一下，端侧检测到就发：

```json
{
  "device_id": "luckfox",
  "timestamp": 1716466205000,
  "ttl_seconds": 300,
  "label_image": "/9j/4AAQSk..."
}
```

- `label_image`：整张标签照片(base64 JPEG)，交给后端 OCR/VLM 读品牌/保质期。
- `ttl_seconds`：标签有效期，后端时间窗配对用（默认 300 秒 = 5 分钟）。
- **没有 bbox**：扫标签时物品还没放进冰箱，位置未知。位置由随后那次 ITEM_IN 提供。

## "扫标签"动作怎么触发（收银机模式）

端侧不需要用户按任何键，靠**视觉停留检测**自动触发（`label_scan.{h,cc}`）：

1. 画面里**没有手**（避免手拿着晃动误触）
2. 有一个非手物体，中心落在画面**中央扫描区**内（默认中间 50% 区域）
3. 该物体连续 `LABELSCAN_HOLD_FRAMES` 帧（默认 8 帧 ≈ 0.8 秒）**基本不动**
4. 不在冷却期（触发后 `LABELSCAN_COOLDOWN_FRAMES` 帧内不重复触发，默认 30 帧 ≈ 3 秒）

满足后裁下该物体区域，发 `/events/label_scan`。参数都在 `fridge_config.h` 顶部，
可按现场调。

## 标签和物品怎么配对（后端做，端侧不管）

按《不知道.md》：后端收到 `label_scan` 写入 `pending_labels`(5 分钟有效)；收到
`ITEM_IN` 时，找"最近、未消费、未过期"的那条标签挂上去。所以操作顺序就是配对
顺序——引导用户 **扫一个 → 放一个** 即可正确对齐。

端侧完全不参与关联，只保证"用户扫标签后 5 分钟内发 ITEM_IN"。

## 整体数据流

```
用户动作            端侧                                   后端
──────────────────────────────────────────────────────────────────────
1. 标签朝镜头停留 → label_scan 检测 → 裁图 → POST /events/label_scan
                                                          → 存 pending_labels
                                                            + OCR 读标签
2. 把物品放进去   → session 判定 IN → 裁图 → POST /events/item (ITEM_IN)
                                                          → 时间窗配对最近标签
                                                          → inventory 既有 bbox
                                                            又有 label_data
3. 关门           → to_json 全量库存 → POST (DOOR_CLOSE，可选)
```

所有上传走**独立后台线程**，推理线程只把裁好的 JPEG + 元数据丢进队列就立即返回，
NPU 推理帧率完全不受网络延迟影响。

## 涉及的文件

| 文件 | 改动 |
|---|---|
| `include/cloud_uploader.h` / `src/cloud_uploader.cc` | 新增。异步上传器(POSIX socket HTTP + base64)，两个端点 |
| `include/label_scan.h` / `src/label_scan.cc` | 新增。扫标签停留检测（收银机模式） |
| `include/fridge_config.h` | 新增 `coarse_category()`、`has_label()`、云端地址 + 扫描参数 |
| `include/inventory.h` / `src/inventory.cc` | `to_json` 改用 [x,y,w,h] + `local_track_id` + 粗类/细类（标签数据在后端，端侧不存） |
| `include/session.h` / `src/session.cc` | `SettlementResult.events` 输出 IN/OUT/MOVED |
| `src/main.cc` | 事件裁图上传 + 扫标签检测上传；启停 uploader |
| `tools/mock_cloud_ocr.py` | 本地 mock 后端（两个端点 + 时间窗配对演示） |

## 粗粒度分类

`fridge_config.h::coarse_category()` 把 43 个细类映射到 5 个粗类：
`fruit_veg` / `meat_seafood` / `beverage_dairy` / `packaged_food` / `interference`。
对外 JSON 的 `category` 用粗类。若后端想要 `fruit/drink/meat` 这种命名，改这个
函数的返回字符串一处即可。

## 配置云端地址

默认值在 `fridge_config.h`：
```cpp
constexpr const char* CLOUD_HOST       = "192.168.168.1";
constexpr int         CLOUD_PORT       = 8000;
constexpr const char* CLOUD_ITEM_PATH  = "/events/item";
constexpr const char* CLOUD_LABEL_PATH = "/events/label_scan";
constexpr const char* CLOUD_DEVICE_ID  = "luckfox";
```
现场不想重新编译时，用环境变量覆盖（在板子上 `export` 后再运行程序）：
```sh
export FRIDGE_CLOUD_HOST=192.168.168.50   # 跑 mock / 真实后端那台机器的 IP
export FRIDGE_CLOUD_PORT=8000
export FRIDGE_ITEM_PATH=/events/item
export FRIDGE_LABEL_PATH=/events/label_scan
export FRIDGE_DEVICE_ID=luckfox
```

## 本地联调步骤

1. 在 PC / WSL（与板子同网段）启动 mock 后端：
   ```sh
   python3 tools/mock_cloud_ocr.py --port 8000
   ```
   它监听 `/events/item` 和 `/events/label_scan`，把上传的图存到 `tools/received/`，
   对 label_scan 模拟 OCR，并在 ITEM_IN 时演示时间窗配对。

2. 交叉编译并上传到板子（照常走 `build.sh` → `scp`）。

3. 板子上设好 `FRIDGE_CLOUD_HOST` 指向 mock 的 IP，运行程序。

4. 演示流程：先把商品标签那面朝摄像头举着停约 1 秒 → 再把商品放进取放区。终端会打印：
   ```
   [LABEL] 检测到扫标签动作: milk_box (置信度 88%) 区域=(...)
   [CLOUD] 标签扫描上报成功 (ttl=300s, NNNN B jpeg)
   [EVENT] 放入: item#3 milk_box ...
   [CLOUD] 事件 ITEM_IN (item#3) 上报成功
   ```
   mock 端会打印 `★ 配对标签 -> {...}`，并在 `tools/received/` 留下两张图供检查。

## 接真实后端

把 `mock_cloud_ocr.py` 里 `fake_ocr()` 换成调用真正的 OCR/VLM：
- 通用大模型：GPT-4V、Qwen-VL、GLM-4V 等，prompt 让它返回品名/品牌/保质期
- 纯文字标签：PaddleOCR / RapidOCR 检测识别后规则抽取日期

`rknn_model_zoo/examples/PPOCR` 里有 PaddleOCR 的 RKNN 版本，若以后想把 OCR 也下
沉到更强的边缘盒子（不是 RV1106）可作参考。端侧不用改。

## 鲁棒性说明

- 队列上限 16，满了丢最旧任务，绝不撑爆内存。
- socket 收发各 5 秒超时，云端不可达时后台线程不会卡死，仅打印失败并继续。
- 网络断了完全不影响端侧识别/库存主流程（上报是"锦上添花"的旁路）。
