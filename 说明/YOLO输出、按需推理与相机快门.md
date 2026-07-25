# YOLO 输出、按需推理与相机快门

## 1. 当前 YOLOv5n 给业务代码的输出

模型原始输出有 3 个 NPU Tensor；`post_process()` 已完成解码、置信度筛选和 NMS。`main.cc` 实际拿到的是：

```cpp
object_detect_result_list od_results;
```

```cpp
od_results.count                 // 本帧最终保留的物品数，最多 128 个
od_results.results[i].box        // { left, top, right, bottom }
od_results.results[i].prop       // 置信度，float
od_results.results[i].cls_id     // 类别编号，int
```

这些 `box` 最初是 **704 × 704 letterbox 输入图**中的坐标；`main.cc` 会用 `mapCoordinates()` 还原到摄像头原图的 1280 × 720 坐标，随后转为：

```cpp
fridge::Detection { box, score, cls_id }
```

类别名称不在结果结构内，需要通过 `cls_id` 查标签表。

## 2. 能否按代码指定时机拍一帧并推理？

可以，完全不必只能固定时间执行。

当前代码本来就在循环中用：

```cpp
RK_MPI_VI_GetChnFrame(0, 0, &stViFrame, -1);
```

取到摄像头下一帧，再将它转换、缩放到 704 × 704、复制进 NPU 输入，最后调用：

```cpp
inference_yolov5n_model(&rknn_app_ctx, &od_results);
```

现在的规则是“门打开时每帧推理，门关闭时跳过”。以后可改成由任意代码条件触发，例如：检测到手、门刚打开、等待稳定快照、定时器到点、收到按键/网络命令时，才对刚取到的一帧调用 YOLO。

摄像头可以持续出流；“拍一张”在这里就是取当前的一帧并决定是否送入 YOLO，不需要真的保存为图片文件。

## 3. 快门速度能否控制？

可以控制，准确说是控制传感器的**曝光时间（电子快门）**。当前代码只启动 ISP：

```cpp
SAMPLE_COMM_ISP_Init(...);
SAMPLE_COMM_ISP_Run(0);
```

没有设置曝光参数，因此目前由自动曝光 AE 决定。

工程的 Rockchip AIQ 头文件已提供接口：

```cpp
rk_aiq_user_api2_ae_setExpSwAttr(...)   // 自动 / 手动曝光模式
rk_aiq_user_api2_ae_setLinExpAttr(...)  // 线性模式曝光参数
```

手动模式可设置：

```text
TimeValue       曝光时间，单位秒，例如 0.005 = 1/200 秒
GainValue       传感器增益
IspDGainValue   ISP 数字增益
```

但当前 `SAMPLE_COMM_ISP_*` 封装没有把 AIQ 上下文直接交给 `main.cc`，所以实现时需要给该封装增加一个“设置曝光”的函数，或取得 AIQ context 后调用上述接口；还需以实际摄像头传感器支持的曝光范围为准。

## 4. 实际取舍

```text
曝光时间短：手移动更清晰，但画面更暗，可能要提高增益（噪点更多）。
曝光时间长：画面更亮，但手和移动物品更容易拖影。
```

帧率与快门不是同一件事；工程虽有 `SAMPLE_COMM_ISP_SetFrameRate()`，但它当前也未调用。若要在“指定时刻拍清晰一帧”，更实用的方式是：保持自动曝光或预设较短曝光，等曝光参数生效数帧后，再取帧送 YOLO。
