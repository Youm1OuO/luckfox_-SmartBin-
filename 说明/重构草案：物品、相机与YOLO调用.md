# 重构草案：物品、相机与 YOLO 调用

> 这是下一次重构的代码设计稿，不会直接修改现有代码。它以当前项目真实链路为基础：
> `RK_MPI_VI_GetChnFrame → YUV 转 BGR → letterbox(704×704) → inference_yolov5n_model → od_results`。
>
> 重点：每个库存物品都拥有自己的 `track`。`item_id` 是长期库存身份；`track` 是本次“稳定快照 A 到稳定快照 B”之间的临时过程记录，快照对比结束后重置。


## 1. 建议的物品与 Track 数据结构

建议以后用下面的内容替换/扩展当前的 `inventory.h`。这里故意**不再依赖旧 ByteTrack 的 `track_id`**：因为 `ItemTrack` 已经属于某个 `InventoryItem`，其拥有者就是 `item_id`，不需要再额外产生一套容易混淆的身份编号。

```cpp
// inventory.h（重构草案）
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "geometry.h"     // 使用项目已有 fridge::BBox

namespace fridge {

// 库存状态：VISIBLE / OCCLUDED 都表示物品仍在冰箱内。
enum class ItemStatus {
    VISIBLE,
    OCCLUDED,
    OUT,
};

// 路径中每一个点来自哪里。
enum class TrackPointSource {
    YOLO,        // YOLO 直接看到了物品
    HAND_PROXY,  // 物品被手完全遮挡，用手的位移推算出的物品位置
};

// 一条路径点。box 永远表示“物品估计框”，不是手的原始框。
// 这样 HAND_PROXY 点才能和后来 YOLO 得到的物品框比较大小、位置。
struct TrackPoint {
    BBox box;    // BBox 里面是 x1、y1、x2、y2 格式 (左上角与右下角)
    int frame_id = -1;
    TrackPointSource source = TrackPointSource::YOLO;
};

// 每一个物品拥有一份 Track; 它只服务于当前操作周期。
struct ItemTrack {
    // 本次操作开始（稳定快照 A）时, 物品的位置。
    BBox start_box;

    // 当前最近的物品估计位置：正常时来自 YOLO; 完全遮挡时由手位移推算。
    BBox latest_box;

    // 若使用 HAND_PROXY，保存上一次手框, 供下一帧计算手移动了多少。
    BBox last_hand_box;
    bool has_last_hand_box = false;

    int operation_start_frame = -1;
    int last_seen_frame = -1;     // 最近一次 YOLO 直接看见物品的帧号
    bool is_hand_proxy = false;   // 当前物品是否正在由手代理移动

    // 从快照 A 到现在的有效路径; 本次快照对比完成后清空。
    std::vector<TrackPoint> path;

    // 新操作开始：用快照 A 的位置作为路径第一个点。
    void begin_operation(const BBox& initial_box, int frame_id) {
        start_box = initial_box;
        latest_box = initial_box;
        operation_start_frame = frame_id;
        last_seen_frame = frame_id;
        has_last_hand_box = false;
        is_hand_proxy = false;
        path.clear();
        path.push_back({initial_box, frame_id, TrackPointSource::YOLO});
    }

    // 记录一个新点。相邻点几乎没动时不重复保存，属于第一层剪枝。
    void append_point(const BBox& box, int frame_id, TrackPointSource source) {
        constexpr float kMinMovePx = 2.0f;
        if (!path.empty() &&
            center_distance(path.back().box, box) < kMinMovePx &&
            path.back().source == source) {
            return;
        }
        latest_box = box;
        path.push_back({box, frame_id, source});
    }

    // YOLO 又确认看到了这个物品：路径已经完成使命，压缩为一个新起点。
    // 这正是“匹配成功后路径变为 0/1 个点”的剪枝。
    void reset_after_reappear(const BBox& yolo_box, int frame_id) {
        latest_box = yolo_box;
        last_seen_frame = frame_id;
        is_hand_proxy = false;
        has_last_hand_box = false;
        path.clear();
        path.push_back({yolo_box, frame_id, TrackPointSource::YOLO});
    }

    // A 与 B 快照比较、库存已经更新后调用：下一段操作重新开始。
    void reset_after_snapshot(const BBox& new_anchor_box, int frame_id) {
        begin_operation(new_anchor_box, frame_id);
    }
};

// 库存中的“一个真实物品”。
struct InventoryItem {
    int item_id = -1;            // 长期库存身份：1、2、3……；不因移动而改变
    int cls_id = -1;             // YOLO 类别，例如 apple
    BBox anchor_box;             // 最近稳定快照确认的位置；快照对比成功后才更新
    BBox visible_box;            // 最近一次 YOLO 直接看到的框
    float score = 0.0f;          // 最近一次 YOLO 的置信度
    ItemStatus status = ItemStatus::VISIBLE;

    int created_frame = -1;
    int updated_frame = -1;
    long long created_time_ms = 0;
    long long out_time_ms = 0;

    ItemTrack track;             // 这个物品自己的临时路径/手代理状态
};

// 注意：每帧 YOLO 的结果还不是库存物品，先用这个结构承接。
struct DetectedItem {
    int cls_id = -1;
    BBox box;
    float score = 0.0f;
};

}  // namespace fridge
```

### 物品初始化：只能来自稳定快照，不应每帧都新建

```cpp
// stable_snapshot 中的一个检测，才可以变成一个新的库存物品。
fridge::InventoryItem make_inventory_item(
        int item_id,
        const fridge::DetectedItem& det,
        int frame_id,
        long long now_ms) {
    fridge::InventoryItem item;
    item.item_id = item_id;              // 例如第一件物品为 1，第二件为 2
    item.cls_id = det.cls_id;
    item.anchor_box = det.box;
    item.visible_box = det.box;
    item.score = det.score;
    item.status = fridge::ItemStatus::VISIBLE;
    item.created_frame = frame_id;
    item.updated_frame = frame_id;
    item.created_time_ms = now_ms;
    item.track.begin_operation(det.box, frame_id);
    return item;
}

// 反例：不能一得到一帧 YOLO Detection 就调用 make_inventory_item()。
// 原因：同一个苹果连续被识别 10 帧，不能变成 10 个库存苹果。
// 新物品只应在：首次稳定快照初始化，或稳定快照对比确认“真正放入”后创建。
```

## 2. 同类 Detection 与物品完整路径的匹配

这段就是计划中的核心比较：

```text
Detection
  × 同 cls_id 的库存物品
  × 该物品本次操作周期的全部 path 点
```

```cpp
// association.h（重构草案）
#pragma once

#include <limits>
#include "inventory.h"

namespace fridge {

// 面积、宽高比要相近；它是很便宜的第一层剪枝。
inline bool similar_object_shape(const BBox& a, const BBox& b) {
    const float area_a = a.area();
    const float area_b = b.area();
    if (area_a <= 1.0f || area_b <= 1.0f || a.h() <= 1.0f || b.h() <= 1.0f) {
        return false;
    }

    const float area_ratio = std::min(area_a, area_b) / std::max(area_a, area_b);
    const float aspect_a = a.w() / a.h();
    const float aspect_b = b.w() / b.h();
    const float aspect_ratio = std::min(aspect_a, aspect_b) / std::max(aspect_a, aspect_b);

    return area_ratio >= 0.55f && aspect_ratio >= 0.55f;
}

// Detection 是否接近某一个路径点。IoU 和中心距离二选一满足即可。
inline float path_point_match_score(const BBox& det_box, const BBox& point_box) {
    if (!similar_object_shape(det_box, point_box)) return -1.0f;

    const float overlap = iou(det_box, point_box);
    const float nearby = normalized_nearby_distance(det_box, point_box);
    if (overlap < 0.25f && nearby > 0.80f) return -1.0f;

    // 仅用于在多个候选中选最佳，不是“外观 ReID 分数”。
    const float distance_score = std::max(0.0f, 1.0f - nearby / 0.80f);
    return std::max(overlap, distance_score);
}

struct ItemMatch {
    int item_id = -1;
    float best_score = -1.0f;
    int matched_path_index = -1;
};

// 从最新路径点向旧路径点扫描：通常物品重新出现的位置更靠近路径末端，
// 所以能更早找到正确结果。这里仍然保留“完整路径可参与比较”的能力。
inline ItemMatch find_best_item_for_detection(
        const DetectedItem& det,
        const std::map<int, InventoryItem>& inventory) {
    ItemMatch best;
    float second_best = -1.0f;

    for (const auto& pair : inventory) {
        const InventoryItem& item = pair.second;

        if (item.status == ItemStatus::OUT) continue;
        if (item.cls_id != det.cls_id) continue;       // 最重要的类别剪枝
        if (!similar_object_shape(det.box, item.track.latest_box)) continue;

        // 同一个 item 的多个 path 点不能互相算“竞争者”。
        // 先只找这个 item 的最佳路径点，再与其他 item 比较唯一性。
        ItemMatch best_for_this_item;
        best_for_this_item.item_id = item.item_id;
        for (int i = static_cast<int>(item.track.path.size()) - 1; i >= 0; --i) {
            const float score = path_point_match_score(det.box, item.track.path[i].box);
            if (score < 0.0f) continue;

            if (score > best_for_this_item.best_score) {
                best_for_this_item.best_score = score;
                best_for_this_item.matched_path_index = i;
            }
        }

        if (best_for_this_item.best_score > best.best_score) {
            second_best = best.best_score;       // 这里才一定是另一个 item 的分数
            best = best_for_this_item;
        } else if (best_for_this_item.best_score > second_best) {
            second_best = best_for_this_item.best_score;
        }
    }

    // 多个同类物品分数太接近时，宁可暂不认领，交给后面的稳定快照裁决。
    constexpr float kAcceptScore = 0.65f;
    constexpr float kUniqueMargin = 0.10f;
    if (best.best_score < kAcceptScore ||
        (second_best >= 0.0f && best.best_score - second_best < kUniqueMargin)) {
        return {};  // item_id = -1：本帧不强行绑定
    }
    return best;
}

// 已将 Detection 认领给 item 后调用。
// 注意：这里不更新 anchor_box；anchor_box 只能由两份稳定快照对比后更新。
inline void accept_yolo_detection(
        InventoryItem& item,
        const DetectedItem& det,
        int frame_id) {
    item.visible_box = det.box;
    item.score = det.score;
    item.status = ItemStatus::VISIBLE;
    item.updated_frame = frame_id;
    item.track.reset_after_reappear(det.box, frame_id);
}

}  // namespace fridge
```

### 完全被手挡住时，怎样把手的移动写入物品 Track

不能直接把“手的 box”塞进物品路径，否则手框通常比苹果框大，之后无法与苹果 Detection 比较。应保留物品自己的宽高，只借用手的**位移**：

```cpp
// hand_proxy.h（重构草案）
inline BBox translate_box(const BBox& box, float dx, float dy) {
    return BBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy);
}

// 调用前已经确认：item 确实被 hand_box 遮挡，且 YOLO 本帧没有检测到 item。
inline void update_item_by_hand_proxy(
        InventoryItem& item,
        const BBox& hand_box,
        int frame_id) {
    ItemTrack& track = item.track;

    if (!track.has_last_hand_box) {
        track.last_hand_box = hand_box;
        track.has_last_hand_box = true;
        track.is_hand_proxy = true;
        return;
    }

    const float dx = hand_box.cx() - track.last_hand_box.cx();
    const float dy = hand_box.cy() - track.last_hand_box.cy();

    // 推算出的还是“物品框”，大小与 latest_box 一样，只移动位置。
    const BBox estimated_object_box = translate_box(track.latest_box, dx, dy);
    track.append_point(estimated_object_box, frame_id, TrackPointSource::HAND_PROXY);
    track.last_hand_box = hand_box;
    track.is_hand_proxy = true;
    item.status = ItemStatus::OCCLUDED;
}
```

当物品在手旁重新被 YOLO 看见时，`find_best_item_for_detection()` 会扫描这段代理路径，命中后 `accept_yolo_detection()` 会把路径压缩成新位置的一个点；因此手之后继续离开冰箱，也不会再把物品继续带走。

## 3. 摄像头初始化、帧率与曝光（快门）设置

下面的初始化与当前 `main.cc`、`luckfox_mpi.cc` 一致。当前分辨率是 1280×720，VI 输出格式为 `RK_FMT_YUV420SP`。

```cpp
// camera_control.cc（重构草案）
#include <cstring>
#include <cstdio>

#include "luckfox_mpi.h"
#include "sample_comm_isp.h"

struct CameraConfig {
    int camera_id = 0;
    int vi_dev = 0;
    int vi_channel = 0;
    int width = 1280;
    int height = 720;

    // 0 表示沿用传感器/IQ 文件默认帧率；大于 0 时调用 SetFrameRate。
    int fps = 0;
    bool flip_180 = true;       // 当前 main.cc 已经对画面执行 cv::flip(..., -1)
};

bool init_camera(const CameraConfig& cfg) {
    const char* iq_dir = "/etc/iqfiles";
    const rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;

    if (SAMPLE_COMM_ISP_Init(cfg.camera_id, hdr_mode, RK_FALSE, iq_dir) != RK_SUCCESS) {
        return false;
    }
    if (SAMPLE_COMM_ISP_Run(cfg.camera_id) != RK_SUCCESS) {
        return false;
    }
    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        return false;
    }
    if (vi_dev_init() != 0 || vi_chn_init(cfg.vi_channel, cfg.width, cfg.height) != 0) {
        return false;
    }
    if (cfg.fps > 0 && SAMPLE_COMM_ISP_SetFrameRate(cfg.camera_id, cfg.fps) != RK_SUCCESS) {
        return false;
    }
    return true;
}
```

当前项目只调用了 `SAMPLE_COMM_ISP_Init/Run`，所以曝光由 ISP 自动曝光 AE 决定。SDK 具备手动电子快门 API，但现有 `SAMPLE_COMM_ISP_*` 封装没有把内部 `rk_aiq_sys_ctx_t*` 返回给 `main.cc`。重构时需要先给封装补一个“取得 AIQ context”或“设置曝光”的函数；下面是拿到 context 后实际要调用的代码：

```cpp
// 需要额外 include：
// #include "rkaiq/uAPI2/rk_aiq_user_api2_ae.h"

// exposure_time_us 例如 5000 = 5ms = 1/200 秒。
// analog_gain 例如 1.0f；光线不足时可提高，但会增加噪声。
bool set_manual_exposure(const rk_aiq_sys_ctx_t* aiq_ctx,
                         int exposure_time_us,
                         float analog_gain) {
    if (aiq_ctx == nullptr || exposure_time_us <= 0 || analog_gain < 1.0f) {
        return false;
    }

    // 先取当前设置再修改，避免用全零结构覆盖 ISP 的其他 AE 参数。
    Uapi_ExpSwAttrV2_t attr{};
    if (rk_aiq_user_api2_ae_getExpSwAttr(aiq_ctx, &attr) != XCAM_RETURN_NO_ERROR) {
        return false;
    }
    attr.sync.sync_mode = RK_AIQ_UAPI_MODE_SYNC;
    attr.AecOpType = RK_AIQ_OP_MODE_MANUAL;
    attr.stManual.LinearAE.ManualTimeEn = true;
    attr.stManual.LinearAE.ManualGainEn = true;
    attr.stManual.LinearAE.ManualIspDgainEn = false;
    attr.stManual.LinearAE.TimeValue = exposure_time_us / 1000000.0f; // SDK 单位：秒
    attr.stManual.LinearAE.GainValue = analog_gain;
    attr.stManual.LinearAE.IspDGainValue = 1.0f;

    return rk_aiq_user_api2_ae_setExpSwAttr(aiq_ctx, attr) == XCAM_RETURN_NO_ERROR;
}

bool set_auto_exposure(const rk_aiq_sys_ctx_t* aiq_ctx) {
    if (aiq_ctx == nullptr) return false;

    Uapi_ExpSwAttrV2_t attr{};
    if (rk_aiq_user_api2_ae_getExpSwAttr(aiq_ctx, &attr) != XCAM_RETURN_NO_ERROR) {
        return false;
    }
    attr.sync.sync_mode = RK_AIQ_UAPI_MODE_SYNC;
    attr.AecOpType = RK_AIQ_OP_MODE_AUTO;
    return rk_aiq_user_api2_ae_setExpSwAttr(aiq_ctx, attr) == XCAM_RETURN_NO_ERROR;
}
```

手移动时需要更清晰的画面，可尝试较短曝光（例如 1/100～1/250 秒）并配合适当补光；具体最短曝光和可用增益仍以实际摄像头传感器支持范围为准。

## 4. 取一帧摄像头图像，并按代码条件送进 YOLOv5n

`RK_MPI_VI_GetChnFrame()` 取的是摄像头的“下一帧”。因此可以持续取流，但只在任意业务条件满足时调用 YOLO；并不要求固定间隔推理。

```cpp
// camera_capture.cc（重构草案）
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// 成功后 bgr_frame 自己持有图像内存；VI 帧可以安全释放。
bool capture_one_bgr_frame(const CameraConfig& cfg, cv::Mat& bgr_frame) {
    VIDEO_FRAME_INFO_S vi_frame{};
    const RK_S32 ret = RK_MPI_VI_GetChnFrame(
            cfg.vi_dev, cfg.vi_channel, &vi_frame, -1);  // -1：等待下一帧
    if (ret != RK_SUCCESS) {
        std::printf("RK_MPI_VI_GetChnFrame failed: 0x%x\\n", ret);
        return false;
    }

    void* data = RK_MPI_MB_Handle2VirAddr(vi_frame.stVFrame.pMbBlk);
    if (data == nullptr) {
        RK_MPI_VI_ReleaseChnFrame(cfg.vi_dev, cfg.vi_channel, &vi_frame);
        return false;
    }

    // 当前 vi_chn_init() 明确设置的格式是 RK_FMT_YUV420SP。
    cv::Mat yuv420sp(cfg.height + cfg.height / 2, cfg.width, CV_8UC1, data);
    cv::cvtColor(yuv420sp, bgr_frame, cv::COLOR_YUV420sp2BGR);

    // cvtColor 输出到 bgr_frame 的自有内存后，才释放 VI 原始缓冲。
    RK_MPI_VI_ReleaseChnFrame(cfg.vi_dev, cfg.vi_channel, &vi_frame);

    if (cfg.flip_180) {
        cv::flip(bgr_frame, bgr_frame, -1);  // 与现有 main.cc 保持一致
    }
    return true;
}
```

## 5. BGR 图像输入 YOLOv5n，并转成 `DetectedItem`

```cpp
// yolo_adapter.cc（重构草案）
#include <algorithm>
#include <cstring>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "fridge_config.h"
#include "inventory.h"
#include "yolov5n.h"

struct LetterboxInfo {
    float scale = 1.0f;
    int left_padding = 0;
    int top_padding = 0;
};

// 原图按比例缩放，再补灰边到模型输入尺寸。当前模型实际是 704×704。
cv::Mat letterbox_for_yolo(const cv::Mat& bgr,
                            int model_width,
                            int model_height,
                            LetterboxInfo& info) {
    const float sx = static_cast<float>(model_width) / bgr.cols;
    const float sy = static_cast<float>(model_height) / bgr.rows;
    info.scale = std::min(sx, sy);

    const int resized_width = static_cast<int>(bgr.cols * info.scale);
    const int resized_height = static_cast<int>(bgr.rows * info.scale);
    info.left_padding = (model_width - resized_width) / 2;
    info.top_padding = (model_height - resized_height) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);
    cv::Mat input(model_height, model_width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(input(cv::Rect(info.left_padding, info.top_padding,
                                  resized_width, resized_height)));
    return input;
}

fridge::BBox model_box_to_camera_box(const image_rect_t& model_box,
                                     const LetterboxInfo& info,
                                     int camera_width,
                                     int camera_height) {
    auto clamp_float = [](float value, float lo, float hi) {
        return std::max(lo, std::min(value, hi));
    };
    auto map_x = [&](int x) {
        return clamp_float((x - info.left_padding) / info.scale,
                           0.0f, static_cast<float>(camera_width));
    };
    auto map_y = [&](int y) {
        return clamp_float((y - info.top_padding) / info.scale,
                           0.0f, static_cast<float>(camera_height));
    };
    return fridge::BBox(map_x(model_box.left), map_y(model_box.top),
                        map_x(model_box.right), map_y(model_box.bottom));
}

bool run_yolov5n_on_frame(rknn_app_context_t& app,
                          const cv::Mat& bgr_frame,
                          std::vector<fridge::DetectedItem>& detections) {
    detections.clear();
    if (bgr_frame.empty() || app.input_mems[0] == nullptr) return false;

    LetterboxInfo info;
    cv::Mat model_input = letterbox_for_yolo(
            bgr_frame, app.model_width, app.model_height, info);

    // 当前模型实测期望 BGR，所以 fridge::INFER_INPUT_BGR2RGB 默认是 false。
    if (fridge::INFER_INPUT_BGR2RGB) {
        cv::cvtColor(model_input, model_input, cv::COLOR_BGR2RGB);
    }
    if (!model_input.isContinuous()) model_input = model_input.clone();

    std::memcpy(app.input_mems[0]->virt_addr, model_input.data,
                static_cast<size_t>(app.model_width) * app.model_height * 3);

    object_detect_result_list raw_results{};
    if (inference_yolov5n_model(&app, &raw_results) < 0) {
        return false;
    }

    detections.reserve(raw_results.count);
    for (int i = 0; i < raw_results.count; ++i) {
        const object_detect_result& raw = raw_results.results[i];
        fridge::DetectedItem det;
        det.cls_id = raw.cls_id;
        det.score = raw.prop;
        det.box = model_box_to_camera_box(raw.box, info, bgr_frame.cols, bgr_frame.rows);
        detections.push_back(det);
    }
    return true;
}
```

`raw_results` 已经是当前 `post_process()` 做完“YOLO 解码、置信度筛选、NMS”后的结果：

```cpp
raw_results.count
raw_results.results[i].box.left / top / right / bottom  // 704×704 模型坐标
raw_results.results[i].prop                             // 置信度
raw_results.results[i].cls_id                           // 类别 ID
```

## 6. 一次“按需取帧 + YOLO + 更新物品 Track”的调用示例

```cpp
// 例如：门打开后、检测到手后、等待稳定快照期间，才令 should_infer = true。
void process_one_camera_event(bool should_infer,
                              const CameraConfig& camera,
                              rknn_app_context_t& yolo,
                              std::map<int, fridge::InventoryItem>& inventory,
                              int frame_id) {
    if (!should_infer) {
        return;  // 不执行 YOLO；并非只能按固定时间执行
    }

    cv::Mat bgr_frame;
    if (!capture_one_bgr_frame(camera, bgr_frame)) return;

    std::vector<fridge::DetectedItem> all_detections;
    if (!run_yolov5n_on_frame(yolo, bgr_frame, all_detections)) return;

    for (const fridge::DetectedItem& det : all_detections) {
        if (fridge::is_hand(det.cls_id)) {
            // 手框先存起来；之后可用来调用 update_item_by_hand_proxy()。
            continue;
        }
        if (!fridge::is_food(det.cls_id)) continue;

        // Detection × 同类库存物品 × 该物品完整 path。
        const fridge::ItemMatch match =
                fridge::find_best_item_for_detection(det, inventory);
        if (match.item_id < 0) {
            // 暂不创建库存 item：可能是新放入物，也可能是暂时无法判定的同类物。
            // 等稳定快照对比后，再决定是 ITEM_IN 还是其他结果。
            continue;
        }

        fridge::InventoryItem& item = inventory.at(match.item_id);
        fridge::accept_yolo_detection(item, det, frame_id);
        // 匹配成功后该 item 的 path 被压缩为这一个最新点。
    }
}
```

## 7. 快照周期中 Track 的重置位置

```cpp
// 伪代码：必须在“稳定快照 B 对比 A，并且库存已更新”之后重置。
void finish_snapshot_comparison(std::map<int, fridge::InventoryItem>& inventory,
                                int frame_id) {
    for (auto& pair : inventory) {
        fridge::InventoryItem& item = pair.second;
        if (item.status == fridge::ItemStatus::OUT) continue;

        // 此时 anchor_box 已是比较结果确认后的新稳定位置。
        item.track.reset_after_snapshot(item.anchor_box, frame_id);
    }
}

// 不能在快照 B 一生成就清空，因为 A→B 的路径还要参与本次整理判定。
```
