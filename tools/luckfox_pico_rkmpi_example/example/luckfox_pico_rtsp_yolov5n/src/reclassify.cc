// ============================================================================
//  reclassify.cc
//  YOLO 输出后的手动纠正层实现。逻辑固定，阈值全部来自 fridge_config.h。
// ============================================================================
#include "reclassify.h"

#include <algorithm>

#include "opencv2/imgproc/imgproc.hpp"

#include "fridge_config.h"

namespace fridge {

namespace {

// 框面积占整幅画面的比例。用比例而非绝对像素，物体远近更稳。
float box_area_ratio(const BBox& box) {
    const float frame_area = FRAME_W * FRAME_H;
    if (frame_area <= 0.0f) return 0.0f;
    return box.area() / frame_area;
}

// 采样框内中心区域的平均颜色(HSV)。向内收缩以避开边缘背景。
// 返回 false 表示框无效/越界，无法采样。
bool sample_box_hsv(const BBox& box, const cv::Mat& frame,
                    float* out_h, float* out_v) {
    if (frame.empty()) return false;

    const float w = box.w();
    const float h = box.h();
    if (w <= 1.0f || h <= 1.0f) return false;

    const float inset_x = w * RECLS_COLOR_INSET_RATIO;
    const float inset_y = h * RECLS_COLOR_INSET_RATIO;

    int x1 = (int)(box.x1 + inset_x);
    int y1 = (int)(box.y1 + inset_y);
    int x2 = (int)(box.x2 - inset_x);
    int y2 = (int)(box.y2 - inset_y);

    // 收缩后若退化，退回用原框，避免采样区为空。
    if (x2 <= x1 || y2 <= y1) {
        x1 = (int)box.x1;
        y1 = (int)box.y1;
        x2 = (int)box.x2;
        y2 = (int)box.y2;
    }

    // 裁剪到图像范围内。
    x1 = std::max(0, std::min(x1, frame.cols - 1));
    y1 = std::max(0, std::min(y1, frame.rows - 1));
    x2 = std::max(0, std::min(x2, frame.cols));
    y2 = std::max(0, std::min(y2, frame.rows));
    if (x2 <= x1 || y2 <= y1) return false;

    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat patch = frame(roi);           // 引用，不拷贝像素
    cv::Mat hsv;
    cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);   // frame 为 BGR
    cv::Scalar mean = cv::mean(hsv);      // [H, S, V, _]，H:0~179 V:0~255

    if (out_h) *out_h = (float)mean[0];
    if (out_v) *out_v = (float)mean[2];
    return true;
}

// 色相是否落在"红"区间（红在 HSV 环两端）。
bool hue_is_red(float h) {
    return h <= RECLS_RED_H_LOW_MAX || h >= RECLS_RED_H_HIGH_MIN;
}

// 色相是否落在"紫/品红"区间。
bool hue_is_purple(float h) {
    return h >= RECLS_PURPLE_H_MIN && h <= RECLS_PURPLE_H_MAX;
}

// egg <-> orange：按框面积纠正。返回是否改动。
bool apply_egg_orange(Detection& det) {
    const float ratio = box_area_ratio(det.box);
    if (det.cls_id == CLS_EGG) {
        if (ratio > RECLS_EGG_TO_ORANGE_AREA_RATIO) {
            det.cls_id = CLS_ORANGE;
            return true;
        }
    } else if (det.cls_id == CLS_ORANGE) {
        if (ratio < RECLS_ORANGE_TO_EGG_AREA_RATIO) {
            det.cls_id = CLS_EGG;
            return true;
        }
    }
    return false;
}

// apple <-> onion：按框内颜色纠正。返回是否改动。
bool apply_apple_onion(Detection& det, const cv::Mat& frame) {
    if (det.cls_id != CLS_APPLE && det.cls_id != CLS_ONION) return false;

    float mean_h = 0.0f, mean_v = 0.0f;
    if (!sample_box_hsv(det.box, frame, &mean_h, &mean_v)) return false;

    const bool dark   = mean_v < RECLS_DARK_V_MAX;
    const bool bright = mean_v > RECLS_BRIGHT_V_MIN;

    if (det.cls_id == CLS_APPLE) {
        // apple 但看起来偏紫/偏暗 → onion
        if (dark || hue_is_purple(mean_h)) {
            det.cls_id = CLS_ONION;
            return true;
        }
    } else {  // CLS_ONION
        // onion 但看起来偏红/偏亮 → apple
        if (bright && hue_is_red(mean_h)) {
            det.cls_id = CLS_APPLE;
            return true;
        }
    }
    return false;
}

// 类别归一化：把难区分的类合并到一个代表类（如 lettuce -> chinese_cabbage）。
// 要新增合并关系，在此仿照再加一段判断即可。
bool apply_class_merge(Detection& det) {
    bool changed = false;
    if (RECLS_MERGE_LETTUCE_CABBAGE && det.cls_id == RECLS_MERGE_FROM) {
        det.cls_id = RECLS_MERGE_TO;
        changed = true;
    }
    return changed;
}

}  // namespace

bool reclassify_detection(Detection& det, const cv::Mat& frame) {
    if (!RECLASSIFY_ENABLED) return false;

    bool changed = false;
    // 先做形状/颜色纠正，再做类别合并：例如 egg->orange 后不涉及合并，
    // 而合并只影响 lettuce/cabbage，互不干扰。
    changed |= apply_egg_orange(det);
    changed |= apply_apple_onion(det, frame);
    changed |= apply_class_merge(det);
    return changed;
}

namespace {

// 纠正后的同类去重（补一次 NMS）。做法 A：只处理"至少一个框被 reclassify 改过
// 类别"的重叠对，绝不碰 YOLO 原生同类框。changed[i] 标记第 i 个框是否被改过。
// 对同 cls_id、IoM 超阈值的一对，压制分数较低者（分数相同则压制后者，保持确定性）。
// suppressed[i]=true 表示第 i 个框应被移除。
void dedup_after_reclassify(const std::vector<Detection>& dets,
                            const std::vector<char>& changed,
                            std::vector<char>* suppressed) {
    const size_t n = dets.size();
    for (size_t i = 0; i < n; ++i) {
        if ((*suppressed)[i]) continue;
        for (size_t j = i + 1; j < n; ++j) {
            if ((*suppressed)[j]) continue;
            // 做法 A：跳过“两个框都不是我们改出来的”的原生对。
            if (!changed[i] && !changed[j]) continue;
            if (dets[i].cls_id != dets[j].cls_id) continue;
            if (iom(dets[i].box, dets[j].box) < RECLS_DEDUP_IOM_THRESH) continue;
            // 同类、且高度重叠：保留分数高者，压制另一个。
            if (dets[j].score > dets[i].score) {
                (*suppressed)[i] = 1;
                break;  // i 已被压制，不再作为基准与后续比较。
            } else {
                (*suppressed)[j] = 1;
            }
        }
    }
}

}  // namespace

void reclassify_detections(std::vector<Detection>& dets, const cv::Mat& frame) {
    if (!RECLASSIFY_ENABLED) return;

    // 第一步：逐框类别纠正，同时记录哪些框被改过（供做法 A 去重）。
    std::vector<char> changed(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        changed[i] = reclassify_detection(dets[i], frame) ? 1 : 0;
    }

    // 第二步：紧跟其后补做一次同类去重，清理纠正制造出的重叠同类框。
    if (!RECLS_DEDUP_ENABLED || dets.size() < 2) return;
    std::vector<char> suppressed(dets.size(), 0);
    dedup_after_reclassify(dets, changed, &suppressed);

    std::vector<Detection> kept;
    kept.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!suppressed[i]) kept.push_back(dets[i]);
    }
    dets.swap(kept);
}

}  // namespace fridge
