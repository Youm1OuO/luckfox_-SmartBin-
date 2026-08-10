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

// 色相是否落在"橙"区间（用于识别真 orange）。
bool hue_is_orange(float h) {
    return h >= RECLS_ORANGE_H_MIN && h <= RECLS_ORANGE_H_MAX;
}

// 假 orange 过滤判据。三条同时满足才判为"假 orange，应删"：
//   cls_id==CLS_ORANGE 且 分数<RECLS_ORANGE_FILTER_SCORE_MAX 且 框内平均色相不橙。
// 在所有逐框转换【之后】调用，因此这里的 orange 既可能是 YOLO 原生的，也可能是
// egg->orange 面积规则转出来的（两个 egg 拼成的大框）——两者都能被这道过滤扫到。
// 采样失败(拿不到颜色)时保守【不删】，避免误伤。
bool is_bogus_orange(const Detection& det, const cv::Mat& frame) {
    if (!RECLS_ORANGE_FILTER_ENABLED) return false;
    if (det.cls_id != CLS_ORANGE) return false;
    if (det.score >= RECLS_ORANGE_FILTER_SCORE_MAX) return false;  // 高分真橙子，保留
    float mean_h = 0.0f;
    if (!sample_box_hsv(det.box, frame, &mean_h, NULL)) return false;  // 采样失败→保守保留
    if (hue_is_orange(mean_h)) return false;  // 确实是橙色→保留
    return true;  // 低分 + 不橙 → 判为假 orange，删框
}

// bitter_gourd -> cabbage：单向，按框长宽比。苦瓜长条(长宽比偏离1:1)，卷心菜圆球(接近1:1)。
// 一个 bitter_gourd 框若“接近正方形”，判它其实是圆的 cabbage。只单向、不反向。
bool apply_bitter_gourd_cabbage(Detection& det) {
    if (!RECLS_BITTER_GOURD_TO_CABBAGE_ENABLED) return false;
    if (det.cls_id != CLS_BITTER_GOURD) return false;
    const float w = det.box.w();
    const float h = det.box.h();
    if (w <= 1.0f || h <= 1.0f) return false;
    const float shorter = std::min(w, h);
    const float longer  = std::max(w, h);
    const float aspect = shorter / longer;   // 1.0=正方形，越小越长条
    if (aspect >= RECLS_SQUARE_ASPECT_MIN) {  // 接近正方形 → 其实是圆的 cabbage
        det.cls_id = CLS_CABBAGE;
        return true;
    }
    return false;
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

// 类别归一化：遍历 fridge_config.h 里的合并表 RECLS_CLASS_MERGES，把 from 类改记成 to 类。
// 要增删合并关系，只改那张表即可，无需改本函数。合并只做一轮、不连锁（找到第一条匹配
// 的 from 就替换并返回），因此表里“同一个 from 只写一行、不要写链式”即可得到预期结果。
bool apply_class_merge(Detection& det) {
    for (int i = 0; i < RECLS_CLASS_MERGES_COUNT; ++i) {
        if (det.cls_id == RECLS_CLASS_MERGES[i].from) {
            det.cls_id = RECLS_CLASS_MERGES[i].to;
            return true;
        }
    }
    return false;
}

}  // namespace

bool reclassify_detection(Detection& det, const cv::Mat& frame) {
    if (!RECLASSIFY_ENABLED) return false;

    bool changed = false;
    // 先做形状/颜色纠正，再做类别合并。各转换作用于不同类别对，互不干扰：
    //   egg<->orange(面积)、apple<->onion(颜色)、bitter_gourd->cabbage(长宽比,单向)。
    // 类别合并(apply_class_merge)放最后，使合并结果不再被上述转换二次加工(避免链式)。
    changed |= apply_egg_orange(det);
    changed |= apply_apple_onion(det, frame);
    changed |= apply_bitter_gourd_cabbage(det);
    changed |= apply_class_merge(det);
    return changed;
}

namespace {

// 统一去重（最后一步，补一次 NMS）。判据：任意两个框（【不分类别、不分是否被改过】），
// 若 IoU >= RECLS_OVERLAP_DEDUP_IOU（默认 0.90，很高）→ 判为“同一物体的重复框”，
// 保留分数高者、压制低者（分数相同则压制后者，保持确定性）。
// 用高 IoU 门槛：只杀“几乎完全重合、大小也相近”的真重复框（如同位置 apple+onion 双标签）；
// 挨着但不重合的真实物体（IoU 达不到 0.90）不受影响，前后遮挡的大小框（IoU 低）也不误删。
// suppressed[i]=true 表示第 i 个框应被移除。
void dedup_overlapping(const std::vector<Detection>& dets,
                       std::vector<char>* suppressed) {
    const size_t n = dets.size();
    for (size_t i = 0; i < n; ++i) {
        if ((*suppressed)[i]) continue;
        for (size_t j = i + 1; j < n; ++j) {
            if ((*suppressed)[j]) continue;
            if (iou(dets[i].box, dets[j].box) < RECLS_OVERLAP_DEDUP_IOU) continue;
            // 高度重合：保留分数高者，压制另一个。
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

    // 第一步：逐框类别纠正（egg<->orange 面积、apple<->onion 颜色、bitter_gourd->cabbage
    // 长宽比、类别合并）。
    for (size_t i = 0; i < dets.size(); ++i) {
        reclassify_detection(dets[i], frame);
    }

    // 第二步：假 orange 过滤。【放在所有逐框转换之后】，因此对“最终是 orange 的框”都生效——
    // 无论它是 YOLO 原生 orange，还是两个 egg 拼成的大框被 egg->orange 面积规则转出来的假
    // orange，都能被扫到并删除。判据(三条同时满足才删)：cls_id==orange + 分数<阈值 + 颜色不橙。
    if (RECLS_ORANGE_FILTER_ENABLED && !dets.empty()) {
        std::vector<Detection> kept0;
        kept0.reserve(dets.size());
        for (size_t i = 0; i < dets.size(); ++i) {
            if (is_bogus_orange(dets[i], frame)) continue;  // 假 orange → 丢弃
            kept0.push_back(dets[i]);
        }
        dets.swap(kept0);
    }

    // 第三步（最后）：统一去重——任意两框 IoU >= RECLS_OVERLAP_DEDUP_IOU 就留高分、删低分。
    // 不分类别、不分是否改过。去重完直接交业务层。
    if (!RECLS_DEDUP_ENABLED || dets.size() < 2) return;
    std::vector<char> suppressed(dets.size(), 0);
    dedup_overlapping(dets, &suppressed);

    std::vector<Detection> kept;
    kept.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!suppressed[i]) kept.push_back(dets[i]);
    }
    dets.swap(kept);
}

}  // namespace fridge
