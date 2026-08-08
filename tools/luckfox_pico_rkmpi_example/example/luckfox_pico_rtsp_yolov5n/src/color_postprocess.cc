// ============================================================================
//  color_postprocess.cc
//  演示专用的少数类别颜色/几何二次判别。
//
//  设计原则：YOLO 是默认答案；只有中心 ROI 的视觉证据足够强时才覆盖
//  apple/onion 或 egg/orange 的 cls_id。所有异常情况都回退原始类别。
// ============================================================================
#include "color_postprocess.h"

#include "fridge_config.h"
#include "geometry.h"

#include "opencv2/imgproc/imgproc.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdio.h>

namespace fridge {
namespace {

constexpr int CLASS_APPLE = 0;
constexpr int CLASS_ORANGE = 2;
constexpr int CLASS_ONION = 15;
constexpr int CLASS_EGG = 18;

// 这些阈值是保守的第一版起点，按实际板端画面调参时只改本文件。
// 颜色比例以中心 ROI 中满足对应 HSV 条件的像素占比表示。
constexpr float ROI_FRACTION = 0.62f;
constexpr int MIN_ROI_WIDTH = 12;
constexpr int MIN_ROI_HEIGHT = 12;
constexpr float MIN_VALID_PIXEL_RATIO = 0.55f;

// OpenCV HSV: H in [0, 179], S/V in [0, 255].
constexpr int COLOR_MIN_SATURATION = 45;
constexpr int COLOR_MIN_VALUE = 35;

// egg / orange 的大小以原图 bbox 的真实像素尺寸判定，不再使用笼统的
// "全画面面积百分比"。这一对的规则就是：小框且偏黄/白/棕黄 -> egg；
// 其他情况 -> orange。
constexpr float SIZE_REFERENCE_WIDTH = 1280.0f;
constexpr float SIZE_REFERENCE_HEIGHT = 720.0f;
constexpr float EGG_MAX_BOX_WIDTH = 155.0f;
constexpr float EGG_MAX_BOX_HEIGHT = 235.0f;
constexpr float EGG_MAX_BOX_AREA = 26000.0f;

struct ColorStats {
    bool valid = false;
    float red_ratio = 0.0f;
    float purple_ratio = 0.0f;
    float orange_ratio = 0.0f;
    float yellow_white_ratio = 0.0f;
    float egg_tone_ratio = 0.0f;
    float mean_saturation = 0.0f;
    float mean_value = 0.0f;
};

struct BoxSize {
    float width = 0.0f;
    float height = 0.0f;
    float area = 0.0f;
    bool egg_sized = false;
};

struct ClassificationResult {
    int cls_id = -1;
    bool egg_color_like = false;
    bool orange_color_like = false;
    bool apple_like = false;
    bool onion_like = false;
};

static bool is_target_pair_class(int cls_id) {
    return cls_id == CLASS_APPLE || cls_id == CLASS_ONION ||
           cls_id == CLASS_EGG || cls_id == CLASS_ORANGE;
}

static bool box_is_usable(const BBox& box, const cv::Mat& frame,
                          cv::Rect* roi_out) {
    if (frame.empty() || frame.channels() != 3 || roi_out == nullptr) {
        return false;
    }
    if (box.w() < static_cast<float>(MIN_ROI_WIDTH) ||
        box.h() < static_cast<float>(MIN_ROI_HEIGHT) ||
        box.area() <= 0.0f) {
        return false;
    }

    const float frame_width = static_cast<float>(frame.cols);
    const float frame_height = static_cast<float>(frame.rows);
    if (frame_width <= 0.0f || frame_height <= 0.0f) return false;

    // 边缘框可能被画面裁断，中心颜色不足以代表完整物体时不改类。
    const float edge_tolerance = 1.0f;
    if (box.x1 <= edge_tolerance || box.y1 <= edge_tolerance ||
        box.x2 >= frame_width - edge_tolerance ||
        box.y2 >= frame_height - edge_tolerance) {
        return false;
    }

    const float cx = (box.x1 + box.x2) * 0.5f;
    const float cy = (box.y1 + box.y2) * 0.5f;
    const float roi_width = box.w() * ROI_FRACTION;
    const float roi_height = box.h() * ROI_FRACTION;
    const int left = std::max(0, static_cast<int>(std::floor(cx - roi_width * 0.5f)));
    const int top = std::max(0, static_cast<int>(std::floor(cy - roi_height * 0.5f)));
    const int right = std::min(frame.cols, static_cast<int>(std::ceil(cx + roi_width * 0.5f)));
    const int bottom = std::min(frame.rows, static_cast<int>(std::ceil(cy + roi_height * 0.5f)));
    if (right <= left || bottom <= top || right - left < MIN_ROI_WIDTH ||
        bottom - top < MIN_ROI_HEIGHT) {
        return false;
    }

    const float expected_area = std::max(1.0f, roi_width * roi_height);
    const float actual_area = static_cast<float>((right - left) * (bottom - top));
    if (actual_area / expected_area < MIN_VALID_PIXEL_RATIO) return false;

    *roi_out = cv::Rect(left, top, right - left, bottom - top);
    return true;
}

static ColorStats extract_color_stats(const cv::Mat& frame, const cv::Rect& roi) {
    ColorStats stats;
    if (roi.width <= 0 || roi.height <= 0) return stats;

    cv::Mat hsv;
    cv::cvtColor(frame(roi), hsv, cv::COLOR_BGR2HSV);

    const int total = hsv.rows * hsv.cols;
    if (total <= 0) return stats;

    int valid_pixels = 0;
    int red_pixels = 0;
    int purple_pixels = 0;
    int orange_pixels = 0;
    int yellow_white_pixels = 0;
    int egg_tone_pixels = 0;
    double saturation_sum = 0.0;
    double value_sum = 0.0;

    for (int y = 0; y < hsv.rows; ++y) {
        const cv::Vec3b* row = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv.cols; ++x) {
            const int h = row[x][0];
            const int s = row[x][1];
            const int v = row[x][2];

            // 极暗像素主要是阴影/噪声，不作为颜色证据，但仍参与亮度统计。
            if (v >= COLOR_MIN_VALUE) {
                ++valid_pixels;
                saturation_sum += s;
                value_sum += v;
            }

            const bool colorful = s >= COLOR_MIN_SATURATION && v >= COLOR_MIN_VALUE;
            const bool red = colorful && (h <= 10 || h >= 170);
            const bool orange = colorful && h >= 8 && h <= 25;
            const bool purple = colorful && h >= 125 && h <= 170;
            // 鸡蛋可能是白、米黄、黄褐或棕黄色。这里保留一套较严格的
            // yellow_white，同时增加 egg_tone，供“小框 + 偏黄”规则使用。
            const bool yellow_white = v >= 80 &&
                ((h >= 15 && h <= 50 && s <= 190) || (s <= 65 && v >= 115));
            const bool egg_tone = v >= 55 &&
                ((h >= 4 && h <= 55) || (s <= 95 && v >= 80));

            if (red) ++red_pixels;
            if (orange) ++orange_pixels;
            if (purple) ++purple_pixels;
            if (yellow_white) ++yellow_white_pixels;
            if (egg_tone) ++egg_tone_pixels;
        }
    }

    if (valid_pixels < static_cast<int>(total * MIN_VALID_PIXEL_RATIO)) return stats;

    stats.valid = true;
    stats.red_ratio = static_cast<float>(red_pixels) / static_cast<float>(total);
    stats.purple_ratio = static_cast<float>(purple_pixels) / static_cast<float>(total);
    stats.orange_ratio = static_cast<float>(orange_pixels) / static_cast<float>(total);
    stats.yellow_white_ratio = static_cast<float>(yellow_white_pixels) / static_cast<float>(total);
    stats.egg_tone_ratio = static_cast<float>(egg_tone_pixels) / static_cast<float>(total);
    stats.mean_saturation = static_cast<float>(saturation_sum / valid_pixels);
    stats.mean_value = static_cast<float>(value_sum / valid_pixels);
    return stats;
}

static bool overlaps_hand(const BBox& object_box,
                          const std::vector<BBox>& hand_boxes) {
    for (const BBox& hand_box : hand_boxes) {
        // 物品小框被手覆盖达到约五分之一时，中心颜色很可能不可信。
        if (overlap_ratio_of_smaller(object_box, hand_box) >= 0.20f) {
            return true;
        }
    }
    return false;
}

static BoxSize measure_box_size(const BBox& box, const cv::Mat& frame) {
    BoxSize size;
    size.width = std::max(0.0f, box.w());
    size.height = std::max(0.0f, box.h());
    size.area = size.width * size.height;
    if (frame.cols <= 0 || frame.rows <= 0) return size;

    const float width_scale = static_cast<float>(frame.cols) / SIZE_REFERENCE_WIDTH;
    const float height_scale = static_cast<float>(frame.rows) / SIZE_REFERENCE_HEIGHT;
    const float max_width = EGG_MAX_BOX_WIDTH * width_scale;
    const float max_height = EGG_MAX_BOX_HEIGHT * height_scale;
    const float max_area = EGG_MAX_BOX_AREA * width_scale * height_scale;
    size.egg_sized = size.width <= max_width &&
                     size.height <= max_height &&
                     size.area <= max_area;
    return size;
}

static ClassificationResult classify_egg_orange(int original_cls_id,
                                                 const BoxSize& size,
                                                 const ColorStats& stats) {
    ClassificationResult result;
    result.cls_id = original_cls_id;
    if (!stats.valid) return result;

    // 黄色鸡蛋在现场光源下可能偏白、黄、米黄或棕黄，不能要求低饱和。
    // 使用宽松的 egg_tone，实际类别仍由“小框”这个必要条件保护。
    result.egg_color_like =
        stats.egg_tone_ratio >= 0.08f;

    if (size.egg_sized && result.egg_color_like) {
        result.cls_id = CLASS_EGG;
    } else {
        // 与原始需求一致：egg/orange 这一对中，不满足“小框且偏黄”的
        // 结果都显示为 orange，不保留这两类中的原始 YOLO cls_id。
        result.cls_id = CLASS_ORANGE;
    }
    return result;
}

static ClassificationResult classify_apple_onion(int original_cls_id,
                                                  const ColorStats& stats) {
    ClassificationResult result;
    result.cls_id = original_cls_id;
    if (!stats.valid) return result;

    result.apple_like =
        stats.red_ratio >= 0.24f &&
        stats.red_ratio >= stats.purple_ratio + 0.05f &&
        stats.mean_saturation >= 55.0f &&
        stats.mean_value >= 65.0f;
    result.onion_like =
        stats.purple_ratio >= 0.20f &&
        stats.purple_ratio >= stats.red_ratio + 0.04f &&
        stats.mean_saturation >= 45.0f;

    if (result.apple_like && !result.onion_like) result.cls_id = CLASS_APPLE;
    if (result.onion_like && !result.apple_like) result.cls_id = CLASS_ONION;
    return result;
}

static bool should_log(int original_cls_id) {
    return ENABLE_COLOR_POSTPROCESS_DEBUG_LOG &&
           (original_cls_id == CLASS_ORANGE || original_cls_id == CLASS_ONION);
}

static void log_skip(int original_cls_id, const BBox& box, const char* reason) {
    if (!should_log(original_cls_id)) return;
    printf("[COLOR] raw_cls=%d box=%.0fx%.0f area=%.0f skip=%s\n",
           original_cls_id, box.w(), box.h(), box.area(), reason);
}

static void log_decision(int original_cls_id, int final_cls_id, const BoxSize& size,
                         const ColorStats& stats, const ClassificationResult& result) {
    if (!should_log(original_cls_id)) return;
    printf("[COLOR] raw_cls=%d final_cls=%d box=%.0fx%.0f area=%.0f egg_size=%d "
           "yellow=%.2f egg_tone=%.2f orange=%.2f red=%.2f purple=%.2f sat=%.0f value=%.0f "
           "egg_color=%d orange_color=%d apple=%d onion=%d\n",
           original_cls_id, final_cls_id, size.width, size.height, size.area,
           size.egg_sized ? 1 : 0, stats.yellow_white_ratio, stats.egg_tone_ratio,
           stats.orange_ratio,
           stats.red_ratio, stats.purple_ratio, stats.mean_saturation, stats.mean_value,
           result.egg_color_like ? 1 : 0, result.orange_color_like ? 1 : 0,
           result.apple_like ? 1 : 0, result.onion_like ? 1 : 0);
}

}  // namespace

void apply_color_postprocess(const cv::Mat& frame,
                             std::vector<Detection>& detections,
                             const std::vector<BBox>& hand_boxes) {
    if (!ENABLE_COLOR_POSTPROCESS) return;
    if (frame.empty() || frame.channels() != 3) return;

    for (Detection& detection : detections) {
        const int original_cls_id = detection.cls_id;
        if (!is_target_pair_class(original_cls_id)) continue;

        const bool is_egg_orange =
            original_cls_id == CLASS_EGG || original_cls_id == CLASS_ORANGE;
        // egg/orange 的展示规则需要在手持时也生效；apple/onion 仍保留手部
        // 保护，避免肤色覆盖使苹果和洋葱的颜色判断失真。
        if (!is_egg_orange && overlaps_hand(detection.box, hand_boxes)) {
            log_skip(original_cls_id, detection.box, "hand_overlap");
            continue;
        }

        cv::Rect roi;
        if (!box_is_usable(detection.box, frame, &roi)) {
            log_skip(original_cls_id, detection.box, "invalid_roi");
            continue;
        }
        const ColorStats stats = extract_color_stats(frame, roi);
        if (!stats.valid) {
            log_skip(original_cls_id, detection.box, "invalid_color");
            continue;
        }

        ClassificationResult result;
        result.cls_id = original_cls_id;
        BoxSize size = measure_box_size(detection.box, frame);
        if (original_cls_id == CLASS_APPLE || original_cls_id == CLASS_ONION) {
            result = classify_apple_onion(original_cls_id, stats);
        } else if (is_egg_orange) {
            result = classify_egg_orange(original_cls_id, size, stats);
        }
        const int corrected_cls_id = result.cls_id;
        log_decision(original_cls_id, corrected_cls_id, size, stats, result);

        // 最终再限制一次目标类别对，防止未来扩展时意外跨组改类。
        const bool apple_onion_pair =
            (original_cls_id == CLASS_APPLE || original_cls_id == CLASS_ONION) &&
            (corrected_cls_id == CLASS_APPLE || corrected_cls_id == CLASS_ONION);
        const bool egg_orange_pair =
            (original_cls_id == CLASS_EGG || original_cls_id == CLASS_ORANGE) &&
            (corrected_cls_id == CLASS_EGG || corrected_cls_id == CLASS_ORANGE);
        if (apple_onion_pair || egg_orange_pair) {
            detection.cls_id = corrected_cls_id;
        }
    }
}

}  // namespace fridge
