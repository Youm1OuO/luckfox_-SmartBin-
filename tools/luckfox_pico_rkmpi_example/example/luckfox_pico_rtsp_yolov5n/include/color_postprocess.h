// ============================================================================
//  color_postprocess.h
//  演示专用的少数类别颜色/几何二次判别。
//
//  该模块只在 YOLO 已经完成解码、NMS 和坐标映射后运行。它不改变 bbox、
//  score 或检测数量；颜色证据不足时始终保留 YOLO 的原始 cls_id。
// ============================================================================
#ifndef __FRIDGE_COLOR_POSTPROCESS_H
#define __FRIDGE_COLOR_POSTPROCESS_H

#include <vector>

#include "opencv2/core/core.hpp"
#include "tracker.h"

namespace fridge {

// 对当前帧的检测做一次保守的颜色/框大小二次判别。
// frame 必须是已经翻转后的原图 BGR；detections 的 bbox 必须已经映射到
// frame 的像素坐标。该函数只允许修改目标四类的 cls_id。
void apply_color_postprocess(const cv::Mat& frame,
                             std::vector<Detection>& detections,
                             const std::vector<BBox>& hand_boxes);

}  // namespace fridge

#endif  // __FRIDGE_COLOR_POSTPROCESS_H
