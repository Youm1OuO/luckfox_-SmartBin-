// ============================================================================
//  reclassify.h
//  YOLO 输出后的手动纠正层
//  ------------------------------------------------------------------------
//  在 [YOLO 输出] 与 [业务使用] 之间，对易混类别做启发式纠正：
//    - egg <-> orange ：按框面积
//    - apple <-> onion ：按框内颜色(HSV)
//    - 类别归一化       ：把难区分的类合并（如 lettuce -> chinese_cabbage）
//
//  所有可调阈值都在 fridge_config.h 的 "reclassify" 配置块，调参只改那里。
//  纠正在原图坐标系下采样颜色，因此必须在图像翻转、坐标映射完成之后调用。
// ============================================================================
#ifndef __RECLASSIFY_H
#define __RECLASSIFY_H

#include <vector>

#include "opencv2/core/core.hpp"

#include "tracker.h"   // fridge::Detection

namespace fridge {

// 对单个检测就地纠正 cls_id。frame 为当前(已翻转的)原图 BGR，det.box 使用该图坐标。
// 若纠正生效，返回 true（cls_id 被改动），否则返回 false。
bool reclassify_detection(Detection& det, const cv::Mat& frame);

// 便捷批处理：对一组检测逐个调用 reclassify_detection。
void reclassify_detections(std::vector<Detection>& dets, const cv::Mat& frame);

}  // namespace fridge

#endif  // __RECLASSIFY_H
