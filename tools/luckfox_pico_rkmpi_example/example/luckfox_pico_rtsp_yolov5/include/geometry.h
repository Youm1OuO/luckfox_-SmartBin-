// ============================================================================
//  geometry.h
//  Bounding Box 几何工具 - IoU、中心点、面积等基础运算
//  纯头文件实现，给 tracker 和 hand_state 共用
// ============================================================================
#ifndef __FRIDGE_GEOMETRY_H
#define __FRIDGE_GEOMETRY_H

#include <algorithm>
#include <cmath>

namespace fridge {

// 标准 BBox 结构：左上角 + 右下角 (x1y1x2y2 格式)
// 这跟现有 image_rect_t (left,top,right,bottom) 完全一致，
// 只是换了个我们自己 namespace 下的、能加方法的版本
struct BBox {
    float x1, y1, x2, y2;

    BBox() : x1(0), y1(0), x2(0), y2(0) {}
    BBox(float a, float b, float c, float d) : x1(a), y1(b), x2(c), y2(d) {}

    // 中心 x 坐标
    float cx() const { return (x1 + x2) * 0.5f; }
    // 中心 y 坐标
    float cy() const { return (y1 + y2) * 0.5f; }
    // 宽
    float w()  const { return x2 - x1; }
    // 高
    float h()  const { return y2 - y1; }
    // 面积
    float area() const { return std::max(0.0f, w()) * std::max(0.0f, h()); }
};

// 计算两个 BBox 的 IoU（交并比）
// IoU = 交集面积 / 并集面积，结果在 [0, 1] 之间
// IoU 越大 → 两个框越像同一个物体
inline float iou(const BBox& a, const BBox& b) {
    // 交集左上角 = 两个框左上角的较大值
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    // 交集右下角 = 两个框右下角的较小值
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);

    // 没交集
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    if (inter <= 0.0f) return 0.0f;

    // 并集 = A + B - 交集
    float uni = a.area() + b.area() - inter;
    if (uni <= 0.0f) return 0.0f;

    return inter / uni;
}

// 两个中心点的欧氏距离（手部移动速度判定用得上）
inline float center_distance(const BBox& a, const BBox& b) {
    float dx = a.cx() - b.cx();
    float dy = a.cy() - b.cy();
    return std::sqrt(dx * dx + dy * dy);
}

// 点 (px, py) 是否落在框 b 内
inline bool point_in_box(float px, float py, const BBox& b) {
    return px >= b.x1 && px <= b.x2 && py >= b.y1 && py <= b.y2;
}

// 交集面积占"较小框"的比例
// 用途：判断"手是否盖住了物品"。
//   IoU 在两个框大小悬殊时会失效（大手框盖住小物品框，IoU 很小），
//   但 overlap_ratio_of_smaller 衡量的是"小框被覆盖了多少"，更鲁棒。
//   手完全盖住物品时，这个值接近 1.0。
inline float overlap_ratio_of_smaller(const BBox& a, const BBox& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    if (inter <= 0.0f) return 0.0f;
    float smaller = std::min(a.area(), b.area());
    if (smaller <= 0.0f) return 0.0f;
    return inter / smaller;
}

}  // namespace fridge

#endif  // __FRIDGE_GEOMETRY_H
