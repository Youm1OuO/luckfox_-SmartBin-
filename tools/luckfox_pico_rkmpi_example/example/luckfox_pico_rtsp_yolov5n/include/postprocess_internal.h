// ============================================================================
//  postprocess_internal.h
//  Internal YOLOv5 decode helpers shared by postprocess implementation files
// ============================================================================
#ifndef __FRIDGE_POSTPROCESS_INTERNAL_H
#define __FRIDGE_POSTPROCESS_INTERNAL_H

#include <stdint.h>
#include <vector>

int process_i8(int8_t* input, int* anchor, int grid_h, int grid_w,
               int height, int width, int stride, std::vector<float>& boxes,
               std::vector<float>& obj_probs, std::vector<int>& class_id,
               float threshold, int32_t zp, float scale);
int process_i8_rv1106(int8_t* input, int* anchor, int grid_h, int grid_w,
                      int height, int width, int stride,
                      std::vector<float>& boxes, std::vector<float>& box_scores,
                      std::vector<int>& class_id, float threshold,
                      int32_t zp, float scale);
int process_fp32(float* input, int* anchor, int grid_h, int grid_w,
                 int height, int width, int stride, std::vector<float>& boxes,
                 std::vector<float>& obj_probs, std::vector<int>& class_id,
                 float threshold);

#endif  // __FRIDGE_POSTPROCESS_INTERNAL_H
