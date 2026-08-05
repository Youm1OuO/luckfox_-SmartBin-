// ============================================================================
//  app_display.h
//  OSD and terminal event presentation helpers for the main loop
// ============================================================================
#ifndef __FRIDGE_APP_DISPLAY_H
#define __FRIDGE_APP_DISPLAY_H

#include <cstddef>
#include <vector>

#include "opencv2/core/core.hpp"

#include "session.h"

bool is_strong_hand_for_osd(const fridge::Detection& det);
void draw_business_input_detection(cv::Mat& frame, const fridge::Detection& det,
                                   size_t input_index, bool is_hand_input);
void trace_business_hand_inputs(
        int frame_id, const std::vector<fridge::Detection>& hand_dets_for_display);
void print_inventory_event(const fridge::InventoryEvent& ev);

#endif  // __FRIDGE_APP_DISPLAY_H
