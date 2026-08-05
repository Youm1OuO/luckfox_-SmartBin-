// ============================================================================
//  app_door_state.h
//  Pure image measurements used by the door-state owner in main.cc
// ============================================================================
#ifndef __FRIDGE_APP_DOOR_STATE_H
#define __FRIDGE_APP_DOOR_STATE_H

#include <deque>

#include "opencv2/core/core.hpp"

double update_no_event_pixel_diff(const cv::Mat& frame, cv::Mat& last_gray_small);
double raw_dark_pixel_ratio(const cv::Mat& y_plane);
double median_brightness(const std::deque<double>& values);
bool is_high_confidence_closing_dark(double mean_brightness,
                                     double dark_pixel_ratio,
                                     double open_brightness_reference);

#endif  // __FRIDGE_APP_DOOR_STATE_H
