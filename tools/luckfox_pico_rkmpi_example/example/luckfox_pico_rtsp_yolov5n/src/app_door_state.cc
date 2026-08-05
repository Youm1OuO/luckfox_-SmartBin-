// ============================================================================
//  app_door_state.cc
//  Door-state image measurements; state ownership stays in main.cc
// ============================================================================
#include "app_door_state.h"

#include <algorithm>
#include <vector>

#include "opencv2/imgproc/imgproc.hpp"

#include "fridge_config.h"

double update_no_event_pixel_diff(const cv::Mat& frame,
                                         cv::Mat& last_gray_small) {
	cv::Mat gray;
	cv::Mat small;
	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
	cv::resize(gray, small, cv::Size(64, 36), 0, 0, cv::INTER_AREA);

	if (last_gray_small.empty()) {
		small.copyTo(last_gray_small);
		return 0.0;
	}

	cv::Mat diff;
	cv::absdiff(small, last_gray_small, diff);
	double mean_diff = cv::mean(diff)[0];
	small.copyTo(last_gray_small);
	return mean_diff;
}

// 门状态机只使用原始 Y 平面，不依赖缩放、翻转或 YOLO 输出。
double raw_dark_pixel_ratio(const cv::Mat& y_plane) {
	if (y_plane.empty() || y_plane.total() == 0) return 0.0;
	cv::Mat dark_mask;
	cv::threshold(y_plane, dark_mask, fridge::DOOR_DARK_PIXEL_THRESHOLD,
	              255, cv::THRESH_BINARY_INV);
	return (double)cv::countNonZero(dark_mask) / (double)y_plane.total();
}

double median_brightness(const std::deque<double>& values) {
	if (values.empty()) return 0.0;
	std::vector<double> ordered(values.begin(), values.end());
	std::sort(ordered.begin(), ordered.end());
	const size_t mid = ordered.size() / 2;
	return ordered.size() % 2 ? ordered[mid] : (ordered[mid - 1] + ordered[mid]) * 0.5;
}

bool is_high_confidence_closing_dark(double mean_brightness,
	                                           double dark_pixel_ratio,
	                                           double open_brightness_reference) {
	return open_brightness_reference > 0.0 &&
	       mean_brightness <= fridge::DOOR_CLOSING_GUARD_THRESHOLD &&
	       mean_brightness <= open_brightness_reference * fridge::DOOR_CLOSING_DROP_RATIO &&
	       dark_pixel_ratio >= fridge::DOOR_CLOSING_DARK_PIXEL_RATIO_THRESHOLD;
}

