// ============================================================================
//  app_display.cc
//  OSD and terminal presentation for the main-loop business inputs
// ============================================================================
#include "app_display.h"

#include <algorithm>
#include <cstdio>

#include "opencv2/imgproc/imgproc.hpp"

#include "fridge_config.h"
#include "yolov5n.h"

bool is_strong_hand_for_osd(const fridge::Detection& det) {
    if (!fridge::is_hand(det.cls_id)) return false;
    if (det.score < fridge::OSD_STRONG_HAND_SCORE_THRESH) return false;

	float frame_area = fridge::FRAME_W * fridge::FRAME_H;
	if (frame_area <= 0.0f) return false;

	float area_ratio = det.box.area() / frame_area;
    return area_ratio >= fridge::OSD_STRONG_HAND_MIN_AREA_RATIO;
}

static cv::Scalar display_color_for_class(int cls_id) {
	static const cv::Scalar colors[] = {
		cv::Scalar(  0, 255,   0),
		cv::Scalar(255,   0,   0),
		cv::Scalar(  0, 165, 255),
		cv::Scalar(255,   0, 255),
		cv::Scalar(255, 255,   0),
		cv::Scalar(  0, 255, 255),
		cv::Scalar(128, 255,   0),
		cv::Scalar(255, 128,   0),
		cv::Scalar(128,   0, 255),
		cv::Scalar(  0, 128, 255),
		cv::Scalar(255,   0, 128),
		cv::Scalar(  0, 255, 128),
	};
	int n = (int)(sizeof(colors) / sizeof(colors[0]));
	int idx = cls_id >= 0 ? cls_id % n : 0;
	return colors[idx];
}

// OSD 必须显示与 SessionManager 完全相同的业务输入。低分框仍然是业务证据，
// 只改变颜色提示，不允许被第二个显示阈值隐藏。
void draw_business_input_detection(cv::Mat& frame,
                                          const fridge::Detection& det,
                                          size_t input_index,
                                          bool is_hand_input) {
	const bool low_confidence =
		det.score < fridge::OSD_LOW_CONFIDENCE_OBJECT_SCORE_THRESH;
	const cv::Scalar color = low_confidence
		? cv::Scalar(0, 165, 255)  // 橙色：进入业务层的低分框
		: (is_hand_input ? cv::Scalar(0, 0, 255)
		                 : display_color_for_class(det.cls_id));
	const int x1 = (int)det.box.x1;
	const int y1 = (int)det.box.y1;
	const int x2 = (int)det.box.x2;
	const int y2 = (int)det.box.y2;
	cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 1);

	// 只显示物品名与置信度；不再显示 H/F 序号前缀和 [BUSINESS]/[BUSINESS-LOW] 标签。
	// 低置信度仍通过框和文字的颜色区分（见上面的 color），不占用文字。
	// input_index 现已不参与显示，但保留形参以兼容调用点。
	(void)input_index;
	char label[128];
	snprintf(label, sizeof(label), "%s %.0f%%",
		         coco_cls_to_name(det.cls_id), det.score * 100.0f);
	cv::putText(frame, label, cv::Point(x1, std::max(14, y1 - 6)),
	            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}

// SessionManager 的 hand_boxes 接口只传坐标。为使日志仍能显示与这些坐标
// 完全同源的类别和分数，在调用 process_frame() 前记录 hand_dets_for_display。
void trace_business_hand_inputs(
	int frame_id, const std::vector<fridge::Detection>& hand_dets_for_display) {
	if (!fridge::FLOW3_DEBUG_TRACE_LOG) return;
	for (size_t i = 0; i < hand_dets_for_display.size(); ++i) {
		const fridge::Detection& det = hand_dets_for_display[i];
		printf("[3.0-TRACE][HAND-DETECTION][frame=%d][HAND] "
		       "input-index=%zu cls=%d score=%.3f box=(%.1f,%.1f,%.1f,%.1f) "
		       "business-input=1 osd-business-visible=1\n",
		       frame_id, i, det.cls_id, det.score,
		       det.box.x1, det.box.y1, det.box.x2, det.box.y2);
	}
}

// 业务层只返回结构化事件；这里统一恢复终端可读的 [EVENT] 日志。
// 遮挡 / 露出是库存状态变化，不会作为出入库事件上传云端。
void print_inventory_event(const fridge::InventoryEvent& ev) {
	const char* name = coco_cls_to_name(ev.cls_id);
	switch (ev.kind) {
		case fridge::EventKind::IN:
			printf("\033[1;32m[EVENT]\033[0m 放入: item#%d %s (置信度 %.0f%%) "
			       "位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
			       ev.item_id, name, ev.score * 100.0f,
			       ev.box.x1, ev.box.y1, ev.box.x2, ev.box.y2);
			break;
		case fridge::EventKind::OUT:
			printf("\033[1;32m[EVENT]\033[0m 取出: item#%d %s "
			       "原位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
			       ev.item_id, name,
			       ev.box.x1, ev.box.y1, ev.box.x2, ev.box.y2);
			break;
		case fridge::EventKind::MOVED:
			printf("\033[1;32m[EVENT]\033[0m 整理: item#%d %s "
			       "原位置=(%.0f,%.0f)~(%.0f,%.0f) -> "
			       "新位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
			       ev.item_id, name,
			       ev.before_box.x1, ev.before_box.y1,
			       ev.before_box.x2, ev.before_box.y2,
			       ev.after_box.x1, ev.after_box.y1,
			       ev.after_box.x2, ev.after_box.y2);
			break;
		case fridge::EventKind::OCCLUDED:
			printf("\033[1;32m[EVENT]\033[0m 遮挡: item#%d %s "
			       "位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
			       ev.item_id, name,
			       ev.box.x1, ev.box.y1, ev.box.x2, ev.box.y2);
			break;
		case fridge::EventKind::REVEALED:
			printf("\033[1;32m[EVENT]\033[0m 露出: item#%d %s "
			       "位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
			       ev.item_id, name,
			       ev.box.x1, ev.box.y1, ev.box.x2, ev.box.y2);
			break;
	}
}

