// ============================================================================
//  main.cc
//  冰箱视觉系统主循环
//
//  每帧流水线：
//    1. 摄像头采集 + YOLO推理 + 坐标映射
//    2. ByteTrack-Lite 每帧更新（只用于 OSD 显示）
//    3. SessionManager 更新轻量 OperationTrack；有手时清空稳定缓冲
//    4. 连续 N 帧无手稳定 → 形成一个稳定快照并直接比较库存表
//    5. 事件上报 + 画面绘制 + RTSP推流
// ============================================================================
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <deque>
#include <vector>
#include <thread>

#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "yolov5n.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// ===== 冰箱视觉系统模块 =====
#include "fridge_config.h"
#include "geometry.h"
#include "tracker.h"
#include "snapshot.h"
#include "session.h"
#include "inventory.h"
#include "cloud_uploader.h"

// 1280*720, 1920*1080
#define DISP_WIDTH  1280
#define DISP_HEIGHT 720

// disp size
int width    = DISP_WIDTH;
int height   = DISP_HEIGHT;

// model size
int model_width = 704;
int model_height = 704;
float scale ;
int leftPadding ;
int topPadding  ;

cv::Mat letterbox(cv::Mat input)
{
	float scaleX = (float)model_width  / (float)width;
	float scaleY = (float)model_height / (float)height;
	scale = scaleX < scaleY ? scaleX : scaleY;

	int inputWidth   = (int)((float)width * scale);
	int inputHeight  = (int)((float)height * scale);

	leftPadding = (model_width  - inputWidth) / 2;
	topPadding  = (model_height - inputHeight) / 2;

	cv::Mat inputScale;
    cv::resize(input, inputScale, cv::Size(inputWidth,inputHeight), 0, 0, cv::INTER_LINEAR);
	cv::Mat letterboxImage(model_height, model_width, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Rect roi(leftPadding, topPadding, inputWidth, inputHeight);
    inputScale.copyTo(letterboxImage(roi));

	return letterboxImage;
}

void mapCoordinates(int *x, int *y) {
	int mx = *x - leftPadding;
	int my = *y - topPadding;

	int rx = (int)((float)mx / scale);
	int ry = (int)((float)my / scale);

	if (rx < 0)      rx = 0;
	if (ry < 0)      ry = 0;
	if (rx > width)  rx = width;
	if (ry > height) ry = height;

	*x = rx;
	*y = ry;
}

static bool request_backend_inventory(const char* device_id,
                                      const char* session_id,
                                      long long opened_at_ms,
                                      std::vector<fridge::InventoryItem>& items,
                                      bool* authoritative_empty) {
	(void)device_id;
	(void)session_id;
	(void)opened_at_ms;
	items.clear();
	if (authoritative_empty) *authoritative_empty = false;

	// 后台库存 GET/POST 协议还没有在端侧落地。这里保留明确接入点：
	// 后续拿到后端返回 JSON 后，只需要在本函数里填充 items 并返回 true。
	// 后台未接入期间，SessionManager 会依据配置决定是否用首次稳定快照建立
	// 本地测试库存；接入可信后台后可关闭该测试兜底。
	printf("[BACKEND] 开门库存获取接口未接入；将按测试配置处理首次快照\n");
	return false;
}

static bool is_strong_hand_for_osd(const fridge::Detection& det) {
    if (!fridge::is_hand(det.cls_id)) return false;
    if (det.score < fridge::OSD_STRONG_HAND_SCORE_THRESH) return false;

	float frame_area = fridge::FRAME_W * fridge::FRAME_H;
	if (frame_area <= 0.0f) return false;

	float area_ratio = det.box.area() / frame_area;
    return area_ratio >= fridge::OSD_STRONG_HAND_MIN_AREA_RATIO;
}

static double update_no_event_pixel_diff(const cv::Mat& frame,
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
static double raw_dark_pixel_ratio(const cv::Mat& y_plane) {
	if (y_plane.empty() || y_plane.total() == 0) return 0.0;
	cv::Mat dark_mask;
	cv::threshold(y_plane, dark_mask, fridge::DOOR_DARK_PIXEL_THRESHOLD,
	              255, cv::THRESH_BINARY_INV);
	return (double)cv::countNonZero(dark_mask) / (double)y_plane.total();
}

static double median_brightness(const std::deque<double>& values) {
	if (values.empty()) return 0.0;
	std::vector<double> ordered(values.begin(), values.end());
	std::sort(ordered.begin(), ordered.end());
	const size_t mid = ordered.size() / 2;
	return ordered.size() % 2 ? ordered[mid] : (ordered[mid - 1] + ordered[mid]) * 0.5;
}

static bool is_high_confidence_closing_dark(double mean_brightness,
	                                           double dark_pixel_ratio,
	                                           double open_brightness_reference) {
	return open_brightness_reference > 0.0 &&
	       mean_brightness <= fridge::DOOR_CLOSING_GUARD_THRESHOLD &&
	       mean_brightness <= open_brightness_reference * fridge::DOOR_CLOSING_DROP_RATIO &&
	       dark_pixel_ratio >= fridge::DOOR_CLOSING_DARK_PIXEL_RATIO_THRESHOLD;
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

// 业务层只返回结构化事件；这里统一恢复终端可读的 [EVENT] 日志。
// 遮挡 / 露出是库存状态变化，不会作为出入库事件上传云端。
static void print_inventory_event(const fridge::InventoryEvent& ev) {
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


int main(int argc, char *argv[]) {
  // 抑制 Rockchip MPP 硬件编码器的调试日志
  setenv("MPP_LOG_LEVEL", "1", 1);
  setenv("mpi_debug", "0", 1);
  freopen("/dev/null", "w", stderr);

  system("RkLunch-stop.sh");
  system("rm -rf ./captures/*");
  system("mkdir -p ./captures");

	RK_S32 s32Ret = 0;
	int sX,sY,eX,eY;

	// Rknn model
	char text[16];
	rknn_app_context_t rknn_app_ctx;
	object_detect_result_list od_results;
    int ret;
	const char *model_path = "./model/yolov5n.rknn";
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
	init_yolov5n_model(model_path, &rknn_app_ctx);
	printf("init rknn model success!\n");
	init_post_process();

	//h264_frame
	VENC_STREAM_S stFrame;
	stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
	RK_U64 H264_PTS = 0;
	RK_U32 H264_TimeRef = 0;
	VIDEO_FRAME_INFO_S stViFrame;

	// Create Pool
	MB_POOL_CONFIG_S PoolCfg;
	memset(&PoolCfg, 0, sizeof(MB_POOL_CONFIG_S));
	PoolCfg.u64MBSize = width * height * 3 ;
	PoolCfg.u32MBCnt = 1;
	PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA;
	MB_POOL src_Pool = RK_MPI_MB_CreatePool(&PoolCfg);
	printf("Create Pool success !\n");

	// Get MB from Pool
	MB_BLK src_Blk = RK_MPI_MB_GetMB(src_Pool, width * height * 3, RK_TRUE);

	// Build h264_frame
	VIDEO_FRAME_INFO_S h264_frame;
	h264_frame.stVFrame.u32Width = width;
	h264_frame.stVFrame.u32Height = height;
	h264_frame.stVFrame.u32VirWidth = width;
	h264_frame.stVFrame.u32VirHeight = height;
	h264_frame.stVFrame.enPixelFormat =  RK_FMT_RGB888;
	h264_frame.stVFrame.u32FrameFlag = 160;
	h264_frame.stVFrame.pMbBlk = src_Blk;
	unsigned char *data = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);
	cv::Mat frame(cv::Size(width,height),CV_8UC3,data);

	// rkaiq init
	RK_BOOL multi_sensor = RK_FALSE;
	const char *iq_dir = "/etc/iqfiles";
	rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
	SAMPLE_COMM_ISP_Init(0, hdr_mode, multi_sensor, iq_dir);
	SAMPLE_COMM_ISP_Run(0);

	// rkmpi init
	if (RK_MPI_SYS_Init() != RK_SUCCESS) {
		RK_LOGE("rk mpi sys init fail!");
		return -1;
	}

	// rtsp init
	rtsp_demo_handle g_rtsplive = NULL;
	rtsp_session_handle g_rtsp_session;
	g_rtsplive = create_rtsp_demo(554);
	g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
	rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
	rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());

	// vi init
	vi_dev_init();
	vi_chn_init(0, width, height);

	// venc init
	RK_CODEC_ID_E enCodecType = RK_VIDEO_ID_AVC;
	venc_init(0, width, height, enCodecType);

	printf("venc init success\n");

	// ============================================================
	//  业务模块初始化
	// ============================================================
	fridge::SessionManager session;
	fridge::CloudUploader cloud;
	bool cloud_enabled = fridge::CLOUD_ENABLED;
	if (const char* env_cloud_enabled = getenv("FRIDGE_CLOUD_ENABLED")) {
		cloud_enabled = atoi(env_cloud_enabled) != 0;
	}
	if (cloud_enabled) {
		cloud.start();
	} else {
		printf("[CLOUD] 离线测试模式：已关闭登录、心跳和事件上传 "
		       "（需要后台时设置 FRIDGE_CLOUD_ENABLED=1）\n");
	}
	int g_frame_id = 0;

	// ============================================================
	//  开关门状态机：CLOSED / OPENING / OPEN / CLOSING
	// ============================================================
	enum class DoorState { CLOSED, OPENING, OPEN, CLOSING };
	DoorState door_state = DoorState::CLOSED;
	int opening_confirm_count = 0;
	int closing_confirm_count = 0;
	std::deque<double> recent_open_brightness;
	double open_brightness_reference = 0.0;
	int door_session_seq = 0;
	char door_session_id[96] = "";
	const double NO_EVENT_PIXEL_DIFF_THRESH = 2.0;
	const int NO_EVENT_STABLE_CONFIRM = 3;
	const long long NO_EVENT_SNAPSHOT_COOLDOWN_MS = 30000;
	cv::Mat no_event_last_gray_small;
	int no_event_stable_count = 0;
	long long last_no_event_upload_ms = 0;

  	while(1)
	{
		// get vi frame
		h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
		h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs();
		s32Ret = RK_MPI_VI_GetChnFrame(0, 0, &stViFrame, -1);
		if(s32Ret == RK_SUCCESS)
		{
			void *vi_data = RK_MPI_MB_Handle2VirAddr(stViFrame.stVFrame.pMbBlk);

			cv::Mat yuv420sp(height + height / 2, width, CV_8UC1, vi_data);
			cv::cvtColor(yuv420sp, frame, cv::COLOR_YUV420sp2BGR);

			// ============================================================
			//  开关门检测：YUV420SP 的前 width*height 字节就是原始 Y 平面。
			//  进入 CLOSING 后立即停止 YOLO、Track、快照与结算，防止暗帧混入。
			// ============================================================
			bool run_yolo_this_frame = false;
			{
				cv::Mat y_plane(height, width, CV_8UC1, vi_data);
				const double mean_y = cv::mean(y_plane)[0];
				const double dark_ratio = raw_dark_pixel_ratio(y_plane);
				const bool high_confidence_closing = is_high_confidence_closing_dark(
					mean_y, dark_ratio, open_brightness_reference);

				if (door_state == DoorState::CLOSED) {
					if (mean_y >= fridge::DOOR_OPENING_CANDIDATE_BRIGHTNESS) {
						door_state = DoorState::OPENING;
						opening_confirm_count = 1;
					}
				} else if (door_state == DoorState::OPENING) {
					if (mean_y >= fridge::DOOR_OPEN_LIGHT_THRESHOLD) {
						++opening_confirm_count;
						if (opening_confirm_count >= fridge::DOOR_OPEN_CONFIRM_FRAME_COUNT) {
							door_state = DoorState::OPEN;
							recent_open_brightness.clear();
							recent_open_brightness.push_back(mean_y);
							open_brightness_reference = mean_y;
							printf("\n\033[1;33m[DOOR]\033[0m 开门 (亮度=%.0f)\n", mean_y);
							RK_U64 ts = TEST_COMM_GetNowUs() / 1000;
							door_session_seq++;
							snprintf(door_session_id, sizeof(door_session_id),
							         "%lld_%d_%s", (long long)ts, door_session_seq,
							         cloud.device_id.c_str());
							session.start_new_session((long long)ts);
							no_event_last_gray_small.release();
							no_event_stable_count = 0;
							last_no_event_upload_ms = 0;

							// 只有本地从未初始化库存时才访问后台；本地库存跨关门保留。
							if (session.needs_backend_inventory()) {
								std::vector<fridge::InventoryItem> backend_items;
								bool authoritative_empty = false;
								if (request_backend_inventory(cloud.device_id.c_str(),
								                              door_session_id,
								                              (long long)ts,
								                              backend_items,
								                              &authoritative_empty)) {
									session.init_from_backend(backend_items, authoritative_empty);
								} else {
									session.mark_backend_unavailable();
								}
							} else {
								printf("[SESSION] 使用跨关门保留的本地库存，不申请后台库存\n");
							}
						}
					} else {
						door_state = DoorState::CLOSED;
						opening_confirm_count = 0;
					}
				} else if (door_state == DoorState::CLOSING) {
					if (high_confidence_closing) {
						++closing_confirm_count;
						if (closing_confirm_count >= fridge::DOOR_CLOSE_CONFIRM_FRAME_COUNT) {
							printf("\n\033[1;33m[DOOR]\033[0m 关门 (亮度=%.0f)\n", mean_y);
							RK_U64 ts = TEST_COMM_GetNowUs() / 1000;
							const bool may_upload_inventory = session.has_local_inventory();
							session.finish_session((long long)ts);
							if (may_upload_inventory && cloud_enabled) {
								std::string json = session.inventory().to_json(
									cloud.device_id.c_str(), (long long)ts, door_session_id);
								printf("%s\n", json.c_str());
								cloud.enqueue_inventory_snapshot(json, (long long)ts);
							} else if (may_upload_inventory) {
								printf("[CLOUD] 离线测试模式：关门库存只保留本地，不上传\n");
							} else {
								printf("[BACKEND] 本次没有可信本地库存，跳过空库存上传\n");
							}
							door_state = DoorState::CLOSED;
							closing_confirm_count = 0;
							no_event_last_gray_small.release();
							no_event_stable_count = 0;
							last_no_event_upload_ms = 0;
						}
					} else {
						// 暗帧误判恢复：只有真的已有 Track 时才丢弃并标记 ambiguous。
						door_state = DoorState::OPEN;
						closing_confirm_count = 0;
						session.resume_after_false_closing();
					}
				} else {  // DoorState::OPEN
					if (high_confidence_closing) {
						door_state = DoorState::CLOSING;
						closing_confirm_count = 1;
						session.begin_closing_guard();
					} else {
						if (mean_y >= fridge::OPEN_REFERENCE_UPDATE_THRESHOLD &&
						    dark_ratio <= fridge::OPEN_DARK_PIXEL_RATIO_LIMIT) {
							recent_open_brightness.push_back(mean_y);
							while ((int)recent_open_brightness.size() >
							       fridge::OPEN_REFERENCE_WINDOW_SIZE) {
								recent_open_brightness.pop_front();
							}
							open_brightness_reference = median_brightness(recent_open_brightness);
						}
						run_yolo_this_frame = true;
					}
				}
			}

			if (!run_yolo_this_frame) {
				memcpy(data, frame.data, width * height * 3);
				RK_MPI_VENC_SendFrame(0, &h264_frame, -1);
				s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, -1);
				if (s32Ret == RK_SUCCESS) {
					if (g_rtsplive && g_rtsp_session) {
						void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
						rtsp_tx_video(g_rtsp_session, (uint8_t *)pData,
						              stFrame.pstPack->u32Len, stFrame.pstPack->u64PTS);
						rtsp_do_event(g_rtsplive);
					}
					RK_MPI_VENC_ReleaseStream(0, &stFrame);
				}
				RK_MPI_VI_ReleaseChnFrame(0, 0, &stViFrame);
				continue;
			}

			// 摄像头物理倒装, 软件翻转 180°
			cv::flip(frame, frame, -1);

			// ============================================================
			//  YOLO 推理
			// ============================================================
			cv::Mat letterboxImage = letterbox(frame);
			if (fridge::INFER_INPUT_BGR2RGB) {
				cv::cvtColor(letterboxImage, letterboxImage, cv::COLOR_BGR2RGB);
			}
			memcpy(rknn_app_ctx.input_mems[0]->virt_addr, letterboxImage.data,
			       model_width*model_height*3);
			inference_yolov5n_model(&rknn_app_ctx, &od_results);

			// ============================================================
			//  坐标映射：letterbox → 原图坐标
			// ============================================================
			std::vector<fridge::Detection> detections;
			std::vector<fridge::Detection> hand_dets_for_display;
			std::vector<fridge::BBox> hand_boxes;
			bool has_strong_hand = false;
			detections.reserve(od_results.count);
			hand_dets_for_display.reserve(od_results.count);

			for (int i = 0; i < od_results.count; i++) {
				const object_detect_result& r = od_results.results[i];
				int x1 = r.box.left;
				int y1 = r.box.top;
				int x2 = r.box.right;
				int y2 = r.box.bottom;
				mapCoordinates(&x1, &y1);
				mapCoordinates(&x2, &y2);

				fridge::Detection d;
				d.box = fridge::BBox((float)x1, (float)y1, (float)x2, (float)y2);
				d.score = r.prop;
				d.cls_id = r.cls_id;
				detections.push_back(d);

				if (fridge::is_hand(d.cls_id)) {
					if (d.score >= fridge::HAND_CONTEXT_SCORE_THRESH) {
						hand_boxes.push_back(d.box);
						hand_dets_for_display.push_back(d);
					}
					if (is_strong_hand_for_osd(d)) {
						has_strong_hand = true;
					}
				}
			}

			// ============================================================
			//  获取时间戳（手检测和事件上报都需要）
			// ============================================================
			RK_U64 now_us = TEST_COMM_GetNowUs();
			long long now_ms = (long long)(now_us / 1000);

			g_frame_id++;

			// 只保留物品检测给业务层（手框单独传入）。
			std::vector<fridge::Detection> food_dets;
			for (const auto& d : detections) {
				if (fridge::is_food(d.cls_id) && d.score >= fridge::SNAPSHOT_MIN_SCORE) {
					food_dets.push_back(d);
				}
			}

			// 新业务层自己维护“连续无手稳定 N 帧”的快照缓冲，并在有手时
			// 更新轻量 OperationTrack；旧 ByteTrack 不参与本程序运行。
			fridge::FrameProcessResult frame_result =
				session.process_frame(food_dets, hand_boxes, g_frame_id, now_ms);

			if (frame_result.stable_snapshot_generated) {
				const fridge::SettlementResult& res = frame_result.settlement;

				// 处理快照对比产生的事件：先打印终端日志；只有真正出入库/整理
				// 才上报云端，遮挡和露出仅更新本地库存状态。
				for (const auto& ev : res.events) {
					print_inventory_event(ev);
					if (ev.kind == fridge::EventKind::OCCLUDED ||
					    ev.kind == fridge::EventKind::REVEALED) {
						continue;
					}

					fridge::UploadJob job;
					switch (ev.kind) {
						case fridge::EventKind::IN:    job.kind = fridge::UploadKind::ITEM_IN;    break;
						case fridge::EventKind::OUT:   job.kind = fridge::UploadKind::ITEM_OUT;   break;
						case fridge::EventKind::MOVED: job.kind = fridge::UploadKind::ITEM_MOVED; break;
						default: break;  // 上面已 continue；保留给编译器完整分支。
					}
					job.local_track_id = ev.item_id;
					job.category       = fridge::cls_id_to_chinese(ev.cls_id);  // 中文细粒度
					job.confidence     = ev.score;
					job.timestamp_ms   = now_ms;

					// 放入(IN) 带物品框截图
					if (ev.kind == fridge::EventKind::IN) {
						int rx1 = std::max(0, (int)ev.box.x1 - 8);
						int ry1 = std::max(0, (int)ev.box.y1 - 8);
						int rx2 = std::min(width, (int)ev.box.x2 + 8);
						int ry2 = std::min(height, (int)ev.box.y2 + 8);
						if (rx2 > rx1 && ry2 > ry1) {
							cv::Rect roi(rx1, ry1, rx2 - rx1, ry2 - ry1);
							cv::Mat crop = frame(roi).clone();
							cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
							std::vector<int> enc_param = {cv::IMWRITE_JPEG_QUALITY, 80};
							std::vector<unsigned char> jpeg;
							if (cv::imencode(".jpg", crop, jpeg, enc_param)) {
								job.jpeg = std::move(jpeg);
								job.x1 = rx1; job.y1 = ry1;
								job.x2 = rx2; job.y2 = ry2;

								// 本地落盘
								char fname[160];
								snprintf(fname, sizeof(fname),
								         "./captures/item%d_%s_%lld.jpg",
								         ev.item_id, coco_cls_to_name(ev.cls_id),
								         (long long)now_ms);
								FILE* fp = fopen(fname, "wb");
								if (fp) {
									fwrite(job.jpeg.data(), 1, job.jpeg.size(), fp);
									fclose(fp);
									printf("\033[1;36m[CAPTURE]\033[0m 放入截图: %s (%zu 字节)\n",
									       fname, job.jpeg.size());
								}
							}
						}
					} else if (ev.kind == fridge::EventKind::MOVED) {
						job.x1 = (int)ev.after_box.x1;
						job.y1 = (int)ev.after_box.y1;
						job.x2 = (int)ev.after_box.x2;
						job.y2 = (int)ev.after_box.y2;
						job.has_before_bbox = true;
						job.before_x1 = (int)ev.before_box.x1;
						job.before_y1 = (int)ev.before_box.y1;
						job.before_x2 = (int)ev.before_box.x2;
						job.before_y2 = (int)ev.before_box.y2;

						int rx1 = std::max(0, (int)ev.after_box.x1 - 8);
						int ry1 = std::max(0, (int)ev.after_box.y1 - 8);
						int rx2 = std::min(width, (int)ev.after_box.x2 + 8);
						int ry2 = std::min(height, (int)ev.after_box.y2 + 8);
						if (rx2 > rx1 && ry2 > ry1) {
							cv::Rect roi(rx1, ry1, rx2 - rx1, ry2 - ry1);
							cv::Mat crop = frame(roi).clone();
							cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
							std::vector<int> enc_param = {cv::IMWRITE_JPEG_QUALITY, 80};
							std::vector<unsigned char> jpeg;
							if (cv::imencode(".jpg", crop, jpeg, enc_param)) {
								job.jpeg = std::move(jpeg);

								char fname[176];
								snprintf(fname, sizeof(fname),
								         "./captures/item%d_moved_after_crop_%lld.jpg",
								         ev.item_id, (long long)now_ms);
								FILE* fp = fopen(fname, "wb");
								if (fp) {
									fwrite(job.jpeg.data(), 1, job.jpeg.size(), fp);
									fclose(fp);
									printf("\033[1;36m[CAPTURE]\033[0m 整理后框截图: %s (%zu 字节)\n",
									       fname, job.jpeg.size());
								}
							}
						}
					} else {
						job.x1 = (int)ev.box.x1;
						job.y1 = (int)ev.box.y1;
						job.x2 = (int)ev.box.x2;
						job.y2 = (int)ev.box.y2;
					}
					if (cloud_enabled) cloud.enqueue(job);
				}

				if (!res.events.empty()) {
					no_event_last_gray_small.release();
					no_event_stable_count = 0;
					last_no_event_upload_ms = 0;
				} else if (cloud_enabled && !session.hand_present() && session.ready()) {
					double pixel_diff =
						update_no_event_pixel_diff(frame, no_event_last_gray_small);
					if (pixel_diff <= NO_EVENT_PIXEL_DIFF_THRESH) {
						no_event_stable_count++;
					} else {
						no_event_stable_count = 0;
					}

					bool cooldown_ok =
						last_no_event_upload_ms == 0 ||
						now_ms - last_no_event_upload_ms >= NO_EVENT_SNAPSHOT_COOLDOWN_MS;
					if (no_event_stable_count >= NO_EVENT_STABLE_CONFIRM && cooldown_ok) {
						cv::Mat full_image = frame.clone();
						cv::cvtColor(full_image, full_image, cv::COLOR_BGR2RGB);
						std::vector<int> enc_param = {cv::IMWRITE_JPEG_QUALITY, 80};
						std::vector<unsigned char> jpeg;
						if (cv::imencode(".jpg", full_image, jpeg, enc_param)) {
							fridge::UploadJob job;
							job.kind = fridge::UploadKind::NO_EVENT_SNAPSHOT;
							job.timestamp_ms = now_ms;
							job.pixel_diff = (float)pixel_diff;
							job.snapshot_reason = "stable_no_event";
							job.jpeg = std::move(jpeg);

							char fname[160];
							snprintf(fname, sizeof(fname),
							         "./captures/no_event_snapshot_%lld.jpg",
							         (long long)now_ms);
							FILE* fp = fopen(fname, "wb");
							if (fp) {
								fwrite(job.jpeg.data(), 1, job.jpeg.size(), fp);
								fclose(fp);
							}

							cloud.enqueue(job);
							last_no_event_upload_ms = now_ms;
							printf("\033[1;36m[CAPTURE]\033[0m 无事件整体图: %s "
							       "(diff=%.2f, stable=%d, %zu 字节)\n",
							       fname, pixel_diff, no_event_stable_count,
							       job.jpeg.size());
						}
					}
				}

				// 正式快照只要成功提交，就打印当前库存；即使本轮只是确认“无变化”，
				// 终端也能看到它确实已经结算完成。
				if (res.committed) {
					printf("\n\033[1;36m[INVENTORY]\033[0m 快照结算已提交:\n");
					session.print_inventory();
				}
			}

			// ============================================================
			//  画面绘制：bbox + 系统状态
			// ============================================================
			for (const auto& d : detections) {
				if (fridge::is_hand(d.cls_id)) continue;
				bool should_display_raw =
					d.score >= fridge::SNAPSHOT_MIN_SCORE;
				if (!should_display_raw) continue;

				int x1 = (int)d.box.x1;
				int y1 = (int)d.box.y1;
				int x2 = (int)d.box.x2;
				int y2 = (int)d.box.y2;
				cv::Scalar color = display_color_for_class(d.cls_id);
				cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 1);

				char label[64];
				snprintf(label, sizeof(label), "%s %.0f%%",
				         coco_cls_to_name(d.cls_id), d.score * 100);
				cv::putText(frame, label, cv::Point(x1, y1 - 6),
				            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
			}

			// 屏幕左上角：系统状态
			{
				bool has_hand_now = session.hand_present();
				cv::Scalar osd_color = has_hand_now
					? cv::Scalar(0, 0, 255)   // 红 = 手在
					: cv::Scalar(0, 255, 0);  // 绿 = 无手
				const fridge::InventoryDB& inv = session.inventory();
				char osd[128];
				snprintf(osd, sizeof(osd), "%s rawH=%zu strongH=%d | V=%zu O=%zu",
				         has_hand_now ? "HAND" : "STABLE",
				         hand_dets_for_display.size(),
					         has_strong_hand ? 1 : 0,
				         inv.count_by_status(fridge::ItemStatus::VISIBLE),
				         inv.count_by_status(fridge::ItemStatus::OCCLUDED));
				cv::putText(frame, osd, cv::Point(8, 22),
				            cv::FONT_HERSHEY_SIMPLEX, 0.6, osd_color, 2);
			}
		}
		memcpy(data, frame.data, width * height * 3);

		// encode H264
		RK_MPI_VENC_SendFrame(0, &h264_frame,-1);

		// rtsp
		s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, -1);
		if(s32Ret == RK_SUCCESS)
		{
			if(g_rtsplive && g_rtsp_session)
			{
				void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
				rtsp_tx_video(g_rtsp_session, (uint8_t *)pData, stFrame.pstPack->u32Len,
				              stFrame.pstPack->u64PTS);
				rtsp_do_event(g_rtsplive);
			}
		}

		// release frame
		s32Ret = RK_MPI_VI_ReleaseChnFrame(0, 0, &stViFrame);
		if (s32Ret != RK_SUCCESS) {
			RK_LOGE("RK_MPI_VI_ReleaseChnFrame fail %x", s32Ret);
		}
		s32Ret = RK_MPI_VENC_ReleaseStream(0, &stFrame);
		if (s32Ret != RK_SUCCESS) {
			RK_LOGE("RK_MPI_VENC_ReleaseStream fail %x", s32Ret);
		}
		memset(text,0,8);
	}

	// Destory MB
	RK_MPI_MB_ReleaseMB(src_Blk);
	// Destory Pool
	RK_MPI_MB_DestroyPool(src_Pool);

	cloud.stop();

	RK_MPI_VI_DisableChn(0, 0);
	RK_MPI_VI_DisableDev(0);

	SAMPLE_COMM_ISP_Stop(0);

	RK_MPI_VENC_StopRecvFrame(0);
	RK_MPI_VENC_DestroyChn(0);

	free(stFrame.pstPack);

	if (g_rtsplive)
		rtsp_del_demo(g_rtsplive);

	RK_MPI_SYS_Exit();

    release_yolov5n_model(&rknn_app_ctx);
	deinit_post_process();

	return 0;
}
