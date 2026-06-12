// ============================================================================
//  main.cc
//  冰箱视觉系统 — 新业务流程7 主循环
//
//  每帧流水线：
//    1. 摄像头采集 + YOLO推理 + 坐标映射
//    2. ByteTrack-Lite 每帧更新
//    3. OperationContext 收集手/HELD/ByteTrack移动证据
//    4. 每帧推入 SnapshotBuffer（多帧投票缓冲区，快照含 has_hand 标记）
//    5. 攒满配置帧数 → 生成 Snapshot → 送入 SessionManager 统一裁决
//    6. 事件上报 + 画面绘制 + RTSP推流
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
#include <vector>
#include <thread>

#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "yolov5.h"

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
	printf("[BACKEND] 开门库存获取接口未接入，使用首个无手稳定快照兜底\n");
	return false;
}

static bool is_snapshot_blocking_hand(const fridge::Detection& det) {
	if (!fridge::is_hand(det.cls_id)) return false;
	if (det.score < fridge::HAND_SNAPSHOT_BLOCK_SCORE_THRESH) return false;

	float frame_area = fridge::FRAME_W * fridge::FRAME_H;
	if (frame_area <= 0.0f) return false;

	float area_ratio = det.box.area() / frame_area;
	return area_ratio >= fridge::HAND_SNAPSHOT_BLOCK_MIN_AREA_RATIO;
}

static bool covered_by_track(const fridge::Detection& det,
                             const std::vector<fridge::Track>& tracks) {
	for (const auto& t : tracks) {
		if (t.cls_id != det.cls_id) continue;
		if (fridge::iou(t.box, det.box) >= 0.50f) {
			return true;
		}
	}
	return false;
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
	const char *model_path = "./model/yolov5.rknn";
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
	init_yolov5_model(model_path, &rknn_app_ctx);
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
	fridge::ByteTrackLite tracker;
	fridge::SessionManager session;
	fridge::SnapshotBuffer snap_buffer(fridge::SNAPSHOT_N, fridge::SNAPSHOT_OBJECT_STABLE_RATIO);
	fridge::CloudUploader cloud;
	cloud.start();
	int g_frame_id = 0;

	// ============================================================
	//  开关门检测（全局亮度阈值法）
	// ============================================================
	bool door_open = false;
	const double DOOR_DARK_THRESH = 50.0;
	int dark_streak = 0;
	int bright_streak = 0;
	const int DOOR_CONFIRM = 5;
	int door_session_seq = 0;
	char door_session_id[96] = "";

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
			//  开关门检测：YUV420SP 的前 width*height 字节就是 Y(亮度)分量
			// ============================================================
			{
				cv::Mat y_plane(height, width, CV_8UC1, vi_data);
				double mean_y = cv::mean(y_plane)[0];

				bool dark_now = (mean_y < DOOR_DARK_THRESH);
				if (dark_now) { dark_streak++; bright_streak = 0; }
				else          { bright_streak++; dark_streak = 0; }

				if (door_open && dark_streak >= DOOR_CONFIRM) {
					door_open = false;
					printf("\n\033[1;33m[DOOR]\033[0m 关门 (亮度=%.0f), 上传库存:\n", mean_y);
					RK_U64 ts = TEST_COMM_GetNowUs() / 1000;
					session.finish_session((long long)ts);
					std::string json = session.inventory().to_json(
						cloud.device_id.c_str(), (long long)ts, door_session_id);
					printf("%s\n", json.c_str());
					cloud.enqueue_inventory_snapshot(json, (long long)ts);
					snap_buffer.reset();
				} else if (!door_open && bright_streak >= DOOR_CONFIRM) {
					door_open = true;
					printf("\n\033[1;33m[DOOR]\033[0m 开门 (亮度=%.0f)\n", mean_y);
					RK_U64 ts = TEST_COMM_GetNowUs() / 1000;
					door_session_seq++;
					snprintf(door_session_id, sizeof(door_session_id),
					         "%lld_%d_%s", (long long)ts, door_session_seq,
					         cloud.device_id.c_str());

					session.start_new_session((long long)ts);
					snap_buffer.reset();

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
				}
			}

			// 关门状态下，跳过识别
			if (!door_open) {
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
			inference_yolov5_model(&rknn_app_ctx, &od_results);

			// ============================================================
			//  坐标映射：letterbox → 原图坐标
			// ============================================================
			std::vector<fridge::Detection> detections;
			std::vector<fridge::Detection> hand_dets_for_display;
			std::vector<fridge::BBox> hand_boxes;
			bool has_snapshot_blocking_hand = false;
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
					if (is_snapshot_blocking_hand(d)) {
						has_snapshot_blocking_hand = true;
					}
				}
			}

			// ============================================================
			//  获取时间戳（手检测和事件上报都需要）
			// ============================================================
			RK_U64 now_us = TEST_COMM_GetNowUs();
			long long now_ms = (long long)(now_us / 1000);

			// ============================================================
			//  ByteTrack 每帧更新
			// ============================================================
			g_frame_id++;
			const std::vector<fridge::Track>& tracks =
				tracker.update(detections, g_frame_id);

			// ============================================================
			//  OperationContext：每帧记录手、HELD代理、ByteTrack移动证据
			//  注意：这里不直接产生库存事件，库存只由后面的稳定快照diff裁决。
			// ============================================================
			session.update_hand(hand_boxes, tracks, g_frame_id, now_ms);

			// ============================================================
			//  快照缓冲：手证据和快照阻塞分开。
			//  OperationContext 可以使用较敏感的 hand_boxes；
			//  快照阻塞只接受更高置信度/足够面积的手，避免弱误检长期卡住库存。
			// ============================================================
			bool has_hand = has_snapshot_blocking_hand;

			// 只把食物检测推入快照缓冲区（不含手）
			std::vector<fridge::Detection> food_dets;
			for (const auto& d : detections) {
				if (fridge::is_food(d.cls_id) && d.score >= fridge::SNAPSHOT_MIN_SCORE) {
					food_dets.push_back(d);
				}
			}
			snap_buffer.push(food_dets, g_frame_id, has_hand);

			if (snap_buffer.full()) {
				fridge::Snapshot snap = snap_buffer.take_snapshot();
				fridge::SettlementResult res = session.push_snapshot(snap, frame);

				// 处理快照对比产生的事件：上报云端
				for (const auto& ev : res.events) {
					fridge::UploadJob job;
					switch (ev.kind) {
						case fridge::EventKind::IN:    job.kind = fridge::UploadKind::ITEM_IN;    break;
						case fridge::EventKind::OUT:   job.kind = fridge::UploadKind::ITEM_OUT;   break;
						case fridge::EventKind::MOVED: job.kind = fridge::UploadKind::ITEM_MOVED; break;
					}
					job.local_track_id = ev.item_id;
					job.category       = fridge::coarse_category(ev.cls_id);
					job.confidence     = ev.score;
					job.timestamp_ms   = now_ms;

					// 放入(IN) 带截图
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
								job.x = rx1; job.y = ry1;
								job.w = rx2 - rx1; job.h = ry2 - ry1;

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
					} else {
						job.x = (int)ev.box.x1;
						job.y = (int)ev.box.y1;
						job.w = (int)(ev.box.x2 - ev.box.x1);
						job.h = (int)(ev.box.y2 - ev.box.y1);
					}
					cloud.enqueue(job);
				}

				if (res.happened && !res.events.empty()) {
					printf("\n\033[1;36m[INVENTORY]\033[0m 快照对比后库存:\n");
					session.inventory().print("  ");
				}
			}

			// ============================================================
			//  画面绘制：bbox + 系统状态
			// ============================================================
			for (const auto& d : detections) {
				bool should_display_raw =
					fridge::is_hand(d.cls_id)
						? d.score >= fridge::HAND_CONTEXT_SCORE_THRESH
						: d.score >= fridge::SNAPSHOT_MIN_SCORE;
				if (!should_display_raw || covered_by_track(d, tracks)) continue;

				int x1 = (int)d.box.x1;
				int y1 = (int)d.box.y1;
				int x2 = (int)d.box.x2;
				int y2 = (int)d.box.y2;
				cv::Scalar color = fridge::is_hand(d.cls_id)
					? cv::Scalar(0, 0, 255)
					: cv::Scalar(0, 255, 0);
				cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 1);

				char label[64];
				snprintf(label, sizeof(label), "%s %.0f%%",
				         coco_cls_to_name(d.cls_id), d.score * 100);
				cv::putText(frame, label, cv::Point(x1, y1 - 6),
				            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
			}

			for (const auto& t : tracks) {
				int x1 = (int)t.box.x1;
				int y1 = (int)t.box.y1;
				int x2 = (int)t.box.x2;
				int y2 = (int)t.box.y2;

				cv::Scalar color = fridge::is_hand(t.cls_id)
					? cv::Scalar(0, 0, 255)      // 红 = 手
					: cv::Scalar(0, 255, 0);     // 绿 = 物品

				cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

				char label[64];
				snprintf(label, sizeof(label), "#%d %s %.0f%%",
				         t.track_id, coco_cls_to_name(t.cls_id), t.score * 100);
				cv::putText(frame, label, cv::Point(x1, y1 - 6),
				            cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
			}

			// 屏幕左上角：系统状态
			{
				bool has_hand_now = session.hand_present();
				cv::Scalar osd_color = has_hand_now
					? cv::Scalar(0, 0, 255)   // 红 = 手在
					: cv::Scalar(0, 255, 0);  // 绿 = 无手
				const fridge::InventoryDB& inv = session.inventory();
				char osd[128];
				snprintf(osd, sizeof(osd), "%s rawH=%zu blockH=%d | V=%zu O=%zu OUT=%zu",
				         has_hand_now ? "HAND" : "STABLE",
				         hand_dets_for_display.size(),
				         has_snapshot_blocking_hand ? 1 : 0,
				         inv.count_by_status(fridge::ItemStatus::VISIBLE),
				         inv.count_by_status(fridge::ItemStatus::OCCLUDED),
				         inv.count_by_status(fridge::ItemStatus::OUT));
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

    release_yolov5_model(&rknn_app_ctx);
	deinit_post_process();

	return 0;
}
