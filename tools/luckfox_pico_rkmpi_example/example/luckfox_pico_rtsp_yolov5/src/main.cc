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

#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "yolov5.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// ===== 冰箱视觉系统模块 =====
#include "fridge_config.h"     // 全局配置（类别 id、阈值）
#include "geometry.h"          // BBox 几何工具
#include "tracker.h"           // ByteTrack-Lite 跟踪器
#include "snapshot.h"          // 稳态快照 + diff
#include "stability.h"         // 稳态监控状态机
#include "session.h"           // 会话管理器（编排 stability + diff + inventory）
#include "inventory.h"         // 本地工作库存
// hand_state.h 暂时保留备用（"主动确认/撤销"等扩展），当前业务路径不再调用

#define DISP_WIDTH  720
#define DISP_HEIGHT 480

// disp size
int width    = DISP_WIDTH;
int height   = DISP_HEIGHT;

// model size
int model_width = 640;
int model_height = 640;	
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
	// 画布尺寸必须等于模型输入(model_width x model_height). 之前硬编码 640 是 bug:
	// 当 model_width=320 时, memcpy 只会取画布左上 320x320 区域, 等于让模型只看到
	// 摄像头画面缩到画布左上的一小块, 周围全是底色, 自然识别不出.
	// padding 颜色也由原本的纯黑(0,0,0) 改回 yolov5 惯例的 (114,114,114).
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

	// 把映射后的坐标 clamp 到原图范围内.
	// 没有 clamp 时, 落在 letterbox padding 区域里的检测框会得到负坐标
	// 或超出 (width, height) 的坐标, 导致后续画框/库存匹配出现 (-119, -67) 这种异常值.
	if (rx < 0)      rx = 0;
	if (ry < 0)      ry = 0;
	if (rx > width)  rx = width;
	if (ry > height) ry = height;

	*x = rx;
	*y = ry;
}


int main(int argc, char *argv[]) {
  system("RkLunch-stop.sh");
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
	//PoolCfg.bPreAlloc = RK_FALSE;
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
	//hdr_mode = RK_AIQ_WORKING_MODE_ISP_HDR2;
	SAMPLE_COMM_ISP_Init(0, hdr_mode, multi_sensor, iq_dir);
	SAMPLE_COMM_ISP_Run(0);
	// 摄像头物理倒装, 需要把画面旋转 180°.
	//
	// 之前用过 SAMPLE_COMM_ISP_SetMirrorFlip(0, 1, 1) 走 ISP/sensor 通路, 但是:
	//   1. 不是所有 sensor 都开放 mirror/flip 寄存器, 调了不报错也不生效;
	//   2. 即便生效, 也只是改 sensor 输出, 排查问题时不直观.
	// 改成软件翻转最稳: 在拿到 BGR 帧后做 cv::flip(frame, frame, -1).
	// 一帧 720x480 BGR 大约 1ms, 没什么开销, 而且 YOLO 推理跟显示用的是同一份
	// 画面, 翻转一次全部生效 (因为 letterbox 输入也来自 frame).
	// 实际翻转见下面 while 循环里 cv::cvtColor 之后那一行.

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
	//  冰箱视觉系统：初始化跟踪器 + 会话管理器
	//  会话管理器内部封装了：稳态监控 + 快照 diff + 工作库存
	// ============================================================
	fridge::ByteTrackLite tracker;
	fridge::SessionManager session;
	int g_frame_id = 0;

	// ============================================================
	//  开关门检测（全局亮度阈值法）
	//   冰箱关门 → 画面变黑 → 整帧平均灰度暴跌
	//   平均灰度 < DOOR_DARK_THRESH 视为关门
	//   关门瞬间把当前库存打包上传后台（这里先打印 JSON，后续接 HTTP/MQTT）
	// ============================================================
	bool door_open = true;        // 假设启动时门是开的
	const double DOOR_DARK_THRESH = 50.0;   // 平均灰度阈值，按实际冰箱调
	int dark_streak = 0;          // 连续多少帧暗
	int bright_streak = 0;        // 连续多少帧亮
	const int DOOR_CONFIRM = 5;   // 连续 5 帧确认开/关，防抖

  	while(1)
	{	
		// get vi frame
		h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
		h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs(); 
		s32Ret = RK_MPI_VI_GetChnFrame(0, 0, &stViFrame, -1);
		if(s32Ret == RK_SUCCESS)
		{
			void *vi_data = RK_MPI_MB_Handle2VirAddr(stViFrame.stVFrame.pMbBlk);	

			// VI 给的是 YUV420SP, 直接转换到 frame 的 buffer (frame 已绑定到 data 上).
			// 之前还多了一个临时 bgr Mat + resize 一次, 但 bgr 和 frame 实际是同一块
			// 内存 + 同样尺寸, resize 是冗余且 src=dst 时 OpenCV 行为未定义, 现已去掉.
			cv::Mat yuv420sp(height + height / 2, width, CV_8UC1, vi_data);
			cv::cvtColor(yuv420sp, frame, cv::COLOR_YUV420sp2BGR);

			// ============================================================
			//  开关门检测：YUV420SP 的前 width*height 字节就是 Y(亮度)分量
			//  直接对 Y 求平均，零额外开销（不用转灰度）
			// ============================================================
			{
				cv::Mat y_plane(height, width, CV_8UC1, vi_data);
				double mean_y = cv::mean(y_plane)[0];

				bool dark_now = (mean_y < DOOR_DARK_THRESH);
				if (dark_now) { dark_streak++; bright_streak = 0; }
				else          { bright_streak++; dark_streak = 0; }

				if (door_open && dark_streak >= DOOR_CONFIRM) {
					// 开 → 关
					door_open = false;
					printf("\n\033[1;33m[DOOR]\033[0m 关门 (亮度=%.0f), 上传库存:\n", mean_y);
					RK_U64 ts = TEST_COMM_GetNowUs() / 1000;  // 毫秒
					std::string json = session.inventory().to_json("luckfox", (long long)ts);
					printf("%s\n", json.c_str());
				} else if (!door_open && bright_streak >= DOOR_CONFIRM) {
					// 关 → 开
					door_open = true;
					printf("\n\033[1;33m[DOOR]\033[0m 开门 (亮度=%.0f)\n", mean_y);
				}
			}

			// 关门状态下，画面是黑的，不跑 YOLO（省算力，也避免黑画面误检）
			if (!door_open) {
				// 直接编码黑画面推流，跳过识别
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
				continue;   // 跳过本帧后续所有识别逻辑
			}

			// 摄像头物理倒装, 这里直接对原图做 180° 旋转 (= 上下+左右各翻一次).
			// flipCode = -1 表示同时翻 X 和 Y 轴.
			// 必须在 letterbox/inference 之前做, 这样 YOLO 看到的是正向画面,
			// 检测框坐标也就直接对齐到旋转后的 frame 上, 不需要再二次映射.
			cv::flip(frame, frame, -1);

			//letterbox
			cv::Mat letterboxImage = letterbox(frame);	
			memcpy(rknn_app_ctx.input_mems[0]->virt_addr, letterboxImage.data, model_width*model_height*3);		
			inference_yolov5_model(&rknn_app_ctx, &od_results);

			// ============================================================
			//  Step A: 把 RKNN 的 detection 结果转成跟踪器需要的格式
			//          同时把 letterbox 坐标映射回原图坐标
			// ============================================================
			std::vector<fridge::Detection> detections;
			detections.reserve(od_results.count);
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
			}

			// ============================================================
			//  Step B: 跑 ByteTrack-Lite，得到带 track_id 的目标列表
			// ============================================================
			g_frame_id++;
			const std::vector<fridge::Track>& tracks =
				tracker.update(detections, g_frame_id);

			// 注: 之前这里有"每 10 帧打印一次 track 列表"的调试日志,
			//     已删除. 检测框直接看 RTSP 视频画面更直观,
			//     终端只保留 SESSION/EVENT/Inventory 这种业务输出, 信号噪声比更高.

			// ============================================================
			//  Step C: 跑会话管理器
			//   - 内部判稳/扰动；
			//   - STABLE→DISTURBED 时冻结 before；
			//   - DISTURBED→STABLE 时取 after，diff，应用到本地工作库存。
			//   多数帧没有事件产生，只在稳态切换那一刻才有。
			// ============================================================
			fridge::SettlementResult settlement = session.update(tracks, g_frame_id);
			(void)settlement;   // 当前还不用上报后台，先把变量留个口子
			//  注意：库存的打印 / 事件的打印都已经在 session.cc / inventory.cc
			//  内部完成（带 [SESSION] / [EVENT] / [Inventory] 前缀），这里
			//  不再重复打印。

			// ============================================================
			//  Step D: 在画面上画 track（含 track_id），方便观察
			//  外加一个右上角的"系统状态"指示（STABLE / DISTURBED）
			// ============================================================
			for (const auto& t : tracks) {
				int x1 = (int)t.box.x1;
				int y1 = (int)t.box.y1;
				int x2 = (int)t.box.x2;
				int y2 = (int)t.box.y2;

				// 手用红框，物品用绿框
				cv::Scalar color = fridge::is_hand(t.cls_id)
					? cv::Scalar(0, 0, 255)      // BGR: 红
					: cv::Scalar(0, 255, 0);     // BGR: 绿

				cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2),
				              color, 2);

				char text[64];
				snprintf(text, sizeof(text), "#%d %s %.0f%%",
				         t.track_id, coco_cls_to_name(t.cls_id),
				         t.score * 100);
				cv::putText(frame, text, cv::Point(x1, y1 - 6),
				            cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
			}

			// ============================================================
			//  Step E: 屏幕左上角叠加系统状态指示
			//   - STABLE   绿色 — 现在画面安静，库存可信
			//   - DISTURBED 红色 — 用户正在操作，库存暂停更新
			//  让答辩 / 调试时一眼能看出"现在系统觉得用户在不在动"
			// ============================================================
			{
				fridge::SystemState ss = session.system_state();
				bool stable = (ss == fridge::SystemState::STABLE);
				cv::Scalar osd_color = stable
					? cv::Scalar(0, 255, 0)
					: cv::Scalar(0, 0, 255);
				char osd[96];
				snprintf(osd, sizeof(osd), "%s | inv=%zu(v=%zu/o=%zu)",
				         fridge::system_state_to_str(ss),
				         session.inventory().size(),
				         session.inventory().count_visible(),
				         session.inventory().count_occluded());
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
				//printf("len = %d PTS = %d \n",stFrame.pstPack->u32Len, stFrame.pstPack->u64PTS);	
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
	
	RK_MPI_VI_DisableChn(0, 0);
	RK_MPI_VI_DisableDev(0);

	SAMPLE_COMM_ISP_Stop(0);
	
	RK_MPI_VENC_StopRecvFrame(0);
	RK_MPI_VENC_DestroyChn(0);

	free(stFrame.pstPack);

	if (g_rtsplive)
		rtsp_del_demo(g_rtsplive);
	
	RK_MPI_SYS_Exit();

	// Release rknn model
    release_yolov5_model(&rknn_app_ctx);		
	deinit_post_process();
	
	return 0;
}
