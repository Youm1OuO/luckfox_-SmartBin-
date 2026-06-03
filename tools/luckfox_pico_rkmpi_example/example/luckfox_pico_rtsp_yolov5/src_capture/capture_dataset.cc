// ============================================================================
//  capture_dataset.cc
//  摄像头定时截图采集工具 — 用于制作 YOLO 训练数据集
//
//  用法: ./capture_dataset [间隔秒] [总数]
//    间隔秒: 每两张截图之间的间隔，默认 2
//    总数:   总共截多少张，默认 100
//
//  示例:
//    ./capture_dataset          # 每 2 秒截一张，共 100 张
//    ./capture_dataset 3 50     # 每 3 秒截一张，共 50 张
//
//  图片保存到 ./dataset_raw/frame_NNNN.jpg
//  采集完后用 PC 端的 tools/make_dataset.py 整理成 YOLO 数据集
//
//  与 main.cc 的区别：
//    - 不加载 RKNN 模型（纯截图，不跑推理）
//    - 不启动 RTSP 推流
//    - 不启动 cloud_uploader / session / tracker
//    - 极轻量，只初始化摄像头 + 保存 JPEG
// ============================================================================
#include <dirent.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>

#include "luckfox_mpi.h"
#include "sample_comm.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// 画面尺寸（与 main.cc 一致）
#define CAP_WIDTH  1920
#define CAP_HEIGHT 1080

// 优雅退出
static volatile sig_atomic_t g_running = 1;
static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

int main(int argc, char *argv[]) {
    // ---------------------------------------------------------------
    //  解析参数
    // ---------------------------------------------------------------
    int interval_sec = 2;    // 默认 2 秒间隔
    int total_count  = 100;  // 默认 100 张

    if (argc >= 2) interval_sec = atoi(argv[1]);
    if (argc >= 3) total_count  = atoi(argv[2]);

    if (interval_sec <= 0) interval_sec = 2;
    if (total_count <= 0)  total_count = 100;

    printf("============================================================\n");
    printf("  冰箱数据集截图工具\n");
    printf("  间隔: %d 秒, 总数: %d 张\n", interval_sec, total_count);
    printf("  保存目录: ./dataset_raw/\n");
    printf("  按 Ctrl+C 可提前退出\n");
    printf("============================================================\n");

    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 创建保存目录
    mkdir("./dataset_raw", 0755);

    // ---------------------------------------------------------------
    //  检测已有图片数量，编号从最后一张往后接续（不覆盖之前的）
    //  这样可以分多次采集：每次运行自动接着上次继续编号
    // ---------------------------------------------------------------
    int start_index = 1;
    {
        // 扫描 dataset_raw/ 下已有的 frame_NNNN.jpg，找到最大编号
        DIR *dir = opendir("./dataset_raw");
        if (dir) {
            struct dirent *ent;
            int max_idx = 0;
            while ((ent = readdir(dir)) != NULL) {
                int idx = 0;
                if (sscanf(ent->d_name, "frame_%d.jpg", &idx) == 1) {
                    if (idx > max_idx) max_idx = idx;
                }
            }
            closedir(dir);
            if (max_idx > 0) {
                start_index = max_idx + 1;
                printf("  检测到已有 %d 张图片，编号从 %d 开始接续\n",
                       max_idx, start_index);
            }
        }
    }

    // ---------------------------------------------------------------
    //  初始化 ISP（摄像头）
    // ---------------------------------------------------------------
    RK_BOOL multi_sensor = RK_FALSE;
    const char *iq_dir = "/etc/iqfiles";
    rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
    SAMPLE_COMM_ISP_Init(0, hdr_mode, multi_sensor, iq_dir);
    SAMPLE_COMM_ISP_Run(0);

    // ---------------------------------------------------------------
    //  初始化 RK MPI 系统
    // ---------------------------------------------------------------
    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        printf("RK_MPI_SYS_Init 失败!\n");
        return -1;
    }

    // ---------------------------------------------------------------
    //  初始化 VI（视频输入）
    // ---------------------------------------------------------------
    int width  = CAP_WIDTH;
    int height = CAP_HEIGHT;

    vi_dev_init();
    vi_chn_init(0, width, height);

    // ---------------------------------------------------------------
    //  主循环：定时截图
    // ---------------------------------------------------------------
    int captured = 0;
    int skip_frames = 10;  // 前 10 帧跳过，等摄像头稳定

    printf("\n开始采集...\n");

    while (g_running && captured < total_count) {
        // 取 VI 帧
        VIDEO_FRAME_INFO_S stViFrame;
        RK_S32 s32Ret = RK_MPI_VI_GetChnFrame(0, 0, &stViFrame, -1);
        if (s32Ret != RK_SUCCESS) {
            usleep(10000);
            continue;
        }

        // 跳过前几帧（摄像头刚启动时曝光/白平衡未稳定）
        if (skip_frames > 0) {
            skip_frames--;
            RK_MPI_VI_ReleaseChnFrame(0, 0, &stViFrame);
            continue;
        }

        void *vi_data = RK_MPI_MB_Handle2VirAddr(stViFrame.stVFrame.pMbBlk);

        // YUV → BGR
        cv::Mat yuv420sp(height + height / 2, width, CV_8UC1, vi_data);
        cv::Mat frame;
        cv::cvtColor(yuv420sp, frame, cv::COLOR_YUV420sp2BGR);

        // 释放 VI 帧（尽早释放，避免占住缓冲区）
        RK_MPI_VI_ReleaseChnFrame(0, 0, &stViFrame);

        // 180° 旋转（与 main.cc 一致：摄像头物理倒装）
        cv::flip(frame, frame, -1);

        // 保存 JPEG
        captured++;
        int file_idx = start_index + captured - 1;
        char fname[128];
        snprintf(fname, sizeof(fname), "./dataset_raw/frame_%04d.jpg", file_idx);

        std::vector<int> enc_param = {cv::IMWRITE_JPEG_QUALITY, 95};
        if (cv::imwrite(fname, frame, enc_param)) {
            printf("  [%d/%d] 已保存: %s\n", captured, total_count, fname);
        } else {
            printf("  [%d/%d] 保存失败: %s\n", captured, total_count, fname);
        }

        // 等待指定间隔
        if (g_running && captured < total_count) {
            // 分段 sleep，以便及时响应 Ctrl+C
            int remain_us = interval_sec * 1000000;
            while (remain_us > 0 && g_running) {
                int chunk = remain_us > 200000 ? 200000 : remain_us;
                usleep(chunk);
                remain_us -= chunk;
            }
        }
    }

    // ---------------------------------------------------------------
    //  清理
    // ---------------------------------------------------------------
    printf("\n采集完成! 共截取 %d 张图片，保存在 ./dataset_raw/\n", captured);
    printf("下一步:\n");
    printf("  1. 把 dataset_raw/ 拷回 PC: scp -r root@<板子IP>:/root/.../dataset_raw/ ./\n");
    printf("  2. 运行 tools/make_dataset.py 整理成 YOLO 格式\n");
    printf("  3. 用 LabelImg 标注\n");

    RK_MPI_VI_DisableChn(0, 0);
    RK_MPI_VI_DisableDev(0);
    SAMPLE_COMM_ISP_Stop(0);
    RK_MPI_SYS_Exit();

    return 0;
}
