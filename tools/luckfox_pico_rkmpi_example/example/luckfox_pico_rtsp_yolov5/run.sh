#!/bin/bash
# 启动 yolov5 推理程序, 自动过滤 RTSP 库的 DEBUG/INFO 噪音日志
# 用法: cd /root/luckfox_pico_rtsp_yolov5_demo && bash run.sh

./luckfox_pico_rtsp_yolov5 2>&1 | grep --line-buffered -v "^\[DEBUG\]\|^\[INFO \]\|rtsp_msg\|rtsp_demo\|rtsp_handle\|rtsp_recv\|rtsp_del_client"
