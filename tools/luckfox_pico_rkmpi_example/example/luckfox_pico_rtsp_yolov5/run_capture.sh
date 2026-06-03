#!/bin/bash
# 板子端快速启动截图采集（自动停掉其他占用摄像头的程序）
# 用法: bash run_capture.sh [间隔秒] [总数]

# 停掉可能占用摄像头的其他程序
RkLunch-stop.sh 2>/dev/null

# 运行截图采集
./capture_dataset "$@"
