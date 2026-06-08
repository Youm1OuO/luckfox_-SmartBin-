@echo off
chcp 65001 >nul
title 冰箱数据集采集

echo ========================================
echo   冰箱数据集采集工具
echo ========================================
echo.

REM === 修改下面这一行，改成你 LuckFox 的 IP 地址 ===
set LUCKFOX_IP=192.168.168.100
set RTSP_URL=rtsp://%LUCKFOX_IP%:554/live/0

REM === 创建保存目录 ===
set SAVE_DIR=dataset_raw
if not exist %SAVE_DIR% mkdir %SAVE_DIR%

REM === 自动编号：找到下一个可用的文件夹名 ===
set BATCH=1
:find_batch
if exist %SAVE_DIR%\batch%BATCH% (
    set /a BATCH+=1
    goto find_batch
)
set SAVE_DIR=%SAVE_DIR%\batch%BATCH%
mkdir %SAVE_DIR%

echo   摄像头地址: %RTSP_URL%
echo   照片保存到: %SAVE_DIR%\
echo   计划拍摄: 200 张（约 2 分钟）
echo.
echo   拍的时候请：
echo     - 把东西一个个放进冰箱，每个换 3~5 个角度
echo     - 也拍几张手拿东西的画面
echo     - 按 q 可以提前结束
echo.
echo   按任意键开始拍摄...
pause >nul

echo.
echo   正在拍摄...请开始往冰箱里放东西！
echo.

ffmpeg -i %RTSP_URL% -vf "fps=2" -q:v 2 %SAVE_DIR%\fridge_%%04d.jpg -v quiet -stats -frames:v 200

echo.
echo ========================================
echo   拍摄完成！
echo   照片保存在: %SAVE_DIR%\
echo ========================================
echo.
echo   下一步：运行 "2_整理数据集.py"
echo.
pause
