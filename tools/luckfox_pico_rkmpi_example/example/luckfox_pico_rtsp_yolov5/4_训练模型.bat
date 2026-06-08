@echo off
chcp 65001 >nul
title 训练冰箱识别模型

echo ========================================
echo   训练冰箱识别模型
echo ========================================
echo.

set SCRIPT_DIR=%~dp0
set DATASET_DIR=%SCRIPT_DIR%dataset
set YAML_FILE=%DATASET_DIR%\data.yaml
set LABELS_DIR=%DATASET_DIR%\labels\train

REM 检查是否已标注
if not exist "%YAML_FILE%" (
    echo [错误] 找不到 data.yaml
    echo   请先运行 "2_整理数据集.py"
    echo.
    pause
    exit /b 1
)

REM 检查标注文件数量
set /a LABEL_COUNT=0
for %%f in ("%LABELS_DIR%\*.txt") do set /a LABEL_COUNT+=1

if %LABEL_COUNT% equ 0 (
    echo [错误] 还没有标注文件
    echo   请先运行 "3_打开标注工具.bat" 完成标注
    echo.
    pause
    exit /b 1
)

echo   找到 %LABEL_COUNT% 个标注文件
echo   数据集配置: %YAML_FILE%
echo.

REM 检查 YOLOv5 是否存在
set YOLOV5_DIR=%SCRIPT_DIR%yolov5
if not exist "%YOLOV5_DIR%\train.py" (
    echo   YOLOv5 还没下载，正在下载...
    echo.
    git clone https://github.com/ultralytics/yolov5.git "%YOLOV5_DIR%"
    if errorlevel 1 (
        echo [错误] 下载 YOLOv5 失败，请检查网络
        pause
        exit /b 1
    )
    echo   正在安装依赖...
    pip install -r "%YOLOV5_DIR%\requirements.txt"
    echo.
)

echo ========================================
echo   开始训练！
echo   预计需要几小时，请耐心等待...
echo ========================================
echo.

python "%YOLOV5_DIR%\train.py" ^
  --data "%YAML_FILE%" ^
  --weights "%YOLOV5_DIR%\yolov5s.pt" ^
  --epochs 100 ^
  --batch-size 16 ^
  --img 704 ^
  --name fridge_model

echo.
if errorlevel 1 (
    echo [错误] 训练失败，请检查上面的错误信息
) else (
    echo ========================================
    echo   训练完成！
    echo   最好的模型在: yolov5\runs\train\fridge_model\weights\best.pt
    echo ========================================
    echo.
    echo   下一步：把 best.pt 转成 rknn 格式部署到 LuckFox
    echo   请联系我帮你处理这一步
)

echo.
pause
