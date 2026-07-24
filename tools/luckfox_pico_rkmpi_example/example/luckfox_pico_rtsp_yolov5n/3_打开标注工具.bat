@echo off
chcp 65001 >nul
title 打开标注工具 LabelImg

echo ========================================
echo   打开标注工具 LabelImg
echo ========================================
echo.

set SCRIPT_DIR=%~dp0
set DATASET_DIR=%SCRIPT_DIR%dataset
set CLASSES_FILE=%DATASET_DIR%\classes.txt
set TRAIN_DIR=%DATASET_DIR%\images\train

if not exist "%TRAIN_DIR%" (
    echo [错误] 找不到训练集图片目录
    echo   请先运行 "2_整理数据集.py"
    echo.
    pause
    exit /b 1
)

if not exist "%CLASSES_FILE%" (
    echo [错误] 找不到类别文件
    echo   请先运行 "2_整理数据集.py"
    echo.
    pause
    exit /b 1
)

REM 检查 labelImg 是否已安装
python -c "import labelImg" 2>nul
if errorlevel 1 (
    echo   LabelImg 还没安装，正在安装...
    pip install labelImg
    echo.
)

echo   正在打开 LabelImg...
echo.
echo   打开后的操作：
echo     1. 左上角把 "PascalVOC" 切换成 "YOLO"
echo     2. 按 W 画框，框住照片里的东西
echo     3. 画完框后选类别（比如 apple）
echo     4. 按 D 看下一张，继续画
echo     5. 按 Ctrl+S 保存
echo.

labelImg "%TRAIN_DIR%" "%CLASSES_FILE%"
