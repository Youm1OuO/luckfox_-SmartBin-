@echo off
REM 一键合并所有公开数据集到 fridge_project/datasets/public_merged/
REM
REM 这个 .bat 假定:
REM   - 当前在 yolov5 仓库根目录被双击,或者用 cmd 进到 yolov5 根目录后运行
REM   - 各数据集已经解压在 yolov5/datasets/<原名>/ 下
REM
REM 如果你之后挪走了某个数据集,直接编辑下面的路径即可。
REM
REM 注意: 每次合并前会先清空 public_merged/ 下的 images/ 和 labels/,
REM 避免上次合并的 train_000001.jpg 残留导致图片/标签错配。

setlocal

REM 切到 .bat 所在目录的上两层 (fridge_project\scripts -> yolov5)
cd /d "%~dp0..\.."

set "OUT_DIR=fridge_project\datasets\public_merged"

echo ============================================================
echo 开始合并公开数据集
echo 当前工作目录: %CD%
echo 输出目录:     %OUT_DIR%
echo ============================================================

REM ---- 清理旧产物 ----
if exist "%OUT_DIR%\images" (
    echo 清理旧的 images\...
    rmdir /S /Q "%OUT_DIR%\images"
)
if exist "%OUT_DIR%\labels" (
    echo 清理旧的 labels\...
    rmdir /S /Q "%OUT_DIR%\labels"
)

python fridge_project\scripts\prepare_dataset.py ^
    --source-yolo "datasets\FOOD-INGREDIENTS dataset.yolov5pytorch" ^
    --source-yolo "datasets\Beverage Containers.yolov5pytorch" ^
    --source-yolo "datasets\Hand Detection.yolov5pytorch" ^
    --source-yolo "datasets\hand.yolov5pytorch" ^
    --source-yolo "datasets\Vegetables.yolov5pytorch" ^
    --source-yolo "datasets\Fruits and Vegetables.yolov5pytorch" ^
    --source-yolo "datasets\fruit detection.yolov5pytorch" ^
    --source-yolo "datasets\Cantaloupe Detection.yolov5pytorch" ^
    --source-yolo "datasets\Watermelon.yolov5pytorch" ^
    --source-yolo "datasets\grape.yolov5pytorch" ^
    --source-yolo "datasets\Milk.yolov5pytorch" ^
    --source-yolo "datasets\papaya.yolov5pytorch" ^
    --source-yolo "datasets\detect can.yolov5pytorch" ^
    --source-yolo "datasets\glass.yolov5pytorch" ^
    --source-yolo "datasets\Plastic Bottle 2.0.yolov5pytorch" ^
    --output      "%OUT_DIR%" ^
    --val-ratio   0.1

if errorlevel 1 (
    echo.
    echo !! 合并失败,请查看上面的错误信息
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 合并完成! 数据集位置:
echo   %OUT_DIR%\
echo 训练时使用 yaml:
echo   %OUT_DIR%\public_merged.yaml
echo ============================================================
pause
