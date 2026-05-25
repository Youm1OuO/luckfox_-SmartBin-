@echo off
REM 一键合并 4 个公开数据集到 fridge_project/datasets/public_merged/
REM
REM 这个 .bat 假定:
REM   - 当前在 yolov5 仓库根目录被双击,或者用 cmd 进到 yolov5 根目录后运行
REM   - 4 个数据集已经解压在 yolov5/datasets/<原名>/ 下
REM
REM 如果你之后挪走了某个数据集,直接编辑下面的路径即可。

setlocal

REM 切到 .bat 所在目录的上两层 (fridge_project/scripts -> yolov5)
cd /d "%~dp0..\.."

echo ============================================================
echo 开始合并公开数据集
echo 当前工作目录: %CD%
echo ============================================================

python fridge_project\scripts\prepare_dataset.py ^
    --source-yolo "datasets\FOOD-INGREDIENTS dataset.yolov5pytorch" ^
    --source-yolo "datasets\Beverage Containers.yolov5pytorch" ^
    --source-yolo "datasets\Hand Detection.yolov5pytorch" ^
    --source-yolo "datasets\hand.yolov5pytorch" ^
    --output      "fridge_project\datasets\public_merged" ^
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
echo   fridge_project\datasets\public_merged\
echo 训练时使用 yaml:
echo   fridge_project\datasets\public_merged\public_merged.yaml
echo ============================================================
pause
