@echo off
chcp 65001 >nul
title 检查标注质量

echo ========================================
echo   检查标注质量
echo ========================================
echo.
echo   会弹出窗口显示前 20 张照片和你标的框
echo   按任意键切换下一张，看完自动关闭
echo.

set SCRIPT_DIR=%~dp0
set DATASET_DIR=%SCRIPT_DIR%dataset\images\train
set LABELS_DIR=%SCRIPT_DIR%dataset\labels\train

if not exist "%DATASET_DIR%" (
    echo [错误] 找不到图片目录
    echo   请先运行 "2_整理数据集.py"
    pause
    exit /b 1
)

python -c "import cv2" 2>nul
if errorlevel 1 (
    echo   正在安装 opencv...
    pip install opencv-python
)

python -c "
import cv2, glob, os, sys

img_dir = r'%DATASET_DIR%'
lbl_dir = r'%LABELS_DIR%'

imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
if not imgs:
    imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))

if not imgs:
    print('[错误] 没有找到图片')
    sys.exit(1)

count = 0
for img_path in imgs[:20]:
    lbl_path = os.path.join(lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
    im = cv2.imread(img_path)
    if im is None:
        continue
    h, w = im.shape[:2]

    if os.path.exists(lbl_path):
        for line in open(lbl_path):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            c, cx, cy, bw, bh = parts
            x1 = int((float(cx) - float(bw)/2) * w)
            y1 = int((float(cy) - float(bh)/2) * h)
            x2 = int((float(cx) + float(bw)/2) * w)
            y2 = int((float(cy) + float(bh)/2) * h)
            cv2.rectangle(im, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(im, c, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(im, 'OK - has labels', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    else:
        cv2.putText(im, 'NO LABELS YET', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow('Check Labels (press any key for next)', im)
    cv2.waitKey(0)
    count += 1

cv2.destroyAllWindows()
print(f'  检查了 {count} 张照片')
"

echo.
echo   检查完成！
echo.
pause
