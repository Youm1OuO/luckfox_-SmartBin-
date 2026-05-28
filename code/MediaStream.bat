@echo off
chcp 65001 >nul
cd /d "%~dp0"

ffplay ^
  -fflags nobuffer ^
  -flags low_delay ^
  -framedrop ^
  -avioflags direct ^
  -probesize 32 ^
  -analyzeduration 0 ^
  -rtsp_transport tcp ^
  -x 800 -y 600 ^
  -an ^
  rtsp://192.168.168.100/live/0

if errorlevel 1 pause
