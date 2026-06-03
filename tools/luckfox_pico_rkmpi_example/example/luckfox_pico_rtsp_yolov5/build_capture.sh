#!/bin/bash
# ============================================================================
#  build_capture.sh
#  单独编译 capture_dataset 截图工具（不影响主程序）
#
#  用法:
#    在 Docker 交叉编译环境中运行:
#      bash build_capture.sh
#
#  产出:
#    install/uclibc/capture_dataset_demo/capture_dataset
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RKMPI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_capture"

# SDK 路径（交叉编译需要）
SDK_PATH="${LUCKFOX_SDK_PATH:-}"
if [ -z "$SDK_PATH" ]; then
    echo "错误: 请先设置 LUCKFOX_SDK_PATH"
    echo "  export LUCKFOX_SDK_PATH=/path/to/luckfox-pico"
    exit 1
fi

COMPILER="$SDK_PATH/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf"

if [ ! -f "${COMPILER}-gcc" ]; then
    echo "错误: 编译器不存在: ${COMPILER}-gcc"
    echo "  请检查 LUCKFOX_SDK_PATH 是否正确"
    exit 1
fi

echo "============================================================"
echo "  编译 capture_dataset (截图采集工具)"
echo "  SDK: $SDK_PATH"
echo "  编译器: ${COMPILER}-gcc"
echo "============================================================"

# 清理旧 build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 编译
${COMPILER}-g++ \
    -g -Wall \
    -DRV1106_1103 -DISP_HW_V30 -DRKPLATFORM=ON -DARCH64=OFF \
    -DROCKIVA -DUAPI2 \
    -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 \
    -I"$RKMPI_DIR/include" \
    -I"$RKMPI_DIR/include/rknn" \
    -I"$RKMPI_DIR/include/librga" \
    -I"$RKMPI_DIR/include/rkaiq" \
    -I"$RKMPI_DIR/include/rkaiq/uAPI2" \
    -I"$RKMPI_DIR/include/rkaiq/common" \
    -I"$RKMPI_DIR/include/rkaiq/xcore" \
    -I"$RKMPI_DIR/include/rkaiq/algos" \
    -I"$RKMPI_DIR/include/rkaiq/iq_parser" \
    -I"$RKMPI_DIR/include/rkaiq/iq_parser_v2" \
    -I"$RKMPI_DIR/include/rkaiq/smartIr" \
    -I"$RKMPI_DIR/common" \
    -I"$RKMPI_DIR/common/isp3.x" \
    -I"$SCRIPT_DIR/include" \
    -I"$RKMPI_DIR/lib/uclibc/lib/cmake/opencv4/../../include/opencv4" \
    -o capture_dataset \
    "$SCRIPT_DIR/src_capture/capture_dataset.cc" \
    "$SCRIPT_DIR/src/luckfox_mpi.cc" \
    -L"$RKMPI_DIR/lib/uclibc" \
    -Wl,-rpath-link,"$RKMPI_DIR/lib/uclibc:/usr/lib" \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
    -lrknnmrt -lrockiva -lsample_comm -lrockit -lrockchip_mpp -lrkaiq \
    -lrtsp -lrga -lpthread -lstdc++ -lm -ldl -lrt

# 安装
INSTALL_DIR="$RKMPI_DIR/install/uclibc/capture_dataset_demo"
mkdir -p "$INSTALL_DIR"
cp capture_dataset "$INSTALL_DIR/"
cp "$SCRIPT_DIR/run_capture.sh" "$INSTALL_DIR/" 2>/dev/null || true

echo ""
echo "============================================================"
echo "  编译成功!"
echo "  可执行文件: $INSTALL_DIR/capture_dataset"
echo ""
echo "  部署到板子:"
echo "    scp $INSTALL_DIR/capture_dataset root@<板子IP>:/root/"
echo ""
echo "  板子上运行:"
echo "    ./capture_dataset 2 100    # 每2秒截一张, 共100张"
echo "============================================================"
