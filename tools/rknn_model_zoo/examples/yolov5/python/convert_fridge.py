#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_fridge.py
=================
冰箱项目专用的 ONNX -> RKNN 转换脚本.

与官方 convert.py 唯一区别:
  - DATASET_PATH 指向 dataset_fridge.txt (你训练集分布的校准图)
  - 默认输出文件名 yolov5.rknn 一致, 编译流程不用改

使用前提:
  先在宿主机跑过校准集生成脚本:
    python3 tools/yolov5/fridge_project/scripts/build_calib_set.py
  它会把图复制到 ../calib_data/ 并写出 ./dataset_fridge.txt

用法 (在 RKNN docker 容器内, 假设仓库挂载到 /workspace):
  cd /workspace/tools/rknn_model_zoo/examples/yolov5/python
  python3 convert_fridge.py <onnx_path> rv1106 i8

  例:
  python3 convert_fridge.py ../model/best.onnx rv1106 i8
"""

import os
import sys
from rknn.api import RKNN

# 这里改了校准集路径; 其他和官方 convert.py 完全一致
DATASET_PATH = './dataset_fridge.txt'
DEFAULT_RKNN_PATH = '../model/yolov5.rknn'
DEFAULT_QUANT = True


def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1103, rv1106, rv1126b, rv1109, rv1126, rk1808]")
        print("       dtype choose from [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1103, rv1106, rv1126b]")
        print("       dtype choose from [u8, fp] for [rv1109, rv1126, rk1808]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path


if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    # ------ 量化前先检查校准集是否就位 ------
    if do_quant and not os.path.isfile(DATASET_PATH):
        print('=' * 64)
        print('ERROR: 找不到校准集索引: {}'.format(DATASET_PATH))
        print('请先在宿主机生成校准集 (项目根目录下执行):')
        print('  python3 tools/yolov5/fridge_project/scripts/build_calib_set.py')
        print('再回到容器内 examples/yolov5/python 目录运行本脚本')
        print('=' * 64)
        exit(1)

    # ------ ONNX 必须存在 ------
    if not os.path.isfile(model_path):
        print('ERROR: ONNX 模型不存在: {}'.format(model_path))
        exit(1)

    rknn = RKNN(verbose=False)

    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform=platform)
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Building model (calibrating with {})'.format(DATASET_PATH))
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    print('Saved to: {}'.format(output_path))

    rknn.release()
