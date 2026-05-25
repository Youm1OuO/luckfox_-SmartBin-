#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_calib_set.py
==================
为 RKNN i8 量化生成"代表你训练分布"的校准集 -- 通用版.

设计原则:
  - 完全不写死数据集名字, 自动扫描 datasets/ 下所有 Roboflow 风格目录
  - 你以后加新数据集 / 删数据集, 直接丢进 datasets/ 重跑本脚本即可
  - 按"每个数据集 split 的图片数"做分层抽样, 防止某个大集吃掉名额

支持的目录布局 (Roboflow / Ultralytics 通用):
  datasets/
    <whatever_name>/
      train/images/ + train/labels/    <- 抽这个
      valid/images/ + valid/labels/    <- 也抽这个 (可选)
      test/images/  + test/labels/     <- 默认不抽 (留作真测试集)
    <another_dataset>/
      train/images/...                  <- 自动发现
    <flat_dataset>/
      images/ + labels/                 <- 也支持 (无 split 的扁平结构)

具体做了什么:
  1) 递归扫描 DATASETS_ROOT, 找出所有 (images_dir, labels_dir) 配对
  2) 每张图必须有非空的 .txt 标注才会被采纳 (空标注对量化无价值)
  3) 按"每个 (images_dir) 含标注图数量"做分层抽样
  4) 复制图片到 calib_data/, 同时生成 dataset_fridge.txt (相对路径)

用法:
  python build_calib_set.py                     # 默认抽 100 张, 用 train+valid
  python build_calib_set.py --num 200           # 抽 200 张
  python build_calib_set.py --splits train      # 仅用 train (不用 valid)
  python build_calib_set.py --include-test      # 把 test 也算上 (不推荐)
  python build_calib_set.py --dry-run           # 仅打印分配, 不真复制
  python build_calib_set.py --datasets-root /path/to/datasets  # 换数据集根目录
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# 默认路径配置
# -----------------------------------------------------------------------------
# 脚本位置: <workspace>/tools/yolov5/fridge_project/scripts/build_calib_set.py
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[3]   # 上推 4 层 = workspace 根

DEFAULT_DATASETS_ROOT = WORKSPACE / "tools" / "yolov5" / "datasets"
DEFAULT_RKNN_YOLOV5 = WORKSPACE / "tools" / "rknn_model_zoo" / "examples" / "yolov5"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Roboflow 风格的 split 子目录名 (按"应不应该用作校准"排序)
SPLIT_DIRS_DEFAULT = ["train", "valid"]   # 默认含 train + valid
SPLIT_DIR_TEST = "test"


# -----------------------------------------------------------------------------
# 数据集发现
# -----------------------------------------------------------------------------
def discover_image_dirs(datasets_root: Path, splits, include_test: bool):
    """
    递归扫描 datasets_root, 返回所有 (images_dir, labels_dir) 配对.

    匹配规则: 任意一个名为 'images' 的目录, 同级必须存在 'labels'.
    例如:
      datasets/<X>/train/images <-> datasets/<X>/train/labels
      datasets/<Y>/images       <-> datasets/<Y>/labels

    splits: 用户指定的 split 名字列表 (用来过滤 train/valid/test);
            扁平结构 (无 split) 不受限, 直接接受.
    """
    if not datasets_root.is_dir():
        return []

    allowed_splits = set(splits)
    if include_test:
        allowed_splits.add(SPLIT_DIR_TEST)

    pairs = []
    # rglob 比手动 walk 简洁, 性能也够用 (Roboflow 数据集层级浅)
    for images_dir in datasets_root.rglob("images"):
        if not images_dir.is_dir():
            continue
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.is_dir():
            continue

        # 判断是否处在某个 split 子目录下 (train/valid/test)
        # 父目录名 = split_name (Roboflow 风格)
        split_name = images_dir.parent.name.lower()

        # 扁平结构: images_dir 的父目录直接就是 dataset 根, 父名通常是数据集名,
        # 不在 train/valid/test 列表里; 此时无条件接受 (因为没有 split 这个概念)
        is_split_layout = split_name in {"train", "valid", "test"}

        if is_split_layout:
            if split_name not in allowed_splits:
                continue
        # 扁平结构: 直接通过

        pairs.append((images_dir, labels_dir))

    # 排序保证可复现 (随机种子 + 输入顺序固定 -> 输出固定)
    pairs.sort(key=lambda p: str(p[0]).lower())
    return pairs


def collect_labeled_images(images_dir: Path, labels_dir: Path):
    """
    扫描 images_dir, 返回 [image_path, ...]; 仅保留有非空 label 的图.
    """
    out = []
    for img_path in images_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.is_file():
            continue
        try:
            content = label_path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not content:
            continue
        out.append(img_path)
    return out


def short_tag_for(images_dir: Path, datasets_root: Path) -> str:
    """
    给抽样到的图加一个短前缀, 防止跨数据集重名.
    用 (datasets_root 下的相对路径) 的关键片段拼成一个紧凑标签.
    """
    try:
        rel = images_dir.relative_to(datasets_root)
    except ValueError:
        rel = Path(images_dir.name)
    parts = [p for p in rel.parts if p.lower() != "images"]
    # 抽前几个非空字符
    tag = "_".join(p.replace(" ", "")[:6] for p in parts) or "set"
    return tag


# -----------------------------------------------------------------------------
# 分层抽样
# -----------------------------------------------------------------------------
def allocate_quotas(per_subset_counts, total_target):
    """
    按各子集大小做比例分配; 至少给非空子集 1 张; 不超过子集容量;
    最后微调让总和等于 total_target.
    """
    grand = sum(per_subset_counts)
    if grand == 0:
        return [0] * len(per_subset_counts)

    quotas = []
    for n in per_subset_counts:
        q = round(total_target * n / grand) if n > 0 else 0
        if n > 0 and q < 1:
            q = 1
        q = min(q, n)
        quotas.append(q)

    # 微调到正好等于 total_target
    diff = total_target - sum(quotas)
    # 优先调整大的子集 (加额度时用大集; 减额度时也从大集减, 影响小)
    order = sorted(range(len(per_subset_counts)),
                   key=lambda i: per_subset_counts[i],
                   reverse=True)

    safety = len(per_subset_counts) * 10 + 10
    while diff != 0 and safety > 0:
        progressed = False
        for idx in order:
            if diff == 0:
                break
            if diff > 0 and quotas[idx] < per_subset_counts[idx]:
                quotas[idx] += 1
                diff -= 1
                progressed = True
            elif diff < 0 and quotas[idx] > 0:
                quotas[idx] -= 1
                diff += 1
                progressed = True
        if not progressed:
            break
        safety -= 1

    return quotas


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--num", type=int, default=100,
                        help="校准图总张数, 推荐 50~200 (默认 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42, 复现用)")
    parser.add_argument("--splits", nargs="+", default=SPLIT_DIRS_DEFAULT,
                        help="用哪些 split (默认 train valid)")
    parser.add_argument("--include-test", action="store_true",
                        help="把 test split 也算上 (默认不算)")
    parser.add_argument("--datasets-root", type=Path,
                        default=DEFAULT_DATASETS_ROOT,
                        help="数据集根目录 (默认 tools/yolov5/datasets)")
    parser.add_argument("--rknn-yolov5", type=Path,
                        default=DEFAULT_RKNN_YOLOV5,
                        help="rknn_model_zoo/examples/yolov5 目录")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印分配, 不真复制")
    args = parser.parse_args()

    random.seed(args.seed)

    calib_dir = args.rknn_yolov5 / "calib_data"
    dataset_txt = args.rknn_yolov5 / "python" / "dataset_fridge.txt"

    # ------ 1. 发现所有 (images, labels) 对 ------
    print(f"[1/4] 扫描 {args.datasets_root}")
    print(f"      使用 split: {args.splits}"
          + ("  + test" if args.include_test else ""))
    pairs = discover_image_dirs(args.datasets_root, args.splits,
                                args.include_test)
    if not pairs:
        print("[ERROR] 未找到任何 'images' + 'labels' 配对; 检查 --datasets-root")
        sys.exit(1)

    # ------ 2. 列出每个子集的有效图 ------
    print(f"\n[2/4] 收集每个子集的带标注图")
    subsets = []  # [(short_tag, display_name, [img_path, ...])]
    for images_dir, labels_dir in pairs:
        imgs = collect_labeled_images(images_dir, labels_dir)
        tag = short_tag_for(images_dir, args.datasets_root)
        try:
            disp = str(images_dir.relative_to(args.datasets_root))
        except ValueError:
            disp = str(images_dir)
        subsets.append((tag, disp, imgs))
        print(f"   - {disp:65s}  {len(imgs):>5d} 张")

    counts = [len(s[2]) for s in subsets]
    if sum(counts) == 0:
        print("[ERROR] 全部子集都空, 没有可用图")
        sys.exit(1)

    # ------ 3. 分层抽样 ------
    print(f"\n[3/4] 分层抽样 总目标 {args.num} 张")
    quotas = allocate_quotas(counts, args.num)

    sampled = []  # [(src_path, dst_filename)]
    used_names = set()
    for (tag, disp, imgs), q in zip(subsets, quotas):
        if q == 0 or not imgs:
            continue
        chosen = random.sample(imgs, q)
        for img_path in chosen:
            base = f"{tag}_{img_path.name}"
            final = base
            n = 1
            while final in used_names:
                final = f"{tag}_{n}_{img_path.name}"
                n += 1
            used_names.add(final)
            sampled.append((img_path, final))
        print(f"   - {disp:65s}  {q:>5d} 张")

    print(f"   实际抽到 {len(sampled)} / {args.num} 张")

    # ------ 4. 复制 + 写 dataset.txt ------
    print(f"\n[4/4] 输出")
    print(f"   复制目标: {calib_dir}")
    print(f"   路径列表: {dataset_txt}")

    if args.dry_run:
        for src, dst in sampled[:5]:
            print(f"   [DRY] {src.name} -> {dst}")
        if len(sampled) > 5:
            print(f"   [DRY] ... 共 {len(sampled)} 张")
        return

    # 清空旧的 calib_data 防止上次的图混进来
    if calib_dir.exists():
        for old in calib_dir.iterdir():
            try:
                old.unlink()
            except Exception:
                pass
    else:
        calib_dir.mkdir(parents=True, exist_ok=True)

    for src, dst in sampled:
        shutil.copy2(src, calib_dir / dst)

    dataset_txt.parent.mkdir(parents=True, exist_ok=True)
    with dataset_txt.open("w", encoding="utf-8", newline="\n") as f:
        for _, dst in sampled:
            # convert.py 在 examples/yolov5/python/ 下运行,
            # 所以这里是 ../calib_data/<file>
            f.write(f"../calib_data/{dst}\n")

    print(f"\nOK!")
    print(f"  校准图: {len(sampled)} 张  -> {calib_dir}")
    print(f"  路径表: {dataset_txt}")
    print(f"\n下一步 (容器内):")
    print(f"  cd /workspace/tools/rknn_model_zoo/examples/yolov5/python")
    print(f"  python3 convert_fridge.py <你的onnx路径> rv1106 i8")


if __name__ == "__main__":
    main()
