"""
prepare_dataset.py — 合并多个公开数据集到统一的 YOLO 格式。

核心逻辑:
  1. 读 configs/classes.yaml 拿到统一类别表 + 别名映射。
  2. 遍历每个传入的数据源,按其格式(YOLO/COCO/VOC)解析。
  3. 把每条标注的类别名通过 aliases 映射到统一类别。
       - 命中 → 写入输出
       - 未命中 → 计入 dropped 统计,该样本若没有任何命中类则跳过
  4. 按 val_ratio 切分 train/val,复制图片、改写 labels。
  5. 输出 <output>/<output_name>.yaml 给 train.py 使用。

支持的输入格式:
  --source-yolo PATH    YOLO 格式(images/labels 子目录,有 data.yaml 或 classes.txt)
  --source-coco PATH    COCO JSON(annotations/*.json + images/)
  --source-voc  PATH    Pascal VOC(JPEGImages/ + Annotations/*.xml)

每个 --source-* 可重复传入多次。

用法示例(Windows cmd):
    python fridge_project\scripts\prepare_dataset.py ^
        --source-yolo D:\datasets\roboflow_fridge ^
        --source-yolo D:\datasets\hand_dataset ^
        --source-coco D:\datasets\coco2017 ^
        --output      fridge_project\datasets\public_merged ^
        --val-ratio   0.1
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

# 让本脚本无论从哪运行,都能找到 configs/classes.yaml
ROOT = Path(__file__).resolve().parent.parent  # fridge_project/
CLASSES_YAML = ROOT / "configs" / "classes.yaml"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# 类别表加载
# =============================================================================

@dataclass
class ClassMap:
    """从 classes.yaml 派生出的查找表。"""
    classes: list[str]                    # 按 ID 顺序的类别名
    name_to_id: dict[str, int]            # 标准名 → ID
    aliases: dict[str, str]               # 外部名(归一化后) → 标准名

    @classmethod
    def load(cls, yaml_path: Path) -> "ClassMap":
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        classes = list(data["classes"])
        name_to_id = {n: i for i, n in enumerate(classes)}

        # 标准名自身也算别名(指向自己)
        aliases: dict[str, str] = {cls._normalize(n): n for n in classes}
        for k, v in (data.get("aliases") or {}).items():
            aliases[cls._normalize(k)] = v
        return cls(classes=classes, name_to_id=name_to_id, aliases=aliases)

    @staticmethod
    def _normalize(s: str) -> str:
        """统一小写、去多余空白、'_' 与空格视作等价。"""
        return re.sub(r"\s+", " ", str(s).strip().lower()).replace("_", " ")

    def map_external(self, external_name: str) -> int | None:
        """外部标签名 → 内部类别 ID,未命中返回 None。"""
        std = self.aliases.get(self._normalize(external_name))
        if std is None:
            return None
        return self.name_to_id.get(std)


# =============================================================================
# 标注收集 — 中间表示
# =============================================================================

@dataclass
class AnnotatedImage:
    """一张图 + 多个 bbox 的中间表示。bbox 已归一化到 [0,1] xywh 中心格式。"""
    image_path: Path
    width: int
    height: int
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    # 每个 box: (cls_id, cx, cy, w, h)


@dataclass
class Stats:
    images_in: int = 0
    images_out: int = 0
    boxes_in: int = 0
    boxes_out: int = 0
    dropped_labels: Counter = field(default_factory=Counter)

    def report(self, source_tag: str) -> str:
        kept = self.boxes_out
        drop = sum(self.dropped_labels.values())
        top = ", ".join(f"{k}({v})" for k, v in self.dropped_labels.most_common(8))
        return (
            f"[{source_tag}] images: {self.images_out}/{self.images_in} kept "
            f"| boxes: {kept}/{self.boxes_in} kept (dropped {drop})\n"
            f"          top dropped labels: {top or '<none>'}"
        )


# =============================================================================
# 解析器:每种输入格式一个生成器
# =============================================================================

def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    """绝对坐标 (x1,y1,x2,y2) → YOLO (cx,cy,w,h),归一化到 [0,1]。"""
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def _read_image_size(path: Path) -> tuple[int, int] | None:
    """读图取尺寸,失败返回 None。优先用 PIL(快),退回 OpenCV。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return w, h
    except Exception:
        return None


def parse_yolo_source(root: Path, cmap: ClassMap, stats: Stats) -> Iterable[AnnotatedImage]:
    """
    YOLO 格式约定:
      root/
        images/         任意层级,放图片
        labels/         同名 .txt 标注
        data.yaml or classes.txt  提供类别名列表
    """
    # 找类别名表
    names: list[str] | None = None
    for cand in [root / "data.yaml", root / "dataset.yaml"]:
        if cand.exists():
            with cand.open("r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
            n = d.get("names")
            if isinstance(n, list):
                names = n
            elif isinstance(n, dict):
                names = [n[i] for i in sorted(n)]
            break
    if names is None:
        cand = root / "classes.txt"
        if cand.exists():
            names = [ln.strip() for ln in cand.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if names is None:
        print(f"  ! YOLO source {root} 缺少 data.yaml 或 classes.txt,跳过", file=sys.stderr)
        return

    # 遍历图像
    images_dir = root / "images"
    if not images_dir.exists():
        # 兼容把图片直接放 root 下的情况
        images_dir = root
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        stats.images_in += 1
        # 找对应 label:替换 images→labels,后缀改 .txt
        try:
            rel = img_path.relative_to(root)
            parts = list(rel.parts)
            if parts and parts[0] == "images":
                parts[0] = "labels"
            label_path = root.joinpath(*parts).with_suffix(".txt")
        except ValueError:
            label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            continue

        size = _read_image_size(img_path)
        if size is None:
            continue
        w, h = size
        ai = AnnotatedImage(image_path=img_path, width=w, height=h)
        for ln in label_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            stats.boxes_in += 1
            try:
                parts = ln.split()
                src_id = int(parts[0])
                cx, cy, bw, bh = (float(x) for x in parts[1:5])
            except (ValueError, IndexError):
                continue
            if not (0 <= src_id < len(names)):
                stats.dropped_labels["<unknown_id>"] += 1
                continue
            new_id = cmap.map_external(names[src_id])
            if new_id is None:
                stats.dropped_labels[names[src_id]] += 1
                continue
            ai.boxes.append((new_id, cx, cy, bw, bh))
            stats.boxes_out += 1
        if ai.boxes:
            stats.images_out += 1
            yield ai


def parse_coco_source(root: Path, cmap: ClassMap, stats: Stats) -> Iterable[AnnotatedImage]:
    """
    COCO 格式约定:
      root/
        annotations/instances_*.json   (找所有 .json)
        images/   或   train2017/  val2017/  ...
    """
    ann_dir = root / "annotations"
    if not ann_dir.exists():
        print(f"  ! COCO source {root} 缺少 annotations/,跳过", file=sys.stderr)
        return
    json_files = sorted(ann_dir.glob("*.json"))
    if not json_files:
        print(f"  ! COCO source {root} 没找到 .json,跳过", file=sys.stderr)
        return

    # 候选图像目录(按常见命名)
    img_dirs = [root / "images", root]
    img_dirs += [d for d in root.iterdir() if d.is_dir() and d.name not in {"annotations"}]

    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            coco = json.load(f)
        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        imgs = {im["id"]: im for im in coco.get("images", [])}
        anns_by_img: dict[int, list[dict]] = defaultdict(list)
        for a in coco.get("annotations", []):
            anns_by_img[a["image_id"]].append(a)

        for img_id, im in imgs.items():
            stats.images_in += 1
            file_name = im["file_name"]
            img_path = None
            for d in img_dirs:
                cand = d / file_name
                if cand.exists():
                    img_path = cand
                    break
                # 子目录递归找一次(慢但稳)
                hits = list(d.rglob(Path(file_name).name))
                if hits:
                    img_path = hits[0]
                    break
            if img_path is None:
                continue
            w = im.get("width") or 0
            h = im.get("height") or 0
            if not (w and h):
                size = _read_image_size(img_path)
                if size is None:
                    continue
                w, h = size

            ai = AnnotatedImage(image_path=img_path, width=w, height=h)
            for a in anns_by_img.get(img_id, []):
                stats.boxes_in += 1
                cat_name = cats.get(a["category_id"], "")
                new_id = cmap.map_external(cat_name)
                if new_id is None:
                    stats.dropped_labels[cat_name or "<no_name>"] += 1
                    continue
                x, y, bw, bh = a["bbox"]  # COCO: 左上角 + 宽高,绝对像素
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                ai.boxes.append((new_id, cx, cy, bw / w, bh / h))
                stats.boxes_out += 1
            if ai.boxes:
                stats.images_out += 1
                yield ai


def parse_voc_source(root: Path, cmap: ClassMap, stats: Stats) -> Iterable[AnnotatedImage]:
    """
    VOC 格式约定:
      root/
        Annotations/*.xml
        JPEGImages/*.jpg   (或 images/)
    """
    ann_dir = root / "Annotations"
    if not ann_dir.exists():
        print(f"  ! VOC source {root} 缺少 Annotations/,跳过", file=sys.stderr)
        return
    img_dir = root / "JPEGImages"
    if not img_dir.exists():
        img_dir = root / "images"

    for xml_path in ann_dir.glob("*.xml"):
        stats.images_in += 1
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        rt = tree.getroot()
        fname = (rt.findtext("filename") or "").strip()
        size_node = rt.find("size")
        if size_node is None:
            continue
        try:
            w = int(size_node.findtext("width") or 0)
            h = int(size_node.findtext("height") or 0)
        except ValueError:
            continue
        if not (w and h):
            continue
        img_path = img_dir / fname if fname else img_dir / (xml_path.stem + ".jpg")
        if not img_path.exists():
            # 尝试别的扩展名
            cand = next((p for p in img_dir.glob(xml_path.stem + ".*") if p.suffix.lower() in IMG_EXTS), None)
            if cand is None:
                continue
            img_path = cand

        ai = AnnotatedImage(image_path=img_path, width=w, height=h)
        for obj in rt.findall("object"):
            stats.boxes_in += 1
            name = (obj.findtext("name") or "").strip()
            new_id = cmap.map_external(name)
            if new_id is None:
                stats.dropped_labels[name or "<no_name>"] += 1
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                x1 = float(bb.findtext("xmin"))
                y1 = float(bb.findtext("ymin"))
                x2 = float(bb.findtext("xmax"))
                y2 = float(bb.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            cx, cy, bw, bh = _xyxy_to_yolo(x1, y1, x2, y2, w, h)
            ai.boxes.append((new_id, cx, cy, bw, bh))
            stats.boxes_out += 1
        if ai.boxes:
            stats.images_out += 1
            yield ai


# =============================================================================
# 写出
# =============================================================================

def write_split(items: list[AnnotatedImage], img_out: Path, lbl_out: Path, prefix: str):
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for i, ai in enumerate(items):
        # 防重名:用 prefix + 序号 + 原文件名
        new_name = f"{prefix}_{i:06d}{ai.image_path.suffix.lower()}"
        try:
            shutil.copy2(ai.image_path, img_out / new_name)
        except OSError as e:
            print(f"  ! copy 失败 {ai.image_path}: {e}", file=sys.stderr)
            continue
        lbl_lines = [
            f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            for cid, cx, cy, bw, bh in ai.boxes
        ]
        (lbl_out / (Path(new_name).stem + ".txt")).write_text(
            "\n".join(lbl_lines) + "\n", encoding="utf-8"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-yolo", action="append", default=[], help="YOLO 格式数据源(可多次)")
    ap.add_argument("--source-coco", action="append", default=[], help="COCO 格式数据源(可多次)")
    ap.add_argument("--source-voc",  action="append", default=[], help="VOC 格式数据源(可多次)")
    ap.add_argument("--output", required=True, help="输出目录")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例(默认 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="输出 yaml 名(默认与输出目录同名)")
    args = ap.parse_args()

    if not (args.source_yolo or args.source_coco or args.source_voc):
        ap.error("至少需要一个 --source-* 参数")

    cmap = ClassMap.load(CLASSES_YAML)
    print(f"已加载类别表:{len(cmap.classes)} 类,别名 {len(cmap.aliases)} 项")

    out_root = Path(args.output).resolve()
    out_name = args.name or out_root.name

    all_items: list[AnnotatedImage] = []
    parsers = [
        ("yolo", args.source_yolo, parse_yolo_source),
        ("coco", args.source_coco, parse_coco_source),
        ("voc",  args.source_voc,  parse_voc_source),
    ]
    for tag, paths, fn in parsers:
        for p in paths:
            stats = Stats()
            print(f"\n→ 解析 [{tag}] {p}")
            for ai in fn(Path(p).resolve(), cmap, stats):
                all_items.append(ai)
            print(stats.report(tag))

    if not all_items:
        print("\n!! 没有任何样本通过类别映射,请检查 classes.yaml 的 aliases 段。", file=sys.stderr)
        sys.exit(1)

    # 切分
    rng = random.Random(args.seed)
    rng.shuffle(all_items)
    n_val = max(1, int(len(all_items) * args.val_ratio))
    val_items = all_items[:n_val]
    train_items = all_items[n_val:]

    print(f"\n总计 {len(all_items)} 张 → train {len(train_items)}, val {len(val_items)}")

    # 写出
    write_split(train_items, out_root / "images" / "train", out_root / "labels" / "train", "train")
    write_split(val_items,   out_root / "images" / "val",   out_root / "labels" / "val",   "val")

    # 写 yaml
    yaml_path = out_root / f"{out_name}.yaml"
    yaml_data = {
        "path": str(out_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(cmap.classes),
        "names": {i: n for i, n in enumerate(cmap.classes)},
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    # 类别分布统计
    dist = Counter(b[0] for ai in all_items for b in ai.boxes)
    print("\n类别分布(细类 ID → 数量):")
    for cid, n in sorted(dist.items()):
        print(f"  {cid:>3} {cmap.classes[cid]:<20} {n}")

    print(f"\n✓ 完成。data yaml: {yaml_path}")


if __name__ == "__main__":
    main()
