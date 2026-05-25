"""
build_data_yaml.py — 给已经准备好的自采数据生成 YOLOv5 data yaml。

前置条件:数据集已经按以下结构放好:
    <dataset_root>/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt   (类别 ID 必须与 configs/classes.yaml 一致)

用法:
    python fridge_project\scripts\build_data_yaml.py ^
        --dataset-root fridge_project\datasets\self_collected ^
        --output       fridge_project\datasets\self_collected\self.yaml
"""
import argparse
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CLASSES_YAML = ROOT / "configs" / "classes.yaml"


def scan_label_distribution(labels_dir: Path) -> Counter:
    """统计 labels/*.txt 中各类别 ID 出现次数,顺便做最低限度的合法性检查。"""
    dist: Counter[int] = Counter()
    n_files = 0
    n_bad = 0
    for txt in labels_dir.rglob("*.txt"):
        n_files += 1
        for ln in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                cid = int(ln.split()[0])
            except (ValueError, IndexError):
                n_bad += 1
                continue
            dist[cid] += 1
    print(f"  扫描 {labels_dir}:{n_files} 个 .txt 文件,{n_bad} 行无法解析")
    return dist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.dataset_root).resolve()
    if not (root / "images" / "train").exists():
        raise SystemExit(f"找不到 {root / 'images' / 'train'},请先准备好数据集结构。")

    with CLASSES_YAML.open("r", encoding="utf-8") as f:
        classes = list(yaml.safe_load(f)["classes"])

    # 扫描分布并检查越界
    print("→ 检查 train labels:")
    train_dist = scan_label_distribution(root / "labels" / "train")
    print("→ 检查 val labels:")
    val_dist = scan_label_distribution(root / "labels" / "val")

    bad_ids = [cid for cid in (train_dist | val_dist) if cid < 0 or cid >= len(classes)]
    if bad_ids:
        raise SystemExit(
            f"发现越界类别 ID: {bad_ids},当前类别表共 {len(classes)} 类。\n"
            f"请检查 labels 是否还在用旧版类别表。"
        )

    yaml_data = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "nc": len(classes),
        "names": {i: n for i, n in enumerate(classes)},
    }
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    print(f"\n类别分布(train + val):")
    total = train_dist + val_dist
    for cid in range(len(classes)):
        n = total.get(cid, 0)
        flag = "  " if n > 0 else " ⚠"
        print(f"  {flag} {cid:>3} {classes[cid]:<20} train={train_dist.get(cid, 0):>5}  val={val_dist.get(cid, 0):>5}")

    print(f"\n✓ 写出 {out_path}")


if __name__ == "__main__":
    main()
