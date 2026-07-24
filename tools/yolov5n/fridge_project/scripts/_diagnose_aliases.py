"""临时脚本:诊断哪些外部标签命中、哪些没命中。"""
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_dataset import ClassMap

CLASSES_YAML = Path("/home/muyou/Projects/workspace/yolov5/fridge_project/configs/classes.yaml")
DATASETS = [
    Path("/home/muyou/Projects/workspace/yolov5/datasets/FOOD-INGREDIENTS dataset.yolov5pytorch"),
    Path("/home/muyou/Projects/workspace/yolov5/datasets/Beverage Containers.yolov5pytorch"),
    Path("/home/muyou/Projects/workspace/yolov5/datasets/Hand Detection.yolov5pytorch"),
    Path("/home/muyou/Projects/workspace/yolov5/datasets/hand.yolov5pytorch"),
]

cmap = ClassMap.load(CLASSES_YAML)
print(f"已加载 {len(cmap.classes)} 类,{len(cmap.aliases)} 别名")
print(f"\n所有标准类:")
for i, c in enumerate(cmap.classes):
    print(f"  {i:>2} {c}")

for ds in DATASETS:
    yml = ds / "data.yaml"
    if not yml.exists():
        continue
    with yml.open() as f:
        d = yaml.safe_load(f)
    names = d.get("names", [])
    print(f"\n\n=== {ds.name} ({len(names)} 类) ===")
    hit, miss = [], []
    for n in names:
        r = cmap.map_external(n)
        if r is not None:
            hit.append((n, cmap.classes[r]))
        else:
            miss.append(n)
    print(f"命中 {len(hit)} / 漏掉 {len(miss)}")
    print("\n-- 命中 --")
    for src, dst in hit:
        marker = " (==自身)" if src.lower().strip() == dst else ""
        print(f"  {src!r:40s} -> {dst}{marker}")
    print("\n-- 漏掉(可能要补 alias) --")
    for n in miss:
        # 找是否有 fuzzy 匹配
        norm = n.lower().strip()
        sugg = []
        for cls in cmap.classes:
            cls_n = cls.replace("_", " ")
            if cls_n in norm or norm in cls_n:
                sugg.append(cls)
        sugg_s = f"  ⚠ 疑似 → {sugg}" if sugg else ""
        print(f"  {n!r}{sugg_s}")
