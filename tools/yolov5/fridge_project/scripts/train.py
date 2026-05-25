"""
train.py — 两阶段训练调度器(包装上层 yolov5/train.py)。

设计:
    - 不修改 yolov5/train.py,而是用 subprocess 调用它。
    - 阶段 B(stage-b):公开数据集训练
        权重起点 = yolov5n.pt (官方 COCO 预训练)
        超参    = data/hyps/hyp.scratch-low.yaml (官方默认)
        epochs  = 100
    - 阶段 C(stage-c):自采数据微调
        权重起点 = 阶段 B 的 best.pt (或用户指定)
        超参    = configs/hyp_finetune.yaml (我们自定义,小学习率)
        冻结    = backbone 前 10 层(--freeze 10)
        epochs  = 50

用法:
    python fridge_project\scripts\train.py stage-b --data ...
    python fridge_project\scripts\train.py stage-c --data ... --weights runs/train/stage_b/weights/best.pt
    python fridge_project\scripts\train.py both    --data-b ... --data-c ...
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

# 关键路径定位
HERE = Path(__file__).resolve().parent           # fridge_project/scripts/
PROJ = HERE.parent                               # fridge_project/
YOLO_ROOT = PROJ.parent                          # yolov5/
TRAIN_PY = YOLO_ROOT / "train.py"
HYP_SCRATCH = YOLO_ROOT / "data" / "hyps" / "hyp.scratch-low.yaml"
HYP_FINETUNE = PROJ / "configs" / "hyp_finetune.yaml"

# 默认权重(官方 COCO 预训练)。如果不存在脚本会提示下载链接。
DEFAULT_PRETRAIN = YOLO_ROOT / "yolov5s.pt"
PRETRAIN_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"


def ensure_pretrain(path: Path):
    if path.exists():
        return
    print(
        f"\n!! 找不到官方预训练权重 {path}\n"
        f"   请手动下载后放到此位置:\n"
        f"     {PRETRAIN_URL}\n"
        f"   或用 PowerShell:\n"
        f"     Invoke-WebRequest -Uri {PRETRAIN_URL} -OutFile {path}\n",
        file=sys.stderr,
    )
    sys.exit(2)


def run_train(args: list[str]):
    """调用 yolov5/train.py。cwd 设为 yolov5 根目录,这样所有相对路径与官方文档一致。"""
    cmd = [sys.executable, str(TRAIN_PY), *args]
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd))
    print(f"  (cwd = {YOLO_ROOT})\n")
    res = subprocess.run(cmd, cwd=str(YOLO_ROOT))
    if res.returncode != 0:
        sys.exit(res.returncode)


# =============================================================================
# 阶段 B
# =============================================================================

def stage_b(opt: argparse.Namespace):
    ensure_pretrain(DEFAULT_PRETRAIN)
    args = [
        "--weights", str(opt.weights or DEFAULT_PRETRAIN),
        "--cfg",     opt.cfg,                       # yolov5n.yaml / yolov5s.yaml
        "--data",    str(Path(opt.data).resolve()),
        "--hyp",     str(HYP_SCRATCH),
        "--epochs",  str(opt.epochs),
        "--batch-size", str(opt.batch_size),
        "--imgsz",   str(opt.imgsz),
        "--project", "runs/train",
        "--name",    opt.name,
        "--workers", str(opt.workers),
        "--cos-lr",                                  # 余弦学习率,微调更稳
    ]
    if opt.device:
        args += ["--device", opt.device]
    if opt.exist_ok:
        args.append("--exist-ok")
    if opt.cache:
        args += ["--cache", opt.cache]
    run_train(args)


# =============================================================================
# 阶段 C
# =============================================================================

def stage_c(opt: argparse.Namespace):
    if not opt.weights:
        # 默认从阶段 B 的 best.pt 接力
        candidate = YOLO_ROOT / "runs" / "train" / "stage_b" / "weights" / "best.pt"
        if not candidate.exists():
            raise SystemExit(
                f"未指定 --weights,且找不到默认 {candidate}。\n"
                f"请先跑 stage-b,或用 --weights 指定起点。"
            )
        opt.weights = candidate
    args = [
        "--weights", str(Path(opt.weights).resolve()),
        # 阶段 C 不传 --cfg,让 train.py 直接从 weights 里读模型结构,
        # 这样类别数自动按数据 yaml 里的 nc 重建 head。
        "--data",    str(Path(opt.data).resolve()),
        "--hyp",     str(HYP_FINETUNE),
        "--epochs",  str(opt.epochs),
        "--batch-size", str(opt.batch_size),
        "--imgsz",   str(opt.imgsz),
        "--project", "runs/train",
        "--name",    opt.name,
        "--workers", str(opt.workers),
        "--freeze",  str(opt.freeze),                # 冻结前 N 层
        "--cos-lr",
        "--patience", "20",                          # 微调早停
    ]
    if opt.device:
        args += ["--device", opt.device]
    if opt.exist_ok:
        args.append("--exist-ok")
    if opt.cache:
        args += ["--cache", opt.cache]
    run_train(args)


# =============================================================================
# both
# =============================================================================

def both(opt: argparse.Namespace):
    # 先跑 B
    b_opt = argparse.Namespace(**{
        "data": opt.data_b, "epochs": opt.epochs_b, "batch_size": opt.batch_size,
        "imgsz": opt.imgsz, "name": "stage_b", "weights": None, "cfg": opt.cfg,
        "workers": opt.workers, "device": opt.device, "cache": opt.cache,
        "exist_ok": opt.exist_ok,
    })
    stage_b(b_opt)
    # 再跑 C,起点用刚训出来的 best.pt
    best = YOLO_ROOT / "runs" / "train" / "stage_b" / "weights" / "best.pt"
    c_opt = argparse.Namespace(**{
        "data": opt.data_c, "epochs": opt.epochs_c, "batch_size": opt.batch_size,
        "imgsz": opt.imgsz, "name": "stage_c", "weights": str(best),
        "workers": opt.workers, "device": opt.device, "cache": opt.cache,
        "exist_ok": opt.exist_ok, "freeze": opt.freeze,
    })
    stage_c(c_opt)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    # 共享参数辅助
    def add_common(s: argparse.ArgumentParser, default_name: str, default_epochs: int):
        s.add_argument("--data", required=True, help="data yaml 路径")
        s.add_argument("--epochs", type=int, default=default_epochs)
        s.add_argument("--batch-size", type=int, default=32)
        s.add_argument("--imgsz", type=int, default=320, help="输入分辨率(端侧推荐 320)")
        s.add_argument("--name", default=default_name)
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--device", default="", help="cuda 设备,例如 0;留空自动")
        s.add_argument("--cache", default=None, help="ram 或 disk")
        s.add_argument("--exist-ok", action="store_true")

    # stage-b
    sb = sub.add_parser("stage-b", help="阶段 B:公开数据集训练")
    add_common(sb, default_name="stage_b", default_epochs=100)
    sb.add_argument("--weights", default=None, help="起始权重(默认 yolov5n.pt)")
    sb.add_argument("--cfg", default="models/yolov5s.yaml",
                    help="模型结构 yaml,默认 yolov5s")
    sb.set_defaults(func=stage_b)

    # stage-c
    sc = sub.add_parser("stage-c", help="阶段 C:自采数据微调")
    add_common(sc, default_name="stage_c", default_epochs=50)
    sc.add_argument("--weights", default=None,
                    help="起始权重(默认用阶段 B 的 best.pt)")
    sc.add_argument("--freeze", type=int, default=10,
                    help="冻结 backbone 前 N 层(默认 10,即整个 backbone)")
    sc.set_defaults(func=stage_c)

    # both
    bo = sub.add_parser("both", help="一次跑完阶段 B + 阶段 C")
    bo.add_argument("--data-b", required=True)
    bo.add_argument("--data-c", required=True)
    bo.add_argument("--epochs-b", type=int, default=100)
    bo.add_argument("--epochs-c", type=int, default=50)
    bo.add_argument("--batch-size", type=int, default=32)
    bo.add_argument("--imgsz", type=int, default=320)
    bo.add_argument("--cfg", default="models/yolov5s.yaml")
    bo.add_argument("--workers", type=int, default=4)
    bo.add_argument("--device", default="")
    bo.add_argument("--cache", default=None)
    bo.add_argument("--exist-ok", action="store_true")
    bo.add_argument("--freeze", type=int, default=10)
    bo.set_defaults(func=both)

    return p


def main():
    parser = build_parser()
    opt = parser.parse_args()
    opt.func(opt)


if __name__ == "__main__":
    main()
