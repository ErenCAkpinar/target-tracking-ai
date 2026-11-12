"""Fine-tune a COCO-pretrained YOLO detector on VisDrone2019-DET.

Usage:
  python training/prepare_visdrone.py --zips data/VisDrone/zips --out data/VisDrone
  python training/train_visdrone.py --epochs 30 --imgsz 640

Why fine-tune instead of training from scratch: the COCO weights already know
edges, textures and object shapes; VisDrone only has to teach the model what
those objects look like from above and how small they are. Starting from the
same `yolov8n.pt` the tracker used before keeps the comparison fair: same
network, different training data.

The best checkpoint (highest validation mAP) ends up in
outputs/train/<name>/weights/best.pt; use it with
  python run_tracker.py --source video.mp4 --weights outputs/train/<name>/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracker.detector import pick_device  # noqa: E402


def use_pillow_decoder() -> None:
    """Decode training JPEGs with Pillow instead of OpenCV.

    On an M2 Mac with opencv-python-headless 5.0, training crashed once mid-epoch with a
    segmentation fault inside OpenCV's JPEG decoder (jsimd_ycc_extbgr_convert_neon). Every image
    decoded fine on its own, so it was not a bad file. Pillow has its own JPEG decoder, so the
    data loader is pointed at that instead. Output is the same: a BGR uint8 array.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    import ultralytics.data.base as base

    original = base.imread

    def imread(filename, flags=cv2.IMREAD_COLOR):
        if flags != cv2.IMREAD_COLOR or not str(filename).lower().endswith((".jpg", ".jpeg")):
            return original(filename, flags)
        try:
            with Image.open(filename) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))   # cv2 also applies EXIF rotation
        except OSError:
            return None
        return np.ascontiguousarray(rgb[:, :, ::-1])                          # RGB -> BGR, like cv2

    base.imread = imread


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolov8n.pt", help="starting weights (COCO-pretrained)")
    p.add_argument("--data", default=str(ROOT / "data/VisDrone/visdrone.yaml"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--time", type=float, default=None,
                   help="train for this many hours instead; the LR schedule is fitted to the time budget")
    p.add_argument("--close-mosaic", type=int, default=10, help="turn mosaic augmentation off for the last N epochs")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--fraction", type=float, default=1.0, help="share of the training set to use (speed tests)")
    p.add_argument("--cache", default=None, help="'disk' stores decoded images as .npy to skip JPEG decoding")
    p.add_argument("--device", default=None, help="cpu, mps or cuda:0 (default: auto)")
    p.add_argument("--name", default="visdrone")
    p.add_argument("--decoder", choices=["pillow", "opencv"], default="pillow", help="JPEG decoder for training images")
    p.add_argument("--resume", action="store_true", help="continue run --name from its last.pt (after a crash)")
    a = p.parse_args()

    from ultralytics import YOLO

    if a.decoder == "pillow":
        use_pillow_decoder()
    if a.resume:
        model = YOLO(str(ROOT / "outputs/train" / a.name / "weights/last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(a.model)
        model.train(data=a.data, epochs=a.epochs, time=a.time, close_mosaic=a.close_mosaic,
                    imgsz=a.imgsz, batch=a.batch, workers=a.workers,
                    fraction=a.fraction, cache=a.cache or False, device=a.device or pick_device(),
                    project=str(ROOT / "outputs/train"), name=a.name, exist_ok=True, seed=0, plots=True)
    # VisDrone val images hold up to 317 objects; the default max_det=300 would cut recall and lower mAP
    # (training raises it by itself, a standalone val() does not)
    metrics = model.val(data=a.data, imgsz=a.imgsz, batch=a.batch, device=a.device or pick_device(), max_det=1000,
                        project=str(ROOT / "outputs/train"), name=f"{a.name}_val", exist_ok=True)
    per_class = {model.names[int(i)]: round(float(metrics.box.maps[int(i)]), 3) for i in metrics.box.ap_class_index}
    print(json.dumps({"mAP50": round(float(metrics.box.map50), 3), "mAP50-95": round(float(metrics.box.map), 3),
                      "mAP50-95 per class": per_class}, indent=2))


if __name__ == "__main__":
    main()
