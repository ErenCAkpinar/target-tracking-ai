"""Unpack VisDrone2019-DET and convert its labels to YOLO format.

Usage:
  python training/prepare_visdrone.py --zips data/VisDrone/zips --out data/VisDrone

VisDrone annotation line (one object per line, pixels):
  x_left, y_top, width, height, score, category, truncation, occlusion
- score 0 marks an "ignored region", which is not a real object, so it is skipped.
- category 0 (ignored region) and 11 (others) are skipped too; 1..10 become YOLO classes 0..9.

YOLO label line (one object per line, normalised to 0..1):
  class x_center y_center width height
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from PIL import Image

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
         "awning-tricycle", "bus", "motor"]
SPLITS = {"VisDrone2019-DET-train": "train", "VisDrone2019-DET-val": "val"}


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract, refusing absolute paths or '..' (a malicious zip could write outside dest)."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe path in {zip_path.name}: {name}")
        z.extractall(dest)


def convert_line(row: list[str], img_w: int, img_h: int) -> str | None:
    x, y, w, h, score, cat = (int(v) for v in row[:6])
    if score == 0 or not 1 <= cat <= 10 or w <= 0 or h <= 0:
        return None
    return f"{cat - 1} {(x + w / 2) / img_w:.6f} {(y + h / 2) / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def convert_split(root: Path, source: str, split: str) -> tuple[int, int]:
    src = root / source
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img in (src / "images").glob("*.jpg"):
        img.rename(img_dir / img.name)
    n_img = n_obj = 0
    for ann in sorted((src / "annotations").glob("*.txt")):
        w, h = Image.open(img_dir / ann.with_suffix(".jpg").name).size
        rows = [r.split(",") for r in ann.read_text().strip().splitlines() if r.strip()]
        lines = [line for r in rows if (line := convert_line(r, w, h))]
        (lbl_dir / ann.name).write_text("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        n_obj += len(lines)
    return n_img, n_obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zips", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.out.resolve()
    for source, split in SPLITS.items():
        if (root / "labels" / split).exists():
            print(f"{split}: already prepared")
        elif not (a.zips / f"{source}.zip").exists():
            print(f"{split}: {source}.zip not found, skipped")
        else:
            safe_extract(a.zips / f"{source}.zip", root)
            n_img, n_obj = convert_split(root, source, split)
            shutil.rmtree(root / source)        # images were moved out; only the raw annotations remain
            print(f"{split}: {n_img} images, {n_obj} objects")
    yaml = root / "visdrone.yaml"
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n"
                    + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    print("wrote", yaml)


if __name__ == "__main__":
    main()
