from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.masks.bbox_mask import (
    create_bbox_mask_from_label_file,
    label_path_from_image,
    load_yolo_boxes,
    overlay_mask_on_image,
    save_mask_npy,
    save_mask_png,
)
from src.xai.saliency_utils import get_split_image_dir, get_split_label_dir, list_image_files, load_yaml, resolve_path


# Ham nay parse tham so command line cho script sinh GT bbox mask.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate binary GT bbox masks from YOLO labels.")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "merged_yolo_grouped" / "dataset.yaml",
        help="Path to dataset YAML.",
    )
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-images", type=int, default=0, help="0 means process the full split.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "masks",
        help="Directory to store generated masks.",
    )
    return parser.parse_args()


# Ham nay luu metadata tong hop cua mot split sau khi sinh mask.
def save_summary(items: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"items": items}, handle, indent=2, ensure_ascii=False)


# Ham nay chay toan bo pipeline tao bbox mask cho mot split cua dataset.
def main() -> None:
    args = parse_args()
    dataset_yaml_path = resolve_path(args.data, base_dir=args.data.parent, repo_root=REPO_ROOT)
    dataset_config = load_yaml(dataset_yaml_path)
    image_dir = get_split_image_dir(dataset_config, dataset_yaml_path, split=args.split, repo_root=REPO_ROOT)
    label_dir = get_split_label_dir(image_dir)

    image_paths = list_image_files(image_dir)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    split_dir = args.output_dir.resolve() / args.split
    png_dir = split_dir / "png"
    npy_dir = split_dir / "npy"
    overlay_dir = split_dir / "overlay"
    png_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        label_path = label_path_from_image(image_path, label_dir)
        boxes = load_yolo_boxes(label_path)
        mask = create_bbox_mask_from_label_file(label_path, image.size)

        save_mask_png(mask, png_dir / f"{image_path.stem}.png")
        save_mask_npy(mask, npy_dir / f"{image_path.stem}.npy")
        overlay_mask_on_image(image, mask).save(overlay_dir / f"{image_path.stem}.png")

        summary.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "num_boxes": len(boxes),
                "mask_shape": list(mask.shape),
                "mask_area_pixels": int(mask.sum()),
                "mask_fill_ratio": float(mask.mean()) if mask.size else 0.0,
            }
        )

    save_summary(summary, split_dir / "summary.json")
    print(json.dumps({"count": len(summary), "output_dir": str(split_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
