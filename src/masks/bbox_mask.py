from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


# Ham nay doc file label YOLO va tra ve danh sach box dang normalize.
def load_yolo_boxes(label_path: Path) -> list[dict[str, float | int]]:
    if not label_path.exists():
        return []

    boxes: list[dict[str, float | int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id_str, x_center_str, y_center_str, width_str, height_str = line.split()
        boxes.append(
            {
                "class_id": int(class_id_str),
                "x_center": float(x_center_str),
                "y_center": float(y_center_str),
                "width": float(width_str),
                "height": float(height_str),
            }
        )
    return boxes


# Ham nay chuyen mot bbox YOLO normalize thanh toa do pixel xyxy.
def yolo_box_to_xyxy(box: dict[str, float | int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x_center = float(box["x_center"]) * width
    y_center = float(box["y_center"]) * height
    box_width = float(box["width"]) * width
    box_height = float(box["height"]) * height

    x1 = int(round(max(0.0, x_center - box_width / 2.0)))
    y1 = int(round(max(0.0, y_center - box_height / 2.0)))
    x2 = int(round(min(float(width), x_center + box_width / 2.0)))
    y2 = int(round(min(float(height), y_center + box_height / 2.0)))
    return x1, y1, x2, y2


# Ham nay tao mask nhi phan tu danh sach bbox YOLO cho mot anh.
def create_bbox_mask(
    boxes: list[dict[str, float | int]],
    image_size: tuple[int, int],
    *,
    dtype: Any = np.uint8,
) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=dtype)

    for box in boxes:
        x1, y1, x2, y2 = yolo_box_to_xyxy(box, image_size)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 1

    return mask


# Ham nay tao mask nhi phan truc tiep tu file label YOLO.
def create_bbox_mask_from_label_file(label_path: Path, image_size: tuple[int, int], *, dtype: Any = np.uint8) -> np.ndarray:
    boxes = load_yolo_boxes(label_path)
    return create_bbox_mask(boxes, image_size=image_size, dtype=dtype)


# Ham nay chuyen mask nhi phan thanh anh PIL de luu hoac hien thi.
def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


# Ham nay ve overlay mask len anh goc de kiem tra nhanh bbox mask.
def overlay_mask_on_image(
    image: Image.Image,
    mask: np.ndarray,
    *,
    alpha: float = 0.35,
    color: tuple[int, int, int] = (255, 64, 64),
) -> Image.Image:
    base = image.convert("RGB")
    overlay = Image.new("RGB", base.size, color=(0, 0, 0))
    mask_image = mask_to_image(mask)

    tint = Image.new("RGB", base.size, color=color)
    overlay.paste(tint, mask=mask_image)
    blended = Image.blend(base, overlay, alpha=alpha)
    return blended


# Ham nay ve vien bbox len anh de doi chieu voi mask khi can debug.
def draw_box_outlines(
    image: Image.Image,
    boxes: list[dict[str, float | int]],
    image_size: tuple[int, int],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    width: int = 2,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        x1, y1, x2, y2 = yolo_box_to_xyxy(box, image_size)
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return canvas


# Ham nay luu mask ra file PNG de phuc vu xem truc quan.
def save_mask_png(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_to_image(mask).save(output_path)


# Ham nay luu mask ra file numpy de su dung lai cho cac buoc tinh metric/loss.
def save_mask_npy(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mask.astype(np.uint8))


# Ham nay tra ve duong dan label tuong ung voi mot anh trong dataset YOLO.
def label_path_from_image(image_path: Path, label_dir: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"
