import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw


# Ham nay doc file YAML va tra ve du lieu dang dict.
def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


# Ham nay resolve duong dan tuong doi thanh duong dan tuyet doi theo repo.
def resolve_path(path_str: str | Path, *, base_dir: Path, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (repo_root / path).resolve()


# Ham nay tim thu muc chua anh cua split can xu ly trong dataset YOLO.
def get_split_image_dir(dataset_config: dict[str, Any], dataset_yaml_path: Path, split: str, repo_root: Path) -> Path:
    dataset_root = resolve_path(dataset_config.get("path", "."), base_dir=dataset_yaml_path.parent, repo_root=repo_root)
    split_rel = dataset_config.get(split)
    if not split_rel:
        raise ValueError(f"Missing `{split}` entry in dataset YAML: {dataset_yaml_path}")
    return (dataset_root / split_rel).resolve()


# Ham nay tim thu muc chua label cua split dua tren cau truc YOLO images/<split> -> labels/<split>.
def get_split_label_dir(image_dir: Path) -> Path:
    if image_dir.parent.name != "images":
        raise ValueError(f"Expected image dir under images/<split>, got: {image_dir}")
    return image_dir.parent.parent / "labels" / image_dir.name


# Ham nay liet ke cac file anh trong mot thu muc split.
def list_image_files(image_dir: Path) -> list[Path]:
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in valid_suffixes)


# Ham nay doc anh goc de phuc vu ve heatmap va bbox.
def load_rgb_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


# Ham nay resize anh va chuyen thanh tensor dau vao cho YOLO.
def preprocess_image(image: Image.Image, imgsz: int, device: str) -> torch.Tensor:
    resized = image.resize((imgsz, imgsz))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    tensor = torch.from_numpy(image_array).unsqueeze(0)
    return tensor.to(device)


# Ham nay chuyen tensor CAM ve dang 2D va normalize ve [0, 1].
def normalize_cam(cam_tensor: torch.Tensor) -> np.ndarray:
    cam = cam_tensor.detach().float().cpu().squeeze().numpy()
    cam = np.maximum(cam, 0.0)
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max - cam_min < 1e-12:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - cam_min) / (cam_max - cam_min)).astype(np.float32)


# Ham nay resize saliency map ve cung kich thuoc voi anh goc.
def resize_cam(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    cam_image = Image.fromarray((cam * 255).astype(np.uint8))
    resized = cam_image.resize(size, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


# Ham nay tao bang mau heatmap don gian de truc quan hoa saliency.
def apply_simple_colormap(cam: np.ndarray) -> np.ndarray:
    red = np.clip(2.0 * cam - 0.1, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * cam - 1.0) * 2.0, 0.0, 1.0)
    blue = np.clip(1.2 - 2.0 * cam, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


# Ham nay tron heatmap voi anh goc de tao anh overlay.
def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    heatmap = apply_simple_colormap(cam)
    blended = np.clip((1.0 - alpha) * image_array + alpha * heatmap, 0.0, 1.0)
    return Image.fromarray((blended * 255).astype(np.uint8))


# Ham nay doc nhan YOLO va chuyen bbox normalize thanh toa do pixel xyxy.
def load_ground_truth_boxes(label_path: Path, image_size: tuple[int, int], names: dict[int, str]) -> list[dict[str, Any]]:
    width, height = image_size
    if not label_path.exists():
        return []

    boxes: list[dict[str, Any]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id_str, x_center_str, y_center_str, box_w_str, box_h_str = line.split()
        class_id = int(class_id_str)
        x_center = float(x_center_str) * width
        y_center = float(y_center_str) * height
        box_w = float(box_w_str) * width
        box_h = float(box_h_str) * height

        x1 = max(0.0, x_center - box_w / 2.0)
        y1 = max(0.0, y_center - box_h / 2.0)
        x2 = min(float(width), x_center + box_w / 2.0)
        y2 = min(float(height), y_center + box_h / 2.0)

        boxes.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return boxes


# Ham nay chuyen du doan cua YOLO thanh danh sach bbox de luu va ve.
def predictions_to_dicts(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    if result.boxes is None:
        return []

    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    confs = result.boxes.conf.detach().cpu().tolist()
    predictions = []
    for box, class_id, conf in zip(xyxy, classes, confs, strict=True):
        class_idx = int(class_id)
        predictions.append(
            {
                "class_id": class_idx,
                "class_name": names.get(class_idx, str(class_idx)),
                "confidence": float(conf),
                "xyxy": [float(value) for value in box],
            }
        )
    return predictions


# Ham nay chon prediction co do tin cay cao nhat de dung lam target cho saliency.
def select_primary_prediction(predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not predictions:
        return None
    return max(predictions, key=lambda item: item["confidence"])


# Ham nay thu thap de quy tat ca tensor nam trong output cua model.
def collect_output_tensors(output: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []

    def _collect(node: Any) -> None:
        if isinstance(node, torch.Tensor):
            tensors.append(node)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                _collect(item)
            return
        if isinstance(node, dict):
            for item in node.values():
                _collect(item)

    _collect(output)
    return tensors


# Ham nay tim tensor du doan phu hop nhat trong output thap cua model YOLO.
def find_primary_output_tensor(output: Any) -> torch.Tensor:
    tensors = collect_output_tensors(output)
    if not tensors:
        raise ValueError("No tensor found in model output.")

    tensors_with_grad = [tensor for tensor in tensors if tensor.requires_grad]
    candidate_pool = tensors_with_grad or tensors

    def _priority(tensor: torch.Tensor) -> tuple[int, int, int]:
        supports_class_scores = 1 if _supports_class_dimension(tensor, num_classes=1) else 0
        return (supports_class_scores, int(tensor.requires_grad), tensor.numel())

    return max(candidate_pool, key=_priority)


# Ham nay kiem tra tensor co dang phu hop de cat class score hay khong.
def _supports_class_dimension(tensor: torch.Tensor, num_classes: int) -> bool:
    if tensor.ndim == 3:
        return tensor.shape[1] >= 4 + num_classes or tensor.shape[2] >= 4 + num_classes
    if tensor.ndim == 4:
        return tensor.shape[1] >= 4 + num_classes or tensor.shape[-1] >= 4 + num_classes
    return False


# Ham nay trich candidate target score tu mot tensor 3D/4D cua detector.
def extract_class_score_candidates(tensor: torch.Tensor, class_id: int, num_classes: int) -> list[torch.Tensor]:
    candidates: list[torch.Tensor] = []

    if tensor.ndim == 3:
        if tensor.shape[1] >= 4 + num_classes:
            class_scores = tensor[:, 4 : 4 + num_classes, :]
            candidates.append(class_scores[:, class_id, :].max())
        if tensor.shape[2] >= 4 + num_classes:
            class_scores = tensor[:, :, 4 : 4 + num_classes]
            candidates.append(class_scores[:, :, class_id].max())
        return candidates

    if tensor.ndim == 4:
        if tensor.shape[1] >= 4 + num_classes:
            class_scores = tensor[:, 4 : 4 + num_classes, :, :]
            candidates.append(class_scores[:, class_id, :, :].max())
        if tensor.shape[-1] >= 4 + num_classes:
            class_scores = tensor[:, :, :, 4 : 4 + num_classes]
            candidates.append(class_scores[:, :, :, class_id].max())

    return candidates


# Ham nay trich scalar target cho mot class tu output tensor cua detector.
def build_detection_target(output: Any, class_id: int, num_classes: int) -> torch.Tensor:
    tensors = collect_output_tensors(output)
    if not tensors:
        raise ValueError("No tensor found in model output.")

    grad_candidates: list[torch.Tensor] = []
    fallback_candidates: list[torch.Tensor] = []

    for tensor in tensors:
        extracted = extract_class_score_candidates(tensor, class_id=class_id, num_classes=num_classes)
        if not extracted:
            continue
        if tensor.requires_grad:
            grad_candidates.extend(extracted)
        else:
            fallback_candidates.extend(extracted)

    if grad_candidates:
        return torch.stack([candidate.reshape(1) for candidate in grad_candidates]).max()
    if fallback_candidates:
        candidate = torch.stack([item.reshape(1) for item in fallback_candidates]).max()
        raise RuntimeError(
            "Found class-score tensors, but none require gradients. "
            "This usually means the selected output branch was detached from autograd."
        )

    tensor = find_primary_output_tensor(output)
    raise ValueError(
        "Could not infer class score layout from model output. "
        f"Selected tensor shape was {tuple(tensor.shape)}."
    )


# Ham nay tim layer conv cuoi cung de dat hook cho cac phuong phap CAM.
def find_last_conv_layer(module: torch.nn.Module) -> torch.nn.Module:
    conv_layers = [child for child in module.modules() if isinstance(child, torch.nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No Conv2d layer found to use as saliency target layer.")
    return conv_layers[-1]


# Ham nay ve bbox ground truth va prediction len anh de de doi chieu.
def draw_boxes(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for box in ground_truth:
        x1, y1, x2, y2 = box["xyxy"]
        label = f"GT: {box['class_name']}"
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1 + 4, max(0.0, y1 - 12)), label, fill=(0, 255, 0))

    for box in predictions:
        x1, y1, x2, y2 = box["xyxy"]
        label = f"Pred: {box['class_name']} {box['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=2)
        draw.text((x1 + 4, min(canvas.height - 12, y1 + 4)), label, fill=(255, 64, 64))

    return canvas


# Ham nay ghep anh goc va anh overlay thanh mot panel de quan sat nhanh.
def make_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    width = left.width + right.width
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


# Ham nay luu metadata cua mot anh saliency de phuc vu phan tich sau nay.
def save_metadata(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


# Ham nay luu ma tran saliency ra file numpy de co the tinh metric o buoc sau.
def save_cam_array(cam: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, cam)


# Ham nay chuyen ten anh thanh duong dan label tuong ung theo dinh dang YOLO.
def label_path_from_image(image_path: Path, label_dir: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


# Ham nay resize activation map len kich thuoc dau vao bang noi suy bilinear.
def upsample_cam(cam_tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(cam_tensor, size=target_size, mode="bilinear", align_corners=False)
