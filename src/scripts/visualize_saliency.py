import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - import guard for local setup
    raise SystemExit(
        "Missing dependency: ultralytics. Install it with `pip install ultralytics` before generating saliency maps."
    ) from exc

from src.xai.eigencam import EigenCAM
from src.xai.gradcam import GradCAM
from src.xai.gradcampp import GradCAMPlusPlus
from src.xai.saliency_utils import (
    draw_boxes,
    find_last_conv_layer,
    get_split_image_dir,
    get_split_label_dir,
    label_path_from_image,
    list_image_files,
    load_ground_truth_boxes,
    load_rgb_image,
    load_yaml,
    make_side_by_side,
    overlay_heatmap,
    predictions_to_dicts,
    preprocess_image,
    resolve_path,
    save_cam_array,
    save_metadata,
    select_primary_prediction,
)


# Ham nay parse tham so command line cho script sinh saliency map.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate saliency maps for a trained YOLO baseline.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to baseline YOLO checkpoint.")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "merged_yolo_grouped" / "dataset.yaml",
        help="Path to YOLO dataset YAML.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        choices=("gradcam", "gradcampp", "eigencam"),
        help="Saliency method to apply.",
    )
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO predictions.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu, cuda:0.")
    parser.add_argument("--max-images", type=int, default=20, help="Maximum number of images to process.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "saliency_maps" / "baseline",
        help="Directory to store saliency outputs.",
    )
    return parser.parse_args()


# Ham nay tao doi tuong CAM tuong ung voi method duoc chon.
def build_cam_generator(method: str, detector: torch.nn.Module, target_layer: torch.nn.Module) -> Any:
    if method == "gradcam":
        return GradCAM(detector, target_layer)
    if method == "gradcampp":
        return GradCAMPlusPlus(detector, target_layer)
    if method == "eigencam":
        return EigenCAM(detector, target_layer)
    raise ValueError(f"Unsupported saliency method: {method}")


# Ham nay sinh mot saliency map cho tung anh va tra ve CAM cung score target.
def generate_saliency_map(
    cam_generator: Any,
    method: str,
    image_tensor: torch.Tensor,
    class_id: int,
    num_classes: int,
) -> tuple[np.ndarray, float | None]:
    if method == "eigencam":
        cam_tensor = cam_generator.generate_normalized(image_tensor=image_tensor)
        return cam_tensor.detach().cpu().numpy(), None

    cam_tensor, target_score = cam_generator.generate_normalized(
        image_tensor=image_tensor,
        class_id=class_id,
        num_classes=num_classes,
    )
    return cam_tensor.detach().cpu().numpy(), target_score


# Ham nay luu tat ca artifact saliency cua mot anh gom overlay, panel, numpy va metadata.
def save_saliency_artifacts(
    image_path: Path,
    output_dir: Path,
    boxed_image: Image.Image,
    overlay_image: Image.Image,
    cam_array: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    heatmap_only = Image.fromarray((cam_array * 255).astype(np.uint8)).convert("L")
    side_by_side = make_side_by_side(boxed_image, overlay_image)

    boxed_image.save(output_dir / f"{stem}_boxes.png")
    overlay_image.save(output_dir / f"{stem}_overlay.png")
    heatmap_only.save(output_dir / f"{stem}_heatmap.png")
    side_by_side.save(output_dir / f"{stem}_panel.png")
    save_cam_array(cam_array, output_dir / f"{stem}_cam.npy")
    save_metadata(metadata, output_dir / f"{stem}_meta.json")


# Ham nay chay toan bo pipeline sinh saliency cho mot checkpoint baseline.
def main() -> None:
    args = parse_args()
    dataset_yaml_path = resolve_path(args.data, base_dir=args.data.parent, repo_root=REPO_ROOT)
    dataset_config = load_yaml(dataset_yaml_path)
    image_dir = get_split_image_dir(dataset_config, dataset_yaml_path, split=args.split, repo_root=REPO_ROOT)
    label_dir = get_split_label_dir(image_dir)

    weights_path = resolve_path(args.weights, base_dir=args.weights.parent, repo_root=REPO_ROOT)
    output_dir = args.output_dir.resolve() / args.method / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    detector = model.model
    detector.eval()
    target_layer = find_last_conv_layer(detector)
    cam_generator = build_cam_generator(args.method, detector=detector, target_layer=target_layer)

    names = {int(key): value for key, value in model.names.items()}
    image_paths = list_image_files(image_dir)[: args.max_images]

    summary: list[dict[str, Any]] = []
    try:
        for image_path in image_paths:
            original_image = load_rgb_image(image_path)
            predictions_result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            predictions = predictions_to_dicts(predictions_result, names)
            primary_prediction = select_primary_prediction(predictions)
            if primary_prediction is None:
                continue

            image_tensor = preprocess_image(original_image, imgsz=args.imgsz, device=args.device)
            cam_array, target_score = generate_saliency_map(
                cam_generator=cam_generator,
                method=args.method,
                image_tensor=image_tensor,
                class_id=int(primary_prediction["class_id"]),
                num_classes=len(names),
            )
            cam_resized = np.array(Image.fromarray((cam_array.squeeze() * 255).astype(np.uint8)).resize(original_image.size)) / 255.0

            ground_truth = load_ground_truth_boxes(
                label_path=label_path_from_image(image_path, label_dir),
                image_size=original_image.size,
                names=names,
            )
            boxed_image = draw_boxes(original_image, ground_truth=ground_truth, predictions=predictions)
            overlay_image = overlay_heatmap(boxed_image, cam_resized)

            metadata = {
                "image_path": str(image_path),
                "method": args.method,
                "target_layer": target_layer.__class__.__name__,
                "primary_prediction": primary_prediction,
                "target_score": target_score,
                "ground_truth": ground_truth,
                "predictions": predictions,
            }
            save_saliency_artifacts(
                image_path=image_path,
                output_dir=output_dir,
                boxed_image=boxed_image,
                overlay_image=overlay_image,
                cam_array=cam_resized,
                metadata=metadata,
            )
            summary.append(metadata)
    finally:
        cam_generator.close()

    save_metadata({"items": summary}, output_dir / "summary.json")


if __name__ == "__main__":
    main()
