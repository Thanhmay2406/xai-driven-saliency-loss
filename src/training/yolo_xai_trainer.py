from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.losses import saliency_alignment_loss
from src.xai.activation_attention import ActivationAttention
from src.xai.eigencam import EigenCAM
from src.xai.gradcam import GradCAM
from src.xai.gradcampp import GradCAMPlusPlus
from src.xai.saliency_utils import find_yolo_saliency_target_layer


@dataclass
class UltralyticsYOLOXAITrainerConfig:
    xai_method: str = "activation"
    lambda_saliency: float = 0.1
    num_classes: int = 1
    target_layer: torch.nn.Module | None = None
    prefer_branch: str = "small"
    use_progressive_lambda: bool = False
    progressive_warmup_epochs: int = 20
    gt_mask_mode: str = "hard"
    soft_mask_sigma: float = 0.35
    shrunk_box_ratio: float = 0.7


@dataclass
class UltralyticsYOLOXAIStepOutput:
    total_loss: torch.Tensor
    detection_loss: torch.Tensor
    saliency_loss: torch.Tensor
    lambda_saliency: float
    saliency_maps: torch.Tensor
    gt_masks: torch.Tensor
    loss_items: Any
    raw_detection_output: Any


class UltralyticsYOLOXAITrainer:
    # Lop nay giu flow huan luyen gan voi baseline YOLO thong qua batch dict cua Ultralytics.
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: UltralyticsYOLOXAITrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config or UltralyticsYOLOXAITrainerConfig()
        self.target_layer = self.config.target_layer or find_yolo_saliency_target_layer(
            self.model,
            prefer_branch=self.config.prefer_branch,
        )
        self.xai = self._build_xai_method()

    # Ham nay dam bao model nam cung device voi anh trong batch truoc khi forward.
    def _ensure_model_device(self, device: torch.device) -> None:
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            return
        if model_device != device:
            self.model.to(device)

    # Dam bao cac tham so cua model khong bi frozen ngoai y muon sau khi load checkpoint/val.
    def _ensure_trainable_model(self) -> None:
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                parameter.requires_grad_(True)

    # Bao loi ro rang neu mot tensor quan trong da roi khoi computational graph.
    def _ensure_grad_path(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.requires_grad:
            return
        raise RuntimeError(
            f"`{name}` does not require grad. "
            "The model is likely running under a no-grad/inference context or was loaded with frozen parameters. "
            "Ensure training uses `xai_method='activation'`, the model is unfrozen, and the step runs under "
            "`torch.enable_grad()`."
        )

    # Chuan hoa loss ve scalar de backward on dinh qua cac phien ban Ultralytics.
    def _reduce_loss_tensor(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        if loss.ndim == 0 or loss.numel() == 1:
            return loss.reshape(())
        if not loss.requires_grad and loss.ndim > 0:
            return loss.sum()
        reduced = loss.sum()
        if reduced.ndim != 0:
            raise RuntimeError(f"`{name}` could not be reduced to a scalar loss. Got shape {tuple(loss.shape)}.")
        return reduced

    # Ham nay tao XAI method dua tren cau hinh.
    def _build_xai_method(self) -> ActivationAttention | GradCAM | GradCAMPlusPlus | EigenCAM:
        method_name = self.config.xai_method.strip().lower()
        if method_name in {"activation", "attention", "activation_attention"}:
            return ActivationAttention(self.model, self.target_layer)
        if method_name == "gradcam":
            return GradCAM(self.model, self.target_layer)
        if method_name in {"gradcam++", "gradcampp"}:
            return GradCAMPlusPlus(self.model, self.target_layer)
        if method_name == "eigencam":
            return EigenCAM(self.model, self.target_layer)
        raise ValueError(f"Unsupported xai_method: {self.config.xai_method}")

    # Ham nay tinh lambda saliency theo epoch neu bat warmup.
    def get_lambda_saliency(self, epoch: int | None = None) -> float:
        base_lambda = float(self.config.lambda_saliency)
        if not self.config.use_progressive_lambda or epoch is None:
            return base_lambda

        warmup_epochs = max(1, int(self.config.progressive_warmup_epochs))
        scale = min(max(epoch, 0) / warmup_epochs, 1.0)
        return base_lambda * scale

    # Ham nay trich detection loss tu output training goc cua model YOLO.
    def compute_detection_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, Any, Any]:
        raw_output = self.model(batch)

        if isinstance(raw_output, torch.Tensor):
            loss = self._reduce_loss_tensor("detection_loss", raw_output)
            loss_items = raw_output.detach() if raw_output.ndim > 0 else None
            return loss, loss_items, raw_output

        if isinstance(raw_output, dict) and "loss" in raw_output:
            loss = raw_output["loss"]
            loss_items = raw_output.get("loss_items")
            if not isinstance(loss, torch.Tensor):
                raise TypeError("Expected `loss` in model output to be a torch.Tensor.")
            return self._reduce_loss_tensor("detection_loss", loss), loss_items, raw_output

        if isinstance(raw_output, (tuple, list)) and raw_output:
            loss = raw_output[0]
            loss_items = raw_output[1] if len(raw_output) > 1 else None
            if not isinstance(loss, torch.Tensor):
                raise TypeError("Expected first item of model output to be a torch.Tensor loss.")
            return self._reduce_loss_tensor("detection_loss", loss), loss_items, raw_output

        raise TypeError(
            "Unsupported YOLO training output. Expected tensor, dict with `loss`, or tuple/list `(loss, loss_items)`."
        )

    # Ham nay tao mask GT tu batch dict cua Ultralytics.
    def build_gt_masks(self, batch: dict[str, Any]) -> torch.Tensor:
        images = batch["img"]
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("Expected `batch['img']` to be a tensor with shape (N, C, H, W).")

        device = images.device
        batch_size, _, height, width = images.shape
        masks = torch.zeros((batch_size, 1, height, width), dtype=torch.float32, device=device)

        batch_idx = torch.as_tensor(batch.get("batch_idx", []), device=device).reshape(-1).long()
        bboxes = torch.as_tensor(batch.get("bboxes", []), dtype=torch.float32, device=device)
        box_format = str(batch.get("bbox_format", "xywhn")).lower()

        if bboxes.numel() == 0:
            return masks
        if bboxes.ndim == 1:
            bboxes = bboxes.unsqueeze(0)
        if bboxes.shape[-1] < 4:
            raise ValueError(f"Expected `bboxes` with 4 values, got shape {tuple(bboxes.shape)}.")
        if batch_idx.numel() != bboxes.shape[0]:
            raise ValueError("`batch_idx` and `bboxes` must have the same number of rows.")

        if box_format == "xywhn":
            x_center = bboxes[:, 0] * width
            y_center = bboxes[:, 1] * height
            box_w = bboxes[:, 2] * width
            box_h = bboxes[:, 3] * height
            x1 = x_center - box_w / 2.0
            y1 = y_center - box_h / 2.0
            x2 = x_center + box_w / 2.0
            y2 = y_center + box_h / 2.0
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        elif box_format == "xyxy":
            boxes_xyxy = bboxes[:, :4]
        else:
            raise ValueError(f"Unsupported bbox_format: {box_format}")

        mask_mode = self.config.gt_mask_mode.strip().lower()
        for sample_index, box in zip(batch_idx.tolist(), boxes_xyxy, strict=True):
            if sample_index < 0 or sample_index >= batch_size:
                continue
            if mask_mode == "shrunk":
                box = self._shrink_box(box, ratio=self.config.shrunk_box_ratio)

            x1 = int(torch.round(box[0]).clamp(0, width).item())
            y1 = int(torch.round(box[1]).clamp(0, height).item())
            x2 = int(torch.round(box[2]).clamp(0, width).item())
            y2 = int(torch.round(box[3]).clamp(0, height).item())
            if x2 <= x1 or y2 <= y1:
                continue

            if mask_mode == "soft":
                soft_region = self._build_soft_box_mask(
                    height=y2 - y1,
                    width=x2 - x1,
                    device=device,
                    sigma=max(float(self.config.soft_mask_sigma), 1e-3),
                )
                masks[sample_index, :, y1:y2, x1:x2] = torch.maximum(
                    masks[sample_index, :, y1:y2, x1:x2],
                    soft_region,
                )
            else:
                masks[sample_index, :, y1:y2, x1:x2] = 1.0

        return masks

    # Ham nay thu nho bbox ve gan tam defect de giam anh huong cua background.
    def _shrink_box(self, box: torch.Tensor, ratio: float) -> torch.Tensor:
        ratio = min(max(float(ratio), 1e-3), 1.0)
        x1, y1, x2, y2 = box.unbind()
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        half_w = (x2 - x1) * ratio / 2.0
        half_h = (y2 - y1) * ratio / 2.0
        return torch.stack(
            [
                center_x - half_w,
                center_y - half_h,
                center_x + half_w,
                center_y + half_h,
            ]
        )

    # Ham nay tao soft mask co gia tri cao o tam bbox va giam dan ra bien.
    def _build_soft_box_mask(self, *, height: int, width: int, device: torch.device, sigma: float) -> torch.Tensor:
        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float32)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        gaussian = torch.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
        gaussian = gaussian / gaussian.max().clamp_min(1e-8)
        return gaussian.unsqueeze(0)

    # Ham nay suy ra class target cho tung anh tu `cls` va `batch_idx`.
    def infer_class_ids(self, batch: dict[str, Any]) -> list[int]:
        images = batch["img"]
        batch_size = int(images.shape[0])
        cls = torch.as_tensor(batch.get("cls", []), device=images.device).reshape(-1)
        batch_idx = torch.as_tensor(batch.get("batch_idx", []), device=images.device).reshape(-1).long()

        class_ids = [0] * batch_size
        if cls.numel() == 0 or batch_idx.numel() == 0:
            return class_ids
        if cls.numel() != batch_idx.numel():
            raise ValueError("`cls` and `batch_idx` must have the same number of rows.")

        for sample_index in range(batch_size):
            matched = torch.nonzero(batch_idx == sample_index, as_tuple=False).flatten()
            if matched.numel() == 0:
                continue
            class_ids[sample_index] = int(cls[matched[0]].item())

        return class_ids

    # Ham nay sinh saliency map tung anh theo dung model YOLO dang huan luyen.
    def compute_saliency_maps(self, batch: dict[str, Any]) -> torch.Tensor:
        images = batch["img"]
        if isinstance(self.xai, ActivationAttention):
            return self.xai.generate_from_activations(target_size=tuple(images.shape[-2:]))

        class_ids = self.infer_class_ids(batch)
        saliency_maps = []

        for sample_index in range(images.shape[0]):
            image = images[sample_index : sample_index + 1]
            class_id = class_ids[sample_index]

            if isinstance(self.xai, EigenCAM):
                saliency = self.xai.generate(image)
            else:
                saliency, _ = self.xai.generate(
                    image_tensor=image,
                    class_id=class_id,
                    num_classes=self.config.num_classes,
                )
            saliency_maps.append(saliency.detach())

        return torch.cat(saliency_maps, dim=0)

    # Ham nay thuc hien mot training step theo batch format baseline YOLO.
    def training_step(self, batch: dict[str, Any], *, epoch: int | None = None) -> UltralyticsYOLOXAIStepOutput:
        images = batch["img"]
        if not isinstance(images, torch.Tensor):
            raise ValueError("Expected `batch['img']` to be a torch.Tensor before training_step.")
        self._ensure_model_device(images.device)
        self._ensure_trainable_model()

        with torch.enable_grad():
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            detection_loss, loss_items, raw_detection_output = self.compute_detection_loss(batch)
            if not isinstance(self.xai, ActivationAttention):
                raise ValueError(
                    "Differentiable saliency training requires `xai_method='activation'`. "
                    "Use Grad-CAM, Grad-CAM++, or EigenCAM for offline visualization/evaluation only."
                )

            saliency_maps = self.compute_saliency_maps(batch)
            gt_masks = self.build_gt_masks(batch)
            saliency_loss = saliency_alignment_loss(saliency_maps, gt_masks)

            lambda_saliency = self.get_lambda_saliency(epoch)
            total_loss = detection_loss + (lambda_saliency * saliency_loss)

            self._ensure_grad_path("detection_loss", detection_loss)
            self._ensure_grad_path("saliency_maps", saliency_maps)
            self._ensure_grad_path("saliency_loss", saliency_loss)
            self._ensure_grad_path("total_loss", total_loss)

            total_loss.backward()
            self.optimizer.step()

        return UltralyticsYOLOXAIStepOutput(
            total_loss=total_loss.detach(),
            detection_loss=detection_loss.detach(),
            saliency_loss=saliency_loss.detach(),
            lambda_saliency=lambda_saliency,
            saliency_maps=saliency_maps.detach(),
            gt_masks=gt_masks.detach(),
            loss_items=loss_items,
            raw_detection_output=raw_detection_output,
        )

    # Ham nay giai phong hook XAI sau khi train.
    def close(self) -> None:
        self.xai.close()
