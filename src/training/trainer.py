from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from src.losses import saliency_alignment_loss
from src.xai.activation_attention import ActivationAttention
from src.xai.eigencam import EigenCAM
from src.xai.gradcam import GradCAM
from src.xai.gradcampp import GradCAMPlusPlus
from src.xai.saliency_utils import find_yolo_saliency_target_layer


DetectionLossFn = Callable[[Any, Any], torch.Tensor]


@dataclass
class XAITrainerConfig:
    xai_method: str = "activation"
    lambda_saliency: float = 0.1
    num_classes: int = 1
    target_layer: torch.nn.Module | None = None
    prefer_branch: str = "small"
    normalize_gt_masks: bool = False
    use_progressive_lambda: bool = False
    progressive_warmup_epochs: int = 20


@dataclass
class XAITrainerStepOutput:
    total_loss: torch.Tensor
    detection_loss: torch.Tensor
    saliency_loss: torch.Tensor
    lambda_saliency: float
    saliency_maps: torch.Tensor
    gt_masks: torch.Tensor
    predictions: Any


class XAITrainer:
    # Lop nay quan ly mot training step gom loss detection va loss saliency.
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        detection_loss_fn: DetectionLossFn,
        config: XAITrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.detection_loss_fn = detection_loss_fn
        self.config = config or XAITrainerConfig()
        self.target_layer = self.config.target_layer or find_yolo_saliency_target_layer(
            self.model,
            prefer_branch=self.config.prefer_branch,
        )
        self.xai = self._build_xai_method()

    # Ham nay dam bao model nam cung device voi batch dau vao truoc khi forward.
    def _ensure_model_device(self, device: torch.device) -> None:
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            return
        if model_device != device:
            self.model.to(device)

    # Khoi phuc training cho model neu checkpoint hoac pipeline truoc do da freeze tham so.
    def _ensure_trainable_model(self) -> None:
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                parameter.requires_grad_(True)

    # Bao loi som neu graph autograd bi mat.
    def _ensure_grad_path(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.requires_grad:
            return
        raise RuntimeError(
            f"`{name}` does not require grad. "
            "The model is likely running under a no-grad/inference context or was loaded with frozen parameters."
        )

    # Chuan hoa loss ve scalar de backward hoat dong on dinh.
    def _reduce_loss_tensor(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        if loss.ndim == 0 or loss.numel() == 1:
            return loss.reshape(())
        reduced = loss.sum()
        if reduced.ndim != 0:
            raise RuntimeError(f"`{name}` could not be reduced to a scalar loss. Got shape {tuple(loss.shape)}.")
        return reduced

    # Ham nay tao bo sinh saliency theo cau hinh da chon.
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

    # Ham nay tinh lambda theo epoch neu muon warmup anh huong cua saliency loss.
    def get_lambda_saliency(self, epoch: int | None = None) -> float:
        base_lambda = float(self.config.lambda_saliency)
        if not self.config.use_progressive_lambda or epoch is None:
            return base_lambda

        warmup_epochs = max(1, int(self.config.progressive_warmup_epochs))
        scale = min(max(epoch, 0) / warmup_epochs, 1.0)
        return base_lambda * scale

    # Ham nay chuyen targets thanh batch mask co cung batch size voi images.
    def build_gt_masks(self, targets: Any, image_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
        height, width = image_hw

        if isinstance(targets, torch.Tensor):
            masks = targets.float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            elif masks.ndim == 4:
                pass
            else:
                raise ValueError(f"Expected gt mask tensor with 3 or 4 dims, got {tuple(masks.shape)}.")
            return masks.to(device)

        if not isinstance(targets, Sequence):
            raise TypeError("targets must be a torch.Tensor or a sequence of per-image annotations.")

        mask_batch = []
        for sample_targets in targets:
            mask = self._build_single_mask(sample_targets, image_hw=image_hw, device=device)
            mask_batch.append(mask)

        return torch.stack(mask_batch, dim=0)

    # Ham nay tao mask cho mot anh tu danh sach bbox theo dinh dang pho bien.
    def _build_single_mask(self, sample_targets: Any, image_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
        height, width = image_hw
        mask = torch.zeros((1, height, width), dtype=torch.float32, device=device)

        if sample_targets is None:
            return mask

        if isinstance(sample_targets, dict):
            if "mask" in sample_targets:
                sample_mask = torch.as_tensor(sample_targets["mask"], dtype=torch.float32, device=device)
                if sample_mask.ndim == 2:
                    sample_mask = sample_mask.unsqueeze(0)
                elif sample_mask.ndim != 3:
                    raise ValueError("Target `mask` must have shape (H, W) or (C, H, W).")
                return sample_mask

            boxes = sample_targets.get("boxes", sample_targets.get("bboxes", []))
            box_format = str(sample_targets.get("box_format", "xyxy")).lower()
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32, device=device)
            return self._rasterize_boxes(boxes_tensor, mask, box_format=box_format)

        boxes_tensor = torch.as_tensor(sample_targets, dtype=torch.float32, device=device)
        return self._rasterize_boxes(boxes_tensor, mask, box_format="xyxy")

    # Ham nay ve bbox len mask, ho tro ca xyxy va xywh normalize.
    def _rasterize_boxes(self, boxes: torch.Tensor, mask: torch.Tensor, box_format: str) -> torch.Tensor:
        if boxes.numel() == 0:
            return mask

        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        if boxes.shape[-1] < 4:
            raise ValueError(f"Expected boxes with 4 values, got shape {tuple(boxes.shape)}.")

        _, height, width = mask.shape

        if box_format == "xywhn":
            x_center = boxes[:, 0] * width
            y_center = boxes[:, 1] * height
            box_w = boxes[:, 2] * width
            box_h = boxes[:, 3] * height
            x1 = x_center - box_w / 2.0
            y1 = y_center - box_h / 2.0
            x2 = x_center + box_w / 2.0
            y2 = y_center + box_h / 2.0
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        elif box_format != "xyxy":
            raise ValueError(f"Unsupported box_format: {box_format}")

        for box in boxes[:, :4]:
            x1 = int(torch.round(box[0]).clamp(0, width).item())
            y1 = int(torch.round(box[1]).clamp(0, height).item())
            x2 = int(torch.round(box[2]).clamp(0, width).item())
            y2 = int(torch.round(box[3]).clamp(0, height).item())
            if x2 <= x1 or y2 <= y1:
                continue
            mask[:, y1:y2, x1:x2] = 1.0

        return mask

    # Ham nay chon class target cho XAI tu ground truth neu co, neu khong thi dung mac dinh 0.
    def infer_class_id(self, sample_targets: Any) -> int:
        if isinstance(sample_targets, dict):
            labels = sample_targets.get("labels", sample_targets.get("class_ids"))
            if labels is not None:
                labels_tensor = torch.as_tensor(labels)
                if labels_tensor.numel() > 0:
                    return int(labels_tensor.flatten()[0].item())
            if "class_id" in sample_targets:
                return int(sample_targets["class_id"])
        return 0

    # Ham nay sinh saliency map cho tung anh trong batch bang XAI method da chon.
    def compute_saliency_maps(self, images: torch.Tensor, targets: Any) -> torch.Tensor:
        if isinstance(self.xai, ActivationAttention):
            return self.xai.generate_from_activations(target_size=tuple(images.shape[-2:]))

        saliency_maps = []
        batch_targets = targets if isinstance(targets, Sequence) else [None] * images.shape[0]

        for index in range(images.shape[0]):
            image = images[index : index + 1]
            class_id = self.infer_class_id(batch_targets[index]) if index < len(batch_targets) else 0

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

    # Ham nay thuc hien mot training step day du theo so do XAI-guided training.
    def training_step(self, images: torch.Tensor, targets: Any, *, epoch: int | None = None) -> XAITrainerStepOutput:
        self._ensure_model_device(images.device)
        self._ensure_trainable_model()

        with torch.enable_grad():
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            predictions = self.model(images)
            detection_loss = self._reduce_loss_tensor("detection_loss", self.detection_loss_fn(predictions, targets))
            if not isinstance(self.xai, ActivationAttention):
                raise ValueError(
                    "Differentiable saliency training requires `xai_method='activation'`. "
                    "Use Grad-CAM, Grad-CAM++, or EigenCAM for offline visualization/evaluation only."
                )

            saliency_maps = self.compute_saliency_maps(images, targets)
            gt_masks = self.build_gt_masks(targets, image_hw=tuple(images.shape[-2:]), device=images.device)
            saliency_loss = saliency_alignment_loss(saliency_maps, gt_masks)

            lambda_saliency = self.get_lambda_saliency(epoch)
            total_loss = detection_loss + (lambda_saliency * saliency_loss)

            self._ensure_grad_path("detection_loss", detection_loss)
            self._ensure_grad_path("saliency_maps", saliency_maps)
            self._ensure_grad_path("saliency_loss", saliency_loss)
            self._ensure_grad_path("total_loss", total_loss)

            total_loss.backward()
            self.optimizer.step()

        return XAITrainerStepOutput(
            total_loss=total_loss.detach(),
            detection_loss=detection_loss.detach(),
            saliency_loss=saliency_loss.detach(),
            lambda_saliency=lambda_saliency,
            saliency_maps=saliency_maps.detach(),
            gt_masks=gt_masks.detach(),
            predictions=predictions,
        )

    # Ham nay giai phong hook cua XAI module sau khi ket thuc training.
    def close(self) -> None:
        self.xai.close()
