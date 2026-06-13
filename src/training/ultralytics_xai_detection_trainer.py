from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model

from src.metrics import energy_in_box, pointing_game_accuracy, saliency_iou
from src.training.yolo_xai_trainer import UltralyticsYOLOXAITrainerConfig
from src.xai.activation_attention import ActivationAttention
from src.xai.saliency_utils import find_yolo_saliency_target_layer
from src.losses import saliency_alignment_loss


EXPECTED_METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)

SUMMARY_METRIC_KEYS = EXPECTED_METRIC_KEYS + ("fitness",)


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _ensure_metric_keys(metrics: dict[str, Any], *, required_keys: tuple[str, ...], context: str) -> None:
    missing_keys = [key for key in required_keys if key not in metrics]
    if missing_keys:
        raise RuntimeError(f"Missing {context} metric keys: {missing_keys}. Found keys: {sorted(metrics.keys())}.")


def _normalize_dataset_names(names: dict | list) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {idx: str(value) for idx, value in enumerate(names)}


def _save_json(payload: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)


def _dataset_split_exists(data: dict[str, Any], split: str) -> bool:
    split_path = data.get(split)
    if not split_path:
        return False
    return Path(split_path).exists()


def _init_base_detection_criterion(model: DetectionModel) -> Any:
    return DetectionModel.init_criterion(model)


def extract_eval_metrics(results: Any, expected_names: dict[int, str] | None = None) -> dict[str, object]:
    results_dict = getattr(results, "results_dict", None)
    if not results_dict:
        raise RuntimeError("Validation did not return `results_dict`.")

    missing_keys = [key for key in SUMMARY_METRIC_KEYS if key not in results_dict]
    if missing_keys:
        raise RuntimeError(f"Missing validation metric keys: {missing_keys}. Found keys: {sorted(results_dict.keys())}")

    names = _normalize_dataset_names(getattr(results, "names", {}) or {})
    if expected_names and names and names != expected_names:
        raise RuntimeError(f"Validation class names mismatch. Expected {expected_names}, got {names}.")

    maps = getattr(results, "maps", None)
    per_class_map50_95 = {}
    if maps is not None:
        if expected_names and len(maps) != len(expected_names):
            raise RuntimeError(f"`results.maps` length {len(maps)} does not match expected nc={len(expected_names)}.")
        for idx, value in enumerate(maps):
            class_name = (expected_names or names).get(idx, str(idx))
            per_class_map50_95[class_name] = float(value)

    return {
        "precision": float(results_dict["metrics/precision(B)"]),
        "recall": float(results_dict["metrics/recall(B)"]),
        "map50": float(results_dict["metrics/mAP50(B)"]),
        "map50_95": float(results_dict["metrics/mAP50-95(B)"]),
        "fitness": float(results_dict["fitness"]),
        "per_class_map50_95": per_class_map50_95,
        "save_dir": str(getattr(results, "save_dir", "")),
        "speed": {key: float(value) for key, value in getattr(results, "speed", {}).items()},
    }


class UltralyticsYOLOXAILoss:
    def __init__(self, model: torch.nn.Module, config: UltralyticsYOLOXAITrainerConfig) -> None:
        if config.xai_method.strip().lower() not in {"activation", "attention", "activation_attention"}:
            raise ValueError(
                "Differentiable saliency training requires `xai_method='activation'`. "
                "Use Grad-CAM variants for offline visualization only."
            )

        self.model = model
        self.config = config
        self.target_layer = config.target_layer or find_yolo_saliency_target_layer(
            model,
            prefer_branch=config.prefer_branch,
        )
        self.detection_criterion = _init_base_detection_criterion(model)
        self.xai = ActivationAttention(model, self.target_layer)
        self.current_epoch = 0
        self.latest_batch_metrics: dict[str, float] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["model"] = None
        state["detection_criterion"] = None
        state["xai"] = None
        state["latest_batch_metrics"] = {}
        return state

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(int(epoch), 0)

    def close(self) -> None:
        if self.xai is not None:
            self.xai.close()
            self.xai = None

    def get_lambda_saliency(self) -> float:
        base_lambda = float(self.config.lambda_saliency)
        if not self.config.use_progressive_lambda:
            return base_lambda
        warmup_epochs = max(1, int(self.config.progressive_warmup_epochs))
        scale = min(self.current_epoch / warmup_epochs, 1.0)
        return base_lambda * scale

    def build_gt_masks(self, batch: dict[str, Any]) -> torch.Tensor:
        images = batch["img"]
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("Expected `batch['img']` with shape (N, C, H, W).")

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
        if batch_idx.numel() != bboxes.shape[0]:
            raise ValueError("`batch_idx` and `bboxes` must have the same number of rows.")

        if box_format == "xywhn":
            x_center = bboxes[:, 0] * width
            y_center = bboxes[:, 1] * height
            box_w = bboxes[:, 2] * width
            box_h = bboxes[:, 3] * height
            boxes_xyxy = torch.stack(
                [
                    x_center - box_w / 2.0,
                    y_center - box_h / 2.0,
                    x_center + box_w / 2.0,
                    y_center + box_h / 2.0,
                ],
                dim=-1,
            )
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
                masks[sample_index, :, y1:y2, x1:x2] = torch.maximum(
                    masks[sample_index, :, y1:y2, x1:x2],
                    self._build_soft_box_mask(
                        height=y2 - y1,
                        width=x2 - x1,
                        device=device,
                        sigma=max(float(self.config.soft_mask_sigma), 1e-3),
                    ),
                )
            else:
                masks[sample_index, :, y1:y2, x1:x2] = 1.0

        return masks

    @staticmethod
    def _shrink_box(box: torch.Tensor, ratio: float) -> torch.Tensor:
        ratio = min(max(float(ratio), 1e-3), 1.0)
        x1, y1, x2, y2 = box.unbind()
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        half_w = (x2 - x1) * ratio / 2.0
        half_h = (y2 - y1) * ratio / 2.0
        return torch.stack([center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h])

    @staticmethod
    def _build_soft_box_mask(*, height: int, width: int, device: torch.device, sigma: float) -> torch.Tensor:
        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float32)
        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        gaussian = torch.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
        gaussian = gaussian / gaussian.max().clamp_min(1e-8)
        return gaussian.unsqueeze(0)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        detection_total, detection_items = self.detection_criterion(preds, batch)
        if not torch.is_grad_enabled() or self.xai is None:
            self.latest_batch_metrics = {}
            zero_saliency_loss = detection_total.new_zeros(())
            loss_items = torch.cat([detection_items, zero_saliency_loss.reshape(1)])
            return detection_total, loss_items

        saliency_maps = self.xai.generate_from_activations(target_size=tuple(batch["img"].shape[-2:]))
        gt_masks = self.build_gt_masks(batch)
        saliency_loss = saliency_alignment_loss(saliency_maps, gt_masks)

        batch_size = int(batch["img"].shape[0])
        lambda_saliency = self.get_lambda_saliency()
        total_loss = detection_total + (lambda_saliency * saliency_loss * batch_size)

        with torch.no_grad():
            self.latest_batch_metrics = {
                "lambda_saliency": float(lambda_saliency),
                "saliency_loss": float(saliency_loss.detach().item()),
                "energy_in_box": float(energy_in_box(saliency_maps.detach(), gt_masks.detach()).mean().item()),
                "pointing_game": float(pointing_game_accuracy(saliency_maps.detach(), gt_masks.detach()).mean().item()),
                "saliency_iou": float(saliency_iou(saliency_maps.detach(), gt_masks.detach()).mean().item()),
            }

        self.xai.state.activations = None
        loss_items = torch.cat([detection_items, saliency_loss.detach().reshape(1)])
        return total_loss, loss_items


class XAIDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        xai_config: UltralyticsYOLOXAITrainerConfig | None = None,
    ) -> None:
        self.xai_config = deepcopy(xai_config or UltralyticsYOLOXAITrainerConfig())
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self) -> UltralyticsYOLOXAILoss:
        return UltralyticsYOLOXAILoss(self, deepcopy(self.xai_config))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["criterion"] = None
        xai_config = state.get("xai_config")
        if xai_config is not None:
            xai_config = deepcopy(xai_config)
            xai_config.target_layer = None
            state["xai_config"] = xai_config
        return state


class UltralyticsYOLOXAIDetectionTrainer(DetectionTrainer):
    def __init__(
        self,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        *,
        xai_config: UltralyticsYOLOXAITrainerConfig | None = None,
    ) -> None:
        self.xai_config = deepcopy(xai_config or UltralyticsYOLOXAITrainerConfig())
        self.xai_history: list[dict[str, Any]] = []
        self._xai_epoch_batches: list[dict[str, float]] = []
        self.metrics_dir: Path | None = None
        self.summary_dir: Path | None = None
        self.history_json_path: Path | None = None
        self.history_csv_path: Path | None = None
        self.best_val_metrics_path: Path | None = None
        self.best_test_metrics_path: Path | None = None
        self.run_summary_path: Path | None = None
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.add_callback("on_pretrain_routine_end", self._on_pretrain_routine_end)
        self.add_callback("on_train_epoch_start", self._on_train_epoch_start)
        self.add_callback("on_train_batch_end", self._on_train_batch_end)
        self.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
        self.add_callback("on_train_end", self._on_train_end)

    def get_model(self, cfg: str | None = None, weights: Any = None, verbose: bool = True) -> DetectionModel:
        model = XAIDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            xai_config=self.xai_config,
        )
        if weights is not None:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "sal_loss")
        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
        )

    def _close_xai_hooks(self) -> None:
        for candidate in (unwrap_model(self.model), getattr(self.ema, "ema", None)):
            if candidate is None:
                continue
            criterion = getattr(candidate, "criterion", None)
            if hasattr(criterion, "close"):
                criterion.close()

    def _ensure_xai_ready(self, candidate: torch.nn.Module | None) -> None:
        if candidate is None:
            return
        criterion = getattr(candidate, "criterion", None)
        if criterion is None or getattr(criterion, "xai", None) is not None:
            return

        target_layer = getattr(criterion, "target_layer", None)
        if target_layer is None:
            xai_config = getattr(candidate, "xai_config", None)
            if xai_config is not None and getattr(xai_config, "target_layer", None) is not None:
                target_layer = xai_config.target_layer
            else:
                prefer_branch = getattr(xai_config, "prefer_branch", "small") if xai_config is not None else "small"
                target_layer = find_yolo_saliency_target_layer(candidate, prefer_branch=prefer_branch)

        criterion.target_layer = target_layer
        criterion.xai = ActivationAttention(candidate, target_layer)
        criterion.latest_batch_metrics = {}
        xai_config = getattr(candidate, "xai_config", None)
        if xai_config is not None:
            xai_config.target_layer = target_layer

    @contextmanager
    def _checkpoint_safe_models(self):
        snapshots: list[tuple[torch.nn.Module, Any, Any, list[tuple[torch.nn.Module, Any, Any, Any]]]] = []
        for candidate in (unwrap_model(self.model), getattr(self.ema, "ema", None)):
            if candidate is None:
                continue
            criterion = getattr(candidate, "criterion", None)
            xai_config = getattr(candidate, "xai_config", None)
            target_layer = getattr(criterion, "target_layer", None)
            if target_layer is None and xai_config is not None:
                target_layer = getattr(xai_config, "target_layer", None)
            if hasattr(criterion, "close"):
                criterion.close()

            module_hooks: list[tuple[torch.nn.Module, Any, Any, Any]] = []
            for module in candidate.modules():
                forward_hooks = getattr(module, "_forward_hooks", None)
                forward_pre_hooks = getattr(module, "_forward_pre_hooks", None)
                backward_hooks = getattr(module, "_backward_hooks", None)
                module_hooks.append(
                    (
                        module,
                        deepcopy(forward_hooks) if forward_hooks is not None else None,
                        deepcopy(forward_pre_hooks) if forward_pre_hooks is not None else None,
                        deepcopy(backward_hooks) if backward_hooks is not None else None,
                    )
                )
                if forward_hooks is not None:
                    forward_hooks.clear()
                if forward_pre_hooks is not None:
                    forward_pre_hooks.clear()
                if backward_hooks is not None:
                    backward_hooks.clear()

            candidate.criterion = None
            if xai_config is not None:
                xai_config.target_layer = None
            snapshots.append((candidate, criterion, target_layer, module_hooks))
        try:
            yield
        finally:
            for candidate, criterion, target_layer, module_hooks in snapshots:
                candidate.criterion = criterion
                xai_config = getattr(candidate, "xai_config", None)
                if xai_config is not None:
                    xai_config.target_layer = target_layer
                for module, forward_hooks, forward_pre_hooks, backward_hooks in module_hooks:
                    if forward_hooks is not None:
                        module._forward_hooks = forward_hooks
                    if forward_pre_hooks is not None:
                        module._forward_pre_hooks = forward_pre_hooks
                    if backward_hooks is not None:
                        module._backward_hooks = backward_hooks
                self._ensure_xai_ready(candidate)

    def save_model(self):
        with self._checkpoint_safe_models():
            return super().save_model()

    def _on_pretrain_routine_end(self, trainer: "UltralyticsYOLOXAIDetectionTrainer") -> None:
        del trainer
        self.metrics_dir = self.save_dir / "metrics"
        self.summary_dir = self.save_dir / "summary"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.history_json_path = self.save_dir / "weights" / "train_history.json"
        self.history_csv_path = self.save_dir / "weights" / "train_history.csv"
        self.best_val_metrics_path = self.metrics_dir / "best_val_metrics.json"
        self.best_test_metrics_path = self.metrics_dir / "best_test_metrics.json"
        self.run_summary_path = self.summary_dir / "run_summary.json"
        _save_json(
            {
                "train_args": vars(self.args),
                "dataset": self.data,
                "xai_config": asdict(self.xai_config),
            },
            self.summary_dir / "run_config.json",
        )

    def _on_train_epoch_start(self, trainer: "UltralyticsYOLOXAIDetectionTrainer") -> None:
        del trainer
        self._xai_epoch_batches = []
        model = unwrap_model(self.model)
        self._ensure_xai_ready(model)
        criterion = getattr(model, "criterion", None)
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(self.epoch)
        if hasattr(criterion, "latest_batch_metrics"):
            criterion.latest_batch_metrics = {}

    def _on_train_batch_end(self, trainer: "UltralyticsYOLOXAIDetectionTrainer") -> None:
        del trainer
        criterion = getattr(unwrap_model(self.model), "criterion", None)
        batch_metrics = getattr(criterion, "latest_batch_metrics", None)
        if batch_metrics:
            self._xai_epoch_batches.append(dict(batch_metrics))

    def _on_fit_epoch_end(self, trainer: "UltralyticsYOLOXAIDetectionTrainer") -> None:
        del trainer
        if not self._xai_epoch_batches or self.history_json_path is None or self.history_csv_path is None:
            return
        if (self.epoch + 1) > self.epochs:
            return

        criterion = getattr(unwrap_model(self.model), "criterion", None)
        lambda_saliency = float(getattr(criterion, "get_lambda_saliency", lambda: self.xai_config.lambda_saliency)())
        xai_means = {
            key: sum(batch[key] for batch in self._xai_epoch_batches) / len(self._xai_epoch_batches)
            for key in self._xai_epoch_batches[0]
        }
        train_losses = self.label_loss_items(self.tloss, prefix="train")
        val_metrics = dict(self.metrics)
        lr_metrics = dict(self.lr)
        if self.args.val:
            _ensure_metric_keys(val_metrics, required_keys=EXPECTED_METRIC_KEYS, context="validation")
        det_loss = sum(float(train_losses.get(f"train/{name}", 0.0)) for name in ("box_loss", "cls_loss", "dfl_loss"))
        sal_loss = float(xai_means.get("saliency_loss", 0.0))

        epoch_record = {
            "epoch": self.epoch + 1,
            "lr": float(next(iter(lr_metrics.values()))) if lr_metrics else 0.0,
            "lambda_saliency": lambda_saliency,
            "train_detection_loss": det_loss,
            "train_saliency_loss": sal_loss,
            "train_total_loss": det_loss + (lambda_saliency * sal_loss),
            "train_energy_in_box": xai_means["energy_in_box"],
            "train_pointing_game": xai_means["pointing_game"],
            "train_saliency_iou": xai_means["saliency_iou"],
            "val_precision": float(val_metrics.get("metrics/precision(B)", 0.0)),
            "val_recall": float(val_metrics.get("metrics/recall(B)", 0.0)),
            "val_map50": float(val_metrics.get("metrics/mAP50(B)", 0.0)),
            "val_map50_95": float(val_metrics.get("metrics/mAP50-95(B)", 0.0)),
            "val_fitness": float(val_metrics.get("fitness", self.fitness)),
            "epoch_seconds": float(self.epoch_time or 0.0),
        }
        self.xai_history.append(epoch_record)
        _save_json(self.xai_history, self.history_json_path)
        with self.history_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.xai_history[0].keys()))
            writer.writeheader()
            writer.writerows(self.xai_history)

        if self.metrics_dir is not None:
            epoch_payload = {
                "epoch": self.epoch + 1,
                "train": epoch_record,
                "val": val_metrics,
                "lr": lr_metrics,
            }
            _save_json(epoch_payload, self.metrics_dir / f"epoch_{self.epoch + 1:03d}_metrics.json")

    def _on_train_end(self, trainer: "UltralyticsYOLOXAIDetectionTrainer") -> None:
        del trainer
        self._close_xai_hooks()
        if self.run_summary_path is None or self.best_val_metrics_path is None or self.best_test_metrics_path is None:
            return

        expected_names = _normalize_dataset_names(self.data["names"])
        val_args = {
            "data": self.args.data,
            "imgsz": self.args.imgsz,
            "batch": self.batch_size,
            "device": self.device.index if self.device.type == "cuda" else self.device.type,
        }
        best_model = YOLO(str(self.best))
        best_val_metrics = extract_eval_metrics(best_model.val(**val_args, split="val"), expected_names)
        best_test_metrics = (
            extract_eval_metrics(best_model.val(**val_args, split="test"), expected_names)
            if _dataset_split_exists(self.data, "test")
            else {}
        )
        _save_json(best_val_metrics, self.best_val_metrics_path)
        _save_json(best_test_metrics, self.best_test_metrics_path)

        best_epoch_by_fitness = max(self.xai_history, key=lambda row: row["val_fitness"])["epoch"] if self.xai_history else 0
        best_epoch_by_map = max(self.xai_history, key=lambda row: row["val_map50_95"])["epoch"] if self.xai_history else 0
        _save_json(
            {
                "best_epoch": best_epoch_by_fitness,
                "best_epoch_by_fitness": best_epoch_by_fitness,
                "best_epoch_by_map50_95": best_epoch_by_map,
                "best_val_map50_95": float(best_val_metrics["map50_95"]),
                "num_epochs_completed": len(self.xai_history),
                "best_checkpoint": str(self.best),
                "last_checkpoint": str(self.last),
                "history_csv": str(self.history_csv_path),
                "history_json": str(self.history_json_path),
                "best_val_metrics": best_val_metrics,
                "best_test_metrics": best_test_metrics,
            },
            self.run_summary_path,
        )


def train_ultralytics_yolo_xai(
    *,
    train_overrides: dict[str, Any],
    xai_config: UltralyticsYOLOXAITrainerConfig | None = None,
) -> UltralyticsYOLOXAIDetectionTrainer:
    trainer = UltralyticsYOLOXAIDetectionTrainer(overrides=train_overrides, xai_config=xai_config)
    trainer.train()
    return trainer
