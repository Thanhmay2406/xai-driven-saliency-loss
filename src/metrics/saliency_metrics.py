from __future__ import annotations

import torch

from src.losses.saliency_alignment_loss import _ensure_nchw, _normalize_saliency_map, _resize_mask


def _prepare_saliency_and_masks(
    saliency_maps: torch.Tensor,
    gt_masks: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    saliency_maps = _ensure_nchw(saliency_maps, name="saliency_maps")
    gt_masks = _ensure_nchw(gt_masks, name="gt_masks")

    if saliency_maps.shape[0] != gt_masks.shape[0]:
        raise ValueError(
            "saliency_maps and gt_masks must have the same batch size. "
            f"Got {saliency_maps.shape[0]} and {gt_masks.shape[0]}."
        )

    saliency_maps = _normalize_saliency_map(saliency_maps, eps=eps)
    gt_masks = _resize_mask(gt_masks, target_hw=saliency_maps.shape[-2:])
    gt_masks = (gt_masks > 0).to(dtype=saliency_maps.dtype)
    return saliency_maps, gt_masks


def energy_in_box(saliency_maps: torch.Tensor, gt_masks: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    saliency_maps, gt_masks = _prepare_saliency_and_masks(saliency_maps, gt_masks, eps=eps)
    inside_energy = (saliency_maps * gt_masks).sum(dim=(1, 2, 3))
    total_energy = saliency_maps.sum(dim=(1, 2, 3)) + eps
    return inside_energy / total_energy


def pointing_game_accuracy(saliency_maps: torch.Tensor, gt_masks: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    saliency_maps, gt_masks = _prepare_saliency_and_masks(saliency_maps, gt_masks, eps=eps)
    flat_saliency = saliency_maps.flatten(start_dim=1)
    peak_indices = flat_saliency.argmax(dim=1)
    flat_masks = gt_masks.flatten(start_dim=1)
    hits = flat_masks.gather(1, peak_indices.unsqueeze(1)).squeeze(1)
    return hits.float()


def saliency_iou(
    saliency_maps: torch.Tensor,
    gt_masks: torch.Tensor,
    *,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    saliency_maps, gt_masks = _prepare_saliency_and_masks(saliency_maps, gt_masks, eps=eps)
    saliency_mask = (saliency_maps >= threshold).to(dtype=gt_masks.dtype)
    intersection = (saliency_mask * gt_masks).sum(dim=(1, 2, 3))
    union = ((saliency_mask + gt_masks) > 0).to(dtype=gt_masks.dtype).sum(dim=(1, 2, 3))
    return intersection / (union + eps)
