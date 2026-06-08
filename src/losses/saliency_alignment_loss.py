from __future__ import annotations

import torch
import torch.nn.functional as F


def _ensure_nchw(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"{name} must have shape (H, W), (N, H, W), or (N, C, H, W). Got {tuple(tensor.shape)}.")


def _normalize_saliency_map(saliency_maps: torch.Tensor, eps: float) -> torch.Tensor:
    saliency_maps = torch.clamp(saliency_maps.float(), min=0.0)
    flat = saliency_maps.flatten(start_dim=1)
    min_vals = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    max_vals = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    return (saliency_maps - min_vals) / (max_vals - min_vals + eps)


def _resize_mask(gt_masks: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if gt_masks.shape[-2:] == target_hw:
        return gt_masks.float()
    return F.interpolate(gt_masks.float(), size=target_hw, mode="nearest")


def saliency_alignment_loss(
    saliency_maps: torch.Tensor,
    gt_masks: torch.Tensor,
    *,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
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

    inside_energy = (saliency_maps * gt_masks).sum(dim=(1, 2, 3))
    total_energy = saliency_maps.sum(dim=(1, 2, 3)) + eps
    losses = 1.0 - (inside_energy / total_energy)

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")
