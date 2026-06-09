from dataclasses import dataclass

import torch

from .saliency_utils import normalize_cam, upsample_cam


@dataclass
class HookState:
    activations: torch.Tensor | None = None


class ActivationAttention:
    # Lop nay tao attention map kha vi sai tu activation cua target layer.
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.state = HookState()
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)

    # Ham nay luu activation map cua lan forward gan nhat.
    def _save_activations(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        del module, inputs
        self.state.activations = output

    # Ham nay tong hop activation theo chieu channel de tao attention map.
    def generate_from_activations(self, target_size: tuple[int, int] | None = None) -> torch.Tensor:
        if self.state.activations is None:
            raise RuntimeError("Hooks did not capture activations for ActivationAttention.")

        activations = self.state.activations
        attention = activations.abs().mean(dim=1, keepdim=True)
        if target_size is not None and attention.shape[-2:] != target_size:
            attention = upsample_cam(attention, target_size=target_size)
        return attention

    # Ham nay phuc vu truc quan hoa ngoai training.
    def generate_normalized(self, target_size: tuple[int, int] | None = None) -> torch.Tensor:
        attention = self.generate_from_activations(target_size=target_size)
        return torch.from_numpy(normalize_cam(attention))

    # Ham nay giai phong hook da dang ky.
    def close(self) -> None:
        self.forward_handle.remove()
