from dataclasses import dataclass

import numpy as np
import torch

from .saliency_utils import normalize_cam, upsample_cam


@dataclass
class HookState:
    activations: torch.Tensor | None = None


class EigenCAM:
    # Ham nay khoi tao doi tuong EigenCAM va dang ky hook vao target layer.
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.state = HookState()
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)

    # Ham nay luu activation map tu target layer trong lan forward.
    def _save_activations(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        del module, inputs
        self.state.activations = output

    # Ham nay tinh saliency map EigenCAM tu principal component cua activation.
    def generate(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _ = self.model(image_tensor)

        if self.state.activations is None:
            raise RuntimeError("Hooks did not capture activations for EigenCAM.")

        activations = self.state.activations.detach().cpu().squeeze(0).numpy()
        channels, height, width = activations.shape
        features = activations.reshape(channels, height * width).T
        features = features - features.mean(axis=0, keepdims=True)

        _, _, vh = np.linalg.svd(features, full_matrices=False)
        principal = features @ vh[0]
        cam = np.abs(principal).reshape(height, width)
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).to(image_tensor.device, dtype=image_tensor.dtype)
        return upsample_cam(cam_tensor, target_size=image_tensor.shape[-2:])

    # Ham nay tra ve saliency map da normalize ve dang numpy.
    def generate_normalized(self, image_tensor: torch.Tensor) -> torch.Tensor:
        cam = self.generate(image_tensor=image_tensor)
        return torch.from_numpy(normalize_cam(cam))

    # Ham nay giai phong cac hook da dang ky.
    def close(self) -> None:
        self.forward_handle.remove()
