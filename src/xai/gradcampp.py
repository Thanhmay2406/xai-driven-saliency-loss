from dataclasses import dataclass

import torch

from .saliency_utils import build_detection_target, normalize_cam, set_batchnorm_eval, upsample_cam


@dataclass
class HookState:
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None


class GradCAMPlusPlus:
    # Ham nay khoi tao doi tuong Grad-CAM++ va dang ky hook vao target layer.
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.state = HookState()
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    # Ham nay luu activation map tu target layer trong lan forward.
    def _save_activations(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        del module, inputs
        self.state.activations = output

    # Ham nay luu gradient cua target layer trong lan backward.
    def _save_gradients(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        del module, grad_input
        self.state.gradients = grad_output[0]

    # Ham nay tinh saliency map Grad-CAM++ cho class duoc chon.
    def generate(self, image_tensor: torch.Tensor, class_id: int, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.model.training
        self.model.zero_grad(set_to_none=True)
        image_tensor = image_tensor.requires_grad_(True)
        with torch.enable_grad():
            self.model.train()
            set_batchnorm_eval(self.model)
            output = self.model(image_tensor)
        target_score = build_detection_target(output, class_id=class_id, num_classes=num_classes)
        target_score.backward(retain_graph=False)
        self.model.train(was_training)

        if self.state.activations is None or self.state.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients for Grad-CAM++.")

        gradients = self.state.gradients
        activations = self.state.activations
        grads_square = gradients.pow(2)
        grads_cube = gradients.pow(3)

        denominator = 2.0 * grads_square + (activations * grads_cube).sum(dim=(2, 3), keepdim=True)
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = grads_square / (denominator + 1e-8)
        positive_gradients = torch.relu(gradients)
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)

        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = upsample_cam(cam, target_size=image_tensor.shape[-2:])
        return cam, target_score.detach()

    # Ham nay tra ve saliency map da normalize ve dang numpy.
    def generate_normalized(self, image_tensor: torch.Tensor, class_id: int, num_classes: int) -> tuple[torch.Tensor, float]:
        cam, target_score = self.generate(image_tensor=image_tensor, class_id=class_id, num_classes=num_classes)
        return torch.from_numpy(normalize_cam(cam)), float(target_score.item())

    # Ham nay giai phong cac hook da dang ky.
    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()
