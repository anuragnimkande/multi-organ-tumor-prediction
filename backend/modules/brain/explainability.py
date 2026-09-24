import io
import base64
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .model.hybrid import HybridModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_gradcam(model: HybridModel, img_tensor: torch.Tensor) -> str:
    """
    Generate Grad-CAM heatmap and return as base64-encoded PNG.
    """
    activations, gradients = {}, {}

    target_layer = list(model.cnn.backbone.children())[-3][-1]

    def forward_hook(module, inp, out):
        activations["val"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["val"] = grad_out[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        img_input = img_tensor.to(DEVICE).requires_grad_(True)
        model.train()
        output = model(img_input)
        model.zero_grad()
        output.backward()

        act  = activations["val"]
        grad = gradients["val"]
        weights = grad.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().detach().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (224, 224))

        orig = img_tensor.squeeze().cpu().detach().numpy()
        orig_rgb = np.stack([orig] * 3, axis=-1)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        blended = np.clip(0.5 * orig_rgb + 0.5 * heatmap, 0, 1)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(blended)
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        print(f"[Grad-CAM] Failed: {e}")
        return None

    finally:
        fh.remove()
        bh.remove()
        model.eval()
