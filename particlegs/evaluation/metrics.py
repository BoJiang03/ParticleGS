"""Evaluation metrics for ParticleGS.

Computes full-frame and masked PSNR/SSIM between render and GT directories.
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tf

try:
    from skimage.metrics import structural_similarity as sk_ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def load_image_as_tensor(path):
    """Load image as [C, H, W] float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return tf.to_tensor(img)


def compute_psnr(img1, img2):
    """PSNR between two [C, H, W] tensors."""
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))).item()


def compute_masked_psnr(img1, img2, bg_threshold=2.5/255.0):
    """Masked PSNR: only non-background pixels (mean brightness > threshold)."""
    gt_brightness = img2.mean(dim=0)
    render_brightness = img1.mean(dim=0)
    mask = (gt_brightness > bg_threshold) | (render_brightness > bg_threshold)
    n_fg = mask.sum().item()
    if n_fg == 0:
        return 0.0

    mask3 = mask.unsqueeze(0).expand_as(img1)
    mse = ((img1 - img2) ** 2 * mask3).sum() / (n_fg * img1.shape[0])
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))).item()


def evaluate_directory(render_dir, gt_dir, compute_ssim=False):
    """Evaluate all images in render_dir vs gt_dir.

    Returns dict with full_psnr, masked_psnr, (optionally ssim), per_image results.
    """
    render_files = sorted([f for f in os.listdir(render_dir) if f.endswith('.png')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])

    assert len(render_files) == len(gt_files), \
        f"Mismatch: {len(render_files)} renders vs {len(gt_files)} GTs"

    results = []
    total_psnr = 0.0
    total_masked_psnr = 0.0
    total_ssim = 0.0

    for rf, gf in zip(render_files, gt_files):
        render_img = load_image_as_tensor(os.path.join(render_dir, rf))
        gt_img = load_image_as_tensor(os.path.join(gt_dir, gf))

        full_psnr = compute_psnr(render_img, gt_img)
        masked_psnr = compute_masked_psnr(render_img, gt_img)

        entry = {
            "name": rf,
            "full_psnr": full_psnr,
            "masked_psnr": masked_psnr,
        }

        if compute_ssim and SKIMAGE_AVAILABLE:
            r_np = render_img.permute(1, 2, 0).numpy()
            g_np = gt_img.permute(1, 2, 0).numpy()
            ssim_val = sk_ssim(r_np, g_np, channel_axis=2, data_range=1.0)
            entry["ssim"] = ssim_val
            total_ssim += ssim_val

        results.append(entry)
        total_psnr += full_psnr
        total_masked_psnr += masked_psnr

    n = len(results)
    summary = {
        "full_psnr": total_psnr / n if n > 0 else 0,
        "masked_psnr": total_masked_psnr / n if n > 0 else 0,
        "n_images": n,
        "per_image": results,
    }
    if compute_ssim and SKIMAGE_AVAILABLE:
        summary["ssim"] = total_ssim / n if n > 0 else 0

    return summary
