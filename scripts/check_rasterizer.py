#!/usr/bin/env python
"""Post-build sanity check for the CUDA rasterizer.

A rasterizer compiled by a mismatched nvcc (for example the host's system CUDA
13.1 against a torch cu130 runtime) miscompiles the Blackwell (sm_120) kernel:
it reads uninitialized memory and asks for a garbage, multi-TiB allocation on
the first forward pass. Every training run then dies at iteration 0 with a
``torch.OutOfMemoryError: Tried to allocate 131071.xx GiB`` even though the GPU
is empty.

This script exercises simple_knn + one diff_gaussian_rasterization forward so
that such a broken build is caught at install time, not 15 h into a
reproduction. install.sh runs it right after building the extensions.

Exit codes:
  0  build works (or no CUDA GPU is visible, in which case the render check is
     skipped with a warning)
  1  build is broken -- do not start a reproduction
"""
import math
import sys


def main() -> int:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"  [check] torch import failed: {e}")
        return 1

    if not torch.cuda.is_available():
        print("  [check] no CUDA GPU visible here -- skipping the runtime render "
              "check. Re-run scripts/check_rasterizer.py on the GPU machine.")
        return 0

    dev = "cuda"
    try:
        from simple_knn._C import distCUDA2
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        N = 100_000
        pts = torch.rand(N, 3, device=dev) * 2 - 1
        d = distCUDA2(pts)
        if torch.isnan(d).any() or torch.isinf(d).any():
            print("  [check] FAILED -- simple_knn distCUDA2 returned nan/inf")
            return 1
        scales = torch.exp(
            torch.log(torch.sqrt(torch.clamp_min(d, 1e-7)))
        ).unsqueeze(1).repeat(1, 3) * 0.01

        means3D = torch.rand(N, 3, device=dev) * 2 - 1
        means2D = torch.zeros_like(means3D, requires_grad=True)
        opacity = torch.ones(N, 1, device=dev) * 0.1
        rot = torch.zeros(N, 4, device=dev)
        rot[:, 0] = 1
        colors = torch.rand(N, 3, device=dev)

        W = H = 512
        fov = 1.0
        n, f = 0.01, 100.0
        t = math.tan(fov / 2) * n
        b, r, l = -t, t, -t  # noqa: E741
        P = torch.zeros(4, 4)
        P[0, 0] = 2 * n / (r - l)
        P[1, 1] = 2 * n / (t - b)
        P[2, 2] = f / (f - n)
        P[2, 3] = -(f * n) / (f - n)
        P[3, 2] = 1
        viewmat = torch.eye(4, device=dev)
        viewmat[2, 3] = 3.0
        full = viewmat.T @ P.to(dev).T

        settings = GaussianRasterizationSettings(
            image_height=H, image_width=W,
            tanfovx=math.tan(fov / 2), tanfovy=math.tan(fov / 2),
            bg=torch.zeros(3, device=dev), scale_modifier=1.0,
            viewmatrix=viewmat.T, projmatrix=full, sh_degree=0,
            campos=torch.zeros(3, device=dev), prefiltered=False,
            debug=False, antialiasing=True,
        )
        out = GaussianRasterizer(raster_settings=settings)(
            means3D=means3D, means2D=means2D, shs=None, colors_precomp=colors,
            opacities=opacity, scales=scales, rotations=rot, cov3D_precomp=None,
        )
        img = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
        if tuple(img.shape) != (3, H, W):
            print(f"  [check] FAILED -- unexpected render shape {tuple(img.shape)}")
            return 1

        cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
        print(f"  [check] OK -- rasterizer renders on "
              f"{torch.cuda.get_device_name(0)} (compute {cap})")
        return 0

    except torch.OutOfMemoryError as e:
        print("  [check] FAILED -- the rasterizer requested a garbage allocation:")
        print(f"          {str(e).splitlines()[0]}")
        print("  This is the hallmark of a bad nvcc / GPU-arch build. The CUDA")
        print("  extensions must build against the pinned conda CUDA 13.0 toolchain")
        print("  (environment.yml), not the host's /usr/local/cuda. Re-create the")
        print("  env from environment.yml and re-run install.sh.")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"  [check] FAILED -- {type(e).__name__}: {str(e).splitlines()[0]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
