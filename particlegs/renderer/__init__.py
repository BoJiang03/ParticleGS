#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from particlegs.model.gaussian_model import GaussianModel
from particlegs.utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, viz_scale_factor=1.0, viz_opacity_factor=1.0, min_scale_pixels=0.0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    viz_scale_factor = torch.as_tensor(viz_scale_factor, device="cuda", dtype=pc.get_xyz.dtype)
    viz_opacity_factor = torch.as_tensor(viz_opacity_factor, device="cuda", dtype=pc.get_xyz.dtype)
    viz_scale_factor = torch.nan_to_num(viz_scale_factor, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.01, 10.0)
    viz_opacity_factor = torch.nan_to_num(viz_opacity_factor, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.05, 10.0)

    # Ensure per-Gaussian factors have correct shape for broadcasting
    if viz_scale_factor.dim() == 1:
        viz_scale_factor = viz_scale_factor.unsqueeze(-1)  # [N] -> [N, 1]
    if viz_opacity_factor.dim() == 1:
        viz_opacity_factor = viz_opacity_factor.unsqueeze(-1)  # [N] -> [N, 1]

    means3D = pc.get_xyz.contiguous()
    means2D = screenspace_points
    log_viz_opacity = torch.log(viz_opacity_factor.clamp(min=1e-6))
    opacity_logit = pc._opacity + log_viz_opacity  # add in logit space
    opacity = torch.sigmoid(opacity_logit)
    opacity = torch.nan_to_num(opacity, nan=0.0, posinf=1.0, neginf=0.0).contiguous()

    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = (pc.get_scaling * viz_scale_factor)
        scales = torch.nan_to_num(scales, nan=1e-3, posinf=1.0, neginf=1e-3)
        if min_scale_pixels > 0:
            xyz = pc.get_xyz
            viewmat = viewpoint_camera.world_view_transform
            depths = (xyz * viewmat[:3, 2].unsqueeze(0)).sum(dim=-1) + viewmat[3, 2]
            depths = depths.clamp(min=0.1)
            focal = viewpoint_camera.image_width / (2.0 * tanfovx)
            min_scale_world = (min_scale_pixels * depths / focal).unsqueeze(-1)
            scales = torch.sqrt(scales ** 2 + min_scale_world ** 2)
        scales = torch.clamp(scales, min=1e-6)
        scales = scales.contiguous()
        rotations = pc.get_rotation.contiguous()

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc.contiguous(), pc.get_features_rest.contiguous()
            else:
                shs = pc.get_features.contiguous()
    else:
        colors_precomp = override_color

    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]

    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }

    return out
