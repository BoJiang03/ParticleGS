import os
import torch
import sys
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser

from particlegs.scene import Scene
from particlegs.model.gaussian_model import GaussianModel
from particlegs.renderer import render
from particlegs.utils.general_utils import safe_state
from particlegs.model.viz_mapper import VizMapper
from particlegs.training.arguments import ModelParams, PipelineParams, get_combined_args

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, viz_mapper, min_scale_pixels=0.0, output_path=None, static_viz=False):
    base = output_path if output_path else model_path
    render_path = os.path.join(base, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(base, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # Precompute per-Gaussian intrinsic features (constant across views)
    g_log_scale = gaussians._scaling.mean(dim=1)
    g_opacity_logit = gaussians._opacity.squeeze(1)
    g_xyz = gaussians.get_xyz.detach() if getattr(viz_mapper, 'use_xyz', False) else None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        N = gaussians.get_xyz.shape[0]
        sf_val = 1.0 if static_viz else view.viz_scale_factor
        of_val = 1.0 if static_viz else view.viz_opacity_factor
        pred_scale_factor, pred_opacity_factor = viz_mapper(
            torch.full((N,), of_val, device="cuda"),
            torch.full((N,), sf_val, device="cuda"),
            gaussian_log_scale=g_log_scale,
            gaussian_opacity_logit=g_opacity_logit,
            xyz=g_xyz,
        )

        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh,
                           viz_scale_factor=pred_scale_factor, viz_opacity_factor=pred_opacity_factor, min_scale_pixels=min_scale_pixels)["render"]

        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, min_scale_pixels: float = 0.0, output_dir=None, factor_delta_opacity: float = 0.3, factor_delta_scale: float = 0.1, mapper_path: str = None, min_opacity_clamp: float = 0.4, static_viz: bool = False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Load Viz Mapper
        viz_mapper_path = mapper_path if mapper_path else os.path.join(dataset.model_path, "viz_mapper_{}.pth".format(scene.loaded_iter))
        if os.path.exists(viz_mapper_path):
            print(f"Loading VizMapper from {viz_mapper_path}")
            raw = torch.load(viz_mapper_path)
            if isinstance(raw, dict) and "model" in raw and "config" in raw:
                config = raw['config']
                viz_mapper = VizMapper(
                    hidden_dim=config.get('hidden_dim', 64),
                    num_layers=config.get('num_layers', 2),
                    use_xyz=config.get('use_xyz', False),
                    factor_delta_opacity=raw.get('factor_delta_opacity', factor_delta_opacity),
                    factor_delta_scale=raw.get('factor_delta_scale', factor_delta_scale),
                    min_opacity_clamp=raw.get('min_opacity_clamp', config.get('min_opacity_clamp', 0.4)),
                ).cuda()
                viz_mapper.load_state_dict(raw['model'])
                print(f"  Config: {config}")
            else:
                if isinstance(raw, dict) and "model" in raw:
                    raw = raw["model"]
                viz_mapper = VizMapper(factor_delta_opacity=factor_delta_opacity, factor_delta_scale=factor_delta_scale, min_opacity_clamp=min_opacity_clamp).cuda()
                sd = raw
                if any(k.startswith('base_net.') for k in sd):
                    viz_mapper.load_from_old(sd)
                elif any(k.startswith('net.') for k in sd):
                    in_dim = sd.get('net.0.weight', torch.empty(0, 0)).shape[-1]
                    if in_dim == 6:
                        viz_mapper.load_state_dict(sd)
                    else:
                        viz_mapper.load_from_old(sd)
                else:
                    viz_mapper.load_state_dict(sd)
            viz_mapper.eval()
        else:
            print(f"[{viz_mapper_path}] not found! Using initialized random weights (This is BAD if evaluating trained model).")
            viz_mapper = VizMapper(factor_delta_opacity=factor_delta_opacity, factor_delta_scale=factor_delta_scale).cuda()

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, viz_mapper, min_scale_pixels=min_scale_pixels, output_path=output_dir, static_viz=static_viz)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, viz_mapper, min_scale_pixels=min_scale_pixels, output_path=output_dir, static_viz=static_viz)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--min_scale_pixels", default=0.0, type=float)
    parser.add_argument("--factor_delta_opacity", default=0.3, type=float)
    parser.add_argument("--factor_delta_scale", default=0.1, type=float)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--mapper_path", default=None, type=str, help="Override VizMapper checkpoint path")
    parser.add_argument("--min_opacity_clamp", default=0.4, type=float, help="VizMapper min opacity clamp")
    parser.add_argument("--static_viz", action="store_true", help="Static model: ignore viz params, render with factor=1.0")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.min_scale_pixels, output_dir=getattr(args, 'output_dir', None), factor_delta_opacity=args.factor_delta_opacity, factor_delta_scale=args.factor_delta_scale, mapper_path=getattr(args, 'mapper_path', None), min_opacity_clamp=getattr(args, 'min_opacity_clamp', 0.4), static_viz=getattr(args, 'static_viz', False))
