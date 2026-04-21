#!/usr/bin/env python3
"""Generate training/eval images with progressive camera strategies.

Supports multi-orbit cameras, per-frame viz parameter sampling (radius/opacity),
and ctypes bypass for fast ParaView rendering of 280M+ particle datasets.

Must be run with pvbatch for ParaView rendering.

Usage:
    pvbatch --force-offscreen-rendering -- generate_images.py \
        --vtp_path /path/to/particles.vtp \
        --output_dir /path/to/output \
        --camera_strategy multi_orbit \
        --orbit_radii 1.0,0.7,0.5 \
        --viz_mode sampled \
        --radius_min 0.0025 --radius_max 0.0175 \
        --opacity_min 0.0125 --opacity_max 0.0875
"""

import argparse
import ctypes
import json
import math
import os
import struct

import numpy as np

try:
    from paraview.simple import *
    PARAVIEW_AVAILABLE = True
except ImportError:
    PARAVIEW_AVAILABLE = False


def _find_double_offset(ptr, value, buf_size=4096):
    """Find offset of a double value in C++ object memory."""
    buf = (ctypes.c_char * buf_size)()
    ctypes.memmove(buf, ptr, buf_size)
    for off in range(0, buf_size - 8):
        v = struct.unpack_from('d', buf, off)[0]
        if abs(v - value) < 1e-15:
            return off
    return None


def _setup_bypass(render_view, display, initial_radius, initial_opacity):
    """Discover memory offsets for ScaleFactor and Opacity, return bypass functions.

    Uses ctypes to directly modify VTK C++ object memory, bypassing Modified()
    calls that trigger expensive pipeline rebuilds for 280M+ point clouds.
    """
    try:
        rw = render_view.GetRenderWindow()
        ren = rw.GetRenderers().GetFirstRenderer()
        actors = ren.GetActors()
        actors.InitTraversal()
        pg_mapper = None
        pg_actor = None
        for _ in range(30):
            a = actors.GetNextActor()
            if a is None:
                break
            m = a.GetMapper()
            if m and "PointGaussian" in type(m).__name__:
                pg_mapper = m
                pg_actor = a
                break

        if pg_mapper is None:
            return None, None

        mapper_ptr = int(pg_mapper.__this__.split('_')[1], 16)
        prop = pg_actor.GetProperty()
        prop_ptr = int(prop.__this__.split('_')[1], 16)

        scale_offset = _find_double_offset(mapper_ptr, initial_radius)
        if scale_offset is None:
            return None, None

        probe_radius = initial_radius * 2.5 + 0.00777
        display.GaussianRadius = probe_radius
        Render()
        verify_offset = _find_double_offset(mapper_ptr, probe_radius)
        if verify_offset != scale_offset:
            display.GaussianRadius = initial_radius
            Render()
            return None, None

        opacity_offset = _find_double_offset(prop_ptr, initial_opacity)
        if opacity_offset is None:
            display.GaussianRadius = initial_radius
            Render()
            return None, None

        probe_opacity = min(initial_opacity * 2.5 + 0.00333, 0.99)
        display.Opacity = probe_opacity
        Render()
        verify_offset = _find_double_offset(prop_ptr, probe_opacity)
        if verify_offset != opacity_offset:
            display.GaussianRadius = initial_radius
            display.Opacity = initial_opacity
            Render()
            return None, None

        struct.pack_into('d', (ctypes.c_char * 8).from_address(mapper_ptr + scale_offset), 0, initial_radius)
        struct.pack_into('d', (ctypes.c_char * 8).from_address(prop_ptr + opacity_offset), 0, initial_opacity)
        Render()

        def set_scale(value):
            struct.pack_into('d', (ctypes.c_char * 8).from_address(mapper_ptr + scale_offset), 0, value)

        def set_opacity(value):
            struct.pack_into('d', (ctypes.c_char * 8).from_address(prop_ptr + opacity_offset), 0, value)

        return set_scale, set_opacity
    except Exception:
        return None, None


# ── Vector math ─────────────────────────────────────────────────────

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def vec_sub(a, b): return [x - y for x, y in zip(a, b)]
def vec_norm(a): return math.sqrt(sum(x * x for x in a))
def vec_normalize(a):
    m = vec_norm(a)
    return [x / m for x in a] if m > 0 else [0.0, 0.0, 1.0]
def vec_cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
def vec_dot(a, b): return sum(x * y for x, y in zip(a, b))


def choose_view_up(dir_vec):
    up = [0.0, 1.0, 0.0]
    if abs(vec_dot(vec_normalize(dir_vec), up)) > 0.95:
        return [0.0, 0.0, 1.0]
    return up


def get_camera_matrix(pos, focal, v_up):
    vec_z = vec_normalize(vec_sub(pos, focal))
    vec_x = vec_cross(v_up, vec_z)
    m_x = vec_norm(vec_x)
    vec_x = vec_normalize(vec_x) if m_x > 0 else [1.0, 0.0, 0.0]
    vec_y = vec_cross(vec_z, vec_x)
    return [
        [vec_x[0], vec_y[0], vec_z[0], pos[0]],
        [vec_x[1], vec_y[1], vec_z[1], pos[1]],
        [vec_x[2], vec_y[2], vec_z[2], pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def load_normalization(path):
    with open(path, "r") as f:
        return json.load(f)


def save_normalization(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def compute_internal_bounds(bounds, center, scale):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    half_x = (xmax - xmin) * 0.5 * scale
    half_y = (ymax - ymin) * 0.5 * scale
    half_z = (zmax - zmin) * 0.5 * scale
    return [
        center[0] - half_x, center[0] + half_x,
        center[1] - half_y, center[1] + half_y,
        center[2] - half_z, center[2] + half_z,
    ]


# ── Main generation ─────────────────────────────────────────────────

def generate_images_and_cameras(
    vtp_path, output_dir, gaussian_radius, opacity,
    width=1920, height=1080, num_frames=200,
    train_ratio=0.8, split_seed=42,
    camera_strategy="external_spiral",
    internal_bounds_scale=0.9,
    normalization_path=None,
    camera_seed=42,
    viz_mode="fixed",
    radius_range=None, opacity_range=None,
    viz_seed=42, viz_distribution="uniform",
    viz_beta_concentration=3.0,
    orbit_radii=None,
    no_norm_radius=False,
):
    if not PARAVIEW_AVAILABLE:
        raise ImportError("paraview.simple is required. Run with pvbatch.")

    images_dir = os.path.join(output_dir, "images")
    ensure_dir(images_dir)

    data_source = OpenDataFile(vtp_path)
    if not data_source:
        raise RuntimeError(f"Could not open VTP file: {vtp_path}")

    render_view = CreateRenderView()
    render_view.ViewSize = [width, height]
    layout = GetLayout(render_view)
    if layout:
        layout.SetSize(width, height)
    render_view.OrientationAxesVisibility = 0
    render_view.UseColorPaletteForBackground = 0
    render_view.Background = [0.0, 0.0, 0.0]
    render_view.Background2 = [0.0, 0.0, 0.0]
    try:
        render_view.BackgroundColorMode = 0
    except AttributeError:
        render_view.UseGradientBackground = 0

    display = Show(data_source, render_view)
    display.Representation = "Point Gaussian"
    display.ShaderPreset = "Gaussian Blur"
    display.GaussianRadius = gaussian_radius
    display.Opacity = opacity
    display.ColorArrayName = [None, ""]
    display.DiffuseColor = [1.0, 1.0, 1.0]

    render_view.ResetCamera()
    Render()

    # Setup ctypes bypass for fast per-frame parameter changes
    _bypass_scale, _bypass_opacity = _setup_bypass(
        render_view, display, gaussian_radius, opacity)
    if _bypass_scale is not None:
        print(f"[pvbatch] ctypes bypass enabled")
    else:
        print(f"[pvbatch] ctypes bypass not available, using standard rendering")

    fov_deg = render_view.CameraViewAngle
    original_pos = list(render_view.CameraPosition)
    original_focal = list(render_view.CameraFocalPoint)
    center = original_focal

    # Override center/radius from normalization if provided
    _norm_override_radius = None
    if normalization_path and os.path.exists(normalization_path):
        _norm_pre = load_normalization(normalization_path)
        _norm_override_center = _norm_pre["center"]
        _norm_override_radius = None if no_norm_radius else _norm_pre.get("radius")
        if _norm_override_center is not None and _norm_override_radius is not None:
            center = list(_norm_override_center)
            print(f"[pvbatch] camera center overridden by normalization: {center}")

    frames_data = []
    positions = []

    if camera_strategy == "external_spiral":
        diff_vec = vec_sub(original_pos, center)
        base_radius = vec_norm(diff_vec) if _norm_override_radius is None else _norm_override_radius
        zoom_factor = 0.65
        radius = base_radius * zoom_factor
        total_loops = int(math.sqrt(num_frames))
        for i in range(num_frames):
            y_n = (1 - (i / float(num_frames - 1)) * 2) * 0.99
            radius_at_y = math.sqrt(max(0, 1 - y_n * y_n))
            theta = 2 * math.pi * total_loops * (i / float(num_frames - 1))
            x_n = math.cos(theta) * radius_at_y
            z_n = math.sin(theta) * radius_at_y
            positions.append([
                center[0] + radius * x_n,
                center[1] + radius * y_n,
                center[2] + radius * z_n,
            ])

    elif camera_strategy == "internal_uniform":
        rng = np.random.RandomState(camera_seed)
        bounds = data_source.GetDataInformation().GetBounds()
        scaled_bounds = compute_internal_bounds(bounds, center, internal_bounds_scale)
        xmin, xmax, ymin, ymax, zmin, zmax = scaled_bounds
        for _ in range(num_frames):
            positions.append([
                rng.uniform(xmin, xmax),
                rng.uniform(ymin, ymax),
                rng.uniform(zmin, zmax),
            ])

    elif camera_strategy == "multi_orbit":
        diff_vec = vec_sub(original_pos, center)
        base_radius = vec_norm(diff_vec) if _norm_override_radius is None else _norm_override_radius
        orbit_radii_list = orbit_radii if orbit_radii else [0.65, 0.45, 0.30]
        n_orbits = len(orbit_radii_list)
        frames_per_orbit = num_frames // n_orbits
        remainder = num_frames - frames_per_orbit * n_orbits
        orbit_frame_counts = []
        for k in range(n_orbits):
            extra = 1 if k < remainder else 0
            orbit_frame_counts.append(frames_per_orbit + extra)

        for k, orbit_frac in enumerate(orbit_radii_list):
            orbit_r = base_radius * orbit_frac
            n_frames_k = orbit_frame_counts[k]
            total_loops = max(1, int(math.sqrt(n_frames_k)))
            for i in range(n_frames_k):
                y_n = (1 - (i / float(max(n_frames_k - 1, 1))) * 2) * 0.99
                radius_at_y = math.sqrt(max(0, 1 - y_n * y_n))
                theta = 2 * math.pi * total_loops * (i / float(max(n_frames_k - 1, 1)))
                x_n = math.cos(theta) * radius_at_y
                z_n = math.sin(theta) * radius_at_y
                positions.append([
                    center[0] + orbit_r * x_n,
                    center[1] + orbit_r * y_n,
                    center[2] + orbit_r * z_n,
                ])

        rng_shuffle = np.random.RandomState(camera_seed)
        shuffle_idx = list(range(len(positions)))
        rng_shuffle.shuffle(shuffle_idx)
        positions = [positions[i] for i in shuffle_idx]
    else:
        raise ValueError(f"Unknown camera_strategy: {camera_strategy}")

    # Compute normalization
    if normalization_path and os.path.exists(normalization_path):
        norm_data = load_normalization(normalization_path)
        norm_center = norm_data["center"]
        scale_factor = norm_data["scale"]
        norm_radius = norm_data.get("radius")
    else:
        if camera_strategy == "external_spiral":
            diff_vec = vec_sub(original_pos, center)
            base_radius = vec_norm(diff_vec)
            norm_radius = base_radius * 0.65
        elif camera_strategy == "multi_orbit":
            diff_vec = vec_sub(original_pos, center)
            base_radius = vec_norm(diff_vec)
            orbit_radii_list = orbit_radii if orbit_radii else [0.65, 0.45, 0.30]
            norm_radius = base_radius * max(orbit_radii_list)
        else:
            dist = [vec_norm(vec_sub(pos, center)) for pos in positions]
            norm_radius = max(dist) if dist else 1.0
        target_radius = 4.0
        scale_factor = target_radius / norm_radius
        norm_center = center

    # Viz parameter sampling
    viz_rng = np.random.RandomState(viz_seed)
    if radius_range is None:
        radius_range = [gaussian_radius, gaussian_radius]
    if opacity_range is None:
        opacity_range = [opacity, opacity]
    radius_min, radius_max = float(radius_range[0]), float(radius_range[1])
    opacity_min, opacity_max = float(opacity_range[0]), float(opacity_range[1])

    for i, pos in enumerate(positions):
        render_view.CameraPosition = pos
        render_view.CameraFocalPoint = center
        view_up = choose_view_up(vec_sub(center, pos))
        render_view.CameraViewUp = view_up
        try:
            render_view.ResetCameraClippingRange()
        except AttributeError:
            pass
        if layout:
            layout.SetSize(width, height)
        try:
            render_view.UseLODForStillRender = 0
        except AttributeError:
            pass
        try:
            render_view.UseLODForInteractiveRender = 0
        except AttributeError:
            pass

        current_radius = gaussian_radius
        current_opacity = opacity
        if viz_mode == "sampled":
            if viz_distribution == "beta":
                a = viz_beta_concentration
                current_radius = radius_min + (radius_max - radius_min) * viz_rng.beta(a, a)
                current_opacity = opacity_min + (opacity_max - opacity_min) * viz_rng.beta(a, a)
            else:
                current_radius = viz_rng.uniform(radius_min, radius_max)
                current_opacity = viz_rng.uniform(opacity_min, opacity_max)
            current_opacity = min(current_opacity, 1.0)
        elif viz_mode != "fixed":
            raise ValueError(f"Unknown viz_mode: {viz_mode}")

        if _bypass_scale is not None:
            _bypass_scale(current_radius)
            _bypass_opacity(current_opacity)
            Render()
        else:
            display.GaussianRadius = current_radius
            display.Opacity = current_opacity
            Render()

        image_name = f"{i:04d}.png"
        image_path = os.path.join(images_dir, image_name)
        SaveScreenshot(image_path, render_view, TransparentBackground=0)

        rel_pos = vec_sub(pos, norm_center)
        norm_pos = [x * scale_factor for x in rel_pos]
        norm_focal = [0.0, 0.0, 0.0]
        c2w = get_camera_matrix(norm_pos, norm_focal, view_up)
        frames_data.append({
            "file_path": f"images/{image_name.split('.')[0]}",
            "transform_matrix": c2w,
            "viz_radius": current_radius,
            "viz_opacity": current_opacity,
        })

    # Write transforms
    fov_rad_y = math.radians(fov_deg)
    aspect_ratio = width / height
    fov_rad_x = 2 * math.atan(math.tan(fov_rad_y / 2) * aspect_ratio)

    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1 (inclusive), got {train_ratio}")
    total_frames = len(frames_data)
    if train_ratio == 1.0:
        train_frames = frames_data
        test_frames = []
    else:
        rng = np.random.RandomState(split_seed)
        indices = np.arange(total_frames)
        rng.shuffle(indices)
        train_count = int(total_frames * train_ratio)
        train_count = max(1, min(train_count, total_frames - 1))
        train_idx = set(indices[:train_count].tolist())
        train_frames = [f for idx, f in enumerate(frames_data) if idx in train_idx]
        test_frames = [f for idx, f in enumerate(frames_data) if idx not in train_idx]

    with open(os.path.join(output_dir, "transforms_train.json"), "w") as f:
        json.dump({"camera_angle_x": fov_rad_x, "frames": train_frames}, f, indent=4)
    with open(os.path.join(output_dir, "transforms_test.json"), "w") as f:
        json.dump({"camera_angle_x": fov_rad_x, "frames": test_frames}, f, indent=4)

    save_normalization(os.path.join(output_dir, "normalization.json"), {
        "center": norm_center,
        "scale": scale_factor,
        "radius": norm_radius,
    })


def main():
    parser = argparse.ArgumentParser(description="Generate images/transforms with progressive camera strategies")
    parser.add_argument("--vtp_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gaussian_radius", type=float, default=0.02)
    parser.add_argument("--opacity", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--camera_strategy", type=str, default="external_spiral")
    parser.add_argument("--internal_bounds_scale", type=float, default=0.9)
    parser.add_argument("--normalization_path", type=str, default=None)
    parser.add_argument("--camera_seed", type=int, default=42)
    parser.add_argument("--viz_mode", type=str, default="fixed", choices=["fixed", "sampled"])
    parser.add_argument("--radius_min", type=float, default=None)
    parser.add_argument("--radius_max", type=float, default=None)
    parser.add_argument("--opacity_min", type=float, default=None)
    parser.add_argument("--opacity_max", type=float, default=None)
    parser.add_argument("--viz_seed", type=int, default=42)
    parser.add_argument("--viz_distribution", type=str, default="uniform",
                        choices=["uniform", "beta"])
    parser.add_argument("--viz_beta_concentration", type=float, default=3.0)
    parser.add_argument("--orbit_radii", type=str, default=None,
                        help="Comma-separated orbit radius fractions (e.g. 0.65,0.45,0.30)")
    parser.add_argument("--no_norm_radius", action="store_true")
    args, _ = parser.parse_known_args()

    orbit_radii_parsed = None
    if args.orbit_radii:
        orbit_radii_parsed = [float(x.strip()) for x in args.orbit_radii.split(",")]

    ensure_dir(args.output_dir)
    generate_images_and_cameras(
        args.vtp_path, args.output_dir, args.gaussian_radius, args.opacity,
        width=args.width, height=args.height,
        num_frames=args.num_frames, train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        camera_strategy=args.camera_strategy,
        internal_bounds_scale=args.internal_bounds_scale,
        normalization_path=args.normalization_path,
        camera_seed=args.camera_seed,
        viz_mode=args.viz_mode,
        radius_range=[
            args.gaussian_radius if args.radius_min is None else args.radius_min,
            args.gaussian_radius if args.radius_max is None else args.radius_max,
        ],
        opacity_range=[
            args.opacity if args.opacity_min is None else args.opacity_min,
            args.opacity if args.opacity_max is None else args.opacity_max,
        ],
        viz_seed=args.viz_seed, viz_distribution=args.viz_distribution,
        viz_beta_concentration=args.viz_beta_concentration,
        orbit_radii=orbit_radii_parsed,
        no_norm_radius=args.no_norm_radius,
    )


if __name__ == "__main__":
    main()
