#!/usr/bin/env python3
"""Unified data preparation: raw .f32 -> VTP -> ParaView renders -> PLY init.

Pipeline steps:
  1. raw .f32 -> VTP (VTK polydata with per-particle vertices)
  2. VTP -> images + cameras (ParaView renders with spiral camera)
  3. raw .f32 + normalization -> PLY (initial 3DGS point cloud)

Steps 1-2 require pvbatch (ParaView headless). Step 3 only needs numpy + plyfile.

Usage (full pipeline):
    pvbatch --force-offscreen-rendering -- prepare_data.py \
        --raw_x data/hacc_raw/xx.f32 \
        --raw_y data/hacc_raw/yy.f32 \
        --raw_z data/hacc_raw/zz.f32 \
        --output_dir /tmp/my_data \
        --gaussian_radius 0.02 --opacity 0.1

Usage (only regenerate PLY):
    python prepare_data.py \
        --raw_x data/hacc_raw/xx.f32 \
        --raw_y data/hacc_raw/yy.f32 \
        --raw_z data/hacc_raw/zz.f32 \
        --output_dir /tmp/my_data \
        --only_ply --num_points_ply 200000
"""

import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import vtk
    from vtk.util import numpy_support
except ImportError:
    vtk = None

try:
    from paraview.simple import *
    PARAVIEW_AVAILABLE = True
except ImportError:
    PARAVIEW_AVAILABLE = False

try:
    from plyfile import PlyData, PlyElement
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def load_raw_data(path, count=None):
    """Load raw float32 binary data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    print(f"Loading raw file: {path}")
    if count is None:
        data = np.fromfile(path, dtype=np.float32)
    else:
        with open(path, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32, count=count)
    return data


# ================= PHASE 1: RAW -> VTP =================

def generate_vtp(input_x, input_y, input_z, output_vtp_path, num_points):
    if num_points is not None and num_points <= 0:
        num_points = None
    if vtk is None:
        raise ImportError("vtk library is required. Install with: pip install vtk")

    if num_points is None:
        print("\n[Phase 1] Generating VTP file with all points...")
    else:
        print(f"\n[Phase 1] Generating VTP file with {num_points} points...")

    x = load_raw_data(input_x, num_points)
    y = load_raw_data(input_y, num_points)
    z = load_raw_data(input_z, num_points)

    min_len = min(len(x), len(y), len(z))
    if len(x) != min_len:
        print(f"Warning: Truncating arrays to shortest length: {min_len}")
        x, y, z = x[:min_len], y[:min_len], z[:min_len]
    num_points = min_len

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(num_points)
    coords = np.column_stack((x, y, z)).astype(np.float32)
    points.SetData(numpy_support.numpy_to_vtk(coords, deep=True))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    vertices = vtk.vtkCellArray()
    vertex_array = np.ones((num_points * 2,), dtype=np.int64)
    vertex_array[::2] = 1
    vertex_array[1::2] = np.arange(num_points)
    vertices.SetCells(num_points, numpy_support.numpy_to_vtkIdTypeArray(vertex_array))
    polydata.SetVerts(vertices)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_vtp_path)
    writer.SetInputData(polydata)
    writer.Write()

    print(f"Saved VTP to {output_vtp_path}")
    return output_vtp_path


# ================= PHASE 2: VTP -> IMAGES & CAMERAS =================

def vec_sub(a, b): return [x - y for x, y in zip(a, b)]
def vec_norm(a): return math.sqrt(sum(x * x for x in a))
def vec_normalize(a):
    m = vec_norm(a)
    return [x / m for x in a] if m > 0 else [0.0, 0.0, 1.0]
def vec_cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]


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
        [0.0, 0.0, 0.0, 1.0]
    ]


def generate_images_and_cameras(
    vtp_path, output_dir, gaussian_radius, opacity,
    width=1920, height=1080, num_frames=400, train_ratio=0.8, split_seed=42,
):
    print(f"\n[Phase 2] Generating Images and Cameras...")
    if not PARAVIEW_AVAILABLE:
        raise ImportError("paraview.simple is required. Run with pvbatch.")

    images_dir = os.path.join(output_dir, "images")
    ensure_dir(images_dir)

    data_source = OpenDataFile(vtp_path)
    if not data_source:
        raise RuntimeError(f"Could not open VTP file: {vtp_path}")

    renderView = CreateRenderView()
    renderView.ViewSize = [width, height]
    layout = GetLayout(renderView)
    if layout:
        layout.SetSize(width, height)
    renderView.OrientationAxesVisibility = 0
    renderView.UseColorPaletteForBackground = 0
    renderView.Background = [0.0, 0.0, 0.0]
    renderView.Background2 = [0.0, 0.0, 0.0]
    try:
        renderView.BackgroundColorMode = 0
    except AttributeError:
        renderView.UseGradientBackground = 0

    display = Show(data_source, renderView)
    display.Representation = 'Point Gaussian'
    display.ShaderPreset = 'Gaussian Blur'
    display.GaussianRadius = gaussian_radius
    display.Opacity = opacity
    display.ColorArrayName = [None, '']
    display.DiffuseColor = [1.0, 1.0, 1.0]

    renderView.ResetCamera()
    Render()

    fov_deg = renderView.CameraViewAngle
    original_pos = list(renderView.CameraPosition)
    original_focal = list(renderView.CameraFocalPoint)
    center = original_focal

    diff_vec = vec_sub(original_pos, center)
    base_radius = vec_norm(diff_vec)
    ZOOM_FACTOR = 0.65
    radius = base_radius * ZOOM_FACTOR

    total_loops = int(math.sqrt(num_frames))
    frames_data = []

    print(f"Rendering {num_frames} frames...")
    for i in range(num_frames):
        y_n = (1 - (i / float(num_frames - 1)) * 2) * 0.99
        radius_at_y = math.sqrt(max(0, 1 - y_n * y_n))
        theta = 2 * math.pi * total_loops * (i / float(num_frames - 1))

        x_n = math.cos(theta) * radius_at_y
        z_n = math.sin(theta) * radius_at_y

        pos = [
            center[0] + radius * x_n,
            center[1] + radius * y_n,
            center[2] + radius * z_n,
        ]

        renderView.CameraPosition = pos
        renderView.CameraFocalPoint = center
        renderView.CameraViewUp = [0, 1, 0]
        try:
            renderView.ResetCameraClippingRange()
        except AttributeError:
            pass

        if layout:
            layout.SetSize(width, height)
        try:
            renderView.UseLODForStillRender = 0
        except AttributeError:
            pass
        try:
            renderView.UseLODForInteractiveRender = 0
        except AttributeError:
            pass

        Render()

        image_name = f"{i:04d}.png"
        image_path = os.path.join(images_dir, image_name)
        SaveScreenshot(image_path, renderView, TransparentBackground=0)

        TARGET_RADIUS = 4.0
        scale_factor = TARGET_RADIUS / radius

        rel_pos = vec_sub(pos, center)
        norm_pos = [x * scale_factor for x in rel_pos]
        norm_focal = [0.0, 0.0, 0.0]
        norm_up = [0, 1, 0]

        c2w = get_camera_matrix(norm_pos, norm_focal, norm_up)
        frames_data.append({
            "file_path": f"images/{image_name.split('.')[0]}",
            "transform_matrix": c2w,
        })

        if i % 50 == 0:
            print(f"  Frame {i}/{num_frames}")

    fov_rad_y = math.radians(fov_deg)
    aspect_ratio = width / height
    fov_rad_x = 2 * math.atan(math.tan(fov_rad_y / 2) * aspect_ratio)

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    total_frames = len(frames_data)
    rng = np.random.RandomState(split_seed)
    indices = np.arange(total_frames)
    rng.shuffle(indices)
    train_count = int(total_frames * train_ratio)
    train_count = max(1, min(train_count, total_frames - 1))
    train_idx = set(indices[:train_count].tolist())
    train_frames = [f for i, f in enumerate(frames_data) if i in train_idx]
    test_frames = [f for i, f in enumerate(frames_data) if i not in train_idx]

    json_data_train = {"camera_angle_x": fov_rad_x, "frames": train_frames}
    json_data_test = {"camera_angle_x": fov_rad_x, "frames": test_frames}

    with open(os.path.join(output_dir, "transforms_train.json"), 'w') as f:
        json.dump(json_data_train, f, indent=4)
    with open(os.path.join(output_dir, "transforms_test.json"), 'w') as f:
        json.dump(json_data_test, f, indent=4)

    norm_data = {"center": center, "scale": scale_factor, "radius": radius}
    norm_path = os.path.join(output_dir, "normalization.json")
    with open(norm_path, 'w') as f:
        json.dump(norm_data, f, indent=4)

    print("Images and camera poses generated.")
    return norm_path


# ================= PHASE 3: RAW -> PLY =================

def generate_ply(input_x, input_y, input_z, output_ply_path, norm_path, num_points, scale_override=None):
    print(f"\n[Phase 3] Generating Initial PLY with {num_points} points...")
    if not PLYFILE_AVAILABLE:
        raise ImportError("plyfile module is required.")

    with open(norm_path, 'r') as f:
        norm_data = json.load(f)
        center = np.array(norm_data["center"])
        scale = norm_data["scale"]

    if scale_override is not None:
        print(f"  [scale override] {scale:.10e} -> {scale_override:.10e}")
        scale = scale_override

    print("Loading all points for random sampling...")
    x = load_raw_data(input_x, count=None)
    y = load_raw_data(input_y, count=None)
    z = load_raw_data(input_z, count=None)

    total_points = min(len(x), len(y), len(z))
    print(f"Total points available: {total_points}")

    if num_points < total_points:
        print(f"Randomly sampling {num_points} points from {total_points}...")
        np.random.seed(42)
        indices = np.random.choice(total_points, size=num_points, replace=False)
        x, y, z = x[indices], y[indices], z[indices]
    else:
        print(f"Using all {total_points} points (requested {num_points})")
        x, y, z = x[:total_points], y[:total_points], z[:total_points]

    xyz = np.column_stack((x, y, z))
    xyz = (xyz - center) * scale

    num_pts = xyz.shape[0]
    rgb = np.full((num_pts, 3), 255, dtype=np.uint8)
    normals = np.zeros_like(xyz)

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ]
    elements = np.empty(num_pts, dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['red'] = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue'] = rgb[:, 2]

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(output_ply_path)

    print(f"Saved PLY to {output_ply_path}")


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser(description="Data preparation for ParticleGS")
    parser.add_argument("--raw_x", required=True, help="Path to xx.f32")
    parser.add_argument("--raw_y", required=True, help="Path to yy.f32")
    parser.add_argument("--raw_z", required=True, help="Path to zz.f32")
    parser.add_argument("--output_dir", required=True, help="Result directory")
    parser.add_argument("--num_points_raw", type=int, default=0,
                        help="Number of points for VTP/Rendering (0 for all)")
    parser.add_argument("--num_points_ply", type=int, default=200000,
                        help="Number of points for PLY initialization")
    parser.add_argument("--ply_scale_override", type=float, default=None,
                        help="Override normalization scale when writing PLY "
                             "(used to reproduce the FIRE-2 legacy PLY scale; "
                             "leave unset for normal runs).")
    parser.add_argument("--gaussian_radius", type=float, default=0.02)
    parser.add_argument("--opacity", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--skip_vtp", action="store_true")
    parser.add_argument("--skip_images", action="store_true")
    parser.add_argument("--only_ply", action="store_true",
                        help="Only regenerate PLY (implies --skip_vtp --skip_images)")
    args, _ = parser.parse_known_args()

    if args.only_ply:
        args.skip_vtp = True
        args.skip_images = True

    ensure_dir(args.output_dir)

    # Step 1: VTP
    vtp_path = os.path.join(args.output_dir, "particles.vtp")
    if args.skip_vtp:
        if os.path.exists(vtp_path):
            print(f"[SKIP] Reusing existing VTP: {vtp_path}")
        else:
            print(f"[WARNING] --skip_vtp set but VTP not found. Generating...")
            generate_vtp(args.raw_x, args.raw_y, args.raw_z, vtp_path, args.num_points_raw)
    else:
        generate_vtp(args.raw_x, args.raw_y, args.raw_z, vtp_path, args.num_points_raw)

    # Step 2: Images
    norm_path = os.path.join(args.output_dir, "normalization.json")
    if args.skip_images:
        if os.path.exists(norm_path):
            print(f"[SKIP] Reusing existing normalization: {norm_path}")
        else:
            print(f"[WARNING] --skip_images set but normalization not found. Generating...")
            norm_path = generate_images_and_cameras(
                vtp_path, args.output_dir, args.gaussian_radius, args.opacity,
                width=args.width, height=args.height, num_frames=args.num_frames,
                train_ratio=args.train_ratio, split_seed=args.split_seed,
            )
    else:
        norm_path = generate_images_and_cameras(
            vtp_path, args.output_dir, args.gaussian_radius, args.opacity,
            width=args.width, height=args.height, num_frames=args.num_frames,
            train_ratio=args.train_ratio, split_seed=args.split_seed,
        )

    # Step 3: PLY
    ply_path = os.path.join(args.output_dir, "points3d.ply")
    generate_ply(args.raw_x, args.raw_y, args.raw_z, ply_path, norm_path,
                 args.num_points_ply, scale_override=args.ply_scale_override)

    print("\n=== Pipeline Complete ===")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
