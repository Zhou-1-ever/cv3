#!/usr/bin/env python3
"""
Multi-view renderer for meshes using nvdiffrast.
Generates images + transforms.json in Blender/NeRF format for 2DGS training.

Usage:
    # For textured mesh (hamburger .obj)
    python render_multiview.py --mesh path/to/model.obj --out rendered_data/hamburger

    # For vertex-colored mesh (cola from checkpoint)
    python render_multiview.py --mesh path/to/mesh.ply --out rendered_data/cola
"""

import os, sys, json, argparse
import numpy as np
from math import cos, sin, pi
from pathlib import Path

import torch
import nvdiffrast.torch as dr
import trimesh
from PIL import Image


def load_mesh(mesh_path):
    """Load mesh with trimesh, return verts, faces, uv, texture, vertex_colors."""
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No Trimesh objects found in scene")
        mesh = max(meshes, key=lambda m: len(m.vertices))
        print(f"  Scene flattened, using mesh with {len(mesh.vertices)} verts")

    verts = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)

    uv = None
    texture = None
    vertex_colors = None

    if mesh.visual.kind == 'texture':
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uv = mesh.visual.uv.astype(np.float32)
        mat = mesh.visual.material
        if mat is not None and hasattr(mat, 'image') and mat.image is not None:
            img = mat.image
            if img.mode != 'RGB':
                img = img.convert('RGB')
            texture = np.array(img, dtype=np.float32) / 255.0
            print(f"  Texture: {texture.shape}, UV: {uv.shape if uv is not None else None}")

    if mesh.visual.kind == 'vertex':
        vc = mesh.visual.vertex_colors
        if vc is not None:
            vertex_colors = np.array(vc, dtype=np.float32) / 255.0
            # Strip alpha if present (RGBA → RGB)
            if vertex_colors.shape[-1] == 4:
                vertex_colors = vertex_colors[..., :3]
            print(f"  Vertex colors: {vertex_colors.shape}")

    # Read text file with vertex colors if exists (same basename, .txt extension)
    if vertex_colors is None:
        txt_path = str(Path(mesh_path).with_suffix('.txt'))
        if os.path.exists(txt_path):
            vertex_colors = np.loadtxt(txt_path, dtype=np.float32)
            print(f"  Vertex colors from .txt: {vertex_colors.shape}")

    if texture is None and vertex_colors is None:
        print("  No texture/vertex colors found, using gray")
        vertex_colors = np.full((len(verts), 3), 0.7, dtype=np.float32)

    return verts, faces, uv, texture, vertex_colors


def look_at(eye, center, up):
    """Standard OpenGL lookAt view matrix (camera looks along -Z)."""
    f = center - eye
    fnorm = np.linalg.norm(f)
    if fnorm < 1e-12:
        f = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        f = f / fnorm
    s = np.cross(f, up)
    snorm = np.linalg.norm(s)
    if snorm < 1e-12:
        s = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        s = s / snorm
    u = np.cross(s, f)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[0, 3] = -float(np.dot(s, eye))
    view[1, 3] = -float(np.dot(u, eye))
    view[2, 3] = float(np.dot(f, eye))
    return view


def view_to_blender_c2w(view):
    """Convert OpenGL view matrix (world-to-camera) to Blender c2w format."""
    # view = [R | t], c2w_blender = inv(view) = [R^T | -R^T @ t]
    R = view[:3, :3]
    t = view[:3, 3]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def perspective(fov_y, aspect, z_near=0.01, z_far=100.0):
    """OpenGL perspective projection matrix."""
    f = 1.0 / np.tan(fov_y / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = -(z_far + z_near) / (z_far - z_near)
    proj[2, 3] = -2.0 * z_far * z_near / (z_far - z_near)
    proj[3, 2] = -1.0
    return proj


def render_view(glctx, verts, faces, uv, texture, vertex_colors,
                view, proj, resolution, bg_color=(1.0, 1.0, 1.0)):
    """Render one view of the mesh given view matrix and projection."""
    H, W = resolution

    # MVP = proj @ view
    mvp = proj @ view

    # Transform vertices to clip space: clip = mvp @ [x,y,z,1]
    v_hom = np.concatenate([verts, np.ones((verts.shape[0], 1), dtype=np.float32)], axis=-1)
    v_clip = v_hom @ mvp.T  # rows = clip space coords

    v_clip_tensor = torch.from_numpy(v_clip[None, ...]).cuda()
    faces_tensor = torch.from_numpy(faces).cuda()

    # Rasterize
    raster_out, _ = dr.rasterize(glctx, v_clip_tensor, faces_tensor, (H, W))
    mask = raster_out[..., 3:4]

    # Shade
    if texture is not None and uv is not None:
        uv_tensor = torch.from_numpy(uv[None, ...]).cuda()
        uv_interp, _ = dr.interpolate(uv_tensor, raster_out, faces_tensor)
        tex_tensor = torch.from_numpy(texture[None, ...]).cuda()
        color = dr.texture(tex_tensor, uv_interp)
    else:
        colors_tensor = torch.from_numpy(vertex_colors[None, ...]).cuda()
        color, _ = dr.interpolate(colors_tensor, raster_out, faces_tensor)

    # Alpha composite on background
    bg = torch.tensor(bg_color, dtype=torch.float32, device='cuda').view(1, 1, 1, 3)
    color = color * mask + bg * (1.0 - mask)

    return color[0], mask[0]


def main():
    parser = argparse.ArgumentParser(description='Multi-view mesh renderer')
    parser.add_argument('--mesh', type=str, required=True, help='Path to mesh file')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--n_views', type=int, default=100, help='Number of views')
    parser.add_argument('--resolution', type=int, default=800, help='Render resolution')
    parser.add_argument('--fov_y', type=float, default=40.0, help='Vertical FOV in degrees')
    parser.add_argument('--radius', type=float, default=3.0, help='Camera sphere radius')
    parser.add_argument('--elev_range', type=float, nargs=2, default=[-20, 60],
                        help='Elevation range [min, max] in degrees')
    parser.add_argument('--azim_start', type=float, default=0.0, help='Starting azimuth (degrees)')
    args = parser.parse_args()

    out_dir = Path(args.out)
    images_dir = out_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # --- Load mesh ---
    print(f"[1/5] Loading mesh: {args.mesh}")
    verts, faces, uv, texture, vertex_colors = load_mesh(args.mesh)

    # --- Normalize mesh (center at origin, max extent = 1) ---
    print(f"[2/5] Normalizing mesh ({len(verts)} verts, {len(faces)} faces)")
    center = verts.mean(axis=0)
    verts = verts - center
    max_ext = np.max(np.linalg.norm(verts, axis=1))
    verts = verts * (1.0 / max_ext)
    print(f"  Centered at {center}, max_extent={max_ext:.4f}, scale={1.0/max_ext:.4f}")

    # --- Setup nvdiffrast ---
    print(f"[3/5] Setting up nvdiffrast")
    glctx = dr.RasterizeCudaContext()

    fov_y_rad = np.deg2rad(args.fov_y)
    res = (args.resolution, args.resolution)
    aspect = 1.0
    proj = perspective(fov_y_rad, aspect)

    # --- Generate camera poses (Fibonacci sphere distribution) ---
    print(f"[4/5] Generating {args.n_views} camera poses")
    elev_min = np.deg2rad(args.elev_range[0])
    elev_max = np.deg2rad(args.elev_range[1])
    azim_start = np.deg2rad(args.azim_start)

    golden_ratio = (1 + 5**0.5) / 2
    frames_data = []

    for i in range(args.n_views):
        t = i / (args.n_views - 1) if args.n_views > 1 else 0.5
        elev = elev_min + t * (elev_max - elev_min)
        azim = azim_start + 2 * pi * i / golden_ratio

        # Camera position on sphere (Y up)
        x = args.radius * cos(elev) * cos(azim)
        y = args.radius * sin(elev)
        z = args.radius * cos(elev) * sin(azim)
        eye = np.array([x, y, z], dtype=np.float32)

        # OpenGL view matrix (camera looks at origin)
        view = look_at(eye, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))

        # Blender c2w for transforms.json
        c2w = view_to_blender_c2w(view)

        # Render
        color, mask = render_view(glctx, verts, faces, uv, texture, vertex_colors,
                                   view, proj, res, (1.0, 1.0, 1.0))

        # Save as RGBA PNG
        img_np = (color.cpu().numpy() * 255).astype(np.uint8)
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        rgba = np.concatenate([img_np, mask_np], axis=-1)

        fname = f'{i:05d}.png'
        Image.fromarray(rgba).save(str(images_dir / fname))

        frames_data.append({
            'file_path': f'images/{i:05d}',
            'transform_matrix': c2w.tolist()
        })

        if (i + 1) % 20 == 0 or i == 0 or i == args.n_views - 1:
            print(f"  Rendered {i + 1}/{args.n_views} (elev={np.rad2deg(elev):.0f}°, azim={np.rad2deg(azim):.0f}°)")

    # --- Write transforms.json ---
    fov_x = 2 * np.arctan(np.tan(fov_y_rad / 2) * aspect)

    transforms = {
        'camera_angle_x': float(fov_x),
        'frames': frames_data
    }

    with open(out_dir / 'transforms_train.json', 'w') as f:
        json.dump(transforms, f, indent=2)
    with open(out_dir / 'transforms_test.json', 'w') as f:
        json.dump(transforms, f, indent=2)

    print(f"\nDone! {args.n_views} views → {out_dir}")
    print(f"  FOV: {args.fov_y}° (Y) → {np.rad2deg(fov_x):.2f}° (X)")
    print(f"  Elevation: {args.elev_range[0]}° to {args.elev_range[1]}°")

    # Verify
    test_img = Image.open(images_dir / f'{0:05d}.png')
    print(f"  Sample image: {test_img.size}, mode={test_img.mode}")


if __name__ == '__main__':
    main()
