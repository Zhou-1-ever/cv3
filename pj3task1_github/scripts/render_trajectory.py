#!/usr/bin/env python3
"""
Multi-view trajectory renderer for fused scene.
Generates a smooth circular camera path and compiles video.
"""
import sys, os, json, subprocess
import numpy as np
import torch
from math import cos, sin, pi
from argparse import Namespace
from pathlib import Path

# === CONFIG ===
PJ = "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1"
MERGED_PLY = os.path.join(PJ, "fusion_output/merged_scene.ply")
OUT_DIR = os.path.join(PJ, "fusion_output/frames_v2")
OUT_VIDEO = os.path.join(PJ, "fusion_output/scene_trajectory_v2.mp4")

os.makedirs(OUT_DIR, exist_ok=True)

# 2DGS paths
for p in [
    os.path.join(PJ, "2dgs/submodules/diff-surfel-rasterization"),
    os.path.join(PJ, "2dgs/submodules/simple-knn"),
    os.path.join(PJ, "2dgs"),
]:
    sys.path.insert(0, p)

from scene import GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera
from torchvision.utils import save_image

# === Load scene ===
device = "cuda"
print("Loading merged scene...")
gaussians = GaussianModel(3)
gaussians.load_ply(MERGED_PLY)
print(f"  {gaussians._xyz.shape[0]} points loaded")

# === Camera trajectory ===
# Scene center: objects are on counter at y≈1.44, x≈[-6,6], z≈[0,4]
# Camera orbits around center (0, 1.8, 2.0) at radius R
CENTER = np.array([0.0, 1.8, 2.0], dtype=np.float32)
RADIUS = 8.0
CAM_HEIGHT_MIN = 0.5   # height above center
CAM_HEIGHT_MAX = 3.5
N_FRAMES = 240         # ~8 seconds at 30fps

pipeline = Namespace(
    compute_cov3D_python=False, debug=False,
    convert_SHs_python=False, depth_ratio=0.0
)
bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
H, W = 720, 1280
fov = 50 * pi / 180

print(f"Rendering {N_FRAMES} frames...")
for i in range(N_FRAMES):
    t = i / N_FRAMES

    # Orbit azimuth: 2 full rotations
    azim = 4 * pi * t
    # Height oscillation: gentle up and down
    height_offset = CAM_HEIGHT_MIN + (CAM_HEIGHT_MAX - CAM_HEIGHT_MIN) * (0.5 + 0.5 * sin(2 * pi * t * 0.5))

    # Camera position on orbit
    cam_x = CENTER[0] + RADIUS * cos(azim)
    cam_z = CENTER[2] + RADIUS * sin(azim)
    cam_y = CENTER[1] + height_offset

    lookat = torch.tensor(CENTER, device=device)
    cam_pos = torch.tensor([cam_x, cam_y, cam_z], device=device)

    # Look-at direction
    z_axis = (lookat - cam_pos) / torch.norm(lookat - cam_pos)
    up = torch.tensor([0., 1., 0.], device=device)
    x_axis = torch.cross(up, z_axis, dim=0) / (torch.norm(torch.cross(up, z_axis, dim=0)) + 1e-12)
    y_axis = torch.cross(z_axis, x_axis, dim=0)

    R_mat = torch.stack([x_axis, y_axis, z_axis], dim=0).cpu().numpy()
    T_vec = -R_mat @ cam_pos.cpu().numpy()

    dummy = torch.ones((3, H, W), device=device)
    cam = Camera(0, R_mat.astype(np.float32), T_vec.astype(np.float32),
                 fov, fov, dummy, None, f"frame_{i:04d}", 0,
                 np.array([0., 0., 0.]), 1.0, device)

    with torch.no_grad():
        out = render(cam, gaussians, pipeline, bg)
        img = out["render"]

    save_image(img, os.path.join(OUT_DIR, f"frame_{i:04d}.png"))

    if (i + 1) % 30 == 0 or i == 0:
        nonwhite = (img < 0.99).sum().item()
        print(f"  [{i+1:3d}/{N_FRAMES}] azim={azim*180/pi:5.1f}°, mean={img.mean().item():.4f}, nw={nonwhite}px")

print(f"\nAll frames saved to {OUT_DIR}")

# === Compile video ===
print("Compiling video...")
cmd = [
    "ffmpeg", "-y",
    "-framerate", "30",
    "-i", os.path.join(OUT_DIR, "frame_%04d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-preset", "medium",
    "-crf", "18",
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # ensure even dimensions
    OUT_VIDEO
]
subprocess.run(cmd, check=True, capture_output=True)
print(f"Video saved: {OUT_VIDEO} ({os.path.getsize(OUT_VIDEO)/1024/1024:.1f} MB)")

print("\nDone!")
