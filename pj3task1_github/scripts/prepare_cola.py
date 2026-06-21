#!/usr/bin/env python3
"""
Extract textured mesh from Magic123 refine checkpoint, save as PLY + vertex colors.
Then render multi-view images for 2DGS training.
"""
import os, sys, json, argparse, subprocess
import numpy as np
import torch
from pathlib import Path

CODE_DIR = "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen"
THREESTUDIO_DIR = CODE_DIR + "/threestudio"
os.chdir(THREESTUDIO_DIR)
sys.path.insert(0, THREESTUDIO_DIR)

CKPT_PATH = CODE_DIR + "/threestudio-outputs/magic123-refine-sd/bk_cola.jpg-a_Burger_King_Coca-Cola_bottle@20260617-091227/ckpts/last.ckpt"
OUT_DIR = CODE_DIR + "/rendered_data/cola"

os.makedirs(OUT_DIR, exist_ok=True)

print("[1/5] Loading checkpoint...")
import threestudio
from threestudio.models.geometry.tetrahedra_sdf_grid import TetrahedraSDFGrid
from threestudio.utils.config import parse_structured

device = torch.device("cuda")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
state = ckpt["state_dict"]
bbox = state["geometry.bbox"]
print(f"  bbox: {bbox}")

geo_cfg = parse_structured(TetrahedraSDFGrid.Config, {
    "radius": 1.0,
    "isosurface_resolution": 128,
    "isosurface_deformable_grid": True,
    "fix_geometry": False,
    "pos_encoding_config": {
        "otype": "HashGrid", "n_levels": 16, "n_features_per_level": 2,
        "log2_hashmap_size": 19, "base_resolution": 16,
        "per_level_scale": 1.447269237440378,
    },
    "mlp_network_config": {
        "otype": "VanillaMLP", "activation": "ReLU",
        "output_activation": "none", "n_neurons": 64, "n_hidden_layers": 2,
    },
})
geo = TetrahedraSDFGrid(geo_cfg)
geo.bbox = bbox
geo.isosurface_bbox = bbox.clone()
geo = geo.to(device)
geo.eval()

geo.load_state_dict(
    {k.replace("geometry.", ""): v for k, v in state.items() if k.startswith("geometry.")},
    strict=False
)

print("[2/5] Extracting mesh via isosurface...")
with torch.no_grad():
    mesh = geo.isosurface()
    v_pos = mesh.v_pos.detach().cpu().numpy()
    t_pos = mesh.t_pos_idx.detach().cpu().numpy()
    print(f"  Mesh: {v_pos.shape[0]} verts, {t_pos.shape[0]} faces")

if v_pos.shape[0] == 0:
    print("ERROR: Empty mesh!"); exit(1)

print("[3/5] Computing vertex colors...")
with torch.no_grad():
    geo_out = geo.forward(mesh.v_pos.to(device))
    raw_colors = geo_out.get("features", None)

    if raw_colors is not None:
        colors = raw_colors.detach().cpu().numpy()
        print(f"  Raw colors shape: {colors.shape}, range: [{colors.min():.4f}, {colors.max():.4f}]")
        if colors.ndim == 1:
            colors = colors[:, None].repeat(3, axis=1)
        # Normalize to [0, 1]
        if colors.min() < 0 or colors.max() > 1:
            colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
    else:
        colors = np.ones((v_pos.shape[0], 3), dtype=np.float32) * 0.7

print(f"  Colors: [{colors.min():.4f}, {colors.max():.4f}]")

print("[4/5] Saving mesh PLY with vertex colors...")
# Save as PLY in trimesh-compatible format
ply_path = os.path.join(OUT_DIR, "cola_mesh.ply")
colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

with open(ply_path, 'w') as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {v_pos.shape[0]}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write(f"element face {t_pos.shape[0]}\n")
    f.write("property list uchar int vertex_indices\n")
    f.write("end_header\n")
    for i in range(v_pos.shape[0]):
        f.write(f"{v_pos[i,0]:.6f} {v_pos[i,1]:.6f} {v_pos[i,2]:.6f} {colors_uint8[i,0]} {colors_uint8[i,1]} {colors_uint8[i,2]}\n")
    for i in range(t_pos.shape[0]):
        f.write(f"3 {t_pos[i,0]} {t_pos[i,1]} {t_pos[i,2]}\n")

file_size = os.path.getsize(ply_path) / 1024 / 1024
print(f"  Saved: {ply_path} ({file_size:.1f} MB)")

print("[5/5] Rendering multi-view images...")
render_cmd = [
    "python", CODE_DIR + "/scripts/render_multiview.py",
    "--mesh", ply_path,
    "--out", OUT_DIR,
    "--n_views", "120",
    "--resolution", "800",
    "--fov_y", "40",
    "--radius", "3.0",
    "--elev_range", "-20", "60",
]

subprocess.run(render_cmd, check=True)
print(f"\nDone! Cola data ready at {OUT_DIR}")
