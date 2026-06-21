#!/usr/bin/env python3
"""
Scene Fusion: Merge all objects (Counter + Banana + Hamburger + Cola)
into one 2DGS scene, then render multi-view trajectory video.

All paths relative to project root (zhouhaowen).
"""
import os, sys, json, subprocess, numpy as np
import torch
from math import cos, sin, pi
from pathlib import Path
from scipy.spatial.transform import Rotation as R_scipy

# ========== CONFIG ==========
ROOT = "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen"
PJ_DIR = ROOT + "/pj3task1"
OUT_DIR = PJ_DIR + "/fusion_output"
os.makedirs(OUT_DIR, exist_ok=True)

# Paths to trained Gaussian PLY files
COUNTER_PLY = ROOT + "/2dgs/output/counter/point_cloud/iteration_30000/point_cloud.ply"
BANANA_PLY  = ROOT + "/2dgs/output/banana/point_cloud/iteration_30000/point_cloud.ply"
HAM_PLY     = PJ_DIR + "/output/2dgs_hamburger/point_cloud/iteration_7000/point_cloud.ply"
COLA_PLY    = PJ_DIR + "/output/2dgs_cola/point_cloud/iteration_7000/point_cloud.ply"

# ========== UTILITIES ==========

def read_ply_header(path):
    """Read PLY header and return (props_list, element_count, header_bytes, is_ascii)."""
    with open(path, "rb") as f:
        hdr = b""
        props = []
        n_verts = 0
        is_ascii = False
        fmt = "binary_little_endian"
        while True:
            line = f.readline()
            hdr += line
            s = line.decode().strip()
            if s.startswith("format"):
                if "ascii" in s:
                    is_ascii = True
                    fmt = "ascii"
                elif "binary_little_endian" in s:
                    fmt = "binary_little_endian"
            if s.startswith("element vertex"):
                n_verts = int(s.split()[-1])
            if s.startswith("property"):
                props.append(s.split()[2])
            if s == "end_header":
                break
    return props, n_verts, len(hdr), is_ascii, fmt


def read_ply_binary(path):
    """Read binary PLY, return structured numpy array."""
    props, n, hdr_len, is_ascii, fmt = read_ply_header(path)
    if is_ascii:
        raise ValueError("ASCII PLY not supported, convert first")
    dt = np.dtype([(p, "<f4") for p in props])
    with open(path, "rb") as f:
        f.seek(hdr_len)
        raw = f.read()
    data = np.frombuffer(raw[:n * dt.itemsize], dtype=dt)
    return data, props


def write_ply_binary(path, data, props):
    """Write structured numpy array as binary PLY (row-major)."""
    n = len(data)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        for p in props:
            f.write(f"property float {p}\n".encode())
        f.write(b"end_header\n")
        # Write row-major (vertex by vertex) -- PLY standard
        out = np.empty(n, dtype=[(p, "<f4") for p in props])
        for p in props:
            out[p] = data[p]
        out.tofile(f)


def quat_to_rotmat(q):
    """Convert (w, x, y, z) quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def rotmat_to_quat(R):
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion."""
    r = R_scipy.from_matrix(R)
    q = r.as_quat()  # (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]])  # (w, x, y, z)


def transform_gaussians(data, props, R, t, scale=1.0):
    """Transform Gaussian positions and rotations.
    R: 3x3 rotation matrix
    t: 3-element translation
    scale: uniform scale factor
    """
    new_data = data.copy()

    # Positions
    pts = np.column_stack([data["x"], data["y"], data["z"]])
    pts = pts * scale
    pts = pts @ R.T + t
    new_data["x"] = pts[:, 0]
    new_data["y"] = pts[:, 1]
    new_data["z"] = pts[:, 2]

    # Normals
    if "nx" in props:
        nrm = np.column_stack([data["nx"], data["ny"], data["nz"]])
        nrm = nrm @ R.T
        norms = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = np.divide(nrm, norms, where=norms > 1e-12, out=np.zeros_like(nrm))
        new_data["nx"] = nrm[:, 0]
        new_data["ny"] = nrm[:, 1]
        new_data["nz"] = nrm[:, 2]

    # Scale values (log-space)
    if "scale_0" in props and scale != 1.0:
        new_data["scale_0"] += np.log(max(scale, 1e-12))
        new_data["scale_1"] += np.log(max(scale, 1e-12))

    # Rotations (quaternions)
    if "rot_0" in props:
        R_tensor = torch.from_numpy(R).float().cuda()
        rot_0 = torch.from_numpy(data["rot_0"].copy()).float().cuda()
        rot_1 = torch.from_numpy(data["rot_1"].copy()).float().cuda()
        rot_2 = torch.from_numpy(data["rot_2"].copy()).float().cuda()
        rot_3 = torch.from_numpy(data["rot_3"].copy()).float().cuda()

        # Build quaternion as (w, x, y, z)
        q = torch.stack([rot_0, rot_1, rot_2, rot_3], dim=-1)  # [N, 4]

        # Normalize
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)

        # Convert to rotation matrix
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R_gauss = torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
            2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
            2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=-1).reshape(-1, 3, 3)  # [N, 3, 3]

        # Apply rotation: R_new = R_transform @ R_old
        R_new = R_tensor[None] @ R_gauss  # [N, 3, 3]

        # Convert back to quaternion
        # Using the standard formula
        tr = R_new[:, 0, 0] + R_new[:, 1, 1] + R_new[:, 2, 2]
        qw = torch.sqrt(torch.clamp(1 + tr, min=0)) / 2
        qx = (R_new[:, 2, 1] - R_new[:, 1, 2]) / (4 * qw + 1e-12)
        qy = (R_new[:, 0, 2] - R_new[:, 2, 0]) / (4 * qw + 1e-12)
        qz = (R_new[:, 1, 0] - R_new[:, 0, 1]) / (4 * qw + 1e-12)

        new_data["rot_0"] = qw.cpu().numpy()
        new_data["rot_1"] = qx.cpu().numpy()
        new_data["rot_2"] = qy.cpu().numpy()
        new_data["rot_3"] = qz.cpu().numpy()

    return new_data


# ========== MAIN ==========

print("=" * 60)
print("Scene Fusion: Counter + Banana + Hamburger + Cola")
print("=" * 60)

# [1] Load Counter (base scene)
print("\n[1/5] Loading Counter base scene...")
counter_data, counter_props = read_ply_binary(COUNTER_PLY)
print(f"  Counter: {len(counter_data)} Gaussians")
print(f"  Props ({len(counter_props)}): {counter_props[:6]}...")

# [2] Load and transform Banana
print("\n[2/5] Loading + transforming Banana...")
banana_data, banana_props = read_ply_binary(BANANA_PLY)
print(f"  Banana: {len(banana_data)} Gaussians")

# Banana transform: center → rotate 90° Z → scale → place on counter
pts_b = np.column_stack([banana_data["x"], banana_data["y"], banana_data["z"]])
center_b = pts_b.mean(axis=0)
pts_b_centered = pts_b - center_b

R_banana = np.array([[cos(pi/2), -sin(pi/2), 0],
                     [sin(pi/2), cos(pi/2), 0],
                     [0, 0, 1]], dtype=np.float32)

scale_b = 2.0 / (pts_b_centered[:, 0].max() - pts_b_centered[:, 0].min() + 1e-12)
t_banana = np.array([-4.0, 1.44, 2.5], dtype=np.float32)

# Adjust Y so bottom sits on counter surface (Y=1.44 guessed from counter scene)
bottom_y_b = (pts_b_centered * scale_b @ R_banana.T)[:, 1].min()
t_banana[1] = 1.44 - bottom_y_b

banana_transformed = transform_gaussians(banana_data, banana_props, R_banana, t_banana, scale_b)
print(f"  Transformed: position={t_banana}, scale={scale_b:.4f}")

# [3] Load and transform Hamburger
print("\n[3/5] Loading + transforming Hamburger...")
ham_data, ham_props = read_ply_binary(HAM_PLY)
print(f"  Hamburger: {len(ham_data)} Gaussians (trained at 7k iters)")

# Hamburger was trained in normalized space (centered, max_extent=1).
# The mesh had center=[-0.0053, 0.0480, -0.1950], max_extent=0.6149
# Normalization: verts = (mesh.verts - center) / max_extent * 1.0
# So to un-normalize: unnormed = (pts * max_extent + center)
# Then place on counter surface.

# Un-normalize from normalized space
ham_max_extent = 0.6149
ham_center = np.array([-0.00531452, 0.04799247, -0.19502233], dtype=np.float32)

pts_h = np.column_stack([ham_data["x"], ham_data["y"], ham_data["z"]])
pts_h = pts_h * ham_max_extent + ham_center

# Remove cola's ground offset (it's y-up) and place on counter
# Rotate to align with counter: 90° around Z
R_ham = np.eye(3, dtype=np.float32)

# Scale to reasonable size in counter scene (~0.3m wide)
ham_scale = 0.3 / (pts_h[:, 0].max() - pts_h[:, 0].min() + 1e-12)
pts_h = pts_h * ham_scale

# Place on counter: right side
bottom_y_h = pts_h[:, 1].min()
t_ham = np.array([2.0, 1.44 - bottom_y_h, 2.0], dtype=np.float32)
pts_h = pts_h + t_ham

# Now reconstruct ham_data with transformed positions
new_ham_data = np.zeros(len(ham_data), dtype=list(zip(counter_props, ["<f4"]*len(counter_props))))
for p in counter_props:
    if p in ham_props:
        new_ham_data[p] = ham_data[p]
    else:
        new_ham_data[p] = np.zeros(len(ham_data), dtype=np.float32)

    # Adjust scale for coordinate space change
    ham_scale_total = ham_max_extent * ham_scale
    if ham_scale_total > 0:
        new_ham_data["scale_0"] += np.log(ham_scale_total)
        new_ham_data["scale_1"] += np.log(ham_scale_total)

new_ham_data["x"] = pts_h[:, 0]
new_ham_data["y"] = pts_h[:, 1]
new_ham_data["z"] = pts_h[:, 2]

# Transform rotation quaternions
if "rot_0" in ham_props:
    R_ham_rot = np.eye(3, dtype=np.float32)  # no rotation for hamburger
    t_ham_rot = np.zeros(3, dtype=np.float32)
    ham_transformed = transform_gaussians(ham_data, ham_props, R_ham_rot, t_ham_rot)
    new_ham_data["rot_0"] = ham_transformed["rot_0"]
    new_ham_data["rot_1"] = ham_transformed["rot_1"]
    new_ham_data["rot_2"] = ham_transformed["rot_2"]
    new_ham_data["rot_3"] = ham_transformed["rot_3"]

print(f"  Position: {t_ham}, scale={ham_scale:.4f}")

# [4] Load and transform Cola (if trained PLY exists)
print("\n[4/5] Loading + transforming Cola...")
if os.path.exists(COLA_PLY):
    cola_data, cola_props = read_ply_binary(COLA_PLY)
    print(f"  Cola: {len(cola_data)} Gaussians")

    # Cola was also trained in normalized space
    cola_max_extent = 1.1535
    cola_center = np.array([0.27896884, 0.03299905, 0.07457108], dtype=np.float32)

    pts_c = np.column_stack([cola_data["x"], cola_data["y"], cola_data["z"]])
    pts_c = pts_c * cola_max_extent + cola_center

    # Scale
    cola_scale = 0.25 / (pts_c[:, 0].max() - pts_c[:, 0].min() + 1e-12)
    pts_c = pts_c * cola_scale

    # Place on left side of counter
    bottom_y_c = pts_c[:, 1].min()
    t_cola = np.array([6.0, 1.44 - bottom_y_c, 1.0], dtype=np.float32)
    pts_c = pts_c + t_cola

    new_cola_data = np.zeros(len(cola_data), dtype=list(zip(counter_props, ["<f4"]*len(counter_props))))
    for p in counter_props:
        if p in cola_props:
            new_cola_data[p] = cola_data[p]
        else:
            new_cola_data[p] = np.zeros(len(cola_data), dtype=np.float32)

    # Adjust scale for coordinate space change
    cola_scale_total = cola_max_extent * cola_scale
    if cola_scale_total > 0:
        new_cola_data["scale_0"] += np.log(cola_scale_total)
        new_cola_data["scale_1"] += np.log(cola_scale_total)

    new_cola_data["x"] = pts_c[:, 0]
    new_cola_data["y"] = pts_c[:, 1]
    new_cola_data["z"] = pts_c[:, 2]

    if "rot_0" in cola_props:
        R_cola_rot = np.eye(3, dtype=np.float32)
        t_cola_rot = np.zeros(3, dtype=np.float32)
        cola_transformed = transform_gaussians(cola_data, cola_props, R_cola_rot, t_cola_rot)
        new_cola_data["rot_0"] = cola_transformed["rot_0"]
        new_cola_data["rot_1"] = cola_transformed["rot_1"]
        new_cola_data["rot_2"] = cola_transformed["rot_2"]
        new_cola_data["rot_3"] = cola_transformed["rot_3"]

    print(f"  Position: {t_cola}, scale={cola_scale:.4f}")
else:
    print("  Cola PLY not found yet! Training probably still running.")
    new_cola_data = None

# [5] Merge all
print("\n[5/5] Merging all Gaussians...")
parts = [counter_data, banana_transformed, new_ham_data]
if new_cola_data is not None:
    parts.append(new_cola_data)

merged = np.concatenate(parts)
print(f"  Total: {len(merged)} Gaussians")
print(f"    Counter:  {len(counter_data)}")
print(f"    Banana:  {len(banana_transformed)}")
print(f"    Hamburger: {len(new_ham_data)}")
if new_cola_data is not None:
    print(f"    Cola:    {len(new_cola_data)}")

# Save merged PLY
out_ply = OUT_DIR + "/merged_scene.ply"
write_ply_binary(out_ply, merged, counter_props)
print(f"\n  Saved: {out_ply} ({os.path.getsize(out_ply)/1024/1024:.1f} MB)")

# Quick quality check
sys.path.insert(0, ROOT + "/2dgs")
from scene import GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera
from argparse import Namespace
from torchvision.utils import save_image

gaussians = GaussianModel(3)
gaussians.load_ply(out_ply)
op = torch.sigmoid(gaussians._opacity)
sc = gaussians.get_scaling
print(f"\n  Quality check:")
print(f"    Points: {gaussians._xyz.shape[0]}")
print(f"    Opacity: mean={op.mean().item():.4f}")
print(f"    Scales: mean={sc.mean().item():.6f} max={sc.max().item():.6f}")
print(f"    XYZ range: x=[{gaussians._xyz[:,0].min().item():.1f},{gaussians._xyz[:,0].max().item():.1f}]")

# Render a test view
pipeline = Namespace(compute_cov3D_python=False, debug=False, convert_SHs_python=False, depth_ratio=0.0)
bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
H, W = 720, 1280
fov = 50 * pi / 180

# Bird's eye view
center = torch.tensor([0.0, 2.0, 0.0], device="cuda")
cam_pos = torch.tensor([8.0, 8.0, 8.0], device="cuda")
z_axis = (cam_pos - center) / torch.norm(cam_pos - center)
up = torch.tensor([0., 1., 0.], device="cuda")
x_axis = torch.cross(up, z_axis) / torch.norm(torch.cross(up, z_axis))
y_axis = torch.cross(z_axis, x_axis)
R = torch.stack([x_axis, y_axis, z_axis], dim=0).cpu().numpy()
T = (-R @ cam_pos.cpu().numpy()).astype(np.float32)
dummy = torch.ones((3, H, W), device="cuda")

cam = Camera(0, R, T, fov, fov, dummy, None, "test", 0, np.array([0.,0.,0.]), 1.0, "cuda")
with torch.no_grad():
    out = render(cam, gaussians, pipeline, bg)
    img = out["render"]
    print(f"    Test render: mean={img.mean().item():.4f}, non-white: {(img < 0.99).sum().item()/1000:.0f}k px")

save_image(img, OUT_DIR + "/test_birds_eye.png")
print(f"    Saved: test_birds_eye.png")

print("\nDone! Run the rendering script separately to generate video.")

# Print paths for reference
print(f"\nPaths:")
print(f"  Merged PLY: {out_ply}")
print(f"  Test render: {OUT_DIR}/test_birds_eye.png")
