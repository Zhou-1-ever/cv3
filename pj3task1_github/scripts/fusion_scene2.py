#!/usr/bin/env python3
import os, sys, json, numpy as np, torch
from math import cos, sin, pi
from pathlib import Path

PJ = "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1"
OUT_DIR = PJ + "/fusion_output"
os.makedirs(OUT_DIR, exist_ok=True)

COUNTER_PLY = PJ + "/2dgs/output/counter/point_cloud/iteration_30000/point_cloud.ply"
BANANA_PLY  = PJ + "/2dgs/output/banana/point_cloud/iteration_30000/point_cloud.ply"
HAM_PLY     = PJ + "/output/2dgs_hamburger/point_cloud/iteration_7000/point_cloud.ply"
COLA_PLY    = PJ + "/output/2dgs_cola/point_cloud/iteration_30000/point_cloud.ply"

def read_ply_header(path):
    with open(path, "rb") as f:
        hdr, props, n_verts, is_ascii = b"", [], 0, False
        while True:
            line = f.readline()
            hdr += line
            s = line.decode().strip()
            if s.startswith("format") and "ascii" in s:
                is_ascii = True
            if s.startswith("element vertex"):
                n_verts = int(s.split()[-1])
            if s.startswith("property"):
                props.append(s.split()[-1])
            if s == "end_header":
                break
    return props, n_verts, len(hdr), is_ascii

def read_ply_binary(path):
    props, n, hdr_len, is_ascii = read_ply_header(path)
    dt = np.dtype([(p, "<f4") for p in props])
    with open(path, "rb") as f:
        f.seek(hdr_len)
        raw = f.read()
    data = np.frombuffer(raw[:n * dt.itemsize], dtype=dt)
    return data, props

def write_ply_binary(path, data, props):
    n = len(data)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        for p in props:
            f.write(f"property float {p}\n".encode())
        f.write(b"end_header\n")
        out = np.empty(n, dtype=[(p, "<f4") for p in props])
        for p in props:
            out[p] = data[p]
        out.tofile(f)

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])

def transform_gaussians(data, props, R, t, scale=1.0):
    new_data = data.copy()
    pts = np.column_stack([data["x"], data["y"], data["z"]])
    pts = pts * scale
    pts = pts @ R.T + t
    new_data["x"] = pts[:,0]
    new_data["y"] = pts[:,1]
    new_data["z"] = pts[:,2]
    if "nx" in props:
        nrm = np.column_stack([data["nx"], data["ny"], data["nz"]]) @ R.T
        norms = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = np.divide(nrm, norms, where=norms>1e-12, out=np.zeros_like(nrm))
        new_data["nx"], new_data["ny"], new_data["nz"] = nrm[:,0], nrm[:,1], nrm[:,2]
    if "scale_0" in props and scale != 1.0:
        new_data["scale_0"] += np.log(max(scale, 1e-12))
        new_data["scale_1"] += np.log(max(scale, 1e-12))
    if "rot_0" in props:
        R_t = torch.from_numpy(R).float().cuda()
        q = torch.stack([torch.from_numpy(data[p].copy()).float().cuda() for p in ["rot_0","rot_1","rot_2","rot_3"]], dim=-1)
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)
        w,x,y,z = q[:,0], q[:,1], q[:,2], q[:,3]
        Rg = torch.stack([1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y,
                          2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x,
                          2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y], dim=-1).reshape(-1,3,3)
        Rn = R_t[None] @ Rg
        tr = Rn[:,0,0] + Rn[:,1,1] + Rn[:,2,2]
        qw = torch.sqrt(torch.clamp(1+tr, min=0))/2
        new_data["rot_0"] = qw.cpu().numpy().astype(np.float32)
        new_data["rot_1"] = ((Rn[:,2,1] - Rn[:,1,2]) / (4*qw + 1e-12)).cpu().numpy()
        new_data["rot_2"] = ((Rn[:,0,2] - Rn[:,2,0]) / (4*qw + 1e-12)).cpu().numpy()
        new_data["rot_3"] = ((Rn[:,1,0] - Rn[:,0,1]) / (4*qw + 1e-12)).cpu().numpy()
    return new_data

print("="*60)
print("Scene Fusion v2: Counter + Banana + Hamburger + Cola")
print("="*60)

# [1] Counter
print("\n[1] Loading Counter...")
counter_data, counter_props = read_ply_binary(COUNTER_PLY)
print("  %d Gaussians" % len(counter_data))

print("  Before merge - counter scale_0: mean=%.4f" % counter_data["scale_0"].mean())

# [2] Banana
print("\n[2] Loading + transforming Banana...")
banana_data, banana_props = read_ply_binary(BANANA_PLY)
print("  %d Gaussians" % len(banana_data))

pts_b = np.column_stack([banana_data["x"], banana_data["y"], banana_data["z"]])
center_b = pts_b.mean(axis=0)
pts_b_centered = pts_b - center_b
R_banana = np.array([[cos(pi/2), -sin(pi/2), 0],
                     [sin(pi/2), cos(pi/2), 0],
                     [0, 0, 1]], dtype=np.float32)
scale_b = 2.0 / (pts_b_centered[:,0].max() - pts_b_centered[:,0].min() + 1e-12)
t_banana = np.array([-4.0, 1.44, 2.5], dtype=np.float32)
bottom_y_b = (pts_b_centered * scale_b @ R_banana.T)[:,1].min()
t_banana[1] = 1.44 - bottom_y_b
banana_transformed = transform_gaussians(banana_data, banana_props, R_banana, t_banana, scale_b)
print("  scale_b=%.4f t=%s" % (scale_b, t_banana))
print("  banana scale_0 after: mean=%.4f" % banana_transformed["scale_0"].mean())

# [3] Hamburger
print("\n[3] Loading + transforming Hamburger...")
ham_data, ham_props = read_ply_binary(HAM_PLY)
pts_h = np.column_stack([ham_data["x"], ham_data["y"], ham_data["z"]])
ham_max_extent = 0.6149
ham_center = np.array([-0.00531452, 0.04799247, -0.19502233], dtype=np.float32)

pts_h = pts_h * ham_max_extent + ham_center
ham_scale = 0.3 / (pts_h[:,0].max() - pts_h[:,0].min() + 1e-12)
pts_h = pts_h * ham_scale
bottom_y_h = pts_h[:,1].min()
t_ham = np.array([2.0, 1.44 - bottom_y_h, 2.0], dtype=np.float32)
pts_h = pts_h + t_ham

new_ham = np.zeros(len(ham_data), dtype=list(zip(counter_props, ["<f4"]*len(counter_props))))
for p in counter_props:
    if p in ham_props:
        new_ham[p] = ham_data[p]
    else:
        new_ham[p] = np.zeros(len(ham_data), dtype=np.float32)
ham_scale_total = ham_max_extent * ham_scale
if ham_scale_total > 0:
    new_ham["scale_0"] += np.log(ham_scale_total)
    new_ham["scale_1"] += np.log(ham_scale_total)
new_ham["x"] = pts_h[:,0]
new_ham["y"] = pts_h[:,1]
new_ham["z"] = pts_h[:,2]

print("  ham_scale=%.4f t=%s ham_scale_total=%.4f" % (ham_scale, t_ham, ham_scale_total))
print("  ham scale_0 mean: original=%.4f after=%.4f" % (ham_data["scale_0"].mean(), new_ham["scale_0"].mean()))

# [4] Cola
print("\n[4] Loading + transforming Cola...")
cola_data, cola_props = read_ply_binary(COLA_PLY)
pts_c = np.column_stack([cola_data["x"], cola_data["y"], cola_data["z"]])
cola_max_extent = 1.1535
cola_center = np.array([0.27896884, 0.03299905, 0.07457108], dtype=np.float32)

pts_c = pts_c * cola_max_extent + cola_center
cola_scale = 0.25 / (pts_c[:,0].max() - pts_c[:,0].min() + 1e-12)
pts_c = pts_c * cola_scale
bottom_y_c = pts_c[:,1].min()
t_cola = np.array([6.0, 1.44 - bottom_y_c, 1.0], dtype=np.float32)
pts_c = pts_c + t_cola

new_cola = np.zeros(len(cola_data), dtype=list(zip(counter_props, ["<f4"]*len(counter_props))))
for p in counter_props:
    if p in cola_props:
        new_cola[p] = cola_data[p]
    else:
        new_cola[p] = np.zeros(len(cola_data), dtype=np.float32)
cola_scale_total = cola_max_extent * cola_scale
if cola_scale_total > 0:
    new_cola["scale_0"] += np.log(cola_scale_total)
    new_cola["scale_1"] += np.log(cola_scale_total)
new_cola["x"] = pts_c[:,0]
new_cola["y"] = pts_c[:,1]
new_cola["z"] = pts_c[:,2]

print("  cola_scale=%.4f t=%s cola_scale_total=%.4f" % (cola_scale, t_cola, cola_scale_total))
print("  cola scale_0 mean: original=%.4f after=%.4f" % (cola_data["scale_0"].mean(), new_cola["scale_0"].mean()))

# [5] Merge
print("\n[5] Merging...")
print("  counter dtype:", counter_data.dtype)
print("  banana dtype:", banana_transformed.dtype)
print("  new_ham dtype:", new_ham.dtype)
print("  new_cola dtype:", new_cola.dtype)

parts = [counter_data, banana_transformed, new_ham, new_cola]
merged = np.concatenate(parts)
print("  Total: %d = %d + %d + %d + %d" % (len(merged), len(counter_data), len(banana_transformed), len(new_ham), len(new_cola)))

# Verify counter portion is unchanged
n = len(counter_data)
print("\n  Verification - counter portion in merged:")
print("    scale_0: original mean=%.4f merged mean=%.4f" % (counter_data["scale_0"].mean(), merged["scale_0"][:n].mean()))
print("    x range: original [%.3f, %.3f] merged [%.3f, %.3f]" % (
    counter_data["x"].min(), counter_data["x"].max(),
    merged["x"][:n].min(), merged["x"][:n].max()))

# Save
out_ply = OUT_DIR + "/merged_scene.ply"
write_ply_binary(out_ply, merged, counter_props)
print("\n  Saved: %s (%.1f MB)" % (out_ply, os.path.getsize(out_ply)/1024/1024))

# Quick render check
sys.path.insert(0, PJ + "/2dgs/submodules/diff-surfel-rasterization")
sys.path.insert(0, PJ + "/2dgs/submodules/simple-knn")
sys.path.insert(0, PJ + "/2dgs")
from scene import GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera
from argparse import Namespace
from torchvision.utils import save_image

gaussians = GaussianModel(3)
gaussians.load_ply(out_ply)

op = torch.sigmoid(gaussians._opacity)
sc = gaussians.get_scaling
print("\n  Quality check:")
print("    Points: %d" % gaussians._xyz.shape[0])
print("    Opacity: mean=%.4f" % op.mean().item())
print("    Scales: mean=%.6f max=%.6f" % (sc.mean().item(), sc.max().item()))

# Render with first training camera
cameras = json.load(open(PJ + "/2dgs/output/counter/cameras.json"))
c = cameras[0]
R = np.array(c["rotation"], dtype=np.float32)
T = -R @ np.array(c["position"], dtype=np.float32)
W, H = c["width"], c["height"]
fovx = 2 * np.arctan(W / (2 * c["fx"]))
fovy = 2 * np.arctan(H / (2 * c["fy"]))
pipeline = Namespace(compute_cov3D_python=False, debug=False, convert_SHs_python=False, depth_ratio=0.0)
bg = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
dummy = torch.ones((3, H, W), device="cuda")
cam = Camera(0, R, T, fovx, fovy, dummy, None, "test", 0, np.array([0.,0.,0.]), 1.0, "cuda")

with torch.no_grad():
    out = render(cam, gaussians, pipeline, bg)
    img = out["render"]
    print("    Render: mean=%.4f nw=%d" % (img.mean().item(), (img<0.99).sum().item()))
    save_image(img, OUT_DIR + "/merged_v2_debug.png")
    print("    Saved: merged_v2_debug.png")

print("\nDone!")
