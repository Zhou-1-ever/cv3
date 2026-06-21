import sys, os
PJ = "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1"
OUT = os.path.join(PJ, "fusion_output")
sys.path.insert(0, PJ + "/2dgs/submodules/diff-surfel-rasterization")
sys.path.insert(0, PJ + "/2dgs/submodules/simple-knn")
sys.path.insert(0, PJ + "/2dgs")
from scene import GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera
from argparse import Namespace
import torch, numpy as np
from math import cos, sin, pi
from torchvision.utils import save_image

device = "cuda"
gaussians = GaussianModel(3)
gaussians.load_ply(os.path.join(PJ, "fusion_output/merged_scene.ply"))
print("Loaded:", gaussians._xyz.shape[0], "points")

# Check object presence by analyzing the scene
xyz = gaussians._xyz.detach().cpu().numpy()
# Counter is points 0:557038, objects are 557038:
n_counter = 557038
obj_xyz = xyz[n_counter:]
print(f"Objects: {len(obj_xyz)} points")
print(f"Range: x[{obj_xyz[:,0].min():.2f},{obj_xyz[:,0].max():.2f}]")
print(f"       y[{obj_xyz[:,1].min():.2f},{obj_xyz[:,1].max():.2f}]")
print(f"       z[{obj_xyz[:,2].min():.2f},{obj_xyz[:,2].max():.2f}]")

# Get colors for object points to verify they have color
op = torch.sigmoid(gaussians._opacity)
obj_op = op[n_counter:].detach().cpu().numpy()
print(f"Object opacity: mean={obj_op.mean():.4f}")

# Render from a good angle
pipeline = Namespace(compute_cov3D_python=False, debug=False, convert_SHs_python=False, depth_ratio=0.0)
bg = torch.tensor([1,1,1], dtype=torch.float32, device=device)
H, W = 720, 1280
fov = 50 * pi / 180

center = torch.tensor([0.0, 1.6, 2.0], device=device)
views = [
    ("overview",    [0.0, 4.0, 12.0]),
    ("front_close", [0.0, 2.5, 6.0]),
    ("side",        [10.0, 2.5, 2.0]),
    ("banana_view", [-4.0, 2.0, 4.0]),
    ("ham_view",    [2.0, 2.0, 3.5]),
    ("cola_view",   [6.0, 2.0, 3.0]),
]

for name, cam_pos in views:
    c = torch.tensor(cam_pos, device=device)
    z_axis = (center - c) / torch.norm(center - c)
    up = torch.tensor([0.,1.,0.], device=device)
    x_axis = torch.cross(up, z_axis) / torch.norm(torch.cross(up, z_axis))
    y_axis = torch.cross(z_axis, x_axis)
    R = torch.stack([x_axis, y_axis, z_axis], dim=0).cpu().numpy()
    T = -R @ c.cpu().numpy().astype(np.float32)
    dummy = torch.ones((3, H, W), device=device)
    cam = Camera(0, R.astype(np.float32), T.astype(np.float32), fov, fov, dummy, None, name, 0, np.array([0.,0.,0.]), 1.0, device)
    with torch.no_grad():
        img = render(cam, gaussians, pipeline, bg)["render"]
    nw = (img < 0.99).sum().item()
    print(f"{name}: mean={img.mean().item():.4f} nw={nw} ({nw/(H*W)*100:.1f}%)")
    save_image(img, os.path.join(OUT, f"quality_{name}.png"))
print("Done")
