import sys, os, json, numpy as np, torch
from math import cos, sin, pi
from argparse import Namespace

PJ = "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1"
OUT = os.path.join(PJ, "fusion_output")

sys.path.insert(0, PJ + "/2dgs/submodules/diff-surfel-rasterization")
sys.path.insert(0, PJ + "/2dgs/submodules/simple-knn")
sys.path.insert(0, PJ + "/2dgs")

from scene import GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera
from torchvision.utils import save_image

device = "cuda"
gaussians = GaussianModel(3)
gaussians.load_ply(os.path.join(PJ, "fusion_output/merged_scene.ply"))
print("Loaded:", gaussians._xyz.shape[0], "points")

pipeline = Namespace(compute_cov3D_python=False, debug=False, convert_SHs_python=False, depth_ratio=0.0)
bg = torch.tensor([1,1,1], dtype=torch.float32, device=device)
H, W = 720, 1280
fov = 50 * pi / 180

views = [
    ("banana_zoom", [-3.5, 1.8, 4.0], [-4.0, 1.6, 2.5]),
    ("ham_zoom",    [2.5, 1.8, 3.5],  [2.0, 1.6, 2.0]),
    ("cola_zoom",   [6.5, 1.8, 2.5],  [6.0, 1.6, 1.0]),
]

for name, cam_pos, lookat in views:
    c = torch.tensor(cam_pos, device=device)
    lt = torch.tensor(lookat, device=device)
    z_axis = (lt - c) / torch.norm(lt - c)
    up = torch.tensor([0.,1.,0.], device=device)
    x_axis = torch.cross(up, z_axis, dim=0) / torch.norm(torch.cross(up, z_axis, dim=0))
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    R = torch.stack([x_axis, y_axis, z_axis], dim=0).cpu().numpy()
    T = -R @ c.cpu().numpy().astype(np.float32)
    dummy = torch.ones((3, H, W), device=device)
    cam = Camera(0, R.astype(np.float32), T.astype(np.float32), fov, fov, dummy, None, name, 0, np.array([0.,0.,0.]), 1.0, device)
    with torch.no_grad():
        img = render(cam, gaussians, pipeline, bg)["render"]
    nw = (img < 0.99).sum().item()
    cy, cx = H//2, W//2
    crop = img[:, cy-150:cy+150, cx-150:cx+150]
    crop_nw = (crop < 0.99).sum().item()
    print(f"{name}: mean={img.mean().item():.4f} nw={nw} crop_nw={crop_nw}/90000")
    save_image(img, os.path.join(OUT, f"zoom_{name}.png"))

print("Done")
