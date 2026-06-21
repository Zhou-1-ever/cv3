import sys, os, json, numpy as np, torch
from math import cos, sin, pi
from argparse import Namespace

PJ = "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1"
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

device = "cuda"

gaussians = GaussianModel(3)
gaussians.load_ply(os.path.join(PJ, "fusion_output/merged_scene.ply"))
print("Loaded merged: %d pts" % gaussians._xyz.shape[0])

xyz = gaussians._xyz
print("x:[%.2f,%.2f] y:[%.2f,%.2f] z:[%.2f,%.2f]" % (
    xyz[:,0].min().item(), xyz[:,0].max().item(),
    xyz[:,1].min().item(), xyz[:,1].max().item(),
    xyz[:,2].min().item(), xyz[:,2].max().item()))

pipeline = Namespace(compute_cov3D_python=False, debug=False, convert_SHs_python=False, depth_ratio=0.0)
bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
H, W = 720, 1280
fov = 50 * pi / 180
out_dir = os.path.join(PJ, "fusion_output")

# Render from different angles looking at possible object positions
views = [
    ("banana_eye", torch.tensor([-4.0, 2.0, 3.5], device=device)),
    ("ham_eye", torch.tensor([2.0, 2.0, 3.0], device=device)),
    ("cola_eye", torch.tensor([6.0, 2.0, 2.0], device=device)),
    ("top_down", torch.tensor([0.0, 10.0, 0.0], device=device)),
    ("side", torch.tensor([15.0, 2.0, 0.0], device=device)),
    ("front_wide", torch.tensor([0.0, 2.0, 15.0], device=device)),
]

for name, cam_pos in views:
    center = cam_pos * 0 + torch.tensor([0.0, 1.5, 0.0], device=device)
    center[0] = cam_pos[0] * 0.3  # look slightly towards center
    z_axis = (cam_pos - center) / torch.norm(cam_pos - center)
    up = torch.tensor([0., 1., 0.], device=device)
    x_axis = torch.cross(up, z_axis, dim=0) / torch.norm(torch.cross(up, z_axis, dim=0))
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    T = -R @ cam_pos

    dummy = torch.ones((3, H, W), device=device)
    cam = Camera(0, R.detach().cpu().numpy(), T.detach().cpu().numpy(), fov, fov, dummy, None, name, 0,
                 np.array([0.,0.,0.]), 1.0, device)

    with torch.no_grad():
        out = render(cam, gaussians, pipeline, bg)
        img = out["render"]
        nw = (img < 0.99).sum().item()
        print("%s: mean=%.4f nw=%d" % (name, img.mean().item(), nw))
        save_image(img, os.path.join(out_dir, "fused_%s.png" % name))

print("\nDone! Check fusion_output/fused_*.png")
