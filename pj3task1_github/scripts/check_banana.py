import sys, os
sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1/2dgs/submodules/diff-surfel-rasterization")
sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1/2dgs/submodules/simple-knn")
sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1/2dgs")
from scene import GaussianModel
import torch, numpy as np

PJ = "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1"

merged = GaussianModel(3)
merged.load_ply(os.path.join(PJ, "fusion_output/merged_scene.ply"))

xyz = merged._xyz.detach().cpu()
n = 557038

obj_xyz = xyz[n:]
obj_np = obj_xyz.numpy()

# Scan for banana cluster
banana_x = obj_np[(obj_np[:, 0] > -7) & (obj_np[:, 0] < -3)]
print(f"Points with x in (-7, -3): {len(banana_x)}")
if len(banana_x) > 0:
    print(f"  y range: [{banana_x[:, 1].min():.3f}, {banana_x[:, 1].max():.3f}]")
    print(f"  z range: [{banana_x[:, 2].min():.3f}, {banana_x[:, 2].max():.3f}]")

# Scan for clusters across x
for x_start in range(-7, 8):
    mask = (obj_np[:, 0] >= x_start) & (obj_np[:, 0] < x_start + 1)
    count = mask.sum()
    if count > 0:
        subset = obj_np[mask]
        print(f"x=[{x_start},{x_start+1}): {count:5d} pts, y=[{subset[:,1].min():.2f},{subset[:,1].max():.2f}], z=[{subset[:,2].min():.2f},{subset[:,2].max():.2f}]")

print("\nDone")
