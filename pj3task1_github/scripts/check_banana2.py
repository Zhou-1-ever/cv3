import sys, os
sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1/2dgs/submodules/diff-surfel-rasterization")
sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1/2dgs/submodules/simple-knn")
sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1/2dgs")
from scene import GaussianModel
import torch, numpy as np

PJ = "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/pj3task1"

# Check merged scene
merged = GaussianModel(3)
merged.load_ply(os.path.join(PJ, "fusion_output/merged_scene.ply"))

xyz = merged._xyz.detach().cpu()
features = merged.get_features.detach().cpu()
opacity = torch.sigmoid(merged._opacity).detach().cpu()
scales = merged.get_scaling.detach().cpu()

n = 557038  # counter points

# Check banana cluster (x between -5 and -3)
banana_mask = (xyz[:, 0] > -5) & (xyz[:, 0] < -3)
banana_xyz = xyz[banana_mask]
banana_feat = features[banana_mask]
banana_op = opacity[banana_mask]
banana_sc = scales[banana_mask]

print(f"Banana points: {len(banana_xyz)}")
print(f"Banana DC features (f_dc_0/1/2): mean={banana_feat[:, 0, 0].mean().item():.4f} {banana_feat[:, 0, 1].mean().item():.4f} {banana_feat[:, 0, 2].mean().item():.4f}")
print(f"Banana opacity: mean={banana_op.mean().item():.4f}")
print(f"Banana scales: mean={banana_sc.mean().item():.6f} max={banana_sc.max().item():.6f}")

# Compare with counter
counter_mask = xyz[:n]
counter_feat = features[:n]
print(f"\nCounter DC features: mean={counter_feat[:, 0, 0].mean().item():.4f} {counter_feat[:, 0, 1].mean().item():.4f} {counter_feat[:, 0, 2].mean().item():.4f}")

# Check ham cluster (x between 1 and 3)
ham_mask = (xyz[:, 0] > 1) & (xyz[:, 0] < 3)
ham_xyz = xyz[ham_mask]
ham_feat = features[ham_mask]
ham_op = opacity[ham_mask]
print(f"\nHamburger points: {len(ham_xyz)}")
print(f"Ham DC features: mean={ham_feat[:, 0, 0].mean().item():.4f} {ham_feat[:, 0, 1].mean().item():.4f} {ham_feat[:, 0, 2].mean().item():.4f}")
print(f"Ham opacity: mean={ham_op.mean().item():.4f}")

# Check the original banana PLY directly for comparison
print("\n\n=== Original Banana PLY ===")
orig_banana = GaussianModel(3)
orig_banana.load_ply(os.path.join(PJ, "2dgs/output/banana/point_cloud/iteration_30000/point_cloud.ply"))
orig_xyz = orig_banana._xyz.detach().cpu()
orig_feat = orig_banana.get_features.detach().cpu()
orig_op = torch.sigmoid(orig_banana._opacity).detach().cpu()
print(f"Points: {len(orig_xyz)}")
print(f"XYZ mean: {orig_xyz.mean(dim=0)}")
print(f"DC features mean: {orig_feat[:, 0, 0].mean().item():.4f} {orig_feat[:, 0, 1].mean().item():.4f} {orig_feat[:, 0, 2].mean().item():.4f}")
print(f"Opacity mean: {orig_op.mean().item():.4f}")

# Check original banana alone renders properly by looking at SH -> color
# The DC component is essentially the base color (SH degree 0)
dc_color = orig_feat[:, 0, :]  # (N, 3)
# Normalize via SH: color = SH(DC) * 0.2820947917 + higher bands
# For degree 0: color = 0.2820947917 * f_dc + 0.5
sh_c0 = 0.2820947917
pred_color = dc_color * sh_c0 + 0.5
print(f"\nOriginal banana predicted color: R={pred_color[:, 0].mean():.4f} G={pred_color[:, 1].mean():.4f} B={pred_color[:, 2].mean():.4f}")

# Same for merged banana
dc_color_b = banana_feat[:, 0, :]
pred_color_b = dc_color_b * sh_c0 + 0.5
print(f"Merged banana predicted color: R={pred_color_b[:, 0].mean():.4f} G={pred_color_b[:, 1].mean():.4f} B={pred_color_b[:, 2].mean():.4f}")

print("\nDone")
