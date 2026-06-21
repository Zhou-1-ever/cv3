# Task1: Multi-Source 3D Asset Generation & Real Scene Fusion

**基于 2DGS 与 AIGC 的多源资产生成与真实场景融合**

---

## 📋 Project Overview

This project fulfills a full-pipeline 3D vision challenge: reconstruct real-world scenes, generate virtual assets via AIGC, and fuse them into a unified 3D scene for rendering.

### Pipeline

```
┌─────────────────────────────────────────────────────┐
│  Object A (Multi-view → 2DGS)    ───────┐           │
│  Object B (Text → DreamFusion → 2DGS)  ───┤         │
│  Object C (Single Image → Magic123 → 2DGS) ─┤      │
│  Background (Mip-NeRF 360 → 2DGS)   ──────┤         │
│                                            ▼        │
│                                   ┌────────────┐    │
│                                   │   Fusion   │    │
│                                   │  (Unified  │    │
│                                   │   2DGS PLY)│    │
│                                   └─────┬──────┘    │
│                                         ▼           │
│                                ┌─────────────────┐  │
│                                │ Multi-view Video │  │
│                                └─────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## 🏗️ Project Structure

```
pj3task1/
├── 2dgs/                          # 2D Gaussian Splatting codebase
│   └── output/
│       ├── counter/               # Background scene (COLMAP + 2DGS)
│       │   ├── point_cloud/iteration_30000/point_cloud.ply
│       │   └── cameras.json
│       └── banana/                # Object A: Real multi-view reconstruction
│           ├── point_cloud/iteration_30000/point_cloud.ply
│           └── cameras.json
├── output/
│   ├── 2dgs_hamburger/            # Object B: Text-to-3D (DreamFusion → 2DGS)
│   └── 2dgs_cola/                 # Object C: Single image to 3D (Magic123 → 2DGS)
├── fusion_output/
│   ├── merged_scene.ply           # Final fused 3D scene (599,912 Gaussians)
│   ├── scene_trajectory_v2.mp4    # Multi-view roaming video
│   ├── frames_v2/                 # Individual video frames (240 frames, 720p)
│   ├── quality_*.png              # Quality check renders
│   └── zoom_*.png                 # Object close-up renders
├── rendered_data/
│   └── cola/                      # Cola multi-view renders for 2DGS training
├── threestudio-outputs/           # Threestudio checkpoints
│   └── magic123-*-sd/
├── scripts/
│   ├── fusion_scene.py            # Scene fusion script (v1)
│   ├── fusion_scene2.py           # Scene fusion script (v2, fixed)
│   ├── render_trajectory.py       # Multi-view trajectory renderer
│   ├── render_multiview.py        # Mesh → multi-view renderer (nvdiffrast)
│   ├── render_fused_views.py      # Fused scene diagnostic renderer
│   ├── prepare_cola.py            # Cola asset extraction from checkpoint
│   ├── check_fusion.py            # Fusion quality diagnostic
│   ├── zoom_check.py              # Object close-up check
│   └── check_banana.py            # Banana position/quality analysis
└── fusion_output/
```

## 🎯 Three Objects

| Object | Method | Source | Gaussians | Training PSNR |
|:------:|:------:|:------:|:---------:|:-------------:|
| **A: Banana** | Multi-view → COLMAP → 2DGS | Phone video (9 views) | 30,763 | 41.47 dB |
| **B: Hamburger** | Text → DreamFusion → 2DGS | Prompt: "a delicious hamburger" | 7,444 | 15.47 dB |
| **C: Cola** | Single image → Magic123 → 2DGS | Phone photo of cola bottle | 3,195 | 16.57 dB |

## 🏪 Background Scene

| Scene | Method | Source | Gaussians | Training PSNR |
|:-----:|:------:|:------:|:---------:|:-------------:|
| **Counter** | COLMAP + 2DGS | Mip-NeRF 360 dataset | 557,038 | 28.65 dB |

## 🔧 Technical Details

### Data Preparation

- **Object A**: 9 photos of a banana on a white background → COLMAP for camera poses → 2DGS training (30K iterations)
- **Object B**: Text prompt → DreamFusion (threestudio, 10K iters) → Export textured mesh → nvdiffrast multi-view rendering (100 views) → 2DGS training (7K iterations)
- **Object C**: Single photo (background removed) → Magic123 Coarse → Magic123 Refine (10K iters each) → Isosurface mesh extraction → nvdiffrast multi-view rendering (120 views) → 2DGS training (30K iterations)
- **Background**: Mip-NeRF 360 "Counter" scene (240 training views) → 2DGS training (30K iterations)

### Scene Fusion

All objects are unified into the 2DGS representation (Gaussian surfels). The fusion process:
1. Load each object's PLY file with Gaussian parameters (position, rotation quaternion, scale, opacity, SH coefficients)
2. Apply coordinate transforms (scale, rotation, translation) to place objects on the counter surface in the background scene's coordinate system
3. Concatenate all Gaussian arrays into a single merged PLY
4. Render with the 2DGS rasterization pipeline

**Common representation**: All assets → 2DGS PLY → single render pipeline

### Bug Fix: Scale Corruption

An indentation bug in `fusion_scene.py` caused the scale log adjustment `+= np.log(scale)` to execute inside a `for p in properties` loop, applying it 16× instead of once. This made object scales catastrophically large (mean=3456, max=198M in log → exp space). Fixed in `fusion_scene2.py`.

### Final Scene
- **Total Gaussians**: 599,912
- **Scene extents**: x[-13.16, 12.83], y[-6.53, 12.76], z[-9.96, 16.69]
- **Rendered video**: 240 frames @ 30fps, 720×1280, 2× orbit with height oscillation
- **Scale check**: mean=0.028 (now normal), opacity check: mean=0.637

## 📊 Quality Comparison

| Aspect | Object A (Multi-view) | Object B (Text-to-3D) | Object C (Single-image) |
|:------:|:---------------------:|:---------------------:|:-----------------------:|
| **Geometry** | High accuracy | Moderate (hallucinated) | Moderate (biased by prior) |
| **Texture** | Photo-realistic | StyleGAN-like | Input-consistent |
| **Training PSNR** | 41.47 dB | 15.47 dB | 16.57 dB |
| **Gaussians** | 30,763 | 7,444 | 3,195 |
| **Training time** | ~97 min | ~83 min | ~113 min |
| **Effort** | Phone video + COLMAP | Just a text prompt | One photo + bg removal |

## 🚀 Usage



### Fusion
```bash
python scripts/fusion_scene2.py
```

### Render trajectory video
```bash
python scripts/render_trajectory.py
```

### Render diagnostic views
```bash
python scripts/render_fused_views.py
```

