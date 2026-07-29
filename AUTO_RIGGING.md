# Phase 5 Research Report: Auto-Rigging

> Single arbitrary image → rigged, animatable 3D mesh with skeleton, skinning weights, and base animation support.

---

## Table of Contents

1. [Single-Image → 3D Mesh Backends](#1-single-image--3d-mesh-backends)
2. [Category / Part Inference](#2-category--part-inference)
3. [Skeleton Prediction](#3-skeleton-prediction)
4. [Skinning Weights](#4-skinning-weights)
5. [Animation Retargeting](#5-animation-retargeting)
6. [Related Work Sweep (2023–2026)](#6-related-work-sweep-2023-2026)
7. [Evaluation & Benchmarking](#7-evaluation--benchmarking)
8. [Edge Cases](#8-edge-cases)
9. [Codebase Integration](#9-codebase-integration)
10. [Summary of Recommendations](#10-summary-of-recommendations)

---

## 1. Single-Image → 3D Mesh Backends

### 1.1 TRELLIS (v1)

| Property | Detail |
|----------|--------|
| **Paper** | "Structured 3D Latents for Scalable and Versatile 3D Generation" (CVPR 2025 Spotlight) — arXiv:2412.01506 |
| **Code** | https://github.com/microsoft/TRELLIS |
| **License** | **MIT License** (code and models). Submodules (diffoctreerast, Flexicubes) have different licenses — see issue #41. The renderers for Gaussians/radiance fields originally used non-commercial components, but community PRs have replaced these with MIT-compatible alternatives. Primary source: https://github.com/microsoft/TRELLIS/blob/main/LICENSE |
| **Output** | 3D Gaussians, Radiance Fields, Mesh (via Flexicubes/Marching Cubes). Exports GLB with baked texture. |
| **Textures** | Yes — vertex colors and textures via UV baking from Gaussians (texture_size up to 1024 in v1, 4096 in v2). |
| **Active** | Yes — last push Nov 2025 (v1), very active community (13K stars, 254 open issues as of mid-2025). |
| **Python API** | `TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")` — full Python API available. Also CLI and Gradio demo. |
| **VRAM** | **Minimum 16GB** (official); users report working on RTX 3090/4090 (24GB) and RTX 3060 12GB with optimizations. Tested on A100, A6000. Source: GitHub README and issue #14. |
| **Speed** | <10s on A100 for image→textured mesh. |
| **Parameters** | 1.2B (image-large model). |

### 1.2 TRELLIS.2 (v2)

| Property | Detail |
|----------|--------|
| **Paper** | "Native and Compact Structured Latents for 3D Generation" — tech report 2025, arXiv not yet assigned at time of research. |
| **Code** | https://github.com/microsoft/TRELLIS.2 |
| **License** | **MIT License** — same structure as v1. |
| **Output** | Mesh with PBR materials. Voxel resolution 512³–1536³. O-Voxel representation. |
| **Textures** | Yes — full PBR (metallic, roughness, normal maps). Texture size up to 4096. |
| **Active** | Yes — released ~Dec 2025, actively maintained (3.7K stars). |
| **Python API** | `Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")`. |
| **VRAM** | **Minimum 24GB**. Verified on A100 and H100. |
| **Parameters** | 4B. |

### 1.3 TripoSR

| Property | Detail |
|----------|--------|
| **Paper** | "TripoSR: Fast 3D Object Reconstruction from a Single Image" — arXiv:2403.02151 |
| **Code** | https://github.com/VAST-AI-Research/TripoSR |
| **License** | **MIT License** — both code and pretrained model. Primary source: https://github.com/VAST-AI-Research/TripoSR (license badge) |
| **Output** | Mesh via NeRF → Marching Cubes extraction. Vertex colors available. |
| **Textures** | Yes — vertex colors baked from NeRF. Not UV-textured. |
| **Active** | Moderate — last push June 2026, 6.7K stars, 103 open issues. By Tripo AI + Stability AI. |
| **Python API** | Yes — `python run.py` with image input. Also Gradio demo. |
| **VRAM** | **~3.5GB** for inference. Very lightweight. A100: <0.5s. |
| **Limitations** | Fixed topology (triplane-NeRF), resolution-limited, sometimes lacks fine detail. |

### 1.4 Hunyuan3D 2.0 / 2.1

| Property | Detail |
|----------|--------|
| **Paper** | "Hunyuan3D 2.0: Scaling Diffusion for High Resolution Textured 3D Generation" — tech report. |
| **Code** | https://github.com/Tencent-Hunyuan/Hunyuan3D-2 (v2.0); https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 (v2.1, PBR) |
| **License** | **TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT** (NOT MIT). Non-commercial restriction: if MAUs >1M, must request license. Restriction on improving other AI models. Primary source: https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE |
| **Output** | Textured mesh (OBJ/GLB). Two-stage: shape generation (DiT) + texture synthesis (Paint). |
| **Textures** | Yes — UV texture maps. v2.1 adds PBR (metallic, roughness, normal). |
| **Active** | Very active — 14.3K stars, frequent releases (Turbo, Mini, MV variants), 40 contributors. |
| **Python API** | `Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')` — diffusers-like API. Also API server (api_server.py) and Gradio. |
| **VRAM** | **Shape only: ~6GB** (with low_vram_mode), **Shape+Texture: ~16GB**. Mini model: ~8GB. Turbo model: ~10GB. Source: GitHub README and issue #16. |
| **Parameters** | 1.1B (DiT-v2-0), 3.0B (DiT-v2-1), 0.6B (Mini). Paint: 1.3B. |

### 1.5 DreamGaussian

| Property | Detail |
|----------|--------|
| **Paper** | "DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation" — ICLR 2024 Oral |
| **Code** | https://github.com/dreamgaussian/dreamgaussian |
| **License** | **MIT License** — code. Primary source: https://github.com/dreamgaussian/dreamgaussian (license badge) |
| **Output** | 3D Gaussians → textured mesh (UV unwrapped) via custom extraction algorithm. |
| **Textures** | Yes — UV texture maps, back-projected from rendered views. |
| **Active** | Low — last push Jan 2024. 4.3K stars. No longer actively maintained (archived effectively). |
| **Python API** | CLI scripts (`main.py`, `main2.py`) with config files. No HuggingFace pipeline. |
| **VRAM** | **<8GB** — tested on V100 16GB, works on consumer GPUs. |
| **Speed** | ~2 min for image→textured mesh. |

### 1.6 Zero123++

| Property | Detail |
|----------|--------|
| **Paper** | "Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model" — arXiv:2310.15110 |
| **Code** | https://github.com/SUDO-AI-3D/zero123plus |
| **License** | **Code: Apache 2.0. Model weights: CC-BY-NC 4.0** (non-commercial). Primary source: https://github.com/SUDO-AI-3D/zero123plus |
| **Output** | Multi-view images (6 canonical views), NOT direct 3D. Must be combined with another reconstruction method. |
| **Active** | Low — last commits 2024. |
| **Python API** | Yes — diffusers-based pipeline. Training code not released. |

### 1.7 Stable Zero123

| Property | Detail |
|----------|--------|
| **Paper** | Stability AI blog (Dec 2023). |
| **Code** | Via threestudio integration. |
| **License** | **Stable Zero123: Non-commercial (SAI-NC Community License). Stable Zero123C: Community License** (requires active Stability AI membership for commercial use). Primary source: https://huggingface.co/stabilityai/stable-zero123 |
| **Output** | View-conditioned images for SDS-based 3D optimization. |

### 1.8 Summary Matrix

| Method | License | Output Type | Texture | VRAM | Speed | Python API | Active |
|--------|---------|-------------|---------|------|-------|-----------|--------|
| **TRELLIS** | MIT | Mesh+GS+RF | UV texture | 16GB | <10s | Yes | Very active |
| **TRELLIS.2** | MIT | Mesh+PBR | PBR 4K | 24GB | <10s | Yes | Very active |
| **TripoSR** | MIT | Mesh(vertex color) | Vertex color | ~3.5GB | <0.5s | Yes | Moderate |
| **Hunyuan3D-2** | Tencent Community | Mesh+UV | UV texture | 6–16GB | ~10s | Yes | Very active |
| **DreamGaussian** | MIT | Mesh+UV | UV texture | <8GB | ~2min | CLI only | Inactive |
| **Zero123++** | Apache+CC-BY-NC | Multi-view images | N/A | ~8GB | ~2s | Yes | Low |

---

## 2. Category / Part Inference

### 2.1 Methods for Inferring Kinematic Structure

| Method | Year | Approach | Category Scope | Code Available? |
|--------|------|----------|---------------|-----------------|
| **RigNet** | 2020 | Joint detection via vertex displacement clustering + connectivity prediction via BoneNet | Category-agnostic (trained on 2.7K diverse characters) | Yes, GPL v3 |
| **TARig** | 2023 | Template-aware adaptation of RigNet with template joint module (TJM) | Humanoid only | No official code |
| **MagicArticulate** | CVPR 2025 | Auto-regressive transformer for skeleton + functional diffusion for skinning | Category-agnostic (Articulation-XL: 59K models) | "Coming soon" |
| **RigAnything** | SIGGRAPH TOG 2025 | Auto-regressive transformer with BFS ordering + diffusion for continuous joint positions | Template-free, arbitrary categories | Yes, non-commercial |
| **Puppeteer** | NeurIPS 2025 Spotlight | Auto-regressive + attention skinning + differentiable animation pipeline | Category-agnostic (Articulation-XL2.0: 59.4K models) | Code page exists |
| **Make-It-Animatable** | CVPR 2025 Highlight | Particle-based shape autoencoder; <1s per character | Humanoid-focused | Yes |

### 2.2 Category Classification from Single Image

- **No single standard taxonomy** exists for rigging purposes.
- Common categories in rigging datasets: **humanoid, quadruped, bird, fish, insect, robot, fictional** (source: RigNet dataset taxonomy).
- MagicPony / 3D-Fauna use implicit category-specific skeletons (horse, bird, etc. — define bone count and connectivity manually per category).
- **3D-Fauna** (CVPR 2024) introduces a **Semantic Bank of Skinned Models (SBSM)** that learns a shared latent space across 100+ quadruped species, automatically discovering base shapes. Source: https://kyleleey.github.io/3DFauna/
- For a general pipeline, you likely need to predict category or at least articulation type. This could be done via:
  - An image classifier trained on categories (e.g., CLIP-based zero-shot)
  - Geometric analysis of the reconstructed mesh (e.g., aspect ratio, symmetry, part volumes)
  - Template matching (try multiple articulation templates and pick the best-fitting)

### 2.3 Template-Based vs. Learned Prediction

| Approach | Pros | Cons |
|----------|------|------|
| **Template-based** (TARig, Pinocchio) | Fast, predictable skeleton structure, works well for humanoids | Limited to specific categories; fails on novel topologies |
| **Learned prediction** (RigNet, MagicArticulate, RigAnything) | Generalizes across categories, handles novel topologies | Requires large datasets, training compute, may produce artistically questionable skeletons |

**Verdict:** Template-based is a pragmatic baseline for well-understood categories (humanoids, quadrupeds). For arbitrary categories, learned prediction (specifically auto-regressive transformers as in MagicArticulate/RigAnything) is the current SOTA.

---

## 3. Skeleton Prediction

### 3.1 RigNet Deep-Dive

**Architecture** (from paper and code, source: https://zhan-xu.github.io/rig-net/):

1. **Joint Detection Stage:**
   - Input: Mesh vertices + normals (1K–5K vertices), surface and geodesic edge graphs
   - Feature extraction: GMEdgeConv (graph convolution on geodesic neighborhoods)
   - Attention module: learns per-vertex importance weights for clustering
   - Vertex displacement: displaces vertices toward joints
   - Mean-shift clustering: extracts joint positions from displaced vertices
   - Loss: Chamfer distance between displaced vertices and GT joints + cluster assignment loss

2. **Connectivity Prediction Stage (BoneNet):**
   - Input: detected joints + shape features
   - Encodes pairwise bone descriptors (length, direction, topology)
   - Binary classifier per joint pair: predicts if they should be connected
   - RootNet: predicts which joint is the skeleton root

3. **Skinning Prediction Stage (SkinNet):**
   - Input: mesh vertices + joints + volumetric geodesic distances
   - 5 nearest bones per vertex based on volumetric geodesic distance
   - MLP predicts skinning weights

**Inputs:** Mesh (OBJ/PLY), optionally pre-computed features
**Outputs:** Joint positions, bone connectivity, per-vertex skinning weights
**License:** **GPL v3** (or commercial license). Primary source: https://github.com/zhan-xu/RigNet
**Dataset:** ModelsResource-RigNetv1 (2,703 rigged characters from https://models-resource.com) — FBX format, available via Google Drive link in repo.
**Limitations:** ~120s per shape; requires rest-pose input; struggles with non-standard poses; bandwidth tuning needed.

### 3.2 TARig

- Extension of RigNet specifically for **humanoid characters**.
- Introduces **template joint module (TJM)**: predicts template joint positions as convex combinations of mesh vertices (replacing mean-shift).
- Faster: ~0.6 min vs RigNet's ~10 min.
- **Limitations:** Humanoid-only; template restricts topology; code not officially released (reimplemented by others based on RigNet codebase).
- **Source:** https://doi.org/10.1016/j.cag.2023.05.018

### 3.3 2025–2026 Papers

| Paper | Venue | Year | Key Idea |
|-------|-------|------|----------|
| **RigAnything** | SIGGRAPH TOG | 2025 | Auto-regressive transformer + diffusion joints + BFS ordering; template-free; <2s per shape |
| **MagicArticulate** | CVPR | 2025 | Auto-regressive skeleton + functional diffusion skinning; Articulation-XL dataset |
| **Puppeteer** | NeurIPS Spotlight | 2025 | Auto-regressive + attention skinning + diff-opt animation; 59.4K models |
| **Make-It-Animatable** | CVPR Highlight | 2025 | Particle-based autoencoder; <1s; humanoid focus |
| **UniRig** | CVPR? | 2025 | Unified rigging framework; outperforms RigNet/TARig on J2J metric |
| **AnyTop** | SIGGRAPH | 2025 | Motion diffusion for arbitrary skeletons (not rigging, but complementary) |
| **RigMo** | arXiv | Jan 2026 | Unifies rig + motion learning; no human annotation needed |

### 3.4 Template Retrieval + Deformation Baseline

A simpler approach than full learned prediction:

1. **Template retrieval:** Given a mesh, find the closest template skeleton from a database (by shape descriptors, category label, or geometric features).
2. **Deformation:** Deform template joints to fit the target mesh using:
   - Laplacian deformation / ARAP (as-rigid-as-possible)
   - Skeleton fitting via iterative closest point (ICP) on target shape
   - Pinocchio-style skeleton embedding

**Existing work:** Pinocchio (Baran & Popovic, SIGGRAPH 2007) does exactly this: embeds a user-provided skeleton into a mesh by optimizing bone positions inside the volume. Code: https://github.com/pear0day/Pinocchio (not rigging but volume-based skeleton fitting).

**Limitations:** Requires a database of templates; may not handle unusual topologies; template bias.

### 3.5 Unpredictable Bone Counts

**How SOTA handles it:**

- **RigNet:** Mean-shift clustering bandwidth controls number of joints; bandwidth must be tuned per shape.
- **MagicArticulate / RigAnything / Puppeteer:** Auto-regressive transformers naturally generate variable-length sequences. A special `<EOS>` token ends generation.
- **RigAnything:** BFS ordering + diffusion-based joint prediction + learned stopping criterion.

**Recommendation:** Auto-regressive transformer approach is the most principled for variable bone counts.

---

## 4. Skinning Weights

### 4.1 Geodesic-Distance-Based Initialization

**Bounded Biharmonic Weights (BBW):**

- Gold standard for automatic skinning weight computation.
- Minimizes Laplacian energy subject to bound constraints (non-negativity, partition of unity).
- Input: mesh + skeleton (joints/bones).
- Output: per-vertex skinning weights.
- **Source:** https://igl.ethz.ch/projects/bbw/
- **Code:** Available in libigl (C++ with Python bindings):
  - `igl::bbw()` — C++ implementation
  - Python: `pyigl` tutorial 403
  - Re-implementation in Python via MOSEK or convex duality (Solomon & Stein, 2025).

**Volumetric Geodesic Distance:**

- Used by RigNet as a strong prior for skinning. For each vertex, compute distance to each bone through the volume, then use inverse distance as the weight initialization.
- **Advantage:** Produces smooth, locally supported weights that respect the shape geometry.
- **Available code:** RigNet's SkinNet uses pre-computed volumetric geodesic distances.

### 4.2 Learned Refinement

- **RigNet SkinNet:** MLP that refines geodesic-based weights.
- **MagicArticulate:** Functional diffusion process that learns residual from volumetric geodesic prior.
- **RigAnything:** Pairwise skinning weight computation using transformer features.
- **Puppeteer:** Attention-based architecture with topology-aware joint attention.
- **Neural Blend Shapes** (Li et al., 2021): Predicts neural blend shapes in addition to skinning weights for better deformation.

### 4.3 Available Code/Libraries

| Library | Function | License | Language |
|---------|----------|---------|----------|
| **libigl** | BBW, LBS, boundary conditions | GPL v3 / Commercial | C++, Python bindings |
| **gptoolbox** | Geodesic distances, skinning utilities | MIT | MATLAB |
| **trimesh** | Mesh processing, geodesic distances | MIT | Python |
| **RigNet SkinNet** | Learned skinning refinement | GPL v3 | Python/PyTorch |
| **PolyFit (SCP)** | Volumetric geodesic distance | MIT | Python |

### 4.4 Evaluation Metrics for Skinning

From RigNet (source: https://ar5iv.labs.arxiv.org/html/2005.00559):

| Metric | Description |
|--------|-------------|
| **Precision** | Fraction of influential bones (weight > threshold) in predicted skinning that match reference |
| **Recall** | Fraction of reference influential bones found in predicted skinning |
| **avg-L1** | Average L1 norm of difference between predicted and reference weight vectors |
| **avg-dist** | Average Euclidean distance between vertices deformed with predicted vs. reference weights under identical bone transformations |
| **max-dist** | Maximum per-vertex distance under same deformation |

---

## 5. Animation Retargeting

### 5.1 Retargeting onto Arbitrary Inferred Skeletons

**Challenge:** Inferred skeletons have unpredictable topology, joint names, and bone hierarchies. Canonical humanoid retargeting methods (e.g., Mixamo's) fail.

**SOTA approaches:**

| Method | Venue | Key Idea | Limitation |
|--------|-------|----------|------------|
| **MoMa** | CVIU 2024 | Masked pose modeling; super-skeleton representation; transformer-based | Moderate generalization |
| **HuMoT** | TOG 2023 | Topology-agnostic transformer autoencoder; conditioned on skeleton templates | Requires skeleton templates |
| **Skeleton-Aware Masked Pose** | ECCV 2024 Workshops | Random masking in space+time; super-skeleton + learnable tokens | Requires all skeletons known at training |
| **AnyTop** | SIGGRAPH 2025 | Diffusion model + Skeletal-Temporal Transformer; textual joint descriptions | Motion generation, not strict retargeting |
| **STaR** | ICCV 2025 | Seamless spatiotemporal retargeting with penetration constraints | Code partially released |
| **SATA** | ICML 2026 | Semantic-aware, topology-agnostic motion encoding; zero-shot cross-species retargeting | Training code not yet released |

### 5.2 Simple Retargeting Recipe

For a practical pipeline with 2–3 canonical animation clips:

1. **Skeleton correspondence** via:
   - Joint name matching (if names are available)
   - Geometric heuristics (e.g., which joint is most upward = head, lowest mid = pelvis)
   - Semantic correspondence via AnyTop's DIFT features or learned embeddings

2. **Joint mapping** using:
   - Hand-crafted mapping rules for common skeleton types (biped, quadruped)
   - Learned correspondence (AnyTop, SATA)
   - Simple fallback: match by topological distance from root

3. **Motion transfer**:
   - Copy rotation data for matched joints
   - Interpolate missing joints (e.g., if target has more spine joints)
   - IK-based foot planting for ground contact

4. **Available codebases**:
   - **AnyTop:** https://github.com/anytop2025/anytop (SIGGRAPH 2025)
   - **STaR:** https://github.com/XiaohangYang829/STaR (ICCV 2025)
   - **MeshRet:** https://github.com/abcyzj/meshret (NeurIPS 2024, Mixamo-only skeletons)
   - **Blender retargeting**: built-in `bpy` API

---

## 6. Related Work Sweep (2023–2026)

### 6.1 Direct Rigging Methods

| Method | Year | Input | Output | License | Code Available? |
|--------|------|-------|--------|---------|-----------------|
| **RigNet** | 2020 | Mesh | Skeleton + skinning | GPL v3 / Commercial | Yes |
| **TARig** | 2023 | Mesh (humanoid) | Skeleton + skinning | UNVERIFIED (no official code) | No (reimpl exists) |
| **Neural Blend Shapes** | 2021 | Mesh (T-pose) | Rig + blend shapes | UNVERIFIED | Yes |
| **MagicArticulate** | 2025 | Mesh | Skeleton + skinning | UNVERIFIED (code not yet released) | Code "coming soon" |
| **RigAnything** | 2025 | Mesh | Skeleton + skinning | Adobe Research (non-commercial) | Yes |
| **Puppeteer** | 2025 | Mesh | Skeleton + skinning + animation | CC BY-NC 4.0 | Code page exists |
| **Make-It-Animatable** | 2025 | Mesh/3DGS | Skinning + bones + pose | UNVERIFIED | Yes |
| **UniRig** | 2025 | Mesh | Skeleton + skinning | UNVERIFIED | UNVERIFIED |

### 6.2 Image → Articulated 3D (Category-Specific)

| Method | Year | Categories | Approach | Code Available? |
|--------|------|-----------|----------|-----------------|
| **MagicPony** | CVPR 2023 | Horses, birds, etc. | Implicit-explicit shape + articulation from single image | Yes |
| **3D-Fauna** | CVPR 2024 | 100+ quadrupeds | Pan-category deformable model with SBSM | Yes |
| **LASSIE** | NeurIPS 2022 | Category-specific | Part discovery from image ensembles | Yes |
| **Hi-LASSIE** | CVPR 2023 | Category-specific | Improved LASSIE with hierarchical parts | Yes |

### 6.3 Commercial Auto-Rigging

| Service | Capabilities | Pricing | Limitations |
|---------|-------------|---------|-------------|
| **Meshy** | Image/Text→3D + auto-rig (humanoid) + animation | API credits | Humanoid only for rigging; ~3 min |
| **Tripo** | Image/Text→3D + auto-rig (biped/quadruped) + retargeting | API credits (25 credits/rig) | Biped default; quad support in v2; $0.01/credit |
| **Mixamo (Adobe)** | Upload→auto-rig + animation library | Free (royalty-free) | Humanoid only; web UI, no API; owned by Adobe |
| **Anything World** | Upload→auto-rig + animation | Subscription | Humanoid focus; ~4 min |
| **DeepMotion** | Video→3D animation | Subscription | Different use case |

**Sources:**
- Meshy Rigging API: https://docs.meshy.ai/en/api/rigging
- Tripo Developers: https://developers.tripo3d.ai/en
- Mixamo: https://www.mixamo.com

### 6.4 Full Image → Rig Pipeline (End-to-End)

**No existing method does the full pipeline** (single image → textured mesh → skeleton → skinning → animation-ready export) **in a single model**. The closest are:

- **Puppeteer** + image-to-3D frontend: Image→3D (external) → Puppeteer (rigging) → Puppeteer (animation)
- **Make-It-Animatable** + image-to-3D: Image→3D (external) → Make-It-Animatable (rigging)
- **Commercial:** Tripo and Meshy both offer image→3D→rig→animation as separate API steps.

### 6.5 Defensible Gap

The key gap is: **No open-source, permissively-licensed, end-to-end pipeline that goes from a single arbitrary image to a fully rigged, animation-ready 3D mesh with skeleton, skinning weights, and retargeted animation capabilities.**

Existing approaches each cover only part of the pipeline:

- Image→3D: TRELLIS (MIT), TripoSR (MIT) — produce static meshes
- Mesh→rig: RigAnything (non-commercial), MagicArticulate (no code yet), Puppeteer (CC BY-NC)
- Motion→retarget: AnyTop (code available), STaR (code available)

**A system combining TRELLIS (MIT) + RigAnything (non-commercial — need alternative) + AnyTop would be blocked by license incompatibility.** A system using TRELLIS + in-house rigging (BBW-based) + simple retargeting would be fully MIT-compatible.

---

## 7. Evaluation & Benchmarking

### 7.1 RigNet-Standard Metrics

From the RigNet paper (source: https://ar5iv.labs.arxiv.org/html/2005.00559):

**Skeleton Metrics:**

| Metric | Description | Formula |
|--------|-------------|---------|
| **CD-J2J (Joint-to-Joint Chamfer)** | Symmetric Chamfer distance between predicted and reference joint sets | Lower is better |
| **CD-J2B (Joint-to-Bone Chamfer)** | Distance from predicted joints to nearest reference bone point + symmetric | Lower is better |
| **CD-B2B (Bone-to-Bone Chamfer)** | Symmetric Chamfer distance between bone line segments | Lower is better |
| **IoU (Intersection over Union)** | Hungarian-matched joints below tolerance (1/2 local shape diameter) | Higher is better |
| **Precision** | Fraction of matched predicted joints below tolerance | Higher is better |
| **Recall** | Fraction of matched reference joints below tolerance | Higher is better |
| **TreeEditDist (ED)** | Min joint deletions/insertions/replacements to transform predicted skeleton to reference | Lower is better |

**Skinning Metrics:**

| Metric | Description |
|--------|-------------|
| **Precision** | Fraction of weight-influenced bones (w > 1/4^4 ≈ 0.0039) matching reference |
| **Recall** | Fraction of reference influential bones matching predicted |
| **avg-L1** | Average L1 difference between predicted and reference weight vectors |
| **avg-dist** | Average vertex position difference under reference vs. predicted weights |
| **max-dist** | Maximum per-vertex distance under same bone transformation |

### 7.2 Existing Benchmarks

| Benchmark | Size | Source | License |
|-----------|------|--------|---------|
| **ModelsResource-RigNetv1** | 2,703 rigged characters | models-resource.com (web archive) | UNVERIFIED (likely CC-like) |
| **Articulation-XL** | 33K → 48K → 59.4K (expanded) | Objaverse-XL subset | Derived from Objaverse-XL (Apache 2.0) |
| **Mixamo** | ~2,500 characters + ~2,000 animations | Adobe | Free for use, royalty-free for commercial |
| **KinematicParts20K** | 20K rigged objects | Objaverse-XL subset | Derived from Objaverse-XL (Apache 2.0) |
| **Truebones** | ~2,000 skeletons + motion data | Truebones | UNVERIFIED |

### 7.3 Building a Benchmark (150–300 Objects)

**Recommended methodology:**

1. **Collect** from Objaverse-XL (Apache 2.0) — filter for rigged/animated objects. Use the ArtXL filter criteria from MagicArticulate.
2. **Source permissive-rigged models:**
   - Objaverse XL rigged subset (Apache 2.0): https://objaverse.allenai.org/
   - Mixamo characters (royalty-free, but Adobe terms apply)
   - Sketchfab CC-licensed rigged models (manually curated)
3. **Annotate** with:
   - Category label (biped/quadruped/winged/wheeled/amorphous/non-articulated)
   - Human-verified skeleton (joint positions + connectivity)
   - Human-verified skinning weights
   - 3–5 canonical motion clips per skeleton
4. **Size:** 50 per major category (biped, quadruped, other) for a total of 150–300.

### 7.4 Sources of Rigged Models with Permissive Licenses

- **Objaverse XL** — Apache 2.0. Contains many rigged characters. Source: https://objaverse.allenai.org/
- **ABO (Amazon Berkeley Objects)** — CC-BY 4.0. 8K objects, not rigged but good for non-articulated evaluation.
- **Articulation-XL2.0** metadata — available on Hugging Face: https://huggingface.co/datasets/Seed3D/Articulation-XL2.0
- **KinematicParts20K** — curated from Objaverse XL, includes skinning weights.

---

## 8. Edge Cases

### 8.1 Non-Articulated Objects (Rocks, Furniture, Plants)

- Current SOTA (RigAnything, MagicArticulate) produces skeletons for any mesh, including non-articulated objects, but the skeleton may be degenerate (one root joint).
- **Recommendation:** Detect non-articulated via:
  - Low variance in curvature / part decomposition
  - Category classifier (CLIP-based)
  - Threshold on skeleton complexity (if only 1–2 joints predicted, mark as non-articulated)
- **Fallback:** Output a static mesh with a single root bone (no deformation).

### 8.2 Objects with Missing Parts (Occluded)

- Single-image reconstruction methods (TRELLIS, TripoSR) hallucinate occluded parts.
- Rigging quality degrades with incomplete geometry.
- **Mitigation:** Use uncertainty-aware rigging; confidence masks for skinning weights.

### 8.3 Symmetric vs. Asymmetric

- **Symmetric objects** (most animals, characters) benefit from symmetry priors (as used in MagicPony, 3D-Fauna).
- **Asymmetric objects** (scissors, tools) need no symmetry enforcement.
- **Approach:** Detect symmetry plane via PCA or Chamfer matching, optionally enforce symmetric skeleton.

### 8.4 Low-Poly vs. High-Poly Input

- **RigNet:** Trained on 1K–5K vertex meshes. Inputs outside this range need remeshing.
- **RigAnything:** Simplifies to 8,192 faces before inference.
- **MagicArticulate:** Point cloud sampling (4,096 points) independent of mesh resolution.
- **Recommendation:** Normalize all inputs to a consistent resolution range (e.g., 5K–20K faces) via remeshing/decimation.

### 8.5 Non-Manifold Geometry

- Most single-image reconstruction methods produce manifold meshes (TRELLIS uses Marching Cubes → manifold).
- **Mitigation:** Run mesh repair (trimesh.fill_holes, remove_non_manifold, etc.) before rigging.
- **Available:** pymeshfix (GPL-3.0), trimesh (MIT), Open3D (MIT).

### 8.6 Multiple Identical Parts (Chair Legs, Table)

- RigAnything explicitly handles this via auto-regressive generation — can generate multiple symmetric joints naturally.
- RigNet's mean-shift clustering may merge close symmetric parts.
- **Recommendation:** Use symmetry-aware post-processing to ensure symmetric parts get symmetric joint placement.

### 8.7 Humans (Ethical Concerns)

- Mixamo, Anything World, and Make-It-Animatable already handle humanoids commercially.
- **Ethical considerations:**
  - Deepfake animation / non-consensual use
  - Bias in training data (predominantly certain body types)
  - Privacy (if reconstructing people from photos without consent)
- **Mitigation:** Add content policy, require affirmative consent, use diverse training data, watermark generated animations.
- Many auto-rigging papers explicitly avoid human subjects (RigNet's dataset is fictional characters, not real humans).

---

## 9. Codebase Integration

### 9.1 Required Code Changes

**New Modules Needed:**

```
src/spatial_ingestion/
├── auto_rigging/                          # NEW PACKAGE
│   ├── __init__.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                       # AutoRigBackend ABC
│   │   ├── rig_anything.py               # RigAnything wrapper
│   │   ├── magic_articulate.py           # MagicArticulate wrapper (when code releases)
│   │   └── bbw_fallback.py              # BBW-based fallback
│   ├── skeleton/
│   │   ├── __init__.py
│   │   ├── prediction.py                 # Skeleton prediction interface
│   │   ├── template_matching.py          # Template retrieval + deformation
│   │   └── evaluation.py                 # J2J, J2B, B2B metrics
│   ├── skinning/
│   │   ├── __init__.py
│   │   ├── bbw.py                        # Bounded Biharmonic Weights
│   │   ├── geodesic.py                   # Volumetric geodesic computation
│   │   └── refinement.py                 # Learned refinement wrapper
│   ├── retargeting/
│   │   ├── __init__.py
│   │   ├── skeleton_correspondence.py    # Joint mapping
│   │   ├── motion_transfer.py            # Rotation retargeting
│   │   └── canonical_clips.py            # Bundled animation clips
│   ├── pipeline.py                       # Orchestrator: image→mesh→rig→retarget→export
│   ├── models.py                         # Pydantic models (rigged mesh, skeleton, etc.)
│   └── export.py                         # FBX/GLB export with skinning
└── reconstruction/
    ├── models.py                         # ADD: RiggedMesh artifact kind
    └── backends/
        └── trellis.py                    # NEW: TRELLIS backend (single-image)
```

**Changes to Existing Code:**

| File | Change |
|------|--------|
| `src/spatial_ingestion/reconstruction/models.py` | Add `RIGGED_MESH` to `ReconstructionArtifactKind` enum. Add `AutoRigConfig` model. |
| `src/spatial_ingestion/reconstruction/backends/base.py` | Add `RIGGED_MESH` handling to plan/execute contract. |
| `src/spatial_ingestion/reconstruction/registry.py` | Register new backends (TRELLIS, auto-rigging). |
| `src/spatial_ingestion/config.py` | Add `AUTO_RIGGING_OUTPUT_ROOT`, `RIGGING_BACKEND_DEFAULT`, canonical animation paths. |
| `src/spatial_ingestion/outcomes_engine/engine.py` | Add rigged mesh export as optional outcome; support FBX and skinned GLB export. |
| `pyproject.toml` | Add dependencies (see below). |

### 9.2 Does Existing Backend ABC Work for Single-Image?

**Yes, with minor extensions.** The current `ReconstructionBackend` ABC (`base.py`) has:

- `supports(job)` — checks mode
- `plan(job)` — returns expected artifacts
- `execute(job)` — runs backend

For single-image backends, add `ReconstructionMode.SINGLE_VIEW` support. The artifact kind `MESH` already exists. Add `RIGGED_MESH` for the skinned output.

**Potential issue:** The current `ReconstructionJob` is designed around multi-view reconstruction (multiple `image_uris`). For single-image, a single URI should suffice. This is already handled since `image_uris` is a list (can be length 1) and `SINGLE_VIEW` mode exists in `GenerationMode` but not in `ReconstructionMode` — need to add it.

### 9.3 New Dependencies for pyproject.toml

```toml
# Core auto-rigging
"torch>=2.4.0",                          # Already present
"trimesh>=4.0.0",                        # Already present
"numpy<2",                               # Already present
"scipy>=1.13.0",                         # Already present

# Mesh processing
"open3d>=0.18.0",                        # Mesh repair, remeshing, geodesic distances
"pymeshlab>=2023.12",                    # Mesh simplification, repair
"libigl>=2.5.0",                         # BBW, bounded biharmonic weights (C++ with Python)

# 3D export
"pygltflib>=1.0.0",                      # GLB/glTF export with skinning
"fbx-sdk>=2020.3",                       # FBX export (optional, via Blender or binary SDK)

# Skeleton processing
"networkx>=3.2",                         # Skeleton graph operations (tree edit distance, BFS ordering)

# Animation
"bvh>=1.0.0",                            # BVH motion file parsing
"smplx>=0.1.28",                         # SMPL skeleton utilities (optional, for humanoids)

# Evaluation
"scikit-learn>=1.5.0",                   # Already present — for Hungarian matching
"chamferdist>=1.0.1",                    # GPU-accelerated Chamfer distance for J2J/J2B/B2B

# HuggingFace integration
"huggingface-hub[torch]>=0.22,<1.0",    # Already present

# Single-image backends
"diffusers>=0.30.0",                     # For HuggingFace pipeline loading
"accelerate>=0.30.0",                    # For device placement
"xformers>=0.0.28",                      # Memory-efficient attention (for TRELLIS etc.)
```

**Optional dependencies (grouped):**

```toml
[project.optional-dependencies]
trellis = [
    "microsoft-trellis @ git+https://github.com/microsoft/TRELLIS.git",
    "diffoctreerast @ git+https://github.com/JeffreyXiang/diffoctreerast.git",
]
hunyuan3d = [
    "hy3dgen @ git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
]
riganything = [
    "riganything @ git+https://github.com/Isabella98Liu/RigAnything.git",
]
```

### 9.4 Outcome Engine Changes

Current `OutcomesEngine` (`outcomes_engine/engine.py`) needs to handle:

| Outcome Type | Export Format | Notes |
|-------------|---------------|-------|
| `rigged_mesh_fbx` | `.fbx` | Full rig with skeleton, skinning weights, mesh, texture. FBX SDK or Blender Python API. |
| `rigged_mesh_glb` | `.glb` (skinned) | glTF 2.0 with skinning. Use `pygltflib` or trimesh export. Defines joints, inverse bind matrices, weights. |
| `animation_clip` | `.fbx` or `.glb` | Pre-baked animated mesh (one frame per animation pose) — useful for visualization. |
| `skeleton_only` | `.bvh` or `.json` | Skeleton hierarchy for retargeting pipeline. |

**Key challenge:** FBX export requires either:

- Blender's `bpy` module (requires Blender installation)
- Autodesk FBX SDK Python bindings (proprietary, but binary packages exist)
- Assimp (via `pyassimp`, GPL-licensed)

**Recommendation:** Use GLB as primary export format (royalty-free, trimesh supports writing skinned GLB via `pygltflib`). Offer FBX as secondary via Blender headless rendering.

### 9.5 Test Infrastructure

```python
# tests/test_auto_rigging/
# ├── test_backends.py       # Test each rigging backend
# ├── test_skeleton.py       # Test skeleton prediction metrics
# ├── test_skinning.py       # Test skinning weight computation
# ├── test_retargeting.py    # Test motion retargeting
# ├── test_pipeline.py       # Test end-to-end pipeline
# ├── test_export.py         # Test FBX/GLB export correctness
# └── test_edge_cases.py     # Test non-articulated, low-poly, etc.
```

**Key test categories:**

1. **Unit tests:** Individual components (skeleton prediction, skinning, export)
2. **Integration tests:** Backend wrappers (TRELLIS→Mesh→Rig→Export)
3. **Edge case tests:** Non-articulated, degenerate geometry, asymmetric objects
4. **Performance tests:** VRAM usage, inference time
5. **Export validation:** Verify GLB/FBX have correct joints, weights, mesh
6. **Metric regression:** Track J2J, J2B, B2B over time

**Test fixtures:**

- Create a small repository of test meshes (5–10) with ground-truth skeletons
- Use synthetic test cases (geometric primitives with known skeletons)
- Include non-articulated controls (sphere, cube, rock)

---

## 10. Summary of Recommendations

1. **Image→3D Backend:** **TRELLIS (MIT)** is the best choice for open-source, permissively-licensed, high-quality single-image reconstruction. TripoSR (MIT) provides a lightweight alternative.

2. **Rigging Backend:** **RigAnything** provides the best template-free rigging but is **non-commercial (Adobe Research License)**. For a commercial pipeline, either:
   - Use **BBW** (via libigl) as a fallback with template-based skeleton placement (Pinocchio-style)
   - Wait for **MagicArticulate** code release (likely permissive for research, unclear commercial)
   - Develop a simple auto-regressive rigging approach

3. **Skinning:** Start with **BBW** (MIT-friendly via libigl) for initialization. Optionally add learned refinement if quality is insufficient.

4. **Retargeting:** Use **AnyTop** (code available) or implement a simple heuristic-based correspondence + rotation transfer. Start with 2–3 canonical clips (idle, walk, wave).

5. **Export:** Use **GLB with skinning** as primary format (trimesh/pygltflib, MIT). Add FBX export as secondary option.

6. **Defensible Gap:** The full **MIT-licensed, end-to-end pipeline** from single image → rigged, animated 3D mesh does not exist. This is the gap z-shift Phase 5 can fill.

7. **License Strategy:** Combine MIT-licensed components (TRELLIS, BBW via libigl) with custom code released under MIT. Avoid GPL components in production unless prepared for GPL obligations.
