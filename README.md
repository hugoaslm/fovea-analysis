# Modeling the Foveal Pit from Radial OCT

Master's Project (Engineering cycle, ISEP Paris) — Computer Vision.  
Goal: **segment retinal layers in radial OCT B-scans**, **register** the slices, **reconstruct** the 3D foveal surface, and **fit** a 2D Gaussian model to extract biomarkers (depth, width, centering).

## Context
Optical Coherence Tomography (OCT) provides high‑resolution cross‑sections of the retina. Around the fovea, radial acquisitions (0–360°) enable 3D reconstruction of the foveal pit and derivation of clinically useful parameters.

## Pipeline
1. **Pre‑processing**: angle sorting, cropping, intensity normalization (percentiles), masking of instrument artefacts.  
2. **Multilayer segmentation (ILM, Hyper‑HRC, Ext‑HRC)**:  
   - vertical gradient (Sobel) + Canny fused into a **cost map**;  
   - **dynamic programming** shortest‑path search per column;  
   - polynomial smoothing;  
   - quality control (local variance, luminance) and optional **active contour** refinement.  
3. **Inter‑slice registration**: affine alignment (from segmented curves) followed by **phase correlation** (FFT) for residual translation.  
4. **3D reconstruction**: polar coordinates \((r,\theta)\) → interpolation → Cartesian grid \((x,y,z)\).  
5. **Mathematical modeling**: **2D Gaussian fit** \(A,\sigma_x,\sigma_y,x_0,y_0,C\) via non‑linear least squares.  
6. **Evaluation & visualization**: overlays, registration RGB checks, 3D surfaces vs. model, metrics (Recall, Precision, Accuracy, F1).

## Key Results
- **ILM → Ext‑HRC surface (robust zone)**: mean **Accuracy 99.17%**, **F1 97.44%** across 8 series.  
- **Hyper‑HRC → Ext‑HRC surface (thinner zone)**: **Accuracy 99.17%**, **F1 87.66%**.  
- **Registration**: typical RMSE **< 0.2** after affine + phase correlation.  

**Interpretation:** ILM→Ext‑HRC is the most stable and precise; inner HRC is more sensitive to noise/artefacts.

## Data
- **8 radial OCT series** centered on the fovea (regular angles), grayscale 2D slices.  
- Three interfaces segmented: **ILM**, **Hyper‑HRC**, **Ext‑HRC**.  
- Ground truth available for quantitative evaluation.

## Installation
```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

**requirements.txt **
```
numpy
scipy
scikit-image
opencv-python
matplotlib
scikit-learn
tqdm
imageio
```

## Technical Highlights
- **Cost map**: weighted fusion of Canny edges and Sobel vertical gradient for robust boundaries.  
- **Dynamic programming** ensures continuous, anatomically plausible interfaces.  
- **Automatic QC** (local variance, luminance) with fallback Otsu + active contour.  
- **Hybrid registration** (curve‑driven affine + phase correlation) for angular consistency.  
- **2D Gaussian model** yields interpretable parameters (A, σx, σy, x0, y0, C).

## Validation
- **Metrics**: Recall, Precision, Accuracy, F1 (per series and per surface), **registration RMSE**.  
- **Qualitative review**: residual peripheral fringes and their impact on reconstruction.

## Roadmap
- Replace affine + phase correlation with a **learning‑based registration** (e.g., self‑supervised descriptors) to reduce local errors.  
- Add a **lightweight CNN refinement** after DP for challenging HRC interfaces.  
- Extend modeling beyond a single Gaussian (ellipsoidal paraboloid, mixtures).

## Authors
Hugo Anselme, Baptiste Aubrée — Engineering cycle (ISEP Paris).
