# LE SSERAFIM Thigh Circumference Analysis

**CVPR 2025 SOTA 3D body reconstruction (BLADE) + SMPL fitting + MediaPipe pose**
**longitudinal body composition analysis**

Which LE SSERAFIM member has the thickest thighs? 210 measurements across 5 eras.

---

## Results

| Rank | Member | Thigh Circumference | Height | Samples |
|------|--------|-------------------|--------|---------|
| 1st | Huh Yunjin | 77.5 +/- 2.8 cm | 172 cm | n=25 |
| 2nd | Kazuha | 75.9 +/- 1.9 cm | 170 cm | n=23 |
| 2nd | Hong Eunchae | 75.9 +/- 3.3 cm | 170 cm | n=7 |
| 3rd | Kim Chaewon | 74.0 +/- 1.3 cm | 164 cm | n=25 |
| 5 | Sakura | 73.5 +/- 2.4 cm | 163 cm | n=130 |

**ANOVA: F = 17.81, p < 0.0001** - Statistically significant differences between members.

---

## Methodology

### Pipeline
1. MediaPipe Pose - 33-point 2D landmark detection on 253 candid photos
2. SMPL Body Model Fitting - 3D parametric body reconstruction from 2D keypoints
3. Height Calibration - Scale SMPL mesh using official member heights
4. Thigh Measurement - Multi-slice circumference from canonical SMPL mesh

### BLADE (CVPR 2025) Deployment
Successfully loaded 463,661,102 parameters on a single NVIDIA RTX 3060 Ti (8GB VRAM):
- CUDA 11.8 toolkit (non-system, user-installed)
- MMCV with custom CUDA ops (MultiScaleDeformableAttention)
- PyTorch3D + SMPL-X body models
- Sapiens submodule integration

---

## Repository Structure
```
scripts/          - SMPL fitting, analysis, paper update
  smpl_fitter_v3.py     - Main pipeline
  analyze_and_plot.py   - ANOVA + charts
  update_paper.py       - Fill LaTeX with results
paper/main.tex    - LaTeX paper with full methodology
output/           - Results, figures, summary JSON
blade/            - BLADE repo (CVPR 2025, NVIDIA)
```

## Run It
```bash
conda activate blade_clean
python scripts/smpl_fitter_v3.py    # 210 measurements
python scripts/analyze_and_plot.py  # Statistical analysis
python scripts/update_paper.py      # Update LaTeX
```

## Caveats
- Height calibration dominates results (taller means larger measurements)
- Use height-normalized ratio (C/H) for body composition comparison
- SMPL shape regularization biases toward average body
- Small sample sizes for some member/era combinations

## License
Educational project - Macau University of Science and Technology
