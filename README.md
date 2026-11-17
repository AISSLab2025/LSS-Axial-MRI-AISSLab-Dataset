
# LSS-Axial-MRI-AISSLab-Dataset

## 🧠 Lumbar Spine MRI — Technical Validation

This repository provides the source code and experimental results used for the technical validation of the **Medical Lumbar Spine 3D Axial MRI Dataset for Anatomical Segmentation, Localization, and Stenosis Severity Classification**.

The implemented deep learning pipelines support reproducible experiments across three primary tasks:

- **Anatomical Segmentation:** Pixel-level delineation of lumbar spine anatomical structures.
- **ROI Localization:** Detection and localization of anatomical regions of interest.
- **Stenosis Severity Classification:** Classification of lumbar spinal stenosis severity.

The repository includes implementation notebooks, quantitative evaluation results, and representative visualizations.

---

## 📊 Anatomical Segmentation

### 1. Quantitative Performance Evaluation

The following table presents the average quantitative performance of six deep learning architectures evaluated on the proposed axial lumbar spine MRI dataset.

**Table 3.** Average quantitative evaluation of anatomical segmentation performance across the proposed axial lumbar spine MRI dataset.

| Model | Dice | IoU | HD95 | ASSD | Pre | Rec | Spe | Accu | Params (M) | FLOPs (G) | FPS |
|:------|-----:|----:|-----:|-----:|----:|----:|----:|-----:|-----------:|----------:|----:|
| DeepLabv3 | 86.88 | 79.04 | **4.83** | **1.42** | 88.13 | 87.85 | 98.07 | 99.35 | 39.63 | 62.66 | 169.8 |
| AttentionUNet | **87.59** | **80.01** | 5.42 | 1.50 | **90.18** | 87.30 | 98.01 | **99.36** | 1.99 | 4.13 | 236.3 |
| UNETR | 82.87 | 73.71 | 7.79 | 2.12 | 83.90 | 84.18 | 97.30 | 99.03 | 115.61 | 38.37 | 101.2 |
| SwinTransformer | 87.10 | 79.44 | 5.02 | **1.42** | 89.37 | 87.24 | 98.03 | **99.36** | 31.63 | 13.57 | 93.12 |
| SegResNet | 87.28 | 79.72 | 5.42 | 1.55 | 88.87 | **87.97** | **98.18** | **99.36** | **0.40** | **1.53** | **369.3** |
| TransUNet | 87.21 | 79.87 | 5.44 | 1.43 | 89.34 | 87.49 | 97.94 | **99.36** | 37.06 | 13.82 | 149.8 |

**Evaluation Metrics:**

- **Dice:** Dice Similarity Coefficient
- **IoU:** Intersection over Union
- **HD95:** 95th Percentile Hausdorff Distance
- **ASSD:** Average Symmetric Surface Distance
- **Pre:** Precision
- **Rec:** Recall
- **Spe:** Specificity
- **Accu:** Accuracy
- **Params (M):** Number of model parameters (millions)
- **FLOPs (G):** Floating-Point Operations (billions)
- **FPS:** Inference throughput (frames per second)

**Note:** Bold values indicate the best performance for each metric. Higher values are preferable for Dice, IoU, precision, recall, specificity, accuracy, and FPS, whereas lower values are preferable for HD95, ASSD, model parameters, and FLOPs.

### 2. Qualitative Segmentation Results

Representative qualitative results of anatomical segmentation on axial lumbar spine MRI images are presented below.

<p align="center">
  <img src="Segmentation_Results/Picture1.png"
       alt="Qualitative Anatomical Segmentation Results"
       width="850">
</p>

<p align="center">
  <em>Figure 1. Representative anatomical segmentation results on axial lumbar spine MRI images.</em>
</p>

### 3. Performance Summary

The quantitative evaluation highlights the following observations:

- **AttentionUNet** achieved the highest Dice score (**87.59%**) and IoU (**80.01%**), demonstrating strong segmentation performance.
- **DeepLabv3** achieved the lowest HD95 (**4.83**), indicating favorable boundary agreement.
- **SegResNet** demonstrated the highest computational efficiency, with only **0.40 million parameters**, **1.53 GFLOPs**, and an inference throughput of **369.3 FPS**.
- **SwinTransformer** and **DeepLabv3** achieved the lowest ASSD (**1.42**).

These results provide a quantitative comparison of segmentation accuracy, boundary agreement, and computational efficiency across the evaluated architectures.

---



## 📚 Citation

If you use this repository, please cite the following publications.

```bibtex
@article{Abdulmahmod2026SciData,
  author  = {Abdulmahmod, Osamah F. and Al-antari, Mugahed A. and Kwon, Hyunwook and Habib, Afnan and Han, Inbo and others},
  title   = {Medical Spine Sagittal MRI Dataset for Segmentation and Foraminal Stenosis Detection},
  journal = {Scientific Data},
  volume  = {13},
  pages    = {809},
  year     = {2026},
  doi      = {10.1038/s41597-026-07138-x}
}
```
```bibtex
@article{AlAntari2025AIR,
  author  = {Al-Antari, Mugahed A. and Salem, Saied and Raza, Mukhlis and Elbadawy, Ahmed S. and B{\"u}t{\"u}n, Ertan and Aydin, Ahmet Arif and Aydo{\u{g}}an, Murat and Ertu{\u{g}}rul, Bilal and Talo, Muhammed and Gu, Yeong Hyeon},
  title   = {Evaluating AI-powered Predictive Solutions for MRI in Lumbar Spinal Stenosis: A Systematic Review},
  journal = {Artificial Intelligence Review},
  volume  = {58},
  number  = {8},
  pages   = {221},
  year    = {2025}
}
```
