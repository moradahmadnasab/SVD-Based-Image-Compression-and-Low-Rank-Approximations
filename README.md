# SVD-Based Image Compression and Low-Rank Approximations

This repository presents a short showcase of **image compression using Singular Value Decomposition (SVD)**.  

**"Applications of Singular Value Decomposition for Image Approximation and Compression"**  
Morad Ahmadnasab, [Self-contained Project, 2025].  

---

## 🔎 Overview
Low-rank approximation of images using SVD is a powerful technique in **image compression**, **denoising**, and **dimensionality reduction**.  
For a given image matrix \(A\), a **rank-\(k\) approximation** \(A_k\) is constructed as:

\[
A_k = \sum_{i=1}^k \sigma_i u_i v_i^T
\]

where \(\sigma_i\) are the singular values, and \(u_i, v_i\) are the corresponding left and right singular vectors.  
This approach allows **efficient storage** while maintaining visual quality.

---

## ✨ Key Features
- Automatic SVD-based **compression for both grayscale and color images**.  
- Calculates **Peak Signal-to-Noise Ratio (PSNR)** and **Compression Ratio (CR)** for each rank \(k\).  
- Outputs **compressed images** and a **LaTeX metrics table** for direct inclusion in reports.  
- Fully reproducible using **Python 3.x** and open-source libraries: `numpy`, `scikit-image`, `matplotlib`.

---

## 📊 Example Figures
<p align="center">
  <img src="results/compressed/color_k50.png" width="600" alt="Color Image Approximation">
</p>

<p align="center">
  <img src="results/compressed/gray_k50.png" width="600" alt="Grayscale Image Approximation">
</p>

*The figures show rank-50 approximations for color and grayscale versions of the same image.*

---

## 🛠 Usage

1. Place your input image(s) in the `figures/` folder.  
2. Run the Python script:

```bash
python svd_image_compression.py
