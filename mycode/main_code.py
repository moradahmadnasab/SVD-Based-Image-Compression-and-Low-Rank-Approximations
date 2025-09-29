"""
SVD Image Compression (GitHub figures folder)
---------------------------------------------
This script performs low-rank approximation of both color and grayscale images
using Singular Value Decomposition (SVD). It automatically picks the first image
found in the 'figures/' folder of the repository.

Outputs:
- Compressed images saved in 'results/compressed/'
- Original images saved in 'results/original/'
- Metrics (PSNR, Compression Ratio) saved in 'results/metrics.tex'
"""

import os
import glob
import numpy as np
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr

# --------------------- CONFIGURATION ---------------------
figures_folder = "figures"
k_values = [10, 20, 50, 100]  # ranks to test

# Output folders
orig_folder = "results/original"
comp_folder = "results/compressed"
os.makedirs(orig_folder, exist_ok=True)
os.makedirs(comp_folder, exist_ok=True)

# --------------------- SELECT FIRST IMAGE ---------------------
image_files = glob.glob(os.path.join(figures_folder, "*.*"))  # match any extension
if not image_files:
    raise FileNotFoundError(f"No image files found in '{figures_folder}' folder")
image_path = image_files[0]
print(f"Using image file: {image_path}")

# --------------------- LOAD IMAGE ---------------------
img_color = io.imread(image_path).astype(np.float64) / 255.0  # normalize [0,1]
img_gray = color.rgb2gray(img_color)

# Save original images
io.imsave(os.path.join(orig_folder, "original_color.png"), img_as_ubyte(img_color))
io.imsave(os.path.join(orig_folder, "original_gray.png"), img_as_ubyte(img_gray))

# --------------------- HELPER FUNCTIONS ---------------------
def compress_svd(img, k):
    """Compress image using top-k SVD."""
    U, S, Vt = np.linalg.svd(img, full_matrices=False)
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vk = Vt[:k, :]
    return Uk @ Sk @ Vk

def compression_ratio(m, n, k, channels=1):
    """Compute compression ratio."""
    return (m * n * channels) / (k * (m + n + 1) * channels)

# --------------------- EXPERIMENTS ---------------------
m, n, _ = img_color.shape
m_gray, n_gray = img_gray.shape

metrics_path = "results/metrics.tex"
with open(metrics_path, "w") as f:
    f.write("% Auto-generated metrics from Python script\n")
    f.write("\\begin{tabular}{c c c}\n")
    f.write("Rank k & Compression Ratio & PSNR (dB) \\\\\n\\hline\n")

    for k in k_values:
        # Color
        comp_color = np.zeros_like(img_color)
        for c in range(3):
            comp_color[:, :, c] = compress_svd(img_color[:, :, c], k)
        cr_color = compression_ratio(m, n, k, channels=3)
        psnr_color = psnr(img_color, comp_color, data_range=1.0)
        io.imsave(os.path.join(comp_folder, f"color_k{k}.png"),
                  img_as_ubyte(np.clip(comp_color, 0, 1)))

        # Grayscale
        comp_gray = compress_svd(img_gray, k)
        cr_gray = compression_ratio(m_gray, n_gray, k)
        psnr_gray = psnr(img_gray, comp_gray, data_range=1.0)
        io.imsave(os.path.join(comp_folder, f"gray_k{k}.png"),
                  img_as_ubyte(np.clip(comp_gray, 0, 1)))

        # Write metrics to LaTeX file
        f.write(f"{k} & {cr_color:.2f} & {psnr_color:.2f} \\\\\n")
        f.write(f"{k} & {cr_gray:.2f} & {psnr_gray:.2f} \\\\\n")

    f.write("\\end{tabular}\n")

print(f"Experiment completed. Compressed images saved in '{comp_folder}'. Metrics saved in '{metrics_path}'.")

