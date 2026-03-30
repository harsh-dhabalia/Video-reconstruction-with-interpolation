# Video-reconstruction with interpolation
# Video Reconstruction via Pixel-wise Temporal Interpolation

![Video Reconstruction](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains a high-performance framework for reconstructing full-length videos from a sparse set of keyframes. By treating every pixel as a temporal signal, we apply various mathematical interpolation techniques to "fill in the blanks" between sampled frames.

## 🚀 Quick Start

To see the pipeline in action immediately without providing your own video, simply run the synthetic demo:

```bash
python demo.py
```
This will:
1. Generate a 5-second `test_input.mp4` with moving synthetic shapes.
2. Sample keyframes using three different node distributions.
3. Reconstruct the full video using three interpolation methods.
4. Calculate and display evaluation metrics (PSNR, SSIM, Smoothness).

---

## 🛠️ Main Components

### 1. `reconstruct.py` (The Core Engine)
This is the primary script used to perform video reconstruction. It is fully vectorised using NumPy, ensuring that interpolation across millions of pixels is computationally efficient.

**Supported Node Types (Sampling Strategies):**
*   **Equidistant**: Uniform spacing between frames. Simple but prone to Runge's phenomenon.
*   **Chebyshev**: Nodes concentrated near the start and end of the sequence. Theoretically optimal for minimising polynomial interpolation error.
*   **Random**: Stochastic sampling (useful for testing robustness).

**Supported Interpolation Methods:**
*   **Lagrange**: Global polynomial interpolation. Excellent with Chebyshev nodes; oscillatory with equidistant nodes.
*   **Hermite (Piecewise Cubic)**: Ensures $C^1$ continuity (smooth velocity). Very stable and local.
*   **Cubic Spline**: Ensures $C^2$ continuity (smooth acceleration). Generally provides the most "natural" motion.

### 2. `evaluate.py` (Performance Analysis)
Quantifies the quality of the reconstruction by comparing it against the original ground truth video.

*   **PSNR (Peak Signal-to-Noise Ratio)**: Measured in dB; higher is better.
*   **SSIM (Structural Similarity Index)**: Measures perceived visual quality (0 to 1).
*   **Motion Smoothness**: Calculates the mean absolute difference between consecutive frames to detect jitter or flickering.

### 3. `demo.py` (Synthetic Pipeline)
A self-contained script that generates a moving-disk synthetic video and runs the entire sampling/reconstruction/evaluation loop. Ideal for verification in environments without external video assets.

---

## 💻 Usage

### Installation
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

### Running Reconstruction
To reconstruct a specific video file:
```bash
python reconstruct.py "your_video.mp4" --n_nodes 20 --outdir output
```

### Running Evaluation
To evaluate the outputs in your `output/` folder against the original:
```bash
python evaluate.py "your_video.mp4" --recon_dir output
```

---

## 📂 Project Structure

*   `reconstruct.py`: Vectorised interpolation logic and video I/O.
*   `evaluate.py`: Metric calculation and comparison logic.
*   `demo.py`: Synthetic data generator and pipeline demonstrator.
*   `REPORT.md`: Detailed mathematical derivation and analysis of results.
*   `requirements.txt`: Python package dependencies.
*   `output/`: (Generated) Directory containing reconstructed `.mp4` files.

---

> [!TIP]
> **Pro Tip**: For the best visual results on real-world footage, use **Cubic Spline** interpolation with **Chebyshev nodes**. This combination minimizes temporal artifacts and ensures smooth motion transitions.
