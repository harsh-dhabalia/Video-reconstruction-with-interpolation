# Video Reconstruction via Pixel-wise Temporal Interpolation

## 1. Problem Setup

A 5-second video is treated as a function

```
f_{x,y}(t) : {0, 1, …, T-1} → [0, 255]^3
```

where `t` is the frame index and `(x, y)` is a pixel location.
Given a *sparse* subset of frames at chosen time nodes
`t_0 < t_1 < … < t_n`, we reconstruct the full sequence by
interpolating each pixel channel independently in time.

---

## 2. Node Types

### 2.1 Equidistant Nodes
```
t_k = t_min + k * (t_max - t_min) / (n-1),   k = 0, …, n-1
```
Simple and intuitive, but susceptible to **Runge's phenomenon**
when used with high-degree polynomial interpolation: oscillations
grow near the endpoints of the interval.

### 2.2 Chebyshev Nodes (1st kind)
```
t_k = (t_min + t_max)/2  +  (t_max - t_min)/2 · cos((2k-1)π / 2n)
```
Concentrated near the endpoints, these nodes minimise the
Lebesgue constant and suppress Runge oscillations for Lagrange
and Hermite interpolation.  They are the theoretically optimal
choice for polynomial approximation.

### 2.3 Random Nodes
Nodes are drawn uniformly at random from `[t_min, t_max]` (fixed
seed for reproducibility). Endpoints are always included to avoid
extrapolation.  Performance varies across runs and tends to be
worse than Chebyshev for polynomial methods.

---

## 3. Interpolation Methods

### 3.1 Lagrange Polynomial Interpolation

Constructs a **single polynomial of degree n-1** passing through all
`n` nodes using the **barycentric form** (numerically stable):

```
p(t) = [ Σ_j  w_j y_j / (t - t_j) ]  /  [ Σ_j  w_j / (t - t_j) ]

where  w_j = 1 / Π_{k≠j} (t_j - t_k)
```

**Advantages**: exact at nodes, simple formula.  
**Disadvantages**: expensive O(n²) weight computation; severe
oscillations (Runge's phenomenon) with equidistant nodes and n > ~15.

### 3.2 Hermite Polynomial Interpolation (Piecewise Cubic)

A **piecewise cubic** is fit on each sub-interval `[t_k, t_{k+1}]`
using function values and first derivatives estimated by centred
finite differences:

```
dy_k ≈ (y_{k+1} - y_{k-1}) / (t_{k+1} - t_{k-1})
```

The cubic Hermite basis functions on `s ∈ [0,1]`:
```
h₀₀ = 2s³ - 3s² + 1
h₁₀ =  s³ - 2s² + s
h₀₁ = -2s³ + 3s²
h₁₁ =  s³ -  s²
```

The reconstruction is C¹ (continuous first derivative) and avoids
global Runge oscillations, though the derivative estimates at
endpoints may be slightly less accurate.

### 3.3 Cubic Spline Interpolation

A **global C² piecewise cubic** that minimises the "bending energy":

```
min  ∫ [p''(t)]² dt
subject to  p(t_k) = y_k  for all k
```

Uses the *not-a-knot* boundary condition (the third derivative is
continuous at the second and second-to-last knots).  Implemented
via `scipy.interpolate.CubicSpline`.

**Advantages**: smooth (C²), no oscillations, computationally
efficient.  Generally the best choice for video reconstruction.

---

## 4. Pipeline Architecture

```
Input video
    │
    ▼
extract_frames()  ──►  full frame list  [f₀, f₁, …, f_{T-1}]
    │
    ▼
Node generator (equidistant / Chebyshev / random)
    │
    ▼
Snap nodes to nearest actual frame indices
    │
    ▼
reconstruct_fast()
    │  For each colour channel c ∈ {R,G,B}:
    │    For each pixel p ∈ {0 … H·W-1}:
    │      y_nodes = known_frames[:, p, c]
    │      y_all   = interp_fn(t_nodes, y_nodes, t_all)
    │
    ▼
Clip to [0, 255], assemble frames
    │
    ▼
write_video()  ──►  output/*.mp4
```

---

## 5. Running the Code

### Install dependencies
```bash
pip install -r requirements.txt
```

### Reconstruct from your own video
```bash
python reconstruct.py "Prof Qun plays with Chewie.mp4" \
    --n_nodes 20 \
    --methods lagrange hermite spline \
    --node_types equidistant chebyshev random \
    --outdir output
```

### Run the self-contained demo (synthetic video, no upload needed)
```bash
python demo.py
```

### Evaluate reconstructed videos
```bash
python evaluate.py "Prof Qun plays with Chewie.mp4" --recon_dir output
```

---

## 6. Expected Results and Discussion

### 6.1 Node Type Comparison

| Node type    | With Lagrange        | With Spline/Hermite |
|--------------|----------------------|----------------------|
| Equidistant  | ❌ Runge oscillations (n > 15) | ✅ Good |
| Chebyshev    | ✅ Best polynomial accuracy | ✅ Excellent |
| Random       | ⚠️ Unpredictable quality | ⚠️ Depends on luck |

**Why Chebyshev nodes work best for Lagrange/Hermite**: the error
bound for polynomial interpolation at the nodes contains the factor
`‖ω‖_∞` where `ω(t) = Π(t - t_k)`.  Chebyshev nodes minimise this
factor over all choices of n nodes on an interval.

**Why random nodes are risky**: clustering in one region leaves other
regions with large gaps, causing large interpolation errors and
visible "jumps" in the reconstructed video.

### 6.2 Method Comparison

| Method   | Global/Local | Continuity | Cost  | Runge risk |
|----------|-------------|------------|-------|------------|
| Lagrange | Global poly | C^∞ but oscillatory | O(n²) | High (equidistant) |
| Hermite  | Piecewise   | C¹         | O(n)  | None |
| Spline   | Piecewise   | C²         | O(n)  | None |

**Cubic spline** produces the smoothest motion because C² continuity
ensures the "acceleration" (second derivative) of pixel brightness
is continuous — closely matching the physics of a smoothly moving
scene.

**Lagrange** works reasonably well with few nodes (< 15) but
produces visible flickering and colour artefacts with many nodes
and equidistant spacing due to Runge's phenomenon.  Chebyshev nodes
largely cure this.

**Hermite** is a good middle ground: more stable than Lagrange,
slightly less smooth than spline (C¹ vs C²), but very fast.

### 6.3 Visual Artefacts

| Artefact | Cause | Fix |
|---|---|---|
| Flickering brightness | Runge oscillations in Lagrange | Use Chebyshev nodes or switch to spline |
| Blocky / stepped motion | Too few nodes | Increase n_nodes |
| Colour fringing | Channel-independent interpolation | Accept as limitation of the per-channel model |
| Blur at fast-motion regions | Low-order method averaging over motion | Use more nodes in dense-motion regions |

### 6.4 Metrics

The evaluation script computes:

- **PSNR** (dB) — peak signal-to-noise ratio; higher is better.
  Typical good reconstruction: > 30 dB.
- **SSIM** — structural similarity (0–1); values > 0.9 indicate
  visually close reconstruction.
- **Motion smoothness** — mean absolute frame-difference; lower
  values mean smoother temporal transitions (closer to original).

Expect spline + Chebyshev to score highest across all three metrics.

---

## 7. Limitations and Extensions

- **Per-pixel assumption**: the model ignores spatial correlations.
  A block-based or optical-flow-guided approach would better handle
  large motions.
- **Colour space**: interpolating in BGR can produce unnatural hues;
  converting to YCbCr before interpolation (and back after) often
  reduces colour artefacts.
- **Adaptive node placement**: automatically placing more nodes
  near frames with large temporal gradients (scene cuts, fast motion)
  would improve quality with fewer total samples.

---

## 8. File Structure

```
video_reconstruction/
├── reconstruct.py      # Main pipeline (nodes + interpolators + I/O)
├── evaluate.py         # PSNR / SSIM / smoothness metrics
├── demo.py             # End-to-end demo with synthetic video
├── requirements.txt
└── output/             # Reconstructed MP4 files written here
```
