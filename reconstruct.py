"""
Video Reconstruction via Pixel-wise Temporal Interpolation
==========================================================
Fully vectorised — all pixel loops replaced with NumPy matrix ops.
Supports:
  Node types  : equidistant, Chebyshev, random
  Methods     : Lagrange, Hermite (finite-diff derivatives), Cubic Spline
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# ─────────────────────────────────────────────
#  Node generation
# ─────────────────────────────────────────────

def equidistant_nodes(n_nodes, t_min, t_max):
    return np.linspace(t_min, t_max, n_nodes)

def chebyshev_nodes(n_nodes, t_min, t_max):
    k = np.arange(1, n_nodes + 1)
    nodes = 0.5*(t_min+t_max) + 0.5*(t_max-t_min)*np.cos((2*k-1)*np.pi/(2*n_nodes))
    return np.sort(nodes)

def random_nodes(n_nodes, t_min, t_max, seed=42):
    if n_nodes <= 2:
        return np.linspace(t_min, t_max, n_nodes)
    rng = np.random.default_rng(seed)
    # Generate n_nodes-2 random nodes in the interior
    inner = rng.uniform(t_min + 1e-5, t_max - 1e-5, n_nodes - 2)
    nodes = np.sort(np.concatenate([[t_min, t_max], inner]))
    return nodes

NODE_GENERATORS = {
    "equidistant": equidistant_nodes,
    "chebyshev":   chebyshev_nodes,
    "random":      random_nodes,
}

# ─────────────────────────────────────────────
#  Vectorised interpolation kernels
#  All functions:
#    t_nodes : (n,)
#    Y       : (n, P)   — P pixels simultaneously
#    t_query : (Q,)
#  → returns  (Q, P)
# ─────────────────────────────────────────────

def lagrange_vec(t_nodes, Y, t_query):
    """Barycentric Lagrange — vectorised over all pixels at once."""
    n = len(t_nodes)
    w = np.ones(n)
    for j in range(n):
        diffs = t_nodes[j] - t_nodes
        diffs[j] = 1.0
        w[j] = 1.0 / np.prod(diffs)

    diffs = t_query[:, None] - t_nodes[None, :]   # (Q, n)
    exact = np.abs(diffs) < 1e-12
    safe_diffs = np.where(exact, 1.0, diffs)
    w_over_d = w[None, :] / safe_diffs             # (Q, n)
    num = w_over_d @ Y                             # (Q, P)
    den = w_over_d.sum(axis=1)                     # (Q,)
    result = num / den[:, None]

    hit_q, hit_n = np.where(exact)
    for q, ni in zip(hit_q, hit_n):
        result[q, :] = Y[ni, :]
    return result


def hermite_vec(t_nodes, Y, t_query):
    """Piecewise cubic Hermite — vectorised over all pixels at once."""
    n = len(t_nodes)
    dY = np.zeros_like(Y)
    dY[0]  = (Y[1]  - Y[0])  / (t_nodes[1]  - t_nodes[0])
    dY[-1] = (Y[-1] - Y[-2]) / (t_nodes[-1] - t_nodes[-2])
    for i in range(1, n - 1):
        dY[i] = (Y[i+1] - Y[i-1]) / (t_nodes[i+1] - t_nodes[i-1])

    idx = np.clip(np.searchsorted(t_nodes, t_query, side='right') - 1, 0, n-2)
    h = t_nodes[idx+1] - t_nodes[idx]
    s = (t_query - t_nodes[idx]) / h

    h00 =  2*s**3 - 3*s**2 + 1
    h10 =    s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 =    s**3 -   s**2

    return (h00[:,None]*Y[idx] + h10[:,None]*h[:,None]*dY[idx]
          + h01[:,None]*Y[idx+1] + h11[:,None]*h[:,None]*dY[idx+1])


def spline_vec(t_nodes, Y, t_query):
    """Cubic spline — batch across pixels in chunks."""
    P = Y.shape[1]
    Q = len(t_query)
    result = np.zeros((Q, P), dtype=np.float32)
    CHUNK = 4096
    for start in range(0, P, CHUNK):
        end = min(start + CHUNK, P)
        cs = CubicSpline(t_nodes, Y[:, start:end], axis=0, bc_type='not-a-knot')
        result[:, start:end] = cs(t_query)
    return result


INTERPOLATORS = {
    "lagrange": lagrange_vec,
    "hermite":  hermite_vec,
    "spline":   spline_vec,
}

# ─────────────────────────────────────────────
#  Frame I/O
# ─────────────────────────────────────────────

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def write_video(frames, out_path, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(np.clip(f, 0, 255).astype(np.uint8))
    out.release()

# ─────────────────────────────────────────────
#  Reconstruction pipeline
# ─────────────────────────────────────────────

def reconstruct(frames_all, t_all, t_nodes, interp_fn, desc=""):
    node_indices = sorted({int(np.argmin(np.abs(t_all - t))) for t in t_nodes})
    t_snapped = t_all[node_indices]
    known = np.stack([frames_all[i] for i in node_indices], axis=0).astype(np.float32)

    n, H, W, C = known.shape
    Q = len(t_all)
    P = H * W
    recon = np.zeros((Q, H, W, C), dtype=np.float32)

    for c in tqdm(range(C), desc=desc):
        Y = known[:, :, :, c].reshape(n, P)   # (n, P)
        out = interp_fn(t_snapped, Y, t_all)   # (Q, P)
        recon[:, :, :, c] = out.reshape(Q, H, W)

    return np.clip(recon, 0, 255)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--n_nodes", type=int, default=20)
    parser.add_argument("--methods", nargs="+",
                        default=["lagrange", "hermite", "spline"],
                        choices=list(INTERPOLATORS.keys()))
    parser.add_argument("--node_types", nargs="+",
                        default=["equidistant", "chebyshev", "random"],
                        choices=list(NODE_GENERATORS.keys()))
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Extracting frames from '{args.video}' …")
    frames, fps = extract_frames(args.video)
    n_frames = len(frames)
    print(f"      {n_frames} frames @ {fps:.2f} fps")

    t_all = np.arange(n_frames, dtype=float)
    t_min, t_max = t_all[0], t_all[-1]

    print("[2/3] Running reconstruction …")
    for node_type in args.node_types:
        t_nodes = NODE_GENERATORS[node_type](args.n_nodes, t_min, t_max)
        for method in args.methods:
            tag = f"{method}_{node_type}_n{args.n_nodes}"
            print(f"  → {tag}")
            recon = reconstruct(frames, t_all, t_nodes, INTERPOLATORS[method], desc=tag)
            out_path = os.path.join(args.outdir, f"{tag}.mp4")
            write_video([recon[i] for i in range(len(t_all))], out_path, fps)
            print(f"     saved: {out_path}")

    print("[3/3] Done.")


if __name__ == "__main__":
    main()
