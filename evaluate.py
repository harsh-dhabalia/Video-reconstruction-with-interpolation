"""
Evaluation: compare reconstructed videos vs. original.
Computes PSNR (non-node frames only), SSIM, and motion smoothness.
"""

import os
import numpy as np
import cv2
from pathlib import Path


def load_frames(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f.astype(np.float32))
    cap.release()
    return np.stack(frames, axis=0)   # (N, H, W, 3)


def get_node_indices(n_frames: int, n_nodes: int, node_type: str) -> np.ndarray:
    """Reconstruct the same node indices used during reconstruction."""
    t_all  = np.arange(n_frames, dtype=float)
    t_min, t_max = t_all[0], t_all[-1]

    if node_type == "equidistant":
        t_nodes = np.linspace(t_min, t_max, n_nodes)
    elif node_type == "chebyshev":
        k = np.arange(1, n_nodes + 1)
        t_nodes = 0.5*(t_min+t_max) + 0.5*(t_max-t_min)*np.cos((2*k-1)*np.pi/(2*n_nodes))
        t_nodes = np.sort(t_nodes)
    elif node_type == "random":
        if n_nodes <= 2:
            t_nodes = np.linspace(t_min, t_max, n_nodes)
        else:
            rng = np.random.default_rng(42)
            inner = rng.uniform(t_min + 1e-5, t_max - 1e-5, n_nodes - 2)
            t_nodes = np.sort(np.concatenate([[t_min, t_max], inner]))
    else:
        raise ValueError(f"Unknown node type: {node_type}")

    # Snap to nearest actual frame indices (same logic as reconstruct.py)
    indices = sorted({int(np.argmin(np.abs(t_all - t))) for t in t_nodes})
    return np.array(indices)


def compute_psnr(original_frames: np.ndarray, recon_frames: np.ndarray,
                 node_indices: np.ndarray) -> float:
    """
    PSNR only on non-node frames (the ones actually reconstructed).
    Node frames are copied exactly from the original so excluding them
    gives a fairer, honest score.
    """
    mask = np.ones(len(original_frames), dtype=bool)
    mask[node_indices] = False          # exclude node frames

    orig  = original_frames[mask].astype(float)
    recon = recon_frames[mask].astype(float)

    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255 ** 2) / mse)


def compute_ssim(orig: np.ndarray, recon: np.ndarray,
                 node_indices: np.ndarray) -> float:
    """Simplified SSIM on non-node frames only."""
    mask = np.ones(len(orig), dtype=bool)
    mask[node_indices] = False

    vals = []
    for o, r in zip(orig[mask], recon[mask]):
        og = cv2.cvtColor(o.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
        rg = cv2.cvtColor(r.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
        C1, C2 = (0.01*255)**2, (0.03*255)**2
        mu1, mu2 = og.mean(), rg.mean()
        s1  = og.std()**2
        s2  = rg.std()**2
        s12 = np.mean((og - mu1) * (rg - mu2))
        vals.append(((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2)))
    return float(np.mean(vals)) if vals else 0.0


def motion_smoothness(frames: np.ndarray) -> float:
    """Mean absolute difference between consecutive frames — lower = smoother."""
    diffs = np.abs(frames[1:].astype(float) - frames[:-1].astype(float))
    return float(diffs.mean())


def parse_tag(filename: str):
    """Extract method, node_type, n_nodes from filename like 'spline_chebyshev_n20'."""
    stem = Path(filename).stem          # e.g. spline_chebyshev_n20
    parts = stem.split("_")
    try:
        n_nodes = int(parts[-1].lstrip("n"))
        node_type = parts[-2]
        method = "_".join(parts[:-2])
        return method, node_type, n_nodes
    except Exception:
        return None, None, None


def evaluate(original_path: str, recon_dir: str = "output"):
    print(f"\n{'='*70}")
    print(f"Evaluation  (PSNR on reconstructed frames only)")
    print(f"Original : {original_path}")
    print(f"{'='*70}")

    orig_frames = load_frames(original_path)
    n_frames    = len(orig_frames)
    orig_smooth = motion_smoothness(orig_frames)

    recon_paths = sorted(Path(recon_dir).glob("*.mp4"))
    if not recon_paths:
        print("No reconstructed videos found in", recon_dir)
        return

    print(f"\n{'Tag':<42} {'PSNR':>8} {'SSIM':>8} {'Smooth':>10} {'#nodes':>7}")
    print("-" * 78)

    results = {}
    for rp in recon_paths:
        method, node_type, n_nodes = parse_tag(rp.name)
        if node_type is None:
            print(f"  Skipping (unrecognised filename): {rp.name}")
            continue

        recon_frames  = load_frames(str(rp))
        node_indices  = get_node_indices(n_frames, n_nodes, node_type)

        # Trim to same length in case of rounding
        n = min(len(orig_frames), len(recon_frames))
        if orig_frames.shape[1:3] != recon_frames.shape[1:3]:
            # Skip if spatial dimensions don't match
            continue
        psnr_val = compute_psnr(orig_frames[:n], recon_frames[:n], node_indices)
        ssim_val = compute_ssim(orig_frames[:n], recon_frames[:n], node_indices)
        smooth   = motion_smoothness(recon_frames[:n])

        tag = rp.stem
        results[tag] = dict(psnr=psnr_val, ssim=ssim_val, smoothness=smooth)
        print(f"{tag:<42} {psnr_val:>8.2f} {ssim_val:>8.4f} {smooth:>10.3f} {n_nodes:>7}")

    print(f"\n{'Original (reference)':<42} {'—':>8} {'—':>8} {orig_smooth:>10.3f}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("original")
    p.add_argument("--recon_dir", default="output")
    args = p.parse_args()
    evaluate(args.original, args.recon_dir)
