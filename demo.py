"""
demo.py  –  Generate a synthetic 5-second test video, then run the full
            reconstruction pipeline so everything can be verified without
            uploading a real clip.

Synthetic scene: a coloured disk that sweeps across the frame (left→right)
while its colour oscillates.  This gives enough temporal variation to
stress-test the interpolation methods.
"""

import numpy as np
import cv2
import os
from pathlib import Path

# ── Parameters ──────────────────────────────────────────────────
FPS        = 24
DURATION   = 5       # seconds
W, H       = 320, 240
OUT_VIDEO  = "test_input.mp4"
N_NODES    = 15      # sampled frames per reconstruction run
# ────────────────────────────────────────────────────────────────


def make_synthetic_video(path: str):
    """Create a simple moving-disk video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    n_frames = int(FPS * DURATION)
    for i in range(n_frames):
        t = i / (n_frames - 1)           # normalised time in [0, 1]
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Background gradient
        frame[:, :, 0] = np.clip(30 + 40 * np.sin(2 * np.pi * t), 0, 255).astype(np.uint8)

        # Moving disk
        cx = int(W * 0.1 + W * 0.8 * t)
        cy = int(H / 2 + H * 0.2 * np.sin(4 * np.pi * t))
        r  = 30
        color = (
            int(np.clip(80  + 120 * np.cos(2 * np.pi * t), 0, 255)),
            int(np.clip(200 - 80  * np.sin(2 * np.pi * t), 0, 255)),
            int(np.clip(160 + 80  * np.sin(4 * np.pi * t), 0, 255)),
        )
        cv2.circle(frame, (cx, cy), r, color, -1)

        # Second smaller disk
        cx2 = int(W * 0.9 - W * 0.7 * t)
        cy2 = int(H * 0.3 + H * 0.3 * np.cos(3 * np.pi * t))
        color2 = (
            int(np.clip(220 - 80 * t, 0, 255)),
            int(np.clip(100 + 100 * t, 0, 255)),
            int(np.clip(80  + 120 * np.sin(np.pi * t), 0, 255)),
        )
        cv2.circle(frame, (cx2, cy2), 20, color2, -1)

        writer.write(frame)
    writer.release()
    print(f"Synthetic video saved → {path}  ({n_frames} frames @ {FPS} fps)")


def run_pipeline():
    Path("output").mkdir(exist_ok=True)

    # 1. Generate test clip
    make_synthetic_video(OUT_VIDEO)

    # 2. Import pipeline components
    from reconstruct import (
        extract_frames, write_video,
        equidistant_nodes, chebyshev_nodes, random_nodes,
        lagrange_vec, hermite_vec, spline_vec,
        reconstruct,
    )

    frames, fps = extract_frames(OUT_VIDEO)
    t_all  = np.arange(len(frames), dtype=float)
    t_min, t_max = t_all[0], t_all[-1]

    NODE_GEN = {
        "equidistant": equidistant_nodes,
        "chebyshev":   chebyshev_nodes,
        "random":      random_nodes,
    }
    METHODS = {
        "lagrange": lagrange_vec,
        "hermite":  hermite_vec,
        "spline":   spline_vec,
    }

    for node_type, gen_fn in NODE_GEN.items():
        t_nodes = gen_fn(N_NODES, t_min, t_max)
        for method, interp_fn in METHODS.items():
            tag = f"{method}_{node_type}_n{N_NODES}"
            print(f"\nProcessing: {tag}")
            recon = reconstruct(frames, t_all, t_nodes, interp_fn, desc=tag)
            out_path = f"output/{tag}.mp4"
            write_video([recon[i] for i in range(len(t_all))], out_path, fps)
            print(f"  → {out_path}")

    # 3. Evaluate
    from evaluate import evaluate
    evaluate(OUT_VIDEO, "output")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run_pipeline()
