"""
Thorough verification script for the video reconstruction project.
Tests every component against the real chewie.mp4 video.
"""
import sys
import numpy as np
from pathlib import Path

# Make sure we import from the project directory
sys.path.insert(0, str(Path(__file__).parent))

from reconstruct import (
    equidistant_nodes, chebyshev_nodes, random_nodes,
    lagrange_vec, hermite_vec, spline_vec,
    extract_frames, reconstruct, write_video,
    NODE_GENERATORS, INTERPOLATORS,
)
from evaluate import get_node_indices, parse_tag

VIDEO = "chewie.mp4"
N_NODES = 20
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
failures = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  —  {detail}")
        failures.append(name)


print("=" * 70)
print("THOROUGH VERIFICATION — chewie.mp4")
print("=" * 70)

# ── 1. Video loading ────────────────────────────────────────────
print("\n[1] VIDEO LOADING")
frames, fps = extract_frames(VIDEO)
n_frames = len(frames)
check("Video loads successfully", n_frames > 0)
check(f"Frame count is reasonable ({n_frames})", 100 <= n_frames <= 300)
check(f"FPS is reasonable ({fps})", 20 <= fps <= 60)
H, W, C = frames[0].shape
check(f"Frame shape ({H}x{W}x{C})", C == 3 and H > 0 and W > 0)

t_all = np.arange(n_frames, dtype=float)
t_min, t_max = t_all[0], t_all[-1]

# ── 2. Node generation ─────────────────────────────────────────
print("\n[2] NODE GENERATION")

for name, gen_fn in NODE_GENERATORS.items():
    nodes = gen_fn(N_NODES, t_min, t_max)
    check(f"{name}: returns {len(nodes)} nodes (expected {N_NODES})",
          len(nodes) == N_NODES)
    check(f"{name}: sorted",
          np.all(np.diff(nodes) >= 0))
    check(f"{name}: includes t_min ({t_min})",
          np.isclose(nodes[0], t_min, atol=0.5),
          f"first node = {nodes[0]}")
    check(f"{name}: includes t_max ({t_max})",
          np.isclose(nodes[-1], t_max, atol=0.5),
          f"last node = {nodes[-1]}")
    check(f"{name}: all within [t_min, t_max]",
          np.all(nodes >= t_min - 1e-6) and np.all(nodes <= t_max + 1e-6))
    # check no duplicates after snapping
    snapped = sorted({int(np.argmin(np.abs(t_all - t))) for t in nodes})
    check(f"{name}: {len(snapped)} unique snapped indices (expected ~{N_NODES})",
          len(snapped) >= N_NODES - 2,  # small tolerance for coincidental snaps
          f"got {len(snapped)}")

# ── 3. Node consistency between reconstruct.py and evaluate.py ─
print("\n[3] RECONSTRUCT ↔ EVALUATE NODE CONSISTENCY")

for node_type in ["equidistant", "chebyshev", "random"]:
    # What reconstruct.py would compute
    r_nodes = NODE_GENERATORS[node_type](N_NODES, t_min, t_max)
    r_snapped = sorted({int(np.argmin(np.abs(t_all - t))) for t in r_nodes})

    # What evaluate.py would compute
    e_indices = get_node_indices(n_frames, N_NODES, node_type)

    check(f"{node_type}: indices match between reconstruct & evaluate",
          list(r_snapped) == list(e_indices),
          f"recon={r_snapped[:5]}... eval={list(e_indices[:5])}...")

# ── 4. Interpolation correctness ───────────────────────────────
print("\n[4] INTERPOLATION CORRECTNESS (small test)")

# Use a small subset of pixels for a quick sanity check
test_nodes = equidistant_nodes(10, t_min, t_max)
snapped_idx = sorted({int(np.argmin(np.abs(t_all - t))) for t in test_nodes})
t_snapped = t_all[snapped_idx]

# Build a small Y matrix: just 100 pixels from the first channel
known_frames = np.stack([frames[i] for i in snapped_idx], axis=0).astype(np.float32)
Y_small = known_frames[:, :, :, 0].reshape(len(snapped_idx), -1)[:, :100]  # (n, 100)

for iname, interp_fn in INTERPOLATORS.items():
    result = interp_fn(t_snapped, Y_small, t_snapped)  # query AT the nodes
    max_err = np.max(np.abs(result - Y_small))
    check(f"{iname}: exact at node points (max err = {max_err:.6f})",
          max_err < 1.0,  # within 1 brightness level
          f"max_err={max_err:.2f}")

# ── 5. Full reconstruction sanity (single combo) ──────────────
print("\n[5] FULL RECONSTRUCTION SANITY (spline + equidistant)")

test_t_nodes = equidistant_nodes(N_NODES, t_min, t_max)
recon = reconstruct(frames, t_all, test_t_nodes, spline_vec, desc="verify")
check(f"Output shape = ({len(t_all)}, {H}, {W}, {C})",
      recon.shape == (len(t_all), H, W, C))
check("All values in [0, 255]",
      np.all(recon >= 0) and np.all(recon <= 255))

# Check that node frames are reproduced exactly
node_idx = sorted({int(np.argmin(np.abs(t_all - t))) for t in test_t_nodes})
for i in node_idx[:3]:  # spot-check first 3
    orig = frames[i].astype(np.float32)
    reconstructed = recon[i]
    max_diff = np.max(np.abs(orig - reconstructed))
    check(f"Node frame {i} reproduced (max diff = {max_diff:.2f})",
          max_diff < 2.0,
          f"max_diff={max_diff:.2f}")

# ── 6. Output video file check ─────────────────────────────────
print("\n[6] OUTPUT VIDEO FILE CHECK")
output_dir = Path("output")
expected_files = [
    f"{m}_{nt}_n{N_NODES}.mp4"
    for nt in ["equidistant", "chebyshev", "random"]
    for m in ["lagrange", "hermite", "spline"]
]
for f in expected_files:
    exists = (output_dir / f).exists()
    size = (output_dir / f).stat().st_size if exists else 0
    check(f"{f} exists and non-empty",
          exists and size > 1000,
          f"exists={exists}, size={size}")

# ── 7. parse_tag validation ────────────────────────────────────
print("\n[7] FILENAME PARSER (parse_tag)")
test_cases = [
    ("spline_chebyshev_n20.mp4", ("spline", "chebyshev", 20)),
    ("lagrange_equidistant_n20.mp4", ("lagrange", "equidistant", 20)),
    ("hermite_random_n15.mp4", ("hermite", "random", 15)),
]
for fname, expected in test_cases:
    result = parse_tag(fname)
    check(f"parse_tag('{fname}') → {result}",
          result == expected,
          f"expected {expected}")

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 70)
if failures:
    print(f"\033[91m{len(failures)} FAILURE(S):\033[0m")
    for f in failures:
        print(f"  • {f}")
else:
    print(f"\033[92mALL CHECKS PASSED\033[0m")
print("=" * 70)
