from __future__ import annotations
import os
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from zstack import ZStack
from plane import Plane
from registration_utils import extract_zstack_plane, match_zstacks_2d


STACK_IN_FILE = "real_data_filtered_algo_VOLUMES_g.csv"
PLANE_OUT_FILE = f"{STACK_IN_FILE}".split(".csv")[0] + "_planes.csv"

USE_FLAT_PLANE = True
AREA_THRESHOLD = 50
MIN_ROI_NUMBER = 8
MAX_ATTEMPTS = 50
TRAIT_FOLDER = "transform_data"  # output folder for CSV export

# Plane generation params
plane_gen_params = {
    "read_filename": PLANE_OUT_FILE,
    "save_filename": PLANE_OUT_FILE,
    "anchor_intensity_threshold": 0,
    "align_intensity_threshold": 0,
    "z_threshold": 0,
    "max_tilt_deg": 30.0,
    "projection_dist_thresh": 0.5,
    "transform_intensity": "quantile",
    "plane_boundaries": [0, 1024, 0, 1024],
    "margin": 2,
    "match_anchors": True,
    "fixed_basis": True,
    "regenerate_planes": False,
    "max_alignments": 500,
    "z_guess": -1,
    "z_range": 0,
    "n_threads": 10,
    "anchor_dist_thresh": 100,
}

# Coarse match params (planes only)
match_plane_params = {
    "bin_match_params": {
        "min_matches": 2,
        "fixed_scale": 1,         # Using same scale at the moment
        "outlier_thresh": 2,
        "angle_tolerance" : 3,
        "mse_threshold": 0.3,
    },
    "traits": {
        "angle": {"weight": 0.35, "max_value": 0.1},
        "magnitude": {"weight": 0.35, "max_value": 0.8},
        "avg_radius": {"weight": 0.05, "max_value": 1},
        "circularity": {"weight": 0.05, "max_value": 0.1},
        "area": {"weight": 0.1, "max_value": 5},
    },
}

plane_list_params = {
    "min_score": 0,
    "max_matches": 2,
    "min_score_modifier": 1.0,
    "max_score_modifier": 1.0,
    "z_guess_a": -1,
    "z_guess_b": -1,
    "z_range": 1,
}

match_params = {
    "plane_gen_params": plane_gen_params,
    "match_plane_params": match_plane_params,
    "plane_list_params": plane_list_params,
    "planes_a_read_file": PLANE_OUT_FILE,
    "planes_b_read_file": None,
    "planes_a_write_file": PLANE_OUT_FILE,
    "planes_b_write_file": None,
    "plot_uoi": False,
    "plot_match": False,
    "use_gpu": True,
    "min_uoi": 0.9,
    "seg_params": {"method": "volume", "eps": 1.5, "min_samples": 5},
    "filter_params": {
        "disable_filtering": True,
        "min_area": 40,
        "max_area": 1000,
        "max_eccentricity": 0.69,
        "preserve_anchor": True,
    },
    "match_planes_only": True,   # Only coarse plane matches
}

# Binning resolution
TX_STEP = 5.0
TY_STEP = 5.0
ANG_STEP = 4.0
S_STEP = 0.1

# ---------- Helpers ----------
def norm_angle_deg(a: float) -> float:
    # Keep angles around zero so bins with ORIGIN=0 make sense
    # (-180, 180] mapping
    a = ((a + 180.0) % 360.0) - 180.0
    if a <= -180.0:
        a += 360.0
    return a

def bin_id(value: float, step: float) -> int:
    # Rigid, deterministic bin id with origin=0
    return math.floor(value / step)

def int_like_range(bin_idx: int, step: float) -> tuple[int, int]:
    """
    Inclusive integer display range: [lo, hi], e.g. tx id=0 -> 0..4 (for 5px),
    id=1 -> 5..9, id=-1 -> -5..-1, etc.
    """
    lo = int(bin_idx * step)
    hi = int(lo + step - 1)    # inclusive
    return lo, hi

def float_range(bin_idx: int, step: float) -> tuple[float, float]:
    """
    Half-open float range: [lo, hi), nice for scale printed to 1dp.
    """
    lo = bin_idx * step
    hi = lo + step
    return lo, hi

@dataclass(frozen=True)
class TransformIndexBin:
    tx_id: int
    ty_id: int
    ang_id: int
    s_id: int

    def label(self) -> str:
        tx_lo, tx_hi = int_like_range(self.tx_id, TX_STEP)
        ty_lo, ty_hi = int_like_range(self.ty_id, TY_STEP)
        ang_lo, ang_hi = int_like_range(self.ang_id, ANG_STEP)
        s_lo,  s_hi  = float_range(self.s_id,  S_STEP)
        # avoid "-0"
        def z(x): return 0 if abs(x) < 1e-12 else x
        return (f"tx={z(tx_lo)}–{z(tx_hi)}px | "
                f"ty={z(ty_lo)}–{z(ty_hi)}px | "
                f"θ={z(ang_lo)}–{z(ang_hi)}° | "
                f"s={s_lo:.1f}–{s_hi:.1f}")


def plot_zstack_planes_3d(zstack, title="Z-planes (ROI centroids)"):
    """
    Plot all ROIs in the zstack grouped by z-plane in 3D, 
    coloring by z-level with a colorbar.
    
    Args:
        zstack (ZStack): your loaded ZStack object
        title (str): plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = [], [], []
    for z in zstack.z_planes:
        rois = zstack.z_planes[z]
        for roi_id, roi_data in rois.items():
            coords = roi_data.get("coords", [])
            if not coords:
                continue
            arr = np.array(coords)
            cx, cy = arr[:,0].mean(), arr[:,1].mean()
            xs.append(cx)
            ys.append(cy)
            zs.append(z)

    if not xs:
        print("[WARN] No ROI centroids found to plot.")
        return

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    sc = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=20, alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Add colorbar by z
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Z-level")

    plt.tight_layout()
    plt.show()

def extract_transforms(matched_planes: Dict) -> List[Tuple[float, float, float, float]]:
    """
    For each matched plane pair, derive (tx, ty, angle_deg, scale).
    - angle_deg  := result['offset']  (deg)
    - scale      := result['scale_factor']
    - (tx, ty)   := from Plane.get_aligned_2d_projection(...), aligning anchors in local 2D
    """
    transforms = []
    for score, entry in matched_planes.items():
        res = entry.get("result", {})
        plane_a: Plane = entry["plane_a"]
        plane_b: Plane = entry["plane_b"]

        if not res or not res.get("match", False):
            continue

        angle_deg = float(res.get("offset", 0.0))
        scale = float(res.get("scale_factor", 1.0))

        # Compute tx, ty consistent with your plane 2D frames
        _, _, tinfo = plane_a.get_aligned_2d_projection(plane_b, angle_deg, scale)
        tx, ty = tinfo["translation"]  # already A-anchored minus B-anchored in A's 2D frame

        transforms.append((tx, ty, angle_deg, scale))

    return transforms

def bin_transforms_by_id(transforms):
    counter = Counter()
    for tx, ty, ang, sc in transforms:
        ang = norm_angle_deg(ang)   # keep around zero, but still origin=0 for bins
        key = TransformIndexBin(
            tx_id = bin_id(tx,  TX_STEP),
            ty_id = bin_id(ty,  TY_STEP),
            ang_id= bin_id(ang, ANG_STEP),
            s_id  = bin_id(sc,  S_STEP),
        )
        counter[key] += 1
    return counter

def plot_top_bins(counter: Counter, top_k: int = 10, title: str = "Top Transform Modes"):
    if not counter:
        print("[WARN] No transform bins to plot.")
        return

    top = counter.most_common(top_k)
    labels = [tb.label() for tb, _ in top]
    counts = [c for _, c in top]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(top)), counts)
    plt.xticks(range(len(top)), labels, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def export_counter_to_csv(counter, out_path: str):
    import csv, os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tx_id","tx_low_px","tx_high_px",
            "ty_id","ty_low_px","ty_high_px",
            "ang_id","ang_low_deg","ang_high_deg",
            "s_id","s_low","s_high","count"
        ])
        for key, cnt in counter.most_common():
            tx_lo, tx_hi = int_like_range(key.tx_id, TX_STEP)
            ty_lo, ty_hi = int_like_range(key.ty_id, TY_STEP)
            ag_lo, ag_hi = int_like_range(key.ang_id, ANG_STEP)
            s_lo,  s_hi  = float_range(key.s_id,  S_STEP)
            w.writerow([
                key.tx_id, tx_lo, tx_hi,
                key.ty_id, ty_lo, ty_hi,
                key.ang_id, ag_lo, ag_hi,
                key.s_id,  round(s_lo, 1), round(s_hi, 1),
                cnt
            ])


def is_flat_normal(n, max_tilt_deg=2.0):
    """
    True if normal is within max_tilt_deg of ±z. Handles sign and non-unit normals.
    """
    n = np.asarray(n, dtype=float)
    norm = np.linalg.norm(n)
    if norm == 0 or not np.isfinite(norm):
        return False
    cos_tilt = abs(n[2]) / norm
    cos_tilt = np.clip(cos_tilt, -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(cos_tilt))
    return tilt_deg <= max_tilt_deg

def scores_by_z_from_matches(matched_dict, aggregate="max", flat_tol_deg=2.0):
    z_to_scores = {}
    total = 0
    kept = 0
    for score_key, entry in matched_dict.items():
        plane_a = entry.get("plane_a")
        if plane_a is None:
            continue
        total += 1

        if not is_flat_normal(plane_a.normal, max_tilt_deg=flat_tol_deg):
            continue
        kept += 1

        # Use anchor z (round to int plane index if needed)
        z_level = int(round(float(plane_a.anchor_point.position[2])))

        # score might be the dict key or stored in entry
        try:
            s = float(score_key)
        except Exception:
            s = float(entry.get("score", np.nan))
        if not np.isfinite(s):
            continue

        z_to_scores.setdefault(z_level, []).append(s)

    if not z_to_scores:
        print(f"[WARN] No near-flat planes found (kept {kept} / {total} from matches); "
              f"try increasing flat_tol_deg from {flat_tol_deg} to e.g. 5–10°.")
        return {}

    if aggregate == "mean":
        return {z: float(np.mean(v)) for z, v in z_to_scores.items()}
    
    print(f"Total flat planes seen = {total}")

    return {z: float(np.max(v)) for z, v in z_to_scores.items()}

def plot_scores_vs_z(matched_dict, expected_z, aggregate="max", title="Plane-match score vs z"):
    """
    Scatter plot of score (y) vs z-level (x) for FLAT planes in zstack A.
    The z == expected_z point is highlighted; a vertical line marks expected_z.
    """
    z2s = scores_by_z_from_matches(matched_dict, aggregate=aggregate)
    if not z2s:
        print("[WARN] No flat-plane scores found to plot.")
        return
    
    print(z2s)

    # Sort by z for a clean plot
    zs = np.array(sorted(z2s.keys()))
    ys = np.array([z2s[z] for z in zs])

    plt.figure(figsize=(10, 5))
    # plot all points
    plt.scatter(zs, ys, s=30, label="flat z-planes", alpha=0.85)

    # highlight expected z (if present)
    exp_mask = (zs == int(round(expected_z)))
    if exp_mask.any():
        plt.scatter(zs[exp_mask], ys[exp_mask], s=80, marker="o", edgecolor="k",
                    label=f"expected z = {int(round(expected_z))}", zorder=3)
    # vertical guide at expected z regardless
    plt.axvline(int(round(expected_z)), linestyle="--", linewidth=1.2, label="expected z", alpha=0.6)

    plt.xlabel("z-level")
    plt.ylabel("match score")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    random.seed(10)
    start = time.perf_counter()

    # Load & plane-gen on full stack
    z_stack = ZStack(data=STACK_IN_FILE)
    # plot_zstack_planes_3d(z_stack) # debug
    z_stack.generate_random_intensities(0, 1000)
    z_stack.generate_planes_gpu(plane_gen_params)

    # Choose a suitable plane as source for a simulated "new stack"
    plane_ids = []
    for idx, plane in enumerate(z_stack.planes):
        if USE_FLAT_PLANE and is_flat_normal(plane.normal):
            plane_ids.append(idx)
        elif not USE_FLAT_PLANE and not np.allclose(plane.normal, [0, 0, 1], atol=1e-6):
            plane_ids.append(idx)

    print(f"num planes found = {len(plane_ids)}")
    # TEST
    true_z_levels = sorted(float(z) for z in z_stack.z_planes.keys())

    def snap_to_nearest_z(z, z_levels=true_z_levels):
        # returns the z-level key (float) with min absolute difference
        # assumes z_levels is sorted and small enough to scan
        return min(z_levels, key=lambda zz: abs(zz - float(z)))

    flat_ids_by_z = defaultdict(list)
    tilts = []

    for idx, pl in enumerate(z_stack.planes):
        if is_flat_normal(pl.normal, max_tilt_deg=2.0):
            z_anchor = float(pl.anchor_point.position[2])
            z_key = snap_to_nearest_z(z_anchor)
            flat_ids_by_z[z_key].append(idx)

            # keep some tilt stats for sanity
            n = np.asarray(pl.normal, float)
            nz = abs(n[2]) / (np.linalg.norm(n) + 1e-12)
            tilt = np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))
            tilts.append(tilt)

    total_flats = sum(len(v) for v in flat_ids_by_z.values())
    print(f"[DBG] flat planes total = {total_flats}, across {len(flat_ids_by_z)} z-slices")
    print(f"[DBG] tilt(deg) min/med/max = "
        f"{(min(tilts) if tilts else None):.3f} / "
        f"{(np.median(tilts) if tilts else None):.3f} / "
        f"{(max(tilts) if tilts else None):.3f}")

    # Optional: show per-z distribution (first few)
    for z in list(sorted(flat_ids_by_z.keys())):
        print(f"  z={int(z)} -> {len(flat_ids_by_z[z])} flat planes")

    # END TEST

    if not plane_ids:
        raise ValueError("No suitable planes found in the z-stack.")
    attempt = 0
    new_stack = None
    selected_idx = None

    while attempt < MAX_ATTEMPTS:
        # selected_idx = random.choice(plane_ids)
        # selected_z = random.choice(list(flat_ids_by_z.keys()))
        # selected_z = 30
        selected_z = 32
        # selected_z = 21
        # selected_z = 22
        # selected_z = 23
        selected_idx = random.choice(flat_ids_by_z[selected_z])
        selected_plane = z_stack.planes[selected_idx]

        temp_stack = extract_zstack_plane(
            z_stack,
            selected_plane,
            threshold=plane_gen_params["projection_dist_thresh"],
            method="volume",
        )

        mean_area = temp_stack._average_roi_area()
        if len(list(temp_stack.z_planes.keys())) > 0:
            z0 = list(temp_stack.z_planes.keys())[0]
            num_rois = len(list(temp_stack.z_planes[z0].keys()))
        else:
            num_rois = 0

        if mean_area > AREA_THRESHOLD and num_rois >= MIN_ROI_NUMBER:
            new_stack = temp_stack
            new_z = selected_plane.anchor_point.position[2] # extract z of anchor point
            print(f"Selected plane ID {selected_idx} with mean ROI area {mean_area:.2f} and anchor z_level = {new_z}")
            break

        attempt += 1

    if new_stack is None:
        raise RuntimeError(f"Could not find a suitable plane with avg ROI area > {AREA_THRESHOLD} after {MAX_ATTEMPTS} attempts.")

    # Generate planes inside the extracted "B" stack (no IO)
    plane_gen_params_b = dict(plane_gen_params)
    plane_gen_params_b.update({
        "read_filename": None,
        "save_filename": None,
        "align_intensity_threshold": 0.0,
        "anchor_intensity_threshold": 0.0,
        "regenerate_planes": True,
    })
    new_stack.generate_planes_gpu(plane_gen_params_b)

    # Restore plane-gen params for potential later calls
    plane_gen_params.update({
        "align_intensity_threshold": 0.4,
        "anchor_intensity_threshold": 0.5,
        "read_filename": PLANE_OUT_FILE,
        "save_filename": PLANE_OUT_FILE,
        "regenerate_planes": False,
    })

    # Match planes (coarse step only)
    matched = match_zstacks_2d(zstack_a=z_stack, zstack_b=new_stack, match_params=match_params)


    # Extract expected z from the sample plane used to build z_stack B
    expected_z = int(round(selected_plane.anchor_point.position[2]))

    # Plot score vs z and highlight the expected z-plane
    plot_scores_vs_z(matched, expected_z, aggregate="max",
                    title="Coarse match score vs z (flat planes)")

    # Extract and bin transforms
    transforms = extract_transforms(matched)
    print(f"[INFO] Extracted {len(transforms)} transforms.")

    counter = bin_transforms_by_id(transforms)
    os.makedirs(TRAIT_FOLDER, exist_ok=True)
    csv_path = os.path.join(TRAIT_FOLDER, f"transform_bins_{selected_idx}.csv")
    export_counter_to_csv(counter, csv_path)

    # Plot top-10
    plot_top_bins(counter, top_k=10, title="Top 10 Transform Modes (tx/ty/θ/scale)")

    end = time.perf_counter()
    print(f"[TIMER] Total run time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
