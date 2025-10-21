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
from collections import  deque
from itertools import product
from shapely.geometry import Polygon as ShPolygon
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon as ShMultiPoly
from scipy.spatial import cKDTree

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
    "max_alignments": 100,
    "combination_max_planes": 1000,       # hard cap on number of sub-planes produced per anchor plane
    "combination_sampling": "deterministic",  # "deterministic" | "random"
    "projection_dist_thresh": 0.5,
    "transform_intensity": "quantile",
    "plane_boundaries": [0, 1024, 0, 1024],
    "margin": 2,
    "match_anchors": True,
    "anchor_dist_thresh": None,
    "fixed_basis": True,
    "regenerate_planes": False,
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
        "angle": {"weight": 0.45, "max_value": 0.1},
        "magnitude": {"weight": 0.35, "max_value": 10},
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

# Debug helpers
from matplotlib.collections import LineCollection

def _anchor_id_only(plane: Plane) -> int | str:
    """
    Return the 'ID' portion of the anchor (not the z). Your Plane.anchor_point.id
    often looks like (z, id).
    """
    aid = plane.anchor_point.id
    if isinstance(aid, (tuple, list)) and len(aid) >= 2:
        return aid[1]
    return aid

def _anchor_z_int(plane: Plane) -> int:
    return int(round(float(plane.anchor_point.position[2])))

def _proj_points_local(plane: Plane) -> dict:
    """
    Return the plane's local 2D coords keyed by the *global* ROI id.
    If Plane.get_local_2d_coordinates() is keyed by per-plane index,
    remap those to PlanePoint.id; otherwise pass keys through.
    """
    local = plane.get_local_2d_coordinates()  # likely keyed by per-plane index (0,1,2,...)
    out = {}

    # Fast path: if keys match plane.plane_points indices, map to .id
    if isinstance(plane.plane_points, dict):
        for k, xy in local.items():
            if k in plane.plane_points:          # per-plane index -> global id
                gid = plane.plane_points[k].id
            else:                                 # already a global id (e.g. (z, roi_id))
                gid = k
            out[gid] = xy
        return out

    # Fallback (shouldn't happen): just return as-is
    return local

def _nearest_pairs_2d(A_pts: np.ndarray, B_pts: np.ndarray) -> list[tuple[int,int]]:
    """
    Return 1-to-1 matches of A->nearest B by centroid (greedy).
    A_pts/B_pts: (N,2)/(M,2)
    """
    if A_pts.size == 0 or B_pts.size == 0:
        return []
    tree = cKDTree(B_pts)
    dists, idxs = tree.query(A_pts, k=1)
    usedB = set()
    pairs = []
    for iA, jB in enumerate(idxs):
        if jB in usedB:
            continue
        usedB.add(jB)
        pairs.append((iA, jB))
    return pairs

def _plot_match_2d(ax, A_xy, A_ids, B_xy, B_ids, pairs,
                   polysA=None, polysB=None,
                   title=None, colorA="C0", colorB="C1", colorI="purple"):
    """
    Original behaviour (single axes) if polysA/polysB are None.

    If polysA & polysB are provided (dicts {roi_id: Nx2 points} in A-frame),
    this function internally creates a 1x2 subplot:
      - Left: original centroid/labels view (what this function used to draw)
      - Right: all ROI polygons (A solid, B dashed) with intersections
               for matched pairs filled.

    Args
    ----
    ax      : matplotlib Axes to draw on (kept for backward compatibility).
              If polys provided, this 'ax' becomes the LEFT subplot.
    A_xy    : (nA,2) ndarray of A points (usually centroids)
    A_ids   : list/array of A ROI ids aligned with A_xy rows
    B_xy    : (nB,2) ndarray of B points already transformed into A frame
    B_ids   : list/array of B ROI ids aligned with B_xy rows
    pairs   : list of (iA, jB) index pairs
    polysA  : optional dict {roi_id: Nx2 array-like} (A polygons in A frame)
    polysB  : optional dict {roi_id: Nx2 array-like} (B polygons in A frame)
    """

    def _draw_centroids(axc):
        axc.scatter(A_xy[:,0], A_xy[:,1], s=25, marker='o', label='A (local)', alpha=0.9)
        axc.scatter(B_xy[:,0], B_xy[:,1], s=25, marker='^', label='B→A (transformed)', alpha=0.9)

        matched_A = set(i for i,_ in pairs)
        matched_B = set(j for _,j in pairs)

        # optional: lines between matches
        segs = [[A_xy[i], B_xy[j]] for i,j in pairs]
        if segs:
            lc = LineCollection(segs, linewidths=0.6, alpha=0.4)
            axc.add_collection(lc)

        # labels
        for i, (x,y) in enumerate(A_xy):
            if i in matched_A:
                j = next(j for ii,j in pairs if ii==i)
                axc.text(x, y, f"A{A_ids[i]}<>B{B_ids[j]}", fontsize=7, ha='left', va='bottom')
            else:
                axc.text(x, y, f"A{A_ids[i]}", fontsize=7, ha='left', va='bottom')

        for j, (x,y) in enumerate(B_xy):
            if j not in matched_B:
                axc.text(x, y, f"B{B_ids[j]}", fontsize=7, ha='left', va='bottom')

        axc.set_aspect('equal', adjustable='box')
        axc.grid(alpha=0.2)
        axc.legend(loc='best')

    # --- No polygons: keep legacy single-axes behaviour
    if polysA is None or polysB is None:
        _draw_centroids(ax)
        return

    # --- Polygons provided: make a twin subplot while preserving existing 'ax'
    fig = ax.figure
    fig.clf()  # clear and rebuild the 1x2 layout using the same figure
    axL, axR = fig.subplots(1, 2, sharex=True, sharey=True)
    if title:
        fig.suptitle(title)

    # Left: original view
    _draw_centroids(axL)
    axL.set_title("Centroids & pair links")

    # Right: polygons + intersections
    axR.set_title("ROI polygons with matched intersections")
    axR.set_aspect("equal", adjustable="box")
    axR.grid(alpha=0.2)

    # Draw all A polygons (solid)
    for rid, pts in polysA.items():
        pts = np.asarray(pts)
        if pts.shape[0] >= 3:
            axR.add_patch(MplPolygon(pts, closed=True, fill=False,
                                     edgecolor=colorA, linewidth=1.3, alpha=0.9))

    # Draw all B polygons (dashed)
    for rid, pts in polysB.items():
        pts = np.asarray(pts)
        if pts.shape[0] >= 3:
            axR.add_patch(MplPolygon(pts, closed=True, fill=False, linestyle="--",
                                     edgecolor=colorB, linewidth=1.1, alpha=0.9))

    # Intersections for matched pairs
    A_idx_to_id = {i: A_ids[i] for i in range(len(A_ids))}
    B_idx_to_id = {j: B_ids[j] for j in range(len(B_ids))}

    def _fill_shapely(poly, axp, fc, ec="none", alpha=0.35, lw=0.0):
        if isinstance(poly, ShMultiPoly):
            for p in poly.geoms:
                x, y = p.exterior.xy
                axp.fill(x, y, facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=lw)
        else:
            x, y = poly.exterior.xy
            axp.fill(x, y, facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=lw)

    for iA, jB in pairs:
        ridA = A_idx_to_id[iA]
        ridB = B_idx_to_id[jB]
        ptsA = np.asarray(polysA.get(ridA, []))
        ptsB = np.asarray(polysB.get(ridB, []))
        if ptsA.shape[0] < 3 or ptsB.shape[0] < 3:
            continue

        polyA = ShPolygon(ptsA);  polyB = ShPolygon(ptsB)
        if not polyA.is_valid: polyA = polyA.buffer(0)
        if not polyB.is_valid: polyB = polyB.buffer(0)
        if not polyA.is_valid or not polyB.is_valid:  # still invalid, skip
            continue

        inter = polyA.intersection(polyB)
        if inter.is_empty:
            continue

        _fill_shapely(inter, axR, fc=colorI, alpha=0.35)
        # optional label on intersection
        try:
            cx, cy = inter.representative_point().xy
            axR.text(cx[0], cy[0], f"A{ridA}∩B{ridB}", fontsize=7, color=colorI)
        except Exception:
            pass

    # Legend proxies
    A_proxy = plt.Line2D([0],[0], color=colorA, lw=1.5, label="A ROI")
    B_proxy = plt.Line2D([0],[0], color=colorB, lw=1.2, ls="--", label="B ROI (transformed)")
    I_proxy = plt.Line2D([0],[0], color=colorI, lw=6, alpha=0.35, label="Intersection")
    axR.legend(handles=[A_proxy, B_proxy, I_proxy], loc="best", fontsize=8)

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


def _visualize_filtered_matches(matched_dict: Dict,
                                zstack_a: ZStack,
                                zstack_b: ZStack,
                                same_z_level: bool = True,
                                same_anchor_id: bool = True):
    """
    For each match passing the filters, plot:
      - Left: A centroids + B centroids (transformed into A), labels 'Axx<>Byy'
      - Right: all ROI polygons (A solid, B dashed) with shaded intersections.

    Assumes (common in your pipeline) these planes are flat slices so polygons
    can be taken from z-plane boundaries and transformed via 2D R·S + T.
    """

    def _anchor_z_int(pl: Plane) -> int:
        return int(round(float(pl.anchor_point.position[2])))

    def _anchor_id_only(pl: Plane):
        # keep whatever your existing convention is; often plane_point.id is (z, rid)
        pid = pl.anchor_point.id
        return pid[1] if isinstance(pid, (tuple, list)) and len(pid) >= 2 else pid

    def _rotation_matrix(deg: float) -> np.ndarray:
        th = np.deg2rad(deg)
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s], [s, c]], dtype=float)

    def _build_polys_for_z(zstack: ZStack, z_level: int) -> dict:
        """Return {roi_id: Nx2 np.array} from zstack’s raw coords at z_level."""
        out = {}
        layer = zstack.z_planes.get(float(z_level))
        if not layer:
            return out
        for rid, rdata in layer.items():
            coords = rdata.get("coords", [])
            if not coords or len(coords) < 3:
                continue
            arr = np.asarray(coords, float)
            # coords may be Nx2 already; ensure Nx2
            if arr.ndim == 2 and arr.shape[1] >= 2:
                out[rid] = arr[:, :2]
        return out

    def _transform_polys(polys: dict, R: np.ndarray, scale: float, t: np.ndarray) -> dict:
        """Apply p' = R @ (scale * p) + t to every polygon point."""
        out = {}
        for rid, xy in polys.items():
            P = (R @ (scale * xy.T)).T + t  # (N,2)
            out[rid] = P
        return out

    # Collect candidate matches after filters
    candidates = []
    for k, entry in matched_dict.items():
        res = entry.get("result", {})
        if not res or not res.get("match", False):
            continue
        pa: Plane = entry.get("plane_a")
        pb: Plane = entry.get("plane_b")
        if not pa or not pb:
            continue

        if same_z_level and _anchor_z_int(pa) != _anchor_z_int(pb):
            continue
        if same_anchor_id and _anchor_id_only(pa) != _anchor_id_only(pb):
            continue

        ang = float(res.get("offset", 0.0))
        sc  = float(res.get("scale_factor", 1.0))

        # Compute the intended global translation (anchor-to-anchor) after R·S:
        R = _rotation_matrix(ang)
        anchorA = np.array(pa.anchor_point.position[:2], float)
        anchorB = np.array(pb.anchor_point.position[:2], float)
        t = anchorA - (R @ (sc * anchorB))

        # Score
        try:
            score = float(k)
        except Exception:
            score = float(res.get("score", 0.0))

        print(entry)

        candidates.append((pa, pb, ang, sc, t, score))

    if not candidates:
        print("[INFO] No matches satisfied the filters.")
        return

    for idx, (pa, pb, ang, sc, t, score) in enumerate(candidates, start=1):
        zA = _anchor_z_int(pa)
        zB = _anchor_z_int(pb)

        # Build polygons in A frame
        polysA = _build_polys_for_z(zstack_a, zA)                  # already in A’s global XY
        polysB_raw = _build_polys_for_z(zstack_b, zB)              # B in its own global XY
        R = _rotation_matrix(ang)
        t = np.asarray(t, float).reshape(1, 2)
        polysB = _transform_polys(polysB_raw, R, sc, t)            # into A frame

        if not polysA or not polysB:
            print(f"[WARN] Missing polygons at zA={zA}, zB={zB} for match {idx}")
            continue

        # Centroids (in A frame) for pairing/labels
        A_ids = list(polysA.keys())
        B_ids = list(polysB.keys())
        A_xy  = np.array([polysA[rid].mean(axis=0) for rid in A_ids], float)
        B_xy  = np.array([polysB[rid].mean(axis=0) for rid in B_ids], float)

        # Nearest-neighbour pairs for labeling on the centroid plot
        pr = _nearest_pairs_2d(A_xy, B_xy)  # expects (iA, jB) indexes

        # Optional IoU line (uses your existing helper on raw stacks/planes)
        tx, ty = float(t[0,0]), float(t[0,1])
        iou0 = compute_plane_iou(zstack_a, zstack_b, pa, pb, tx, ty, ang, sc)

        # Draw using your enhanced plotter (creates a 1x2 view internally)
        fig, ax = plt.subplots(figsize=(7, 6))
        _plot_match_2d(
            ax, A_xy, A_ids, B_xy, B_ids, pr,
            polysA=polysA, polysB=polysB,
            title=(f"Match {idx}: "
                   f"A(z={zA}, anc={_anchor_id_only(pa)}) vs "
                   f"B(z={zB}, anc={_anchor_id_only(pb)})  |  "
                   f"score={score:.3f}, R={ang:.1f}°, S={sc:.3f}, T=({tx:.1f},{ty:.1f})\n"
                   f"IoU={iou0:.3f}")
        )
        plt.show()


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
    tx_id : int
    ty_id : int
    ang_id :int
    s_id : int

    tx_step : int
    ty_step : int
    ang_step : int
    s_step :int

    def label(self) -> str:
        tx_lo, tx_hi = int_like_range(self.tx_id, self.tx_step)
        ty_lo, ty_hi = int_like_range(self.ty_id, self.ty_step)
        ang_lo, ang_hi = int_like_range(self.ang_id, self.ang_step)
        s_lo,  s_hi  = float_range(self.s_id,  self.s_step)
        # avoid "-0"
        def z(x): return 0 if abs(x) < 1e-12 else x
        return (f"tx={z(tx_lo)}–{z(tx_hi)}px | "
                f"ty={z(ty_lo)}–{z(ty_hi)}px | "
                f"θ={z(ang_lo)}–{z(ang_hi)}° | "
                f"s={s_lo:.1f}–{s_hi:.1f}")


@dataclass
class ZScoreParams:
    w_count: float = 1.0
    w_mean:  float = 1.0
    w_max:   float = 1.0

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

def extract_transforms(matched_planes: Dict) -> List[TransformMatch]:
    """
    For each matched plane pair, derive a full TransformMatch:
      - tx, ty: translation in A's local 2D
      - ang_deg: rotation (degrees)
      - scale: scale factor
      - score: match score from the coarse step
      - z_a, z_b: anchor z-levels (rounded ints)
      - key_bin: TransformIndexBin computed with (TX_STEP, TY_STEP, ANG_STEP, S_STEP)
    """
    out = []
    for score_key, entry in matched_planes.items():
        res = entry.get("result", {})
        plane_a: Plane = entry.get("plane_a")
        plane_b: Plane = entry.get("plane_b")
        if not plane_a or not plane_b:
            continue
        if not res or not res.get("match", False):
            continue

        # score might be the dict key or stored in entry
        try:
            score = float(score_key)
        except Exception:
            score = float(res.get("score", np.nan))
        if not np.isfinite(score):
            continue

        ang = float(res.get("offset", 0.0))
        sc  = float(res.get("scale_factor", 1.0))
        _, _, tinfo = plane_a.get_aligned_2d_projection(plane_b, ang, sc)
        tx, ty = tinfo["translation"]

        # bin key (rigid)
        ang_n = norm_angle_deg(ang)
        key = TransformIndexBin(
            tx_id = bin_id(tx,  TX_STEP),
            ty_id = bin_id(ty,  TY_STEP),
            ang_id= bin_id(ang_n, ANG_STEP),
            s_id  = bin_id(sc,  S_STEP),
            tx_step=TX_STEP, ty_step=TY_STEP, ang_step=ANG_STEP, s_step=S_STEP
        )

        out.append(TransformMatch(
            tx=tx, ty=ty, ang_deg=ang, scale=sc, score=score,
            plane_a=plane_a, plane_b=plane_b,
            z_a=int(round(float(plane_a.anchor_point.position[2]))),
            z_b=int(round(float(plane_b.anchor_point.position[2]))),
            key_bin=key
        ))
    return out

def bin_transforms_by_id(items):
    """
    Accepts a list of TransformMatch OR a list of (tx, ty, angle_deg, scale) tuples.
    Returns Counter[TransformIndexBin] as before.
    """
    counter = Counter()
    for it in items:
        if isinstance(it, TransformMatch):
            tx, ty, ang, sc = it.tx, it.ty, it.ang_deg, it.scale
        else:
            # assume tuple-like
            tx, ty, ang, sc = it

        ang_n = norm_angle_deg(ang)
        key = TransformIndexBin(
            tx_id = bin_id(tx,  TX_STEP),
            ty_id = bin_id(ty,  TY_STEP),
            ang_id= bin_id(ang_n, ANG_STEP),
            s_id  = bin_id(sc,  S_STEP),
            tx_step=TX_STEP, ty_step=TY_STEP, ang_step=ANG_STEP, s_step=S_STEP
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


ANG_PERIOD_IDS = int(round(360.0 / ANG_STEP))  # e.g., 360/4 -> 90 angle bins

# -- helpers for best plane picks --
def most_represented_bucket(counter: Counter):
    """Return (bucket_key, count)."""
    if not counter:
        return None, 0
    return counter.most_common(1)[0]

def _match_in_bin(m: TransformMatch, bucket) -> bool:
    # support TransformIndexBin or MergedIndexRange
    if isinstance(bucket, TransformIndexBin):
        return (bin_id(m.tx, TX_STEP)  == bucket.tx_id and
                bin_id(m.ty, TY_STEP)  == bucket.ty_id and
                bin_id(norm_angle_deg(m.ang_deg), ANG_STEP) == bucket.ang_id and
                bin_id(m.scale, S_STEP) == bucket.s_id)

    # MergedIndexRange: ranges per axis (inclusive in bin-IDs)
    if isinstance(bucket, MergedIndexRange):
        tx_id = bin_id(m.tx, TX_STEP)
        ty_id = bin_id(m.ty, TY_STEP)
        s_id  = bin_id(m.scale, S_STEP)
        ang_id= bin_id(norm_angle_deg(m.ang_deg), ANG_STEP) % ANG_PERIOD_IDS

        if not (bucket.tx_min_id <= tx_id <= bucket.tx_max_id): return False
        if not (bucket.ty_min_id <= ty_id <= bucket.ty_max_id): return False
        if not (bucket.s_min_id  <= s_id  <= bucket.s_max_id):  return False

        # angle: inside circular span?
        span_ids = angle_ids_from_span(bucket.ang_start_id, bucket.ang_len)
        return (ang_id % ANG_PERIOD_IDS) in span_ids

    # unknown bucket type
    return False

def score_z_levels(matches, params: ZScoreParams, which="a"):
    """
    Compute weighted z-scores for either zstack A or B.

    Args:
        matches: list of TransformMatch objects
        params:  ZScoreParams (weights for count, mean, max)
        which:   "a" or "b" -> use plane_a or plane_b

    Returns:
        best_z  (int)
        table   (dict[z_level] = composite score)
    """
    # --- Gather per-z scores ---
    z_to_scores = defaultdict(list)
    for m in matches:
        plane = m.plane_a if which == "a" else m.plane_b
        z_val = int(round(float(plane.anchor_point.position[2])))
        z_to_scores[z_val].append(m.score)

    if not z_to_scores:
        return None, {}

    # --- Compute components ---
    z_levels = sorted(z_to_scores.keys())
    counts = np.array([len(z_to_scores[z]) for z in z_levels], float)
    means  = np.array([np.mean(z_to_scores[z]) for z in z_levels], float)
    maxes  = np.array([np.max(z_to_scores[z]) for z in z_levels], float)

    # --- Normalize each component 0–1 ---
    def minmax(x):
        rng = x.max() - x.min()
        return (x - x.min()) / (rng + 1e-9)

    counts_n, means_n, maxes_n = map(minmax, [counts, means, maxes])

    # --- Weighted linear score ---
    total = (
        params.w_count * counts_n +
        params.w_mean  * means_n +
        params.w_max   * maxes_n
    )

    # --- Pick best ---
    z_scores = {z: float(s) for z, s in zip(z_levels, total)}
    best_z = z_levels[int(np.argmax(total))]
    return best_z, z_scores



def filter_matches_in_bucket(matches: List[TransformMatch], bucket) -> List[TransformMatch]:
    return [m for m in matches if _match_in_bin(m, bucket)]


def select_matches_on_z(matches: List[TransformMatch], best_z_a: int, best_z_b: int) -> List[TransformMatch]:
    return [m for m in matches if m.z_a == best_z_a and m.z_b == best_z_b]


def choose_best_transform_fast(cands: List[TransformMatch]) -> TransformMatch | None:
    if not cands:
        return None
    return max(cands, key=lambda m: m.score)


# ---- IoU evaluation (placeholder) ----
def _transform_xy(coords_2d: np.ndarray, ang_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    """
    coords_2d: (N,2) array of [x,y] in A's plane frame.
    Applies: u' = R(ang_deg) * (scale * u) + [tx,ty]
    """
    if coords_2d.size == 0:
        return coords_2d
    th = np.deg2rad(ang_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=float)
    U = coords_2d.astype(float).T  # (2,N)
    U2 = (R @ (scale * U)).T       # (N,2)
    U2[:, 0] += tx
    U2[:, 1] += ty
    return U2

def _polys_at_z(zstack: ZStack, z_level: int):
    """Return list of (roi_id, np.array Nx2) for the given integer z."""
    out = []
    layer = zstack.z_planes.get(float(z_level), None)
    if not layer:
        return out
    for roi_id, roi_data in layer.items():
        coords = roi_data.get("coords")
        if not coords or len(coords) < 3:
            continue
        arr = np.asarray(coords, dtype=float)
        if arr.shape[1] == 2:
            xy = arr
        else:
            # coords may be [[x,y,z], ...]
            xy = arr[:, :2]
        out.append((roi_id, xy))
    return out

def _avg_pairwise_iou(polysA: list[tuple[int, np.ndarray]],
                      polysB: list[tuple[int, np.ndarray]],
                      max_nn_dist: float = np.inf) -> float:
    """
    polysA/B: lists of (roi_id, Nx2 coords)
    Match B to A by nearest centroid (1-to-1 via KD-tree). 
    Compute IoU per pair; return mean IoU over matched pairs.
    """
    if not polysA or not polysB:
        return 0.0

    # centroids
    A_xy   = [p for _, p in polysA]
    A_cent = np.array([xy.mean(axis=0) for xy in A_xy])  # (NA,2)

    B_xy   = [p for _, p in polysB]
    B_cent = np.array([xy.mean(axis=0) for xy in B_xy])  # (NB,2)

    tree = cKDTree(B_cent)
    dists, idxs = tree.query(A_cent, k=1)
    pairs = []
    usedB = set()
    for iA, (d, jB) in enumerate(zip(dists, idxs)):
        if max_nn_dist < np.inf and d > max_nn_dist:
            continue
        if jB in usedB:
            # enforce 1-to-1: skip this A if its nearest B already taken
            continue
        usedB.add(jB)
        pairs.append((iA, jB))

    if not pairs:
        return 0.0

    ious = []
    for iA, jB in pairs:
        polyA = ShPolygon(A_xy[iA])
        polyB = ShPolygon(B_xy[jB])
        if not polyA.is_valid or not polyB.is_valid:
            continue
        inter = polyA.intersection(polyB).area
        uni   = polyA.union(polyB).area
        if uni <= 0:
            continue
        ious.append(inter / uni)

    return float(np.mean(ious)) if ious else 0.0

def compute_plane_iou(zstack_a: ZStack, zstack_b: ZStack, 
                      plane_a: Plane, plane_b: Plane,
                      tx: float, ty: float, ang_deg: float, scale: float,
                      max_nn_dist: float = np.inf) -> float:
    """
    Average pairwise IoU of nearest-neighbour matched ROIs (by centroid)
    between z-slice of A and transformed z-slice of B.
    """
    zA = int(round(float(plane_a.anchor_point.position[2])))
    zB = int(round(float(plane_b.anchor_point.position[2])))

    # Polygons at those z-levels
    polysA = _polys_at_z(zstack_a, zA)          # [(roi_id, Nx2)]
    polysB = _polys_at_z(zstack_b, zB)
    if not polysA or not polysB:
        return 0.0

    # Transform B polygons into A's 2D frame using candidate (R,S,T)
    transB = []
    for rid, xy in polysB:
        xy2 = _transform_xy(xy, ang_deg=ang_deg, scale=scale, tx=tx, ty=ty)
        transB.append((rid, xy2))

    # Average IoU over NN centroid matches
    return _avg_pairwise_iou(polysA, transB, max_nn_dist=max_nn_dist)


def choose_best_transform_slow(zstack_a: ZStack, zstack_b: ZStack,
                               cands: List[TransformMatch],
                               patience: int = 20) -> Tuple[TransformMatch | None, float]:
    """
    Evaluate IoU for each candidate (R,T,S). Early-stop if no improvement for 'patience' steps.
    Returns (best_match, best_iou).
    """
    best = None
    best_iou = -1.0
    no_improve = 0

    for m in sorted(cands, key=lambda x: -x.score):  # try high-scoring first
        iou = compute_plane_iou(zstack_a, zstack_b, m.plane_a, m.plane_b, m.tx, m.ty, m.ang_deg, m.scale)
        if iou > best_iou:
            best_iou = iou
            best = m
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best, best_iou


# ---------- helpers for circular angle axis ----------
def ang_mod_id(i: int) -> int:
    return int(i % ANG_PERIOD_IDS)

def angle_dilate(ids: set[int], N: int) -> set[int]:
    """Dilate a set of angle IDs by N (circularly)."""
    if N <= 0: return ids
    out = set()
    for v in ids:
        for k in range(-N, N+1):
            out.add((v + k) % ANG_PERIOD_IDS)
    return out

def angle_min_cover(ids: set[int]) -> tuple[int, int]:
    """Smallest inclusive covering arc (start_id, length_in_bins)."""
    P = ANG_PERIOD_IDS
    if not ids: return 0, 0
    if len(ids) >= P: return 0, P

    arr = sorted({i % P for i in ids})
    best_gap = -1; best_a = arr[0]; best_b = arr[0]
    for i in range(len(arr)):
        a = arr[i]; b = arr[(i+1) % len(arr)]
        gap = (b - a) % P
        if gap > best_gap:
            best_gap = gap; best_a, best_b = a, b

    start  = best_b % P
    length = ((best_a - best_b) % P) + 1  # <-- inclusive (+1)
    if length > P: length = P
    return start, length

# ---------- working bin (mutable while merging) ----------
@dataclass
class _WorkBin:
    tx_min: int; tx_max: int
    ty_min: int; ty_max: int
    s_min:  int; s_max:  int
    ang_start: int; ang_len: int        # circular span
    count: int

    @property
    def size(self) -> int:
        # product of 4D spans (angle uses number of angle bins in the span)
        return (self.tx_max - self.tx_min + 1) * \
               (self.ty_max - self.ty_min + 1) * \
               (self.s_max  - self.s_min  + 1) * \
               max(self.ang_len, 1)

# ---------- public merged-key (hashable; with label() for your plotter/exporter) ----------
@dataclass(frozen=True)
class MergedIndexRange:
    tx_min_id: int; tx_max_id: int
    ty_min_id: int; ty_max_id: int
    s_min_id: int;  s_max_id: int
    ang_start_id: int; ang_len: int


    def label(self) -> str:
        tx_lo, tx_hi = int_like_range(self.tx_min_id, TX_STEP)[0], int_like_range(self.tx_max_id, TX_STEP)[1]
        ty_lo, ty_hi = int_like_range(self.ty_min_id, TY_STEP)[0], int_like_range(self.ty_max_id, TY_STEP)[1]
        s_lo,  _    = float_range(self.s_min_id, S_STEP)
        _,    s_hi  = float_range(self.s_max_id, S_STEP)
        ang_txt = angle_span_label(self.ang_start_id, self.ang_len)
        return (f"tx={tx_lo}–{tx_hi}px | "
                f"ty={ty_lo}–{ty_hi}px | "
                f"{ang_txt} | "
                f"s={s_lo:.1f}–{s_hi:.1f}")

@dataclass
class TransformMatch:
    tx: float
    ty: float
    ang_deg: float
    scale: float
    score: float
    plane_a: Plane
    plane_b: Plane
    z_a: int
    z_b: int
    key_bin: TransformIndexBin  # bin the match fell into (rigid binning)


# ---------- interval compatibility on linear axes ----------
def _gap_1d(a_min: int, a_max: int, b_min: int, b_max: int) -> int:
    """0 if overlap, else positive gap between two closed integer intervals."""
    if a_max < b_min:
        return b_min - a_max
    if b_max < a_min:
        return a_min - b_max
    return 0

def _in_seed_window(seed: _WorkBin, cand: _WorkBin, bucket_dist: int) -> bool:
    """
    True if 'cand' lies within the frozen window of 'seed' expanded by ±bucket_dist
    on linear axes and by bucket_dist in angle bin IDs (circularly).
    """
    # linear axes: intersection with seed±N
    if _gap_1d(seed.tx_min - bucket_dist, seed.tx_max + bucket_dist, cand.tx_min, cand.tx_max) > 0: return False
    if _gap_1d(seed.ty_min - bucket_dist, seed.ty_max + bucket_dist, cand.ty_min, cand.ty_max) > 0: return False
    if _gap_1d(seed.s_min  - bucket_dist, seed.s_max  + bucket_dist, cand.s_min,  cand.s_max)  > 0: return False

    # angle: candidate must intersect seed's dilated angle-ID set
    seed_ids = angle_ids_from_span(seed.ang_start, seed.ang_len)
    win_ids  = angle_dilate(seed_ids, bucket_dist)
    cand_ids = angle_ids_from_span(cand.ang_start, cand.ang_len)
    return bool(win_ids & cand_ids)

def angle_ids_from_span(start_id: int, length: int) -> set[int]:
    """Inclusive set of angle bin IDs covered by (start_id, length) on [0..P-1]."""
    P = ANG_PERIOD_IDS
    if length <= 0:
        return set()
    if length >= P:
        return set(range(P))
    start = start_id % P
    return { (start + k) % P for k in range(length) }

def angle_span_label(start_id: int, length: int) -> str:
    """
    Human-readable, always-inclusive angle label for a circular span.
    Uses the start bin's *low-degree edge* as base, then adds length*ANG_STEP - 1
    to get the inclusive 'hi' bound, finally wraps both to [-180, 180].
    """
    if length <= 0:
        return "θ=∅"

    # Bin low/high in degrees for an ID
    def bin_deg_bounds(bid: int) -> tuple[int, int]:
        lo, hi = int_like_range(bid, ANG_STEP)  # e.g. 176..179
        return lo, hi

    def wrap180(d: float) -> int:
        d = ((d + 180) % 360) - 180
        return int(d if d > -180 else d + 360)

    start = start_id % ANG_PERIOD_IDS
    lo_deg, _ = bin_deg_bounds(start)

    # Prefer starting near 0/negative side for spans at the seam,
    # so -4..7 appears instead of 176..-173 for short arcs.
    if lo_deg > 180 - ANG_STEP:
        lo_deg -= 360  # shift to negative side

    hi_deg_unwrapped = lo_deg + length * ANG_STEP - 1  # inclusive upper degree
    return f"θ={wrap180(lo_deg)}–{wrap180(hi_deg_unwrapped)}°"

def _min_circ_dist_sets(A: set[int], B: set[int], P: int) -> int:
    """
    Minimum circular distance (in bins) between any a∈A and b∈B on [0..P-1].
    0 => overlap/touch, 1 => immediate neighbours, etc.
    """
    if not A or not B:
        return 10**9
    best = 10**9
    for a in A:
        for b in B:
            d = abs(a - b) % P
            d = min(d, P - d)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best

def _ang_compatible(a_start: int, a_len: int,
                    b_start: int, b_len: int,
                    bucket_dist: int) -> bool:
    A_ids = angle_ids_from_span(a_start, a_len)
    B_ids = angle_ids_from_span(b_start, b_len)
    return _min_circ_dist_sets(A_ids, B_ids, ANG_PERIOD_IDS) <= bucket_dist

def _merge_into(a: _WorkBin, b: _WorkBin) -> None:
    a.tx_min = min(a.tx_min, b.tx_min); a.tx_max = max(a.tx_max, b.tx_max)
    a.ty_min = min(a.ty_min, b.ty_min); a.ty_max = max(a.ty_max, b.ty_max)
    a.s_min  = min(a.s_min,  b.s_min);  a.s_max  = max(a.s_max,  b.s_max)

    # Inclusive angle union across the circle
    A = angle_ids_from_span(a.ang_start, a.ang_len)
    B = angle_ids_from_span(b.ang_start, b.ang_len)
    ids = A | B
    a.ang_start, a.ang_len = angle_min_cover(ids)  # must return inclusive length (+1)
    a.count += b.count


def merge_bins_with_groups(counter: Counter, bucket_dist: int = 1, passes: int = 3):
    # same algorithm as your current merge_bins(...), but:
    #  - keep track of member bins for each merged work bin
    #  - return (merged_counter, groups) where groups is a list of dicts
    if not counter:
        return Counter(), []

    # ---- seed work bins ----
    work = []
    groups = []  # parallel to work; each entry is dict{TransformIndexBin: count}
    for k, c in counter.items():
        work.append(_WorkBin(
            tx_min=k.tx_id, tx_max=k.tx_id,
            ty_min=k.ty_id, ty_max=k.ty_id,
            s_min=k.s_id,   s_max=k.s_id,
            ang_start=ang_mod_id(k.ang_id), ang_len=1,
            count=int(c),
        ))
        groups.append({k: int(c)})

    # ---- passes ----
    for _ in range(max(1, passes)):
        # stable order: smallest first (then by count)
        order = list(range(len(work)))
        order.sort(key=lambda i: (work[i].size, work[i].count))

        # ordered views (indices for this pass are 0..n-1)
        W = [work[i]   for i in order]
        G = [groups[i] for i in order]
        C = [w.count   for w in W]
        n = len(W)
        if n <= 1:
            break

        # ---- Phase 0: build FROZEN compatibility graph (undirected, no new nodes) ----
        # edge if either node lies in the other's seed-window
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if _in_seed_window(W[i], W[j], bucket_dist) or _in_seed_window(W[j], W[i], bucket_dist):
                    adj[i].add(j)
                    adj[j].add(i)

        # ---- Phase 1: select disjoint pairs for degree-1 nodes ----
        unmatched = set(range(n))
        pairs = []  # list of (i,j) to merge this pass

        # collect degree-1 edges
        deg1_edges = []
        for i in range(n):
            if i not in unmatched:
                continue
            nbrs = adj[i] & unmatched
            if len(nbrs) == 1:
                j = next(iter(nbrs))
                # weight = sum of counts; store as (weight, i, j) with i<j
                a, b = (i, j) if i < j else (j, i)
                deg1_edges.append((C[a] + C[b], a, b))

        # select all degree-1 edges (they are disjoint by definition)
        # to be extra safe, pick in descending weight so conflicts resolve best-possible
        deg1_edges.sort(reverse=True)
        taken = set()
        for w, i, j in deg1_edges:
            if i in unmatched and j in unmatched and i not in taken and j not in taken:
                pairs.append((i, j))
                unmatched.discard(i); unmatched.discard(j)
                taken.add(i); taken.add(j)

        # ---- Phase 2: greedy maximum-weight matching on remaining nodes ----
        # build all candidate edges among unmatched nodes
        edges = []
        for i in sorted(unmatched):
            for j in adj[i]:
                if j in unmatched and i < j:
                    edges.append((C[i] + C[j], i, j))

        # Test
        def _ang_gap_bins(a, b):
            Ai = angle_ids_from_span(a.ang_start, a.ang_len)
            Bi = angle_ids_from_span(b.ang_start, b.ang_len)
            return _min_circ_dist_sets(Ai, Bi, ANG_PERIOD_IDS)

        edges.sort(key=lambda e: (-(e[0]), _ang_gap_bins(W[e[1]], W[e[2]])))

        # edges.sort(reverse=True)  # pick largest totals first

        for w, i, j in edges:
            if i in unmatched and j in unmatched:
                pairs.append((i, j))
                unmatched.discard(i); unmatched.discard(j)

        merges_this_pass = len(pairs)

        # ---- Phase 3: apply merges simultaneously to build next state ----
        new_work, new_groups = [], []

        # helper: merge two bins to a new _WorkBin and merged member-map
        def _merge_pair(i, j):
            acc = _WorkBin(**W[i].__dict__)  # shallow copy fields
            _merge_into(acc, W[j])
            grp = G[i].copy()
            for k, c in G[j].items():
                grp[k] = grp.get(k, 0) + c
            return acc, grp

        # add all merged pairs
        for i, j in pairs:
            acc, grp = _merge_pair(i, j)
            new_work.append(acc)
            new_groups.append(grp)

        # carry over any unmatched nodes unchanged
        for i in sorted(unmatched):
            new_work.append(W[i])
            new_groups.append(G[i])

        work, groups = new_work, new_groups

        if merges_this_pass == 0:
            break  # early-exit if nothing merged this pass

    # ---- emit merged counter and group list ----
    merged = Counter()
    merged_groups = []
    for w, g in zip(work, groups):
        key = MergedIndexRange(
            tx_min_id=w.tx_min, tx_max_id=w.tx_max,
            ty_min_id=w.ty_min, ty_max_id=w.ty_max,
            s_min_id=w.s_min,   s_max_id=w.s_max,
            ang_start_id=w.ang_start, ang_len=w.ang_len
        )
        total = sum(g.values())

        merged[key] = total
        merged_groups.append((key, g))

    return merged, merged_groups

def explain_merged_bar(merged_groups, target_key, top_n=20):
    """
    Print the member pre-merge bins (and counts) for a given merged key.
    Pass the 'target_key' you see in the post-merge top-10 (the MergedIndexRange).
    """
    for key, members in merged_groups:
        if key == target_key:
            print(f"[GROUP] {key.label()}  -> total {sum(members.values())}")
            top = sorted(members.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            for (bin_key, cnt) in top:
                print(f"  {bin_key.label():<50}  {cnt}")
            break


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
        selected_z = 30
        # selected_z = 32
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
            # APPLY AFFINE TRANSFORMATIONS
            new_stack = new_stack.affine_transformed(
                angle_deg=50,
                t=(10,10),
                s=1, # dont touch this unless you unfix scale estimation
                origin=(0, 0))
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

    # DEBUG
    # Visualize only "correct" style pairs:
    #   - same z-level anchors
    #   - same anchor IDs
    _visualize_filtered_matches(
        matched_dict=matched,
        zstack_a=z_stack,
        zstack_b=new_stack,
        same_z_level=True,
        same_anchor_id=True
    )
    # END DEBUG

    # Extract expected z from the sample plane used to build z_stack B
    expected_z = int(round(selected_plane.anchor_point.position[2]))

    # Plot score vs z and highlight the expected z-plane
    plot_scores_vs_z(matched, expected_z, aggregate="max",
                    title="Coarse match score vs z (flat planes)")

    # Extract and bin transforms
    transforms = extract_transforms(matched)
    print(f"[INFO] Extracted {len(transforms)} transforms.")

    pre = bin_transforms_by_id(transforms)                  # pre-merge
    post, groups = merge_bins_with_groups(pre, bucket_dist=1, passes=2)

    # Show the post-merge top bar and its components
    # top_key, _ = post.most_common(1)[0]
    for key in post:
        explain_merged_bar(groups, key)

    # # Plot top-10
    plot_top_bins(post, top_k=10, title="Top 10 Transform Modes (tx/ty/θ/scale)")

    # 1) find most represented bucket (use post-merge if you want the merged modes)
    top_bucket, top_count = most_represented_bucket(post if post else pre)
    if top_bucket is None:
        print("[WARN] No buckets found.")
        return

    # 2) extract full matches and filter into that bucket
    matches = extract_transforms(matched)           # rich records
    matches_top = filter_matches_in_bucket(matches, top_bucket)
    print(f"[INFO] Using top bucket ({top_bucket.label() if hasattr(top_bucket,'label') else top_bucket}) with {len(matches_top)} matches")

    # 3) best z-plane (dataset A and B) with weighted scoring
    params = ZScoreParams(w_count=1.0, w_mean=1.0, w_max=1.0) # TO BE TUNED
    best_z_a, table_a = score_z_levels(matches_top, params, which="a")
    best_z_b, table_b = score_z_levels(matches_top, params, which="b")
    print(f"[Z*] best_z_a={best_z_a}, best_z_b={best_z_b}")

    # 4) restrict to candidates that hit those z-levels
    cands = select_matches_on_z(matches_top, best_z_a, best_z_b)
    if not cands:
        print("[WARN] No candidates at chosen z-levels; falling back to top by score in top bucket.")
        cands = matches_top

    # 5a) FAST: take highest coarse score
    best_fast = choose_best_transform_fast(cands)
    if best_fast:
        print(f"[FAST] best score={best_fast.score:.6f}, "
            f"R={best_fast.ang_deg:.2f}°, T=({best_fast.tx:.1f},{best_fast.ty:.1f}), S={best_fast.scale:.3f}, "
            f"zA={best_fast.z_a}, zB={best_fast.z_b}")

    # 5b) SLOW: refine by IoU (needs compute_plane_iou implementation)
    best_slow, best_iou = choose_best_transform_slow(z_stack, new_stack, cands, patience=20)
    if best_slow:
        print(f"[SLOW] best IoU={best_iou:.4f}, "
            f"R={best_slow.ang_deg:.2f}°, T=({best_slow.tx:.1f},{best_slow.ty:.1f}), S={best_slow.scale:.3f}, "
            f"zA={best_slow.z_a}, zB={best_slow.z_b}")


    end = time.perf_counter()
    print(f"[TIMER] Total run time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
