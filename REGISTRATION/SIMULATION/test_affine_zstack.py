#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from zstack import ZStack

def pick_middle_z(zstack: ZStack) -> int:
    zs = sorted(zstack.z_planes.keys())
    if not zs:
        raise RuntimeError("No z-planes found.")
    return int(round(zs[len(zs)//2]))

def slice_points(zstack: ZStack, z_level: int):
    pts = []
    layer = zstack.z_planes.get(float(z_level), None)
    if not layer:
        return pts
    for _, info in layer.items():
        coords = info.get("coords", [])
        if not coords:
            continue
        arr = np.asarray(coords, float)
        if arr.shape[1] >= 2:
            pts.append(arr[:, :2])
    return pts

def plot_overlay(zA: ZStack, zB: ZStack, z_level: int, title: str):
    A_polys = slice_points(zA, z_level)
    B_polys = slice_points(zB, z_level)

    fig, ax = plt.subplots(figsize=(8, 8))
    for poly in A_polys:
        ax.plot(poly[:,0], poly[:,1], '-', alpha=0.5, linewidth=1.0, label='A' if 'A' not in ax.get_legend_handles_labels()[1] else "")
    for poly in B_polys:
        ax.plot(poly[:,0], poly[:,1], '-', alpha=0.9, linewidth=1.2, label='B (transformed)' if 'B (transformed)' not in ax.get_legend_handles_labels()[1] else "")

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{title}\nOverlay at z={z_level}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.show()

CSV_PATH = "real_data_filtered_algo_VOLUMES_g.csv"

def main():
    # Load original
    A = ZStack(data=CSV_PATH)  # original
    # Copy and transform
    B = A.affine_transformed(
        angle_deg=50,
        t=(10,10),
        s=1,
        origin=(0, 0),
    )

    Z = 30
    # choose z
    z_level = int(round(Z)) if Z is not None else pick_middle_z(A)
    plot_overlay(A, B, z_level, title="ZStack Affine Transform Demo")

if __name__ == "__main__":
    main()
