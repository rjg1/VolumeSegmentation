import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import itertools
import pandas as pd
import pyqtgraph as pg
import tifffile as tiff

# --- Qt / Graph imports (try PySide6, fall back to PyQt5) ---
try:
    from PySide6 import QtWidgets, QtCore, QtGui
    QT_LIB = 'PySide6'
except Exception:
    from PyQt5 import QtWidgets, QtCore, QtGui  # type: ignore
    QT_LIB = 'PyQt5'

try:
    from skimage import measure as skmeasure  # type: ignore
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False
    skmeasure = None  # type: ignore

# Optional Hungarian
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    HUNGARIAN_AVAILABLE = True
except Exception:
    HUNGARIAN_AVAILABLE = False
    def linear_sum_assignment(cost_matrix: np.ndarray):
        m = cost_matrix.copy()
        used_rows, used_cols, pairs = set(), set(), []
        flat = [(i, j, m[i, j]) for i in range(m.shape[0]) for j in range(m.shape[1])]
        flat.sort(key=lambda x: x[2])
        for i, j, _ in flat:
            if i not in used_rows and j not in used_cols:
                used_rows.add(i); used_cols.add(j); pairs.append((i, j))
        if not pairs:
            return np.array([], dtype=int), np.array([], dtype=int)
        rows, cols = zip(*pairs)
        return np.array(rows, dtype=int), np.array(cols, dtype=int)

# Optional convex hull
try:
    from scipy.spatial import ConvexHull  # type: ignore
    HULL_AVAILABLE = True
except Exception:
    HULL_AVAILABLE = False
    ConvexHull = None  # type: ignore

# Optional precise polygon IoU (Shapely)
try:
    from shapely.geometry import Polygon  # type: ignore
    SHAPELY = True
except Exception:
    SHAPELY = False
    Polygon = None  # type: ignore


# ---------------------------- Math utils ----------------------------

def estimate_similarity_transform_2d(src: np.ndarray, dst: np.ndarray, allow_reflection: bool = False):
    """Return (s, R, t) such that approx: dst ≈ s * R @ src + t."""
    assert src.shape == dst.shape and src.shape[1] == 2 and src.shape[0] >= 2
    mu_src = src.mean(axis=0); mu_dst = dst.mean(axis=0)
    X = src - mu_src; Y = dst - mu_dst
    Sigma = X.T @ Y / src.shape[0]
    U, D, Vt = np.linalg.svd(Sigma)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = U @ Vt
    var_src = (X ** 2).sum() / src.shape[0]
    s = 1.0 if var_src < 1e-12 else float(np.trace(np.diag(D)) / var_src)
    t = mu_dst - s * (R @ mu_src)
    return float(s), R, t


def apply_similarity_transform_2d(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    return (s * (R @ points.T)).T + t


def two_point_similarity(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray):
    """Similarity from two point pairs: p1->q1, p2->q2.
       Returns (s, R, t). If p1==p2, falls back to translation from p1->q1.
    """
    vP = p2 - p1
    vQ = q2 - q1
    nP = np.linalg.norm(vP)
    if nP < 1e-8:
        s = 1.0; R = np.eye(2); t = q1 - p1
        return s, R, t
    s = float(np.linalg.norm(vQ) / nP) if nP > 0 else 1.0
    aP = np.arctan2(vP[1], vP[0])
    aQ = np.arctan2(vQ[1], vQ[0])
    ang = aQ - aP
    c, d = np.cos(ang), np.sin(ang)
    R = np.array([[c, -d], [d, c]], dtype=float)
    t = q1 - s * (R @ p1)
    return s, R, t


# ---------------------------- Data model ----------------------------

ROIKey = Tuple[float, int]  # (z, ROI_ID)

def norm_z(zval: float) -> float:
    return float(np.round(float(zval), 6))

# ---------------------------- Atlas ----------------------------
@dataclass
class Atlas:
    """Holds a many-dataset mapping to a shared Common_ID."""
    # columns are dataset labels, e.g. ["Session1","Session2",...]
    columns: List[str] = field(default_factory=list)
    # rows are Common_ID -> { dataset_label: "z,ROI" }
    rows: Dict[int, Dict[str, str]] = field(default_factory=dict)

    def _next_common_id(self) -> int:
        return (max(self.rows.keys()) + 1) if self.rows else 1

    @staticmethod
    def roikey(z: float, rid: int) -> str:
        return f"{float(z)},{int(rid)}"  # CSV-friendly

    @staticmethod
    def parse_roikey(s: str) -> Optional[Tuple[float,int]]:
        s = (s or "").strip()
        if not s: return None
        try:
            z_str, rid_str = s.split(",")
            return float(z_str), int(rid_str)
        except Exception:
            return None

    def ensure_column(self, label: str):
        if label not in self.columns:
            self.columns.append(label)
            # ensure all existing rows have an empty cell for the new column
            for row in self.rows.values():
                row.setdefault(label, "")

    def add_pair(self, labelA: str, zA: float, ridA: int,
                 labelB: str, zB: float, ridB: int,
                 prefer_existing='atlas') -> int:
        """
        Insert/merge a pair (A,B) into the atlas; returns the Common_ID used.

        Rules:
        - If either ROI already exists in some row, use that Common_ID.
        - If both exist but in different rows:
            - If prefer_existing='atlas' -> merge rows using the Common_ID where A/B were found first.
            - If prefer_existing='tool'  -> let the second override the first.
        - Otherwise mint a new Common_ID.
        """
        self.ensure_column(labelA)
        self.ensure_column(labelB)
        ra = self.roikey(zA, ridA)
        rb = self.roikey(zB, ridB)

        # Find rows containing ra/rb
        row_a = None; row_b = None
        for cid, row in self.rows.items():
            if row.get(labelA, "") == ra:
                row_a = cid
            if row.get(labelB, "") == rb:
                row_b = cid

        # Case: neither exists → new row
        if row_a is None and row_b is None:
            cid = self._next_common_id()
            self.rows[cid] = {col: "" for col in self.columns}
            self.rows[cid][labelA] = ra
            self.rows[cid][labelB] = rb
            return cid

        # Case: one side exists → extend that row
        if row_a is not None and row_b is None:
            cid = row_a
            self.rows[cid][labelB] = rb
            return cid
        if row_b is not None and row_a is None:
            cid = row_b
            self.rows[cid][labelA] = ra
            return cid

        # Case: both exist
        if row_a == row_b:
            return row_a  # already consistent
        # conflict: two rows for the same cell → merge policy
        keep, drop = (row_a, row_b) if prefer_existing == 'atlas' else (row_b, row_a)
        # merge all columns from drop into keep (prefer keep's non-empty)
        for col in self.columns:
            v_keep = self.rows[keep].get(col, "")
            v_drop = self.rows[drop].get(col, "")
            if not v_keep and v_drop:
                self.rows[keep][col] = v_drop
        # delete dropped row
        del self.rows[drop]
        return keep

    def to_dataframe(self) -> pd.DataFrame:
        cols = self.columns + ["Common_ID"]
        data = []
        for cid, row in sorted(self.rows.items()):
            rec = {c: row.get(c, "") for c in self.columns}
            rec["Common_ID"] = cid
            data.append(rec)
        return pd.DataFrame(data, columns=cols)

    @classmethod
    def from_csv(cls, path: str) -> "Atlas":
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c != "Common_ID"]
        atlas = cls(columns=cols, rows={})
        for _, r in df.iterrows():
            cid = int(r["Common_ID"])
            atlas.rows[cid] = {c: str(r.get(c, "") if not pd.isna(r.get(c, "")) else "") for c in cols}
        return atlas

    def save_csv(self, path: str):
        self.to_dataframe().to_csv(path, index=False)


@dataclass
class Dataset:
    name: str = ""
    df: Optional[pd.DataFrame] = None

    # SAFE defaults so they exist even before CSV load
    z_values: List[float] = field(default_factory=list)
    roi_to_points_by_z: Dict[float, Dict[int, np.ndarray]] = field(default_factory=dict)
    roi_centroids_xy_by_z: Dict[float, Dict[int, np.ndarray]] = field(default_factory=dict)

    current_z: Optional[float] = None
    cur_roi_ids_sorted: List[int] = field(default_factory=list)
    cur_centroid_array: Optional[np.ndarray] = None
    cur_roi_centroids_xy: Dict[int, np.ndarray] = field(default_factory=dict)

    labels_by_z: Dict[float, Dict[int, int]] = field(default_factory=dict)  # z -> {ROI_ID: label}
    has_labels: bool = False

    tif_path: Optional[str] = None
    tif_stack: Optional[np.ndarray] = None  # (H,W) or (Z,H,W)
    def clear(self):
        self.df = None
        self.z_values = []
        self.roi_to_points_by_z = {}
        self.roi_centroids_xy_by_z = {}
        self.current_z = None
        self.cur_roi_ids_sorted = []
        self.cur_centroid_array = None
        self.cur_roi_centroids_xy = {}
        self.tif_path = None
        self.tif_stack = None
        self.labels_by_z = {}
        self.has_labels = False

    def load_csv(self, path: str):
        self.name = os.path.basename(path)
        df = pd.read_csv(path)
        required = {"x", "y", "z", "ROI_ID"}
        missing = required - set(map(str, df.columns))
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        df['x'] = pd.to_numeric(df['x'])
        df['y'] = pd.to_numeric(df['y'])
        df['z'] = pd.to_numeric(df['z'])
        df['ROI_ID'] = pd.to_numeric(df['ROI_ID'], downcast='integer')
        
        has_label_col = 'label' in df.columns

        if has_label_col:
            # normalize to integer labels; missing/NaN -> 0
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
            self.has_labels = bool((df['label'] > 0).any())
        else:
            df['label'] = 0
            self.has_labels = False
        
        self.df = df

        self.z_values = sorted([norm_z(z) for z in df['z'].unique()])
        self.roi_to_points_by_z = {}
        self.roi_centroids_xy_by_z = {}
        for z in self.z_values:
            subz = df[np.isclose(df['z'], z)]
            rid2pts: Dict[int, np.ndarray] = {}
            rid2cent: Dict[int, np.ndarray] = {}
            for rid, sub in subz.groupby('ROI_ID'):
                pts = sub[["x", "y", "z"]].to_numpy(dtype=float)
                rid2pts[int(rid)] = pts
                rid2cent[int(rid)] = pts[:, :2].mean(axis=0)
            self.roi_to_points_by_z[z] = rid2pts
            self.roi_centroids_xy_by_z[z] = rid2cent
        if self.z_values:
            self.set_current_z(self.z_values[0])

            self.labels_by_z = {}
        
        for z in self.z_values:
            ...
            # build labels_by_z
            if self.has_labels:
                subz = df[np.isclose(df['z'], z)]
                self.labels_by_z[z] = {int(rid): int(lbl) for rid, lbl
                                    in subz.groupby('ROI_ID')['label'].first().items()}
            else:
                self.labels_by_z[z] = {}

    

    def get_label(self, z: float, rid: int) -> int:
        z = norm_z(z)
        return int((self.labels_by_z.get(z) or {}).get(int(rid), 0))

    def set_label(self, z: float, rid: int, label: int):
        """Set label in memory (both map + df). Creates 'label' col if missing."""
        z = norm_z(z); rid = int(rid); label = int(label)
        self.labels_by_z.setdefault(z, {})[rid] = label
        if self.df is None: return
        if 'label' not in self.df.columns:
            self.df['label'] = 0
        m = (np.isclose(self.df['z'], z)) & (self.df['ROI_ID'].astype(int) == rid)
        self.df.loc[m, 'label'] = label
        # also refresh has_labels flag
        self.has_labels = bool((self.df['label'] > 0).any())

    def load_tif(self, path: str):
        arr = tiff.imread(path)
        if arr.ndim == 2:
            self.tif_stack = arr.astype(np.float32)
        elif arr.ndim >= 3:
            self.tif_stack = (arr if arr.ndim == 3 else arr[..., 0]).astype(np.float32)
        else:
            raise ValueError("Unsupported TIF dims")
        self.tif_path = path

    def set_current_z(self, z: float):
        z = norm_z(z)
        self.current_z = z
        rid2cent = (self.roi_centroids_xy_by_z or {}).get(z, {})
        self.cur_roi_centroids_xy = rid2cent
        self.cur_roi_ids_sorted = sorted(rid2cent.keys())
        if rid2cent:
            self.cur_centroid_array = np.array(
                [rid2cent[rid] for rid in self.cur_roi_ids_sorted], dtype=float
            )
        else:
            self.cur_centroid_array = np.zeros((0, 2), dtype=float)

    def get_boundary_xy_for_z(self, z: float) -> np.ndarray:
        if self.df is None:
            return np.zeros((0, 2), dtype=float)
        z = norm_z(z)
        subz = self.df[np.isclose(self.df['z'], z)]
        if subz.empty:
            return np.zeros((0, 2), dtype=float)
        return subz[["x", "y"]].to_numpy(dtype=float)

    def get_tif_slice_for_z(self, z: float) -> Optional[np.ndarray]:
        if self.tif_stack is None:
            return None
        if self.tif_stack.ndim == 2:
            return self.tif_stack
        zvals = self.z_values
        if len(zvals) == self.tif_stack.shape[0] and len(zvals) > 0:
            z_norm = norm_z(z)
            try:
                idx = zvals.index(z_norm)
            except ValueError:
                idx = int(np.argmin(np.abs(np.array(zvals) - z_norm)))
        else:
            idx = int(np.clip(int(round(z)), 0, self.tif_stack.shape[0] - 1))
        return self.tif_stack[idx]


# ---------------------------- Graphics helpers ----------------------------

class ClickableScatter(pg.ScatterPlotItem):
    def __init__(self, *args, on_left_click=None, on_right_click=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_left = on_left_click
        self._on_right = on_right_click
    def mouseClickEvent(self, ev):
        if ev.button() not in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            super().mouseClickEvent(ev); return
        pts = self.pointsAt(ev.pos())
        if pts is None or len(pts) == 0:
            ev.ignore(); return
        rid = pts[0].data()
        if ev.button() == QtCore.Qt.LeftButton and self._on_left:
            self._on_left(rid); ev.accept()
        elif ev.button() == QtCore.Qt.RightButton and self._on_right:
            self._on_right(rid); ev.accept()
        else:
            ev.ignore()


class ClickablePathItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, path: QtGui.QPainterPath, on_click=None, key=None):
        super().__init__(path)
        self._on_click = on_click
        self._key = key

    def mousePressEvent(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton and self._on_click:
            self._on_click(self._key)   # notify MainWindow
            ev.accept()
            return
        super().mousePressEvent(ev)


# ---------------------------- Main Window ----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Registration GUI")
        self.resize(2100, 1100)

        # Composite image of A and B
        self._additiveItem = None

        # --- Atlas state ---
        self.atlas = Atlas()   # auto-created on startup
        self.dataset_labels: Dict[str, Optional[str]] = {'A': None, 'B': None}

        # Cached zoomed values of plots
        self._savedRanges = {'A': None, 'B': None, 'O': None}  # overlay=O

        # Cached outlines
        self.outlineItemsA = []
        self.outlineItemsB = []

        # Cached pairs
        self._two_pair_cache: Dict[Tuple[ROIKey, ROIKey, ROIKey, ROIKey], Tuple[float, np.ndarray, np.ndarray, float]] = {}
        self._best_combo_cache: Dict[frozenset, Tuple[float, np.ndarray, np.ndarray, float]] = {}


        # TIF rects (per-z)
        self.imgRectA_by_z = {}  # z -> QRectF
        self.imgRectB_by_z = {}

        # Data
        self.datasetA = Dataset(name="A")
        self.datasetB = Dataset(name="B")

        # Pairing maps (1:1) using keys (z, ROI_ID)
        self.A2B: Dict[ROIKey, ROIKey] = {}
        self.B2A: Dict[ROIKey, ROIKey] = {}

        # Current staged selections (ROI_ID only; z from dataset state)
        self.selA: Optional[int] = None
        self.selB: Optional[int] = None

        # Transform from B->A (None until computed dynamically)
        self.transform: Optional[Tuple[float, np.ndarray, np.ndarray]] = None

        # (Legacy) initial 90° step toggles, kept for completeness but not used for live UI
        self.rotA_k = 0
        self.rotB_k = 0

        # Base levels cache per dataset per z (lo, hi)
        self._baseLevelsA_by_z = {}  # {z: (lo, hi)}
        self._baseLevelsB_by_z = {}

        # ---------------- UI ----------------
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Three views: A | B | Overlay
        self.viewA = pg.PlotWidget(); self.viewB = pg.PlotWidget(); self.viewOverlay = pg.PlotWidget()
        for v, title in [
            (self.viewA, "Dataset A (target)"),
            (self.viewB, "Dataset B (source)"),
            (self.viewOverlay, "Overlay (A+B) — live transform & IoU")
        ]:
            v.setBackground('w'); v.showGrid(x=True, y=True, alpha=0.2); v.setAspectLocked(False)
            ax = v.getAxis('left'); ax.setPen(pg.mkPen(color=(80,80,80)))
            ax = v.getAxis('bottom'); ax.setPen(pg.mkPen(color=(80,80,80)))
            v.setTitle(title)

        layout.addWidget(self.viewA, 1)
        layout.addWidget(self.viewB, 1)
        layout.addWidget(self.viewOverlay, 1)

        self.viewA.getViewBox().sigRangeChanged.connect(lambda *_: self._save_range('A'))
        self.viewB.getViewBox().sigRangeChanged.connect(lambda *_: self._save_range('B'))
        self.viewOverlay.getViewBox().sigRangeChanged.connect(lambda *_: self._save_range('O'))

        # Match matplotlib origin='upper'
        for v in (self.viewA, self.viewB, self.viewOverlay):
            v.invertY(True)

        # Background click selection
        self.viewA.scene().sigMouseClicked.connect(self._on_viewA_click)
        self.viewB.scene().sigMouseClicked.connect(self._on_viewB_click)

        # Control panel (right)
        ctrl = QtWidgets.QWidget(); ctrl.setFixedWidth(600); layout.addWidget(ctrl)
        f = QtWidgets.QFormLayout(ctrl)

        self.btnLoadMasksA = QtWidgets.QPushButton("Load Masks (A)…")
        self.btnLoadMasksB = QtWidgets.QPushButton("Load Masks (B)…")
        self.lblMasksA = QtWidgets.QLabel("<i>Not loaded</i>")
        self.lblMasksB = QtWidgets.QLabel("<i>Not loaded</i>")
        self.btnLoadMasksA.clicked.connect(lambda: self.load_masks(self.datasetA, self.lblMasksA, which='A'))
        self.btnLoadMasksB.clicked.connect(lambda: self.load_masks(self.datasetB, self.lblMasksB, which='B'))
        f.addRow(self.btnLoadMasksA, self.lblMasksA)
        f.addRow(self.btnLoadMasksB, self.lblMasksB)

        self.btnLoadImageA = QtWidgets.QPushButton("Load Image (A)…")
        self.btnLoadImageB = QtWidgets.QPushButton("Load Image (B)…")
        self.lblImageA = QtWidgets.QLabel("<i>Not loaded</i>")
        self.lblImageB = QtWidgets.QLabel("<i>Not loaded</i>")
        self.btnLoadImageA.clicked.connect(lambda: self.load_image(self.datasetA, self.lblImageA, which='A'))
        self.btnLoadImageB.clicked.connect(lambda: self.load_image(self.datasetB, self.lblImageB, which='B'))
        f.addRow(self.btnLoadImageA, self.lblImageA)
        f.addRow(self.btnLoadImageB, self.lblImageB)

         # --- Atlas controls ---
        self.btnAtlasNew  = QtWidgets.QPushButton("New")
        self.btnAtlasLoad = QtWidgets.QPushButton("Load")
        self.btnAtlasSave = QtWidgets.QPushButton("Export")

        self.btnAtlasNew.clicked.connect(self._atlas_new)
        self.btnAtlasLoad.clicked.connect(self._atlas_load)
        self.btnAtlasSave.clicked.connect(self._atlas_export)

        atlas_row = QtWidgets.QWidget()
        hb = QtWidgets.QHBoxLayout(atlas_row); hb.setContentsMargins(0,0,0,0); hb.setSpacing(8)
        hb.addWidget(self.btnAtlasNew); hb.addWidget(self.btnAtlasLoad); hb.addWidget(self.btnAtlasSave)
        f.addRow(QtWidgets.QLabel("<b>Atlas</b>"), atlas_row)

        # Show image + Show outlines (same row per dataset)
        self.chkShowImgA = QtWidgets.QCheckBox("Show image (A)")
        self.chkShowImgB = QtWidgets.QCheckBox("Show image (B)")
        self.chkShowOutlineA = QtWidgets.QCheckBox("Show outlines (A)")
        self.chkShowOutlineB = QtWidgets.QCheckBox("Show outlines (B)")
        self.chkShowImgA.setChecked(False); self.chkShowImgB.setChecked(False)
        self.chkShowOutlineA.setChecked(True); self.chkShowOutlineB.setChecked(True)

        self.chkShowImgA.toggled.connect(lambda _: (self.populate_view_A(), self.update_overlay_view()))
        self.chkShowImgB.toggled.connect(lambda _: (self.populate_view_B(), self.update_overlay_view()))
        self.chkShowOutlineA.toggled.connect(self._on_toggle_outlineA)
        self.chkShowOutlineB.toggled.connect(self._on_toggle_outlineB)

        rowA = QtWidgets.QWidget(); hA = QtWidgets.QHBoxLayout(rowA); hA.setContentsMargins(0,0,0,0); hA.setSpacing(12)
        hA.addWidget(self.chkShowImgA); hA.addWidget(self.chkShowOutlineA); hA.addStretch(1)
        rowB = QtWidgets.QWidget(); hB = QtWidgets.QHBoxLayout(rowB); hB.setContentsMargins(0,0,0,0); hB.setSpacing(12)
        hB.addWidget(self.chkShowImgB); hB.addWidget(self.chkShowOutlineB); hB.addStretch(1)
        f.addRow(QtWidgets.QLabel("A display:"), rowA)
        f.addRow(QtWidgets.QLabel("B display:"), rowB)

        # --- Image intensity (sliders) ---
        self.sldIntensityA = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldIntensityA.setRange(5, 500)          # 5 -> 0.05×, 500 -> 5.00×
        self.sldIntensityA.setValue(100)              # 1.00× default
        self.lblIntensityA = QtWidgets.QLabel("1.00×")
        self.sldIntensityA.valueChanged.connect(lambda v: self.on_intensity_changed('A', v))

        wrapA = QtWidgets.QWidget()
        hlA = QtWidgets.QHBoxLayout(wrapA); hlA.setContentsMargins(0, 0, 0, 0); hlA.setSpacing(8)
        hlA.addWidget(self.sldIntensityA); hlA.addWidget(self.lblIntensityA)
        f.addRow(QtWidgets.QLabel("Image intensity A:"), wrapA)

        self.sldIntensityB = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldIntensityB.setRange(5, 500)
        self.sldIntensityB.setValue(100)
        self.lblIntensityB = QtWidgets.QLabel("1.00×")
        self.sldIntensityB.valueChanged.connect(lambda v: self.on_intensity_changed('B', v))

        wrapB = QtWidgets.QWidget()
        hlB = QtWidgets.QHBoxLayout(wrapB); hlB.setContentsMargins(0, 0, 0, 0); hlB.setSpacing(8)
        hlB.addWidget(self.sldIntensityB); hlB.addWidget(self.lblIntensityB)
        f.addRow(QtWidgets.QLabel("Image intensity B:"), wrapB)

         # --- Image opacity (sliders) ---
        self.sldOpacityA = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldOpacityA.setRange(0, 100)      # 0%..100%
        self.sldOpacityA.setValue(100)          # default 100%
        self.lblOpacityA = QtWidgets.QLabel("100%")
        self.sldOpacityA.valueChanged.connect(lambda v: self.on_opacity_changed('A', v))
        wrapOA = QtWidgets.QWidget()
        hlOA = QtWidgets.QHBoxLayout(wrapOA); hlOA.setContentsMargins(0,0,0,0); hlOA.setSpacing(8)
        hlOA.addWidget(self.sldOpacityA); hlOA.addWidget(self.lblOpacityA)
        f.addRow(QtWidgets.QLabel("Image opacity A:"), wrapOA)

        self.sldOpacityB = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldOpacityB.setRange(0, 100)
        self.sldOpacityB.setValue(100)
        self.lblOpacityB = QtWidgets.QLabel("100%")
        self.sldOpacityB.valueChanged.connect(lambda v: self.on_opacity_changed('B', v))
        wrapOB = QtWidgets.QWidget()
        hlOB = QtWidgets.QHBoxLayout(wrapOB); hlOB.setContentsMargins(0,0,0,0); hlOB.setSpacing(8)
        hlOB.addWidget(self.sldOpacityB); hlOB.addWidget(self.lblOpacityB)
        f.addRow(QtWidgets.QLabel("Image opacity B:"), wrapOB)


        # ---------------- Rotation (deg) + flips ----------------
        self.spinRotA = QtWidgets.QDoubleSpinBox()
        self.spinRotA.setRange(-3600.0, 3600.0); self.spinRotA.setDecimals(2)
        self.spinRotA.setSingleStep(0.5); self.spinRotA.setSuffix("°"); self.spinRotA.setValue(0.0)
        self.spinRotB = QtWidgets.QDoubleSpinBox()
        self.spinRotB.setRange(-3600.0, 3600.0); self.spinRotB.setDecimals(2)
        self.spinRotB.setSingleStep(0.5); self.spinRotB.setSuffix("°"); self.spinRotB.setValue(0.0)
        self.spinRotA.valueChanged.connect(lambda _: self.on_user_orientation_changed('A'))
        self.spinRotB.valueChanged.connect(lambda _: self.on_user_orientation_changed('B'))
        f.addRow(QtWidgets.QLabel("Rotation A (deg):"), self.spinRotA)
        f.addRow(QtWidgets.QLabel("Rotation B (deg):"), self.spinRotB)

        self.chkFlipHA = QtWidgets.QCheckBox("Flip A horizontally")
        self.chkFlipVA = QtWidgets.QCheckBox("Flip A vertically")
        self.chkFlipHB = QtWidgets.QCheckBox("Flip B horizontally")
        self.chkFlipVB = QtWidgets.QCheckBox("Flip B vertically")
        self.chkFlipHA.toggled.connect(lambda _: self.on_user_orientation_changed('A'))
        self.chkFlipVA.toggled.connect(lambda _: self.on_user_orientation_changed('A'))
        self.chkFlipHB.toggled.connect(lambda _: self.on_user_orientation_changed('B'))
        self.chkFlipVB.toggled.connect(lambda _: self.on_user_orientation_changed('B'))
        f.addRow(self.chkFlipHA, self.chkFlipVA)
        f.addRow(self.chkFlipHB, self.chkFlipVB)

        # ---------------- Z selection: entry boxes (not dropdowns) ----------------
        self.editZA = QtWidgets.QLineEdit(); self.editZB = QtWidgets.QLineEdit()
        self.editZA.setPlaceholderText("type z (e.g., 82)")
        self.editZB.setPlaceholderText("type z (e.g., 82)")
        self.editZA.setValidator(QtGui.QDoubleValidator()); self.editZB.setValidator(QtGui.QDoubleValidator())
        self.editZA.editingFinished.connect(lambda: self.change_z_text('A'))
        self.editZB.editingFinished.connect(lambda: self.change_z_text('B'))
        f.addRow(QtWidgets.QLabel("Z (A):"), self.editZA)
        f.addRow(QtWidgets.QLabel("Z (B):"), self.editZB)

        # ---------------- Live transform only: overlay show/hide ----------------
        self.chkShowOverlay = QtWidgets.QCheckBox("Show overlay / transform pane")
        self.chkShowOverlay.setChecked(True)
        self.chkShowOverlay.toggled.connect(self.toggle_overlay_visible)
        f.addRow(self.chkShowOverlay)

        self.chkCompositeAdd = QtWidgets.QCheckBox("Single composite (sum colors)")
        self.chkCompositeAdd.setChecked(True)
        self.chkCompositeAdd.toggled.connect(self.update_overlay_view)
        f.addRow(self.chkCompositeAdd)


        # --- Manual T/R/S override ---
        self.chkManualTRS = QtWidgets.QCheckBox("Manual T / R / S override")
        self.chkManualTRS.setChecked(False)
        f.addRow(self.chkManualTRS)

        rowTRS = QtWidgets.QWidget(); hTRS = QtWidgets.QGridLayout(rowTRS)
        hTRS.setContentsMargins(0,0,0,0); hTRS.setHorizontalSpacing(8); hTRS.setVerticalSpacing(4)

        self.spinTx = QtWidgets.QDoubleSpinBox(); self.spinTx.setRange(-1e6, 1e6); self.spinTx.setDecimals(3)
        self.spinTy = QtWidgets.QDoubleSpinBox(); self.spinTy.setRange(-1e6, 1e6); self.spinTy.setDecimals(3)
        self.spinScale = QtWidgets.QDoubleSpinBox(); self.spinScale.setRange(1e-4, 1e4); self.spinScale.setDecimals(6); self.spinScale.setValue(1.0)
        self.spinTheta = QtWidgets.QDoubleSpinBox(); self.spinTheta.setRange(-3600.0, 3600.0); self.spinTheta.setDecimals(3); self.spinTheta.setSuffix("°")

        hTRS.addWidget(QtWidgets.QLabel("Tx"), 0, 0); hTRS.addWidget(self.spinTx,   0, 1)
        hTRS.addWidget(QtWidgets.QLabel("Ty"), 0, 2); hTRS.addWidget(self.spinTy,   0, 3)
        hTRS.addWidget(QtWidgets.QLabel("Scale"), 1, 0); hTRS.addWidget(self.spinScale, 1, 1)
        hTRS.addWidget(QtWidgets.QLabel("Angle"), 1, 2); hTRS.addWidget(self.spinTheta, 1, 3)
        f.addRow(rowTRS)

        # Disable the fields when override is off
        for w in (self.spinTx, self.spinTy, self.spinScale, self.spinTheta):
            w.setEnabled(False)

        # Re-render overlay when manual values change (only if override is ON)
        def _maybe_update_overlay(_):
            if self.chkManualTRS.isChecked():
                self.update_overlay_view()

        self.spinTx.valueChanged.connect(_maybe_update_overlay)
        self.spinTy.valueChanged.connect(_maybe_update_overlay)
        self.spinScale.valueChanged.connect(_maybe_update_overlay)
        self.spinTheta.valueChanged.connect(_maybe_update_overlay)

        def _on_override_toggled(checked: bool):
            for w in (self.spinTx, self.spinTy, self.spinScale, self.spinTheta):
                w.setEnabled(checked)
            # When turning ON, seed fields from the latest auto so user can fine-tune
            if checked and hasattr(self, "_last_auto_sRt") and (self._last_auto_sRt is not None):
                s, R, t = self._last_auto_sRt
                self._set_ui_from_srt(s, R, t, block_signals=True)
            self.update_overlay_view()

        self.chkManualTRS.toggled.connect(_on_override_toggled)



        # ---------------- Auto-match (IoU) ----------------
        self.btnHungarian = QtWidgets.QPushButton("Auto-match (IoU ≥ threshold, current slices)")
        self.btnHungarian.clicked.connect(self.run_auto_iou)  # <-- new handler
        self.btnHungarian.setEnabled(False)

        self.spinMaxDist = QtWidgets.QDoubleSpinBox()
        self.spinMaxDist.setRange(0.0, 1.0)          # IoU in [0,1]
        self.spinMaxDist.setDecimals(2)
        self.spinMaxDist.setSingleStep(0.05)
        self.spinMaxDist.setValue(0.50)              # default 0.50 IoU
        self.spinMaxDist.setSuffix(" min IoU")       # <-- label reflects IoU now

        f.addRow(self.btnHungarian, self.spinMaxDist)


        # ---------------- Pairs table ----------------
        self.pairTable = QtWidgets.QTableWidget(0, 3)
        self.pairTable.setHorizontalHeaderLabels(["A (z,ROI)", "B (z,ROI)","Cell ID"])
        self.pairTable.horizontalHeader().setStretchLastSection(True)
        self.pairTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.pairTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        f.addRow(QtWidgets.QLabel("<b>Pairs</b>")); f.addRow(self.pairTable)

        # Hold intersection items keyed by (akey, bkey)
        self._overlay_intersections: Dict[Tuple[ROIKey, ROIKey], List[QtWidgets.QGraphicsPathItem]] = {}

        # React to user selecting a pair row
        self.pairTable.itemSelectionChanged.connect(self._on_pair_selection_changed)

        # Pair table actions
        hbtns = QtWidgets.QHBoxLayout()
        self.btnRemovePair = QtWidgets.QPushButton("Remove Selected Pair")
        self.btnClearPairs = QtWidgets.QPushButton("Clear All Pairs")
        self.btnRemovePair.clicked.connect(self.remove_selected_pair)
        self.btnClearPairs.clicked.connect(self.clear_pairs)
        hcont = QtWidgets.QWidget(); hcont.setLayout(hbtns)
        hbtns.addWidget(self.btnRemovePair); hbtns.addWidget(self.btnClearPairs)
        f.addRow(hcont)

        # Export
        self.btnExport = QtWidgets.QPushButton("Export Labeled CSVs…")
        self.btnExport.clicked.connect(self.export_labeled_csvs)
        self.btnExport.setEnabled(False)
        f.addRow(self.btnExport)

        # Status
        self.status = QtWidgets.QLabel("")
        f.addRow(self.status)

        # Graphics state
        self.imgItemA: Optional[pg.ImageItem] = None
        self.imgItemB: Optional[pg.ImageItem] = None
        self.centroidsA: Optional[ClickableScatter] = None
        self.centroidsB: Optional[ClickableScatter] = None
        self.labelsA: List[pg.TextItem] = []
        self.labelsB: List[pg.TextItem] = []
        self.overlay_B_on_A = None
        self.hullsA: Dict[int, QtGui.QPainterPath] = {}
        self.hullsB: Dict[int, QtGui.QPainterPath] = {}
        self.selOutlineA = None
        self.selOutlineB = None

        # Colors (current z)
        self.colorA: Dict[int, QtGui.QColor] = {}
        self.colorB: Dict[int, QtGui.QColor] = {}
    # ------------------------ Utility ------------------------
    def _atlas_new(self):
        self.atlas = Atlas()
        self.status.setText("Created new empty atlas.")
        # keep existing dataset labels so we can start filling rows

    def _atlas_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Atlas CSV", os.getcwd(), "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.atlas = Atlas.from_csv(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Atlas load error", str(e)); return
        self.status.setText(f"Loaded atlas: {os.path.basename(path)}")

        # Prompt labels for any loaded datasets and try to apply pairs
        for which in ('A','B'):
            ds_loaded = (self.datasetA.df is not None) if which=='A' else (self.datasetB.df is not None)
            if ds_loaded:
                self._prompt_dataset_label(which)

        la = self.dataset_labels.get('A'); lb = self.dataset_labels.get('B')
        if la and (la not in self.atlas.columns):
            self.atlas.ensure_column(la)
        if lb and (lb not in self.atlas.columns):
            self.atlas.ensure_column(lb)

        if la and lb:
            # Ask user whether to load pairs (replace/merge)
            has_pairs = (len(self.A2B) > 0)
            default = QtWidgets.QMessageBox.No if has_pairs else QtWidgets.QMessageBox.Yes
            choice = QtWidgets.QMessageBox.question(
                self, "Apply atlas pairs?",
                "Apply atlas pairs for the current A & B labels to this session?\n"
                "Yes = REPLACE current pairs, No = MERGE (add non-conflicting).",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                default
            )
            if choice != QtWidgets.QMessageBox.Cancel:
                mode = 'replace' if choice == QtWidgets.QMessageBox.Yes else 'merge'
                self._apply_pairs_from_atlas(la, lb, mode)

        # Also ingest any current matches into the atlas (fills missing cells/rows)
        self._atlas_ingest_current_pairs()

    def _atlas_export(self):
        if not self.atlas.columns:
            QtWidgets.QMessageBox.information(self, "Empty atlas", "Nothing to export yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Atlas CSV", os.getcwd(), "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.atlas.save_csv(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Atlas export error", str(e)); return
        self.status.setText(f"Exported atlas to {os.path.basename(path)}")

    def _prompt_dataset_label(self, which: str) -> Optional[str]:
        current = self.dataset_labels.get(which) or ""
        text, ok = QtWidgets.QInputDialog.getText(self, "Dataset label",
                                                  f"Enter atlas label for dataset {which} (column name):",
                                                  QtWidgets.QLineEdit.Normal, current)
        if not ok: 
            return None
        label = text.strip()
        if not label:
            return None
        self.dataset_labels[which] = label
        self.atlas.ensure_column(label)
        return label

    def _atlas_after_dataset_loaded(self, which: str):
        """Called whenever A or B finishes loading (CSV / Suite2P)."""
        # Ask user to name this dataset for the atlas
        label = self._prompt_dataset_label(which)
        if not label:
            return  # atlas still exists; dataset just not labeled yet

        # If label already exists in atlas, we may be able to prefill matches
        # If both datasets are labeled, offer to load/merge/replace pairs from the atlas.
        other = 'B' if which == 'A' else 'A'
        label_other = self.dataset_labels.get(other)

        if label_other:
            # Check if the atlas has any rows containing either column
            has_any_for_both = any( (row.get(label) and row.get(label_other)) for row in self.atlas.rows.values() )
            if has_any_for_both:
                # If user has already made matches, ask merge/replace
                has_pairs = (len(self.A2B) > 0)
                if has_pairs:
                    choice = QtWidgets.QMessageBox.question(
                        self, "Atlas pairs found",
                        "This atlas already contains matches for these two dataset labels.\n"
                        "Do you want to REPLACE your current matches with atlas pairs?\n"
                        "Choose 'No' to MERGE (keep yours, add missing atlas pairs).",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                        QtWidgets.QMessageBox.No
                    )
                    if choice == QtWidgets.QMessageBox.Cancel:
                        return
                    mode = 'replace' if choice == QtWidgets.QMessageBox.Yes else 'merge'
                else:
                    mode = 'replace'  # nothing made yet, just load atlas pairs

                self._apply_pairs_from_atlas(labelA=self.dataset_labels['A'],
                                             labelB=self.dataset_labels['B'],
                                             mode=mode)

        # If label was new to atlas, fold in any existing tool pairs
        self._atlas_ingest_current_pairs()


    def _apply_pairs_from_atlas(self, labelA: Optional[str], labelB: Optional[str], mode: str):
        """Load pairs for current z-slices from atlas rows where both columns are populated."""
        if not labelA or not labelB:
            return
        zA = self.datasetA.current_z; zB = self.datasetB.current_z
        added = 0
        atlas_pairs = []
        for cid, row in self.atlas.rows.items():
            a = self.atlas.parse_roikey(row.get(labelA, ""))
            b = self.atlas.parse_roikey(row.get(labelB, ""))
            if not a or not b:
                continue
            za, ra = a; zb, rb = b
            if (np.isclose(za, zA) and np.isclose(zb, zB)):
                atlas_pairs.append( ((za,int(ra)), (zb,int(rb)), cid) )

        if mode == 'replace':
            self.clear_pairs()

        # Apply pairs (respect 1:1)
        for (az, arid), (bz, brid), cid in atlas_pairs:
            akey = (az, arid); bkey = (bz, brid)
            if akey in self.A2B or bkey in self.B2A:
                continue
            self.A2B[akey] = bkey
            self.B2A[bkey] = akey
            # also set dataset 'label' = Common_ID
            self.datasetA.set_label(az, arid, cid)
            self.datasetB.set_label(bz, brid, cid)
            added += 1

        if added:
            self.update_pair_table(); self.refresh_controls(); self.update_overlay_view()
            self.status.setText(f"Loaded {added} pair(s) from atlas for zA={zA}, zB={zB}")

    def _atlas_ingest_current_pairs(self):
        la = self.dataset_labels.get('A'); lb = self.dataset_labels.get('B')
        if not la and not lb:
            return

        for akey, bkey in list(self.A2B.items()):
            # New guard: skip stale pairs that don't exist in the current files
            if not self._roi_exists_in_dataset(self.datasetA, akey): 
                continue
            if not self._roi_exists_in_dataset(self.datasetB, bkey): 
                continue

            (az, arid), (bz, brid) = akey, bkey

            if la and lb:
                cid = self.atlas.add_pair(la, az, arid, lb, bz, brid, prefer_existing='atlas')
                # Force both datasets’ labels to the atlas Common_ID (prevents drift/dupes)
                self.datasetA.set_label(az, arid, cid)
                self.datasetB.set_label(bz, brid, cid)
            elif la:
                cid = self.atlas.add_pair(la, az, arid, la, az, arid)
                self.datasetA.set_label(az, arid, cid)
            elif lb:
                cid = self.atlas.add_pair(lb, bz, brid, lb, bz, brid)
                self.datasetB.set_label(bz, brid, cid)


    def _atlas_add_pair_incremental(self, akey: ROIKey, bkey: ROIKey):
        """Call this whenever the user creates a new pair."""
        la = self.dataset_labels.get('A'); lb = self.dataset_labels.get('B')
        if not la or not lb:
            return
        (az, arid), (bz, brid) = akey, bkey
        cid = self.atlas.add_pair(la, az, arid, lb, bz, brid, prefer_existing='atlas')
        # sync dataset 'label' fields
        self.datasetA.set_label(az, arid, cid)
        self.datasetB.set_label(bz, brid, cid)


    def _outline_from_xy_pixels(self, xpix: np.ndarray, ypix: np.ndarray,
                                H: int, W: int) -> np.ndarray:
        """
        Build a 2D polygon outline from Suite2P ROI pixels.
        Returns Nx2 float array in the SAME oriented frame as your pre-oriented images
        (we apply the same CW rot + horiz flip you used for rasters, which is
        equivalent to swapping x/y).
        """
        xpix = np.asarray(xpix).astype(int)
        ypix = np.asarray(ypix).astype(int)
        if xpix.size == 0:
            return np.zeros((0, 2), dtype=float)

        # Preferred: exact contour via marching squares
        if SKIMAGE_AVAILABLE:
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[ypix, xpix] = 1
            pad = np.pad(mask, 1, mode='constant')
            contours = skmeasure.find_contours(pad, level=0.5)
            if len(contours) > 0:
                # pick longest contour (outer boundary)
                c = max(contours, key=lambda a: a.shape[0])
                c = c - 1.0  # undo padding
                # skimage gives (row, col) = (y, x). Convert to (x, y):
                poly = np.stack([c[:,1], c[:,0]], axis=1)
            else:
                poly = np.stack([xpix, ypix], axis=1).astype(float)
        # Fallback: convex hull of pixels (coarser but dependency-free)
        elif HULL_AVAILABLE and xpix.size >= 3:
            pts = np.stack([xpix, ypix], axis=1).astype(float)
            try:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
            except Exception:
                poly = pts
        else:
            pts = np.stack([xpix, ypix], axis=1).astype(float)
            if pts.shape[0] == 0:
                return np.zeros((0,2), float)
            mn = pts.min(axis=0); mx = pts.max(axis=0)
            poly = np.array([[mn[0], mn[1]],[mx[0], mn[1]],[mx[0], mx[1]],[mn[0], mx[1]]], float)

        # IMPORTANT: apply the same one-time raster orientation you do for TIFs
        # (90° CW + horizontal flip). For points this is equivalent to swapping axes.
        # See your S matrix derivation; the net effect is (x, y) -> (y, x).
        poly = poly[:, ::-1]  # swap columns

        return poly


    def _load_suite2p_stat_into_dataset(self, dataset: Dataset, stat_path: str, ops_path: Optional[str] = None):
        """
        Read Suite2P stat.npy (and optionally ops.npy for exact image size),
        extract an outline polygon per ROI, and populate the Dataset the same way
        your CSV loader does (roi_to_points_by_z, centroids, df, etc).
        """
        # Read stat
        stats = np.load(stat_path, allow_pickle=True)
        stats_list = list(stats) if isinstance(stats, (list, tuple)) else list(stats.ravel())

        # Image size (preferred from ops.npy; else from max coords)
        H = W = None
        if ops_path and os.path.exists(ops_path):
            ops = np.load(ops_path, allow_pickle=True).item()
            H = int(ops.get('Ly', 0)) or None
            W = int(ops.get('Lx', 0)) or None
        if H is None or W is None:
            all_x = np.concatenate([np.asarray(s['xpix']).ravel() for s in stats_list]) if stats_list else np.array([0])
            all_y = np.concatenate([np.asarray(s['ypix']).ravel() for s in stats_list]) if stats_list else np.array([0])
            W = int(all_x.max()) + 2
            H = int(all_y.max()) + 2

        # Reset dataset and fill
        dataset.clear()
        dataset.name = os.path.basename(stat_path)
        z = 0.0
        dataset.z_values = [z]
        dataset.roi_to_points_by_z = {z: {}}
        dataset.roi_centroids_xy_by_z = {z: {}}
        dataset.labels_by_z = {z: {}}
        dataset.has_labels = False

        rows = []
        rid_counter = 1
        for s in stats_list:
            xpix = np.asarray(s.get('xpix', []))
            ypix = np.asarray(s.get('ypix', []))
            if xpix.size == 0:
                continue
            poly = self._outline_from_xy_pixels(xpix, ypix, H, W)
            if poly.shape[0] < 3:
                continue

            rid = rid_counter; rid_counter += 1
            pts3 = np.c_[poly, np.full((poly.shape[0],), z, dtype=float)]
            dataset.roi_to_points_by_z[z][rid] = pts3
            dataset.roi_centroids_xy_by_z[z][rid] = poly.mean(axis=0)

            # populate df rows so the rest of your pipeline works unchanged
            for (x, y) in poly:
                rows.append({"x": float(x), "y": float(y), "z": z, "ROI_ID": rid, "label": 0})

        dataset.df = pd.DataFrame(rows, columns=["x", "y", "z", "ROI_ID", "label"])
        dataset.set_current_z(z)


    def _load_suite2p_ops_array(self, ops_path: str) -> np.ndarray:
        """Return 2D or 3D array from ops.npy (meanImgE preferred), no pre-orientation here."""
        raw = np.load(ops_path, allow_pickle=True)
        # ops can be dict, 0-d object array of dict, or 1-d array/list of dicts (multi-plane)
        def to_ops_list(obj):
            if isinstance(obj, dict):
                return [obj]
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                if obj.ndim == 0:
                    return [obj.item()]
                if obj.ndim == 1:
                    return list(obj)
            if isinstance(obj, list):
                return obj
            raise ValueError("Unrecognized ops.npy format")

        ops_list = to_ops_list(raw)

        imgs = []
        for op in ops_list:
            # img = op.get('meanImgE', None)
            img = None
            if img is None:
                img = op.get('meanImg', None)
            if img is None:
                continue
            img = np.asarray(img, dtype=np.float32)
            imgs.append(img)

        if not imgs:
            raise ValueError("ops.npy has no meanImg/meanImgE")

        if len(imgs) == 1:
            return imgs[0]
        else:
            # stack along Z: (Z,H,W)
            # ensure same shape
            H, W = imgs[0].shape
            imgs = [im if im.shape == (H, W) else np.resize(im, (H, W)) for im in imgs]
            return np.stack(imgs, axis=0).astype(np.float32)



    def _purge_pairs_for_dataset(self, which: str):
        """Remove pairs tied to the dataset being (re)loaded, keep the other side's labels intact."""
        if which == 'A':
            # remove all A-side keys
            for akey, bkey in list(self.A2B.items()):
                del self.A2B[akey]
                if bkey in self.B2A:
                    del self.B2A[bkey]
        else:
            for bkey, akey in list(self.B2A.items()):
                del self.B2A[bkey]
                if akey in self.A2B:
                    del self.A2B[akey]
        self.update_pair_table()
        self.refresh_controls()
        self.update_overlay_view()

    def _roi_exists_in_dataset(self, ds: Dataset, key: ROIKey) -> bool:
        z, rid = key
        z = norm_z(z); rid = int(rid)
        return (z in (ds.roi_to_points_by_z or {})) and (rid in (ds.roi_to_points_by_z[z] or {}))


    def load_masks(self, dataset: Dataset, label_widget: QtWidgets.QLabel, which: str):
        """CSV (legacy) or Suite2P stat.npy."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load masks for {dataset.name or which}", os.getcwd(),
            "Masks (*.csv *.npy);;CSV (*.csv);;Suite2P stat (*.npy)"
        )
        if not path:
            return

        try:
            if path.lower().endswith(".csv"):
                dataset.load_csv(path)
            elif path.lower().endswith(".npy"):
                self._load_suite2p_stat_into_dataset(dataset, path)
            else:
                raise ValueError("Unsupported masks file. Use .csv or Suite2P stat.npy")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return

        
        # clear pairs tied to this dataset (keep the other dataset's labels for propagation)
        self._purge_pairs_for_dataset(which)
        self._atlas_after_dataset_loaded(which)

        # show filename
        label_widget.setText(os.path.basename(path))



        # draw views
        if which == 'A':
            self.editZA.setText(str(dataset.current_z) if dataset.current_z is not None else "")
            self.populate_view_A()
        else:
            self.editZB.setText(str(dataset.current_z) if dataset.current_z is not None else "")
            self.populate_view_B()

        self.status.setText(
            f"Loaded masks {os.path.basename(path)} → {len(dataset.df) if dataset.df is not None else 0} points, "
            f"{len(dataset.z_values)} z-levels"
        )
        self.refresh_controls()
        self.update_overlay_view()

        # If both datasets present, try autopair by labels (still works with CSV)
        if self.datasetA.df is not None and self.datasetB.df is not None:
            self._autopair_by_labels()
        

    def load_image(self, dataset: Dataset, label_widget: QtWidgets.QLabel, which: str):
        """TIFF (legacy) or Suite2P ops.npy."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load image for {dataset.name or which}", os.getcwd(),
            "Images (*.tif *.tiff *.npy);;TIFF (*.tif *.tiff);;Suite2P ops (*.npy)"
        )
        if not path:
            return

        try:
            if path.lower().endswith((".tif", ".tiff")):
                dataset.load_tif(path)
                # match orientation used elsewhere
                dataset.tif_stack = self._pre_orient_tif_stack(dataset.tif_stack)
            elif path.lower().endswith(".npy"):
                arr = self._load_suite2p_ops_array(path)  # raw array(s)
                dataset.tif_stack = arr                  # ← no pre-orient for Suite2P
                dataset.tif_path = path
            else:
                raise ValueError("Unsupported image file. Use .tif/.tiff or Suite2P ops.npy")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Image load error", str(e))
            return

        # Seed z if needed (for 2D: 0)
        if dataset.current_z is None:
            dataset.set_current_z(0.0)
            if which == 'A': self.editZA.setText(str(dataset.current_z))
            else:            self.editZB.setText(str(dataset.current_z))

        label_widget.setText(os.path.basename(path))

        # Try to fit & render immediately
        try:
            self.auto_fit_image_to_points(which)
        except Exception:
            pass

        if which == 'A':
            self.chkShowImgA.blockSignals(True); self.chkShowImgA.setChecked(True); self.chkShowImgA.blockSignals(False)
            self.populate_view_A()
        else:
            self.chkShowImgB.blockSignals(True); self.chkShowImgB.setChecked(True); self.chkShowImgB.blockSignals(False)
            self.populate_view_B()

        self.update_overlay_view()
        self.status.setText(f"Loaded image for {dataset.name or which}: {os.path.basename(path)}")

    def _levels_for_array(self, which: str, slc: np.ndarray, z: float) -> Tuple[float, float]:
        if slc is None:
            return (0.0, 1.0)
        self._ensure_base_levels_for_z(which, z, slc)  # seeds caches, robust range
        v = slc[np.isfinite(slc)].astype(np.float32)
        if v.size == 0:
            return (0.0, 1.0)

        wide_lo, wide_hi   = float(np.percentile(v, 0.2)),  float(np.percentile(v, 99.8))
        tight_lo, tight_hi = float(np.percentile(v, 10.0)), float(np.percentile(v, 90.0))
        if tight_hi <= tight_lo:
            tight_lo, tight_hi = wide_lo, wide_hi
        if wide_hi <= wide_lo:
            wide_lo, wide_hi = float(np.min(v)), float(np.max(v))
            if wide_hi <= wide_lo:
                wide_hi = wide_lo + 1.0

        f   = self._get_intensity_factor(which)      # 0.05..5.0
        mix = (f - 0.05) / (5.0 - 0.05)
        mix = max(0.0, min(1.0, mix))
        lo = (1.0 - mix) * wide_lo + mix * tight_lo
        hi = (1.0 - mix) * wide_hi + mix * tight_hi

        full = float(np.max(v) - np.min(v))
        minw = max(1e-6, 0.001 * full)
        if (hi - lo) < minw:
            c = 0.5 * (hi + lo)
            lo, hi = c - 0.5 * minw, c + 0.5 * minw
        return (float(lo), float(hi))

    def _normalize_image(self, which: str, slc: np.ndarray, z: float) -> np.ndarray:
        lo, hi = self._levels_for_array(which, slc, z)
        arr = (slc.astype(np.float32) - lo) / max(hi - lo, 1e-6)
        return np.clip(arr, 0.0, 1.0)

    def _world_to_indices(self, X: np.ndarray, Y: np.ndarray,
                        rect: QtCore.QRectF, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map world coords (X,Y) to (row i, col j) indices of an image placed by `rect`.
        Uses pixel centers, so j = round((x - x0) * W/rect.w - 0.5).
        """
        sx = W / max(rect.width(), 1e-12)
        sy = H / max(rect.height(), 1e-12)
        j = np.rint((X - rect.x()) * sx - 0.5).astype(int)  # columns (x)
        i = np.rint((Y - rect.y()) * sy - 0.5).astype(int)  # rows (y)
        return i, j


    def _draw_additive_composite(self):
        """
        Build ONE RGB image on A's canvas by resampling B into A using the
        current B→A similarity, then additively blending:
            A -> blue,  B -> orange, overlap -> yellow.

        Only includes A and/or B if their "Show image" checkboxes are checked.
        """
        # --- Respect 'Show image (A/B)' checkboxes ---
        useA = self.chkShowImgA.isChecked()
        useB = self.chkShowImgB.isChecked()

        # If neither should draw, remove existing composite and bail
        if not (useA or useB):
            if hasattr(self, "_additiveItem") and self._additiveItem is not None:
                try: self.viewOverlay.removeItem(self._additiveItem)
                except Exception: pass
                self._additiveItem = None
            return

        # --- Guards (keep existing expectations: both stacks present) ---
        if self.datasetA.tif_stack is None or self.datasetB.tif_stack is None:
            return
        slcA = self.oriented_slice(self.datasetA, 'A')
        slcB = self.oriented_slice(self.datasetB, 'B')
        if slcA is None or slcB is None:
            return

        # --- Rects as-is (A is the canvas) ---
        rectA = self.imgRectA_by_z.get(self.datasetA.current_z)
        if rectA is None:
            Ha, Wa = slcA.shape[:2]
            rectA = QtCore.QRectF(-0.5, -0.5, Wa, Ha)
        else:
            Ha, Wa = slcA.shape[:2]

        rectB = self.imgRectB_by_z.get(self.datasetB.current_z)
        if rectB is None:
            Hb, Wb = slcB.shape[:2]
            rectB = QtCore.QRectF(-0.5, -0.5, Wb, Hb)
        else:
            Hb, Wb = slcB.shape[:2]

        # --- Transform choice (unchanged) ---
        if self.chkManualTRS.isChecked():
            sRt = self._srt_from_ui()
        else:
            sRt = self.dynamic_transform_from_pairs()

        # Pre-orientation matrix (unchanged)
        S = np.array([[0.0, 1.0],
                    [1.0, 0.0]], dtype=float)

        if sRt is None:
            s, R_img, t_img = 1.0, np.eye(2), np.zeros(2, dtype=float)
        else:
            s_raw, R_raw, t_raw = sRt
            R_img = S @ R_raw @ S
            t_img = S @ t_raw
            s = float(s_raw)

        # --- World grid on A’s canvas (unchanged) ---
        xs = rectA.x() + (np.arange(Wa, dtype=float) + 0.5) * (rectA.width()  / Wa)
        ys = rectA.y() + (np.arange(Ha, dtype=float) + 0.5) * (rectA.height() / Ha)
        X, Y = np.meshgrid(xs, ys)                 # (Ha, Wa)
        XY = np.vstack([X.ravel(), Y.ravel()])     # (2, Ha*Wa)

        # --- Map A centers → B image frame (unchanged math) ---
        XYB = (1.0 / max(s, 1e-12)) * (R_img.T @ (XY - t_img.reshape(2, 1)))
        XB = XYB[0, :].reshape(Ha, Wa)
        YB = XYB[1, :].reshape(Ha, Wa)

        # --- Convert world coords to B fractional indices (unchanged) ---
        dxB = rectB.width()  / Wb
        dyB = rectB.height() / Hb
        ix = (XB - rectB.x()) / max(dxB, 1e-12) - 0.5
        iy = (YB - rectB.y()) / max(dyB, 1e-12) - 0.5

        # --- Normalization helpers (unchanged) ---
        def _levels_for(which: str, slc: np.ndarray) -> Tuple[float, float]:
            v = slc.astype(np.float32, copy=False)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return 0.0, 1.0
            p = np.percentile
            wide_lo, wide_hi   = float(p(v, 0.2)),  float(p(v, 99.8))
            tight_lo, tight_hi = float(p(v, 10.0)), float(p(v, 90.0))
            if tight_hi <= tight_lo: tight_lo, tight_hi = wide_lo, wide_hi
            if wide_hi <= wide_lo:
                wide_lo, wide_hi = float(np.min(v)), float(np.max(v))
            f = self._get_intensity_factor(which)  # 0.05..5.00 → 0..1
            mix = (f - 0.05) / (5.0 - 0.05); mix = max(0.0, min(1.0, mix))
            lo = (1.0 - mix) * wide_lo + mix * tight_lo
            hi = (1.0 - mix) * wide_hi + mix * tight_hi
            if hi <= lo: hi = lo + 1.0
            return float(lo), float(hi)

        # --- Normalize A (only if enabled) ---
        A01 = None
        if useA:
            loA, hiA = _levels_for('A', slcA)
            A01 = np.clip((slcA.astype(np.float32) - loA) / (hiA - loA), 0.0, 1.0)
        opaA = self._get_opacity('A') if useA else 0.0

        # --- Sample & normalize B (only if enabled) ---
        B01 = None
        opaB = 0.0
        if useB:
            # Bilinear sampling
            ix0 = np.floor(ix).astype(np.int32); iy0 = np.floor(iy).astype(np.int32)
            ix1 = ix0 + 1;                         iy1 = iy0 + 1
            wx = ix - ix0;                         wy = iy - iy0

            def clip_idx(i, imax): return np.clip(i, 0, imax - 1)
            ix0c = clip_idx(ix0, Wb); ix1c = clip_idx(ix1, Wb)
            iy0c = clip_idx(iy0, Hb); iy1c = clip_idx(iy1, Hb)

            v00 = slcB[iy0c, ix0c]; v10 = slcB[iy0c, ix1c]
            v01 = slcB[iy1c, ix0c]; v11 = slcB[iy1c, ix1c]
            w00 = (1.0 - wx) * (1.0 - wy); w10 = wx * (1.0 - wy)
            w01 = (1.0 - wx) * wy;         w11 = wx * wy

            sampB = (w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11)

            # true OOB → zero
            oob = (ix < 0) | (iy < 0) | (ix > (Wb - 1)) | (iy > (Hb - 1))
            sampB[oob] = 0.0

            loB, hiB = _levels_for('B', slcB)
            B01 = np.clip((sampB.astype(np.float32) - loB) / (hiB - loB), 0.0, 1.0)
            opaB = self._get_opacity('B')

        # --- Colorize & additive blend (now gated) ---
        comp = np.zeros((Ha, Wa, 3), dtype=np.float32)
        if useA:
            comp[..., 2] += (A01 * opaA)              # A → blue
        if useB:
            comp[..., 0] += (B01 * opaB)              # B → red
            comp[..., 1] += (B01 * opaB * 0.55)       # B → green (to make orange)
        np.clip(comp, 0.0, 1.0, out=comp)

        rgb8 = (comp * 255.0 + 0.5).astype(np.uint8)

        # --- Show it in the overlay as a single ImageItem pinned to A's rect ---
        if hasattr(self, "_additiveItem") and self._additiveItem is not None:
            try:
                self.viewOverlay.removeItem(self._additiveItem)
            except Exception:
                pass
            self._additiveItem = None

        img = pg.ImageItem(rgb8)
        img.setRect(rectA)
        img.setZValue(-210)  # beneath outlines & intersection patches
        self.viewOverlay.addItem(img)
        self._additiveItem = img


   

    def _on_intersection_clicked(self, key: Tuple[ROIKey, ROIKey]):
        """Select the pair row and highlight the two ROIs in A/B views."""
        akey, bkey = key
        # select row in the pairs table
        for r in range(self.pairTable.rowCount()):
            aitem = self.pairTable.item(r, 0); bitem = self.pairTable.item(r, 1)
            if not aitem or not bitem:
                continue
            if aitem.data(QtCore.Qt.UserRole) == akey and bitem.data(QtCore.Qt.UserRole) == bkey:
                self.pairTable.setCurrentCell(r, 0)
                self.pairTable.selectRow(r)
                break

        # highlight outlines in A/B using existing function
        (az, arid), (bz, brid) = akey, bkey
        self._clear_highlight('A'); self._clear_highlight('B')
        self._highlight_roi('A', int(arid))
        self._highlight_roi('B', int(brid))


    def _next_global_label(self) -> int:
        """Next label = 1 + max label seen across both datasets (0 if none)."""
        def _maxlab(ds: Dataset) -> int:
            if ds.df is None or 'label' not in ds.df.columns or ds.df['label'].empty:
                return 0
            try:
                return int(pd.to_numeric(ds.df['label'], errors='coerce').fillna(0).max())
            except Exception:
                return 0
        return max(_maxlab(self.datasetA), _maxlab(self.datasetB)) + 1

    def _assign_or_propagate_label_for_pair(self, akey: ROIKey, bkey: ROIKey):
        az, arid = akey; bz, brid = bkey
        labA = self.datasetA.get_label(az, arid)
        labB = self.datasetB.get_label(bz, brid)
        if labA > 0 and labB == 0:
            self.datasetB.set_label(bz, brid, labA)
        elif labB > 0 and labA == 0:
            self.datasetA.set_label(az, arid, labB)
        elif labA == 0 and labB == 0:
            new_lab = self._next_global_label()
            self.datasetA.set_label(az, arid, new_lab)
            self.datasetB.set_label(bz, brid, new_lab)
        # if both >0 and disagree, leave as-is

    def _resolve_pair_label(self, akey, bkey) -> int:
        az, arid = akey; bz, brid = bkey
        labA = self.datasetA.get_label(az, arid)
        labB = self.datasetB.get_label(bz, brid)
        return labA if labA > 0 else (labB if labB > 0 else 0)


    def _autopair_by_labels(self):
        """If BOTH datasets have labels, auto-create pairs where labels match (per-z)."""
        if not (self.datasetA.has_labels and self.datasetB.has_labels):
            return
        made = 0
        # Use intersection of z-levels that exist in both
        zs = sorted(set(self.datasetA.z_values) & set(self.datasetB.z_values))
        for z in zs:
            labA = self.datasetA.labels_by_z.get(z, {})
            labB = self.datasetB.labels_by_z.get(z, {})
            if not labA or not labB: 
                continue
            # invert: label -> rid (assume labels are unique per session)
            invA = {lab: rid for rid, lab in labA.items() if lab > 0}
            invB = {lab: rid for rid, lab in labB.items() if lab > 0}

            common = set(invA.keys()) & set(invB.keys())
            for lab in sorted(common):
                akey = (z, int(invA[lab]))
                bkey = (z, int(invB[lab]))
                # enforce 1:1
                if akey in self.A2B and self.A2B[akey] != bkey:
                    continue
                if bkey in self.B2A and self.B2A[bkey] != akey:
                    continue
                if akey not in self.A2B and bkey not in self.B2A:
                    self.A2B[akey] = bkey
                    self.B2A[bkey] = akey
                    made += 1
        if made:
            self.status.setText(f"Auto-loaded {made} pairs from existing registration labels.")
            self.update_pair_table()
            self.refresh_controls()
            self.update_overlay_view()


    def _on_pair_selection_changed(self):
        # reset everything to "normal" style
        normal_pen   = QtGui.QPen(QtGui.QColor(0, 200, 0, 160)); normal_pen.setWidthF(1.0)
        normal_brush = QtGui.QBrush(QtGui.QColor(0, 200, 0, 80))
        for items in self._overlay_intersections.values():
            for it in items:
                it.setPen(normal_pen)
                it.setBrush(normal_brush)
                it.setZValue(-50)

        # brighten any selected rows
        sel_rows = sorted(set(ix.row() for ix in self.pairTable.selectedIndexes()))
        if not sel_rows:
            return
        sel_pen   = QtGui.QPen(QtGui.QColor(0, 255, 0, 255)); sel_pen.setWidthF(2.0)
        sel_brush = QtGui.QBrush(QtGui.QColor(0, 255, 0, 150))
        for r in sel_rows:
            aitem = self.pairTable.item(r, 0)
            bitem = self.pairTable.item(r, 1)
            if not aitem or not bitem:
                continue
            akey: ROIKey = aitem.data(QtCore.Qt.UserRole)
            bkey: ROIKey = bitem.data(QtCore.Qt.UserRole)
            key = (akey, bkey)
            if key in self._overlay_intersections:
                for it in self._overlay_intersections[key]:
                    it.setPen(sel_pen)
                    it.setBrush(sel_brush)
                    it.setZValue(-45)  # slightly above normal patches

    def _save_range(self, which: str):
        vb = {'A': self.viewA, 'B': self.viewB, 'O': self.viewOverlay}[which].getViewBox()
        xr, yr = vb.viewRange()  # [[xMin,xMax],[yMin,yMax]]
        self._savedRanges[which] = (tuple(xr), tuple(yr))

    def _restore_range(self, which: str):
        rng = self._savedRanges.get(which)
        if rng:
            vb = {'A': self.viewA, 'B': self.viewB, 'O': self.viewOverlay}[which].getViewBox()
            vb.setRange(xRange=rng[0], yRange=rng[1], padding=0)
            return True
        return False

    def _srt_from_ui(self) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        try:
            s = float(self.spinScale.value())
            th = float(self.spinTheta.value()) * np.pi / 180.0
            c, d = np.cos(th), np.sin(th)
            R = np.array([[c, -d], [d,  c]], dtype=float)
            t = np.array([float(self.spinTx.value()), float(self.spinTy.value())], dtype=float)
            return (s, R, t)
        except Exception:
            return None

    def _set_ui_from_srt(self, s: float, R: np.ndarray, t: np.ndarray, block_signals: bool = False):
        # recover angle (deg) from R
        theta = float(np.arctan2(R[1, 0], R[0, 0])) * 180.0 / np.pi
        widgets = [self.spinScale, self.spinTheta, self.spinTx, self.spinTy]
        if block_signals:
            for w in widgets: w.blockSignals(True)
        try:
            self.spinScale.setValue(float(s))
            self.spinTheta.setValue(theta)
            self.spinTx.setValue(float(t[0]))
            self.spinTy.setValue(float(t[1]))
        finally:
            if block_signals:
                for w in widgets: w.blockSignals(False)


    def on_opacity_changed(self, which: str, v: int):
        txt = f"{int(v)}%"
        if which == 'A':
            if hasattr(self, "lblOpacityA"): self.lblOpacityA.setText(txt)
            self.update_image_on_view('A')
        else:
            if hasattr(self, "lblOpacityB"): self.lblOpacityB.setText(txt)
            self.update_image_on_view('B')
        # Overlay uses the same opacity settings
        self.update_overlay_view()

    def _slider_to_opacity(self, v: int) -> float:
        return max(0.0, min(1.0, v / 100.0))

    def _get_opacity(self, which: str) -> float:
        if which == 'A':
            return self._slider_to_opacity(self.sldOpacityA.value())
        else:
            return self._slider_to_opacity(self.sldOpacityB.value())


    def _on_toggle_outlineA(self, checked: bool):

        if not checked:
            self._clear_outlines('A')
            self._clear_labels('A')
            self._clear_highlight('A')          # <-- clear selection outline too
            self.update_overlay_view()          # <-- refresh overlay so it hides there as well

        self.populate_view_A()
        self.update_overlay_view()

    def _on_toggle_outlineB(self, checked: bool):
        if not checked:
            self._clear_outlines('B')
            self._clear_labels('B')
            self._clear_highlight('B')          # <-- clear selection outline too
            self.update_overlay_view()          # <-- refresh overlay

        self.populate_view_B()
        self.update_overlay_view()

    def _draw_outlines(self, which: str):
        if (which == 'A' and not self.chkShowOutlineA.isChecked()) or (which == 'B' and not self.chkShowOutlineB.isChecked()):
            return
        view = self.viewA if which == 'A' else self.viewB
        items = self.outlineItemsA if which == 'A' else self.outlineItemsB
        # clear previous
        for it in items:
            try: view.removeItem(it)
            except Exception: pass
        items.clear()

        hulls = self.hullsA if which == 'A' else self.hullsB
        pen = pg.mkPen((0,120,255), width=1.2) if which == 'A' else pg.mkPen((255,140,0), width=1.2)

        for rid, path in hulls.items():
            poly = path.toFillPolygon()
            if poly.isEmpty():
                continue
            pts = np.array([[p.x(), p.y()] for p in poly], dtype=float)
            if pts.shape[0] >= 2:
                if not np.allclose(pts[0], pts[-1]):
                    pts = np.vstack([pts, pts[0]])
                item = pg.PlotDataItem(pts[:, 0], pts[:, 1], pen=pen)
                view.addItem(item)
                items.append(item)

        if which == 'A':
            self.outlineItemsA = items
        else:
            self.outlineItemsB = items


    def on_intensity_changed(self, which: str, v: int):
        """Slider callback: update label, redraw images, refresh overlay."""
        # map slider 5..500 → factor 0.05..5.00
        fct = max(0.05, min(5.0, v / 100.0))

        if which == 'A':
            if hasattr(self, "lblIntensityA"):
                self.lblIntensityA.setText(f"{fct:.2f}×")
            self.update_image_on_view('A')
        else:
            if hasattr(self, "lblIntensityB"):
                self.lblIntensityB.setText(f"{fct:.2f}×")
            self.update_image_on_view('B')

        # overlay uses the same per-z levels; refresh it too
        self.update_overlay_view()


    def _slider_to_factor(self, v: int) -> float:
        # 5..500 → 0.05..5.00
        return max(0.05, min(5.0, v / 100.0))

    def _get_intensity_factor(self, which: str) -> float:
        if which == 'A':
            return self._slider_to_factor(self.sldIntensityA.value())
        else:
            return self._slider_to_factor(self.sldIntensityB.value())

    def _ensure_base_levels_for_z(self, which: str, z: float, slc: np.ndarray):
        """Compute/caches robust base levels (lo, hi) per (dataset, z).
        Falls back to wider percentiles/minmax if the window is too narrow.
        """
        if slc is None:
            return
        cache = self._baseLevelsA_by_z if which == 'A' else self._baseLevelsB_by_z
        if z in cache:
            return

        vals = slc.astype(np.float32, copy=False)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            cache[z] = (0.0, 1.0)
            return

        # Start with robust percentiles
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        width = hi - lo

        # Overall dynamic range (robust)
        dyn = float(np.percentile(vals, 99.9) - np.percentile(vals, 0.1))

        # If too narrow, widen to 0.1–99.9
        if width < 1e-6 or width < 0.01 * max(dyn, 1.0):
            lo = float(np.percentile(vals, 0.1))
            hi = float(np.percentile(vals, 99.9))
            width = hi - lo

        # If still too narrow (e.g., almost-constant images), center on median with min width
        if width < 1e-6 or width < 0.005 * max(dyn, 1.0):
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            if vmax <= vmin:
                vmax = vmin + 1.0
            med = float(np.median(vals))
            base_width = max(0.05 * (vmax - vmin), 1.0)  # at least 5% of full range or 1 DN
            lo = med - 0.5 * base_width
            hi = med + 0.5 * base_width

        cache[z] = (float(lo), float(hi))

    def _reset_pairs_on_dataset_load(self, which: str):
        """
        Called when either dataset A or B is (re)loaded.
        Clears cross-dataset pair mappings and UI, but does NOT touch
        labels stored inside either dataset (so they can seed the new file).
        """
        # Clear pair mappings and caches
        self.A2B.clear()
        self.B2A.clear()
        self._two_pair_cache.clear()
        self._best_combo_cache.clear()

        # Clear selection & highlights
        self.selA = None
        self.selB = None
        self._clear_highlight('A')
        self._clear_highlight('B')

        # Clear intersection overlays
        for items in self._overlay_intersections.values():
            for it in items:
                try:
                    self.viewOverlay.removeItem(it)
                except Exception:
                    pass
        self._overlay_intersections = {}

        # Clear table UI
        self.pairTable.clearContents()
        self.pairTable.setRowCount(0)

        # Refresh views
        self.update_overlay_view()
        self.refresh_controls()

        # Status message helps debugging
        self.status.setText(
            f"Cleared old A↔B pairs after loading dataset {which}. "
            "Existing labels on the other dataset are preserved."
        )

    def _make_tint_lut(self, rgb, n=256, alpha_mode='constant', max_alpha=220):
        """Return (n,4) uint8 LUT. alpha_mode: 'constant' or 'ramp' (intensity-weighted)."""
        rgb = np.array(rgb, dtype=float)[None, :] / 255.0          # (1,3)
        ramp = np.linspace(0.0, 1.0, n)[:, None]                   # (n,1)
        rgb_part = (ramp * rgb * 255.0).astype(np.uint8)           # (n,3)

        if alpha_mode == 'ramp':
            a = np.clip(ramp * max_alpha, 0, 255).astype(np.uint8) # brighter ⇒ more opaque
        else:
            a = np.full((n,1), int(max_alpha), dtype=np.uint8)     # fixed opacity per pixel

        return np.hstack([rgb_part, a])                            # (n,4)

    def _apply_tinted_lut(self, img_item: pg.ImageItem, which: str, alpha_mode='constant'):
        if which == 'A':
            if alpha_mode == 'ramp':
                if getattr(self, "_lutA_ramp", None) is None:
                    self._lutA_ramp = self._make_tint_lut((0,120,255), alpha_mode='ramp',  max_alpha=220)
                lut = self._lutA_ramp
            else:
                if getattr(self, "_lutA_const", None) is None:
                    self._lutA_const = self._make_tint_lut((0,120,255), alpha_mode='constant', max_alpha=220)
                lut = self._lutA_const
        else:
            if alpha_mode == 'ramp':
                if getattr(self, "_lutB_ramp", None) is None:
                    self._lutB_ramp = self._make_tint_lut((255,140,0), alpha_mode='ramp',  max_alpha=220)
                lut = self._lutB_ramp
            else:
                if getattr(self, "_lutB_const", None) is None:
                    self._lutB_const = self._make_tint_lut((255,140,0), alpha_mode='constant', max_alpha=220)
                lut = self._lutB_const

        img_item.setLookupTable(lut)




    def _apply_image_levels(self, imgItem: pg.ImageItem, which: str, slc: np.ndarray, z: Optional[float] = None):
        """
        Stable contrast slider:
        - Compute robust wide and tight windows per-z.
        - Slider (0.05..5.0) maps linearly to mix between wide (low contrast) and tight (high contrast).
        """
        if imgItem is None or slc is None:
            return
        if z is None:
            z = self.datasetA.current_z if which == 'A' else self.datasetB.current_z

        # compute once per (dataset, z)
        self._ensure_base_levels_for_z(which, z, slc)
        cache = self._baseLevelsA_by_z if which == 'A' else self._baseLevelsB_by_z
        base = cache.get(z)
        if base is None:
            return

        v = slc.astype(np.float32, copy=False)
        v = v[np.isfinite(v)]
        if v.size == 0:
            imgItem.setLevels((0.0, 1.0))
            return

        # wide and tight windows (robust)
        p = np.percentile
        wide_lo, wide_hi   = float(p(v, 0.2)),  float(p(v, 99.8))
        tight_lo, tight_hi = float(p(v, 10.0)), float(p(v, 90.0))

        # ensure order
        if tight_hi <= tight_lo:
            tight_lo, tight_hi = wide_lo, wide_hi
        if wide_hi <= wide_lo:
            wide_lo, wide_hi = float(np.min(v)), float(np.max(v))
            if wide_hi <= wide_lo:
                wide_hi = wide_lo + 1.0

        # slider factor 0.05..5.00  →  mix 0..1 linearly
        f   = self._get_intensity_factor(which)              # 0.05..5
        mix = (f - 0.05) / (5.0 - 0.05)                      # 0..1
        mix = max(0.0, min(1.0, mix))

        lo = (1.0 - mix) * wide_lo + mix * tight_lo
        hi = (1.0 - mix) * wide_hi + mix * tight_hi

        # enforce a tiny minimal width to avoid binary look
        full = float(np.max(v) - np.min(v))
        minw = max(1e-6, 0.001 * full)
        if (hi - lo) < minw:
            c = 0.5 * (hi + lo)
            lo, hi = c - 0.5 * minw, c + 0.5 * minw

        imgItem.setLevels((float(lo), float(hi)))


    def toggle_overlay_visible(self, checked: bool):
        self.viewOverlay.setVisible(checked)

    def change_z_text(self, which: str):
        ds = self.datasetA if which == 'A' else self.datasetB
        edit = self.editZA if which == 'A' else self.editZB
        if ds.df is None:
            return
        txt = edit.text().strip()
        if not txt:
            return
        try:
            zval = float(txt)
        except ValueError:
            return
        # snap to nearest available z from CSV
        if ds.z_values:
            zarr = np.asarray(ds.z_values, dtype=float)
            zsel = float(zarr[np.argmin(np.abs(zarr - zval))])
        else:
            zsel = zval
        ds.set_current_z(zsel)
        edit.setText(str(zsel))
        if which == 'A': self.populate_view_A()
        else:            self.populate_view_B()
        self.update_overlay_view()

    def on_user_orientation_changed(self, which: str):
        if which == 'A':
            self.populate_view_A()
        else:
            self.populate_view_B()
        self.update_overlay_view()


    def _rect_for_dataset(self, which: str, slc_shape: Tuple[int,int]) -> QtCore.QRectF:
        """Use existing per-z rect if set; otherwise default to pixel-center rect."""
        rect_map = self.imgRectA_by_z if which == 'A' else self.imgRectB_by_z
        ds = self.datasetA if which == 'A' else self.datasetB
        rect = rect_map.get(ds.current_z)
        if rect is not None:
            return rect
        H, W = slc_shape
        return QtCore.QRectF(-0.5, -0.5, W, H)

    def _user_affine(self, which: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (A, b) for XY' = A @ XY + b, applying flipH/flipV then rotation about the image-rect center."""
        angle_deg = (self.spinRotA.value() if which == 'A' else self.spinRotB.value())
        flipH = (self.chkFlipHA.isChecked() if which == 'A' else self.chkFlipHB.isChecked())
        flipV = (self.chkFlipVA.isChecked() if which == 'A' else self.chkFlipVB.isChecked())

        # Need current slice shape to define the default rect / center
        ds = self.datasetA if which == 'A' else self.datasetB
        slc = ds.get_tif_slice_for_z(ds.current_z) if ds.tif_stack is not None else None
        if slc is None:
            # fallback from points extent
            pts = ds.get_boundary_xy_for_z(ds.current_z)
            if pts.size == 0:
                cx = cy = 0.0
            else:
                cx = 0.5 * (pts[:,0].min() + pts[:,0].max())
                cy = 0.5 * (pts[:,1].min() + pts[:,1].max())
        else:
            rect = self._rect_for_dataset(which, (slc.shape[0], slc.shape[1]))
            cx = rect.x() + rect.width() * 0.5
            cy = rect.y() + rect.height() * 0.5

        # 2x2
        ang = np.deg2rad(angle_deg)
        c, s = np.cos(ang), np.sin(ang)
        Arot = np.array([[c, -s], [s, c]], dtype=float)
        Sflip = np.diag([ -1.0 if flipH else 1.0, -1.0 if flipV else 1.0 ])
        A = Arot @ Sflip  # flips first, then rotate

        # translate so pivot = (cx, cy)
        p = np.array([cx, cy], dtype=float)
        b = p - A @ p
        return A, b

    def _apply_user_affine_to_points(self, which: str, XY: np.ndarray) -> np.ndarray:
        if XY is None or XY.size == 0:
            return XY
        A, b = self._user_affine(which)
        return (A @ XY.T).T + b


    def _pre_orient_tif_stack(self, arr: np.ndarray) -> np.ndarray:
        """
        One-time raster-only orientation: 90° CW + *horizontal* flip on spatial axes.
        This matches what you saw when using matplotlib imshow(origin='upper').
        """
        if arr is None:
            return arr

        if arr.ndim == 2:
            out = np.rot90(arr, k=-1)   # 90° CW
            out = out[:, ::-1]          # flip columns (left-right) 
            return out

        if arr.ndim == 3:
            # Assume (Z,H,W). If you ever need to be robust, detect spatial axes first.
            # Here we keep it simple since you said Z format is fine.
            out = np.rot90(arr, k=-1, axes=(1, 2))  # rotate on (H,W)
            out = out[:, :, ::-1]                   # flip columns (W axis) 
            return out

        return arr

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Escape:
            self.selA = None; self.selB = None
            self._clear_highlight('A'); self._clear_highlight('B')
            ev.accept(); return
        super().keyPressEvent(ev)

    def set_rot(self, which: str):
        idx = self.comboRotA.currentIndex() if which == 'A' else self.comboRotB.currentIndex()
        k = [0, 1, 2, 3][idx]
        if which == 'A': self.rotA_k = k
        else:            self.rotB_k = k

    def apply_xy_options(self, XY: np.ndarray) -> np.ndarray:
        return XY

    def _qtransform_from_affine(self, A: np.ndarray, t: np.ndarray) -> QtGui.QTransform:
        """
        Build QTransform for XY' = A @ XY + t (no perspective).
        Qt maps points as:
            x' = m11*x + m21*y + m31
            y' = m12*x + m22*y + m32
            w  = m13*x + m23*y + m33
        For affine: m13=m23=0, m33=1.
        """
        return QtGui.QTransform(
            float(A[0, 0]), float(A[1, 0]), 0.0,   # m11, m12, m13
            float(A[0, 1]), float(A[1, 1]), 0.0,   # m21, m22, m23
            float(t[0]),    float(t[1]),    1.0    # m31, m32, m33
        )

    def qtransform_from_similarity(self, s: float, R: np.ndarray, t: np.ndarray) -> QtGui.QTransform:
        return self._qtransform_from_affine(s * R, t)

    def _user_qtransform(self, which: str) -> QtGui.QTransform:
        A, b = self._user_affine(which)
        return self._qtransform_from_affine(A, b)

    def qtransform_from_similarity(self, s: float, R: np.ndarray, t: np.ndarray) -> QtGui.QTransform:
        return self._qtransform_from_affine(s * R, t)

    def _user_qtransform(self, which: str) -> QtGui.QTransform:
        A, b = self._user_affine(which)
        return self._qtransform_from_affine(A, b)



    # ------------------------ File loading + z ------------------------
    def load_csv(self, dataset: Dataset, label_widget: QtWidgets.QLabel, which: str):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load {dataset.name} CSV", os.getcwd(), "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            dataset.load_csv(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return

        # Reset pairs
        self._reset_pairs_on_dataset_load(which)

        # Show filename beside the Load button
        label_widget.setText(os.path.basename(path))

        # Pick the first z-level (if any) and set current
        if dataset.z_values:
            z0 = dataset.z_values[0]
            dataset.set_current_z(z0)
            if which == 'A':
                self.editZA.setText(str(z0))
            else:
                self.editZB.setText(str(z0))

        self.status.setText(
            f"Loaded {dataset.name}: {len(dataset.df)} points, {len(dataset.z_values)} z-levels"
        )

        # Draw views
        if which == 'A':
            self.populate_view_A()
        else:
            self.populate_view_B()

        self.refresh_controls()
        self.update_overlay_view()
        if self.datasetA.df is not None and self.datasetB.df is not None:
            self._autopair_by_labels()


    def load_tif(self, dataset: Dataset, which: str):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load {dataset.name} TIF", os.getcwd(), "TIFF Files (*.tif *.tiff)"
        )
        if not path:
            return

        try:
            dataset.load_tif(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "TIF load error", str(e))
            return

        # One-time pre-orientation (your fixed version that rotates CW then flips columns)
        dataset.tif_stack = self._pre_orient_tif_stack(dataset.tif_stack)

        # If z not set yet, initialize it (for 2D: 0; for 3D: 0th slice)
        if dataset.current_z is None:
            z_init = 0.0
            dataset.set_current_z(z_init)
            if which == 'A':
                self.editZA.setText(str(dataset.current_z))
            else:
                self.editZB.setText(str(dataset.current_z))

        # Update filename label beside the TIF button
        if which == 'A':
            self.lblTifA.setText(os.path.basename(path))
            self._baseLevelsA_by_z.clear()
        else:
            self.lblTifB.setText(os.path.basename(path))
            self._baseLevelsB_by_z.clear()

        # Optionally scale/position image to match points on this z (no harm if no points)
        try:
            self.auto_fit_image_to_points(which)
        except Exception:
            pass  # safe fallback; rect will default to pixel coords

        # Turn on "Show image" and draw immediately
        if which == 'A':
            self.chkShowImgA.blockSignals(True)
            self.chkShowImgA.setChecked(True)
            self.chkShowImgA.blockSignals(False)
            self.populate_view_A()
        else:
            self.chkShowImgB.blockSignals(True)
            self.chkShowImgB.setChecked(True)
            self.chkShowImgB.blockSignals(False)
            self.populate_view_B()

        self.update_overlay_view()
        self.status.setText(f"Loaded TIF for {dataset.name}: {os.path.basename(path)}")

    def change_z(self, which: str):
        dataset = self.datasetA if which == 'A' else self.datasetB
        combo = self.comboZA if which == 'A' else self.comboZB
        if dataset.df is None or combo.count() == 0:
            return
        z = norm_z(float(combo.currentText()))
        dataset.set_current_z(z)
        if which == 'A': self.populate_view_A()
        else:            self.populate_view_B()
        self.refresh_controls(); self.update_overlay_view()

    # ------------------------ Controls enable ------------------------
    def refresh_controls(self):
        loadedA = self.datasetA.df is not None
        loadedB = self.datasetB.df is not None
        both_loaded = loadedA and loadedB

        # Enable Hungarian only when both datasets are loaded and have centroids on the current z
        hasA = both_loaded and (self.datasetA.cur_centroid_array is not None) and (self.datasetA.cur_centroid_array.size > 0)
        hasB = both_loaded and (self.datasetB.cur_centroid_array is not None) and (self.datasetB.cur_centroid_array.size > 0)
        if hasattr(self, "btnHungarian"):
            self.btnHungarian.setEnabled(hasA and hasB)

        # Enable export when at least one pair exists
        if hasattr(self, "btnExport"):
            self.btnExport.setEnabled(both_loaded and len(self.A2B) > 0)


    # ------------------------ Drawing ------------------------
    def populate_view_A(self):
        self._save_range('A') 
        self.viewA.clear(); self._clear_labels('A')
        if self.datasetA.df is None:
            return
        self.update_image_on_view('A')
        z = self.datasetA.current_z
        pts = self.apply_xy_options(self.datasetA.get_boundary_xy_for_z(z))
        pts = self._apply_user_affine_to_points('A', pts)
        if pts.size and self.chkShowOutlineA.isChecked():
            self.viewA.addItem(
                pg.PlotDataItem(
                    pts[:, 0], pts[:, 1],
                    pen=None,
                    symbol='o',
                    symbolSize=3,
                    symbolPen=pg.mkPen(0,120,255, 220),
                    symbolBrush=pg.mkBrush(0,120,255, 140),
                )
            )
        self.centroidsA = self._draw_centroids(self.viewA, self.datasetA, is_A=True)
        self._draw_labels(self.viewA, self.datasetA, 'A')
        self.rebuild_hulls_for(self.datasetA, 'A')
        # if getattr(self, 'chkShowOutlineA', None) and self.chkShowOutlineA.isChecked():
        #     self._draw_outlines('A')
        self.update_overlay_on_A()

        if not self._restore_range('A'):   # keep user zoom if we have it
            self.viewA.autoRange()         # otherwise autorange only on first draw

    def populate_view_B(self):
        self._save_range('B')
        self.viewB.clear(); self._clear_labels('B')
        if self.datasetB.df is None:
            return
        self.update_image_on_view('B')
        z = self.datasetB.current_z
        pts = self.apply_xy_options(self.datasetB.get_boundary_xy_for_z(z))
        pts = self._apply_user_affine_to_points('B', pts)
        if pts.size and self.chkShowOutlineB.isChecked():
            self.viewB.addItem(
                pg.PlotDataItem(
                    pts[:, 0], pts[:, 1],
                    pen=None,
                    symbol='o',
                    symbolSize=3,
                    symbolPen=pg.mkPen(255,140,0, 220),
                    symbolBrush=pg.mkBrush(255,140,0, 140),
                )
            )
        self.centroidsB = self._draw_centroids(self.viewB, self.datasetB, is_A=False)
        self._draw_labels(self.viewB, self.datasetB, 'B')
        self.rebuild_hulls_for(self.datasetB, 'B')
        # if getattr(self, 'chkShowOutlineB', None) and self.chkShowOutlineB.isChecked():
        #     self._draw_outlines('B')

        if not self._restore_range('B'):
                self.viewB.autoRange()

    def oriented_slice(self, dataset, which: str) -> Optional[np.ndarray]:
        # Return the current slice as-is (already pre-oriented at load time).
        return dataset.get_tif_slice_for_z(dataset.current_z)

    def _clear_outlines(self, which: str):
        view = self.viewA if which == 'A' else self.viewB
        plot = view.getPlotItem()

        items = self.outlineItemsA if which == 'A' else self.outlineItemsB

        for it in items:
            # 1) Proper owner removal (PlotItem)
            try:
                plot.removeItem(it)
            except Exception:
                pass

            try:
                view.removeItem(it)
            except Exception:
                pass

            # 2) Hard fallback: detach and remove from scene
            try:
                gi = it  # PlotDataItem is already a GraphicsObject
                if gi.scene() is not None:
                    gi.setParentItem(None)
                    gi.hide()
                    gi.scene().removeItem(gi)
            except Exception:
                pass

            # 3) Nuke its data as a last resort (instant visual clear)
            try:
                if hasattr(it, "setData"):
                    it.setData(x=[], y=[])
            except Exception:
                pass

        # Reset the cache list
        if which == 'A':
            self.outlineItemsA = []
        else:
            self.outlineItemsB = []

        # Force a redraw so the removal is visible immediately
        plot.update()
        view.update()
        QtWidgets.QApplication.processEvents()

    def update_image_on_view(self, which: str):
        dataset = self.datasetA if which == 'A' else self.datasetB
        view    = self.viewA    if which == 'A' else self.viewB
        chk     = self.chkShowImgA if which == 'A' else self.chkShowImgB
        imgItem = self.imgItemA if which == 'A' else self.imgItemB
        if imgItem is not None:
            try: view.removeItem(imgItem)
            except Exception: pass
            if which == 'A': self.imgItemA = None
            else:            self.imgItemB = None
        if not chk.isChecked() or dataset.tif_stack is None:
            return
        slc = self.oriented_slice(dataset, which)
        if slc is None:
            return
        new_img = pg.ImageItem(slc)
        new_img.setOpacity(self._get_opacity(which))  # <— add this


        rect_map = self.imgRectA_by_z if which == 'A' else self.imgRectB_by_z
        rect = rect_map.get(dataset.current_z)
        if rect is None:
            H, W = slc.shape[0], slc.shape[1]
            rect = QtCore.QRectF(-0.5, -0.5, W, H)
        new_img.setRect(rect)

        # User orientation (flip + any-angle rot) applied to the image
        T_user = self._user_qtransform(which)

        # Optionally compose B→A similarity in B view
        T = T_user
        if which == 'B' and self.transform is not None and self.chkApplyTransformToB.isChecked():
            s, R, t = self.transform
            T = self.qtransform_from_similarity(s, R, t) * T_user

        new_img.setTransform(T)
        cur_z = self.datasetA.current_z if which == 'A' else self.datasetB.current_z
        self._apply_image_levels(new_img, which, slc, z=cur_z)
        new_img.setZValue(-100)
        view.addItem(new_img)
        if which == 'A': self.imgItemA = new_img
        else:            self.imgItemB = new_img

    def _draw_centroids(self, view: pg.PlotWidget, dataset: Dataset, is_A: bool) -> ClickableScatter:
        colors = {}; n = max(1, len(dataset.cur_roi_ids_sorted))
        for i, rid in enumerate(dataset.cur_roi_ids_sorted):
            colors[rid] = pg.intColor(i, hues=n)
        if is_A: self.colorA = colors
        else:    self.colorB = colors
        spots = []
        for rid in dataset.cur_roi_ids_sorted:
            xy = dataset.cur_roi_centroids_xy[rid]
            xy = self.apply_xy_options(np.array(xy))
            xy = self._apply_user_affine_to_points('A' if is_A else 'B', np.array([xy]))[0]
            col = colors[rid]
            spots.append({'pos': (xy[0], xy[1]), 'size': 10, 'brush': col,
                          'pen': pg.mkPen('k', width=0.8), 'data': rid, 'symbol': 'o'})
        scatter = ClickableScatter(on_left_click=(self.on_click_A if is_A else self.on_click_B),
                                   on_right_click=(self.on_right_A if is_A else self.on_right_B))
        scatter.addPoints(spots)
        view.addItem(scatter)
        return scatter

    def _draw_labels(self, view: pg.PlotWidget, dataset: Dataset, which: str):
        # Don't draw labels if outlines are hidden
        if (which == 'A' and not self.chkShowOutlineA.isChecked()) or \
        (which == 'B' and not self.chkShowOutlineB.isChecked()):
            # ensure any old labels are cleared
            self._clear_labels(which)
            return

        lbls: List[pg.TextItem] = []
        ids = dataset.cur_roi_ids_sorted or []
        for rid in ids:
            base_xy = dataset.cur_roi_centroids_xy.get(rid)
            if base_xy is None:
                continue
            xy = np.asarray(base_xy, dtype=float)
            xy = self.apply_xy_options(xy)
            xy = self._apply_user_affine_to_points(which, xy[None, :])[0]

            label_item = pg.TextItem(text=str(rid), anchor=(0, 1))
            label_item.setPos(float(xy[0]) + 3.0, float(xy[1]) - 3.0)
            view.addItem(label_item)
            lbls.append(label_item)

        if which == 'A':
            self.labelsA = lbls
        else:
            self.labelsB = lbls



    def _clear_labels(self, which: str):
        lbls = self.labelsA if which == 'A' else self.labelsB
        for ti in lbls:
            try: (self.viewA if which=='A' else self.viewB).removeItem(ti)
            except Exception: pass
        if which == 'A': self.labelsA = []
        else:            self.labelsB = []

    # ------------------------ Hulls + background click ------------------------
    def rebuild_hulls_for(self, dataset: Dataset, which: str):
        hulls: Dict[int, QtGui.QPainterPath] = {}
        rid2pts = dataset.roi_to_points_by_z.get(dataset.current_z, {})
        for rid, pts3 in rid2pts.items():
            pts = self.apply_xy_options(pts3[:, :2])
            pts = self._apply_user_affine_to_points(which, pts)
            poly = None
            if pts.shape[0] >= 3 and HULL_AVAILABLE:
                try:
                    hull = ConvexHull(pts)
                    poly = pts[hull.vertices]
                except Exception:
                    poly = None
            if poly is None:
                if pts.shape[0] == 0:
                    continue
                mn = pts.min(axis=0); mx = pts.max(axis=0)
                poly = np.array([[mn[0], mn[1]],[mx[0], mn[1]],[mx[0], mx[1]],[mn[0], mx[1]]], float)
            path = QtGui.QPainterPath(); qp = QtGui.QPolygonF([QtCore.QPointF(p[0], p[1]) for p in poly])
            path.addPolygon(qp); path.closeSubpath()
            hulls[int(rid)] = path
        if which == 'A': self.hullsA = hulls
        else:            self.hullsB = hulls

    def find_roi_at_point(self, dataset: Dataset, which: str, x: float, y: float) -> Optional[int]:
        hulls = self.hullsA if which == 'A' else self.hullsB
        qp = QtCore.QPointF(x, y)
        for rid, path in hulls.items():
            if path.contains(qp):
                return rid
        # XY = dataset.cur_centroid_array
        # if XY is not None and XY.size:
        #     XY2 = self.apply_xy_options(XY)
        #     d = np.linalg.norm(XY2 - np.array([x, y]), axis=1)
        #     idx = int(np.argmin(d))
        #     return dataset.cur_roi_ids_sorted[idx]
        return None

    def _on_viewA_click(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or ev.isAccepted():
            return
        vb = self.viewA.getViewBox(); p = vb.mapSceneToView(ev.scenePos())
        rid = self.find_roi_at_point(self.datasetA, 'A', p.x(), p.y())
        if rid is None: return
        if self.selA == rid:
            self.selA = None; self._clear_highlight('A')
        else:
            self.selA = rid; self._highlight_roi('A', rid)
        self.try_make_pair(); self.update_overlay_view()

    def _on_viewB_click(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or ev.isAccepted():
            return
        vb = self.viewB.getViewBox(); p = vb.mapSceneToView(ev.scenePos())
        rid = self.find_roi_at_point(self.datasetB, 'B', p.x(), p.y())
        if rid is None: return
        if self.selB == rid:
            self.selB = None; self._clear_highlight('B')
        else:
            self.selB = rid; self._highlight_roi('B', rid)
        self.try_make_pair(); self.update_overlay_view()

    def _highlight_roi(self, which: str, rid: int):
        self._clear_highlight(which)
        ds = self.datasetA if which == 'A' else self.datasetB
        pts3 = ds.roi_to_points_by_z.get(ds.current_z, {}).get(int(rid))
        if pts3 is None or pts3.size == 0:
            return

        # Apply user orientation (so highlight aligns with the view)
        XY = self._apply_user_affine_to_points(which, pts3[:, :2].copy())
        if XY.shape[0] >= 3:
            XY = np.vstack([XY, XY[0]])

        # Fixed highlight colours (lighter shades of A/B)
        if which == 'A':
            pen = pg.mkPen((150, 200, 255), width=2)   # light blue
            item = pg.PlotDataItem(XY[:, 0], XY[:, 1], pen=pen)
            self.selOutlineA = item; self.viewA.addItem(item)
        else:
            pen = pg.mkPen((255, 200, 120), width=2)   # light orange
            item = pg.PlotDataItem(XY[:, 0], XY[:, 1], pen=pen)
            self.selOutlineB = item; self.viewB.addItem(item)


    def _clear_highlight(self, which: str):
        if which == 'A' and self.selOutlineA is not None:
            try: self.viewA.removeItem(self.selOutlineA)
            except Exception: pass
            self.selOutlineA = None
        if which == 'B' and self.selOutlineB is not None:
            try: self.viewB.removeItem(self.selOutlineB)
            except Exception: pass
            self.selOutlineB = None


    # ------------------------ Click handlers on centroids ------------------------
    def on_click_A(self, rid: int):
        if rid not in self.datasetA.cur_roi_centroids_xy:
            return
        if self.selA == rid:
            self.selA = None; self._clear_highlight('A')
        else:
            self.selA = rid; self._highlight_roi('A', rid)
        self.status.setText(f"Selected A (z={self.datasetA.current_z}) ROI {rid}")
        self.try_make_pair(); self.update_overlay_view()

    def on_click_B(self, rid: int):
        if rid not in self.datasetB.cur_roi_centroids_xy:
            return
        if self.selB == rid:
            self.selB = None; self._clear_highlight('B')
        else:
            self.selB = rid; self._highlight_roi('B', rid)
        self.status.setText(f"Selected B (z={self.datasetB.current_z}) ROI {rid}")
        self.try_make_pair(); self.update_overlay_view()

    def on_right_A(self, rid: int):
        akey: ROIKey = (self.datasetA.current_z, int(rid))
        if akey in self.A2B:
            bkey = self.A2B[akey]
            del self.A2B[akey]
            if bkey in self.B2A:
                del self.B2A[bkey]
            self.status.setText(f"Unpaired A {akey} ↔ B {bkey}")
            self.update_pair_table(); self.refresh_controls(); self.update_overlay_on_A(); self.update_overlay_view()

    def on_right_B(self, rid: int):
        bkey: ROIKey = (self.datasetB.current_z, int(rid))
        if bkey in self.B2A:
            akey = self.B2A[bkey]
            del self.B2A[bkey]
            if akey in self.A2B:
                del self.A2B[akey]
            self.status.setText(f"Unpaired A {akey} ↔ B {bkey}")
            self.update_pair_table(); self.refresh_controls(); self.update_overlay_on_A(); self.update_overlay_view()

    def try_make_pair(self):
        if self.selA is None or self.selB is None:
            return
        akey: ROIKey = (self.datasetA.current_z, int(self.selA))
        bkey: ROIKey = (self.datasetB.current_z, int(self.selB))
        if akey in self.A2B:
            prev_b = self.A2B[akey];
            if prev_b in self.B2A: del self.B2A[prev_b]
        if bkey in self.B2A:
            prev_a = self.B2A[bkey];
            if prev_a in self.A2B: del self.A2B[prev_a]
        self.A2B[akey] = bkey; self.B2A[bkey] = akey
        self._atlas_add_pair_incremental(akey, bkey) # Update current atlas
        self._assign_or_propagate_label_for_pair(akey, bkey)

        self.selA = None; self.selB = None
        self._clear_highlight('A'); self._clear_highlight('B')
        self.status.setText(f"Paired A {akey} ↔ B {bkey}")
        self.update_pair_table(); self.refresh_controls(); self.update_overlay_on_A(); self.update_overlay_view()

    def is_key_on_current_slices(self, akey: ROIKey, bkey: ROIKey) -> bool:
        return (np.isclose(akey[0], self.datasetA.current_z) and np.isclose(bkey[0], self.datasetB.current_z))

    def update_pair_table(self):
        self.pairTable.setRowCount(0)
        for i, (akey, bkey) in enumerate(sorted(self.A2B.items())):
            self.pairTable.insertRow(i)
            itemA = QtWidgets.QTableWidgetItem(f"({akey[0]}, {akey[1]})")
            itemB = QtWidgets.QTableWidgetItem(f"({bkey[0]}, {bkey[1]})")
            itemA.setData(QtCore.Qt.UserRole, akey); itemB.setData(QtCore.Qt.UserRole, bkey)
            self.pairTable.setItem(i, 0, itemA); self.pairTable.setItem(i, 1, itemB)

            cell_id = self._resolve_pair_label(akey, bkey)
            itemC = QtWidgets.QTableWidgetItem("" if cell_id <= 0 else str(cell_id))
            itemC.setData(QtCore.Qt.UserRole, cell_id)
            self.pairTable.setItem(i, 2, itemC)


    def remove_selected_pair(self):
        rows = sorted(set([ix.row() for ix in self.pairTable.selectedIndexes()]), reverse=True)
        for r in rows:
            aitem = self.pairTable.item(r, 0); bitem = self.pairTable.item(r, 1)
            if not aitem or not bitem: continue
            akey: ROIKey = aitem.data(QtCore.Qt.UserRole); bkey: ROIKey = bitem.data(QtCore.Qt.UserRole)
            if akey in self.A2B: del self.A2B[akey]
            if bkey in self.B2A: del self.B2A[bkey]
            self.pairTable.removeRow(r)
        self.refresh_controls(); self.update_overlay_on_A(); self.update_overlay_view()

    def clear_pairs(self):
        self.A2B.clear(); self.B2A.clear()
        self.update_pair_table(); self.refresh_controls(); self.update_overlay_on_A(); self.update_overlay_view()

    # ------------------------ Transform & overlay ------------------------
    def compute_transform(self):
        valid_pairs = [(ak, bk) for ak, bk in self.A2B.items() if self.is_key_on_current_slices(ak, bk)]
        if len(valid_pairs) < 3:
            QtWidgets.QMessageBox.warning(self, "Need ≥3 pairs", "Provide at least 3 A↔B pairs on the current z-slices.")
            return
        A_pts, B_pts = [], []
        for (az, arid), (bz, brid) in valid_pairs:
            A_pts.append(self.datasetA.cur_roi_centroids_xy[arid])
            B_pts.append(self.datasetB.cur_roi_centroids_xy[brid])
        A_pts = np.array(A_pts, dtype=float); B_pts = np.array(B_pts, dtype=float)
        try:
            s, R, t = estimate_similarity_transform_2d(B_pts, A_pts, allow_reflection=False)
            self.transform = (s, R, t)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Transform error", str(e)); return
        self.status.setText(f"Estimated B→A on zA={self.datasetA.current_z}, zB={self.datasetB.current_z}: scale={self.transform[0]:.4f}")
        self.update_overlay_on_A(); self.populate_view_B(); self.update_overlay_view()

    def clear_transform(self):
        self.transform = None
        self.update_overlay_on_A(); self.populate_view_B(); self.update_overlay_view()

    def update_overlay_on_A(self): # removed
        return

    # ------------------------ Live overlay view (A+B) with IoU ------------------------
    # def dynamic_transform_from_pairs(self):
    #     pairs = [(ak, bk) for ak, bk in self.A2B.items() if self.is_key_on_current_slices(ak, bk)]
    #     n = len(pairs)
    #     if n == 0:
    #         return None

    #     # RAW centroids (no user affine)
    #     A_pts = np.array([ self.datasetA.cur_roi_centroids_xy[ak[1]] for ak, _ in pairs ], float)
    #     B_pts = np.array([ self.datasetB.cur_roi_centroids_xy[bk[1]] for _, bk in pairs ], float)

    #     if n == 1:
    #         p1, q1 = B_pts[0], A_pts[0]
    #         s, R, t = 1.0, np.eye(2), q1 - p1
    #     elif n == 2:
    #         s, R, t = two_point_similarity(B_pts[0], B_pts[1], A_pts[0], A_pts[1])
    #     else:
    #         s, R, t = estimate_similarity_transform_2d(B_pts, A_pts, allow_reflection=False)
    #     return (s, R, t)
    def dynamic_transform_from_pairs(self):
        """
        1 pair  -> translation only
        2 pairs -> similarity from those two pairs
        3+      -> choose the best 2 pairs (i,j) that maximize mean IoU across all current pairs.
                - cache per (pair_i, pair_j)
                - early stop when (t_x, t_y, s, theta) stabilizes.
        NOTE: This operates in RAW coordinates (no user-affine) so overlay should not apply user transforms.
        """
        # Collect valid current-slice pairs
        pairs = [(ak, bk) for ak, bk in self.A2B.items() if self.is_key_on_current_slices(ak, bk)]
        n = len(pairs)
        if n == 0:
            return None

        # Raw centroids (no user affine)
        A_pts = np.array([self.datasetA.cur_roi_centroids_xy[ak[1]] for ak, _ in pairs], dtype=float)
        B_pts = np.array([self.datasetB.cur_roi_centroids_xy[bk[1]] for _, bk in pairs], dtype=float)

        # ---- 1 pair: translation ----
        if n == 1:
            p1 = B_pts[0]; q1 = A_pts[0]
            s, R, t = 1.0, np.eye(2), q1 - p1
            return (s, R, t)

        # ---- 2 pairs: direct similarity ----
        if n == 2:
            s, R, t = two_point_similarity(B_pts[0], B_pts[1], A_pts[0], A_pts[1])
            return (s, R, t)

        # ---- 3+ pairs: choose the best (i, j) by IoU over ALL pairs ----

        # Precompute raw polygons (no user affine)
        polyA = []
        polyB = []
        for (ak, bk) in pairs:
            pA = self.polygon_from_roi(self.datasetA, ak[1])
            pB = self.polygon_from_roi(self.datasetB, bk[1])
            polyA.append(pA if (pA is not None and pA.shape[0] >= 3) else None)
            polyB.append(pB if (pB is not None and pB.shape[0] >= 3) else None)

        # If we have no usable polygons at all, fall back to SVD on all points
        if not any(p is not None for p in polyA) or not any(p is not None for p in polyB):
            s, R, t = estimate_similarity_transform_2d(B_pts, A_pts, allow_reflection=False)
            return (s, R, t)

        # Helper: mean IoU across all matched polygons (skip Nones)
        def score_transform(s: float, R: np.ndarray, t: np.ndarray) -> float:
            scores = []
            for k in range(n):
                pa = polyA[k]; pb = polyB[k]
                if pa is None or pb is None:
                    continue
                pb_t = apply_similarity_transform_2d(pb, s, R, t)
                val = self.iou_between(pa, pb_t)
                if val is not None:
                    scores.append(val)
            if not scores:
                return -1.0  # nothing to score on
            return float(np.mean(scores))

        # Candidate combos of two pairs
        idx_pairs = list(itertools.combinations(range(n), 2))

        # Cache key for current set of pairs (so we can reuse the best result if nothing changed)
        set_key = frozenset(pairs)
        if set_key in self._best_combo_cache:
            s0, R0, t0, _ = self._best_combo_cache[set_key]
            return (s0, R0, t0)

        best_score = -1.0
        best_sRt = None  # (s,R,t)

        # Early stopping settings (convergence on parameters)
        k_window = 6
        tol_t = 0.25      # px
        tol_s = 1e-3
        tol_theta = 1e-3  # rad ~ 0.057deg
        stable_hits = 0
        param_window = []

        def params_of(R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
            theta = math.atan2(R[1, 0], R[0, 0])  # rotation from R
            return np.array([t[0], t[1], s, theta], dtype=float)

        # For large n, evaluating all C(n,2) can be costly—iterate in a deterministic but
        # semi-spread order (middle, edges) to find good candidates early.
        # (Simple heuristic: evaluate mid indices first.)
        mid = n // 2
        idx_pairs.sort(key=lambda ij: abs((ij[0]+ij[1]) * 0.5 - mid))

        for (i, j) in idx_pairs:
            # Per-two-pair cache key (order-invariant)
            key = (pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1])
            # normalize to keep (i,j) vs (j,i) identical
            if key not in self._two_pair_cache:
                s, R, t = two_point_similarity(B_pts[i], B_pts[j], A_pts[i], A_pts[j])
                sc = score_transform(s, R, t)
                self._two_pair_cache[key] = (s, R, t, sc)
            else:
                s, R, t, sc = self._two_pair_cache[key]

            # Track best
            if sc > best_score:
                best_score = sc
                best_sRt = (s, R, t)

            # ---- Early stopping on parameter stability ----
            p = params_of(R, t, s)
            param_window.append(p)
            if len(param_window) > k_window:
                param_window.pop(0)

            if len(param_window) == k_window:
                span = np.ptp(np.stack(param_window, axis=0), axis=0)
                # if translation, scale and angle are all stable for a couple of consecutive windows
                if (max(span[0], span[1]) < tol_t) and (span[2] < tol_s) and (span[3] < tol_theta):
                    stable_hits += 1
                    if stable_hits >= 2:  # "majority" convergence proxy
                        break
                else:
                    stable_hits = 0

        # Fallback if somehow nothing scored
        if best_sRt is None:
            s, R, t = estimate_similarity_transform_2d(B_pts, A_pts, allow_reflection=False)
            return (s, R, t)

        # Remember the best for this exact set of pairs
        s, R, t = best_sRt
        self._best_combo_cache[set_key] = (s, R, t, best_score)
        return (s, R, t)



    def polygon_from_roi(self, dataset: Dataset, rid: int) -> Optional[np.ndarray]:
        pts3 = dataset.roi_to_points_by_z.get(dataset.current_z, {}).get(int(rid))
        if pts3 is None or pts3.size == 0:
            return None
        pts = self.apply_xy_options(pts3[:, :2])
        if pts.shape[0] >= 3 and HULL_AVAILABLE:
            try:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
            except Exception:
                poly = pts
        else:
            poly = pts
        return poly

    def iou_between(self, polyA: np.ndarray, polyB: np.ndarray) -> Optional[float]:
        if polyA is None or polyB is None or polyA.shape[0] < 3 or polyB.shape[0] < 3:
            return None
        if SHAPELY:
            try:
                pA = Polygon(polyA)
                pB = Polygon(polyB)
                inter = pA.intersection(pB).area
                uni = pA.union(pB).area
                if uni <= 0: return 0.0
                return float(inter / uni)
            except Exception:
                return None
        # Fallback: coarse raster IoU
        all_pts = np.vstack([polyA, polyB])
        mn = np.floor(all_pts.min(axis=0)).astype(int); mx = np.ceil(all_pts.max(axis=0)).astype(int)
        w = max(1, int(mx[0]-mn[0])); h = max(1, int(mx[1]-mn[1]))
        if w*h > 2_000_000:  # limit
            return None
        grid_x, grid_y = np.meshgrid(np.arange(mn[0], mx[0]), np.arange(mn[1], mx[1]))
        pts = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        from matplotlib.path import Path  # lightweight
        maskA = Path(polyA).contains_points(pts).reshape(h, w)
        maskB = Path(polyB).contains_points(pts).reshape(h, w)
        inter = np.logical_and(maskA, maskB).sum()
        uni = np.logical_or(maskA, maskB).sum()
        if uni == 0: return 0.0
        return float(inter/uni)

    def update_overlay_view(self):
        self._save_range('O')
        self.viewOverlay.clear()

        for items in self._overlay_intersections.values():
            for it in items:
                try: self.viewOverlay.removeItem(it)
                except Exception: pass
        self._overlay_intersections = {}

        if self.datasetA.df is None or self.datasetB.df is None:
            return

        # Choose transform: manual override (UI) or auto (dynamic)
        if self.chkManualTRS.isChecked():
            sRt = self._srt_from_ui()
        else:
            sRt = self.dynamic_transform_from_pairs()
            # keep UI boxes synced to auto values
            if sRt is not None:
                s, R, t = sRt
                self._last_auto_sRt = (s, R.copy(), t.copy())
                # Only update fields if override is OFF (so the user sees the auto values)
                self._set_ui_from_srt(s, R, t, block_signals=True)

        # ---- draw images (overlay should be in RAW frame; no user-affine here) ----
        if self.chkCompositeAdd.isChecked():
            # Single summed-RGB composite
            self._draw_additive_composite()
        else:
            # A image (identity transform)
            if self.chkShowImgA.isChecked() and self.datasetA.tif_stack is not None:
                slcA = self.oriented_slice(self.datasetA, 'A')
                if slcA is not None:
                    imgA = pg.ImageItem(slcA); imgA.setZValue(-200)
                    imgA.setOpacity(self._get_opacity('A')) 
                    self._apply_image_levels(imgA, 'A', slcA, z=self.datasetA.current_z)
                    # Apply blend
                    self._apply_tinted_lut(imgA, 'A',
                        alpha_mode='constant')
                    rectA = self.imgRectA_by_z.get(self.datasetA.current_z)
                    if rectA is None:
                        Ha, Wa = slcA.shape[0], slcA.shape[1]
                        rectA = QtCore.QRectF(-0.5, -0.5, Wa, Ha)
                    imgA.setRect(rectA)
                    imgA.setTransform(QtGui.QTransform())  # RAW frame
                    self.viewOverlay.addItem(imgA)

            # B image (apply only the chosen similarity)
            if self.chkShowImgB.isChecked() and self.datasetB.tif_stack is not None:
                slcB = self.oriented_slice(self.datasetB, 'B')
                if slcB is not None:
                    imgB = pg.ImageItem(slcB); imgB.setZValue(-190)
                    imgB.setOpacity(self._get_opacity('B'))  # <— use slider value
                    self._apply_image_levels(imgB, 'B', slcB, z=self.datasetB.current_z)
                    self._apply_tinted_lut(imgB, 'B',
                        alpha_mode='constant')
                    rectB = self.imgRectB_by_z.get(self.datasetB.current_z)
                    if rectB is None:
                        Hb, Wb = slcB.shape[0], slcB.shape[1]
                        rectB = QtCore.QRectF(-0.5, -0.5, Wb, Hb)
                    imgB.setRect(rectB)
                    T_total_B = QtGui.QTransform()
                    if sRt is not None:
                        s, R, t = sRt
                        T_total_B = self.qtransform_from_similarity(s, R, t)
                    imgB.setTransform(T_total_B)
                    self.viewOverlay.addItem(imgB)
                
        # Draw polygons & centroids
        # A in blue; B in orange (transformed by dynamic transform)
        dyn = sRt
        sRt = None if dyn is None else dyn
        penA = pg.mkPen((0, 120, 255), width=2)
        penB = pg.mkPen((255, 140, 0), width=2)

        # A polygons (RAW)
        rid2polyA = {}
        for rid in self.datasetA.cur_roi_ids_sorted:
            poly = self.polygon_from_roi(self.datasetA, rid)   # RAW
            rid2polyA[rid] = poly
            if poly is not None and poly.shape[0] >= 3 and self.chkShowOutlineA.isChecked():
                poly2 = np.vstack([poly, poly[0]])
                self.viewOverlay.addItem(pg.PlotDataItem(poly2[:,0], poly2[:,1], pen=penA))

        # B polygons (RAW → similarity only)
        for rid in self.datasetB.cur_roi_ids_sorted:
            poly = self.polygon_from_roi(self.datasetB, rid)   # RAW
            if poly is None or poly.shape[0] < 3: 
                continue
            if sRt is not None:
                s, R, t = sRt
                poly = apply_similarity_transform_2d(poly, s, R, t)
            poly2 = np.vstack([poly, poly[0]])
            if self.chkShowOutlineB.isChecked():
                self.viewOverlay.addItem(pg.PlotDataItem(poly2[:,0], poly2[:,1], pen=penB))

        show_overlap = self.chkShowOutlineA.isChecked() or self.chkShowOutlineB.isChecked()
        if SHAPELY and show_overlap:
            # colors
            # base_pen   = QtGui.QPen(QtGui.QColor(0, 200, 0, 160)); base_pen.setWidthF(1.0)
            # base_brush = QtGui.QBrush(QtGui.QColor(0, 200, 0, 80))
            base_pen   = QtGui.QPen(QtGui.QColor(255, 220, 0, 220)); base_pen.setWidthF(1.2)
            base_brush = QtGui.QBrush(QtGui.QColor(255, 220, 0, 110))

            # We already have sRt and rid2polyA (RAW). For each current-slice pair:
            pairs_cur = [(ak, bk) for ak, bk in self.A2B.items() if self.is_key_on_current_slices(ak, bk)]
            for (az, arid), (bz, brid) in pairs_cur:
                polyA = rid2polyA.get(arid)                 # RAW A polygon
                polyB = self.polygon_from_roi(self.datasetB, brid)  # RAW B polygon
                if polyA is None or polyB is None or polyA.shape[0] < 3 or polyB.shape[0] < 3:
                    continue

                # Transform B into A frame (RAW similarity only)
                if sRt is not None:
                    s, R, t = sRt
                    polyB = apply_similarity_transform_2d(polyB, s, R, t)

                # Shapely intersection (may be MultiPolygon)
                pA = Polygon(polyA)
                pB = Polygon(polyB)
                inter = pA.intersection(pB)
                if inter.is_empty or inter.area <= 0:
                    continue

                key = ((az, int(arid)), (bz, int(brid)))
                self._overlay_intersections[key] = []

                geoms = [inter] if inter.geom_type == "Polygon" else list(inter.geoms)
                for g in geoms:
                    try:
                        coords = np.asarray(g.exterior.coords, dtype=float)
                        if coords.shape[0] < 3:
                            continue
                        # Build a filled path item in data coords
                        path = QtGui.QPainterPath()
                        path.moveTo(coords[0,0], coords[0,1])
                        for k in range(1, coords.shape[0]):
                            path.lineTo(coords[k,0], coords[k,1])
                        path.closeSubpath()

                        gi = ClickablePathItem(path, on_click=self._on_intersection_clicked, key=key)
                        gi.setPen(base_pen)
                        gi.setBrush(base_brush)
                        gi.setZValue(-50)  # above images, below outlines
                        self.viewOverlay.addItem(gi)
                        self._overlay_intersections[key].append(gi)
                    except Exception:
                        pass  # robust to odd geometries

        # IoU labels for matched pairs (current slices)
        pairs_sorted = sorted(self.A2B.items())

        show_iou = self.chkShowOutlineA.isChecked() and self.chkShowOutlineB.isChecked()
        if show_iou:
            pairs_cur = [(ak, bk) for ak, bk in self.A2B.items() if self.is_key_on_current_slices(ak, bk)]
            for (az, arid), (bz, brid) in pairs_cur:
                polyA = rid2polyA.get(arid)  # RAW polygon in A
                polyB = self.polygon_from_roi(self.datasetB, brid)  # RAW polygon in B
                if polyA is None or polyB is None or polyA.shape[0] < 3 or polyB.shape[0] < 3:
                    continue

                # Transform B → A (similarity only)
                if sRt is not None:
                    s, R, t = sRt
                    polyB = apply_similarity_transform_2d(polyB, s, R, t)

                # Compute intersection center (fallback to A centroid if empty)
                cx, cy = self.datasetA.cur_roi_centroids_xy[arid]
                iou_val = self.iou_between(polyA, polyB) or 0.0

                if SHAPELY:
                    try:
                        pA = Polygon(polyA)
                        pB = Polygon(polyB)
                        inter = pA.intersection(pB)
                        if (not inter.is_empty) and inter.area > 0:
                            pt = inter.representative_point()  # guaranteed inside
                            cx, cy = float(pt.x), float(pt.y)
                            # recompute IoU using shapely areas for consistency
                            denom = pA.union(pB).area
                            if denom > 0:
                                iou_val = float(inter.area / denom)
                    except Exception:
                        pass

                cell_id = self._resolve_pair_label((az, arid), (bz, brid))
                label = f"IoU {iou_val:.2f}" + (f"\nID {cell_id}" if cell_id > 0 else "")
                ti = pg.TextItem(text=label, anchor=(0.5, 0.5))  # center of block at (cx, cy)
                ti.setColor('k')
                ti.setZValue(-40)  # above the green patch
                ti.setPos(cx, cy)
                self.viewOverlay.addItem(ti)

        self._on_pair_selection_changed()

        if not self._restore_range('O'):
            self.viewOverlay.autoRange()

    # ------------------------ Hungarian auto-match ------------------------
    def run_auto_iou(self):
        if self.datasetA.df is None or self.datasetB.df is None:
            return

        # Unpaired candidates on current slices only
        zA = self.datasetA.current_z
        zB = self.datasetB.current_z
        A_unpaired = [rid for rid in self.datasetA.cur_roi_ids_sorted
                    if (zA, rid) not in self.A2B]
        B_unpaired = [rid for rid in self.datasetB.cur_roi_ids_sorted
                    if (zB, rid) not in self.B2A]

        if not A_unpaired or not B_unpaired:
            QtWidgets.QMessageBox.information(self, "No unpaired ROIs", "Nothing to match on current slices.")
            return

        # Choose the same transform used in the overlay:
        # manual override if enabled, else dynamic from current pairs (RAW coords)
        if self.chkManualTRS.isChecked():
            sRt = self._srt_from_ui()
        else:
            sRt = self.dynamic_transform_from_pairs()

        # If no transform, IoU is measured in RAW frames (likely small).
        if sRt is None:
            QtWidgets.QMessageBox.information(
                self, "No transform in use",
                "No manual or auto transform is active. IoU will be computed in raw coordinates and may be near zero. "
                "Seed with 1–2 pairs or enter manual T/R/S, then try again."
            )

        # Precompute polygons (RAW). Skip unusable ones.
        polysA = {rid: self.polygon_from_roi(self.datasetA, rid) for rid in A_unpaired}
        polysB = {rid: self.polygon_from_roi(self.datasetB, rid) for rid in B_unpaired}

        # Build all IoUs above (or near) threshold
        tau = float(self.spinMaxDist.value())
        tau = max(0.0, min(1.0, tau))

        matches = []  # (iou, ridA, ridB)
        if sRt is not None:
            s, R, t = sRt
        for ridA in A_unpaired:
            pA = polysA.get(ridA)
            if pA is None or pA.shape[0] < 3:
                continue
            for ridB in B_unpaired:
                pB = polysB.get(ridB)
                if pB is None or pB.shape[0] < 3:
                    continue
                pBt = apply_similarity_transform_2d(pB, s, R, t) if sRt is not None else pB
                iou = self.iou_between(pA, pBt)
                if iou is None:
                    continue
                if iou >= tau:
                    matches.append((float(iou), ridA, ridB))

        if not matches:
            self.status.setText(f"IoU auto-match: no pairs found ≥ {tau:.2f}")
            return

        # Greedy 1:1: take global best IoU, then remove that A and B from consideration, repeat
        matches.sort(key=lambda x: x[0], reverse=True)
        usedA, usedB = set(), set()
        added = 0
        for iou, ridA, ridB in matches:
            if ridA in usedA or ridB in usedB:
                continue
            akey = (zA, int(ridA))
            bkey = (zB, int(ridB))
            # (They’re unpaired by construction, but keep 1:1 guardrails anyway)
            if akey in self.A2B or bkey in self.B2A:
                continue
            self.A2B[akey] = bkey
            self.B2A[bkey] = akey
            # atlas sync
            self._atlas_add_pair_incremental(akey, bkey)
            self._assign_or_propagate_label_for_pair(akey, bkey) 
            usedA.add(ridA); usedB.add(ridB)
            added += 1

        self.status.setText(
            f"IoU auto-match added {added} pair(s) on zA={zA}, zB={zB} (≥ {tau:.2f})"
        )
        self.update_pair_table()
        self.refresh_controls()
        self.update_overlay_on_A()
        self.populate_view_B()
        self.update_overlay_view()


    # ------------------------ Export ------------------------
    def export_labeled_csvs(self):
        if self.datasetA.df is None or self.datasetB.df is None:
            return
        if len(self.A2B) == 0:
            QtWidgets.QMessageBox.warning(self, "No pairs", "Create at least one pair before exporting.")
            return

        pairs_sorted = sorted(self.A2B.items())

        # Ensure 'label' column exists in both dataframes
        for ds in (self.datasetA, self.datasetB):
            if ds.df is not None and 'label' not in ds.df.columns:
                ds.df['label'] = 0

        # Determine labels pair-by-pair:
        next_new = 1 + max(
            int(self.datasetA.df['label'].max()) if self.datasetA.df is not None else 0,
            int(self.datasetB.df['label'].max()) if self.datasetB.df is not None else 0,
        )

        conflict_cnt = 0
        for (akey, bkey) in pairs_sorted:
            az, arid = akey; bz, brid = bkey
            labA = self.datasetA.get_label(az, arid)
            labB = self.datasetB.get_label(bz, brid)

            if labA > 0 and labB > 0:
                # if both present but differ, prefer A deterministically, update B (rare)
                label = labA
                if labA != labB:
                    conflict_cnt += 1
                    self.datasetB.set_label(bz, brid, label)
            elif labA > 0:
                label = labA
                self.datasetB.set_label(bz, brid, label)
            elif labB > 0:
                label = labB
                self.datasetA.set_label(az, arid, label)
            else:
                # both zero: mint a new label
                label = next_new
                next_new += 1
                self.datasetA.set_label(az, arid, label)
                self.datasetB.set_label(bz, brid, label)

        # Now write both CSVs (their df already updated in-place)
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output directory", os.getcwd())
        if not out_dir:
            return
        baseA = os.path.splitext(self.datasetA.name)[0]; baseB = os.path.splitext(self.datasetB.name)[0]
        outA = os.path.join(out_dir, f"{baseA}_labeled.csv")
        outB = os.path.join(out_dir, f"{baseB}_labeled.csv")
        try:
            self.datasetA.df.to_csv(outA, index=False)
            self.datasetB.df.to_csv(outB, index=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", str(e)); return

        msg = f"Exported:\n{outA}\n{outB}"
        if conflict_cnt:
            msg += f"\n(Resolved {conflict_cnt} label conflicts by preferring A)"
        self.status.setText(msg)



# ---------------------------- main ----------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
