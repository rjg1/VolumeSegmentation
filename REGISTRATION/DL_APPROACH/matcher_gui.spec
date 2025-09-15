# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files

base_dir = os.path.abspath(os.path.dirname(__file__))

# Detect which Qt binding you actually have at build time.
# Your code falls back (PySide6 -> PyQt5); we exclude the one you don't use
# to keep the EXE smaller and avoid plugin duplication.
excludes = []
try:
    import PySide6  # noqa:F401
    excludes.append("PyQt5")
    qt_binding = "PySide6"
except Exception:
    try:
        import PyQt5  # noqa:F401
        excludes.append("PySide6")
        qt_binding = "PyQt5"
    except Exception:
        qt_binding = None  # (shouldn’t happen if the app runs)

# Minimal hidden imports. Most are discovered automatically, but these help
# when imports happen dynamically (inside functions / try/except).
hiddenimports = [
    # core libs your app relies on
    "numpy",
    "pandas",
    "pyqtgraph",
    "scipy.optimize",
    "scipy.spatial",
    "tifffile",
    "shapely.geometry",
    "skimage.measure",
    # NOTE: If you want the raster IoU fallback (without Shapely), uncomment:
    # "matplotlib.path",
]

# Big packages you have installed but the app doesn’t import at runtime.
# (PyInstaller won’t bundle them unless imported, but this is a safety valve.)
excludes += [
    "torch", "cupy", "vtk", "pyvista",
    "ipykernel", "ipython", "notebook", "jupyter", "jupyterlab",
]

# Data files for a few libs that need binary/runtime assets.
datas = []
# shapely ships GEOS DLLs that must be bundled
try:
    import shapely  # noqa:F401
    datas += collect_data_files("shapely", include_py_files=False)
except Exception:
    pass
# tifffile + imagecodecs binaries (if installed) help with compressed TIFFs
try:
    import tifffile  # noqa:F401
    datas += collect_data_files("tifffile", include_py_files=False)
except Exception:
    pass
try:
    import imagecodecs  # noqa:F401
    datas += collect_data_files("imagecodecs", include_py_files=False)
except Exception:
    pass
# pyqtgraph sometimes needs small resource files
try:
    import pyqtgraph  # noqa:F401
    datas += collect_data_files("pyqtgraph", include_py_files=False)
except Exception:
    pass

block_cipher = None

a = Analysis(
    ['matcher_gui.py'],
    pathex=[base_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='matcher_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,          # set to False if your antivirus is picky
    console=False,     # GUI app (no console)
    icon=None,         # put a path to an .ico here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='matcher_gui'
)
