# roi_matcher.spec
block_cipher = None

import os
from PyInstaller.utils.hooks import collect_all

qt_excludes = []

# Collect everything from numpy (datas, binaries, hiddenimports)
np_datas, np_binaries, np_hidden = collect_all('numpy')

a = Analysis(
    ['matcher_gui.py'],
    pathex=[os.getcwd()],
    binaries=np_binaries,   # pass binaries here
    datas=np_datas,         # pass datas here
    hiddenimports=list(np_hidden),  # pass hidden imports here
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=qt_excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='roi_matcher',
    console=False,
    strip=True,
    upx=True,
    disable_windowed_traceback=True,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False,
    upx=True,
    name='roi_matcher'
)
