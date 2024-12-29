# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

# the data files that the app depends on
# the assets folder of faster-whisper, containing necessary onnx files,
# the same for silero_vad
datas = collect_data_files("faster_whisper", subdir="assets") + collect_data_files(
    "silero_vad", subdir="data"
)
# the qss themes
datas.append(("app/styles", "./styles"))

# binaries dependencies
# ffmpeg related files
binaries = []
binaries.append(("app/binaries/ffmpeg.exe", '.'))
binaries.append(("app/binaries/ffplay.exe", '.'))
binaries.append(("app/binaries/ffprobe.exe", '.'))


a = Analysis(
    ["app\\app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=["importlib_resources", "silero_vad", "silero_vad.data"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="STT app",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # leave console as True for debug reason and to avoid Windows Defender flag app as Trojan
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="app",
)
