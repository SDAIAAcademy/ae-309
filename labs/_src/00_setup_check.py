# %% [markdown]
# # Setup check
#
# Run this once before Day 1. It imports every library the labs use, draws the Al-Manar data
# pack (a few minutes the first time), loads one supplied checkpoint and says what is missing.

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ae-309.git"
    if not Path("/content/computer-vision").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/computer-vision"], check=True)
    os.chdir("/content/computer-vision/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import importlib

missing = []
for name in ("numpy", "PIL", "matplotlib", "sklearn", "torch", "torchvision", "timm", "open_clip", "onnx", "onnxruntime"):
    try:
        mod = importlib.import_module(name)
        print(f"ok   {name:12s} {getattr(mod, '__version__', '')}")
    except Exception as err:  # noqa: BLE001
        missing.append(name)
        print(f"MISSING {name}: {err}")

# %%
import roadkit as U

manifest = U.ensure_data()
print(f"mode {U.MODE}, device {U.DEV}")
print(f"data pack: {manifest['segments']} road segments, {manifest['frames']} frames, {manifest['tiles']} defect crops")
print(f"this mode uses {len(U.frames())} frames and {len(U.tiles())} crops")

model = U.load_ckpt(U.TileNet(), "pilot_classifier")
crops = U.tiles("val")[:8]
print("pilot classifier says:", [U.TILE_CLASSES[k] for k in U.predict(model, U.tile_images(crops)).argmax(1)])
print("the labels say:       ", [t["label"] for t in crops])

# %%
if missing:
    print("Install what is missing, then run this notebook again:", ", ".join(missing))
else:
    print("Ready. On Colab, choose a GPU runtime for Day 2 and Day 3 (Runtime > Change runtime type).")
    if U.DEV == "cpu" and not U.QUICK:
        print("No GPU found: set the environment variable CV_QUICK=1 before starting Jupyter to use the quick mode.")
