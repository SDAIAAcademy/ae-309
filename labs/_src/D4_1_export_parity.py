# %% [markdown]
# # Lab 4.1 · Export and check parity
#
# Day 4 · Session 1 (Export to an interchange format) · 25 min.
#
# The roadside unit has no PyTorch. Yousef Al-Maliki's unit runs an inference runtime that
# loads a model as a graph in an interchange format. You export the supplied detector to ONNX,
# prove that the copy answers like the original, first on raw outputs and then on final boxes,
# and then find out why the prototype's serving code gets different boxes from the same file.
#
# **Tasks**
# - [ ] Export the detector with a dynamic input size · 6 min
# - [ ] Run it in the runtime and compare the raw outputs · 6 min
# - [ ] Compare the final boxes after decoding · 6 min
# - [ ] Find and fix the planted preprocessing mismatch · 7 min
# - [ ] ✱ Stretch · Check parity at a second input size
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the
# export or the checks. Everything here runs on a processor.

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/computer-vision.git"
    if not Path("/content/computer-vision").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/computer-vision"], check=True)
    os.chdir("/content/computer-vision/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import numpy as np
import onnx
import onnxruntime as ort
import torch
import roadkit as U

U.ensure_data()
val = U.frames("val")[:32]
Xv, Gv = U.frame_images(val), [U.det_gt(f["objects"]) for f in val]
model = U.load_ckpt(U.RoadDet(), "detector").cpu()
OUT = U.CACHE / "export"
OUT.mkdir(parents=True, exist_ok=True)
print("torch", torch.__version__, "· onnx", onnx.__version__, "· onnxruntime", ort.__version__)

# %% [markdown]
# ## 1 · Export the detector with a dynamic input size
#
# `torch.onnx.export` runs the model once on an example input and records the operations. By
# default every dimension of that example is frozen into the graph. The unit will feed one
# frame at a time at a resolution you have not chosen yet (lab 4.2), so the batch, the height
# and the width must stay free.
#
# ✏️ **Exercise** · Complete `dynamic_axes`: for the input `image` the axes 0, 2 and 3 are
# free (name them `batch`, `height`, `width`); for the outputs `cls` and `box` the axes 0, 2
# and 3 are free too.

# %%
def export(model, path):
    example = torch.rand(1, 3, 192, 192)
    # BEGIN SOLUTION
    dynamic_axes = {"image": {0: "batch", 2: "height", 3: "width"},
                    "cls": {0: "batch", 2: "h", 3: "w"},
                    "box": {0: "batch", 2: "h", 3: "w"}}
    # END SOLUTION
    torch.onnx.export(model.eval(), example, str(path), input_names=["image"], output_names=["cls", "box"],
                      dynamic_axes=dynamic_axes, opset_version=17, dynamo=False)
    return path

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
path = OUT / "detector.onnx"
try:
    export(model, path)
except NotImplementedError:
    U.export_onnx(model, path)
onnx.checker.check_model(onnx.load(str(path)))
sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
shape = sess.get_inputs()[0].shape
assert not isinstance(shape[0], int) and not isinstance(shape[2], int), f"the input shape is frozen: {shape}"
print(f"exported: {path.name}, {path.stat().st_size / 1e6:.1f} MB, input shape {shape}")

# %% [markdown]
# ## 2 · Run it in the runtime and compare the raw outputs
#
# The same eight frames through both. Floating-point arithmetic is not associative and the
# runtime fuses and reorders operations, so the outputs will not be bit-identical. They must
# be equal within a tolerance you state.
#
# ✏️ **Exercise** · Return the largest absolute difference between the original's outputs and
# the runtime's, over both output maps. The runtime call is
# `sess.run(None, {"image": x})` with `x` a float32 NumPy array; it returns `[cls, box]`.

# %%
x = U.to_input(Xv[:8])  # float32, 8 x 3 x 192 x 192, RGB in 0..1


def max_difference(model, sess, x):
    # BEGIN SOLUTION
    with torch.no_grad():
        cls_t, box_t = model(torch.from_numpy(x))
    cls_o, box_o = sess.run(None, {"image": x})
    return float(max(np.abs(cls_t.numpy() - cls_o).max(), np.abs(box_t.numpy() - box_o).max()))
    # END SOLUTION

# %%
try:
    diff = max_difference(model, sess, x)
except NotImplementedError:
    diff = U.reference("d4_1_diff")
TOLERANCE = 1e-3
print(f"largest absolute difference: {diff:.2e}  (tolerance {TOLERANCE:.0e})")
assert diff < TOLERANCE
U.save_reference("d4_1_diff", diff)

# %% [markdown]
# ## 3 · Compare the final boxes after decoding
#
# Raw outputs within tolerance is necessary. What the maintenance team receives is boxes, and
# a tiny difference in a score can move a box across a threshold. So compare the product.
#
# ✏️ **Exercise** · For two lists of detections (one array per frame, rows of x0, y0, x1, y1,
# score, class), return the share of the first list's boxes that have a partner of the same
# class in the same frame of the second list with an IoU of at least 0.99.

# %%
def box_parity(dets_a, dets_b):
    # BEGIN SOLUTION
    hit = total = 0
    for a, b in zip(dets_a, dets_b):
        for row in a:
            total += 1
            same = b[b[:, 5] == row[5]]
            if len(same) and U.box_iou(row[None, :4], same[:, :4]).max() >= 0.99:
                hit += 1
    return hit / max(total, 1)
    # END SOLUTION

# %%
def _box_parity(dets_a, dets_b):
    hit = total = 0
    for a, b in zip(dets_a, dets_b):
        for row in a:
            total += 1
            same = b[b[:, 5] == row[5]]
            hit += bool(len(same) and U.box_iou(row[None, :4], same[:, :4]).max() >= 0.99)
    return hit / max(total, 1)

original = U.detect(model, Xv, score_thr=0.3)
exported = U.detect_onnx(sess, Xv, score_thr=0.3)
try:
    parity = box_parity(original, exported)
except NotImplementedError:
    box_parity = _box_parity
    parity = box_parity(original, exported)
print(f"{sum(len(d) for d in original)} boxes from the original, {sum(len(d) for d in exported)} from the export, parity {parity:.3f}")
assert parity == 1.0 and sum(len(d) for d in original) == sum(len(d) for d in exported)

# %% [markdown]
# ## 4 · Find and fix the planted preprocessing mismatch
#
# The prototype on the roadside unit loads the same `detector.onnx` and reports almost no
# potholes. Its serving function is below. The model file is proven: so the fault is in what
# the function feeds it.

# %%
def serve_prototype(sess, frame_bgr):
    """The roadside prototype. Its camera library hands over frames as BGR, uint8."""
    x = frame_bgr.astype(np.float32)[None].transpose(0, 3, 1, 2)
    cls, box = sess.run(None, {"image": x})
    return U.decode(cls, box, score_thr=0.3)[0]


camera = [np.ascontiguousarray(f[..., ::-1]) for f in Xv]   # what the unit's camera delivers
proto = [serve_prototype(sess, f) for f in camera]
print(f"prototype: {sum(len(d) for d in proto)} boxes, parity with the original {box_parity(original, proto):.3f}")

# %% [markdown]
# Compare `x` in the prototype with what the model was trained on: RGB, values from 0 to 1
# (`U.to_input` shows the training-side convention; the normalisation by ImageNet mean and
# standard deviation is inside the model, so it must not be applied again).
#
# ✏️ **Exercise** · Write the serving function correctly for a BGR uint8 frame.

# %%
def serve_fixed(sess, frame_bgr):
    # BEGIN SOLUTION
    rgb = frame_bgr[..., ::-1].astype(np.float32) / 255.0
    x = np.ascontiguousarray(rgb[None].transpose(0, 3, 1, 2))
    cls, box = sess.run(None, {"image": x})
    return U.decode(cls, box, score_thr=0.3)[0]
    # END SOLUTION

# %%
try:
    fixed = [serve_fixed(sess, f) for f in camera]
except NotImplementedError:
    fixed = exported
fixed_parity = box_parity(original, fixed)
print(f"fixed:     {sum(len(d) for d in fixed)} boxes, parity with the original {fixed_parity:.3f}")
assert fixed_parity == 1.0
U.save_reference("d4_1_parity", {"prototype": box_parity(original, proto), "fixed": fixed_parity})

# %% [markdown]
# **Look at** how the prototype failed: no exception, no warning, a plausible list of boxes,
# only a wrong one. Two faults were stacked: channel order and value range. The defence is
# the test you just wrote, kept in the repository and run on every export: ten frames, the
# original's boxes stored beside them, parity required.

# %% [markdown]
# ## ✱ Stretch · A second input size
#
# Run both models on the frames resized to 160 (`U.detect(model, Xv, size=160)` and
# `U.detect_onnx(sess, Xv, size=160)`). Is parity still 1.0? What would you have seen with the
# default export, without `dynamic_axes`?

# %% [markdown]
# ## Check yourself
#
# 1. The raw outputs differ by 0.00001 and you accept. What would make you refuse?
# 2. The normalisation sits inside the exported graph. What does that protect against, and
#    what does it leave open?
# 3. Why keep the parity test in the repository once it has passed?
#
# <details><summary>Answers</summary>
#
# 1. A difference orders of magnitude above float32 rounding, or any final box that differs:
#    signs of an operation exported differently, a layer left in training mode, a wrong opset.
# 2. Against a serving team forgetting or doubling the mean and standard deviation. Channel
#    order, value range and resizing still happen outside the graph, as the prototype showed.
# 3. Exports are repeated: a new checkpoint, a new library version, a quantised copy. Each is
#    a new chance for a silent difference, and the test costs seconds.
# </details>
#
# ## Carry forward
#
# `U.export_onnx`, `U.onnx_session` and `U.detect_onnx` wrap what you did here. Lab 4.2 starts
# from the same supplied checkpoint and asks what the exported model costs per frame.
