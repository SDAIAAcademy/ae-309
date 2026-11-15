# %% [markdown]
# # Lab 4.2 · Fit the latency budget
#
# Day 4 · Session 2 (Resolution and precision against latency) · 25 min.
#
# Yousef Al-Maliki's roadside unit has a two-core processor, no accelerator, and gets its
# model over a metered cellular link. Its budget for Rasid's detector:
#
# - **latency**: at most 60 ms per frame at the 95th percentile, on the unit;
# - **size**: a model file of at most 2 MB;
# - the unit runs about **3 times slower** than one core pair of a current laptop, so the
#   laptop target is 20 ms.
#
# You measure latency and accuracy at three input resolutions and two numeric precisions,
# build the benchmark table and choose the operating point.
#
# **Tasks**
# - [ ] Latency at three resolutions: median and 95th percentile · 6 min
# - [ ] Average precision at each resolution · 5 min
# - [ ] Quantise to 8-bit integers with calibration frames · 6 min
# - [ ] The benchmark table: latency, size, accuracy · 4 min
# - [ ] Choose the operating point for the unit's budget · 4 min
# - [ ] ✱ Stretch · Repeat the latency runs with one thread
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the
# measurements or the choice. Close other programs: latency is measured on your processor.

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

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
import roadkit as U

U.ensure_data()
BUDGET_MS, UNIT_FACTOR, BUDGET_MB, THREADS = 60, 3, 2.0, 2
SIZES = (192, 160, 128)
val = U.frames("val")[: U.pick(400, 120)]
Xv, Gv = U.frame_images(val), [U.det_gt(f["objects"]) for f in val]
calibration = U.frame_images(U.frames("train")[: U.pick(64, 32)])
OUT = U.CACHE / "export"
OUT.mkdir(parents=True, exist_ok=True)
fp32_path = U.export_onnx(U.load_ckpt(U.RoadDet(), "detector"), OUT / "detector_fp32.onnx")
fp32 = U.onnx_session(fp32_path, threads=THREADS)
print(f"mode {U.MODE}: {len(val)} validation frames, model file {fp32_path.stat().st_size / 1e6:.2f} MB")

# %% [markdown]
# ## 1 · Latency at three resolutions: median and 95th percentile
#
# A unit that is fast on average and slow on one frame in twenty drops that frame. So the
# budget is written on the 95th percentile. `U.latency_ms(fn, n)` warms up, then times `fn`
# n times and returns the milliseconds.
#
# ✏️ **Exercise** · Return `(median, p95)` of a list of latencies, in milliseconds.

# %%
def summarise(ms):
    # BEGIN SOLUTION
    return float(np.median(ms)), float(np.percentile(ms, 95))
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    assert summarise(list(range(1, 101))) == (50.5, 95.05)
except NotImplementedError:
    def summarise(ms):
        return float(np.median(ms)), float(np.percentile(ms, 95))


def measure(sess, size, n=U.pick(100, 40)):
    x = U.to_input(Xv[:1], size)  # one frame, as the unit sees it
    return summarise(U.latency_ms(lambda: sess.run(None, {"image": x}), n=n))


latency = {("fp32", s): measure(fp32, s) for s in SIZES}
for (prec, s), (med, p95) in latency.items():
    print(f"{prec} {s}x{s}: median {med:6.2f} ms   p95 {p95:6.2f} ms   on the unit about {p95 * UNIT_FACTOR:6.1f} ms")

# %% [markdown]
# **Look at** the ratio between 192 and 128. The frame has (192/128)² = 2.25 times the
# pixels, and a convolutional network does work in proportion to pixels. Is your ratio close?
# What else is in a "per frame" latency that does not shrink with the picture?

# %% [markdown]
# ## 2 · Average precision at each resolution
#
# Smaller frames are cheaper, and a small pothole shrinks with them. `U.detect_onnx` resizes
# the frames, runs the session and returns boxes in the pixels of the original frame.
#
# ✏️ **Exercise** · Return the mean AP at IoU 0.5 of a session at one input size on the
# validation frames (`U.mean_ap(preds, Gv, 0.5)`).

# %%
def accuracy(sess, size):
    # BEGIN SOLUTION
    return U.mean_ap(U.detect_onnx(sess, Xv, size=size), Gv, 0.5)
    # END SOLUTION

# %%
try:
    accuracy(fp32, 192)
except NotImplementedError:
    def accuracy(sess, size):
        return U.mean_ap(U.detect_onnx(sess, Xv, size=size), Gv, 0.5)
ap = {("fp32", s): accuracy(fp32, s) for s in SIZES}
for (prec, s), v in ap.items():
    print(f"{prec} {s}x{s}: AP50 {v:.3f}")
assert ap[("fp32", 192)] > ap[("fp32", 128)]

# %% [markdown]
# ## 3 · Quantise to 8-bit integers with calibration frames
#
# The weights and activations are 32-bit floats. Static quantisation stores and computes them
# as 8-bit integers: a quarter of the file, and integer arithmetic that processors run
# faster. To map floats to integers the tool must know the range of every activation, which
# it learns by running a few real frames: the **calibration** set. It asks for them through a
# reader with one method.
#
# ✏️ **Exercise** · Write `get_next`: each call returns the next calibration frame as
# `{"image": array}` (float32, 1 x 3 x size x size, from `U.to_input(frames, size)`), and
# `None` when the frames are used up.

# %%
class Reader(CalibrationDataReader):
    def __init__(self, frames, size):
        self.frames, self.size, self.i = frames, size, 0

    def get_next(self):
        # BEGIN SOLUTION
        if self.i >= len(self.frames):
            return None
        x = U.to_input(self.frames[self.i:self.i + 1], self.size)
        self.i += 1
        return {"image": x}
        # END SOLUTION

# %%
class _Reader(CalibrationDataReader):
    def __init__(self, frames, size):
        self.it = iter([{"image": U.to_input(frames[i:i + 1], size)} for i in range(len(frames))])

    def get_next(self):
        return next(self.it, None)

try:
    Reader(calibration[:1], 192).get_next()
except NotImplementedError:
    Reader = _Reader

int8_path = OUT / "detector_int8.onnx"
quantize_static(str(fp32_path), str(int8_path), Reader(calibration, 192), quant_format=QuantFormat.QDQ,
                per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
int8 = U.onnx_session(int8_path, threads=THREADS)
mb = {"fp32": fp32_path.stat().st_size / 1e6, "int8": int8_path.stat().st_size / 1e6}
print(f"fp32 {mb['fp32']:.2f} MB  ->  int8 {mb['int8']:.2f} MB")
for s in SIZES:
    latency[("int8", s)] = measure(int8, s)
    ap[("int8", s)] = accuracy(int8, s)  # re-measure accuracy after every change to the model

# %% [markdown]
# ## 4 · The benchmark table

# %%
table = []
print(f"{'precision':9s} {'input':>7s} {'median':>8s} {'p95':>8s} {'unit p95':>9s} {'file MB':>8s} {'AP50':>6s}")
for prec in ("fp32", "int8"):
    for s in SIZES:
        med, p95 = latency[(prec, s)]
        row = {"precision": prec, "size": s, "median_ms": round(med, 2), "p95_ms": round(p95, 2),
               "unit_p95_ms": round(p95 * UNIT_FACTOR, 1), "file_mb": round(mb[prec], 2), "ap50": round(ap[(prec, s)], 3)}
        table.append(row)
        print(f"{prec:9s} {s:4d}x{s:<3d}{med:8.2f} {p95:8.2f} {row['unit_p95_ms']:9.1f} {row['file_mb']:8.2f} {row['ap50']:6.3f}")
U.save_reference("d4_2_table", table)

# %% [markdown]
# **Look at** three things. What did eight bits cost in AP50 at each size, and is the cost the
# same at every size? What did they buy in latency on your processor (the gain depends on the
# processor's integer instructions: report what you measured, never what a blog promised)?
# And which rows the 2 MB limit removes before latency is even considered.

# %% [markdown]
# ## 5 · Choose the operating point
#
# ✏️ **Exercise** · Among the rows that meet both budgets (`unit_p95_ms` at most
# `BUDGET_MS`, `file_mb` at most `BUDGET_MB`), return the one with the highest `ap50`. Return
# `None` when no row qualifies.

# %%
def choose(table):
    # BEGIN SOLUTION
    ok = [r for r in table if r["unit_p95_ms"] <= BUDGET_MS and r["file_mb"] <= BUDGET_MB]
    return max(ok, key=lambda r: r["ap50"]) if ok else None
    # END SOLUTION

# %%
try:
    point = choose(table)
except NotImplementedError:
    ok = [r for r in table if r["unit_p95_ms"] <= BUDGET_MS and r["file_mb"] <= BUDGET_MB]
    point = max(ok, key=lambda r: r["ap50"]) if ok else None
if point is None:
    print("No row meets the budget on this processor. That is a result: report it, with the nearest row.")
else:
    assert point["file_mb"] <= BUDGET_MB and point["unit_p95_ms"] <= BUDGET_MS
    print(f"operating point: {point['precision']} at {point['size']}x{point['size']}: AP50 {point['ap50']}, "
          f"about {point['unit_p95_ms']} ms p95 on the unit, {point['file_mb']} MB")
U.save_reference("d4_2_point", point)

# %% [markdown]
# **Look at** the number you did not measure: the factor of 3 between your laptop and the
# unit is the case's assumption. A real release measures on the unit itself, at its
# temperature, with the camera pipeline running. Your table says which rows are worth
# carrying there.

# %% [markdown]
# ## ✱ Stretch · One thread
#
# Build the sessions again with `U.onnx_session(path, threads=1)` and repeat the latency
# runs. Does halving the threads double the latency? At which input size is the gap smallest,
# and why?

# %% [markdown]
# ## Check yourself
#
# 1. The median meets the budget and the 95th percentile does not. Does the model fit?
# 2. Why must accuracy be measured again after quantisation, on the same frames?
# 3. Batch size 8 gives more frames per second than batch size 1. Why does the unit not use it?
#
# <details><summary>Answers</summary>
#
# 1. No. The budget is on the tail: one frame in twenty would arrive late or be dropped.
# 2. Quantisation changes every number in the network a little. Most models lose little, some
#    lose a lot on small or low-contrast objects, and nothing but a measurement tells which.
# 3. Batching trades latency for throughput: the first frame waits for the eighth. A camera
#    delivers one frame at a time and the budget is per frame.
# </details>
#
# ## Carry forward
#
# The benchmark table and the chosen operating point are the evidence for outcome 5. The
# capstone asks for the same table against the budget you state, measured on your machine,
# with the machine named.
