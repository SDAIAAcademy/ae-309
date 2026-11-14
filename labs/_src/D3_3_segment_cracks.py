# %% [markdown]
# # Lab 3.3 · Segment cracks
#
# Day 3 · Session 3 (Segmenting cracks) · 25 min.
#
# A box around a crack is mostly road. The maintenance team wants crack length and area, so
# Rasid needs a mask: one decision per pixel. Before training anything you build the baseline
# that needs no training, "cracks are darker than the road", and score it. Then you fine-tune
# the segmenter from a supplied checkpoint and compare the two, overall and by lighting.
#
# **Tasks**
# - [ ] A thresholding baseline and its overlap score · 6 min
# - [ ] Fine-tune the segmenter from the warm checkpoint · 8 min
# - [ ] IoU and Dice against the baseline · 5 min
# - [ ] Compare the two by lighting · 6 min
# - [ ] ✱ Stretch · Score with a two-pixel tolerance and explain the change
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the
# baseline, the metrics or the training. On Colab choose a GPU runtime; on a laptop set `CV_QUICK=1`.

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
import roadkit as U

U.ensure_data()


def with_cracks(split, n):
    rows = [f for f in U.frames(split) if any(o["cls"] in U.CRACK_CLASSES for o in f["objects"])]
    return rows[:n]


train, val = with_cracks("train", U.pick(800, 240)), with_cracks("val", U.pick(300, 80))
X, M = U.frame_images(train), np.stack([U.crack_mask(f) for f in train])
Xv, Mv = U.frame_images(val), np.stack([U.crack_mask(f) for f in val])
print(f"mode {U.MODE}, device {U.DEV}: {len(train)} training frames, {len(val)} validation frames")
print(f"crack pixels are {Mv.mean():.1%} of all validation pixels")
U.show(list(Xv[:4]) + list(Mv[:4]), cols=4, size=3.0, titles=["frame"] * 4 + ["crack mask"] * 4)

# %% [markdown]
# ## 1 · A thresholding baseline and its overlap score
#
# ✏️ **Exercise** · Return a boolean mask for one frame: the pixels whose grey level (the mean
# of the three channels) is more than `k` standard deviations below the frame's mean grey level.

# %%
def threshold_baseline(img, k=2.0):
    # BEGIN SOLUTION
    grey = img.astype(np.float32).mean(axis=-1)
    return grey < grey.mean() - k * grey.std()
    # END SOLUTION

# %% [markdown]
# Two overlap scores for masks, both from the same three counts (pixels in both masks, pixels
# in the prediction, pixels in the truth):
#
# - IoU = both / (prediction + truth - both)
# - Dice = 2 x both / (prediction + truth)
#
# ✏️ **Exercise** · Write both for stacks of masks (N x H x W, boolean), counting pixels over
# the whole stack. Return 0.0 when the denominator is zero.

# %%
def iou(pred, truth):
    # BEGIN SOLUTION
    both = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return float(both / union) if union else 0.0
    # END SOLUTION


def dice(pred, truth):
    # BEGIN SOLUTION
    both = np.logical_and(pred, truth).sum()
    total = pred.sum() + truth.sum()
    return float(2 * both / total) if total else 0.0
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    threshold_baseline(Xv[0])
except NotImplementedError:
    def threshold_baseline(img, k=2.0):
        grey = img.astype(np.float32).mean(axis=-1)
        return grey < grey.mean() - k * grey.std()
try:
    iou(Mv[:1], Mv[:1]), dice(Mv[:1], Mv[:1])
except NotImplementedError:
    def iou(pred, truth):
        union = np.logical_or(pred, truth).sum()
        return float(np.logical_and(pred, truth).sum() / union) if union else 0.0

    def dice(pred, truth):
        total = pred.sum() + truth.sum()
        return float(2 * np.logical_and(pred, truth).sum() / total) if total else 0.0
a, b = np.zeros((1, 4, 4), bool), np.zeros((1, 4, 4), bool)
a[0, :2], b[0, 1:3] = True, True
assert abs(iou(a, b) - 1 / 3) < 1e-9 and abs(dice(a, b) - 0.5) < 1e-9

base = np.stack([threshold_baseline(im) for im in Xv])
print(f"thresholding baseline: IoU {iou(base, Mv):.3f}, Dice {dice(base, Mv):.3f}")
U.show(list(Xv[:4]) + list(base[:4]), cols=4, size=3.0, titles=["frame"] * 4 + ["baseline"] * 4)

# %% [markdown]
# **Look at** what the baseline marks besides cracks: potholes, patch seams, shadows, the dark
# side of a lane marking. "Dark" is not "crack". It is still a number to beat, and it cost
# four lines.

# %% [markdown]
# ## 2 · Fine-tune the segmenter from the warm checkpoint
#
# `RoadSeg` uses the same ResNet-18 trunk as the detector as an encoder, and a small decoder
# that brings the features back to full resolution with skip connections, because a crack two
# pixels wide vanishes at one-eighth resolution. The checkpoint `segmenter_warm` has had one
# short pass.

# %%
model = U.load_ckpt(U.RoadSeg(), "segmenter_warm")
warm = U.segment(model, Xv) > 0.5
print(f"warm checkpoint:      IoU {iou(warm, Mv):.3f}, Dice {dice(warm, Mv):.3f}")
U.fit_segmenter(model, X, M, epochs=U.pick(3, 3), lr=1e-3)
tuned = U.segment(model, Xv) > 0.5
supplied = U.segment(U.load_ckpt(U.RoadSeg(), "segmenter"), Xv) > 0.5  # the full schedule, made for you

# %% [markdown]
# ## 3 · IoU and Dice against the baseline

# %%
scores = {"thresholding baseline": [iou(base, Mv), dice(base, Mv)],
          "segmenter, warm": [iou(warm, Mv), dice(warm, Mv)],
          "segmenter, fine-tuned": [iou(tuned, Mv), dice(tuned, Mv)],
          "segmenter, supplied": [iou(supplied, Mv), dice(supplied, Mv)]}
for name, (i_, d_) in scores.items():
    print(f"{name:24s} IoU {i_:.3f}   Dice {d_:.3f}")
assert scores["segmenter, fine-tuned"][0] > scores["thresholding baseline"][0]
U.save_reference("d3_3_scores", scores)
U.show(list(Xv[:4]) + list(Mv[:4]) + list(tuned[:4]), cols=4, size=3.0,
       titles=["frame"] * 4 + ["truth"] * 4 + ["segmenter"] * 4)

# %% [markdown]
# **Look at** Dice against IoU: Dice is always the larger of the two for the same masks
# (Dice = 2 IoU / (1 + IoU)). A report that says "0.7" without saying which has not said much.
# The last row is the checkpoint `segmenter`: the same model after five passes over every
# training frame with cracks, about half an hour on a laptop processor. Your
# short schedule sits somewhere between the warm row and that one.
# And look at the masks: on a line two or three pixels wide, a prediction one pixel off loses
# a large share of its overlap. Thin structures make every overlap score look harsh.

# %% [markdown]
# ## 4 · Compare the two by lighting
#
# ✏️ **Exercise** · Return `{condition: IoU}` for one stack of predicted masks over the
# validation frames (`val[i]["condition"]`), for the conditions that occur.

# %%
def iou_by_condition(pred):
    # BEGIN SOLUTION
    out = {}
    for c in U.CONDITIONS:
        idx = [i for i, f in enumerate(val) if f["condition"] == c]
        if idx:
            out[c] = iou(pred[idx], Mv[idx])
    return out
    # END SOLUTION

# %%
def _iou_by_condition(pred):
    out = {}
    for c in U.CONDITIONS:
        idx = [i for i, f in enumerate(val) if f["condition"] == c]
        if idx:
            out[c] = iou(pred[idx], Mv[idx])
    return out

try:
    by_light = {"baseline": iou_by_condition(base), "fine-tuned": iou_by_condition(tuned), "supplied": iou_by_condition(supplied)}
except NotImplementedError:
    by_light = {"baseline": _iou_by_condition(base), "fine-tuned": _iou_by_condition(tuned), "supplied": _iou_by_condition(supplied)}
print(f"{'lighting':8s} {'frames':>6s} {'baseline':>9s} {'fine-tuned':>11s} {'supplied':>9s}")
for c in by_light["baseline"]:
    n = sum(f["condition"] == c for f in val)
    print(f"{c:8s} {n:6d} {by_light['baseline'][c]:9.3f} {by_light['fine-tuned'][c]:11.3f} {by_light['supplied'][c]:9.3f}")
U.save_reference("d3_3_by_light", by_light)

# %% [markdown]
# **Look at** where the baseline collapses and where it holds. A rule about darkness is a
# rule about lighting: the same crack is "dark" at noon and invisible to the rule at night or
# under glare. The segmenter's weakest column is your next data request.

# %% [markdown]
# ## ✱ Stretch · A two-pixel tolerance
#
# Grow the truth masks by two pixels (a 5 x 5 maximum filter: `scipy.ndimage.binary_dilation`
# with `iterations=2`, or shifted copies combined with `|`). Count a predicted pixel as correct
# when it falls inside the grown truth, and a truth pixel as found when a grown prediction
# covers it. How do precision and recall change, and what does that say about the errors?

# %% [markdown]
# ## Check yourself
#
# 1. Crack pixels are a few percent of all pixels. Why is pixel accuracy a useless score here?
# 2. The baseline's IoU is far from zero in daylight. Why train a model at all?
# 3. The segmenter's IoU is 0.6. A colleague calls that poor against a detector's AP of 0.9.
#    What is wrong with the comparison?
#
# <details><summary>Answers</summary>
#
# 1. A mask of all "no crack" scores in the high nineties. Overlap scores ignore the
#    true-negative pixels that dominate.
# 2. Daylight is one condition. The table by lighting shows where a rule about brightness
#    fails; and the baseline cannot tell a crack from any other dark thing.
# 3. Different metrics on different tasks. For structures two pixels wide, an IoU of 0.6 can
#    mean masks that follow every crack; judge by the tolerance score and by looking.
# </details>
#
# ## Carry forward
#
# The fully trained segmenter is supplied as the checkpoint `segmenter`. An inspection capstone
# may use the detector, the segmenter or both, and reports overlap by lighting and district.
