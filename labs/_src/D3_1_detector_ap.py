# %% [markdown]
# # Lab 3.1 · Train and score a detector
#
# Day 3 · Session 1 (Detection and its metrics) · 25 min.
#
# Rasid's detector finds potholes and patches in a road frame. It predicts, for every 8 x 8
# pixel cell, a score per class and a box; non-maximum suppression then keeps one box per
# object. You start from a supplied checkpoint that has seen one short pass of the data, write
# the suppression and the metric yourself, train a short schedule and measure what it bought,
# at two overlap thresholds.
#
# **Tasks**
# - [ ] Load the warm checkpoint and look at detections before and after suppression · 5 min
# - [ ] Train a short schedule · 7 min
# - [ ] Write average precision from sorted detections · 8 min
# - [ ] Average precision at 0.5 and 0.75, before and after · 5 min
# - [ ] ✱ Stretch · Sweep the suppression threshold and plot average precision
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the
# suppression, the metric or the training. On Colab choose a GPU runtime; on a laptop set `CV_QUICK=1`.

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
import torch
import roadkit as U

U.ensure_data()
train = U.frames("train")[: U.pick(1600, 400)]
val = U.frames("val")[: U.pick(600, 120)]
X, G = U.frame_images(train), [U.det_gt(f["objects"]) for f in train]
Xv, Gv = U.frame_images(val), [U.det_gt(f["objects"]) for f in val]
print(f"mode {U.MODE}, device {U.DEV}: {len(train)} training frames, {len(val)} validation frames,"
      f" {sum(len(g) for g in Gv)} boxes to find ({', '.join(U.DET_CLASSES)})")

model = U.load_ckpt(U.RoadDet(), "detector_warm")

# %% [markdown]
# ## 1 · Detections before and after suppression
#
# Every cell near an object predicts a box for it, so the raw output holds many boxes per
# pothole. Greedy non-maximum suppression: take the highest-scoring box, drop every remaining
# box that overlaps it by more than a threshold, repeat.
#
# ✏️ **Exercise** · Write it for one class. `boxes` is n x 4, `scores` has n values. Return
# the indices of the boxes you keep, best first. `U.box_iou(a, b)` gives an IoU matrix.

# %%
def nms(boxes, scores, iou_thr=0.5):
    # BEGIN SOLUTION
    order = list(np.argsort(-np.asarray(scores)))
    keep = []
    while order:
        best = order.pop(0)
        keep.append(int(best))
        if order:
            iou = U.box_iou(boxes[best][None], boxes[order])[0]
            order = [i for i, v in zip(order, iou) if v <= iou_thr]
    return keep
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
raw = U.detect(model, Xv[:8], score_thr=0.3, nms_iou=None)
one = raw[int(np.argmax([len(r) for r in raw]))]
one = one[one[:, 5] == np.bincount(one[:, 5].astype(int)).argmax()]
try:
    kept = nms(one[:, :4], one[:, 4])
except NotImplementedError:
    from torchvision.ops import nms as tv_nms
    kept = tv_nms(torch.from_numpy(one[:, :4]), torch.from_numpy(one[:, 4]), 0.5).tolist()
from torchvision.ops import nms as tv_nms
assert kept == tv_nms(torch.from_numpy(one[:, :4]), torch.from_numpy(one[:, 4]), 0.5).tolist(), "not the same boxes as torchvision.ops.nms"
print(f"one frame, one class: {len(one)} raw boxes above 0.3, {len(kept)} after suppression")

clean = U.detect(model, Xv[:8], score_thr=0.3)
U.show(list(Xv[:4]) + list(Xv[:4]), boxes=raw[:4] + clean[:4], cols=4, size=3.0,
       titles=["raw"] * 4 + ["after suppression"] * 4)

# %% [markdown]
# ## 2 · Train a short schedule
#
# Keep the warm model's detections first: they are the "before". Then train.
# `U.fit_detector` is a plain loop: a focal loss on the score maps, a box-overlap loss on the
# cells that sit on an object, mirrored frames as the only augmentation.

# %%
before = U.detect(model, Xv)
history = U.fit_detector(model, X, G, epochs=U.pick(3, 2), lr=2e-4)
after = U.detect(model, Xv)

# %% [markdown]
# ## 3 · Write average precision from sorted detections
#
# Sort all detections of a class by score. Walking down the list, each one is a true positive
# (it overlaps a still-unclaimed ground-truth box by at least the threshold) or a false
# positive. `U.tp_fp` does that matching and hands you `(scores, is_tp, n_gt)`. From there:
#
# - recall after k detections = true positives so far / n_gt
# - precision after k detections = true positives so far / k
# - make precision non-increasing from the right (each value becomes the maximum of itself
#   and everything after it)
# - average precision = the area under that curve: the sum of (step in recall) x (precision)
#
# ✏️ **Exercise** · Write it. Return 0.0 when there is no ground truth or no detection.

# %%
def average_precision(scores, is_tp, n_gt):
    # BEGIN SOLUTION
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    tp = np.cumsum(is_tp)
    recall = tp / n_gt
    precision = tp / np.arange(1, len(tp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(steps * precision))
    # END SOLUTION

# %%
try:
    toy = average_precision(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)
except NotImplementedError:
    average_precision = U.ap_from
    toy = average_precision(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)
assert abs(toy - (0.5 * 1.0 + 0.5 * 2 / 3)) < 1e-9, toy
print("toy example:", round(toy, 4))

# %% [markdown]
# ## 4 · Average precision at 0.5 and 0.75, before and after

# %%
def ap_table(preds):
    row = {}
    for thr in (0.5, 0.75):
        per_class = [average_precision(*U.tp_fp(preds, Gv, c, thr)) for c in range(len(U.DET_CLASSES))]
        row[f"AP{int(thr * 100)}"] = {**dict(zip(U.DET_CLASSES, per_class)), "mean": float(np.mean(per_class))}
    return row


results = {"before": ap_table(before), "after": ap_table(after)}
print(f"{'':8s} {'':5s} " + " ".join(f"{c:>8s}" for c in U.DET_CLASSES + ["mean"]))
for when, row in results.items():
    for thr, vals in row.items():
        print(f"{when:8s} {thr:5s} " + " ".join(f"{vals[c]:8.3f}" for c in U.DET_CLASSES + ["mean"]))
assert results["after"]["AP50"]["mean"] > results["before"]["AP50"]["mean"]
U.save_reference("d3_1_results", results)

# %% [markdown]
# **Look at** the distance between the two thresholds. At 0.5 a box counts when it is roughly
# on the object; at 0.75 it must fit. Remember lab 1.2: one of Al-Manar's annotators leaves a
# margin around every box and the others draw tight, so the ground truth itself does not agree
# with itself at 0.75. No training schedule removes that gap; a guideline line does.

# %% [markdown]
# ## ✱ Stretch · The suppression threshold
#
# Run `U.detect(model, Xv, nms_iou=t)` for t in 0.3, 0.5, 0.7 and `None`, and compute the mean
# AP50 each time (`U.mean_ap(preds, Gv, 0.5)`). Why does AP fall when suppression is too
# loose? What would a threshold that is too tight do to two potholes side by side?

# %% [markdown]
# ## Check yourself
#
# 1. A detector has AP50 of 0.90 and AP75 of 0.45. What kind of error dominates?
# 2. Why are detections sorted by score before precision and recall are computed?
# 3. Two detections overlap the same pothole by 0.8 and 0.7. How many true positives is that?
#
# <details><summary>Answers</summary>
#
# 1. Localisation: the boxes are on the objects and do not fit them. Or the labels disagree
#    with each other about how tight a box is.
# 2. Average precision summarises every operating point at once: each position in the sorted
#    list is the precision and recall you would get with the threshold set just there.
# 3. One. The better-scoring detection claims the box; the other is a false positive, a
#    duplicate. Suppression exists to remove it before it is counted.
# </details>
#
# ## Carry forward
#
# `U.mean_ap` is your `average_precision` averaged over the two classes. The fully trained
# detector is supplied as the checkpoint `detector`: labs 3.2, 4.1, 4.2 and 4.3 start from it.
