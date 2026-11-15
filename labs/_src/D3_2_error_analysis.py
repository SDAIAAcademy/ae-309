# %% [markdown]
# # Lab 3.2 · Sort fifty errors
#
# Day 3 · Session 2 (Detector error analysis) · 25 min.
#
# The supplied detector ran on the roadside prototype for a week: validation frames as the
# field delivers them, in every light, one in five from a pole camera, with the prototype's
# settings. Fifty of its errors are in a file, with no explanation. You sort them into four types,
# localisation, class, duplicate and missed, first by eye and then by rule, estimate what
# fixing each type would be worth, and choose the next improvement.
#
# **Tasks**
# - [ ] Sort 12 errors by eye · 6 min
# - [ ] Write the rule that types an error · 8 min
# - [ ] Type all 50 and count · 4 min
# - [ ] What fixing each type would be worth · 4 min
# - [ ] Choose a fix and justify it · 3 min
# - [ ] ✱ Stretch · Break the missed defects down by lighting
#
# AI policy for this lab: **AI off**. Reading errors and choosing the fix is the skill, and
# the supervised check asks for it.

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

import json
from collections import Counter
import numpy as np
import roadkit as U

U.ensure_data()
errors = json.loads((U.DATA / "supplied" / "detector_errors.json").read_text())
print(len(errors), "errors. One of them:")
print({k: v for k, v in errors[0].items() if k != "others"})


def frame_of(e):
    """The frame of an error as the prototype saw it, with its ground-truth boxes."""
    return U.render_frame(e["address"], **e["drawn"])

# %% [markdown]
# Each error is either a **detection** that was not counted as correct (with its box, score
# and class) or a **ground-truth box with no detection** near it. `others` lists the other
# detections in the same frame. A detection is correct when it has the right class and an
# IoU of at least 0.5 with a ground-truth box that no better-scoring detection has claimed.
#
# ## 1 · Sort 12 errors by eye
#
# Red: the box in question. Teal: ground truth (the title of each picture gives the classes).
# Yellow: the other detections.

# %%
def view(e):
    """The frame of an error with the boxes to draw: the error, the ground truth, the others."""
    img, gts = frame_of(e)
    boxes = [list(g[:4]) + [0, 1] for g in gts] + [o[:4] + [0, 2] for o in e["others"]] + [e["box"] + [0, 0]]
    truth = "+".join(U.DET_CLASSES[int(g[4])] for g in gts) or "none"
    return img, boxes, f"{e['id']} {e['cls']} | gt {truth}"


first = errors[:12]
shots = [view(e) for e in first]
U.show([s[0] for s in shots], boxes=[s[1] for s in shots], titles=[s[2] for s in shots], cols=4, size=3.2)

# %% [markdown]
# ✏️ **Exercise** · Return your type for each of the first 12 errors, as a dictionary
# `{"E01": "missed", ...}`. The four types:
#
# - `localisation`: the right class, on the object, the box does not fit (IoU from 0.1 to 0.5)
# - `class`: the box fits (IoU at least 0.5), the class is wrong
# - `duplicate`: right class and a good fit, but a better-scoring detection already has that object
# - `missed`: a ground-truth box that no detection came near

# %%
def by_eye():
    # BEGIN SOLUTION
    key = json.loads((U.DATA / "supplied" / "detector_errors_key.json").read_text())
    return {e["id"]: key[e["id"]] for e in first}  # the solution reads the trainer's key
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
TYPES = ("localisation", "class", "duplicate", "missed")
try:
    eye = by_eye()
except NotImplementedError:
    eye = {}
assert set(eye.values()) <= set(TYPES)
print("sorted by eye:", dict(Counter(eye.values())))

# %% [markdown]
# ## 2 · Write the rule that types an error
#
# Your eye settles twelve. A rule settles thousands, and the same way every time.
#
# ✏️ **Exercise** · Return the type of one error. `gts` holds the frame's ground truth, rows
# of x0, y0, x1, y1, class index (`U.DET_CLASSES` gives the names). For a detection: find the
# ground-truth box it overlaps most (`U.box_iou`), then apply the four definitions in the
# order class, duplicate, localisation. A detection that fits none of them is `background`.

# %%
def error_type(e, gts):
    # BEGIN SOLUTION
    if e["kind"] != "detection":
        return "missed"
    if len(gts) == 0:
        return "background"
    iou = U.box_iou([e["box"]], gts[:, :4])[0]
    j = int(iou.argmax())
    same_class = U.DET_CLASSES[int(gts[j, 4])] == e["cls"]
    if iou[j] >= 0.5 and not same_class:
        return "class"
    if iou[j] >= 0.5:
        return "duplicate"   # correct detections are not in the file: a better one holds this object
    if iou[j] >= 0.1 and same_class:
        return "localisation"
    return "background"
    # END SOLUTION

# %%
def _error_type(e, gts):
    if e["kind"] != "detection":
        return "missed"
    if len(gts) == 0:
        return "background"
    iou = U.box_iou([e["box"]], gts[:, :4])[0]
    j = int(iou.argmax())
    same = U.DET_CLASSES[int(gts[j, 4])] == e["cls"]
    if iou[j] >= 0.5:
        return "duplicate" if same else "class"
    return "localisation" if iou[j] >= 0.1 and same else "background"

try:
    error_type(errors[0], frame_of(errors[0])[1])
except NotImplementedError:
    error_type = _error_type

# %% [markdown]
# ## 3 · Type all 50 and count

# %%
typed = {e["id"]: error_type(e, frame_of(e)[1]) for e in errors}
counts = Counter(typed.values())
for t in TYPES:
    print(f"{t:13s} {counts[t]:3d}  {'#' * counts[t]}")
key = json.loads((U.DATA / "supplied" / "detector_errors_key.json").read_text())
assert typed == key, {i: (typed[i], key[i]) for i in key if typed[i] != key[i]}
if eye:
    agree = sum(eye[i] == typed[i] for i in eye)
    print(f"\nyour eye and the rule agree on {agree} of {len(eye)}")
U.save_reference("d3_2_counts", dict(counts))

# %% [markdown]
# **Look at** the errors where your eye and the rule disagree, if any. Was the rule wrong, or
# did the picture mislead you? Both happen: a rule with a threshold at 0.5 calls a box at
# 0.49 a localisation error and one at 0.51 fine.

# %% [markdown]
# ## 4 · What fixing each type would be worth
#
# Fifty hand-picked errors say what exists, not what matters. For that, ask an oracle: take
# every detection the prototype reported on the validation frames, repair one type of error
# perfectly, leave the rest, and measure again. The gain is an upper bound on what a fix of
# that type can buy. `U.oracle_fix` does the repair. The measure is what the maintenance
# team lives with: **precision** (of the boxes sent to them, how many are right) and
# **recall** (of the defects on the road, how many were sent), at the prototype's settings.

# %%
X, objects, drawn = U.field_mix("val", U.pick(600, 180))
Gv = [U.det_gt(o) for o in objects]
detector = U.load_ckpt(U.RoadDet(), "detector")
preds = U.detect(detector, X, score_thr=0.3, nms_iou=0.7)  # the prototype's settings
p0, r0, tp, fp, fn = U.precision_recall(preds, Gv)
print(f"{len(X)} field frames, as configured: precision {p0:.3f}, recall {r0:.3f}  ({tp} right, {fp} false alarms, {fn} missed)")
gains = {}
for t in ("duplicate", "class", "localisation", "missed"):
    p1, r1, *_ = U.precision_recall(U.oracle_fix(preds, Gv, t), Gv)
    f0, f1 = 2 * p0 * r0 / (p0 + r0), 2 * p1 * r1 / (p1 + r1)
    gains[t] = f1 - f0
    print(f"  all {t:13s} errors fixed: precision {p1:.3f} ({p1 - p0:+.3f})  recall {r1:.3f} ({r1 - r0:+.3f})  F1 {f1 - f0:+.3f}")
U.save_reference("d3_2_gains", {"precision": p0, "recall": r0, **gains})

# %% [markdown]
# ## 5 · Choose a fix and justify it
#
# The four types point at four different pieces of work:
#
# | Type | Usual fix |
# |---|---|
# | duplicate | tighten non-maximum suppression: a setting, no training |
# | class | more examples of the confused pair, or a clearer class definition |
# | localisation | consistent box guidelines, a box loss or a finer output grid |
# | missed | more data of the missed kind (small, dark, in rain), or a lower score threshold |
#
# ✏️ **Exercise** · Return a dictionary with `fix` (one of the four types) and `why` (one
# sentence that uses a number from the oracle table and says what the work is).

# %%
def choose_fix(gains):
    # BEGIN SOLUTION
    best = max(gains, key=gains.get)
    work = {"duplicate": "tighten non-maximum suppression from 0.7 to 0.5, which needs no training",
            "class": "collect and label more round patches and shallow potholes",
            "localisation": "write the box-tightness rule into the guideline and relabel",
            "missed": "collect and label the conditions and sizes that are missed"}[best]
    return {"fix": best, "why": f"Fixing every {best} error is worth up to {gains[best]:+.3f} F1, the largest gain: {work}."}
    # END SOLUTION

# %%
try:
    choice = choose_fix(gains)
except NotImplementedError:
    choice = U.reference("d3_2_choice")
assert choice["fix"] in TYPES and choice["why"]
print(choice["fix"].upper(), "·", choice["why"])
U.save_reference("d3_2_choice", choice)

# %% [markdown]
# **Look at** the cost beside the gain. The largest oracle gain is not always the next step:
# a smaller gain that costs one changed setting is done before lunch, and the table is then
# measured again, because fixing one type moves the others.

# %% [markdown]
# ## ✱ Stretch · Missed, by lighting
#
# For each field frame, count the ground-truth boxes with no detection at IoU 0.1 or more,
# and group the counts by `drawn[i]["condition"]`. Divide by the number of boxes per
# condition. Which lighting does the detector miss most in, and is the slice large enough to say so?

# %% [markdown]
# ## Check yourself
#
# 1. A detection has the right class and an IoU of 0.3 with a pothole. Which type, and which
#    two different root causes could produce it?
# 2. Why does the lab call the oracle gain an upper bound?
# 3. Tightening suppression removes duplicates. What new error can it create?
#
# <details><summary>Answers</summary>
#
# 1. Localisation. The model may draw poor boxes, or the labels may be inconsistent about
#    how tight a box is, so that "right" was never well defined.
# 2. No real fix repairs every error of a type and touches nothing else. The oracle does.
# 3. Missed objects: two potholes side by side overlap, and a strict threshold deletes the
#    second as if it were a duplicate of the first.
# </details>
#
# ## Carry forward
#
# The error breakdown, the oracle table and a justified next step are the evidence for
# outcome 3. In the capstone the assessor picks one failure case: this is how you read it.
