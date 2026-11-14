# %% [markdown]
# # Lab 4.3 · Field-condition tests
#
# Day 4 · Session 3 (Corruption and shift tests) · 25 min.
#
# On Day 1 a classifier that looked ready fell apart in the field. Before Rasid's detector is
# released you run the test that would have caught it: the same test frames drawn again under
# heavy rain, glare, at night and from a camera that was moved to a pole. You report the drop
# for each with an interval and make the release decision by the department's rule.
#
# **Tasks**
# - [ ] Average precision on the clean test frames · 4 min
# - [ ] Average precision under rain, glare, night and a moved camera · 8 min
# - [ ] The drop per condition with bootstrap intervals · 7 min
# - [ ] The release decision against the department's rule · 6 min
# - [ ] ✱ Stretch · Sweep the severity of the worst condition
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the
# evaluation or the decision.

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
model = U.load_ckpt(U.RoadDet(), "detector")
CONDITIONS = ("rain", "glare", "night", "pole")
N = U.pick(240, 90)

# %% [markdown]
# ## 1 · Average precision on the clean test frames
#
# `U.field_frames(condition, n)` draws the first n test frames under one condition: `"clean"`
# (daylight, sharp, the survey vehicle's view), `"rain"`, `"glare"`, `"night"` at high
# severity, or `"pole"` for the moved camera. The road spots are the same every time, so the
# sets are **paired**: frame 17 of the rain set is frame 17 of the clean set, in the rain.
#
# ✏️ **Exercise** · Return `(preds, gts)` for one condition: the detector's output on the
# frames and the ground truth of each frame (`U.det_gt(objects)`).

# %%
def run(condition):
    # BEGIN SOLUTION
    images, objects, _ = U.field_frames(condition, N)
    return U.detect(model, images), [U.det_gt(o) for o in objects]
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _run(condition):
    images, objects, _ = U.field_frames(condition, N)
    return U.detect(model, images), [U.det_gt(o) for o in objects]

try:
    results = {"clean": run("clean")}
except NotImplementedError:
    run = _run
    results = {"clean": run("clean")}
clean_ap = U.mean_ap(*results["clean"], 0.5)
print(f"clean test frames: {N}, AP50 {clean_ap:.3f}")

# %% [markdown]
# ## 2 · Average precision under rain, glare, night and a moved camera

# %%
for c in CONDITIONS:
    results[c] = run(c)
    print(f"{c:6s} AP50 {U.mean_ap(*results[c], 0.5):.3f}")

shown = {c: U.field_frames(c, 4)[0] for c in ("clean",) + CONDITIONS}
U.show([im for c in shown for im in shown[c]], cols=4, size=2.6,
       boxes=[b for c in shown for b in results[c][0][:4]], titles=[c for c in shown for _ in range(4)])

# %% [markdown]
# **Look at** the same four road spots down the columns. Which objects does the detector keep
# under every condition, and which does it lose first?

# %% [markdown]
# ## 3 · The drop per condition with bootstrap intervals
#
# "AP falls by 0.2 at night" on ninety frames is an estimate. Because the sets are paired,
# resample the *frames*: draw frame numbers with replacement, compute the clean AP and the
# condition's AP on the same drawn frames, and keep the difference. The spread of those
# differences is the interval of the drop.
#
# ✏️ **Exercise** · Return `(drop, low, high)`: the drop in mean AP50 from clean to the
# condition on all frames, and the 2.5th and 97.5th percentiles of the drop over `n_boot`
# resamples of the frame numbers. Use `np.random.default_rng(seed)` and `U.mean_ap`.

# %%
def paired_drop(clean, cond, n_boot=U.pick(300, 120), seed=0):
    # BEGIN SOLUTION
    (pc, gc), (pf, gf) = clean, cond
    drop = U.mean_ap(pc, gc, 0.5) - U.mean_ap(pf, gf, 0.5)
    r = np.random.default_rng(seed)
    n, vals = len(pc), []
    for _ in range(n_boot):
        idx = r.integers(0, n, n)
        vals.append(U.mean_ap([pc[i] for i in idx], [gc[i] for i in idx], 0.5)
                    - U.mean_ap([pf[i] for i in idx], [gf[i] for i in idx], 0.5))
    return float(drop), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    # END SOLUTION

# %%
try:
    drops = {c: paired_drop(results["clean"], results[c]) for c in CONDITIONS}
except NotImplementedError:
    drops = {c: tuple(v) for c, v in U.reference("d4_3_drops").items()}
for c, (d, lo, hi) in drops.items():
    assert lo <= d <= hi or abs(d - lo) < 0.05, "the interval should sit around the drop"
    print(f"{c:6s} drop in AP50 {d:+.3f}   95% interval [{lo:+.3f}, {hi:+.3f}]")
U.save_reference("d4_3_drops", {c: [round(x, 4) for x in v] for c, v in drops.items()})

# %% [markdown]
# ## 4 · The release decision against the department's rule
#
# Rakan Al-Shammari's rule for Rasid, written before the test was run:
#
# - a condition **passes** when the upper end of its drop interval is at most 0.10;
# - a condition is **released with a human check** when the upper end is at most 0.30: every
#   detection under that condition goes to an inspector before a repair crew is sent, and the
#   model card says so;
# - otherwise the condition is **held**: the detector does not run under it.
#
# ✏️ **Exercise** · Return the decision for one condition from the upper end of its interval:
# `"pass"`, `"human check"` or `"hold"`.

# %%
def decide(upper):
    # BEGIN SOLUTION
    return "pass" if upper <= 0.10 else "human check" if upper <= 0.30 else "hold"
    # END SOLUTION

# %%
try:
    decisions = {c: decide(v[2]) for c, v in drops.items()}
except NotImplementedError:
    decisions = {c: ("pass" if v[2] <= 0.10 else "human check" if v[2] <= 0.30 else "hold") for c, v in drops.items()}
assert set(decisions.values()) <= {"pass", "human check", "hold"}
memo = {"clean_ap50": round(clean_ap, 3), "frames": N,
        "conditions": {c: {"drop": round(drops[c][0], 3), "upper": round(drops[c][2], 3), "decision": decisions[c]} for c in CONDITIONS}}
print(f"Release memo · detector on {N} paired test frames · clean AP50 {clean_ap:.3f}")
for c, row in memo["conditions"].items():
    print(f"  {c:6s} drop {row['drop']:+.3f} (upper {row['upper']:+.3f})  ->  {row['decision'].upper()}")
U.save_reference("d4_3_memo", memo)

# %% [markdown]
# **Look at** the decision column as a whole. "Release" is rarely one word: it is a list of
# conditions, each with what happens under it. And note what the rule used: the upper end of
# the interval, not the estimate. A condition tested on few frames has a wide interval and
# fails the rule until someone collects more frames. That is the rule working.

# %% [markdown]
# ## ✱ Stretch · How bad before it breaks
#
# For the condition with the largest drop, draw the frames at severities 0.3, 0.5, 0.7 and 0.9
# (`roadkit` imports the generator as `U.P`: `U.P.render(*frame["address"], condition="night",
# severity=0.5, blur=0)` returns the image and the objects) and plot AP50 against severity. Is
# the fall gradual or a cliff? Where would you put the limit in the model card?

# %% [markdown]
# ## Check yourself
#
# 1. The training data held some rain and night frames, and the detector still drops in this
#    test. Why?
# 2. Why does pairing the frames give a narrower interval than two separate test sets would?
# 3. The moved camera is the worst condition. Is more night data the right next step?
#
# <details><summary>Answers</summary>
#
# 1. Training saw mild versions; the test uses heavy ones. And a few percent of the training
#    frames per condition teach less than the daylight majority does.
# 2. Each road spot is its own control: what differs between the two numbers is the condition,
#    not which potholes happened to be drawn.
# 3. No. A moved camera is a different shift: scale and angle, not lighting. The fixes are
#    frames from pole cameras, scale and rotation augmentation, and the camera-drift monitor
#    of lab 4.4 so that a moved camera is noticed at all.
# </details>
#
# ## Carry forward
#
# The memo is the core of the model card's results section: one row per condition with its
# interval and what happens under it. Add districts as rows the same way, so that no part of
# the city is served worse unnoticed.
