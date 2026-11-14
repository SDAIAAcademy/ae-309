# %% [markdown]
# # Lab 4.4 · Blur, monitor, choose
#
# Day 4 · Session 4 (Privacy and monitoring in the field) · 25 min.
#
# Street cameras see faces and number plates: personal data. Wejdan Al-Rashidi, the data
# protection officer, has one rule for Rasid: nothing readable is stored. You add blurring
# before storage, measure what still leaks and repair it, define two monitors that tell a
# moved or dirty camera from a rainy day, and choose your capstone task.
#
# **Tasks**
# - [ ] Blur faces and plates before storage · 7 min
# - [ ] Measure what still leaks and fix it · 6 min
# - [ ] Two camera-drift monitors on the 30-day stream · 8 min
# - [ ] Choose your capstone task and write its budget · 4 min
# - [ ] ✱ Stretch · Add a third monitor on detections per frame that ignores rain
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the
# blurring or the monitors. Every face and plate here is a drawing: no real person or vehicle.

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
from PIL import Image, ImageFilter
import roadkit as U

records = U.street()
print(len(records), "street frames;", sum(len(r["truth"]) for r in records), "faces and plates in them")
print("one detection:", records[0]["detections"][0])
U.show([U.street_image(r["id"]) for r in records[:4]], cols=4, size=3.2)

# %% [markdown]
# ## 1 · Blur faces and plates before storage
#
# A face and plate detector has already run on every frame: `detections` holds its boxes with
# a score. (The frames also carry `truth`, the regions that are really there. A deployed system
# never has it; you use it only to measure.)
#
# ✏️ **Exercise** · Return a copy of the image with every box blurred. Grow each box by `pad`
# pixels on every side first (stay inside the image). With Pillow: crop the region, apply
# `ImageFilter.GaussianBlur(4)`, paste it back.

# %%
def blur_regions(img, boxes, pad=0):
    # BEGIN SOLUTION
    im = Image.fromarray(img)
    w, h = im.size
    for x0, y0, x1, y1 in boxes:
        box = (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(w, int(np.ceil(x1 + pad))), min(h, int(np.ceil(y1 + pad))))
        if box[2] > box[0] and box[3] > box[1]:
            im.paste(im.crop(box).filter(ImageFilter.GaussianBlur(4)), box[:2])
    return np.asarray(im)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _blur_regions(img, boxes, pad=0):
    im = Image.fromarray(img)
    w, h = im.size
    for x0, y0, x1, y1 in boxes:
        box = (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(w, int(np.ceil(x1 + pad))), min(h, int(np.ceil(y1 + pad))))
        if box[2] > box[0] and box[3] > box[1]:
            im.paste(im.crop(box).filter(ImageFilter.GaussianBlur(4)), box[:2])
    return np.asarray(im)

try:
    blur_regions(U.street_image("ST01"), [[10, 10, 30, 30]])
except NotImplementedError:
    blur_regions = _blur_regions


def store(record, score_thr=0.5, pad=0):
    """What reaches the disk: the frame with every detection of at least score_thr blurred."""
    boxes = [d["box"] for d in record["detections"] if d["score"] >= score_thr]
    return blur_regions(U.street_image(record["id"]), boxes, pad)


first = store(records[0])
assert first.shape == U.street_image("ST01").shape and (first != U.street_image("ST01")).any()
U.show([U.street_image(r["id"]) for r in records[:2]] + [store(r) for r in records[:2]], cols=4, size=3.2,
       titles=["camera", "camera", "stored", "stored"])

# %% [markdown]
# ## 2 · Measure what still leaks and fix it
#
# "We blur faces" is a claim. The measurement: for every true face or plate, how much fine
# detail is left in the stored frame compared with the camera frame? `U.detail(img, box)`
# measures fine detail in a region. Call a region **leaked** when more than 40% of it is left.
#
# ✏️ **Exercise** · Return the share of true regions that leaked, over all frames, for a score
# threshold and a padding.

# %%
def leak_rate(score_thr=0.5, pad=0):
    # BEGIN SOLUTION
    leaked = total = 0
    for r in records:
        before, after = U.street_image(r["id"]), store(r, score_thr, pad)
        for g in r["truth"]:
            total += 1
            leaked += U.detail(after, g["box"]) > 0.4 * U.detail(before, g["box"])
    return leaked / total
    # END SOLUTION

# %%
def _leak_rate(score_thr=0.5, pad=0):
    leaked = total = 0
    for r in records:
        before, after = U.street_image(r["id"]), store(r, score_thr, pad)
        for g in r["truth"]:
            total += 1
            leaked += U.detail(after, g["box"]) > 0.4 * U.detail(before, g["box"])
    return leaked / total

try:
    leak_rate()
except NotImplementedError:
    leak_rate = _leak_rate


def blurred_area(score_thr, pad):
    """The share of all pixels that get blurred: what the fix costs the road engineers."""
    changed = [(store(r, score_thr, pad) != U.street_image(r["id"])).any(-1).mean() for r in records]
    return float(np.mean(changed))


leaks = {}
for thr, pad in [(0.5, 0), (0.5, 3), (0.2, 0), (0.2, 3)]:
    leaks[f"{thr}|{pad}"] = [round(leak_rate(thr, pad), 3), round(blurred_area(thr, pad), 4)]
    print(f"score >= {thr}, padding {pad}:  leaked {leaks[f'{thr}|{pad}'][0]:.1%} of faces and plates,"
          f"  blurred {leaks[f'{thr}|{pad}'][1]:.2%} of pixels")
assert leaks["0.2|3"][0] < leaks["0.5|0"][0]
U.save_reference("d4_4_leaks", leaks)

# %% [markdown]
# **Look at** the two causes separately. Padding repairs boxes that ran tight. The lower
# threshold catches the faces and plates the detector was unsure of, at the price of a few
# blurred patches of wall. For a road department that price is nothing: for privacy the
# operating point is high recall, the opposite of the pothole detector's. And the last row is
# still not zero. The model card states that number, and the retention rule covers the rest.

# %% [markdown]
# ## 3 · Two camera-drift monitors on the 30-day stream
#
# Each pole camera reports four daily statistics: `brightness`, `sharpness`, `offset_px` (how
# far the lane markings sit from where they were at installation) and `detections_per_frame`.
# A first monitor, "alert when brightness leaves its first-week range", shows the problem.

# %%
stream = U.camera_stream()
cameras = sorted({r["camera"] for r in stream})


def series(camera, field):
    return [r[field] for r in sorted(stream, key=lambda r: r["day"]) if r["camera"] == camera]


naive = [c for c in cameras
         if any(abs(v - np.mean(series(c, "brightness")[:7])) > 4 * np.std(series(c, "brightness")[:7]) for v in series(c, "brightness")[7:])]
print(f"brightness monitor: {len(naive)} of {len(cameras)} cameras alert")

# %% [markdown]
# It rained on three days and every camera went dark together. A monitor that pages the
# field team for weather is switched off within a week. A useful one fires on what a
# technician can fix, and stays quiet otherwise.
#
# ✏️ **Exercise** · Write `first_alert`: the first day (1-based) on which `run` consecutive
# values, that day and the days before it, all lie outside `[low, high]`; `None` if that
# never happens.

# %%
def first_alert(values, low, high, run):
    # BEGIN SOLUTION
    streak = 0
    for day, v in enumerate(values, start=1):
        streak = streak + 1 if (v < low or v > high) else 0
        if streak >= run:
            return day
    return None
    # END SOLUTION

# %%
def _first_alert(values, low, high, run):
    streak = 0
    for day, v in enumerate(values, start=1):
        streak = streak + 1 if (v < low or v > high) else 0
        if streak >= run:
            return day
    return None

try:
    assert first_alert([0, 9, 0, 9, 9, 9], -1, 1, 3) == 6 and first_alert([0, 0], -1, 1, 1) is None
except NotImplementedError:
    first_alert = _first_alert

# %% [markdown]
# ✏️ **Exercise** · Define the two monitors. Each is a field, a band and a run length. For
# sharpness the band is relative to the camera's own first week (its median over days 1 to 7):
# give `low` and `high` as fractions of it. Choose values that catch a camera that was moved
# and a lens that is getting dirty, and that stay silent on rain.

# %%
def my_monitors():
    # BEGIN SOLUTION
    return {
        "moved camera": {"field": "offset_px", "low": -5, "high": 5, "run": 2, "relative": False},
        "dirty lens": {"field": "sharpness", "low": 0.85, "high": 1.5, "run": 3, "relative": True},
    }
    # END SOLUTION

# %%
try:
    monitors = my_monitors()
except NotImplementedError:
    monitors = U.reference("d4_4_monitors")

alerts = {}
for name, m in monitors.items():
    for c in cameras:
        v = series(c, m["field"])
        ref = float(np.median(v[:7])) if m["relative"] else 1.0
        day = first_alert(v, m["low"] * ref, m["high"] * ref, m["run"])
        if day is not None:
            alerts[f"{name}|{c}"] = day
            print(f"{name:13s} {c}: alert on day {day}")
assert set(alerts) == {"moved camera|CAM-07", "dirty lens|CAM-03"}, f"expected one moved camera and one dirty lens, got {sorted(alerts)}"
U.save_reference("d4_4_monitors", monitors)
U.save_reference("d4_4_alerts", alerts)

# %% [markdown]
# **Look at** the days. CAM-07 was knocked on day 19, and the lens of CAM-03 has been gathering
# dust since day 9. How many days did each monitor need, and what did the camera count wrongly in
# between? Shorter runs and tighter bands alert sooner and page for rain: say which you chose.

# %% [markdown]
# ## 4 · Choose your capstone task and write its budget
#
# The capstone is an inspection service (the pothole detector or the crack segmenter) or a
# counting service, with a stated latency and accuracy budget and a field-conditions test set.
#
# ✏️ **Exercise** · Return your choice as a dictionary with `task` (`"inspection"` or
# `"counting"`), `accuracy_budget` (a metric and a floor, in words), `latency_budget_ms` (per
# frame, 95th percentile, on the roadside unit) and `field_conditions` (at least three).

# %%
def my_capstone():
    # BEGIN SOLUTION
    return {"task": "inspection",
            "accuracy_budget": "average precision at IoU 0.5 of at least 0.80 on clean frames and no condition more than 0.15 below it",
            "latency_budget_ms": 60,
            "field_conditions": ["rain", "glare", "night", "moved camera"]}
    # END SOLUTION

# %%
try:
    plan = my_capstone()
except NotImplementedError:
    plan = U.reference("d4_4_capstone")
assert plan["task"] in ("inspection", "counting") and len(plan["field_conditions"]) >= 3 and plan["latency_budget_ms"] > 0
for k, v in plan.items():
    print(f"{k:18s} {v}")
U.save_reference("d4_4_capstone", plan)

# %% [markdown]
# ## ✱ Stretch · Detections per frame, without the rain
#
# A moved camera also counts fewer vehicles. But so does every camera on a rainy day. Divide
# each camera's `detections_per_frame` by the median over all cameras on the same day, then
# monitor that ratio. Which cameras alert now, and on which day?

# %% [markdown]
# ## Check yourself
#
# 1. The pothole detector runs at a score threshold that balances false alarms and misses.
#    Why does the blurring run at a much lower one?
# 2. Blurring happens "before storage". Name two places where an unblurred frame could still
#    end up if the order were wrong.
# 3. Why is the sharpness band relative to each camera's first week and the offset band absolute?
#
# <details><summary>Answers</summary>
#
# 1. The costs are unequal. A blurred patch of wall costs nothing; one stored readable plate
#    is a breach of the rule. So recall is bought with false alarms.
# 2. A debug folder or log written by the detection step, a cache or message queue between
#    camera and storage, a backup of the raw stream, thumbnails in a labelling tool.
# 3. Cameras differ in lens and scene, so "sharp" is only meaningful against the camera's own
#    normal. A lane marking that moved by thirteen pixels has moved, on any camera.
# </details>
#
# ## Carry forward
#
# The leak rate, the two monitors and your capstone plan go into the model card: results per
# condition, the privacy controls with their measured recall, the monitors and who is paged.
