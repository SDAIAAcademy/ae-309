# %% [markdown]
# # Lab 3.4 · Track and count
#
# Day 3 · Session 4 (Tracking and counting) · 25 min.
#
# Abeer Al-Juhani's traffic team wants vehicle counts from pole cameras. A detector gives
# boxes per frame; a count needs to know that the box in this frame is the vehicle of the
# last one. You build tracking by detection on twelve synthetic clips with ground-truth tracks:
# association by overlap, a counting rule that counts once, then the two numbers that judge
# it, count error per clip and identity switches.
#
# **Tasks**
# - [ ] Associate detections to tracks by overlap · 8 min
# - [ ] Count each track once when it crosses the line · 5 min
# - [ ] Count error per clip · 4 min
# - [ ] Identity switches against the ground-truth tracks · 4 min
# - [ ] Tune patience and minimum hits on the gantry and night clips · 4 min
# - [ ] ✱ Stretch · Predict each track forward at constant velocity
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the tracker.
# The clips are stored as tracks and detections, not as video: the detections imitate a vehicle
# detector, with misses, jitter and false alarms.

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

clips = U.clips()
for c in clips[:4]:
    print(c["id"], f"{c['kind']:7s}", c["frames"], "frames,", len(c["gt"]), "vehicles,", c["gt_count"], "cross the line")

# %%
clip = clips[2]
U.show([U.clip_frame(clip, t, clip["detections"][t]) for t in range(40, 80, 5)], titles=[f"frame {t}" for t in range(40, 80, 5)], size=2.0)

# %% [markdown]
# Four kinds of clip: **day**, **queue** (traffic slows to a crawl at the line), **gantry** (a
# sign gantry hides part of the road) and **night** (more misses, and headlight reflections
# that slide down the wet road). Red boxes are detections, the yellow line is the counting line.
#
# ## 1 · Associate detections to tracks by overlap
#
# Each frame, every open track holds its last box. A detection continues the track it
# overlaps most. Greedy association: take the pair with the highest IoU, remove both from the
# pool, repeat while the best IoU is at least the threshold.
#
# ✏️ **Exercise** · Return the list of `(track_index, detection_index)` pairs.
# `U.box_iou(track_boxes, det_boxes)` gives the IoU matrix.

# %%
def associate(track_boxes, det_boxes, iou_thr=0.3):
    # BEGIN SOLUTION
    if len(track_boxes) == 0 or len(det_boxes) == 0:
        return []
    iou = U.box_iou(track_boxes, det_boxes)
    pairs = []
    while True:
        t, d = np.unravel_index(iou.argmax(), iou.shape)
        if iou[t, d] < iou_thr:
            return pairs
        pairs.append((int(t), int(d)))
        iou[t, :] = -1
        iou[:, d] = -1
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _associate(track_boxes, det_boxes, iou_thr=0.3):
    if len(track_boxes) == 0 or len(det_boxes) == 0:
        return []
    iou, pairs = U.box_iou(track_boxes, det_boxes), []
    while iou.max() >= iou_thr:
        t, d = np.unravel_index(iou.argmax(), iou.shape)
        pairs.append((int(t), int(d)))
        iou[t, :], iou[:, d] = -1, -1
    return pairs

try:
    got = associate([[0, 0, 10, 10], [20, 0, 30, 10]], [[21, 0, 31, 10], [1, 0, 11, 10], [50, 50, 60, 60]])
except NotImplementedError:
    associate = _associate
    got = associate([[0, 0, 10, 10], [20, 0, 30, 10]], [[21, 0, 31, 10], [1, 0, 11, 10], [50, 50, 60, 60]])
assert sorted(got) == [(0, 1), (1, 0)], got


def run_tracker(detections, iou_thr=0.3, max_age=0, min_hits=1, predict=False):
    """Tracking by detection. detections[t] is a list of x0, y0, x1, y1, score.
    max_age: frames a track may go unmatched before it is closed (its patience).
    min_hits: detections a track needs before it is reported.
    Returns the reported tracks: {"id": int, "boxes": {frame: box}}."""
    open_tracks, done, next_id = [], [], 0
    for t, dets in enumerate(detections):
        boxes = [d[:4] for d in dets]
        guess = [tr["guess"] if predict else tr["last"] for tr in open_tracks]
        pairs = associate(guess, boxes, iou_thr)
        matched_t, matched_d = {a for a, _ in pairs}, {b for _, b in pairs}
        for a, b in pairs:
            tr = open_tracks[a]
            tr["velocity"] = [(n - o) / (tr["missed"] + 1) for n, o in zip(boxes[b], tr["last"])]
            tr["boxes"][t], tr["last"], tr["missed"] = boxes[b], boxes[b], 0
        for a, tr in enumerate(open_tracks):
            if a not in matched_t:
                tr["missed"] += 1
            tr["guess"] = [v + w * (tr["missed"] + 1) for v, w in zip(tr["last"], tr["velocity"])]
        done += [tr for tr in open_tracks if tr["missed"] > max_age]
        open_tracks = [tr for tr in open_tracks if tr["missed"] <= max_age]
        for b, box in enumerate(boxes):
            if b not in matched_d:
                open_tracks.append({"id": next_id, "boxes": {t: box}, "last": box, "guess": box,
                                    "velocity": [0, 0, 0, 0], "missed": 0})
                next_id += 1
    return [{"id": tr["id"], "boxes": tr["boxes"]} for tr in done + open_tracks if len(tr["boxes"]) >= min_hits]


tracks = run_tracker(clips[0]["detections"])
print(clips[0]["id"], "-", len(tracks), "tracks for", len(clips[0]["gt"]), "vehicles")

# %% [markdown]
# ## 2 · Count each track once when it crosses the line
#
# A first idea: every time a box centre moves from above the line to below it, add one. In a
# queue a vehicle creeps over the line while its box jitters by a pixel or two, and the centre
# crosses several times.
#
# ✏️ **Exercise** · Count the tracks that cross: a track counts **once**, at the first pair of
# consecutive boxes whose centre goes from above `line_y` (smaller y) to on or below it.

# %%
def crossings_naive(tracks, line_y):
    """Every downward crossing of every track: counts a jittering vehicle more than once."""
    n = 0
    for tr in tracks:
        cy = [(b[1] + b[3]) / 2 for _, b in sorted(tr["boxes"].items())]
        n += sum(a < line_y <= b for a, b in zip(cy, cy[1:]))
    return n


def count_crossings(tracks, line_y):
    # BEGIN SOLUTION
    n = 0
    for tr in tracks:
        cy = [(b[1] + b[3]) / 2 for _, b in sorted(tr["boxes"].items())]
        n += any(a < line_y <= b for a, b in zip(cy, cy[1:]))
    return n
    # END SOLUTION

# %%
def _count_crossings(tracks, line_y):
    total = 0
    for tr in tracks:
        cy = [(b[1] + b[3]) / 2 for _, b in sorted(tr["boxes"].items())]
        total += any(a < line_y <= b for a, b in zip(cy, cy[1:]))
    return total

try:
    count_crossings(tracks, 84)
except NotImplementedError:
    count_crossings = _count_crossings
queue = clips[1]
q_tracks = run_tracker(queue["detections"])
print(f"{queue['id']} (queue): truth {queue['gt_count']}, every crossing {crossings_naive(q_tracks, queue['line_y'])}, "
      f"once per track {count_crossings(q_tracks, queue['line_y'])}")

# %% [markdown]
# ## 3 · Count error per clip
#
# ✏️ **Exercise** · Run the tracker on one clip with the given settings and return
# `(count, error)`, where the error is the count minus the clip's `gt_count` (signed: a
# positive error is an overcount).

# %%
def count_clip(clip, **settings):
    # BEGIN SOLUTION
    count = count_crossings(run_tracker(clip["detections"], **settings), clip["line_y"])
    return count, count - clip["gt_count"]
    # END SOLUTION

# %%
try:
    count_clip(clips[0])
except NotImplementedError:
    def count_clip(clip, **settings):
        n = count_crossings(run_tracker(clip["detections"], **settings), clip["line_y"])
        return n, n - clip["gt_count"]

# %% [markdown]
# ## 4 · Identity switches against the ground-truth tracks
#
# A count can be right while the tracker loses and re-finds every vehicle. For each true
# vehicle, `matched_ids` lists which reported track covered it in each frame. An identity
# switch is a change of that id from one covered frame to the next.
#
# ✏️ **Exercise** · Return the number of identity switches in a clip: over all ground-truth
# vehicles, the number of times the matched id changes along its frames.

# %%
def matched_ids(gt_track, tracks, iou_thr=0.5):
    """For one true vehicle: the id of the reported track that overlaps it in each frame where one does."""
    ids = []
    for t, box in sorted((int(k), v) for k, v in gt_track["boxes"].items()):
        cands = [(float(U.box_iou([box], [tr["boxes"][t]])[0, 0]), tr["id"]) for tr in tracks if t in tr["boxes"]]
        if cands and max(cands)[0] >= iou_thr:
            ids.append(max(cands)[1])
    return ids


def identity_switches(clip, tracks):
    # BEGIN SOLUTION
    switches = 0
    for g in clip["gt"]:
        ids = matched_ids(g, tracks)
        switches += sum(a != b for a, b in zip(ids, ids[1:]))
    return switches
    # END SOLUTION

# %%
def _identity_switches(clip, tracks):
    total = 0
    for g in clip["gt"]:
        ids = matched_ids(g, tracks)
        total += sum(a != b for a, b in zip(ids, ids[1:]))
    return total

try:
    identity_switches(clips[0], tracks)
except NotImplementedError:
    identity_switches = _identity_switches


def evaluate(**settings):
    """Per kind of clip: the sum of absolute count errors and of identity switches."""
    out = {}
    for c in clips:
        tr = run_tracker(c["detections"], **settings)
        row = out.setdefault(c["kind"], {"abs_count_error": 0, "identity_switches": 0, "truth": 0})
        row["abs_count_error"] += abs(count_crossings(tr, c["line_y"]) - c["gt_count"])
        row["identity_switches"] += identity_switches(c, tr)
        row["truth"] += c["gt_count"]
    return out


baseline = evaluate(max_age=0, min_hits=1)
print("baseline: patience 0, minimum hits 1")
for kind, row in baseline.items():
    print(f"  {kind:7s} vehicles {row['truth']:3d}   count error {row['abs_count_error']:3d}   identity switches {row['identity_switches']:3d}")
U.save_reference("d3_4_baseline", baseline)

# %% [markdown]
# **Look at** the day row first. With no patience, every missed detection ends a track and
# the next detection starts a new one: far more switches than vehicles, in good light. Then
# compare count error with switches across the rows: the two numbers do not move together. A
# switch costs a count only when it happens at the line.

# %% [markdown]
# ## 5 · Tune patience and minimum hits
#
# Two settings: `max_age`, how many frames a track survives without a detection, and
# `min_hits`, how many detections a track needs before it is believed.
#
# ✏️ **Exercise** · Try `max_age` in 0, 2, 5 and `min_hits` in 1, 4. Return the pair with the
# smallest total count error over all clips; break ties by fewer identity switches.

# %%
def tune():
    # BEGIN SOLUTION
    best = None
    for max_age in (0, 2, 5):
        for min_hits in (1, 4):
            res = evaluate(max_age=max_age, min_hits=min_hits)
            key = (sum(r["abs_count_error"] for r in res.values()), sum(r["identity_switches"] for r in res.values()))
            if best is None or key < best[0]:
                best = (key, (max_age, min_hits))
    return best[1]
    # END SOLUTION

# %%
try:
    max_age, min_hits = tune()
except NotImplementedError:
    max_age, min_hits = U.reference("d3_4_best")
tuned = evaluate(max_age=max_age, min_hits=min_hits)
print(f"chosen: patience {max_age}, minimum hits {min_hits}")
for kind, row in tuned.items():
    print(f"  {kind:7s} vehicles {row['truth']:3d}   count error {row['abs_count_error']:3d}   identity switches {row['identity_switches']:3d}")
assert sum(r["abs_count_error"] for r in tuned.values()) < sum(r["abs_count_error"] for r in baseline.values())
U.save_reference("d3_4_best", [max_age, min_hits])
U.save_reference("d3_4_tuned", tuned)

# %% [markdown]
# **Look at** what tuning could not fix: the identity switches under the gantry. A vehicle
# hidden for several frames comes out tens of pixels further on, where its last box overlaps
# nothing. Patience alone keeps the track open and still cannot re-attach it. The gantry
# count is right all the same: the line lies beyond the gantry, and each re-found vehicle
# crosses it once.

# %% [markdown]
# ## ✱ Stretch · Constant velocity
#
# `run_tracker(..., predict=True)` matches each detection against where the track *should be
# by now*: its last box moved by its last velocity for every missed frame. Evaluate
# `max_age=20, min_hits=4, predict=True`. What happens to the gantry row? This prediction step
# is the heart of the SORT family of trackers, which use a Kalman filter for it.

# %% [markdown]
# ## Check yourself
#
# 1. A clip has 14 true crossings. The tracker reports 14 with 11 identity switches. Is the
#    counter ready for the traffic team?
# 2. Why does requiring four hits remove the headlight reflections and almost no vehicles?
# 3. The gantry moves so that it covers the counting line. What breaks, and what are two fixes?
#
# <details><summary>Answers</summary>
#
# 1. For counting on this camera, the evidence says yes. For anything that follows a vehicle
#    (speed between two lines, turning movements, dwell time) it is not: those need identities.
# 2. A reflection lives two or three frames; a vehicle is in view for twenty or more.
# 3. Tracks end before the line and new ones start after it, so neither crosses: counts drop.
#    Move the line to where the camera sees clearly, or predict tracks through the gap.
# </details>
#
# ## Carry forward
#
# Count error per clip and the number of identity switches are the evidence for outcome 4. A
# counting capstone reports both, by kind of clip, and says where the line is and why.
