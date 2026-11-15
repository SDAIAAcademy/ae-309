# %% [markdown]
# # Lab 1.3 · Split by road segment
#
# Day 1 · Session 3 (Leakage by site and near-duplicates) · 25 min.
#
# The survey vehicle films every road spot several times, so the pack is full of frames that
# are almost the same picture. The split you inherited (`split_random`) was drawn frame by
# frame. You find the near-duplicates, show that they sit on both sides of that split, repair
# the missing segment ids, split again by road segment and write the split report.
#
# **Tasks**
# - [ ] Difference hash and near-duplicate pairs · 7 min
# - [ ] Count the pairs that straddle the inherited split · 4 min
# - [ ] Recover the missing segment ids from the duplicates · 5 min
# - [ ] Split by segment and write the split report · 5 min
# - [ ] Score one model on both test sets · 4 min
# - [ ] ✱ Stretch · Find the duplicates with embeddings instead of hashes
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the split.

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
from PIL import Image
import roadkit as U

U.ensure_data()
frames = U.frames()
images = U.frame_images(frames)
print(len(frames), "frames;", sum(f["segment"] is None for f in frames), "have no segment id")
print({s: sum(f["split_random"] == s for f in frames) for s in ("train", "val", "test")}, "(inherited split)")

# %% [markdown]
# ## 1 · Difference hash and near-duplicate pairs
#
# A difference hash shrinks a grey image to 17 x 16 pixels and writes one bit per pixel: is it
# brighter than its left neighbour? Two frames of the same spot give nearly the same 256 bits,
# whatever the exposure. The number of differing bits is the Hamming distance.
#
# ✏️ **Exercise** · Return the 256 bits of one image as a flat boolean array.
# `Image.fromarray(img).convert("L").resize((17, 16), Image.BILINEAR)` does the shrinking.

# %%
def dhash(img):
    # BEGIN SOLUTION
    g = np.asarray(Image.fromarray(img).convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).ravel()
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _dhash(img):
    g = np.asarray(Image.fromarray(img).convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).ravel()

try:
    H = np.stack([dhash(im) for im in images]).astype(np.float32)
except NotImplementedError:
    H = np.stack([_dhash(im) for im in images]).astype(np.float32)
assert H.shape == (len(frames), 256)

# Hamming distance of every pair, as two matrix products
D = H @ (1 - H).T + (1 - H) @ H.T
THRESHOLD = 30  # at most 30 of 256 bits differ
i, j = np.triu_indices(len(frames), 1)
near = D[i, j] <= THRESHOLD
pairs = list(zip(i[near].tolist(), j[near].tolist()))
print(len(pairs), "near-duplicate pairs at a distance of", THRESHOLD, "bits or less")

# %%
show = pairs[:6]
U.show([images[k] for p in show for k in p], titles=[frames[k]["id"] for p in show for k in p], cols=6, size=2.0)

# %% [markdown]
# **Look at** the pairs, side by side. Same cracks, same lane marking, a shift of a pixel or
# two. Which of each pair would you allow into the test set if the other is in training?

# %% [markdown]
# ## 2 · Count the pairs that straddle the inherited split
#
# ✏️ **Exercise** · Count the near-duplicate pairs whose two frames have a different value of
# the given split field.

# %%
def straddling(pairs, frames, key):
    # BEGIN SOLUTION
    return sum(frames[a][key] != frames[b][key] for a, b in pairs)
    # END SOLUTION

# %%
try:
    leaks_random = straddling(pairs, frames, "split_random")
except NotImplementedError:
    leaks_random = U.reference("d1_3_leaks")[0]
print(f"{leaks_random} of {len(pairs)} near-duplicate pairs sit on two sides of the inherited split")

# %% [markdown]
# ## 3 · Recover the missing segment ids from the duplicates
#
# A split by segment needs a segment for every frame, and some frames lost theirs in an export.
# A frame with no segment takes the segment of a near-duplicate that has one.
#
# ✏️ **Exercise** · Return a list with one segment per frame: the frame's own, or one
# recovered through a pair, or `None` when no duplicate knows it.

# %%
def recover_segments(frames, pairs):
    # BEGIN SOLUTION
    seg = [f["segment"] for f in frames]
    for a, b in pairs:
        if seg[a] is None and seg[b] is not None:
            seg[a] = seg[b]
        if seg[b] is None and seg[a] is not None:
            seg[b] = seg[a]
    return seg
    # END SOLUTION

# %%
try:
    seg = recover_segments(frames, pairs)
except NotImplementedError:
    seg = [f["segment"] for f in frames]
    for _ in range(2):
        for a, b in pairs:
            seg[a], seg[b] = seg[a] or seg[b], seg[b] or seg[a]
missing_before = sum(f["segment"] is None for f in frames)
missing_after = sum(s is None for s in seg)
assert missing_after < missing_before
print(f"frames with no segment: {missing_before} before, {missing_after} after")

# %% [markdown]
# **Look at** what is left. Hashes found the easy duplicates and repaired a few ids, no more.
# A frame whose segment is still unknown cannot be placed safely: in training it may leak a
# test segment, in the test set it may repeat a training one. It leaves both.

# %% [markdown]
# ## 4 · Split by segment and write the split report
#
# ✏️ **Exercise** · Give every frame a split: `"drop"` when its segment is unknown, otherwise
# the split of its segment from `segment_split` (a function that maps a segment id to
# `"train"`, `"val"` or `"test"`).

# %%
def segment_split(segment_id, seed=309):
    """A fixed, pseudo-random split of segments: about 70% train, 15% val, 15% test."""
    u = np.random.default_rng([seed, int(segment_id.split("-")[1])]).random()
    return "train" if u < 0.70 else "val" if u < 0.85 else "test"


def split_by_segment(seg):
    # BEGIN SOLUTION
    return ["drop" if s is None else segment_split(s) for s in seg]
    # END SOLUTION

# %%
try:
    split = split_by_segment(seg)
except NotImplementedError:
    split = ["drop" if s is None else segment_split(s) for s in seg]


def split_report(frames, seg, split, pairs):
    """Counts per split, segments shared between splits and duplicate pairs across splits."""
    rep = {"frames": {s: split.count(s) for s in ("train", "val", "test", "drop")}}
    segs = {s: {g for g, q in zip(seg, split) if q == s} for s in ("train", "val", "test")}
    rep["segments"] = {s: len(v) for s, v in segs.items()}
    rep["segments_shared"] = len((segs["train"] & segs["val"]) | (segs["train"] & segs["test"]) | (segs["val"] & segs["test"]))
    rep["duplicate_pairs_across_splits"] = sum(
        split[a] != split[b] and "drop" not in (split[a], split[b]) for a, b in pairs)
    return rep


report = split_report(frames, seg, split, pairs)
for k, v in report.items():
    print(f"{k:32s} {v}")
assert report["segments_shared"] == 0, "no road segment may appear in two splits"
U.save_reference("d1_3_leaks", [leaks_random, report["duplicate_pairs_across_splits"]])
U.save_reference("d1_3_report", report)

# %% [markdown]
# **Look at** `duplicate_pairs_across_splits`. If it is not zero, the hash matched two frames
# of different segments: open one such pair and judge whether the hash or the segment id is
# wrong.

# %% [markdown]
# ## 5 · Score one model on both test sets
#
# How much did the leak flatter a model? Take a model that can only memorise: one nearest
# neighbour on 16 x 16 grey pixels of each defect crop. It answers with the label of the most
# similar training crop.

# %%
tiles = U.tiles()
Xt = U.tile_images(tiles)
small = np.stack([np.asarray(Image.fromarray(x).convert("L").resize((16, 16)), dtype=np.float32).ravel() for x in Xt])
small = (small - small.mean(1, keepdims=True)) / (small.std(1, keepdims=True) + 1e-6)
yt = U.labels(tiles)
frame_split = {f["id"]: s for f, s in zip(frames, split)}


def one_nn_accuracy(train, test):
    sim = small[test] @ small[train].T
    return float((yt[train][sim.argmax(1)] == yt[test]).mean())


inherited = np.array([t["split_random"] for t in tiles])
by_segment = np.array([frame_split[t["frame"]] for t in tiles])
acc_random = one_nn_accuracy(inherited == "train", inherited == "test")
acc_segment = one_nn_accuracy(by_segment == "train", by_segment == "test")
print(f"1-nearest-neighbour accuracy, inherited split:  {acc_random:.3f}")
print(f"1-nearest-neighbour accuracy, split by segment: {acc_segment:.3f}")
print(f"the leak was worth {100 * (acc_random - acc_segment):.0f} points to a model that learnt nothing general")
assert acc_random > acc_segment
U.save_reference("d1_3_inflation", [acc_random, acc_segment])

# %% [markdown]
# ## ✱ Stretch · Embeddings instead of hashes
#
# `U.predict(U.load_ckpt(U.TileNet(), "pilot_classifier"), images, embed=True)` gives a
# 128-number embedding per frame. Normalise the rows, take cosine similarity, and count pairs
# above 0.995. Compare with the hash pairs using the true segments. Does a classifier's
# embedding make a good duplicate detector? Why might it not?

# %% [markdown]
# ## Check yourself
#
# 1. The hash found only a fraction of the same-spot pairs. Why is the split by segment still
#    safe?
# 2. Why are the frames with an unknown segment dropped and not put into training?
# 3. A colleague removes one frame of every near-duplicate pair and keeps the random split. Is
#    the leak gone?
#
# <details><summary>Answers</summary>
#
# 1. The split never relies on the hash. Whole segments go to one side, so every frame of a
#    spot goes with it, found or not. The hash is the evidence of the problem, the group key
#    is the fix.
# 2. One of them may belong to a test segment. In training it would teach the model that
#    segment's surface, and the test would no longer be unseen road.
# 3. No. Frames of one spot under another light, or of the next spot of the same segment,
#    still share surface, markings and defects without being near-duplicates.
# </details>
#
# ## Carry forward
#
# From here every lab uses the supplied split by segment (`split` in each frame: 70% train,
# 15% validation, 15% test, no segment shared). The split report goes into your dataset card.
