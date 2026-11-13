# %% [markdown]
# # Lab 1.4 · Augment without changing labels
#
# Day 1 · Session 4 (Augmentation and what it assumes) · 25 min.
#
# An augmentation is a claim: "this change does not alter what the picture shows". You build a
# pipeline that moves an image and its mask together, measure for six transforms how often
# the claim is false on Al-Manar's defect crops, and show what one false claim does to a model.
#
# **Tasks**
# - [ ] Build the pipeline for images and masks · 7 min
# - [ ] Label-change rate of six transforms · 8 min
# - [ ] Keep the safe transforms, remove the two that change meaning · 4 min
# - [ ] Train a probe with and without the unsafe transform · 6 min
# - [ ] ✱ Stretch · Add a night transform and measure the night slice
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the pipeline.

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
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
import roadkit as U

U.ensure_data()

# Defect crops in daylight with their own masks (alligator cracking is left out: lab 1.2)
rows = [t for t in U.tiles("train") if t["label"] != "clear" and t["cls"] != "alligator"
        and t["condition"] == "day" and not t["blur"]][: U.pick(600, 240)]
X, M = U.tile_images(rows), U.tile_masks(rows)
names = [t["label"] for t in rows]
print(len(rows), "crops:", {c: names.count(c) for c in U.TILE_CLASSES[1:]})

# %% [markdown]
# ## 1 · Build the pipeline for images and masks
#
# A flip that moves the pixels and leaves the mask behind produces a wrong label silently.
# `torchvision.transforms.v2` moves everything together when each input says what it is:
# `tv_tensors.Image` for the picture, `tv_tensors.Mask` for the mask (and
# `tv_tensors.BoundingBoxes` for boxes).
#
# ✏️ **Exercise** · Apply one transform to a crop (uint8, 96 x 96 x 3) and its mask (bool,
# 96 x 96) together, and return both as NumPy arrays of the same kinds.

# %%
def apply(transform, img, mask):
    # BEGIN SOLUTION
    image = tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1))
    target = tv_tensors.Mask(torch.from_numpy(mask))
    image, target = transform(image, target)
    return image.permute(1, 2, 0).numpy(), target.numpy().astype(bool)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _apply(transform, img, mask):
    image, target = transform(tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)), tv_tensors.Mask(torch.from_numpy(mask)))
    return image.permute(1, 2, 0).numpy(), target.numpy().astype(bool)

try:
    apply(v2.RandomHorizontalFlip(p=1.0), X[0], M[0])
except NotImplementedError:
    apply = _apply
img2, mask2 = apply(v2.RandomHorizontalFlip(p=1.0), X[0], M[0])
assert img2.shape == X[0].shape and (mask2 == M[0][:, ::-1]).all(), "the mask must flip with the image"
print("image and mask move together")

# %% [markdown]
# ## 2 · Label-change rate of six transforms
#
# What would a careful annotator call the crop *after* the transform? `meaning` answers from
# the mask: a crop that lost most of its defect is `clear`; a crack is longitudinal when it
# runs along the direction of travel (up the image) and transverse when it runs across.

# %%
def meaning(mask, label, pixels_before):
    """The label a careful annotator would give this crop now."""
    if mask.sum() < 0.4 * pixels_before:
        return "clear"
    if label.startswith("crack"):
        ys, xs = np.nonzero(mask)
        return "crack_long" if ys.std() >= xs.std() else "crack_trans"
    return label


CANDIDATES = {
    "mirror left-right": v2.RandomHorizontalFlip(p=1.0),
    "mirror up-down": v2.RandomVerticalFlip(p=1.0),
    "quarter turn": v2.RandomRotation((90, 90)),
    "brightness and contrast": v2.ColorJitter(brightness=0.4, contrast=0.4),
    "tight random crop": v2.RandomResizedCrop(96, scale=(0.08, 0.3), antialias=True),
    "gaussian blur": v2.GaussianBlur(5, sigma=(0.5, 1.5)),
}

# %% [markdown]
# ✏️ **Exercise** · Return the share of crops whose meaning after the transform differs from
# their label. Use `apply` and `meaning`; `M[k].sum()` is the defect's pixel count before.

# %%
def change_rate(transform):
    # BEGIN SOLUTION
    changed = 0
    for k in range(len(rows)):
        _, mask = apply(transform, X[k], M[k])
        changed += meaning(mask, names[k], M[k].sum()) != names[k]
    return changed / len(rows)
    # END SOLUTION

# %%
torch.manual_seed(0)
try:
    rates = {name: change_rate(t) for name, t in CANDIDATES.items()}
except NotImplementedError:
    rates = U.reference("d1_4_rates")
baseline = float(np.mean([meaning(M[k], names[k], M[k].sum()) != names[k] for k in range(len(rows))]))
print(f"{'no transform':26s} {baseline:.2f}")
for name, rate in rates.items():
    print(f"{name:26s} {rate:.2f}  {'#' * int(rate * 40)}")
U.save_reference("d1_4_rates", rates)

# %% [markdown]
# **Look at** the two transforms that stand out, and at which classes they hit. Print the rate
# per class for the quarter turn if you have a minute: potholes and patches do not care.

# %% [markdown]
# ## 3 · Keep the safe transforms, remove the two that change meaning
#
# ✏️ **Exercise** · Return the names of the transforms you keep: those that change the meaning
# of fewer than 5% of crops.

# %%
def safe_transforms(rates):
    # BEGIN SOLUTION
    return [name for name, rate in rates.items() if rate < 0.05]
    # END SOLUTION

# %%
try:
    safe = safe_transforms(rates)
except NotImplementedError:
    safe = U.reference("d1_4_safe")
unsafe = [n for n in CANDIDATES if n not in safe]
assert "quarter turn" in unsafe and "mirror left-right" in safe
print("keep:  ", safe)
print("remove:", unsafe)
U.save_reference("d1_4_safe", safe)

pipeline = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
                       v2.ColorJitter(brightness=0.4, contrast=0.4), v2.GaussianBlur(5, sigma=(0.5, 1.5))])
torch.manual_seed(1)
U.show([apply(pipeline, X[0], M[0])[0] for _ in range(8)], titles=[names[0]] * 8)

# %% [markdown]
# ## 4 · Train a probe with and without the unsafe transform
#
# Does one wrong claim matter? Train a small classifier on top of frozen features twice: once
# on the crops plus safely augmented copies, once on the crops plus copies given a quarter
# turn, the labels left as they were. Then score both on validation crops, crack classes only.

# %%
from sklearn.linear_model import LogisticRegression

encoder = U.load_ckpt(U.TileNet(), "pilot_classifier")
y = np.array([U.TILE_CLASSES.index(n) for n in names])
val = [t for t in U.tiles("val") if t["label"].startswith("crack") and t["cls"] != "alligator"
       and t["condition"] == "day" and not t["blur"]]
Fv, yv = U.predict(encoder, U.tile_images(val), embed=True), U.labels(val)


def probe_accuracy(transform, seed=0):
    torch.manual_seed(seed)
    copies = np.stack([apply(transform, X[k], M[k])[0] for k in range(len(rows))])
    F = U.predict(encoder, np.concatenate([X, copies]), embed=True)
    clf = LogisticRegression(max_iter=3000, C=10).fit(F, np.concatenate([y, y]))
    return float(clf.score(Fv, yv))


acc_safe = probe_accuracy(pipeline)
acc_unsafe = probe_accuracy(v2.RandomRotation((90, 90)))
print(f"crack accuracy on {len(val)} validation crops")
print(f"  trained with the safe pipeline:      {acc_safe:.3f}")
print(f"  trained with quarter-turned copies:  {acc_unsafe:.3f}")
assert acc_safe > acc_unsafe
U.save_reference("d1_4_probe", [acc_safe, acc_unsafe])

# %% [markdown]
# **Look at** the gap between the two numbers. Every quarter-turned copy of a crack shows the
# other crack class under the old label: the probe is being taught that direction does not
# matter, on a task where direction is the label. Nothing failed loudly on the way.

# %% [markdown]
# ## ✱ Stretch · A night transform
#
# Write a transform that imitates night (multiply by 0.2, add noise, a touch of blue) as a
# `v2.Lambda` on the image only. Add it to the pipeline with probability 0.3, retrain the
# probe and score it on night validation crops. Does imitated night help on rendered night?

# %% [markdown]
# ## Check yourself
#
# 1. For which of Al-Manar's classes is a quarter turn a safe augmentation?
# 2. The tight crop removed the defect from many crops. Name two repairs that keep cropping.
# 3. A mirror up-down puts the far end of the road at the bottom. Why does it count as safe here?
#
# <details><summary>Answers</summary>
#
# 1. Pothole and patch: nothing in their definition mentions direction. For the two crack
#    classes the direction is the definition.
# 2. Crop less tightly (a larger scale range), or recompute the label from the mask after the
#    crop and call a crop with too little defect `clear`. Detection pipelines do the second
#    with boxes: they drop a box that keeps too little of its area.
# 3. These crops look straight down at the surface: there is no sky or horizon to turn upside
#    down, and a longitudinal crack stays longitudinal. On a forward-facing street camera the
#    same flip would produce pictures no camera takes.
# </details>
#
# ## Carry forward
#
# The safe pipeline is the one Day 2 and Day 3 train with. The method stays with you: before a
# transform enters a pipeline, say which label it could change and measure how often it does.
