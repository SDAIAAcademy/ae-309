# %% [markdown]
# # Lab 2.1 · Two backbones by slice
#
# Day 2 · Session 1 (Convolutional networks and vision transformers) · 25 min.
#
# You fine-tune two pretrained backbones on Al-Manar's defect crops, a ResNet-18 (a
# convolutional network) and a ViT-Tiny (a vision transformer), on the same short schedule,
# and compare them the only way that counts in the field: by lighting slice, with an interval
# on every number.
#
# **Tasks**
# - [ ] Put a five-class head on each backbone · 4 min
# - [ ] Fine-tune each on a short schedule · 9 min
# - [ ] Accuracy by lighting slice · 5 min
# - [ ] Bootstrap intervals and the comparison table · 7 min
# - [ ] ✱ Stretch · Bootstrap the difference between the two models on paired crops
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the
# training or the comparison. On Colab choose a GPU runtime. On a laptop set `CV_QUICK=1`.

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

import time
import numpy as np
import timm
import torch.nn as nn
import torchvision
import roadkit as U

U.ensure_data()
rng = np.random.default_rng(21)
train = U.tiles("train")
train = [train[i] for i in rng.permutation(len(train))[: U.pick(3000, 320)]]
test = U.tiles("test")[: U.pick(1500, 320)]
Xtr, ytr = U.tile_images(train), U.labels(train)
Xte, yte = U.tile_images(test), U.labels(test)
print(f"mode {U.MODE}, device {U.DEV}: {len(train)} training crops, {len(test)} test crops")

# %% [markdown]
# ## 1 · Put a five-class head on each backbone
#
# Both backbones were trained on ImageNet and end in a 1,000-class layer. Keep everything
# before it, replace the last layer with one for our five classes.
#
# ✏️ **Exercise** · In the ResNet the last layer is `m.fc`. In timm, `create_model(...,
# num_classes=5)` builds the new head for you; pass `img_size=128` so the transformer's
# position embeddings are resampled for 128-pixel inputs.

# %%
def resnet18_with_head(n_classes=5):
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # BEGIN SOLUTION
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    # END SOLUTION
    return nn.Sequential(U.Norm(), m)


def vit_tiny_with_head(n_classes=5):
    # BEGIN SOLUTION
    m = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True,
                          num_classes=n_classes, img_size=128)
    # END SOLUTION
    return nn.Sequential(U.Norm(), m)

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    models = {"resnet18": resnet18_with_head(), "vit_tiny": vit_tiny_with_head()}
except NotImplementedError:
    models = {name: U.backbone(name) for name in ("resnet18", "vit_tiny")}
for name, m in models.items():
    out = U.predict(m, Xte[:2], size=128)
    assert out.shape == (2, 5), "the head must give five numbers per crop"
    print(f"{name:9s} {sum(p.numel() for p in m.parameters()) / 1e6:5.1f} million parameters")

# %% [markdown]
# ## 2 · Fine-tune each on a short schedule
#
# The same crops, the same number of passes, the same 128-pixel input for both. The learning
# rate differs: transformers are fine-tuned with a smaller one. `U.fit` is the plain loop
# you have written before (AdamW, cross-entropy, one cycle).

# %%
LR = {"resnet18": 3e-4, "vit_tiny": 1e-4}
minutes = {}
for name, m in models.items():
    print(name)
    t = time.time()
    U.fit(m, Xtr, ytr, epochs=U.pick(3, 1), lr=LR[name], bs=32, size=128)
    minutes[name] = (time.time() - t) / 60

correct = {name: U.predict(m, Xte, size=128).argmax(1) == yte for name, m in models.items()}
for name in models:
    print(f"{name:9s} test accuracy {correct[name].mean():.3f}   training took {minutes[name]:.1f} min")

# %% [markdown]
# ## 3 · Accuracy by lighting slice
#
# ✏️ **Exercise** · For one model, return `{condition: array of 0/1 per crop}` over the test
# crops (`t["condition"]` is day, glare, dust, rain or night). Keep the per-crop values: the
# intervals of task 4 need them.

# %%
def by_condition(correct_one):
    # BEGIN SOLUTION
    out = {}
    for c in U.CONDITIONS:
        out[c] = np.array([ok for t, ok in zip(test, correct_one) if t["condition"] == c], dtype=float)
    return out
    # END SOLUTION

# %%
def _by_condition(correct_one):
    return {c: np.array([ok for t, ok in zip(test, correct_one) if t["condition"] == c], dtype=float) for c in U.CONDITIONS}

try:
    slices = {name: by_condition(correct[name]) for name in models}
except NotImplementedError:
    slices = {name: _by_condition(correct[name]) for name in models}
assert sum(len(v) for v in slices["resnet18"].values()) == len(test)
for c in U.CONDITIONS:
    print(f"{c:6s} n={len(slices['resnet18'][c]):4d}  " + "  ".join(f"{n} {slices[n][c].mean():.3f}" for n in models))

# %% [markdown]
# ## 4 · Bootstrap intervals and the comparison table
#
# "The transformer wins at night by four points" means little on forty crops. The bootstrap
# answers "how much would this number move with another sample of the same size?": draw the
# crops again with replacement, many times, and read the spread of the mean.
#
# ✏️ **Exercise** · Return `(mean, low, high)`: the mean of `values` and the 2.5th and 97.5th
# percentiles of the means of `n_boot` resamples. Use `np.random.default_rng(seed)`.

# %%
def bootstrap_ci(values, n_boot=1000, seed=0):
    # BEGIN SOLUTION
    values = np.asarray(values, dtype=float)
    r = np.random.default_rng(seed)
    means = values[r.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    # END SOLUTION

# %%
try:
    bootstrap_ci([1, 0, 1, 1])
except NotImplementedError:
    bootstrap_ci = U.boot_ci
m_, lo_, hi_ = bootstrap_ci(np.r_[np.ones(80), np.zeros(20)])
assert abs(m_ - 0.8) < 1e-9 and 0.70 < lo_ < 0.76 and 0.85 < hi_ < 0.90, "check the percentiles"

table = {}
print(f"{'slice':8s} {'n':>4s}   {'resnet18':24s} {'vit_tiny':24s} intervals overlap?")
for c in ["all"] + U.CONDITIONS:
    row = {}
    for name in models:
        v = correct[name].astype(float) if c == "all" else slices[name][c]
        row[name] = bootstrap_ci(v)
    a, b = row["resnet18"], row["vit_tiny"]
    overlap = a[1] <= b[2] and b[1] <= a[2]
    table[c] = {"n": int(len(v)), **{k: [round(x, 3) for x in r] for k, r in row.items()}, "overlap": bool(overlap)}
    fmt = lambda r: f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]"
    print(f"{c:8s} {len(v):4d}   {fmt(a):24s} {fmt(b):24s} {'yes' if overlap else 'NO'}")
U.save_reference("d2_1_table", table)

# %% [markdown]
# **Look at** the widths. The interval on a slice of a few dozen crops is several times wider
# than the one on all crops. Where the intervals overlap you have no evidence that one
# backbone is better there; where they do not, you have a finding to write down.

# %% [markdown]
# ## ✱ Stretch · The paired difference
#
# Both models saw the same test crops, so resample the *difference* per crop
# (`correct["vit_tiny"].astype(float) - correct["resnet18"]`) and put an interval on it.
# A paired interval is narrower than comparing two separate intervals: why?

# %% [markdown]
# ## Check yourself
#
# 1. Both models got the same crops, passes and input size. Name one thing that still differs
#    and could explain a gap.
# 2. On the night slice the intervals overlap. Can you say the two models are equally good at night?
# 3. Why is the comparison table sliced by lighting and not by class?
#
# <details><summary>Answers</summary>
#
# 1. The learning rate (and with it how far each has converged on a short schedule), and the
#    pretraining data: this ViT saw ImageNet-21k before ImageNet-1k, the ResNet only the latter.
# 2. No. You can say this test cannot tell them apart at night. To decide, collect more night
#    crops or use the paired difference.
# 3. Because the release question is "does it hold in the conditions the roadside meets".
#    Slicing by class answers a different, also useful question: which defect is hard.
# </details>
#
# ## Carry forward
#
# `bootstrap_ci` is in `roadkit` as `U.boot_ci` from here on. The comparison table by slice
# with intervals is the evidence the course asks for whenever two models are compared.
