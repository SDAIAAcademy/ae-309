# %% [markdown]
# # Lab 2.2 · Probe or fine-tune
#
# Day 2 · Session 2 (How much to fine-tune) · 25 min.
#
# Labels are the department's scarce resource: Sultan Al-Yami's inspectors label a few hundred
# crops a week. You train the cheapest thing, a linear probe on frozen features with 10% of the
# labels, and set it beside supplied runs of full fine-tuning on 10% and on 100%. The table
# tells you whether the next week should buy labels or training.
#
# **Tasks**
# - [ ] Freeze the backbone and extract features for 10% of the labels · 6 min
# - [ ] Train the linear probe · 5 min
# - [ ] Load the supplied full fine-tuning runs · 4 min
# - [ ] One table: method, labels, accuracy by slice, minutes · 6 min
# - [ ] Decide what to do next: label or train · 4 min
# - [ ] ✱ Stretch · Unfreeze the last block only and place it in the table
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the probe.

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

import json
import time
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import roadkit as U

U.ensure_data()
by_id = {t["id"]: t for t in U.tiles("train")}
ten = [by_id[i] for i in json.loads((U.DATA / "runs" / "ten_percent_ids.json").read_text()) if i in by_id]
test = U.tiles("test")
Xten, yten = U.tile_images(ten), U.labels(ten)
Xte, yte = U.tile_images(test), U.labels(test)
print(f"mode {U.MODE}: {len(ten)} labelled crops (the 10% budget), {len(test)} test crops")

# %% [markdown]
# ## 1 · Freeze the backbone and extract features
#
# A linear probe never changes the backbone. So run the backbone once, keep what comes out
# just before the classification layer, and train on those numbers.
#
# ✏️ **Exercise** · Turn the pretrained model into a feature extractor by replacing its last
# layer with `nn.Identity()` (`model[1].fc` for the ResNet, `model[1].head` for the ViT), then
# return `U.predict(model, X, size=128)`.

# %%
def frozen_features(name, X):
    model = U.backbone(name)  # pretrained weights, a new head we do not use
    # BEGIN SOLUTION
    if name == "resnet18":
        model[1].fc = nn.Identity()
    else:
        model[1].head = nn.Identity()
    return U.predict(model, X, size=128)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _frozen_features(name, X):
    model = U.backbone(name)
    setattr(model[1], "fc" if name == "resnet18" else "head", nn.Identity())
    return U.predict(model, X, size=128)

feats, probe_minutes = {}, {}
for name in ("resnet18", "vit_tiny"):
    t = time.time()
    try:
        feats[name] = (frozen_features(name, Xten), frozen_features(name, Xte))
    except NotImplementedError:
        feats[name] = (_frozen_features(name, Xten), _frozen_features(name, Xte))
    probe_minutes[name] = (time.time() - t) / 60
    print(f"{name:9s} features {feats[name][0].shape}  in {probe_minutes[name]:.1f} min")
assert feats["resnet18"][0].shape[1] == 512 and feats["vit_tiny"][0].shape[1] == 192

# %% [markdown]
# ## 2 · Train the linear probe
#
# ✏️ **Exercise** · Fit a logistic regression on the training features (standardise them
# first: subtract the training mean, divide by the training standard deviation) and return
# its predictions for the test features.

# %%
def linear_probe(F_train, y_train, F_test):
    # BEGIN SOLUTION
    mu, sd = F_train.mean(0), F_train.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((F_train - mu) / sd, y_train)
    return clf.predict((F_test - mu) / sd)
    # END SOLUTION

# %%
def _linear_probe(F_train, y_train, F_test):
    mu, sd = F_train.mean(0), F_train.std(0) + 1e-6
    return LogisticRegression(max_iter=2000).fit((F_train - mu) / sd, y_train).predict((F_test - mu) / sd)

preds = {}
for name in feats:
    try:
        preds[(name, "linear probe", 10)] = linear_probe(feats[name][0], yten, feats[name][1])
    except NotImplementedError:
        preds[(name, "linear probe", 10)] = _linear_probe(feats[name][0], yten, feats[name][1])
    print(f"{name:9s} linear probe on {len(ten)} labels: test accuracy {(preds[(name, 'linear probe', 10)] == yte).mean():.3f}")

# %% [markdown]
# ## 3 · Load the supplied full fine-tuning runs
#
# Four runs were made for you on the full pack: each backbone, every weight trained, on the
# same 10% of labels and on 100%. A run file holds the settings and one prediction per test crop.

# %%
run_minutes = {}
test_ids = [t["id"] for t in test]
for name in ("resnet18", "vit_tiny"):
    for pct in (10, 100):
        run = json.loads((U.DATA / "runs" / f"{name}_full_{pct}.json").read_text())
        lookup = dict(zip(run["test_ids"], run["pred"]))
        preds[(name, "full fine-tuning", pct)] = np.array([lookup[i] for i in test_ids])
        run_minutes[(name, pct)] = run["minutes_on_laptop_processor"]
        print(f"{name:9s} full fine-tuning on {pct:3d}%: {run['train_crops']} crops, {run['epochs']} passes, "
              f"{run['minutes_on_laptop_processor']} min on a laptop processor")

# %% [markdown]
# ## 4 · One table: method, labels, accuracy by slice, minutes
#
# ✏️ **Exercise** · Return the accuracy of one prediction array on all test crops and on each
# lighting condition, as a dictionary with the keys `all`, `day`, `glare`, `dust`, `rain`, `night`.

# %%
def accuracy_by_slice(pred):
    # BEGIN SOLUTION
    ok = pred == yte
    out = {"all": float(ok.mean())}
    for c in U.CONDITIONS:
        out[c] = float(np.mean([o for t, o in zip(test, ok) if t["condition"] == c]))
    return out
    # END SOLUTION

# %%
def _accuracy_by_slice(pred):
    ok = pred == yte
    return {"all": float(ok.mean()), **{c: float(np.mean([o for t, o in zip(test, ok) if t["condition"] == c])) for c in U.CONDITIONS}}

try:
    accuracy_by_slice(yte)
except NotImplementedError:
    accuracy_by_slice = _accuracy_by_slice
assert accuracy_by_slice(yte)["all"] == 1.0

table = {}
print(f"{'backbone':9s} {'method':17s} {'labels':>6s}  " + " ".join(f"{c:>6s}" for c in ["all"] + U.CONDITIONS) + "  minutes")
for (name, method, pct), p in sorted(preds.items()):
    acc = accuracy_by_slice(p)
    mins = probe_minutes[name] if method == "linear probe" else run_minutes[(name, pct)]
    table[f"{name}|{method}|{pct}"] = {**{k: round(v, 3) for k, v in acc.items()}, "minutes": round(mins, 1)}
    print(f"{name:9s} {method:17s} {pct:5d}%  " + " ".join(f"{acc[c]:6.3f}" for c in ["all"] + U.CONDITIONS) + f"  {mins:6.1f}")
U.save_reference("d2_2_table", table)

# %% [markdown]
# **Look at** three gaps: probe against full fine-tuning at the same 10% (what training more
# weights buys), 10% against 100% for full fine-tuning (what ten times the labels buy), and
# the worst lighting column of each row. The minutes of the probe were measured on this
# machine, those of the runs on a laptop processor: compare orders of magnitude only.
#
# One caution in quick mode: your probe was trained on the part of the 10% that lies in the
# quick pack (about 140 crops) and is scored on about 300 test crops, while the supplied runs
# were trained on all 717. The gap between the probe and full fine-tuning is then wider than
# a like-for-like comparison gives. In full mode both use the same 717 crops and all 1,575
# test crops: there the probes reach 0.81 (ResNet-18) and 0.76 (ViT-Tiny).

# %% [markdown]
# ## 5 · Decide what to do next
#
# ✏️ **Exercise** · Return a dictionary with `next` (`"label"` or `"train"`) and `why` (one
# sentence with two numbers from your table). Rule of thumb for this lab: if ten times the
# labels add more accuracy than moving from the probe to full fine-tuning at 10%, labels are
# the better buy.

# %%
def decide(table):
    # BEGIN SOLUTION
    best = max(("resnet18", "vit_tiny"), key=lambda n: table[f"{n}|full fine-tuning|100"]["all"])
    probe = table[f"{best}|linear probe|10"]["all"]
    ft10 = table[f"{best}|full fine-tuning|10"]["all"]
    ft100 = table[f"{best}|full fine-tuning|100"]["all"]
    labels_gain, training_gain = ft100 - ft10, ft10 - probe
    return {"next": "label" if labels_gain > training_gain else "train",
            "why": f"On {best}, ten times the labels add {labels_gain:+.3f} accuracy; "
                   f"training every weight on the same 10% adds {training_gain:+.3f} over the probe."}
    # END SOLUTION

# %%
try:
    decision = decide(table)
except NotImplementedError:
    decision = U.reference("d2_2_decision")
assert decision["next"] in ("label", "train") and decision["why"]
print(decision["next"].upper(), "·", decision["why"])
U.save_reference("d2_2_decision", decision)

# %% [markdown]
# ## ✱ Stretch · Partial fine-tuning
#
# Between the probe and full fine-tuning: train the head and the last block only
# (`model[1].layer4` and `model[1].fc` in the ResNet). Set `requires_grad = False` on every
# other parameter, call `U.fit(model, Xten, yten, epochs=8, lr=3e-4, size=128)` and add the
# row to the table. Where does it land in accuracy and in minutes?

# %% [markdown]
# ## Check yourself
#
# 1. Why can the probe's features be computed once and reused for every later experiment?
# 2. A probe beats full fine-tuning on 10% of the labels. Give one reason this can happen.
# 3. Full fine-tuning on 100% wins overall but its worst slice is still weak. What does the
#    table tell you to label next?
#
# <details><summary>Answers</summary>
#
# 1. The backbone is frozen, so its output for a crop never changes. Only the small linear
#    layer is trained.
# 2. With few labels, moving millions of weights can overfit or forget what pretraining
#    taught; a probe moves a few thousand and keeps the pretrained features intact.
# 3. More crops of the weak condition, not more crops overall. A larger random sample mostly
#    adds daylight, which the model already handles.
# </details>
#
# ## Carry forward
#
# The table of this lab is the form of the evidence for outcome 2: methods as rows,
# conditions as columns. Lab 2.3 adds one more row, the model that used no labels at all.
