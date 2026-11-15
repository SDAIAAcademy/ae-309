# %% [markdown]
# # Lab 1.1 · Errors by condition
#
# Day 1 · Session 1 (How vision systems fail in the field) · 25 min.
#
# Al-Manar's road maintenance team trained a pilot classifier on crops from its survey vehicle:
# clean daylight, a sharp camera. It scored well and was about to go onto the roadside units.
# You run it on 200 crops as the field sees them and find where it breaks, by lighting, blur
# and viewpoint.
#
# **Tasks**
# - [ ] Load the pilot classifier and score 200 field crops · 5 min
# - [ ] Accuracy by lighting, blur and viewpoint · 8 min
# - [ ] Look at 16 errors of the worst slice · 6 min
# - [ ] Write a three-line failure note for the head of maintenance · 6 min
# - [ ] ✱ Stretch · Cross lighting with viewpoint and find the worst cell
#
# AI policy for this lab: an assistant may explain code and errors; it does not write your analysis.

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

U.ensure_data()  # draws the Al-Manar pack on the first run (a few minutes), then reuses it

# %% [markdown]
# ## 1 · Load the pilot classifier and score 200 field crops
#
# The supplied checkpoint `pilot_classifier` sorts a 96-pixel crop into five classes: clear,
# pothole, longitudinal crack, transverse crack, patch. First its score on the kind of data it
# was built with, then on the field batch.

# %%
model = U.load_ckpt(U.TileNet(), "pilot_classifier")

survey = [t for t in U.tiles("val") if t["condition"] == "day" and not t["blur"]]
survey_acc = float((U.predict(model, U.tile_images(survey)).argmax(1) == U.labels(survey)).mean())
print(f"survey-style validation crops: {len(survey)}, accuracy {survey_acc:.3f}")

X, rows = U.field_tiles(200)
y = U.labels(rows)
pred = U.predict(model, X).argmax(1)
correct = pred == y
print(f"field crops: {len(rows)}, accuracy {correct.mean():.3f}")
print("one row:", rows[0])

# %%
U.show(X[:16], titles=[f"{r['lighting']}/{r['viewpoint'][:4]}" for r in rows[:16]])

# %% [markdown]
# ## 2 · Accuracy by lighting, blur and viewpoint
#
# One number hides the field. Every row carries three fields: `lighting` (day, glare, dust,
# rain, night), `blur` (sharp, blurred) and `viewpoint` (vehicle, pole).
#
# ✏️ **Exercise** · Return, for one field, a dictionary `{value: (count, accuracy)}`.

# %%
def by_slice(rows, correct, key):
    # BEGIN SOLUTION
    out = {}
    for value in sorted({r[key] for r in rows}):
        hit = [c for r, c in zip(rows, correct) if r[key] == value]
        out[value] = (len(hit), float(np.mean(hit)))
    return out
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    slices = {k: by_slice(rows, correct, k) for k in ("lighting", "blur", "viewpoint")}
except NotImplementedError:
    slices = {k: {a: tuple(b) for a, b in v.items()} for k, v in U.reference("d1_1_slices").items()}
assert sum(n for n, _ in slices["lighting"].values()) == len(rows)
for key, table in slices.items():
    print(key)
    for value, (n, acc) in table.items():
        print(f"  {value:8s} n={n:3d}  accuracy {acc:.2f}  {'#' * int(acc * 30)}")
U.save_reference("d1_1_slices", slices)

# %% [markdown]
# **Look at** the counts as well as the accuracies. A slice of 30 crops moves by 3 points when
# one crop flips: tomorrow you put intervals on numbers like these.

# %% [markdown]
# ## 3 · Look at 16 errors of the worst slice
#
# ✏️ **Exercise** · Return the `(key, value)` of the slice with the lowest accuracy among the
# slices that hold at least 20 crops.

# %%
def worst_slice(slices, min_count=20):
    # BEGIN SOLUTION
    best = None
    for key, table in slices.items():
        for value, (n, acc) in table.items():
            if n >= min_count and (best is None or acc < best[2]):
                best = (key, value, acc)
    return best[0], best[1]
    # END SOLUTION

# %%
try:
    wkey, wvalue = worst_slice(slices)
except NotImplementedError:
    wkey, wvalue = U.reference("d1_1_worst")
assert slices[wkey][wvalue][0] >= 20
print("worst slice:", wkey, "=", wvalue, slices[wkey][wvalue])
U.save_reference("d1_1_worst", [wkey, wvalue])

bad = [i for i, r in enumerate(rows) if r[wkey] == wvalue and not correct[i]][:16]
U.show(X[bad], titles=[f"{rows[i]['label'][:9]}>{U.TILE_CLASSES[pred[i]][:9]}" for i in bad])

# %% [markdown]
# **Look at** the titles (true class > predicted class). Is the classifier guessing one class
# for the whole slice? Could you name the defect yourself in these crops? If a person can and
# the model cannot, the model has not seen this condition. If a person cannot either, no
# training fixes it and the camera or the lighting must change.

# %% [markdown]
# ## 4 · Write a three-line failure note
#
# Rakan Al-Shammari, head of road maintenance, will read three lines. Fill them from your
# numbers: what was promised, what the field shows, what you ask for.
#
# ✏️ **Exercise** · Return the note as a dictionary with the keys `promised`, `field` and `ask`.
# `field` must name the worst slice and its accuracy.

# %%
def failure_note():
    # BEGIN SOLUTION
    n, acc = slices[wkey][wvalue]
    return {
        "promised": f"{survey_acc:.0%} accuracy on survey-style crops (clean daylight, vehicle camera).",
        "field": f"{correct.mean():.0%} on 200 field crops; worst slice {wkey}={wvalue}: {acc:.0%} on {n} crops.",
        "ask": "Do not deploy. Collect and label crops under dust, night, rain and glare and from pole "
               "cameras, and report every later model by these slices.",
    }
    # END SOLUTION

# %%
try:
    note = failure_note()
except NotImplementedError:
    note = U.reference("d1_1_note")
assert set(note) == {"promised", "field", "ask"} and wvalue in note["field"]
for k, v in note.items():
    print(f"{k:9s} {v}")
U.save_reference("d1_1_note", note)

# %% [markdown]
# ## ✱ Stretch · The worst cell
#
# Cross `lighting` with `viewpoint` (ten cells). Which cell is worst, and how many crops does
# it hold? What would you need before you trusted its accuracy?

# %% [markdown]
# ## Check yourself
#
# 1. The pilot scored about 95% on its own validation crops and under half on the field
#    batch. Was the validation number wrong?
# 2. Why does the lab ask for a minimum count before naming the worst slice?
# 3. Which of the three fields could the department fix without retraining?
#
# <details><summary>Answers</summary>
#
# 1. No. It was a correct answer to the wrong question: how the model does on clean daylight
#    crops from the survey vehicle. The field asks about other conditions.
# 2. A slice of five crops can read 0% or 100% by chance. A small slice is a reason to
#    collect more, not a finding.
# 3. Blur (a faster shutter or a slower survey speed) and viewpoint (mounting rules for pole
#    cameras). Lighting and weather come with the job.
# </details>
#
# ## Carry forward
#
# The slice table is the shape of every result this week: by lighting, by weather, later by
# district. Lab 4.3 runs the same idea on the detector, with intervals.
