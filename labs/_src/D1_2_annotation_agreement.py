# %% [markdown]
# # Lab 1.2 · Agree on a new class
#
# Day 1 · Session 2 (Annotation guidelines and agreement) · 25 min.
#
# The department's old labels have two crack classes. Sultan Al-Yami, the lead inspector, wants
# a third: **alligator cracking**, the crocodile-skin pattern that means the road base is
# failing and a patch will not hold. You and a partner label 30 crops with a first guideline,
# measure how far you agree, repair the guideline and measure again.
#
# **Tasks**
# - [ ] Read guideline v1 and label the 30 crops · 8 min
# - [ ] Raw agreement and Cohen's kappa with your partner · 6 min
# - [ ] Read the disagreements and write the rule that settles them · 6 min
# - [ ] Kappa on the second pass and overlap of the boxes · 5 min
# - [ ] ✱ Stretch · Score each annotator against the adjudicated labels
#
# AI policy for this lab: **AI off**. Labelling and judging a guideline are the skill.

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
import roadkit as U

U.ensure_data()
batch = U.annot_batch()
ids = [item["id"] for item in batch["items"]]
print(len(ids), "crops:", ids[0], "to", ids[-1])

# %% [markdown]
# ## 1 · Read guideline v1 and label the 30 crops
#
# > **Guideline v1 · Alligator cracking.** Cracks that join into a pattern like crocodile
# > skin. Label 1 when the crop shows alligator cracking, otherwise 0.
#
# Look at the crops, then type your 30 labels. Do not confer yet: agreement is only worth
# measuring when the two passes are independent.

# %%
U.show([U.annot_image(i) for i in ids], titles=ids, cols=10, size=1.5)

# %% [markdown]
# ✏️ **Exercise** · Return your labels as a dictionary `{"A01": 0, "A02": 1, ...}` with all 30 ids.

# %%
def my_labels():
    # BEGIN SOLUTION
    return dict(batch["pass1"]["A"])  # the solution plays annotator A, who reads v1 generously
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    mine = my_labels()
except NotImplementedError:
    mine = dict(batch["pass1"]["A"])
assert set(mine) == set(ids) and set(mine.values()) <= {0, 1}, "30 labels, each 0 or 1"

# Your partner's labels. Working alone? Keep annotator B, who read the same guideline strictly.
partner = dict(batch["pass1"]["B"])
print("you said yes to", sum(mine.values()), "crops; your partner to", sum(partner.values()))

# %% [markdown]
# ## 2 · Raw agreement and Cohen's kappa
#
# Raw agreement is the share of crops with the same label. Two people who both say "no" to
# most crops agree often by chance alone. Cohen's kappa removes that chance agreement:
#
# kappa = (p_observed - p_chance) / (1 - p_chance), with
# p_chance = P(a says 1) x P(b says 1) + P(a says 0) x P(b says 0).
#
# ✏️ **Exercise** · Write both. Each takes two dictionaries with the same ids.

# %%
def raw_agreement(a, b):
    # BEGIN SOLUTION
    return float(np.mean([a[i] == b[i] for i in a]))
    # END SOLUTION


def cohen_kappa(a, b):
    # BEGIN SOLUTION
    po = raw_agreement(a, b)
    pa, pb = np.mean([a[i] for i in a]), np.mean([b[i] for i in a])
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe))
    # END SOLUTION

# %%
try:
    agree1, kappa1 = raw_agreement(mine, partner), cohen_kappa(mine, partner)
    assert abs(cohen_kappa({"x": 1, "y": 0, "z": 1, "w": 0}, {"x": 1, "y": 0, "z": 0, "w": 0}) - 0.5) < 1e-9
except NotImplementedError:
    agree1, kappa1 = U.reference("d1_2_pass1")
print(f"pass 1: raw agreement {agree1:.2f}, Cohen's kappa {kappa1:.2f}")
U.save_reference("d1_2_pass1", [agree1, kappa1])

# %% [markdown]
# **Look at** the two numbers together. A common reading of kappa: below 0.4 the guideline is
# not working, 0.6 to 0.8 is usable, above 0.8 is strong. These bands are a convention, not a
# law: say the number and the task.

# %% [markdown]
# ## 3 · Read the disagreements and write the rule that settles them
#
# Do not vote and do not average. Look at every crop you disagreed on and ask what the
# guideline failed to say.

# %%
split = [i for i in ids if mine[i] != partner[i]]
print(len(split), "disagreements:", split)
U.show([U.annot_image(i) for i in split], titles=[f"{i}: you {mine[i]}, partner {partner[i]}" for i in split],
       cols=max(4, min(8, len(split))), size=2.0)

# %% [markdown]
# Write your rule in one sentence before you open the department's version.
#
# <details><summary>Guideline v2, as Sultan Al-Yami wrote it</summary>
#
# > **Guideline v2 · Alligator cracking.** Interconnected cracks that enclose **at least three
# > closed cells**. Two cracks that cross are not alligator cracking. One or two closed cells
# > are block cracking: label 0 and leave a note. When a cell is cut by the edge of the crop,
# > count it only if three of its sides are visible. Draw the box around the outermost cells.
#
# The rule counts something a second person can count. "Like crocodile skin" asked for a
# judgement of resemblance, and two careful people judged differently.
# </details>

# %% [markdown]
# ## 4 · Kappa on the second pass, and overlap of the boxes
#
# Both annotators labelled the 30 crops again with guideline v2 and drew a box on every
# positive. (Label again yourself if time allows and put your dictionary in place of A's.)
#
# ✏️ **Exercise** · For the crops both annotators boxed, return the mean IoU of their two boxes
# and the share of pairs with IoU of at least 0.5. `U.box_iou(a, b)` gives an IoU matrix.

# %%
second_a, second_b = batch["pass2"]["A"], batch["pass2"]["B"]
boxes_a, boxes_b = batch["boxes_pass2"]["A"], batch["boxes_pass2"]["B"]


def box_agreement(boxes_a, boxes_b):
    # BEGIN SOLUTION
    both = sorted(set(boxes_a) & set(boxes_b))
    ious = [float(U.box_iou([boxes_a[i]], [boxes_b[i]])[0, 0]) for i in both]
    return float(np.mean(ious)), float(np.mean([v >= 0.5 for v in ious]))
    # END SOLUTION

# %%
try:
    kappa2 = cohen_kappa(second_a, second_b)
    mean_iou, share = box_agreement(boxes_a, boxes_b)
except NotImplementedError:
    kappa2, mean_iou, share = U.reference("d1_2_pass2")
assert kappa2 > kappa1, "the second pass should agree more than the first"
print(f"pass 2: kappa {kappa2:.2f} (pass 1: {kappa1:.2f})")
print(f"boxes: mean IoU {mean_iou:.2f}, share with IoU >= 0.5: {share:.2f}")
U.save_reference("d1_2_pass2", [kappa2, mean_iou, share])

# %% [markdown]
# **Look at** the box overlap. The two annotators now agree on *which* crops, and still draw
# different boxes: one hugs the cracks, the other leaves a margin. A detector trained on a mix
# of the two learns a blurred edge, and an evaluation at IoU 0.75 punishes it for the
# annotators' habit. The guideline needs a line on how tight a box is.

# %% [markdown]
# ## ✱ Stretch · Each annotator against the adjudicated labels
#
# `U.annot_truth()` holds the adjudicated label of each crop (`truth`) and how many closed
# cells it has (`cells`). Compute each annotator's kappa against it, for both passes. Who
# improved more, and on which kind of crop?

# %% [markdown]
# ## Check yourself
#
# 1. Two annotators agree on 90% of crops and kappa is 0.2. How can both be true?
# 2. Why not settle the disagreements by a majority vote of three annotators?
# 3. After v2, kappa is high but box IoU between annotators averages well under 1. What does
#    that put a ceiling on?
#
# <details><summary>Answers</summary>
#
# 1. The class is rare. If both say "no" to nearly everything they agree by chance on most
#    crops; kappa measures agreement beyond that.
# 2. A vote fixes these 30 labels and leaves the guideline broken for the next 3,000. The
#    disagreement is information about the rule.
# 3. On any score that uses a strict overlap threshold: a model cannot be scored as more
#    consistent with the labels than the annotators are with each other.
# </details>
#
# ## Carry forward
#
# Guideline v2 and the two kappas go into the dataset card (the capstone asks for one). The
# annotation tools named in the course (CVAT, Label Studio) export boxes in the COCO format
# that `cache/roads/annotations_coco.json` uses.
