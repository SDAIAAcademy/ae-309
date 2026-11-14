# %% [markdown]
# # Lab 2.4 · Which backbone may ship
#
# Day 2 · Session 4 (Licences of open-weight backbones) · 25 min.
#
# "Open weights" does not mean "free to deploy". Yousef Al-Maliki wants to put one of three
# backbones on Al-Manar's roadside units, which a contractor operates under a paid contract.
# You read what the model hub records about each, separate the terms of the code, the weights
# and the training data, apply the department's rule and write the deployment record.
#
# **Tasks**
# - [ ] Read the licence metadata of three backbones · 6 min
# - [ ] Separate code, weights and training-data terms · 7 min
# - [ ] Apply the department's deployment rule · 6 min
# - [ ] Write the deployment record · 6 min
# - [ ] ✱ Stretch · Check the licence of a detection library the same way
#
# AI policy for this lab: an assistant may explain a licence term; the decision and the record
# are yours. This lab teaches a method, not law: the department's legal officer signs the record.

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
import timm
import roadkit as U

CANDIDATES = [
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "vit_small_patch16_dinov3.lvd1689m",
]
print("timm", timm.__version__)

# %% [markdown]
# ## 1 · Read the licence metadata of three backbones
#
# timm ships a small record with every set of pretrained weights. No weights are downloaded
# here: `timm.get_pretrained_cfg(name)` reads the record only.
#
# ✏️ **Exercise** · For one model name, return a dictionary with the keys `name`,
# `weights_licence` (the record's `license`) and `hub_id` (its `hf_hub_id`).

# %%
def read_metadata(name):
    # BEGIN SOLUTION
    cfg = timm.get_pretrained_cfg(name)
    return {"name": name, "weights_licence": cfg.license, "hub_id": cfg.hf_hub_id}
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    meta = [read_metadata(n) for n in CANDIDATES]
except NotImplementedError:
    meta = U.reference("d2_4_meta")
assert [m["name"] for m in meta] == CANDIDATES and all(m["weights_licence"] for m in meta)
for m in meta:
    print(f"{m['name']:44s} weights: {m['weights_licence']:16s} hub: {m['hub_id']}")
U.save_reference("d2_4_meta", meta)

# %% [markdown]
# **Look at** the three values. They are identifiers, not the terms. Open the model page of
# each (huggingface.co/ followed by the hub id) and find the licence text itself before you
# rely on an identifier.

# %% [markdown]
# ## 2 · Separate code, weights and training-data terms
#
# Three things carry terms, and they can differ:
#
# - the **code** that builds the network (timm's own code is Apache-2.0),
# - the **weights**, which their publisher licenses separately,
# - the **training data**, whose terms of access may reach the weights trained on it.
#
# The tag after the dot in a timm name says where the weights come from. The table below is
# the department's reading aid for licence identifiers and data tags.

# %%
LICENCE_KIND = {
    "apache-2.0": "permissive",        # commercial use allowed, keep the notices
    "mit": "permissive",
    "bsd-3-clause": "permissive",
    "cc-by-nc-4.0": "non-commercial",  # no commercial use
    "dinov3-license": "custom",        # a publisher's own agreement: read and accept it
    "agpl-3.0": "network copyleft",    # offering the service obliges you to offer its source
}
DATA_TAGS = {
    "in1k": "ImageNet-1k", "in21k": "ImageNet-21k", "in22k": "ImageNet-22k",
    "lvd1689m": "LVD-1689M (web images collected by the publisher)", "fcmae": "self-supervised pretraining on ImageNet",
}

# %% [markdown]
# ✏️ **Exercise** · Add two fields to a metadata record and return it: `licence_kind` from
# `LICENCE_KIND` (use `"unknown"` for an identifier that is not in the table) and
# `training_data`, the list of data sets named by the tag of the model name (the part after the
# first dot, split on `_`, each piece looked up in `DATA_TAGS`; ignore pieces that are not there).

# %%
def add_terms(m):
    # BEGIN SOLUTION
    tag = m["name"].split(".", 1)[1]
    data = [DATA_TAGS[p] for p in tag.split("_") if p in DATA_TAGS]
    return {**m, "licence_kind": LICENCE_KIND.get(m["weights_licence"], "unknown"), "training_data": data}
    # END SOLUTION

# %%
try:
    terms = [add_terms(m) for m in meta]
except NotImplementedError:
    terms = U.reference("d2_4_terms")
assert terms[0]["licence_kind"] == "permissive" and "ImageNet-21k" in terms[0]["training_data"]
for t in terms:
    print(f"{t['name']:44s} {t['licence_kind']:15s} data: {', '.join(t['training_data'])}")
U.save_reference("d2_4_terms", terms)

# %% [markdown]
# ## 3 · Apply the department's deployment rule
#
# Wejdan Al-Rashidi and the legal officer wrote the rule for roadside units:
#
# 1. The units are run by a contractor under a paid contract: that is **commercial use**.
# 2. Weights under a **permissive** licence may be deployed.
# 3. Weights under a **non-commercial** licence may not be deployed. They may be used in
#    experiments that never leave the lab.
# 4. Weights under a **custom** or **unknown** licence wait for legal review.
# 5. **Network copyleft** needs a commercial licence from the publisher, because the
#    department does not publish the source of its services.
# 6. Whatever the decision, the record names the training data, so the legal officer can read
#    the data's own terms of access.
#
# ✏️ **Exercise** · Return the decision for one record: `"deploy"`, `"do not deploy"`,
# `"hold for legal review"` or `"needs a commercial licence"`.

# %%
def decide(t):
    # BEGIN SOLUTION
    return {"permissive": "deploy", "non-commercial": "do not deploy",
            "network copyleft": "needs a commercial licence"}.get(t["licence_kind"], "hold for legal review")
    # END SOLUTION

# %%
try:
    decisions = [decide(t) for t in terms]
except NotImplementedError:
    decisions = U.reference("d2_4_decisions")
assert decisions == ["deploy", "do not deploy", "hold for legal review"], decisions
for t, d in zip(terms, decisions):
    print(f"{t['name']:44s} {d}")
U.save_reference("d2_4_decisions", decisions)

# %% [markdown]
# ## 4 · Write the deployment record
#
# A decision nobody can find is made again by the next engineer, differently. The record is a
# small file in the repository, next to the model card.
#
# ✏️ **Exercise** · Return the record: a list with one dictionary per backbone holding `name`,
# `weights_licence`, `licence_kind`, `training_data`, `decision`, `checked_on` (today's date
# as text) and `checked_with` (the version of the tool the metadata came from).

# %%
def deployment_record(terms, decisions):
    # BEGIN SOLUTION
    import datetime
    return [{"name": t["name"], "weights_licence": t["weights_licence"], "licence_kind": t["licence_kind"],
             "training_data": t["training_data"], "decision": d,
             "checked_on": datetime.date.today().isoformat(), "checked_with": f"timm {timm.__version__}"}
            for t, d in zip(terms, decisions)]
    # END SOLUTION

# %%
try:
    record = deployment_record(terms, decisions)
except NotImplementedError:
    record = [{**t, "decision": d, "checked_on": "see the solution", "checked_with": f"timm {timm.__version__}"}
              for t, d in zip(terms, decisions)]
assert all({"name", "weights_licence", "decision", "checked_on", "checked_with"} <= set(r) for r in record)
out = U.CACHE / "deployment_record.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record[0], indent=2))
print("\nwritten:", out.name, "with", len(record), "entries")

# %% [markdown]
# **Look at** `checked_on` and `checked_with`. A publisher can change a licence between
# versions of the same weights, and a hub record can be corrected. The record states what was
# true on one day according to one source.

# %% [markdown]
# ## ✱ Stretch · A detection library
#
# Ultralytics publishes its detection library and models under AGPL-3.0, with an enterprise
# licence as the alternative (read its licensing page). Run a record
# `{"name": "ultralytics.yolo", "weights_licence": "agpl-3.0", "hub_id": None}` through
# `LICENCE_KIND` and `decide`. Which decision comes out, and which two ways forward does the
# department have?

# %% [markdown]
# ## Check yourself
#
# 1. A backbone's code is Apache-2.0 and its weights are CC-BY-NC-4.0. May the department
#    deploy the model on its roadside units?
# 2. You fine-tune non-commercial weights on the department's own images. Are the fine-tuned
#    weights now the department's to deploy?
# 3. Why does the record keep the training data even for a permissive licence?
#
# <details><summary>Answers</summary>
#
# 1. Not under the rule: what runs on the unit is the weights, and their licence forbids
#    commercial use. The code's licence does not override it.
# 2. Do not assume so. Fine-tuned weights are derived from the original ones and carry their
#    terms unless the licence says otherwise: a question for the legal officer, and a reason
#    to choose the starting weights before any training.
# 3. The licence of the weights is the publisher's permission. The data the publisher trained
#    on has terms of its own, and torchvision's documentation reminds users that pretrained
#    models may carry terms derived from their training data.
# </details>
#
# ## Carry forward
#
# The deployment record goes into the capstone repository beside the model card. Day 3 and
# Day 4 use a ResNet-18 trunk with torchvision's ImageNet weights: add it to your record.
