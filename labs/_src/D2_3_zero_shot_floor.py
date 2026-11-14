# %% [markdown]
# # Lab 2.3 · The zero-shot floor
#
# Day 2 · Session 3 (Zero-shot baselines) · 25 min.
#
# An open-vocabulary model was trained on image and text pairs from the web. Give it a
# sentence per class and it classifies a crop with no training at all. Whatever it scores on
# Al-Manar's defect classes is the **floor**: a trained model that does not clearly beat it
# has not earned its labels. You measure that floor, see how much it depends on the wording,
# and put it in yesterday's table.
#
# **Tasks**
# - [ ] Write one prompt per class · 5 min
# - [ ] Score the open-vocabulary model on the test crops · 7 min
# - [ ] Compare two prompt sets · 6 min
# - [ ] Place the floor beside the probe and the fine-tuned runs · 4 min
# - [ ] Name the class the floor is worst at · 3 min
# - [ ] ✱ Stretch · Build a five-shot prototype classifier from image embeddings
#
# AI policy for this lab: an assistant may explain code and errors; it does not write the scoring.
# The model (CLIP ViT-B/32, about 350 MB) downloads on the first run.

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
import numpy as np
import open_clip
import torch
import roadkit as U

U.ensure_data()
test = U.tiles("test")[: U.pick(1500, 300)]
Xte, yte = U.tile_images(test), U.labels(test)

clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
clip = clip.to(U.DEV).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
MEAN = torch.tensor(open_clip.OPENAI_DATASET_MEAN, device=U.DEV).view(1, 3, 1, 1)
STD = torch.tensor(open_clip.OPENAI_DATASET_STD, device=U.DEV).view(1, 3, 1, 1)


@torch.no_grad()
def embed_images(images, bs=64):
    """One unit-length embedding per crop, from the model's image tower (input 224 x 224)."""
    out = []
    for i in range(0, len(images), bs):
        x = (U.to_tensor(images[i:i + bs], size=224) - MEAN) / STD
        f = clip.encode_image(x).float()
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out)


image_emb = embed_images(Xte)
print(f"mode {U.MODE}: {len(test)} test crops embedded, {tuple(image_emb.shape)}")

# %% [markdown]
# ## 1 · Write one prompt per class
#
# The classifier is the list of sentences. Start with the bare class names, as a colleague
# in a hurry would.
#
# ✏️ **Exercise** · Return five sentences in the order of `U.TILE_CLASSES`: clear, pothole,
# crack_long, crack_trans, patch. Describe what the camera sees: these crops look straight
# down at asphalt.

# %%
BARE = ["clear", "pothole", "longitudinal crack", "transverse crack", "patch"]


def my_prompts():
    # BEGIN SOLUTION
    return ["a top-down photo of a smooth asphalt road surface with no damage",
            "a top-down photo of an asphalt road surface with a pothole, a dark hole",
            "a top-down photo of an asphalt road surface with a long vertical crack",
            "a top-down photo of an asphalt road surface with a horizontal crack across it",
            "a top-down photo of an asphalt road surface with a rectangular repair patch of darker asphalt"]
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    DESCRIBED = my_prompts()
except NotImplementedError:
    DESCRIBED = U.reference("d2_3_prompts")
assert len(DESCRIBED) == 5 and all(isinstance(p, str) and p for p in DESCRIBED)
U.save_reference("d2_3_prompts", DESCRIBED)

# %% [markdown]
# ## 2 · Score the open-vocabulary model on the test crops
#
# The text tower turns each sentence into an embedding in the same space as the images. A crop
# gets the class whose sentence its embedding is most similar to.
#
# ✏️ **Exercise** · Embed the prompts (`clip.encode_text(tokenizer(prompts).to(U.DEV))`),
# scale each embedding to unit length, take the similarity of every image embedding with every
# text embedding (a matrix product) and return the index of the best prompt per crop as a
# NumPy array.

# %%
@torch.no_grad()
def zero_shot(prompts, image_emb):
    # BEGIN SOLUTION
    t = clip.encode_text(tokenizer(prompts).to(U.DEV)).float()
    t = (t / t.norm(dim=-1, keepdim=True)).cpu()
    return (image_emb @ t.T).argmax(dim=1).numpy()
    # END SOLUTION

# %%
@torch.no_grad()
def _zero_shot(prompts, image_emb):
    t = clip.encode_text(tokenizer(prompts).to(U.DEV)).float()
    return (image_emb @ (t / t.norm(dim=-1, keepdim=True)).cpu().T).argmax(dim=1).numpy()

try:
    pred_described = zero_shot(DESCRIBED, image_emb)
except NotImplementedError:
    zero_shot = _zero_shot
    pred_described = zero_shot(DESCRIBED, image_emb)
assert pred_described.shape == (len(test),) and set(pred_described) <= set(range(5))
acc_described = float((pred_described == yte).mean())
print(f"zero-shot accuracy, your prompts: {acc_described:.3f}")

# %% [markdown]
# ## 3 · Compare two prompt sets

# %%
pred_bare = zero_shot(BARE, image_emb)
acc_bare = float((pred_bare == yte).mean())
majority = float(np.bincount(yte).max() / len(yte))
print(f"always the most common class: {majority:.3f}")
print(f"bare class names:             {acc_bare:.3f}")
print(f"described prompts:            {acc_described:.3f}")
print("\npredictions per class (bare | described):")
for k, c in enumerate(U.TILE_CLASSES):
    print(f"  {c:12s} {int((pred_bare == k).sum()):5d} | {int((pred_described == k).sum()):5d}   (true: {int((yte == k).sum())})")

# %% [markdown]
# **Look at** how the predictions spread over the classes. A zero-shot model often pours most
# crops into one or two sentences. The wording moved the score without one new label: a
# zero-shot number is a property of the model *and* the prompts, so report both.

# %% [markdown]
# ## 4 · Place the floor beside the probe and the fine-tuned runs
#
# ✏️ **Exercise** · Return the floor: the better of your two zero-shot accuracies, or the
# accuracy of always answering the most common class when that is higher.

# %%
def floor_to_beat():
    # BEGIN SOLUTION
    return max(acc_bare, acc_described, majority)
    # END SOLUTION

# %%
try:
    floor = floor_to_beat()
except NotImplementedError:
    floor = max(acc_bare, acc_described, majority)
ids = [t["id"] for t in test]
print(f"{'the floor to beat':38s} {floor:.3f}")
for name in ("resnet18", "vit_tiny"):
    for pct in (10, 100):
        run = json.loads((U.DATA / "runs" / f"{name}_full_{pct}.json").read_text())
        lookup = dict(zip(run["test_ids"], run["pred"]))
        acc = float(np.mean([lookup[i] == y for i, y in zip(ids, yte)]))
        print(f"{name + ', full fine-tuning, ' + str(pct) + '% labels':38s} {acc:.3f}   ({acc - floor:+.3f} over the floor)")
U.save_reference("d2_3_scores", {"majority": majority, "bare": acc_bare, "described": acc_described, "floor": floor})

# %% [markdown]
# ## 5 · Name the class the floor is worst at
#
# ✏️ **Exercise** · Return the name of the class with the lowest recall under your prompts
# (recall of a class: the share of its crops that were given that class).

# %%
def worst_class(pred):
    # BEGIN SOLUTION
    recall = [float((pred[yte == k] == k).mean()) for k in range(5)]
    return U.TILE_CLASSES[int(np.argmin(recall))]
    # END SOLUTION

# %%
try:
    worst = worst_class(pred_described)
except NotImplementedError:
    worst = U.reference("d2_3_worst")
assert worst in U.TILE_CLASSES
print("lowest recall under the described prompts:", worst)
U.save_reference("d2_3_worst", worst)

# %% [markdown]
# ## ✱ Stretch · Five labelled crops per class
#
# Take five training crops per class, embed them with `embed_images`, average the embeddings
# per class and scale to unit length: five "prototypes". Classify the test crops by the nearest
# prototype. How far above the zero-shot floor do 25 labels take you, without any training?

# %% [markdown]
# ## Check yourself
#
# 1. The zero-shot model scores below the majority-class answer. Is it useless?
# 2. Why is the floor a fairer baseline than "chance, 20%" when you report a fine-tuned model?
# 3. The described prompts beat the bare names. What did the longer sentence give the model?
#
# <details><summary>Answers</summary>
#
# 1. As a classifier for this task, on this evidence, yes: say so and move on. Its image
#    embeddings may still be useful (the stretch item), and on another task the floor may be high.
# 2. It is what the department gets for no labels and no training. The value of the labelling
#    budget is the distance above that, not above a coin.
# 3. The context its training captions had: a viewpoint, a surface, what the defect looks
#    like. The bare word "patch" or "clear" fits thousands of unrelated web pictures.
# </details>
#
# ## Carry forward
#
# The floor is one more row of the Day 2 table. Vision-language models as systems (asking
# questions about images, grounding, multimodal applications) are the subject of AE-408.
