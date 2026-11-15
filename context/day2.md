---
course: "Computer Vision Engineering"
code: "AE-309"
version: "2027.1.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-309/slides/day2.html"
---

# Computer Vision Engineering

## Day 2: Backbones and transfer

Two families of pretrained backbone, how much of one to train, the zero-shot floor, and what a licence lets the department deploy

#### [day2#title] Backbones and transfer

*SDAIA Academy · Computer Vision Engineering*

Day 2: Two families of pretrained backbone, how much of one to train, the zero-shot floor, and what a licence lets the department deploy

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-309

#### [day2#agenda] Backbones and transfer

*Today*

- **Session 1 · Convolutional networks and vision transformers**: What a convolutional network and a vision transformer assume about an image, and how to compare them fairly
- **Session 2 · How much to fine-tune**: Linear probe, partial and full fine-tuning: what each costs in labels, time and risk
- **Session 3 · Zero-shot baselines**: An open-vocabulary model classifies from text prompts with no training: the floor every trained model must beat
- **Session 4 · Licences of open-weight backbones**: Code, weights and training data each carry terms: which backbones a municipality may deploy

#### [day2#review-day-1] Which slice goes into the failure note?

*Review of Day 1*

**Question:** A field batch is sliced by lighting and viewpoint. The cell "night, pole camera" holds 6 crops and scores 17%; "dust, survey vehicle" holds 38 and scores 21%. What does the failure note say?

- A. Night from a pole camera is the worst slice, at 17%
- B. The two worst slices average 19%
- C. Dust from the vehicle is the worst slice with enough crops; the night cell needs more
- D. Slices this small say nothing: report the overall score only

### Session 1: Convolutional networks and vision transformers

#### [day2#s1] Convolutional networks and vision transformers

*Session 1*

- Contrast the assumptions of convolutional networks and vision transformers
- Fine-tune two pretrained backbones on the same crops
- Compare two models by condition slice with bootstrap intervals

#### [day2#start-from-a-backbone] Nobody trains a vision model from nothing: start from a backbone

*Transfer learning*

1. **1 · Pretrained backbone** trained by someone else on a million or more images
2. **2 · New head** its last layer replaced by one for your five classes
3. **3 · Fine-tune** a short schedule on your own crops

> **What transfers:** edges, textures, shapes. A network that has learnt them from a million photographs needs hundreds of road crops, not hundreds of thousands, to learn "pothole".

#### [day2#what-a-convnet-assumes] A convolutional network assumes that nearby pixels belong together

*Family one*

- **Locality.** Each unit looks at a small window. Edges first, then textures, then parts.
- **Shared weights.** The same filter slides over the whole image: a crack is a crack wherever it lies.
- **A pyramid.** Resolution falls and the view widens, layer by layer.

> **These assumptions are true of photographs,** so the network does not have to learn them from data. That is why it does well with little.

#### [day2#what-a-vit-assumes] A vision transformer assumes almost nothing and looks everywhere at once

*Family two*

1. **1 · Patches** a 128-pixel crop is cut into 64 patches of 16 by 16
2. **2 · Tokens** each patch becomes a vector, plus where it was
3. **3 · Attention** every patch can use every other patch, from the first layer
4. **4 · Class token** one summary vector feeds the head

> **Fewer assumptions, more data.** What a convolutional network is given, a transformer must learn, so it depends more on large-scale pretraining and on a careful schedule.

#### [day2#a-fair-comparison] A fair comparison fixes everything that is not the question

*The experiment*

|  | ResNet-18 | ViT-Tiny |
| --- | --- | --- |
| Training crops, passes, input size | The same: 320 crops, one pass, 128 pixels (lab 2.1, quick mode) |  |
| Test crops | The same 304, from road segments neither has seen |  |
| Learning rate | 0.0003 | 0.0001 |
| Weights | 11.2 million | 5.5 million |
| Pretraining data | ImageNet-1k | ImageNet-21k, then ImageNet-1k |

> **The last three rows still differ.** You compare two available starting points under one budget. You do not compare two families.

#### [day2#bootstrap-interval] The bootstrap asks: how far would this number move on another sample?

*The interval*

```python
# file: lab 2.1
def bootstrap_ci(values, n_boot=1000, seed=0):
    values = np.asarray(values, dtype=float)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(values),
                     (n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(values.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))
```

- `values` holds one 0 or 1 per test crop
- Draw the crops again **with replacement**, a thousand times
- The middle 95% of the resampled means is the interval
- It needs no formula, so it works for any metric

#### [day2#two-backbones-by-slice] After one short pass the ResNet leads, and only two rows can say so

*From the lab*

| Slice | Crops | ResNet-18 | ViT-Tiny | Intervals overlap |
| --- | --- | --- | --- | --- |
| All | 304 | 0.73 (0.68 to 0.78) | 0.51 (0.45 to 0.56) | No |
| Day | 185 | 0.78 (0.72 to 0.84) | 0.52 (0.46 to 0.60) | No |
| Glare | 35 | 0.66 (0.51 to 0.80) | 0.54 (0.40 to 0.69) | Yes |
| Dust | 33 | 0.73 (0.58 to 0.88) | 0.67 (0.52 to 0.82) | Yes |
| Rain | 35 | 0.57 (0.40 to 0.71) | 0.31 (0.17 to 0.49) | Yes |
| Night | 16 | 0.56 (0.31 to 0.81) | 0.31 (0.13 to 0.56) | Yes |

> **Sixteen night crops give an interval fifty points wide.** A gap of 25 points at night is not yet a finding.

#### [day2#reading-overlap] Overlap means "this test cannot tell", not "these are equal"

*Reading the table*

- **Intervals apart.** A finding on this test set. Write it down, with the slice and its count.
- **Intervals overlap.** No evidence either way. Collect more crops of that condition, or pair the comparison.
- **Never.** "Equally good at night", or "better at night by 25 points". Neither is supported by 16 crops.

> **Slice by condition, because the release question is about conditions.** Slicing by class answers another useful question: which defect is hard.

#### [day2#checkpoint-1] How do you settle it?

*Checkpoint*

**Question:** Two backbones are scored on the same 320 test crops. A reaches 88% with an interval from 84% to 91%, B 90% with an interval from 87% to 93%. The department must choose one this week. What is the next step?

- A. Bootstrap the difference between the two per crop and read its interval
- B. Choose B: it is two points better
- C. Declare a tie and choose the smaller model
- D. Train both for longer until the intervals separate

#### [day2#lab-2-1] Two backbones by slice

Lab 2.1 · 25 min

Fine-tune a ResNet-18 and a ViT-Tiny on defect crops on a short schedule and compare them by lighting slice with bootstrap intervals.

[D2_1_two_backbones.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D2_1_two_backbones.ipynb)

- [ ] Put a five-class head on each backbone (4 min)
- [ ] Fine-tune each on a short schedule (9 min)
- [ ] Accuracy by lighting slice (5 min)
- [ ] Bootstrap intervals and the comparison table (7 min)
- [ ] ✱ Stretch: Bootstrap the difference between the two models on paired crops

### Session 2: How much to fine-tune

#### [day2#s2] How much to fine-tune

*Session 2*

- Train a linear probe on frozen features
- Compare a probe with full fine-tuning at two label budgets
- Decide whether more labels or more training is the next step

#### [day2#three-depths] Fine-tuning has three depths

*How much to train*

- **Linear probe.** The backbone is frozen. Only the new last layer is trained: 2,565 weights on a ResNet-18.
- **Partial fine-tuning.** The last block and the head are trained. The early layers, which hold edges and textures, stay as they are.
- **Full fine-tuning.** Every weight moves: 11.2 million on a ResNet-18.

> **The deeper you train, the more labels and care it takes.** Millions of weights and a few hundred labels can overfit, or forget what pretraining taught.

#### [day2#features-once] A probe runs the backbone once and trains on what comes out

*The probe*

```python
# file: lab 2.2
def frozen_features(name, X):
    model = U.backbone(name)
    if name == "resnet18":
        model[1].fc = nn.Identity()
    else:
        model[1].head = nn.Identity()
    return U.predict(model, X, size=128)

def linear_probe(F_train, y_train, F_test):
    mu, sd = F_train.mean(0), F_train.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000)
    clf.fit((F_train - mu) / sd, y_train)
    return clf.predict((F_test - mu) / sd)
```

- A frozen backbone gives the same output for a crop every time: **compute it once**
- The classifier is a logistic regression on 512 or 192 numbers per crop
- Under half a minute on a laptop for 717 crops
- It measures how much of the task the pretrained features **already** solve

#### [day2#probe-against-fine-tuning] Training every weight bought 8 to 16 points; ten times the labels bought 1 to 3

*From the lab*

| Backbone | Method | Labels | All | Day | Glare | Dust | Rain | Night | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-18 | Linear probe | 717 | 0.81 | 0.87 | 0.80 | 0.76 | 0.66 | 0.75 | 0.4 |
| ResNet-18 | Full fine-tuning | 717 | 0.90 | 0.94 | 0.82 | 0.86 | 0.87 | 0.81 | 6.6 |
| ResNet-18 | Full fine-tuning | 7,171 | 0.92 | 0.95 | 0.89 | 0.91 | 0.90 | 0.89 | 15.0 |
| ViT-Tiny | Linear probe | 717 | 0.76 | 0.81 | 0.75 | 0.77 | 0.69 | 0.44 | 0.4 |
| ViT-Tiny | Full fine-tuning | 717 | 0.92 | 0.94 | 0.91 | 0.90 | 0.88 | 0.88 | 4.1 |
| ViT-Tiny | Full fine-tuning | 7,171 | 0.94 | 0.96 | 0.93 | 0.89 | 0.90 | 0.89 | 10.0 |

Lab 2.2, full mode: accuracy on 1,575 test crops (895 day, 219 glare, 174 dust, 183 rain, 104 night). Minutes on a laptop processor.

#### [day2#label-or-train] The table answers the budget question: label more, or train more?

*Deciding*

| Item | Value |
| --- | --- |
| Train every weight | +0.163 |
| Ten times the labels | +0.014 |

- On ViT-Tiny, from the probe on 717 labels: **train**. The compute is minutes; labels cost inspectors' time
- Then spend labels where the table is weak: glare, dust, rain, night
- A larger random sample mostly adds daylight

> **The answer belongs to this data.** On another task ten times the labels may be worth twenty points. The method transfers; the conclusion does not.

#### [day2#when-the-probe-wins] Start shallow, and go deeper only when the table says so

*Choosing a depth*

| You have | Start with | Because |
| --- | --- | --- |
| Tens of labels per class | A linear probe | A few thousand weights cannot overfit much, and it runs in seconds |
| Hundreds per class, a task close to photographs | The last block and the head | Early features already fit; fewer weights to move |
| Hundreds per class, pictures unlike the pretraining data | Full fine-tuning at a low rate | The features themselves must change: night, dust, a view from above |

> **A probe that beats full fine-tuning is a warning about the fine-tuning,** not a miracle: too few labels, too high a rate, or too many passes.

#### [day2#checkpoint-2] What does this result tell you?

*Checkpoint*

**Question:** On 150 labelled crops a linear probe scores 79%. Full fine-tuning of the same backbone on the same crops, for 30 passes at a from-scratch learning rate, scores 71%. What is the likely explanation?

- A. The backbone is unsuited to road images
- B. The fine-tuning code has a bug: it can never score below a probe
- C. There are too few labels: 2,000 more crops must come first
- D. The fine-tuning damaged the pretrained features: too high a rate, too many passes

#### [day2#lab-2-2] Probe or fine-tune

Lab 2.2 · 25 min

Train a linear probe on 10% of the labels and compare it with the supplied full fine-tuning runs on 10% and 100%.

[D2_2_probe_or_finetune.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D2_2_probe_or_finetune.ipynb)

- [ ] Freeze the backbone and extract features for 10% of the labels (6 min)
- [ ] Train the linear probe (5 min)
- [ ] Load the supplied full fine-tuning runs (4 min)
- [ ] One table: method, labels, accuracy by slice, minutes (6 min)
- [ ] Decide what to do next: label or train (4 min)
- [ ] ✱ Stretch: Unfreeze the last block only and place it in the table

### Session 3: Zero-shot baselines

#### [day2#s3] Zero-shot baselines

*Session 3*

- Explain how an image-text model classifies without task labels
- Score a zero-shot baseline on the defect classes
- Use the floor to judge whether training was worth its labels

#### [day2#images-and-text-in-one-space] An image-text model puts pictures and sentences in one space

*Open-vocabulary models*

- **How it was trained.** On hundreds of millions of pictures with their captions: an image tower and a text tower learn to give a picture and its own caption nearby vectors.
- **What that allows.** Describe a class in words, and the model can score any picture against the description. No task labels, no training.

> **Open vocabulary:** the classes are not fixed when the model is trained. You name them when you use it.

#### [day2#prompts-are-the-classifier] The classifier is a list of sentences

*Zero-shot classification*

```python
# file: lab 2.3
@torch.no_grad()
def zero_shot(prompts, image_emb):
    t = clip.encode_text(
        tokenizer(prompts).to(U.DEV)).float()
    t = (t / t.norm(dim=-1, keepdim=True)).cpu()
    return (image_emb @ t.T).argmax(dim=1).numpy()
```

1. Write one sentence per class
2. Embed the sentences and the crops, scale each vector to unit length
3. Similarity is a matrix product
4. Each crop gets the class of its most similar sentence

#### [day2#the-zero-shot-floor] With no labels at all: 51%. With careless prompts: worse than a constant answer

*From the lab*

| Item | Value |
| --- | --- |
| Bare class names | 28% |
| Always the most common class | 32% |
| Described prompts | 51% |
| Full fine-tuning, 717 labels | 94% |

Lab 2.3, quick mode: 300 test crops, five classes. Last row: the supplied ViT-Tiny run on the same crops.

#### [day2#wording-is-a-setting] The wording moved the score by 23 points without one new label

*Prompts*

| Class | True | Bare names | Described |
| --- | --- | --- | --- |
| Clear | 97 | 14 | 67 |
| Pothole | 57 | 33 | 38 |
| Longitudinal crack | 69 | 6 | 58 |
| Transverse crack | 47 | 225 | 66 |
| Patch | 30 | 22 | 71 |

**Bare:** "clear", "pothole", "longitudinal crack", "transverse crack", "patch".

**Described:** "a top-down photo of an asphalt road surface with a long vertical crack", and four more of the kind.

> **Three crops in four went to one sentence.** The bare word "clear" or "patch" fits thousands of unrelated web pictures.

#### [day2#what-a-floor-is-for] A floor tells you what your labels were worth

*Using the floor*

- **Far above the floor.** Rasid: 94% against 51%. The labels and the training earned their cost.
- **Just above it.** A few points for thousands of labels: ask whether the simple baseline, with no training to maintain, is the better system.
- **Below a constant answer.** The zero-shot model is no classifier for this task. Say so and move on.

> **The same model can still help with the data:** searching frames by a sentence, proposing labels for a person to confirm, a few-shot classifier from five labelled crops per class.

#### [day2#checkpoint-3] What is the floor?

*Checkpoint*

**Question:** An image-text model scores 28% on five defect classes with the class names as prompts. Always answering the most common class scores 32%. A colleague reports: "the floor is 28%". What do you correct?

- A. Nothing: 28% is above the 20% of chance
- B. The floor is the best answer without labels: 32% so far, and better prompts come first
- C. The floor is dropped from the report: the model is unsuited
- D. The model is fine-tuned on the defect crops, and that score is the floor

#### [day2#lab-2-3] The zero-shot floor

Lab 2.3 · 25 min

Score an open-vocabulary model on the defect classes from text prompts, compare two prompt sets and record the floor to beat.

[D2_3_zero_shot_floor.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D2_3_zero_shot_floor.ipynb)

- [ ] Write one prompt per class (5 min)
- [ ] Score the open-vocabulary model on the test crops (7 min)
- [ ] Compare two prompt sets (6 min)
- [ ] Place the floor beside the probe and the fine-tuned runs (4 min)
- [ ] Name the class the floor is worst at (3 min)
- [ ] ✱ Stretch: Build a five-shot prototype classifier from image embeddings

### Session 4: Licences of open-weight backbones

#### [day2#s4] Licences of open-weight backbones

*Session 4*

- Separate the licence of the code, of the weights and of the training data
- Read licence metadata from a model hub
- Record a deployment decision with its conditions

#### [day2#three-things-carry-terms] "Open" is not one thing: code, weights and data each carry terms

*Licences*

- **The code.** What builds the network. Usually under a permissive software licence.
- **The weights.** What runs on the roadside unit. Licensed separately by whoever trained them, and often differently.
- **The training data.** Its terms of access may reach the weights trained on it. The record names it for the legal officer.

> **The library's licence says nothing about a model's weights.** One library ships hundreds of sets of weights under a dozen different terms.

#### [day2#licence-kinds] Four kinds of terms, four different answers for the department

*Reading an identifier*

| Kind | Examples | For Al-Manar's roadside units |
| --- | --- | --- |
| **Permissive** | Apache-2.0, MIT, BSD | May be deployed; keep the notices |
| **Non-commercial** | CC-BY-NC-4.0 | Not on the units; experiments that never leave the lab |
| **Custom** | A publisher's own agreement | Wait for legal review: someone reads and accepts the text |
| **Network copyleft** | AGPL-3.0 | Needs a commercial licence, or the service's source is published |

#### [day2#read-the-metadata] Three backbones, three different answers

*From the lab*

| Backbone (name in timm) | Weights | Training data | Decision |
| --- | --- | --- | --- |
| `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | apache-2.0 | ImageNet-21k, ImageNet-1k | **Deploy** |
| `convnextv2_tiny.fcmae_ft_in22k_in1k` | cc-by-nc-4.0 | ImageNet, ImageNet-22k, ImageNet-1k | **Do not deploy** |
| `vit_small_patch16_dinov3.lvd1689m` | dinov3-license | LVD-1689M, web images collected by the publisher | **Hold for legal review** |

```python
# file: lab 2.4
cfg = timm.get_pretrained_cfg(name)
cfg.license, cfg.hf_hub_id
```

> **The tag after the dot says where the weights come from.** No weights are downloaded: the library ships a small record with each set.

#### [day2#deployment-record] A decision nobody can find is made again, differently

*The record*

```json
// file: deployment_record.json (one entry)
{
  "name": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
  "weights_licence": "apache-2.0",
  "licence_kind": "permissive",
  "training_data": ["ImageNet-21k", "ImageNet-1k"],
  "decision": "deploy",
  "checked_on": "2026-11-14",
  "checked_with": "timm 1.0.31"
}
```

- One entry per backbone **considered**, including those refused
- The date and the tool version: a publisher can change terms between versions, and a record can be corrected
- It lives in the repository, beside the model card

#### [day2#terms-travel] Terms travel with the weights, and libraries have licences too

*Two traps*

- **Fine-tuning does not wash a licence.** Weights fine-tuned from non-commercial weights are derived from them. Do not assume they are yours to deploy: choose the starting weights before any training.
- **The detection library.** Ultralytics publishes its models and code under AGPL-3.0, with an enterprise licence as the alternative. Under the department's rule: a commercial licence, or another library.

> **Run the licence check before the first experiment,** not after the best model has been found.

#### [day2#checkpoint-4] What goes into the record?

*Checkpoint*

**Question:** A backbone's record gives the weights licence as "mit". Its model page says the training dataset was released "for research purposes only". The rule lets permissive weights be deployed. What do you write?

- A. Deploy; the licence is permissive, and nothing else is recorded
- B. Do not deploy; a dataset's terms always override the weights licence
- C. The rule's decision, the dataset with its stated terms, and a flag for the legal officer
- D. Deploy; the remark is left out, since the record field is what counts

#### [day2#lab-2-4] Which backbone may ship

Lab 2.4 · 25 min

Check the licence and data terms of three backbones and record which the department may deploy on its roadside units.

[D2_4_backbone_licences.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D2_4_backbone_licences.ipynb)

- [ ] Read the licence metadata of three backbones (6 min)
- [ ] Separate code, weights and training-data terms (7 min)
- [ ] Apply the department's deployment rule (6 min)
- [ ] Write the deployment record (6 min)
- [ ] ✱ Stretch: Check the licence of a detection library the same way

#### [day2#wrap-up] Three things to keep

*Day 2 recap*

- **A table by slice, with intervals.** Two backbones differed on all crops and in daylight; on 16 night crops no test could tell them apart.
- **Probe, then floor, then train.** No labels: 0.51 with good prompts. On 717 labels the transformer's probe reached 0.76 and full fine-tuning 0.92. Ten times the labels added a point or two.
- **Check the licence first.** Of three backbones one may ship. Weights, code and data carry separate terms, and the record says what was checked and when.

#### [day2#next] Day 3 · Detect, segment and track

*Tomorrow*

A pothole detector and what its number means, its errors by type, a crack segmenter against a four-line baseline, and vehicles counted across a line.

- On Colab, keep the GPU runtime for labs 3.1 and 3.3
- Lab 3.2 is an AI-off lab: you sort fifty detector errors yourself
