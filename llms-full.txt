---
course: "Computer Vision Engineering"
code: "AE-309"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-309/"
---

# Computer Vision Engineering

Build vision systems that are measured by condition, fitted to a roadside processor and released with evidence

Participants build vision systems for a fictional municipal roads and traffic department that must keep working in glare, dust, rain and at night. On public road-damage images and synthetic traffic-camera clips they audit and extend annotations, fine-tune convolutional and transformer backbones, train a pothole detector and a crack segmenter, track and count vehicles, and fit the chosen model to the latency and memory budget of a processor-only roadside unit. Each lab ends with a test under changed lighting, blur or camera position. Vision-language models and multimodal applications are AE-408.

- **Audience:** Practitioners. Engineers who have completed the level 200 set or can show equivalent work: a deployed model or language model application they built and can explain.
- **Prerequisites:** AE-201 Machine Learning Foundations and Model Evaluation; AE-202 Deep Learning and Generative AI Fundamentals; Trains a neural network with a hand-written training loop; Computes precision and recall from a confusion matrix; Setup: Google Colab with a free GPU runtime, or a laptop with uv (every lab has a quick mode for a laptop processor). Day 2 downloads pretrained backbones from their public hubs. Run 00_setup_check before Day 1; see labs/README.md.
- **Case study:** Al-Manar Municipality Roads and Traffic Department: A fictional city department that inspects road damage from survey-vehicle cameras and counts vehicles from pole cameras, on a processor-only roadside unit that must keep working in glare, dust, rain and at night. Participants build Rasid, its inspection and counting system: they audit and extend the annotations, split by road segment, fine-tune two backbones, train a pothole detector and a crack segmenter, track and count vehicles, fit the detector to the unit's latency and size budget and test it under field conditions before a release decision. All data is synthetic and drawn with a fixed seed: 6,000 road frames in 300 segments over five districts, crack masks, traffic clips with ground-truth counts, street frames with invented faces and plates, and a 30-day stream of camera statistics.

## Outcomes

- LO1 (Applied): Prepare an image dataset with annotation guidelines, an agreement check, splits that keep each road segment in one split, and augmentations that preserve label meaning, given road images with partial labels and near-duplicate frames
- LO2 (Applied): Implement transfer learning with two pretrained backbones and compare them by condition slice with intervals, given defect crops and a free notebook GPU
- LO3 (Performance): Build a detector and a segmenter and diagnose their errors by type to choose the next improvement, given pothole boxes and crack masks
- LO4 (Applied): Measure tracking and counting accuracy for vehicles crossing a line in video, given synthetic camera clips with ground-truth counts
- LO5 (Performance): Optimise a model to a per-frame latency and memory budget on a processor through export, input resolution and quantisation, re-measuring accuracy at each step, given the roadside unit's stated budget
- LO6 (Performance): Evaluate the model under corruption and camera shift and decide whether to release it, given rain, glare, night and moved-camera test sets

## Day 1: Visual data first

Before any model: how the field breaks a classifier, what a label means, who drew it, and which frames may share a split

#### [day1#title] Visual data first

*SDAIA Academy · Computer Vision Engineering*

Day 1: Before any model: how the field breaks a classifier, what a label means, who drew it, and which frames may share a split

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-309

#### [day1#four-days-one-system] Four days, one system that reaches the roadside, and a capstone

*The course at a glance*

- **Day 1 · Visual data first.** How the field breaks a model, annotation and agreement, leakage by site, augmentation.
- **Day 2 · Backbones and transfer.** Convolutional networks and transformers, how much to fine-tune, the zero-shot floor, licences.
- **Day 3 · Detect, segment, track.** A pothole detector and its errors, a crack segmenter, vehicles counted across a line.
- **Day 4 · Fit the budget and release.** Export, latency and size on a processor, field-condition tests, privacy and monitoring.

> **Day 5 · Capstone.** An inspection or counting service that meets a stated budget, in a public repository, defended on a failure case the assessor picks.

#### [day1#how-we-work] Every lab ends with a test under changed conditions

*How this course works*

- **Data before models.** The first model is trained on Day 2. Day 1 decides what the labels mean and which frames may meet.
- **Slices and intervals.** No result is one number. It is a table by lighting, weather and district, with an interval on each cell.
- **Short schedules.** Each lab starts from a supplied checkpoint and trains for minutes, on a free notebook GPU or a laptop processor.

> **AI policy.** AI off in the supervised checks and in labs 1.2 and 3.2. Elsewhere an assistant may explain code and errors, not write the pipeline. In the capstone, anything goes.

#### [day1#al-manar-and-rasid] Al-Manar's roads department and its system, Rasid

*One case study all week*

A fictional city. Survey vehicles film the road surface, pole cameras watch the traffic. You build **Rasid** (راصد), which finds road damage and counts vehicles.

- **Road frames.** 6,000 frames from 300 road segments in five districts: potholes, cracks, patches, under five kinds of light.
- **Traffic clips.** Twelve clips with ground-truth tracks and counts: day, queue, a gantry, night.
- **A roadside unit.** A two-core processor, no accelerator, a metered link. The model must fit its budget.

#### [day1#five-kinds-of-light] One road spot, five kinds of light and a moved camera

*The data*

Figure: Left to right: day, glare, dust, rain, night, and the same spot from a pole camera. Synthetic frames from the course generator.

#### [day1#setup-check] Open Colab, or use uv on a laptop, and run the setup check

*Lab environment*

1. **Colab:** open a lab from the course page. Its first cell clones the repository and installs what Colab lacks.
2. Run `00_setup_check.ipynb`. It draws the data pack once, a few minutes.
3. Each lab starts from a **supplied checkpoint** and the supplied split. No lab needs an earlier one.

```bash
# file: terminal (laptop option)
git clone https://github.com/SDAIAAcademy/ae-309
cd computer-vision/labs
uv sync
export CV_QUICK=1        # the quick mode for a processor
uv run jupyter lab
```

#### [day1#agenda] Visual data first

*Today*

- **Session 1 · How vision systems fail in the field**: A pilot classifier that scores well in daylight fails on the road: accuracy by lighting, blur and viewpoint
- **Session 2 · Annotation guidelines and agreement**: Boxes, masks and their formats; a guideline that two people can follow; agreement as a number
- **Session 3 · Leakage by site and near-duplicates**: Frames of one road spot look alike: a random split puts them on both sides and inflates the test score
- **Session 4 · Augmentation and what it assumes**: Every augmentation claims that a change does not matter: some claims are false for a road defect

### Session 1: How vision systems fail in the field

#### [day1#s1] How vision systems fail in the field

*Session 1*

- Name the conditions a roadside camera meets that a survey set does not
- Break one accuracy number into slices by lighting, blur and viewpoint
- Write a failure note that names the worst slice and its size

#### [day1#pilot-that-passed] A classifier that passed its test and was about to ship

*The pilot*

- Trained on crops from the survey vehicle: **clean daylight, a sharp camera**
- Five classes: clear, pothole, longitudinal crack, transverse crack, patch
- Validated on held-out crops **of the same kind**
- Next step in the plan: the roadside units

**95.5%** accuracy on survey-style validation crops (lab 1.1, 156 crops)

#### [day1#the-field-number] On 200 crops as the roadside sees them, it is right less than half the time

*The same model, the field*

| Item | Value |
| --- | --- |
| Day | 88% |
| Glare | 65% |
| Night | 34% |
| Rain | 18% |
| Dust | 12% |

Lab 1.1: 200 field crops, 31 to 47 per lighting condition. Overall accuracy 43%.

#### [day1#three-ways-the-field-differs] The field differs from the survey in three ways

*Distribution shift*

- **Corruption.** The same scene, a worse picture: glare, dust, rain, night, motion blur, a dirty lens.
- **Viewpoint.** The same road from elsewhere: a pole in place of a vehicle changes scale and angle.
- **Population.** Other roads: a district with older asphalt, other markings, defects the survey never met.

> **Each has its own test.** Corruption and viewpoint can be imitated on frames you already have. A new population can only be collected.

#### [day1#slice-before-you-average] Slice before you average

*The method*

```python
# file: lab 1.1
def by_slice(rows, correct, key):
    out = {}
    for value in sorted({r[key] for r in rows}):
        hit = [c for r, c in zip(rows, correct)
               if r[key] == value]
        out[value] = (len(hit), float(np.mean(hit)))
    return out
```

- Every test item carries **metadata**: lighting, blur, viewpoint, district
- The result is **count and score** per value, never the score alone
- A small slice is a reason to collect, not a finding

#### [day1#person-test] Could a person name the defect in this crop?

*Reading errors*

- **Yes, and the model cannot.** The model has not seen the condition. Data, augmentation or fine-tuning can fix it.
- **No, nobody can.** The information is not in the picture. Only the camera, the lighting or the time of capture can fix it.

> **Look at sixteen errors of the worst slice before you plan anything.** Five minutes of looking decides which of the two you are facing.

#### [day1#failure-note] A failure note is three lines: promised, found, asked

*Reporting*

| Line | From lab 1.1 |
| --- | --- |
| **Promised** | 95.5% on survey-style crops: clean daylight, vehicle camera |
| **Found** | 43% on 200 field crops; worst slice dust, 12% on 40 crops |
| **Asked** | Do not deploy. Label crops under dust, night, rain and glare and from pole cameras; report every later model by these slices |

#### [day1#checkpoint-1] Where do you look first?

*Checkpoint*

**Question:** A crack classifier scores 94% on its validation set. On 300 crops from roadside cameras it scores 61%. What do you do first?

- A. Train a larger backbone on the same training crops
- B. Report the average of the two numbers as the expected accuracy
- C. Break the 300 crops down by lighting, blur and viewpoint, with counts
- D. Add every augmentation available and retrain

#### [day1#lab-1-1] Errors by condition

Lab 1.1 · 25 min

Run the pilot classifier on 200 field crops and sort its errors by lighting, blur and viewpoint.

[D1_1_errors_by_condition.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D1_1_errors_by_condition.ipynb)

- [ ] Load the pilot classifier and score 200 field crops (5 min)
- [ ] Accuracy by lighting, blur and viewpoint (8 min)
- [ ] Look at 16 errors of the worst slice (6 min)
- [ ] Write a three-line failure note for the head of maintenance (6 min)
- [ ] ✱ Stretch: Cross lighting with viewpoint and find the worst cell

### Session 2: Annotation guidelines and agreement

#### [day1#s2] Annotation guidelines and agreement

*Session 2*

- Read boxes and masks in a common annotation format
- Compute raw agreement and Cohen's kappa between two annotators
- Revise a guideline from the disagreements and measure again

#### [day1#what-a-label-is] A label is a class, a place and a decision someone made

*Annotation*

- **Class.** One name for the whole crop. Cheap, and enough for a classifier.
- **Box.** Four numbers around an object. What a detector learns from. A crack's box is mostly road.
- **Mask.** One decision per pixel. Needed for length and area. Many times the labelling cost of a box.

Figure: Al-Manar's five crop classes: clear, pothole, longitudinal crack, transverse crack, patch.

#### [day1#annotation-formats] The same box is written in at least three ways

*Formats*

| Convention | A box is |
| --- | --- |
| COCO | x, y of the top-left corner, width, height, in pixels |
| Pascal VOC, torchvision | x0, y0, x1, y1 of two corners, in pixels |
| YOLO text files | centre x, centre y, width, height, as fractions of the image |

```json
// file: annotations_coco.json (one box)
{"id": 0, "image_id": 0,
 "category_id": 1,
 "bbox": [131, 84, 21, 26],
 "iscrowd": 0}
```

> **A wrong convention does not raise an error.** It trains a detector on boxes that are somewhere else. Draw ten boxes on their images after every conversion.

#### [day1#new-class-alligator] The inspectors want a class the old labels never had

*A new class*

**Alligator cracking:** cracks that join into a pattern like crocodile skin. It means the road base is failing, and a patch will not hold.

In the old labels each such area was filed under one of the two crack classes, more or less at random.

Figure: Clearly alligator cracking; two cracks that cross; one closed cell. Which of the last two counts?

#### [day1#guideline-anatomy] A guideline is written so that two strangers agree

*Guidelines*

- **A definition that can be checked:** something to count or measure, not a resemblance
- **Positive and negative examples,** with the nearest wrong class beside each
- **The borderline cases named** and decided: cut by the edge, partly hidden, two classes at once
- **How to draw:** how tight a box, where a mask ends, the smallest object worth a label
- **What to do when unsure:** a flag, never a guess

#### [day1#raw-agreement-misleads] Raw agreement flatters a rare class

*Agreement*

Two annotators, 100 crops, a class that appears in about 5.

- Each says "no" to about 95
- They agree on about 90 by saying "no" alike
- **Raw agreement: 90%**, and they may not agree on one positive

- **Cohen's kappa.** kappa = (pobserved − pchance) / (1 − pchance) pchance = P(a says 1) × P(b says 1) + P(a says 0) × P(b says 0) 0 is agreement by chance, 1 is perfect agreement.

#### [day1#kappa-before-after] One rule that can be counted moved kappa from 0.44 to 0.85

*From the lab*

- **Guideline v1.** "Cracks that join into a pattern like crocodile skin." **0.44** kappa, 9 disagreements in 30 crops
- **Guideline v2.** "Interconnected cracks that enclose at least three closed cells." **0.85** kappa on the second pass

#### [day1#boxes-disagree-too] Annotators can agree on what and still disagree on where

*Label noise*

- After v2 the two annotators box the same crops
- One hugs the cracks, one leaves a margin: **mean IoU 0.71** between their boxes
- A detector cannot be more consistent with the labels than the labels are with themselves

> **Label noise has kinds.** A wrong class, a loose or shifted box, a missing object. Each one costs a different metric, and an agreement check on a sample is how you find out which you have.

#### [day1#checkpoint-2] What does a low kappa call for?

*Checkpoint*

**Question:** Two inspectors label 200 crops for "edge break". Raw agreement is 88%, Cohen's kappa is 0.31. What is the next step?

- A. Read the crops they disagree on and rewrite the definition
- B. Accept the labels, since they agree on 88% of crops
- C. Add a third inspector and take the majority vote
- D. Ask both to label the 200 crops again, more carefully

#### [day1#lab-1-2] Agree on a new class

Lab 1.2 · 25 min

Annotate 30 crops for the new class, alligator cracking, with a partner, compute agreement, revise the guideline and measure again. AI assistants stay off.

[D1_2_annotation_agreement.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D1_2_annotation_agreement.ipynb)

- [ ] Read guideline v1 and label the 30 crops (8 min)
- [ ] Raw agreement and Cohen's kappa with your partner (6 min)
- [ ] Read the disagreements and write the rule that settles them (6 min)
- [ ] Kappa on the second pass and overlap of the boxes (5 min)
- [ ] ✱ Stretch: Score each annotator against the adjudicated labels

### Session 3: Leakage by site and near-duplicates

#### [day1#s3] Leakage by site and near-duplicates

*Session 3*

- Find near-duplicate frames with a difference hash
- Split by road segment so that no segment is shared
- Report how much a leaking split inflates a score

#### [day1#same-spot-twice] The survey vehicle films every spot several times

*Near-duplicates*

Figure: Three pairs found by a difference hash in lab 1.3. Each pair is one road spot on two passes.

#### [day1#random-split-leaks] A split drawn frame by frame puts one spot on both sides

*Leakage*

**150 of 274** near-duplicate pairs straddle the inherited split (lab 1.3, 1,200 frames)

- The model is tested on pictures it has, in effect, already seen
- The test score rises, and the rise is not skill
- The better a model memorises, the more it gains

#### [day1#difference-hash] A difference hash turns a frame into 256 bits

*Finding them*

1. **1 · Shrink** grey, 17 by 16 pixels
2. **2 · Compare** is each pixel brighter than its left neighbour?
3. **3 · Count** bits that differ between two frames
4. **4 · Threshold** 30 bits or fewer: a near-duplicate

> **What it finds and what it misses.** The same spot in the same light, shifted by a pixel or two: found. The same spot at night, or shifted by a car's width: missed.

#### [day1#split-by-group] Split by road segment, and every frame of a segment moves together

*The repair*

```python
# file: lab 1.3
def split_by_segment(seg):
    return ["drop" if s is None
            else segment_split(s)
            for s in seg]
```

- The split is a function of the **segment id**, never of the frame
- It does not rely on the hash having found every duplicate
- A frame whose segment is unknown is **dropped**: it is safe on neither side

#### [day1#split-report] The split report states what is shared: nothing

*Evidence*

| Split | Frames | Segments |
| --- | --- | --- |
| Train | 780 | 44 |
| Validation | 130 | 7 |
| Test | 164 | 9 |
| Dropped, segment unknown | 126 |  |
| Segments in more than one split |  | 0 |
| Near-duplicate pairs across splits | 0 |  |

#### [day1#what-the-leak-was-worth] The leak was worth 33 points to a model that learnt nothing general

*The price*

| Item | Value |
| --- | --- |
| Inherited split | 63% |
| Split by segment | 30% |

A one-nearest-neighbour classifier on 16 by 16 grey pixels: it answers with the label of the most similar training crop. It can only memorise. (Lab 1.3, quick pack.)

> **A leaking split rewards the wrong model.** It ranks memorisers above models that generalise, so it misleads every choice made with it.

#### [day1#checkpoint-3] Why did the score fall?

*Checkpoint*

**Question:** A crack classifier scores 96% with a random split of frames and 81% when whole road segments are held out. Nothing else changed. Which number do you report to the department?

- A. 96%: a random split is the standard way to evaluate
- B. The mean of the two, about 88%
- C. Neither: the model got worse and must be retrained
- D. 81%: it is the score on roads the model has not seen

#### [day1#lab-1-3] Split by road segment

Lab 1.3 · 25 min

Find near-duplicate frames, give frames without a segment the segment of their duplicates, re-split by road segment and write the split report.

[D1_3_split_by_segment.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D1_3_split_by_segment.ipynb)

- [ ] Difference hash and near-duplicate pairs (7 min)
- [ ] Count the pairs that straddle the inherited split (4 min)
- [ ] Recover the missing segment ids from the duplicates (5 min)
- [ ] Split by segment and write the split report (5 min)
- [ ] Score one model on both test sets (4 min)
- [ ] ✱ Stretch: Find the duplicates with embeddings instead of hashes

### Session 4: Augmentation and what it assumes

#### [day1#s4] Augmentation and what it assumes

*Session 4*

- Apply one transform to an image, its boxes and its mask together
- Measure how often a transform changes what the label means
- Choose augmentations from the conditions the field will bring

#### [day1#augmentation-is-a-claim] Every augmentation claims that a change does not matter

*Augmentation*

| Transform | The claim it makes |
| --- | --- |
| Mirror left-right | The label does not depend on which side things are |
| Brightness and contrast | The label does not depend on exposure |
| Blur | The label survives a softer picture |
| Quarter turn | The label does not depend on direction |
| Tight random crop | Any part of the picture carries the label |

#### [day1#move-labels-with-pixels] The pixels, the boxes and the mask move together

*Mechanics*

```python
# file: lab 1.4
from torchvision import tv_tensors
from torchvision.transforms import v2

def apply(transform, img, mask):
    image = tv_tensors.Image(
        torch.from_numpy(img).permute(2, 0, 1))
    target = tv_tensors.Mask(torch.from_numpy(mask))
    image, target = transform(image, target)
    return (image.permute(1, 2, 0).numpy(),
            target.numpy().astype(bool))
```

- Tell the library **what each input is**: image, mask, boxes
- A geometric transform then moves all of them; a colour transform touches the image only
- The failure it prevents is silent: a flipped picture with its old boxes

#### [day1#label-change-rate] Two of six transforms change what the crop shows

*From the lab*

| Item | Value |
| --- | --- |
| Mirror left-right | 0% |
| Mirror up-down | 0% |
| Brightness, contrast | 0% |
| Gaussian blur | 0% |
| Quarter turn | 42% |
| Tight random crop | 42% |

Figure: Original, mirrored, quarter turn (now transverse), tight crop (now clear).

#### [day1#one-wrong-claim] One false claim took crack accuracy from 95% to 76%

*The cost*

| Item | Value |
| --- | --- |
| Safe pipeline | 95% |
| Quarter-turned copies | 76% |

A probe on frozen features, trained on 240 crops plus one augmented copy of each, scored on 37 validation crops of the two crack classes. (Lab 1.4, quick pack.)

> **Nothing failed loudly.** The training ran, the loss fell. Half of the copies of each crack class showed the other class under the old label.

#### [day1#choose-from-the-field] Choose augmentations from the conditions the field will bring

*Choosing*

- **Start from the failure table.** Session 1 said: dust, rain, night, blur, pole cameras. Augment towards those: exposure, haze, noise, motion blur, scale and small rotations.
- **Then check each claim.** Which label could this transform change? How often does it? Measure on a sample with masks before it enters the pipeline.

> **Imitation is not the condition.** Darkened daylight is not night. Augmentation narrows the gap; the field test on Day 4 says by how much.

#### [day1#checkpoint-4] Which transform breaks this label?

*Checkpoint*

**Question:** A new class, "faded lane marking", is defined as paint that has lost most of its brightness against the road. Which augmentation is unsafe for it?

- A. A left-right mirror
- B. Strong random brightness and contrast changes
- C. A shift of a few pixels in any direction
- D. A rotation by a few degrees

#### [day1#lab-1-4] Augment without changing labels

Lab 1.4 · 25 min

Build an augmentation pipeline for crops with masks, measure the label-change rate of six transforms and find the ones that change what a label means.

[D1_4_augmentation.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D1_4_augmentation.ipynb)

- [ ] Build the pipeline for images and masks (7 min)
- [ ] Label-change rate of six transforms (8 min)
- [ ] Keep the safe transforms, remove the two that change meaning (4 min)
- [ ] Train a probe with and without the unsafe transform (6 min)
- [ ] ✱ Stretch: Add a night transform and measure the night slice

#### [day1#wrap-up] Three things to keep

*Day 1 recap*

- **Slice before you average.** 95.5% in the survey's conditions was 43% in the field, and 12% in dust. Report by condition, with counts.
- **A label is a decision.** A rule that can be counted took kappa from 0.44 to 0.85. Disagreement is information about the guideline.
- **Keep sites together.** Split by road segment. Check what every augmentation claims about the label before it enters the pipeline.

#### [day1#next] Day 2 · Backbones and transfer

*Tomorrow*

Two families of pretrained backbone on the same crops, how much of one to train, the zero-shot floor, and which weights the department may deploy.

- On Colab, switch the runtime to a GPU before the first lab
- Lab 2.3 downloads an open-vocabulary model of about 600 MB: start it on a good connection

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

## Day 3: Detect, segment and track

A pothole detector and what its number means, its errors by type, a crack segmenter against a simple baseline, and vehicles counted across a line

#### [day3#title] Detect, segment and track

*SDAIA Academy · Computer Vision Engineering*

Day 3: A pothole detector and what its number means, its errors by type, a crack segmenter against a simple baseline, and vehicles counted across a line

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-309

#### [day3#agenda] Detect, segment and track

*Today*

- **Session 1 · Detection and its metrics**: Boxes, overlap, non-maximum suppression and average precision at more than one threshold
- **Session 2 · Detector error analysis**: A detector is wrong in four ways, and each way has a different fix
- **Session 3 · Segmenting cracks**: Masks, overlap metrics and why thin structures are hard; a baseline that needs no training
- **Session 4 · Tracking and counting**: Tracking by detection: association, identity switches and a counting rule that counts once

#### [day3#review-day-2] May this model be deployed?

*Review of Day 2*

**Question:** A colleague fine-tuned weights under a non-commercial licence on the department's own crops. It is the best model in the table. A contractor will run the units under a paid contract. What happens next?

- A. Deploy it: fine-tuning on its own data made the weights the department's
- B. Do not deploy it: record that, and compare again from weights the rule allows
- C. Deploy it: the library is under Apache-2.0
- D. Deploy it as a pilot: a pilot is not commercial use

### Session 1: Detection and its metrics

#### [day3#s1] Detection and its metrics

*Session 1*

- Explain how a detector turns score maps into boxes
- Apply non-maximum suppression and say what it removes
- Compute average precision at two overlap thresholds

#### [day3#what-a-detector-returns] A detector answers three questions for every object at once

*Detection*

- **Where.** A box: two corners in pixels of the frame.
- **What.** A class: for Rasid, pothole or patch.
- **How sure.** A score from 0 to 1. Everything downstream depends on where you cut it.

> **Why patch is a class.** A repaired pothole is the thing a pothole detector most often takes for a pothole. Teaching the detector to name it is cheaper than filtering it afterwards.

#### [day3#score-maps-to-boxes] Every cell of a grid votes for a class and draws a box

*How it works*

1. **1 · Trunk** ResNet-18 turns a 192-pixel frame into a 24 by 24 grid of features
2. **2 · Score map** per cell and class: is the centre of an object here?
3. **3 · Box map** per cell: distances to the left, top, right and bottom edge
4. **4 · Decode** keep cells above a score, turn distances into corners

> **Neighbouring cells see the same pothole.** Each draws its own box for it. The raw output holds several boxes per object.

#### [day3#iou-is-the-ruler] Intersection over union is the ruler for "the same box"

*Overlap*

- **IoU of two boxes.** IoU = area of overlap / area covered by either Identical boxes: 1. Boxes that do not touch: 0.

| A 40 by 40 box against the truth | IoU |
| --- | --- |
| Shifted by 3 pixels each way | 0.75 |
| Shifted by 5 pixels each way | 0.62 |
| Shifted by 8 pixels each way | 0.47 |
| Same centre, 6 pixels of margin all round | 0.59 |

> **Two common thresholds.** At 0.5 a box counts when it is roughly on the object. At 0.75 it has to fit.

#### [day3#non-maximum-suppression] Non-maximum suppression keeps the best box and drops its echoes

*Suppression*

Figure: Top: raw boxes above a score of 0.3. Bottom: after suppression at IoU 0.5.

1. Sort the boxes of one class by score
2. Keep the best one
3. Drop every other box that overlaps it by more than a threshold
4. Repeat with what is left

#### [day3#precision-recall-per-threshold] Sort the detections by score and walk down the list

*The metric*

| Score | Counts as | Precision | Recall |
| --- | --- | --- | --- |
| 0.9 | true positive | 1/1 | 1/2 |
| 0.8 | false positive | 1/2 | 1/2 |
| 0.7 | true positive | 2/3 | 2/2 |

- Two potholes exist. Each row is the result if the score threshold sat **just there**
- A detection is a true positive when it overlaps a **still unclaimed** truth box by at least the IoU threshold
- **Average precision** is the area under precision against recall: here 0.5 × 1 + 0.5 × 2/3 = 0.83

#### [day3#ap-at-two-thresholds] A short schedule moved AP at 0.75 far more than AP at 0.5

*From the lab*

| Mean over pothole and patch | AP at IoU 0.5 | AP at IoU 0.75 |
| --- | --- | --- |
| Warm checkpoint | 0.871 | 0.306 |
| After two passes over 400 frames | 0.894 | 0.483 |
| Supplied detector, full schedule | 0.968 | 0.785 |

> **Report both.** AP at 0.5 says the detector finds the objects. AP at 0.75 says its boxes fit. The gap between them names the next problem.

#### [day3#checkpoint-1] Which detector for this use?

*Checkpoint*

**Question:** Detector A scores an AP of 0.90 at IoU 0.5 and 0.70 at IoU 0.75. Detector B scores 0.93 and 0.45. The department will use the boxes to estimate how much asphalt each repair needs. Which do you choose?

- A. B: it finds more potholes, and finding them is what counts
- B. B: AP at IoU 0.5 is the standard metric for detectors
- C. A: it has the higher score at both thresholds
- D. A: its boxes fit the potholes, and area is read from the box

#### [day3#lab-3-1] Train and score a detector

Lab 3.1 · 25 min

Train the pothole detector on a short schedule from a supplied checkpoint and compute average precision at two overlap thresholds.

[D3_1_detector_ap.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D3_1_detector_ap.ipynb)

- [ ] Load the warm checkpoint and look at detections before and after suppression (5 min)
- [ ] Train a short schedule (7 min)
- [ ] Write average precision from sorted detections (8 min)
- [ ] Average precision at 0.5 and 0.75, before and after (5 min)
- [ ] ✱ Stretch: Sweep the suppression threshold and plot average precision

### Session 2: Detector error analysis

#### [day3#s2] Detector error analysis

*Session 2*

- Sort detector errors into localisation, class, duplicate and missed
- Estimate what fixing each error type would be worth
- Justify the next improvement from the error breakdown

#### [day3#a-number-is-not-a-plan] An average precision says how good, never what to do next

*After the metric*

**41** false alarms beside 148 correct boxes and 10 missed defects, on 180 field frames with the prototype's settings (lab 3.2)

- Precision 0.78, recall 0.94: the crews drive to one wrong address in five
- More data? A bigger model? A setting?
- Each answer costs something different, and only the errors themselves can choose

#### [day3#four-error-types] Every detector error is one of a few types

*Types*

Figure: Red: the box in question. Teal: ground truth. Yellow: other detections. Left to right: localisation, class, duplicate, missed.

| Localisation | Class | Duplicate | Missed |
| --- | --- | --- | --- |
| Right class, IoU from 0.1 to 0.5 | IoU at least 0.5, wrong class | Right and fitting, but a better box holds the object | A truth box no detection came near |

#### [day3#type-points-to-fix] Each type points to a different piece of work

*From type to work*

| Type | Usual fix | What it costs |
| --- | --- | --- |
| Duplicate | Tighten non-maximum suppression | A setting, no training |
| Class | More examples of the confused pair, or a clearer definition | Labelling |
| Localisation | A box guideline everyone follows, a finer grid, a box loss | Relabelling or retraining |
| Missed | Data of the missed kind: small, dark, in rain. Or a lower score threshold | Collection and labelling |

#### [day3#fifty-errors-sorted] Fifty supplied errors, sorted by one rule

*From the lab*

| Item | Value |
| --- | --- |
| Missed | 18 |
| Localisation | 14 |
| Duplicate | 10 |
| Class | 8 |

```python
# file: lab 3.2
iou = U.box_iou([e["box"]], gts[:, :4])[0]
j = int(iou.argmax())
same = CLASSES[int(gts[j, 4])] == e["cls"]
if iou[j] >= 0.5 and not same:
    return "class"
if iou[j] >= 0.5:
    return "duplicate"
if iou[j] >= 0.1 and same:
    return "localisation"
return "background"
```

> **Fifty hand-picked errors say what exists, not what matters.** Someone chose them. The bars are that person's sample.

#### [day3#oracle-fix] Ask an oracle what each fix could be worth

*What matters*

| All errors of one type repaired | Precision | Recall | Gain in F1 |
| --- | --- | --- | --- |
| As configured | 0.783 | 0.937 |  |
| Duplicate | 0.897 | 0.937 | +0.063 |
| Missed | 0.789 | 0.968 | +0.016 |
| Class | 0.783 | 0.937 | 0.000 |
| Localisation | 0.783 | 0.937 | 0.000 |

> **The largest gain is also the cheapest fix:** suppression set at 0.7 on the prototype lets echoes through. Set it to 0.5, then measure again.

#### [day3#justified-next-step] A justified next step is one sentence with a number and a cost

*Reporting*

- **Not this.** "The detector needs more training data, especially at night."
- **This.** "Fixing duplicates is worth up to +0.063 F1, the largest gain in the oracle table, and costs one setting: suppression from 0.7 to 0.5. Missed objects are next, at +0.016, and need labelled night and rain frames."

> **This is the evidence for outcome 3:** the metrics, the error breakdown, and a next step someone can approve or refuse.

#### [day3#checkpoint-2] Which fix do you fund first?

*Checkpoint*

**Question:** An oracle table for a crack-sealing detector shows: localisation errors repaired +0.09 F1, missed +0.07, class +0.01, duplicate +0.01. The team's boxes come from three annotators with no rule on tightness. What do you propose first?

- A. Write the box-tightness rule, relabel a sample and measure localisation again
- B. Tighten non-maximum suppression, because it needs no training
- C. Train a larger detector on the same labels
- D. Lower the score threshold so that fewer objects are missed

#### [day3#lab-3-2] Sort fifty errors

Lab 3.2 · 25 min

Sort 50 supplied detector errors into localisation, class, duplicate and missed, and choose a fix. AI assistants stay off.

[D3_2_error_analysis.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D3_2_error_analysis.ipynb)

- [ ] Sort 12 errors by eye (6 min)
- [ ] Write the rule that types an error (8 min)
- [ ] Type all 50 and count (4 min)
- [ ] What fixing each type would be worth (4 min)
- [ ] Choose a fix and justify it (3 min)
- [ ] ✱ Stretch: Break the missed defects down by lighting

### Session 3: Segmenting cracks

#### [day3#s3] Segmenting cracks

*Session 3*

- Compute intersection over union and Dice for masks
- Fine-tune a segmenter on a short schedule
- Compare a segmenter with a thresholding baseline by condition

#### [day3#a-box-cannot-measure-a-crack] A box cannot measure a crack; a mask can

*Segmentation*

- **Semantic segmentation.** One class per pixel: crack or road. Enough for length and area. What Rasid needs.
- **Instance segmentation.** A separate mask per object: this crack and that crack. Needed to count or follow objects.
- **The cost.** Masks are the most expensive labels to draw, and thin structures are the hardest to draw consistently.

> **In Al-Manar's frames crack pixels are 1.1% of all pixels.** A model that answers "road" everywhere is 98.9% accurate and useless.

#### [day3#encoder-decoder] Down to understand, back up to draw

*How it works*

1. **1 · Encoder** the same ResNet-18 trunk shrinks the frame to one eighth
2. **2 · Decoder** upsamples step by step back to full size
3. **3 · Skip connections** hand the fine detail of early layers across
4. **4 · One score per pixel** cut at 0.5 for the mask

> **A crack two pixels wide does not exist at one-eighth resolution.** Without the skip connections the decoder would have to guess where it was.

#### [day3#iou-and-dice] IoU and Dice count the same three things differently

*Overlap metrics*

- **From pixels in both, in the prediction, in the truth.** IoU = both / (prediction + truth − both) Dice = 2 × both / (prediction + truth) Dice = 2 × IoU / (1 + IoU), so Dice is always the larger.

- Both ignore the pixels that are road in both masks, which is nearly all of them
- An IoU of 0.38 is a Dice of 0.55: **say which one you report**
- A predicted line one pixel beside a two-pixel crack loses half its overlap

#### [day3#dark-is-not-crack] Four lines of thresholding are the number to beat

*The baseline*

```python
# file: lab 3.3
def threshold_baseline(img, k=2.0):
    grey = img.astype(np.float32).mean(axis=-1)
    return grey < grey.mean() - k * grey.std()
```

Figure: Columns: frame, truth, baseline, supplied segmenter. Rows: day, rain, night.

> **"Dark" is not "crack".** The rule also marks potholes, patch seams and shadows, and it finds nothing when the whole frame is dark.

#### [day3#segmenter-against-baseline] The baseline fails where the light changes; the segmenter holds

*From the lab*

| IoU on validation frames | All | Day | Glare | Dust | Rain | Night |
| --- | --- | --- | --- | --- | --- | --- |
| Frames | 80 | 37 | 13 | 13 | 11 | 6 |
| Thresholding baseline | 0.224 | 0.266 | 0.072 | 0.215 | 0.258 | 0.008 |
| Segmenter, three passes over 240 frames | 0.378 | 0.408 | 0.385 | 0.360 | 0.368 | 0.294 |
| Supplied segmenter, full schedule | 0.809 | 0.861 | 0.740 | 0.824 | 0.842 | 0.618 |

> **Six night frames.** The night column is the weakest of every row and the least certain. It is the next data request, not a finding.

#### [day3#reading-an-overlap-score] An overlap score needs its object beside it

*Reading the number*

- **The wrong comparison.** "The segmenter's IoU is 0.6 and the detector's AP is 0.9, so the segmenter is the weak part." Two metrics on two tasks share no scale.
- **The right ones.** Against the baseline on the same frames. Against the same model under another light. With a tolerance of two pixels. And by looking at twenty masks.

> **For the maintenance team, convert:** metres of crack found against metres present is a number they can plan with. IoU is the engineer's number.

#### [day3#checkpoint-3] Why is the overlap low?

*Checkpoint*

**Question:** A crack segmenter scores an IoU of 0.52. Its masks, overlaid on the frames, follow every crack and look about one pixel wider than the labels. What do you report?

- A. The model is poor and needs a larger backbone
- B. Pixel accuracy, which is 99% and describes the model better
- C. The IoU with the object's width, and precision and recall at a two-pixel tolerance
- D. Dice, because it is higher for the same masks

#### [day3#lab-3-3] Segment cracks

Lab 3.3 · 25 min

Fine-tune the crack segmenter on a short schedule and compare its overlap score with a thresholding baseline.

[D3_3_segment_cracks.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D3_3_segment_cracks.ipynb)

- [ ] A thresholding baseline and its overlap score (6 min)
- [ ] Fine-tune the segmenter from the warm checkpoint (8 min)
- [ ] IoU and Dice against the baseline (5 min)
- [ ] Compare the two by lighting (6 min)
- [ ] ✱ Stretch: Score with a two-pixel tolerance and explain the change

### Session 4: Tracking and counting

#### [day3#s4] Tracking and counting

*Session 4*

- Associate detections to tracks by overlap
- Count line crossings once per track
- Measure count error and identity switches against ground truth

#### [day3#counting-needs-identity] To count vehicles, you have to know which box is the same vehicle

*Video*

Figure: Al-Manar's four kinds of clip: day, queue, gantry, night. The yellow line is where vehicles are counted.

#### [day3#tracking-by-detection] Tracking by detection: link each new box to the track it overlaps most

*How it works*

```python
# file: lab 3.4
def associate(track_boxes, det_boxes, iou_thr=0.3):
    iou = U.box_iou(track_boxes, det_boxes)
    pairs = []
    while True:
        t, d = np.unravel_index(iou.argmax(), iou.shape)
        if iou[t, d] < iou_thr:
            return pairs
        pairs.append((int(t), int(d)))
        iou[t, :] = -1
        iou[:, d] = -1
```

- Every open track remembers its last box
- A matched detection **continues** its track
- An unmatched detection **starts** a new track
- An unmatched track **waits**, then closes

#### [day3#identity-switches] One missed detection, and a vehicle becomes two

*The failure*

**49** tracks for 16 vehicles in a daytime clip, with a tracker that has no patience (lab 3.4)

- The detector misses a vehicle for one frame; its track closes
- The next detection opens a new track with a new identity
- An **identity switch**: the track that covers a true vehicle changes from one frame to the next

> **Two measures, two questions.** Count error asks whether the total is right. Identity switches ask whether each vehicle stayed one vehicle.

#### [day3#counting-rule] Count a track once, the first time its centre crosses the line

*Counting rules*

- **Every crossing.** Add one whenever a box centre moves from above the line to below it. **14** counted in a queue clip with 13 true crossings
- **Once per track.** A track counts at its first crossing and never again. **13** counted in the same clip

#### [day3#patience-and-hits] Two settings removed nearly every count error

*From the lab*

| Kind of clip | True crossings | Patience 0, minimum hits 1 | Patience 5, minimum hits 4 |  |  |
| --- | --- | --- | --- | --- | --- |
| Count error | Switches | Count error | Switches |  |  |
| Day | 38 | 3 | 67 | 0 | 1 |
| Queue | 39 | 5 | 106 | 0 | 18 |
| Gantry | 39 | 4 | 105 | 0 | 40 |
| Night | 35 | 14 | 172 | 1 | 6 |

> **Patience** lets a track survive missed frames. **Minimum hits** ignores tracks that live two or three frames: the headlight reflections.

#### [day3#right-count-wrong-tracks] A right count with broken tracks is ready for counting and for nothing else

*What the numbers allow*

- **Needs the count only.** Vehicles per hour across one line. The gantry clips count correctly because the line lies beyond the gantry.
- **Needs identities.** Speed between two lines, turning movements, time spent waiting. Forty switches make each of these wrong.

> **Where the line is drawn is a design decision.** Put it where the camera sees clearly. Move the gantry over the line and the counts drop.

#### [day3#checkpoint-4] Is this counter ready?

*Checkpoint*

**Question:** On a new camera the tracker reports 52 crossings against 50 true ones, with 3 identity switches. The clips are at dusk, and short tracks appear beside some vehicles. Which change do you try first?

- A. Require more hits before a track is counted
- B. Raise the patience so that tracks survive longer gaps
- C. Count every crossing of the line, not one per track
- D. Retrain the detector on dusk frames before changing anything

#### [day3#lab-3-4] Track and count

Lab 3.4 · 25 min

Track and count vehicles across a line in synthetic clips and measure count error and identity switches against the labels.

[D3_4_track_count.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D3_4_track_count.ipynb)

- [ ] Associate detections to tracks by overlap (8 min)
- [ ] Count each track once when it crosses the line (5 min)
- [ ] Count error per clip (4 min)
- [ ] Identity switches against the ground-truth tracks (4 min)
- [ ] Tune patience and minimum hits on the gantry and night clips (4 min)
- [ ] ✱ Stretch: Predict each track forward at constant velocity

#### [day3#wrap-up] Three things to keep

*Day 3 recap*

- **Two thresholds, two messages.** AP at 0.5 says the objects are found, AP at 0.75 says the boxes fit. Rasid's short schedule: 0.89 and 0.48.
- **Type the errors, then price the fixes.** Fifty errors showed mostly misses. The oracle table said duplicates: +0.063 F1 for one setting.
- **Beat the cheap baseline, name the metric.** Thresholding 0.22, the segmenter 0.81 IoU. A count can be right with 40 identity switches: say what it is fit for.

#### [day3#next] Day 4 · Fit the budget and release

*Tomorrow*

The detector leaves the training framework, meets the roadside unit's budget, faces rain, glare, night and a moved camera, and gets a release decision.

- Day 4 runs on a laptop processor: no GPU runtime is needed
- Lab 4.2 measures latency: close other programs before you run it
- Think about your capstone: an inspection service or a counting service

## Day 4: Fit the budget and release

Export, the latency and size budget of the roadside unit, tests under rain, glare, night and a moved camera, privacy before storage, and the release decision

#### [day4#title] Fit the budget and release

*SDAIA Academy · Computer Vision Engineering*

Day 4: Export, the latency and size budget of the roadside unit, tests under rain, glare, night and a moved camera, privacy before storage, and the release decision

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-309

#### [day4#agenda] Fit the budget and release

*Today*

- **Session 1 · Export to an interchange format**: A model leaves the training framework as a graph: export it, then prove the copy behaves like the original
- **Session 2 · Resolution and precision against latency**: Resolution, batch size and numeric precision trade accuracy for latency and memory: measure, then choose
- **Session 3 · Corruption and shift tests**: The test that decides a release: the same frames under rain, glare, night and from a moved camera
- **Session 4 · Privacy and monitoring in the field**: Faces and plates are personal data: blur before storage, watch the cameras for drift, document the model

#### [day4#review-day-3] What do you do with this error?

*Review of Day 3*

**Question:** On a rain frame the detector puts two pothole boxes on one pothole: one with a score of 0.84 and an IoU of 0.81 with the truth, one with 0.61 and 0.66. How is the second box counted, and what removes it?

- A. A true positive; nothing needs removing
- B. A localisation error; a box guideline removes it
- C. A duplicate, counted as a false positive; tighter suppression removes it
- D. A background error; more rain frames remove it

### Session 1: Export to an interchange format

#### [day4#s1] Export to an interchange format

*Session 1*

- Export a model to an interchange format with a dynamic input size
- Check that the runtime's outputs match the original within a tolerance
- Find a preprocessing mismatch between training and serving

#### [day4#leave-the-framework] The model leaves the framework it was trained in

*Deployment*

- **Where it was trained.** Python, a training framework, a notebook with gigabytes of libraries, a graphics processor when one is free.
- **Where it will run.** Al-Manar's roadside unit: two processor cores, no accelerator, a metered link, a small disk, one frame at a time.

> **An interchange format carries the model across:** the network as a graph of standard operations plus its weights, in one file that an inference runtime can execute.

#### [day4#what-export-records] Export runs the model once and writes down what happened

*Export*

```python
# file: lab 4.1
example = torch.rand(1, 3, 192, 192)
dynamic_axes = {
    "image": {0: "batch", 2: "height", 3: "width"},
    "cls": {0: "batch", 2: "h", 3: "w"},
    "box": {0: "batch", 2: "h", 3: "w"},
}
torch.onnx.export(model.eval(), example, path,
                  input_names=["image"],
                  output_names=["cls", "box"],
                  dynamic_axes=dynamic_axes,
                  opset_version=17, dynamo=False)
```

- The exporter follows one example input through the network
- By default **every size of that example is frozen** into the graph
- Name the axes that must stay free: here batch, height and width
- `model.eval()` first: the outputs you compare against must be inference outputs too

#### [day4#parity-within-tolerance] A copy is proven, not assumed: compare outputs within a stated tolerance

*Parity*

**0.00002** largest difference between the original's and the runtime's raw outputs on eight frames, against a tolerance of 0.001 (lab 4.1)

- The outputs are **never bit-identical**: the runtime fuses and reorders operations, and float arithmetic depends on the order
- State the tolerance before you look
- Orders of magnitude above rounding means something was exported differently

#### [day4#compare-the-product] Then compare what the user receives: the boxes

*Parity, second level*

1. **1 · Raw outputs** score and box maps agree within the tolerance
2. **2 · Final boxes** same count, same class, IoU of at least 0.99 with a partner
3. **3 · Kept as a test** ten frames and their boxes, run on every export

> **From the lab:** 30 boxes from the original, 30 from the export, parity 1.000. A tiny difference in a score can move a box across a threshold, so the raw outputs alone do not close the question.

#### [day4#preprocessing-mismatch] The file was right and the prototype still saw potholes everywhere

*The planted fault*

```python
# file: the roadside prototype
def serve_prototype(sess, frame_bgr):
    x = frame_bgr.astype(np.float32)[None]
    x = x.transpose(0, 3, 1, 2)
    cls, box = sess.run(None, {"image": x})
    return U.decode(cls, box, score_thr=0.3)[0]
```

**816** boxes from the prototype on 32 frames where the original reports 30; parity 0.000

- The camera library delivers **BGR**; the model was trained on RGB
- Values arrive as **0 to 255**; the model expects 0 to 1

> **No exception, no warning.** A plausible list of boxes, only a wrong one. Most deployment bugs in vision live in the lines before the model.

#### [day4#inside-and-outside-the-graph] Write down what the graph does and what the caller must do

*The contract*

| Step | Rasid | Who can get it wrong |
| --- | --- | --- |
| Channel order | RGB, outside the graph | The serving code |
| Value range | 0 to 1, outside the graph | The serving code |
| Resize | Outside the graph, bilinear | The serving code |
| Mean and standard deviation | Inside the graph | Nobody |
| Decoding and suppression | Outside the graph, shared code | Whoever copies the settings |

#### [day4#checkpoint-1] The parity test fails the day before release

*Checkpoint*

**Question:** After a runtime upgrade the stored parity test reports a largest difference of 0.004 against a tolerance of 0.001, and 2 of 30 boxes have no partner. The model file did not change. What do you do?

- A. Raise the tolerance to 0.01, since 28 of 30 boxes still agree
- B. Release with the proven versions, and find what changed before upgrading
- C. Release with the upgrade: differences this small are float rounding
- D. Retrain the detector with the upgraded library installed

#### [day4#lab-4-1] Export and check parity

Lab 4.1 · 25 min

Export the supplied detector to ONNX and check that its outputs match the original within tolerance.

[D4_1_export_parity.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D4_1_export_parity.ipynb)

- [ ] Export the detector with a dynamic input size (6 min)
- [ ] Run it in the runtime and compare the raw outputs (6 min)
- [ ] Compare the final boxes after decoding (6 min)
- [ ] Find and fix the planted preprocessing mismatch (7 min)
- [ ] ✱ Stretch: Check parity at a second input size

### Session 2: Resolution and precision against latency

#### [day4#s2] Resolution and precision against latency

*Session 2*

- Measure per-frame latency as a median and a 95th percentile
- Quantise a model to 8-bit integers and re-measure its accuracy
- Choose an operating point that meets a stated budget

#### [day4#the-units-budget] The roadside unit states its budget before any model exists

*The budget*

- **60 ms** per frame at the 95th percentile, on the unit
- **2 MB** of model file, sent over a metered link
- **2** processor threads, no accelerator

> **The case's assumption:** the unit takes three times as long per frame as the reference laptop. Your table says which settings are worth carrying to the unit, where the real measurement is made.

#### [day4#measure-the-tail] Measure the median and the 95th percentile, after a warm-up

*Measuring*

```python
# file: lab 4.2
def summarise(ms):
    return (float(np.median(ms)),
            float(np.percentile(ms, 95)))

x = U.to_input(Xv[:1], size)   # one frame
ms = U.latency_ms(
    lambda: sess.run(None, {"image": x}), n=100)
```

- One frame at a time, as the camera delivers them
- The first runs are slow: discard them
- A unit that is fast on average and slow on one frame in twenty **drops that frame**
- Name the machine and the number of threads beside every latency

#### [day4#three-levers] Three levers trade accuracy for latency and memory

*The levers*

| Lever | What it buys | What it can cost |
| --- | --- | --- |
| **Input resolution** | Work falls with the number of pixels | Small objects shrink below what the grid can see |
| **Numeric precision** | 8-bit integers: a quarter of the file, faster integer arithmetic | Every number moves a little; some models lose accuracy |
| **Batch size** | More frames per second in total | Latency per frame: the first frame waits for the last |

> **Re-measure accuracy after every change.** No rule of thumb says which model loses how much. Only the same validation frames, run again, do.

#### [day4#quantisation-needs-calibration] Quantisation maps floats to 8-bit integers, and has to see real frames to do it

*Quantisation*

1. **1 · Float model** weights and activations as 32-bit floats: 3.53 MB
2. **2 · Calibrate** run 32 training frames and record the range of every activation
3. **3 · Quantise** store and compute in 8-bit integers: 0.93 MB
4. **4 · Re-measure** parity is gone by design: measure accuracy again

> **Calibration frames come from the training split and look like the field.** Calibrate on daylight only, and the ranges are wrong at night.

#### [day4#benchmark-table] The benchmark table: one row per operating point

*From the lab*

| Precision | Input | Median ms | p95 ms | On the unit, p95 ms | File MB | AP at 0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| Float, 32-bit | 192 by 192 | 25.3 | 35.0 | 105.1 | 3.53 | 0.973 |
| Float, 32-bit | 160 by 160 | 18.0 | 28.5 | 85.6 | 3.53 | 0.973 |
| Float, 32-bit | 128 by 128 | 10.1 | 15.7 | 47.2 | 3.53 | 0.921 |
| Integer, 8-bit | 192 by 192 | 8.7 | 41.7 | 125.0 | 0.93 | 0.973 |
| Integer, 8-bit | 160 by 160 | 8.3 | 10.2 | 30.5 | 0.93 | 0.973 |
| Integer, 8-bit | 128 by 128 | 6.9 | 10.3 | 31.0 | 0.93 | 0.919 |

Lab 4.2, quick mode: 120 validation frames, two threads, one laptop processor. Unit latency is the laptop's times three, the case's assumption.

#### [day4#choose-the-operating-point] The operating point is the most accurate row that meets every limit

*Deciding*

```python
# file: lab 4.2
def choose(table):
    ok = [r for r in table
          if r["unit_p95_ms"] <= BUDGET_MS
          and r["file_mb"] <= BUDGET_MB]
    return max(ok, key=lambda r: r["ap50"]) \
        if ok else None
```

- Limits first, accuracy second: a row over the limit is not a candidate
- From this run: **8-bit at 160 by 160**, AP 0.973, about 31 ms on the unit, 0.93 MB
- **No row qualifies** is a result too: report it with the nearest row and what would close the gap

> **This table and the chosen row are the evidence for outcome 5.** The capstone asks for the same against the budget you state, with the machine named.

#### [day4#checkpoint-2] Does it fit?

*Checkpoint*

**Question:** The budget is 50 ms per frame at the 95th percentile. On the unit, model A has a median of 31 ms and a 95th percentile of 64 ms; model B 38 ms and 46 ms. Model A is one point more accurate. Which do you deploy?

- A. Model A: its median is lower and it is more accurate
- B. Model A: the mean of its two numbers is under 50 ms
- C. Neither, until both are measured again with batch size 8
- D. Model B: it is the only one inside the budget

#### [day4#lab-4-2] Fit the latency budget

Lab 4.2 · 25 min

Measure latency and accuracy at three resolutions and two precisions on a laptop processor and choose the operating point for the roadside unit.

[D4_2_latency_budget.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D4_2_latency_budget.ipynb)

- [ ] Latency at three resolutions: median and 95th percentile (6 min)
- [ ] Average precision at each resolution (5 min)
- [ ] Quantise to 8-bit integers with calibration frames (6 min)
- [ ] The benchmark table: latency, size, accuracy (4 min)
- [ ] Choose the operating point for the unit's budget (4 min)
- [ ] ✱ Stretch: Repeat the latency runs with one thread

### Session 3: Corruption and shift tests

#### [day4#s3] Corruption and shift tests

*Session 3*

- Build paired test sets for corruption and camera shift
- Report the drop per condition with an interval
- Decide to release, release with conditions or hold

#### [day4#back-to-day-1] The pilot fell from 95% to 43% in the field. Will this model?

*The question from Day 1*

- **What the pilot did.** Trained on clean daylight, tested on clean daylight, and met dust, rain and night for the first time on the roadside.
- **What changes today.** The field conditions are met in a test, on purpose, before the release. The test decides, and its rule is written first.

> **Three kinds of shift, from Day 1:** corruption (rain, glare, night), viewpoint (a moved camera), population (another district). The first two can be imitated on frames you hold.

#### [day4#paired-test-sets] Paired sets: the same road spots under each condition

*The test sets*

Figure: Rows: clean, rain, glare, night, moved camera.

- Each column is one test frame, drawn again: frame 17 of the rain set is frame 17 of the clean set, in the rain
- Each spot is **its own control**: what differs is the condition, not which potholes were drawn
- Only **test-split** segments, at a stated severity
- Find the frame where the box is gone

#### [day4#paired-drop-with-interval] Report the drop per condition, with an interval on the drop

*The statistic*

```python
# file: lab 4.3
r = np.random.default_rng(seed)
vals = []
for _ in range(n_boot):
    idx = r.integers(0, n, n)   # frame numbers
    vals.append(
        U.mean_ap([pc[i] for i in idx],
                  [gc[i] for i in idx], 0.5)
        - U.mean_ap([pf[i] for i in idx],
                    [gf[i] for i in idx], 0.5))
low, high = np.percentile(vals, [2.5, 97.5])
```

- Resample the **frames**, and use the same drawn frames for the clean and the field score
- Keep the difference each time
- The spread of the differences is the interval of the drop
- Pairing makes it narrower than two separate intervals would be

#### [day4#drops-by-condition] Night costs the most, and its interval is the widest

*From the lab*

| Condition | AP at 0.5 | Drop | 95% interval of the drop |
| --- | --- | --- | --- |
| Clean | 1.000 |  |  |
| Glare | 1.000 | 0.000 | 0.000 to 0.000 |
| Rain | 0.987 | 0.013 | 0.001 to 0.035 |
| Moved camera | 0.987 | 0.013 | 0.000 to 0.042 |
| Night | 0.924 | 0.076 | 0.022 to 0.159 |

Lab 4.3, quick mode: the supplied detector on 90 paired test frames, 120 resamples.

#### [day4#rule-before-result] The release rule is written before the test is run

*The decision*

| Upper end of the drop interval | Decision | From the lab |
| --- | --- | --- |
| At most 0.10 | **Pass** | Glare, rain, moved camera |
| At most 0.30 | **Release with a human check:** an inspector confirms each detection before a crew is sent | Night (upper end 0.159) |
| Above 0.30 | **Hold:** the detector does not run under this condition | None |

> **The rule uses the upper end, not the estimate.** A condition tested on few frames has a wide interval and fails until someone collects more. That is the rule working.

#### [day4#release-is-a-list] "Release" is rarely one word: it is a list of conditions

*The memo*

- **The release memo, one page.** The model and its version. The test: 90 paired frames, clean AP 1.000. One row per condition: drop, interval, decision. What happens under each decision, and who does it.
- **What the test did not cover.** Dust. Motion blur. Districts one by one. Conditions together, such as rain at night. Real cameras. Say so in the memo.

> **Add districts as rows.** Results by lighting, weather and district, so that no part of the city is served worse unnoticed.

#### [day4#checkpoint-3] Release, check or hold?

*Checkpoint*

**Question:** The rule: pass when the upper end of the drop interval is at most 0.10, human check up to 0.30, otherwise hold. In fog the detector's drop is 0.08 with an interval from 0.01 to 0.33, measured on 25 frames. What is the decision?

- A. Hold in fog, and collect more fog frames to narrow the interval
- B. Pass: the drop of 0.08 is below 0.10
- C. Release with a human check: the truth is probably in between
- D. Pass: with only 25 frames the interval is too wide to count

#### [day4#lab-4-3] Field-condition tests

Lab 4.3 · 25 min

Run the detector on rain, glare, night and moved-camera sets, report the drop for each and decide on release.

[D4_3_field_tests.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D4_3_field_tests.ipynb)

- [ ] Average precision on the clean test frames (4 min)
- [ ] Average precision under rain, glare, night and a moved camera (8 min)
- [ ] The drop per condition with bootstrap intervals (7 min)
- [ ] The release decision against the department's rule (6 min)
- [ ] ✱ Stretch: Sweep the severity of the worst condition

### Session 4: Privacy and monitoring in the field

#### [day4#s4] Privacy and monitoring in the field

*Session 4*

- Blur faces and plates before a frame is stored and measure what still leaks
- Define monitors that separate a moved or dirty camera from weather
- Write the sections of a model card for a vision system

#### [day4#faces-and-plates] Street frames hold faces and number plates: blur them before anything is stored

*Privacy*

Figure: Top: as the camera delivers them. Bottom: as stored. Drawn frames; no real person or vehicle.

- A road-damage system has **no use** for a face or a plate
- They are personal data, so the pipeline blurs them **before storage**
- "Before" is an order of steps: no log, cache, queue or backup sees the raw frame

#### [day4#measure-the-blur] "We blur faces" is a claim until the leak is measured

*From the lab*

| Blur detections with a score of at least | Padding | Faces and plates still readable | Pixels blurred |
| --- | --- | --- | --- |
| 0.5 | none | 29.9% | 1.2% |
| 0.5 | 3 pixels | 17.4% | 2.6% |
| 0.2 | none | 16.0% | 1.5% |
| 0.2 | 3 pixels | 0.7% | 3.2% |

> **Two separate causes.** Boxes that ran tight, repaired by padding. Detections the detector was unsure of, caught by a lower threshold. Either fix alone leaves one in six.

#### [day4#recall-is-the-operating-point] For privacy the operating point is recall, bought with false alarms

*The operating point*

- **The pothole detector.** A false alarm sends a crew for nothing; a miss leaves a hole. The two costs are balanced, and the threshold sits in the middle.
- **The face and plate blur.** A false alarm blurs a patch of wall, which costs the road engineers nothing. A miss stores a readable plate. The threshold goes low.

> **The price in the lab:** blurred pixels rise from 1.2% to 3.2% of the frame. For a road department that price is nothing.

#### [day4#a-monitor-that-cries-rain] A monitor that pages for weather is switched off within a week

*Monitoring*

**12 of 12** cameras raise an alert when the monitor is "brightness left its first-week range" (lab 4.4, 30 days)

- It rained on three days and every camera went dark together
- Nothing was wrong with any camera
- A useful monitor fires on **what a technician can fix**, and stays quiet otherwise

#### [day4#two-drift-monitors] Two monitors, each with a signal, a band and a run length

*From the lab*

| Monitor | Signal | Band | Run | Caught |
| --- | --- | --- | --- | --- |
| **Moved camera** | Offset of the lane markings, in pixels | Within 5 pixels of the installed position | 2 days | CAM-07 on day 20, one day after the knock |
| **Dirty lens** | Sharpness against the camera's own first week | At least 85% of its first-week median | 3 days | CAM-03 on day 14, five days after the dust began |

> **Both stay silent on the three rainy days.** Rain darkens a frame; it does not move the lane markings, and it does not last three days.

#### [day4#model-card-sections] The model card is where the week's evidence is kept

*Documentation*

- **What it is for.** The task, the cameras, the classes. What it must not be used for.
- **Data.** The guideline and its agreement score, the split report, the licence record of the weights.
- **Results.** By lighting, weather and district, with intervals. The error breakdown.
- **How to run.** The operating point, the benchmark table, the preprocessing contract.
- **Limits and people.** The conditions in which a person checks a detection before a crew is sent.
- **Privacy and monitoring.** The blur and its measured leak rate, the monitors and who is paged.

#### [day4#checkpoint-4] Which monitor do you add?

*Checkpoint*

**Question:** A counting camera's daily vehicle count falls by a third and stays there. The counts of the neighbouring cameras are unchanged, and it has not rained. Which monitor would have told the field team why?

- A. The brightness of the frames against the first week
- B. The count itself against last month's average
- C. The offset of the lane markings from their installed position
- D. The mean score of the detections per day

#### [day4#lab-4-4] Blur, monitor, choose

Lab 4.4 · 25 min

Add face and plate blurring before storage and define two camera-drift monitors, then choose the capstone task.

[D4_4_privacy_monitoring.ipynb](https://sdaiaacademy.github.io/ae-309/labs/D4_4_privacy_monitoring.ipynb)

- [ ] Blur faces and plates before storage (7 min)
- [ ] Measure what still leaks and fix it (6 min)
- [ ] Two camera-drift monitors on the 30-day stream (8 min)
- [ ] Choose your capstone task and write its budget (4 min)
- [ ] ✱ Stretch: Add a third monitor on detections per frame that ignores rain

#### [day4#wrap-up] Three things to keep

*Day 4 recap*

- **Prove the copy.** Raw outputs within tolerance, the same boxes, and the test kept. The prototype's 816 boxes came from the lines before the model.
- **Budget on the tail.** Median and 95th percentile per operating point, accuracy measured again after every change, the machine named.
- **Rule first, then the test.** Drops per condition with intervals, a decision per condition, a blur whose leak is measured, monitors that stay silent on rain.

#### [day4#next] Capstone · A service that meets its budget

*Next*

An inspection or counting service for Al-Manar in a public repository, with a field-conditions test set, a model card and a release memo, defended on a failure case the assessor picks.

- Bring the task and the budget you wrote in lab 4.4
- Make the repository public, with one command that re-runs the evaluation
- Expect a held-back condition: your evaluation is re-run on it live

## Capstone: Capstone: a service that meets its budget, defended on a failure

An inspection or counting service for Al-Manar with its field-conditions test set, model card and release memo

#### [capstone#title] Capstone: a service that meets its budget, defended on a failure

*SDAIA Academy · Computer Vision Engineering*

Capstone: An inspection or counting service for Al-Manar with its field-conditions test set, model card and release memo

AI Engineering Track · 4 teaching days + capstone day · 8 min talk and demo · 4 min Q&A · AE-309

#### [capstone#agenda] How the day runs

*Capstone*

- **The brief, the rubric and final prep**: what is submitted, how it is scored, a last rehearsal
- **Defences**: eight minutes of talk, then four minutes on a failure case the assessor picks
- **Review and closing**: peer feedback, the trainer's review, what next

#### [capstone#the-brief] A service for Al-Manar that meets a budget you stated

*The brief*

- **Inspection.** Find potholes and patches, or segment cracks, in road frames. Accuracy is average precision or mask overlap, by lighting and district.
- **Counting.** Count vehicles across a line in traffic clips. Accuracy is count error per clip, with identity switches reported beside it.

> **You wrote the budget in lab 4.4:** a per-frame latency at the 95th percentile on the roadside unit, an accuracy floor, and at least three field conditions.

#### [capstone#what-is-submitted] Six things the assessor looks for in the public repository

*The repository*

- **Data card and split report.** What the data is, the guideline, the agreement score, no site shared across splits.
- **Field-conditions test set.** Paired with the clean test set, and how it was made.
- **Benchmark table.** Latency, size and accuracy per operating point, the machine named, the chosen row marked.
- **Model card.** Results per condition and district with intervals, the privacy controls and their measured recall, the monitors.
- **Release memo.** One page: release, release with a human check, or hold, per condition, against a rule stated in advance.
- **The trail.** Training logs, checkpoints, commit history, and one command that re-runs the evaluation.

#### [capstone#rubric] The rubric

*How the repository is scored*

| Item | Value |
| --- | --- |
| Data and splits | 20% |
| Model quality and error analysis | 25% |
| Latency budget | 20% |
| Field-condition tests and privacy | 25% |
| Reproducibility | 10% |

> **A claim without a number is a plan, not a result.** And a number without its condition, its count and its interval is half a result.

#### [capstone#format] Eight minutes of talk, then four minutes on a failure the assessor picks

*Format*

- **1 · The budget and the result.** The task, the budget you stated, the operating point you chose. Two minutes.
- **2 · The evidence.** The table by condition with intervals, the error breakdown, the release decision. Four minutes.
- **3 · What you would not release.** The weakest condition, what happens under it, the next step. Two minutes.

> **Then the failure case.** The assessor shows one frame or clip your service gets wrong, and names a held-back condition. You re-run the evaluation on it and explain what you see.

#### [capstone#running-order] Who presents when

*Running order*

| # | Presenter | Service |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first defence*

Clone your repository into a new folder and run the evaluation command once.

#### [capstone#now-presenting] Talk

*Now presenting*

##### Failure case

#### [capstone#review] What the defences showed

*Trainer review*

- **Strongest evidence.** Tables and decisions that several projects got right.
- **Common gaps.** What most repositories would fix next.
- **One surprise.** A failure nobody expected, and what it taught.

#### [capstone#what-next] Keep going

*After the course*

- **The follow-up task:** run your service on one condition you did not test this week, and add the row to your model card
- **Your repository stays yours:** keep the evaluation command working as the model changes
- **Next Academy course:** AE-408 Multimodal and Speech Systems Engineering, for vision-language models and multimodal applications

#### [capstone#thanks] Thank you

*SDAIA Academy*

Computer Vision Engineering
