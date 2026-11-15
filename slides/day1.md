---
course: "Computer Vision Engineering"
code: "AE-309"
version: "2027.1.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-309/slides/day1.html"
---

# Computer Vision Engineering

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
