---
course: "Computer Vision Engineering"
code: "AE-309"
version: "2027.1.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-309/slides/day3.html"
---

# Computer Vision Engineering

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
