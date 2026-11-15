---
course: "Computer Vision Engineering"
code: "AE-309"
version: "2027.1.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-309/slides/day4.html"
---

# Computer Vision Engineering

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
