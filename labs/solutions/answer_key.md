# Computer Vision Engineering: answer key for the trainer

What is planted in the Al-Manar data pack, which lab finds it, and what a quick-mode run (`CV_QUICK=1`) of each solution
prints on a laptop processor. Everything is drawn from one seed, so the counts repeat exactly. Scores of models trained in
a lab move in the second decimal between machines and library versions. Latencies belong to the machine.

No lab calls a language model. Labs 2.1 to 2.3 download pretrained weights on first use; every other lab runs offline.

## Day 1 · Visual data first

| Lab | Planted | A quick-mode run of the solution |
|---|---|---|
| 1.1 Errors by condition | The pilot classifier was trained on clean daylight crops from the survey vehicle only | 95.5% on 156 survey-style validation crops; 43% on the 200 field crops; by lighting: day 88%, glare 65%, night 34%, rain 18%, dust 12%; worst slice with at least 20 crops: dust (40 crops) |
| 1.2 Agree on a new class | Guideline v1 for alligator cracking asks for a resemblance; the two supplied annotators read it differently. One annotator boxes with a margin | Kappa 0.44 on the first pass (raw agreement 0.70, 9 disagreements in 30 crops), 0.85 after guideline v2 ("at least three closed cells"); mean IoU between the two annotators' boxes 0.71 |
| 1.3 Split by road segment | The inherited split was drawn frame by frame; 170 of 1,200 frames lost their segment id | 274 near-duplicate pairs at 30 bits or fewer, 150 of them across splits; 44 segment ids recovered through a duplicate, 126 frames dropped; new split 780 / 130 / 164 frames in 44 / 7 / 9 segments, no segment shared; one-nearest-neighbour accuracy 0.629 on the inherited split and 0.300 on the split by segment |
| 1.4 Augment without changing labels | Two of six transforms change what a crop shows: the quarter turn swaps the two crack classes, the tight crop loses the defect | Label-change rate 0 for the mirror left-right, the mirror up-down, brightness and contrast, and blur; 0.42 for the quarter turn; 0.42 for the tight crop; probe accuracy on the crack classes 0.946 with the safe pipeline and 0.757 with quarter-turned copies (37 crops) |

## Day 2 · Backbones and transfer

| Lab | Planted | A quick-mode run of the solution |
|---|---|---|
| 2.1 Two backbones by slice | Small slices: 16 night crops among 304 test crops | 11.2 and 5.5 million weights; after one pass over 320 crops, test accuracy 0.727 (ResNet-18) and 0.507 (ViT-Tiny); intervals apart on all crops and on day, overlapping on glare, dust, rain and night |
| 2.2 Probe or fine-tune | Nothing planted. The four full fine-tuning runs are supplied in `data/runs/` | 139 labelled crops, 304 test crops; probes 0.757 (ResNet-18) and 0.674 (ViT-Tiny); supplied runs 0.938, 0.938, 0.941, 0.944 on those crops; decision: train. In quick mode the probe sees about 140 labels and the supplied runs saw 717. Full mode, 717 labels and 1,575 test crops: probes 0.814 and 0.758; full fine-tuning 0.896 and 0.921 on 10% of the labels, 0.924 and 0.935 on all of them |
| 2.3 The zero-shot floor | Bare class names make a poor classifier | On 300 test crops: most common class 0.323, bare names 0.280 (225 crops sent to "transverse crack"), the solution's described prompts 0.510; lowest recall on "clear"; the supplied runs stand 0.43 above the floor |
| 2.4 Which backbone may ship | Three backbones chosen to give three answers | apache-2.0, cc-by-nc-4.0, dinov3-license (read from timm 1.0.31); decisions: deploy, do not deploy, hold for legal review. A newer timm may report another identifier: the participant records what they see |

## Day 3 · Detect, segment and track

| Lab | Planted | A quick-mode run of the solution |
|---|---|---|
| 3.1 Train and score a detector | One annotator's boxes carry a margin, so the truth disagrees with itself at IoU 0.75 | 7 raw boxes and 1 after suppression on the example frame; mean AP at 0.5 and 0.75: 0.871 and 0.306 from the warm checkpoint, 0.894 and 0.483 after two passes over 400 frames; toy example 0.8333. The supplied detector scores 0.968 and 0.785 on the full validation split |
| 3.2 Sort fifty errors | The roadside prototype ran suppression at IoU 0.7, so duplicates survive. The fifty errors were hand-picked to show every type | 14 localisation, 8 class, 10 duplicate, 18 missed (the key is `data/supplied/detector_errors_key.json`); on 180 field frames precision 0.783 and recall 0.937 (148 right, 41 false alarms, 10 missed); oracle gains in F1: duplicate +0.063, missed +0.016, class 0.000, localisation 0.000; chosen fix: duplicate |
| 3.3 Segment cracks | A rule about darkness fails when the light changes | Crack pixels 1.1%; IoU and Dice: thresholding baseline 0.224 and 0.366, warm checkpoint 0.218 and 0.358, after three passes over 240 frames 0.378 and 0.549, supplied segmenter 0.809 and 0.894; baseline IoU 0.072 under glare and 0.008 at night |
| 3.4 Track and count | Queue clips jitter at the line; a gantry hides vehicles for several frames; night clips carry headlight reflections | 49 tracks for 16 vehicles in the first clip without patience; the queue clip counts 14 by every crossing and 13 once per track (truth 13); baseline count errors 3, 5, 4, 14 and identity switches 67, 106, 105, 172 for day, queue, gantry, night; chosen settings patience 5 and minimum hits 4: count errors 0, 0, 0, 1 and switches 1, 18, 40, 6 |

## Day 4 · Fit the budget and release

| Lab | Planted | A quick-mode run of the solution |
|---|---|---|
| 4.1 Export and check parity | The prototype's serving function feeds BGR frames with values from 0 to 255 | File 3.5 MB with free batch, height and width; largest difference 2.24e-05 against a tolerance of 1e-03; 30 boxes on each side, parity 1.000; prototype 816 boxes, parity 0.000; fixed 30 boxes, parity 1.000 |
| 4.2 Fit the latency budget | The float file is over the 2 MB limit, so quantisation is needed | Files 3.53 MB and 0.93 MB; AP at 0.5: 0.973, 0.973, 0.921 for the float model at 192, 160, 128 pixels and 0.973, 0.973, 0.919 for the 8-bit one. Latency depends on the machine: on the author's laptop the chosen point was 8-bit at 160 pixels, about 31 ms at the 95th percentile on the unit. "No row qualifies" is a valid result on a slow or busy machine |
| 4.3 Field-condition tests | Night at high severity costs the detector most | On 90 paired test frames: clean AP 1.000; rain 0.987, glare 1.000, night 0.924, moved camera 0.987; drops with intervals: rain 0.013 (0.001 to 0.035), glare 0.000, night 0.076 (0.022 to 0.159), moved camera 0.013 (0.000 to 0.042); decisions: pass, pass, human check, pass |
| 4.4 Blur, monitor, choose | Tight boxes and unsure detections leave faces and plates readable. It rains on three days of the stream; camera CAM-07 is knocked on day 19; the lens of CAM-03 gathers dust from day 9 | 144 faces and plates in 40 frames; leaked 29.9% (score 0.5, no padding), 17.4% (0.5, 3 pixels), 16.0% (0.2, none), 0.7% (0.2, 3 pixels); the brightness monitor alerts on 12 of 12 cameras; with the solution's monitors CAM-07 alerts on day 20 and CAM-03 on day 14 |

## How the supplied files were made

`_tools/train_checkpoints.py` trained the pilot classifier, the detector, the segmenter and the four fine-tuning runs on
the full pack on a laptop processor, and wrote the fifty detector errors with their key. `data/reference/*.json` holds the
results the notebooks fall back on when an exercise is not done yet; they come from the quick-mode runs above.
