<!-- academy:title -->
# Computer Vision Engineering

هندسة رؤية الحاسب

AI Engineering Track · SDAIA Academy · AE-309 · Level 300

REPLACE: one line under the title: what participants can do after the course

Participants build vision systems for a fictional municipal roads and traffic department that must keep working in glare, dust, rain and at night. On public road-damage images and synthetic traffic-camera clips they audit and extend annotations, fine-tune convolutional and transformer backbones, train a pothole detector and a crack segmenter, track and count vehicles, and fit the chosen model to the latency and memory budget of a processor-only roadside unit. Each lab ends with a test under changed lighting, blur or camera position. Vision-language models and multimodal applications are AE-408.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/computer-vision/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Prepare an image dataset with annotation guidelines, an agreement check, splits that keep each road segment in one split, and augmentations that preserve label meaning, given road images with partial labels and near-duplicate frames | Applied | a dataset card, an agreement score and a split report with no road segment shared across splits |
| LO2 | Implement transfer learning with two pretrained backbones and compare them by condition slice with intervals, given defect crops and a free notebook GPU | Applied | a comparison table by lighting and weather slice with bootstrap intervals |
| LO3 | Build a detector and a segmenter and diagnose their errors by type to choose the next improvement, given pothole boxes and crack masks | Performance | average precision and overlap metrics, an error breakdown and a justified next step |
| LO4 | Measure tracking and counting accuracy for vehicles crossing a line in video, given synthetic camera clips with ground-truth counts | Applied | count error per clip and the number of identity switches |
| LO5 | Optimise a model to a per-frame latency and memory budget on a processor through export, input resolution and quantisation, re-measuring accuracy at each step, given the roadside unit's stated budget | Performance | a benchmark table and the chosen operating point |
| LO6 | Evaluate the model under corruption and camera shift and decide whether to release it, given rain, glare, night and moved-camera test sets | Performance | a model card with results per condition, privacy controls and a release decision defended live |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Visual data first | How vision systems fail in the field; Annotation guidelines and agreement; Leakage by site and near-duplicates; Augmentation and what it assumes | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Backbones and transfer | Convolutional networks and vision transformers; How much to fine-tune; Zero-shot baselines; Licences of open-weight backbones | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Detect, segment and track | Detection and its metrics; Detector error analysis; Segmenting cracks; Tracking and counting | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Fit the budget and release | Export to an interchange format; Resolution and precision against latency; Corruption and shift tests; Privacy and monitoring in the field | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Run a pretrained classifier on 200 road images and sort its errors by lighting, blur and viewpoint | [`D1_1_run_pretrained_classifier.ipynb`](labs/D1_1_run_pretrained_classifier.ipynb) |
| 1.2 | Day 1 · Session 2 | Annotate 30 images for a new defect class with a partner and compute agreement | [`D1_2_annotate_30_images.ipynb`](labs/D1_2_annotate_30_images.ipynb) |
| 1.3 | Day 1 · Session 3 | Find near-duplicate frames and re-split the data by road segment | [`D1_3_find_near_duplicate.ipynb`](labs/D1_3_find_near_duplicate.ipynb) |
| 1.4 | Day 1 · Session 4 | Build an augmentation pipeline and find the transform that changes what a label means | [`D1_4_build_augmentation_pipeline.ipynb`](labs/D1_4_build_augmentation_pipeline.ipynb) |
| 2.1 | Day 2 · Session 1 | Fine-tune two pretrained backbones on defect crops and compare them by slice | [`D2_1_fine_tune_two.ipynb`](labs/D2_1_fine_tune_two.ipynb) |
| 2.2 | Day 2 · Session 2 | Train a linear probe on 10% of the labels and compare it with supplied full fine-tuning runs on 10% and 100% | [`D2_2_train_linear_probe.ipynb`](labs/D2_2_train_linear_probe.ipynb) |
| 2.3 | Day 2 · Session 3 | Score an open-vocabulary model on the defect classes as the floor to beat | [`D2_3_score_open_vocabulary.ipynb`](labs/D2_3_score_open_vocabulary.ipynb) |
| 2.4 | Day 2 · Session 4 | Check the licence and data terms of three backbones and record which the department may deploy | [`D2_4_check_licence_data.ipynb`](labs/D2_4_check_licence_data.ipynb) |
| 3.1 | Day 3 · Session 1 | Train a pothole detector on a short schedule and compute average precision at two overlap thresholds | [`D3_1_train_pothole_detector.ipynb`](labs/D3_1_train_pothole_detector.ipynb) |
| 3.2 | Day 3 · Session 2 | Sort 50 supplied detector errors into localisation, class, duplicate and missed, and choose a fix | [`D3_2_sort_50_supplied.ipynb`](labs/D3_2_sort_50_supplied.ipynb) |
| 3.3 | Day 3 · Session 3 | Fine-tune a segmenter and compare its overlap score with a thresholding baseline | [`D3_3_fine_tune_segmenter.ipynb`](labs/D3_3_fine_tune_segmenter.ipynb) |
| 3.4 | Day 3 · Session 4 | Track and count vehicles across a line in a clip and measure count error against the labels | [`D3_4_track_count_vehicles.ipynb`](labs/D3_4_track_count_vehicles.ipynb) |
| 4.1 | Day 4 · Session 1 | Export the supplied detector and check its outputs match the original within tolerance | [`D4_1_export_supplied_detector.ipynb`](labs/D4_1_export_supplied_detector.ipynb) |
| 4.2 | Day 4 · Session 2 | Measure latency and accuracy at three resolutions and two precisions on a laptop processor | [`D4_2_measure_latency_accuracy.ipynb`](labs/D4_2_measure_latency_accuracy.ipynb) |
| 4.3 | Day 4 · Session 3 | Run the model on rain, glare, night and moved-camera sets and report the drop for each | [`D4_3_run_model_rain.ipynb`](labs/D4_3_run_model_rain.ipynb) |
| 4.4 | Day 4 · Session 4 | Add face and plate blurring before storage and define two camera-drift monitors, then choose the capstone task | [`D4_4_add_face_plate.ipynb`](labs/D4_4_add_face_plate.ipynb) |
<!-- /academy:labs -->

## Capstone

An inspection or counting service that meets a stated latency and accuracy budget, submitted as a public repository with a field-conditions test set, model card and release memo, defended on a failure case the assessor picks

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- AE-201 Machine Learning Foundations and Model Evaluation
- AE-202 Deep Learning and Generative AI Fundamentals
- Trains a neural network with a hand-written training loop
- Computes precision and recall from a confusion matrix

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Deep learning framework and pretrained vision models:** PyTorch with torchvision, timm
- **Detection and segmentation library:** torchvision detection and segmentation models, Ultralytics
- **Annotation tool:** CVAT, Label Studio
- **Dataset inspection:** FiftyOne, cleanlab
- **Image and video processing:** OpenCV, Pillow
- **Interchange format and inference runtime:** ONNX Runtime, OpenVINO, LiteRT
- **Compute:** a free notebook GPU, Academy GPU hours
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
- NDMO Data Classification Policy: REPLACE: the address of the official text
- SDAIA AI Ethics Principles: REPLACE: the address of the official text
- National AI Risk Management Framework: REPLACE: the address of the official text
