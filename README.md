<!-- academy:title -->
# Computer Vision Engineering

هندسة الرؤية الحاسوبية

AI Engineering Track · SDAIA Academy · AE-309 · Level 300

Build vision systems that are measured by condition, fitted to a roadside processor and released with evidence

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
| Capstone | Capstone: a service that meets its budget, defended on a failure |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**Al-Manar Municipality Roads and Traffic Department**, a fictional city department, and **Rasid (راصد)**, the system
participants build for it: road-damage inspection from survey-vehicle cameras and vehicle counting from pole cameras, on a
processor-only roadside unit that must keep working in glare, dust, rain and at night.

All data is synthetic and drawn from one seed by `labs/prepare_data.py`: 6,000 road frames from 300 road segments in five
districts (potholes, cracks, patches, alligator cracking, with manhole covers and shadows as confusers, under day, glare,
dust, rain and night), instance maps for boxes and crack masks, twelve traffic clips with ground-truth tracks and counts,
forty street frames with drawn faces and plates, and a 30-day stream of camera statistics. No real road, camera or person
appears. Problems are planted for the labs to find: a pilot classifier trained on clean daylight only, a random split that
shares road spots, missing segment ids, a class two annotators read differently, transforms that change a label, a gantry
that breaks tracks, a serving function with the wrong preprocessing, a camera that was knocked and a lens gathering dust.

The catalog describes this course on public road-damage images. This edition runs on the synthetic pack, so that no data is
downloaded and every lab runs on a laptop processor in a quick mode, with known planted problems. The capstone may use a public road-damage
set (for example RDD2022) under its own licence.

## How the labs run

Google Colab with a free GPU runtime, or a laptop with [uv](https://docs.astral.sh/uv/) and the quick mode (`CV_QUICK=1`).
Each lab starts from a supplied checkpoint and the supplied split by road segment; no lab needs an earlier one. Day 2
downloads pretrained backbones from their public hubs. No lab calls a language model. See [labs/README.md](labs/README.md).

## How the course is assessed

- Supervised entry and exit checks without AI assistance, on metrics, splits and error types: 20%
- The capstone repository on a five-criterion rubric (data and splits 20, model quality and error analysis 25, latency
  budget 20, field-condition tests and privacy 25, reproducibility 10): 60%
- A live defence on a failure case the assessor picks: 20%

Passing is 70% overall and at least 60% on the capstone rubric. AI assistants are off in the supervised checks and in the
annotation and error-analysis labs (1.2 and 3.2), may explain code and errors in the other labs, and are unrestricted in
the capstone, which is defended live.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Errors by condition | [`D1_1_errors_by_condition.ipynb`](labs/D1_1_errors_by_condition.ipynb) |
| 1.2 | Day 1 · Session 2 | Agree on a new class | [`D1_2_annotation_agreement.ipynb`](labs/D1_2_annotation_agreement.ipynb) |
| 1.3 | Day 1 · Session 3 | Split by road segment | [`D1_3_split_by_segment.ipynb`](labs/D1_3_split_by_segment.ipynb) |
| 1.4 | Day 1 · Session 4 | Augment without changing labels | [`D1_4_augmentation.ipynb`](labs/D1_4_augmentation.ipynb) |
| 2.1 | Day 2 · Session 1 | Two backbones by slice | [`D2_1_two_backbones.ipynb`](labs/D2_1_two_backbones.ipynb) |
| 2.2 | Day 2 · Session 2 | Probe or fine-tune | [`D2_2_probe_or_finetune.ipynb`](labs/D2_2_probe_or_finetune.ipynb) |
| 2.3 | Day 2 · Session 3 | The zero-shot floor | [`D2_3_zero_shot_floor.ipynb`](labs/D2_3_zero_shot_floor.ipynb) |
| 2.4 | Day 2 · Session 4 | Which backbone may ship | [`D2_4_backbone_licences.ipynb`](labs/D2_4_backbone_licences.ipynb) |
| 3.1 | Day 3 · Session 1 | Train and score a detector | [`D3_1_detector_ap.ipynb`](labs/D3_1_detector_ap.ipynb) |
| 3.2 | Day 3 · Session 2 | Sort fifty errors | [`D3_2_error_analysis.ipynb`](labs/D3_2_error_analysis.ipynb) |
| 3.3 | Day 3 · Session 3 | Segment cracks | [`D3_3_segment_cracks.ipynb`](labs/D3_3_segment_cracks.ipynb) |
| 3.4 | Day 3 · Session 4 | Track and count | [`D3_4_track_count.ipynb`](labs/D3_4_track_count.ipynb) |
| 4.1 | Day 4 · Session 1 | Export and check parity | [`D4_1_export_parity.ipynb`](labs/D4_1_export_parity.ipynb) |
| 4.2 | Day 4 · Session 2 | Fit the latency budget | [`D4_2_latency_budget.ipynb`](labs/D4_2_latency_budget.ipynb) |
| 4.3 | Day 4 · Session 3 | Field-condition tests | [`D4_3_field_tests.ipynb`](labs/D4_3_field_tests.ipynb) |
| 4.4 | Day 4 · Session 4 | Blur, monitor, choose | [`D4_4_privacy_monitoring.ipynb`](labs/D4_4_privacy_monitoring.ipynb) |
<!-- /academy:labs -->

## Capstone

An inspection or counting service that meets a stated latency and accuracy budget, submitted as a public repository with a field-conditions test set, model card and release memo, defended on a failure case the assessor picks

**Submission:** a public GitHub repository link: the service, a field-conditions test set, a benchmark table against the
stated latency and accuracy budget, a model card with results per condition and district and the privacy controls, a
release memo, and the training logs, checkpoints and commit history. Then a live defence: an eight-minute talk and four
minutes on a failure case the assessor picks, with the evaluation re-run on a held-back condition.

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

### Frameworks and model libraries

- [torchvision: models and pretrained weights](https://docs.pytorch.org/vision/stable/models.html)
- [torchvision: transforms v2](https://docs.pytorch.org/vision/stable/transforms.html)
- [timm (PyTorch Image Models) documentation](https://huggingface.co/docs/timm/index)
- [OpenCLIP: an open source implementation of CLIP](https://github.com/mlfoundations/open_clip)
- [Ultralytics: licensing](https://www.ultralytics.com/license)

### Export and inference runtimes

- [PyTorch: torch.onnx](https://docs.pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime: quantize ONNX models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [OpenVINO documentation](https://docs.openvino.ai/)
- [LiteRT overview](https://ai.google.dev/edge/litert)

### Papers

- [He et al., Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)
- [Dosovitskiy et al., An Image is Worth 16x16 Words (2020)](https://arxiv.org/abs/2010.11929)
- [Radford et al., Learning Transferable Visual Models From Natural Language Supervision (2021)](https://arxiv.org/abs/2103.00020)
- [Tian et al., FCOS: Fully Convolutional One-Stage Object Detection (2019)](https://arxiv.org/abs/1904.01355)
- [Lin et al., Focal Loss for Dense Object Detection (2017)](https://arxiv.org/abs/1708.02002)
- [Bolya et al., TIDE: A General Toolbox for Identifying Object Detection Errors (2020)](https://arxiv.org/abs/2008.08115)
- [Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)](https://arxiv.org/abs/1505.04597)
- [Bewley et al., Simple Online and Realtime Tracking (2016)](https://arxiv.org/abs/1602.00763)
- [Hendrycks and Dietterich, Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (2019)](https://arxiv.org/abs/1903.12261)
- [Mitchell et al., Model Cards for Model Reporting (2018)](https://arxiv.org/abs/1810.03993)
- [Arya et al., RDD2022: A multi-national image dataset for automatic Road Damage Detection (2022)](https://arxiv.org/abs/2209.08538)

### Annotation and dataset tools

- [CVAT documentation](https://docs.cvat.ai/)
- [Label Studio documentation](https://labelstud.io/guide/)
- [FiftyOne documentation](https://docs.voxel51.com/)
- [cleanlab documentation](https://docs.cleanlab.ai/)
- [COCO: data format](https://cocodataset.org/#format-data)
- [COCO: detection evaluation](https://cocodataset.org/#detection-eval)

### National references

- [Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)
- [Data Classification Policy (National Data Management Office)](https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf)
- [Principles of AI Ethics](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)
<!-- /academy:sources -->

## National references

- Personal Data Protection Law: https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf
- Implementing Regulation of the Personal Data Protection Law: https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf
- Data Classification Policy (National Data Management Office): https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf
- Principles of AI Ethics (SDAIA): https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf
- National AI Risk Management Framework: cited by name; no official address was found for it when this edition was written
