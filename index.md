---
course: "Computer Vision Engineering"
code: "AE-309"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-309/index.html"
---

# Computer Vision Engineering

هندسة الرؤية الحاسوبية

AI Engineering Track · 4 teaching days + capstone day · 26 hours · Level 300

Build vision systems that are measured by condition, fitted to a roadside processor and released with evidence

Participants build vision systems for a fictional municipal roads and traffic department that must keep working in glare, dust, rain and at night. On public road-damage images and synthetic traffic-camera clips they audit and extend annotations, fine-tune convolutional and transformer backbones, train a pothole detector and a crack segmenter, track and count vehicles, and fit the chosen model to the latency and memory budget of a processor-only roadside unit. Each lab ends with a test under changed lighting, blur or camera position. Vision-language models and multimodal applications are AE-408.

**Who it is for:** Practitioners. Engineers who have completed the level 200 set or can show equivalent work: a deployed model or language model application they built and can explain.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Al-Manar Municipality Roads and Traffic Department.** A fictional city department that inspects road damage from survey-vehicle cameras and counts vehicles from pole cameras, on a processor-only roadside unit that must keep working in glare, dust, rain and at night. Participants build Rasid, its inspection and counting system: they audit and extend the annotations, split by road segment, fine-tune two backbones, train a pothole detector and a crack segmenter, track and count vehicles, fit the detector to the unit's latency and size budget and test it under field conditions before a release decision. All data is synthetic and drawn with a fixed seed: 6,000 road frames in 300 segments over five districts, crack masks, traffic clips with ground-truth counts, street frames with invented faces and plates, and a 30-day stream of camera statistics.

## The course

### Day 1: Visual data first

Slides: https://sdaiaacademy.github.io/ae-309/slides/day1.md

- **Session 1: How vision systems fail in the field**
  - Name the conditions a roadside camera meets that a survey set does not
  - Break one accuracy number into slices by lighting, blur and viewpoint
  - Write a failure note that names the worst slice and its size
  - [Lab 1.1: Errors by condition (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D1_1_errors_by_condition.ipynb)
- **Session 2: Annotation guidelines and agreement**
  - Read boxes and masks in a common annotation format
  - Compute raw agreement and Cohen's kappa between two annotators
  - Revise a guideline from the disagreements and measure again
  - [Lab 1.2: Agree on a new class (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D1_2_annotation_agreement.ipynb)
- **Session 3: Leakage by site and near-duplicates**
  - Find near-duplicate frames with a difference hash
  - Split by road segment so that no segment is shared
  - Report how much a leaking split inflates a score
  - [Lab 1.3: Split by road segment (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D1_3_split_by_segment.ipynb)
- **Session 4: Augmentation and what it assumes**
  - Apply one transform to an image, its boxes and its mask together
  - Measure how often a transform changes what the label means
  - Choose augmentations from the conditions the field will bring
  - [Lab 1.4: Augment without changing labels (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D1_4_augmentation.ipynb)

### Day 2: Backbones and transfer

Slides: https://sdaiaacademy.github.io/ae-309/slides/day2.md

- **Session 1: Convolutional networks and vision transformers**
  - Contrast the assumptions of convolutional networks and vision transformers
  - Fine-tune two pretrained backbones on the same crops
  - Compare two models by condition slice with bootstrap intervals
  - [Lab 2.1: Two backbones by slice (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D2_1_two_backbones.ipynb)
- **Session 2: How much to fine-tune**
  - Train a linear probe on frozen features
  - Compare a probe with full fine-tuning at two label budgets
  - Decide whether more labels or more training is the next step
  - [Lab 2.2: Probe or fine-tune (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D2_2_probe_or_finetune.ipynb)
- **Session 3: Zero-shot baselines**
  - Explain how an image-text model classifies without task labels
  - Score a zero-shot baseline on the defect classes
  - Use the floor to judge whether training was worth its labels
  - [Lab 2.3: The zero-shot floor (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D2_3_zero_shot_floor.ipynb)
- **Session 4: Licences of open-weight backbones**
  - Separate the licence of the code, of the weights and of the training data
  - Read licence metadata from a model hub
  - Record a deployment decision with its conditions
  - [Lab 2.4: Which backbone may ship (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D2_4_backbone_licences.ipynb)

### Day 3: Detect, segment and track

Slides: https://sdaiaacademy.github.io/ae-309/slides/day3.md

- **Session 1: Detection and its metrics**
  - Explain how a detector turns score maps into boxes
  - Apply non-maximum suppression and say what it removes
  - Compute average precision at two overlap thresholds
  - [Lab 3.1: Train and score a detector (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D3_1_detector_ap.ipynb)
- **Session 2: Detector error analysis**
  - Sort detector errors into localisation, class, duplicate and missed
  - Estimate what fixing each error type would be worth
  - Justify the next improvement from the error breakdown
  - [Lab 3.2: Sort fifty errors (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D3_2_error_analysis.ipynb)
- **Session 3: Segmenting cracks**
  - Compute intersection over union and Dice for masks
  - Fine-tune a segmenter on a short schedule
  - Compare a segmenter with a thresholding baseline by condition
  - [Lab 3.3: Segment cracks (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D3_3_segment_cracks.ipynb)
- **Session 4: Tracking and counting**
  - Associate detections to tracks by overlap
  - Count line crossings once per track
  - Measure count error and identity switches against ground truth
  - [Lab 3.4: Track and count (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D3_4_track_count.ipynb)

### Day 4: Fit the budget and release

Slides: https://sdaiaacademy.github.io/ae-309/slides/day4.md

- **Session 1: Export to an interchange format**
  - Export a model to an interchange format with a dynamic input size
  - Check that the runtime's outputs match the original within a tolerance
  - Find a preprocessing mismatch between training and serving
  - [Lab 4.1: Export and check parity (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D4_1_export_parity.ipynb)
- **Session 2: Resolution and precision against latency**
  - Measure per-frame latency as a median and a 95th percentile
  - Quantise a model to 8-bit integers and re-measure its accuracy
  - Choose an operating point that meets a stated budget
  - [Lab 4.2: Fit the latency budget (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D4_2_latency_budget.ipynb)
- **Session 3: Corruption and shift tests**
  - Build paired test sets for corruption and camera shift
  - Report the drop per condition with an interval
  - Decide to release, release with conditions or hold
  - [Lab 4.3: Field-condition tests (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D4_3_field_tests.ipynb)
- **Session 4: Privacy and monitoring in the field**
  - Blur faces and plates before a frame is stored and measure what still leaks
  - Define monitors that separate a moved or dirty camera from weather
  - Write the sections of a model card for a vision system
  - [Lab 4.4: Blur, monitor, choose (25 min)](https://sdaiaacademy.github.io/ae-309/labs/D4_4_privacy_monitoring.ipynb)

### Capstone: Capstone: a service that meets its budget, defended on a failure

Slides: https://sdaiaacademy.github.io/ae-309/slides/capstone.md

## Capstone

An inspection or counting service that meets a stated latency and accuracy budget, submitted as a public repository with a field-conditions test set, model card and release memo, defended on a failure case the assessor picks

**What you submit:** A public GitHub repository link: the service, a field-conditions test set, a benchmark table against the stated latency and accuracy budget, a model card with results per condition and district and the privacy controls, a release memo, and the training logs, checkpoints and commit history. Then a live defence: an eight-minute talk and four minutes on a failure case the assessor picks, with the evaluation re-run on a held-back condition.

**Rubric:**

- Data and splits: 20%
- Model quality and error analysis: 25%
- Latency budget: 20%
- Field-condition tests and privacy: 25%
- Reproducibility: 10%

## Outcomes

- **LO1** Prepare an image dataset with annotation guidelines, an agreement check, splits that keep each road segment in one split, and augmentations that preserve label meaning, given road images with partial labels and near-duplicate frames (Applied) Evidence: a dataset card, an agreement score and a split report with no road segment shared across splits
- **LO2** Implement transfer learning with two pretrained backbones and compare them by condition slice with intervals, given defect crops and a free notebook GPU (Applied) Evidence: a comparison table by lighting and weather slice with bootstrap intervals
- **LO3** Build a detector and a segmenter and diagnose their errors by type to choose the next improvement, given pothole boxes and crack masks (Performance) Evidence: average precision and overlap metrics, an error breakdown and a justified next step
- **LO4** Measure tracking and counting accuracy for vehicles crossing a line in video, given synthetic camera clips with ground-truth counts (Applied) Evidence: count error per clip and the number of identity switches
- **LO5** Optimise a model to a per-frame latency and memory budget on a processor through export, input resolution and quantisation, re-measuring accuracy at each step, given the roadside unit's stated budget (Performance) Evidence: a benchmark table and the chosen operating point
- **LO6** Evaluate the model under corruption and camera shift and decide whether to release it, given rain, glare, night and moved-camera test sets (Performance) Evidence: a model card with results per condition, privacy controls and a release decision defended live

## Tools

- **Deep learning framework and pretrained vision models:** PyTorch with torchvision, timm
- **Detection and segmentation library:** torchvision detection and segmentation models, Ultralytics
- **Annotation tool:** CVAT, Label Studio
- **Dataset inspection:** FiftyOne, cleanlab
- **Image and video processing:** OpenCV, Pillow
- **Interchange format and inference runtime:** ONNX Runtime, OpenVINO, LiteRT
- **Compute:** a free notebook GPU, Academy GPU hours

## Prerequisites

- AE-201 Machine Learning Foundations and Model Evaluation
- AE-202 Deep Learning and Generative AI Fundamentals
- Trains a neural network with a hand-written training loop
- Computes precision and recall from a confusion matrix
- Setup: Google Colab with a free GPU runtime, or a laptop with uv (every lab has a quick mode for a laptop processor). Day 2 downloads pretrained backbones from their public hubs. Run 00_setup_check before Day 1; see labs/README.md.

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

---

SDAIA Academy · AI Engineering Track · Computer Vision Engineering

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-15 · commit d41c694
