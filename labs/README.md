# Computer Vision Engineering: labs

Sixteen 25-minute labs on one case study, the roads and traffic department of the fictional city of Al-Manar and its
system Rasid. Every lab starts from a supplied checkpoint and the supplied split by road segment, so no lab needs an
earlier one. No lab calls a language model.

## Quick start

**Google Colab (recommended for Days 2 and 3).** Open a notebook from the course page or from the table below. Its first
cell clones this repository and installs what Colab lacks (`requirements-colab.txt`). Choose a GPU runtime
(Runtime > Change runtime type). Run `00_setup_check.ipynb` first.

**A laptop with [uv](https://docs.astral.sh/uv/).** Every lab has a quick mode that runs on a processor.

```
git clone https://github.com/SDAIAAcademy/computer-vision
cd computer-vision/labs
uv sync
export CV_QUICK=1          # PowerShell: $env:CV_QUICK = "1"
uv run jupyter lab
```

`uv sync` installs the processor build of PyTorch. For a local graphics processor, install the matching PyTorch build
yourself.

## The data

Nothing is downloaded: `prepare_data.py` draws the whole pack from one seed into `labs/cache/` the first time a lab asks
for it (about four minutes for the full pack, under a minute for the quick one). Everything is synthetic; no real road,
camera or person appears.

| Part | What it holds | Used by |
|---|---|---|
| Road frames | 6,000 frames of 192 by 192 pixels from 300 road segments (4 spots each, 5 passes per spot) in five districts, under day, glare, dust, rain and night; instance maps for boxes and crack masks; a COCO annotation file | Days 1 to 4 |
| Defect crops | About 10,000 crops of 96 by 96 pixels in five classes: clear, pothole, longitudinal crack, transverse crack, patch | Days 1 and 2 |
| Annotation batch | 30 crops for the new class "alligator cracking", with two passes by two annotators | Lab 1.2 |
| Traffic clips | 12 clips of 180 frames with ground-truth tracks, counts and imitated detections: day, queue, gantry, night | Lab 3.4 |
| Street frames | 40 frames with drawn faces and plates, their true regions and imitated detections | Lab 4.4 |
| Camera stream | 30 days of daily statistics from 12 pole cameras | Lab 4.4 |

Quick mode (`CV_QUICK=1`) uses the first 60 road segments (1,200 frames) and shorter training schedules. The quick pack is
exactly the first fifth of the full one, and the field batches of labs 1.1 and 4.3 are the same frames in both modes.

## What is supplied

| File | What it is |
|---|---|
| `roadkit.py` | The helper module (`import roadkit as U`): data access, the three compact models, training loops, detection metrics, export helpers |
| `prepare_data.py` | The data generator |
| `data/checkpoints/pilot_classifier.pt` | The pilot crop classifier of Day 1, trained on clean daylight crops only |
| `data/checkpoints/detector_warm.pt`, `detector.pt` | The pothole and patch detector after one short pass, and fully trained |
| `data/checkpoints/segmenter_warm.pt`, `segmenter.pt` | The crack segmenter after one short pass, and fully trained |
| `data/runs/*.json` | Full fine-tuning runs of two backbones on 10% and 100% of the labels, with a prediction per test crop |
| `data/supplied/detector_errors.json` | Fifty errors of the detector for lab 3.2 (the key for the trainer is beside it) |
| `data/reference/*.json` | Reference results that keep a notebook running when an exercise is not done yet |
| `solutions/` | The solved notebooks and `answer_key.md` for the trainer |

The checkpoints were trained on the full pack on a laptop processor by `_tools/train_checkpoints.py`. The three models
share a ResNet-18 trunk (the ImageNet weights of torchvision) and are small on purpose, so that their weights fit in the
repository and a short schedule means minutes. The library detectors and segmenters named in the course (torchvision,
Ultralytics) follow the same training and evaluation steps.

## What downloads from the internet

| Lab | Download | Size |
|---|---|---|
| 2.1, 2.2 | ResNet-18 weights (torchvision) and ViT-Tiny weights (timm) | about 45 MB and 25 MB |
| 2.3 | An open-vocabulary image-text model, CLIP ViT-B/32 (OpenCLIP) | about 600 MB |
| 2.4 | Nothing: licence records are read from the installed timm package | |

Days 1, 3 and 4 run offline once the repository and its Python packages are installed.

## The notebooks

| Lab | Notebook | AI policy |
|---|---|---|
| Setup | [`00_setup_check.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/00_setup_check.ipynb) | |
| 1.1 Errors by condition | [`D1_1_errors_by_condition.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D1_1_errors_by_condition.ipynb) | tutor |
| 1.2 Agree on a new class | [`D1_2_annotation_agreement.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D1_2_annotation_agreement.ipynb) | off |
| 1.3 Split by road segment | [`D1_3_split_by_segment.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D1_3_split_by_segment.ipynb) | tutor |
| 1.4 Augment without changing labels | [`D1_4_augmentation.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D1_4_augmentation.ipynb) | tutor |
| 2.1 Two backbones by slice | [`D2_1_two_backbones.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D2_1_two_backbones.ipynb) | tutor |
| 2.2 Probe or fine-tune | [`D2_2_probe_or_finetune.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D2_2_probe_or_finetune.ipynb) | tutor |
| 2.3 The zero-shot floor | [`D2_3_zero_shot_floor.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D2_3_zero_shot_floor.ipynb) | tutor |
| 2.4 Which backbone may ship | [`D2_4_backbone_licences.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D2_4_backbone_licences.ipynb) | tutor |
| 3.1 Train and score a detector | [`D3_1_detector_ap.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D3_1_detector_ap.ipynb) | tutor |
| 3.2 Sort fifty errors | [`D3_2_error_analysis.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D3_2_error_analysis.ipynb) | off |
| 3.3 Segment cracks | [`D3_3_segment_cracks.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D3_3_segment_cracks.ipynb) | tutor |
| 3.4 Track and count | [`D3_4_track_count.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D3_4_track_count.ipynb) | tutor |
| 4.1 Export and check parity | [`D4_1_export_parity.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D4_1_export_parity.ipynb) | tutor |
| 4.2 Fit the latency budget | [`D4_2_latency_budget.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D4_2_latency_budget.ipynb) | tutor |
| 4.3 Field-condition tests | [`D4_3_field_tests.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D4_3_field_tests.ipynb) | tutor |
| 4.4 Blur, monitor, choose | [`D4_4_privacy_monitoring.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D4_4_privacy_monitoring.ipynb) | tutor |

"Tutor" means an AI assistant may explain code and errors but does not write the pipeline. "Off" means no assistant.

## How a notebook is built

Each exercise is a function with a marked gap that raises `NotImplementedError`. The cell after it checks your result and,
when the exercise is not done yet, falls back on a reference so that every later cell still runs. The solved notebooks are
in `solutions/`.

The notebooks are generated from the sources in `_src/` by `_tools/py2nb.py` (`uv run python _tools/py2nb.py` from
`labs/`). Edit the sources, not the notebooks.

## For the trainer

- Run every solution once before the course: `CV_QUICK=1 uv run --with nbclient jupyter execute solutions/<name>.ipynb`.
- `solutions/answer_key.md` lists every planted problem, the lab that finds it and the numbers of a quick-mode run.
- Lab 4.2 measures latency on the processor of each participant: its numbers differ from machine to machine, and so may
  the chosen operating point.
- The Colab links work once the repository is public.
