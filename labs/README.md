# Computer Vision Engineering: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · How vision systems fail in the field | [`D1_1_run_pretrained_classifier.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D1_1_run_pretrained_classifier.ipynb) |
| Day 1 | Session 2 · Annotation guidelines and agreement | [`D1_2_annotate_30_images.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D1_2_annotate_30_images.ipynb) |
| Day 1 | Session 3 · Leakage by site and near-duplicates | [`D1_3_find_near_duplicate.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D1_3_find_near_duplicate.ipynb) |
| Day 1 | Session 4 · Augmentation and what it assumes | [`D1_4_build_augmentation_pipeline.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D1_4_build_augmentation_pipeline.ipynb) |
| Day 2 | Session 1 · Convolutional networks and vision transformers | [`D2_1_fine_tune_two.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D2_1_fine_tune_two.ipynb) |
| Day 2 | Session 2 · How much to fine-tune | [`D2_2_train_linear_probe.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D2_2_train_linear_probe.ipynb) |
| Day 2 | Session 3 · Zero-shot baselines | [`D2_3_score_open_vocabulary.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D2_3_score_open_vocabulary.ipynb) |
| Day 2 | Session 4 · Licences of open-weight backbones | [`D2_4_check_licence_data.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D2_4_check_licence_data.ipynb) |
| Day 3 | Session 1 · Detection and its metrics | [`D3_1_train_pothole_detector.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D3_1_train_pothole_detector.ipynb) |
| Day 3 | Session 2 · Detector error analysis | [`D3_2_sort_50_supplied.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D3_2_sort_50_supplied.ipynb) |
| Day 3 | Session 3 · Segmenting cracks | [`D3_3_fine_tune_segmenter.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D3_3_fine_tune_segmenter.ipynb) |
| Day 3 | Session 4 · Tracking and counting | [`D3_4_track_count_vehicles.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D3_4_track_count_vehicles.ipynb) |
| Day 4 | Session 1 · Export to an interchange format | [`D4_1_export_supplied_detector.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D4_1_export_supplied_detector.ipynb) |
| Day 4 | Session 2 · Resolution and precision against latency | [`D4_2_measure_latency_accuracy.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D4_2_measure_latency_accuracy.ipynb) |
| Day 4 | Session 3 · Corruption and shift tests | [`D4_3_run_model_rain.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D4_3_run_model_rain.ipynb) |
| Day 4 | Session 4 · Privacy and monitoring in the field | [`D4_4_add_face_plate.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/computer-vision/blob/main/labs/D4_4_add_face_plate.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.
