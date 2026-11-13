"""Produce the supplied checkpoints, runs and error list of the course (the author's tool).

    uv run python _tools/train_checkpoints.py pilot runs detector segmenter errors

Runs on the full data pack (never in quick mode) on a laptop processor. Everything it writes
goes to labs/data/ and is committed, so no lab has to train from scratch.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import roadkit as U  # noqa: E402

assert not U.QUICK, "run without CV_QUICK"
RUNS = U.DATA / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
(U.DATA / "supplied").mkdir(parents=True, exist_ok=True)


def write(path, obj, indent=None):
    path.write_text(json.dumps(obj, indent=indent) + "\n", encoding="utf-8", newline="\n")


def acc(model, rows, size=None):
    return float((U.predict(model, U.tile_images(rows), size).argmax(1) == U.labels(rows)).mean())


def pilot():
    """The pilot classifier of Day 1: trained on clean daylight crops from the survey vehicle only."""
    rows = [t for t in U.tiles("train") if t["condition"] == "day" and not t["blur"]]
    torch.manual_seed(1)
    m = U.TileNet(pretrained=True)
    U.fit(m, U.tile_images(rows), U.labels(rows), epochs=6, lr=1e-3, bs=64)
    U.save_ckpt(m, "pilot_classifier")
    val = U.tiles("val")
    clean = [t for t in val if t["condition"] == "day" and not t["blur"]]
    print("pilot: train crops", len(rows), "val clean acc", acc(m, clean), "val all acc", acc(m, val))


def runs():
    """Full fine-tuning of the two Day 2 backbones on 10% and on 100% of the training labels."""
    train, test = U.tiles("train"), U.tiles("test")
    Xte, yte = U.tile_images(test), U.labels(test)
    order = np.random.default_rng(22).permutation(len(train))
    write(RUNS / "ten_percent_ids.json", [train[i]["id"] for i in order[: len(train) // 10]])
    for name in ("resnet18", "vit_tiny"):
        for pct in (10, 100):
            rows = [train[i] for i in order[: len(train) * pct // 100]]
            torch.manual_seed(2)
            m = U.backbone(name)
            t = time.time()
            ep = 8 if pct == 10 else 2
            U.fit(m, U.tile_images(rows), U.labels(rows), epochs=ep,
                  lr=3e-4 if name == "resnet18" else 1e-4, bs=32, size=128)
            mins = (time.time() - t) / 60
            pred = U.predict(m, Xte, 128).argmax(1)
            write(RUNS / f"{name}_full_{pct}.json", {
                "backbone": name, "weights": U.BACKBONES[name], "method": "full fine-tuning",
                "labels_percent": pct, "train_crops": len(rows), "epochs": ep, "input": 128,
                "minutes_on_laptop_processor": round(mins, 1),
                "test_ids": [q["id"] for q in test], "pred": pred.tolist()})
            print(name, pct, "acc", float((pred == yte).mean()), "minutes", round(mins, 1))


def detector():
    rows = U.frames("train")
    X, G = U.frame_images(rows), [U.det_gt(f["objects"]) for f in rows]
    val = U.frames("val")
    Xv, Gv = U.frame_images(val), [U.det_gt(f["objects"]) for f in val]
    torch.manual_seed(3)
    m = U.RoadDet(pretrained=True)
    U.fit_detector(m, X[:1200], G[:1200], epochs=1, lr=1e-3)
    U.save_ckpt(m, "detector_warm")
    p = U.detect(m, Xv)
    print("warm  AP50", U.mean_ap(p, Gv, 0.5), "AP75", U.mean_ap(p, Gv, 0.75))
    U.fit_detector(m, X, G, epochs=6, lr=1e-3, seed=1)
    U.save_ckpt(m, "detector")
    p = U.detect(m, Xv)
    print("final AP50", U.mean_ap(p, Gv, 0.5), "AP75", U.mean_ap(p, Gv, 0.75))


def seg_rows(split):
    r = np.random.default_rng(5)
    return [f for f in U.frames(split)
            if any(o["cls"] in U.CRACK_CLASSES for o in f["objects"]) or r.random() < 0.15]


def iou(prob, mask, thr=0.5):
    p = prob > thr
    return float((p & mask).sum() / max((p | mask).sum(), 1))


def segmenter():
    rows, val = seg_rows("train"), seg_rows("val")
    X, M = U.frame_images(rows), np.stack([U.crack_mask(f) for f in rows])
    Xv, Mv = U.frame_images(val), np.stack([U.crack_mask(f) for f in val])
    torch.manual_seed(4)
    m = U.RoadSeg(pretrained=True)
    U.fit_segmenter(m, X[:400], M[:400], epochs=1, lr=1e-3)
    U.save_ckpt(m, "segmenter_warm")
    print("warm  IoU", iou(U.segment(m, Xv), Mv))
    U.fit_segmenter(m, X, M, epochs=5, lr=1e-3, seed=1)
    U.save_ckpt(m, "segmenter")
    print("final IoU", iou(U.segment(m, Xv), Mv), "frames", len(rows))


def error_type(det, gts, used):
    """The error type of one detection, by the rules of lab 3.2."""
    if len(gts) == 0:
        return "background", -1
    iou_ = U.box_iou(det[None, :4], gts[:, :4])[0]
    j = int(iou_.argmax())
    same = gts[j, 4] == det[5]
    if iou_[j] >= 0.5 and same:
        return ("duplicate" if j in used else "correct"), j
    if iou_[j] >= 0.5:
        return "class", j
    if iou_[j] >= 0.1 and same:
        return "localisation", j
    return "background", j


def errors():
    """50 errors of the supplied detector on validation frames, for lab 3.2, with no type given.
    Non-maximum suppression is set loose (IoU 0.7), as the roadside prototype had it."""
    m = U.load_ckpt(U.RoadDet(), "detector")
    val = U.frames("val")
    preds = U.detect(m, U.frame_images(val), score_thr=0.3, nms_iou=0.7)
    pool = {k: [] for k in ("localisation", "class", "duplicate", "missed", "background")}
    for f, p in zip(val, preds):
        g = U.det_gt(f["objects"])
        used = set()
        for d in p[np.argsort(-p[:, 4])]:
            kind, j = error_type(d, g, used)
            if kind == "correct":
                used.add(j)
                continue
            pool[kind].append({"frame": f["id"], "address": f["address"], "kind": "detection",
                               "box": [round(float(v), 1) for v in d[:4]], "score": round(float(d[4]), 2),
                               "cls": U.DET_CLASSES[int(d[5])], "truth": kind,
                               "others": [[round(float(v), 2) for v in q[:5]] + [U.DET_CLASSES[int(q[5])]]
                                          for q in p if not np.array_equal(q, d)]})
        for j in range(len(g)):
            if j not in used and (len(p) == 0 or U.box_iou(g[j:j + 1, :4], p[:, :4]).max() < 0.1):
                pool["missed"].append({"frame": f["id"], "address": f["address"],
                                       "kind": "ground truth with no detection",
                                       "box": [round(float(v), 1) for v in g[j, :4]], "score": None,
                                       "cls": U.DET_CLASSES[int(g[j, 4])], "truth": "missed", "others": []})
    print({k: len(v) for k, v in pool.items()})
    r = np.random.default_rng(32)
    want = {"localisation": 14, "class": 8, "duplicate": 10, "missed": 18}
    chosen = []
    for k, n in want.items():
        chosen += [pool[k][i] for i in r.permutation(len(pool[k]))[:n]]
    spare = [e for k in ("missed", "localisation", "duplicate", "class") for e in pool[k] if e not in chosen]
    chosen += spare[: 50 - len(chosen)]
    chosen = [chosen[i] for i in r.permutation(len(chosen))][:50]
    key = {}
    for n, e in enumerate(chosen):
        e["id"] = f"E{n + 1:02d}"
        key[e["id"]] = e.pop("truth")
    write(U.DATA / "supplied" / "detector_errors.json", chosen, 1)
    write(U.DATA / "supplied" / "detector_errors_key.json", key, 1)
    print("errors written:", {k: list(key.values()).count(k) for k in sorted(set(key.values()))},
          "| background false alarms on validation:", len(pool["background"]))


if __name__ == "__main__":
    for stage in sys.argv[1:]:
        t0 = time.time()
        print("==", stage, flush=True)
        globals()[stage]()
        print(f"== {stage} done in {(time.time() - t0) / 60:.1f} min", flush=True)
