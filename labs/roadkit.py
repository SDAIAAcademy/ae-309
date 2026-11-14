"""roadkit: the shared helpers of the Computer Vision Engineering labs (import roadkit as U).

Data access for the Al-Manar pack, the three compact models of the course (a crop classifier,
an anchor-free detector and a crack segmenter, all on a ResNet-18 trunk), training and
prediction loops, detection metrics, and small utilities. The labs keep the interesting lines
for you; this module holds what you have written before (AE-201, AE-202).
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

import prepare_data as P

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
ROADS = CACHE / "roads"
CKPT = DATA / "checkpoints"
S, TILE = P.S, P.TILE
TILE_CLASSES, DET_CLASSES, CONDITIONS, DISTRICTS = P.TILE_CLASSES, P.DET_CLASSES, P.CONDITIONS, P.DISTRICTS
CRACK_CLASSES = ("crack_long", "crack_trans", "alligator")

QUICK = os.environ.get("CV_QUICK", "0") == "1"
MODE = "quick" if QUICK else "full"


def pick(full, quick):
    """The full value, or the small one in quick mode (CV_QUICK=1)."""
    return quick if QUICK else full


N_SEG = pick(300, 60)


def ensure_data(log=print):
    """Draw the data pack when it is missing or smaller than this mode needs."""
    man = CACHE / "manifest.json"
    have = json.loads(man.read_text())["segments"] if man.exists() else 0
    if have < N_SEG:
        P.build(N_SEG, log)
    return json.loads(man.read_text())


# ---------------------------------------------------------------- reference results

def reference(name):
    """The reference result of an exercise: what the safety-net cells fall back on."""
    d = json.loads((DATA / "reference" / f"{name}.json").read_text(encoding="utf-8"))
    return d[MODE] if MODE in d else next(iter(d.values()))


def save_reference(name, value):
    """Record a solution's result (only when CV_WRITE_REF=1: the course author's run)."""
    if os.environ.get("CV_WRITE_REF") != "1":
        return
    path = DATA / "reference" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    d[MODE] = value
    path.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- frames and crops

@lru_cache(maxsize=1)
def _frames():
    ensure_data()
    rows = json.loads((ROADS / "frames.json").read_text(encoding="utf-8"))
    return [f for f in rows if f["address"][0] < N_SEG]


def frames(split=None, key="split"):
    """Road frames as dicts: id, segment (may be None), district, split, split_random,
    condition, severity, blur, view, objects."""
    return [f for f in _frames() if split is None or f[key] == split]


@lru_cache(maxsize=1)
def _tiles():
    by_id = {f["id"]: f for f in _frames()}
    rows = json.loads((ROADS / "tiles.json").read_text(encoding="utf-8"))
    out = []
    for t in rows:
        f = by_id.get(t["frame"])
        if f is None:
            continue
        out.append({**t, **{k: f[k] for k in ("segment", "district", "split", "split_random",
                                               "condition", "blur", "view", "address")},
                    "y": TILE_CLASSES.index(t["label"])})
    return out


def tiles(split=None, key="split"):
    """Defect crops as dicts: id, frame, x0, y0, label, y (class index) and the frame's fields."""
    return [t for t in _tiles() if split is None or t[key] == split]


def frame_image(fid):
    return np.asarray(Image.open(ROADS / "frames" / f"{fid}.jpg").convert("RGB"))


def frame_inst(fid):
    return np.asarray(Image.open(ROADS / "inst" / f"{fid}.png"))


def frame_images(rows):
    return np.stack([frame_image(f["id"]) for f in rows])


def tile_images(rows):
    """The pixels of a list of crops: uint8, N x 96 x 96 x 3."""
    out = np.zeros((len(rows), TILE, TILE, 3), np.uint8)
    last, img = None, None
    for i, t in enumerate(rows):
        if t["frame"] != last:
            last, img = t["frame"], frame_image(t["frame"])
        out[i] = img[t["y0"]:t["y0"] + TILE, t["x0"]:t["x0"] + TILE]
    return out


def tile_masks(rows):
    """For each crop, the mask of its own defect (all False for a clear crop)."""
    out = np.zeros((len(rows), TILE, TILE), bool)
    for i, t in enumerate(rows):
        if t["iid"]:
            inst = frame_inst(t["frame"])
            out[i] = inst[t["y0"]:t["y0"] + TILE, t["x0"]:t["x0"] + TILE] == t["iid"]
    return out


def labels(rows):
    return np.array([t["y"] for t in rows])


def det_gt(objects):
    """Ground-truth boxes of a frame for the detector: rows of x0, y0, x1, y1, class index."""
    g = [o["box"] + [DET_CLASSES.index(o["cls"])] for o in objects if o["cls"] in DET_CLASSES]
    return np.array(g, np.float32).reshape(-1, 5)


def crack_mask(frame, inst=None):
    """The crack mask of a frame from its instance map."""
    inst = frame_inst(frame["id"]) if inst is None else inst
    ids = [o["iid"] for o in frame["objects"] if o["cls"] in CRACK_CLASSES]
    return np.isin(inst, ids)


def field_frames(condition, n=None):
    """Test frames drawn again under one field condition: 'rain', 'glare', 'night' at high
    severity, 'pole' for the moved camera, or 'clean' for the same frames in daylight.
    Returns (images, objects per frame, instance maps): the same road spots every time."""
    rows = frames("test")[: n or pick(240, 60)]
    kw = {"clean": dict(condition="day", severity=0.0, blur=0),
          "pole": dict(condition="day", severity=0.0, blur=0, view="pole")}.get(
              condition, dict(condition=condition, blur=0))
    imgs, objs, insts = [], [], []
    for f in rows:
        im, inst, ob, _ = P.render(*f["address"], **kw)
        imgs.append(im), objs.append(ob), insts.append(inst)
    return np.stack(imgs), objs, np.stack(insts)


def field_tiles(n=200):
    """The field batch of lab 1.1: n test crops drawn again as the roadside sees them, each
    with a lighting condition, sometimes motion blur, sometimes from a pole camera."""
    imgs, rows = [], []
    for k, t in enumerate(tiles("test")):
        r = np.random.default_rng([P.SEED, 60, k])
        cond = CONDITIONS[int(r.integers(5))]
        blur = 9 if r.random() < 0.25 else 0
        view = "pole" if r.random() < 0.25 else "vehicle"
        im, inst, objs, _ = P.render(*t["address"], condition=cond, severity=0.0 if cond == "day" else 0.8,
                                     blur=blur, view=view)
        x0, y0 = t["x0"], t["y0"]
        if t["iid"]:
            o = [o for o in objs if o["iid"] == t["iid"]]
            if not o:
                continue
            b = o[0]["box"]
            x0 = int(np.clip((b[0] + b[2]) // 2 - TILE // 2, 0, S - TILE))
            y0 = int(np.clip((b[1] + b[3]) // 2 - TILE // 2, 0, S - TILE))
        elif view == "pole" or (inst[y0:y0 + TILE, x0:x0 + TILE] > 0).mean() > 0.01:
            continue
        imgs.append(im[y0:y0 + TILE, x0:x0 + TILE])
        rows.append({"id": t["id"], "label": t["label"], "y": t["y"], "district": t["district"],
                     "lighting": cond, "blur": "blurred" if blur else "sharp", "viewpoint": view})
        if len(rows) == n:
            break
    return np.stack(imgs), rows


# ---------------------------------------------------------------- other packs

def annot_batch():
    return json.loads((CACHE / "annot" / "batch.json").read_text(encoding="utf-8"))


def annot_truth():
    return json.loads((CACHE / "annot" / "truth.json").read_text(encoding="utf-8"))


def annot_image(aid):
    return np.asarray(Image.open(CACHE / "annot" / f"{aid}.png").convert("RGB"))


def clips():
    ensure_data()
    return json.loads((CACHE / "traffic" / "clips.json").read_text(encoding="utf-8"))


def clip_frame(clip, t, boxes=None):
    """Draw frame t of a traffic clip (the clips are stored as tracks, not as video)."""
    from PIL import ImageDraw
    night = clip["kind"] == "night"
    im = Image.new("RGB", (clip["width"], clip["height"]), (28, 30, 38) if night else (104, 104, 108))
    d = ImageDraw.Draw(im)
    for x in (20, 60, 100, 140):
        d.line([(x, 0), (x, clip["height"])], fill=(70, 70, 76) if night else (200, 200, 196))
    for g in clip["gt"]:
        b = g["boxes"].get(str(t)) or g["boxes"].get(t)
        if b:
            shade = 60 + (g["tid"] * 47) % 150
            d.rectangle(b, fill=(shade // 3, shade // 3, shade // 2) if night else (shade, 90, 200 - shade))
    if clip["occluder"]:
        d.rectangle([0, clip["occluder"][0], clip["width"], clip["occluder"][1]], fill=(40, 70, 60))
    d.line([(0, clip["line_y"]), (clip["width"], clip["line_y"])], fill=(240, 200, 40))
    for b in boxes or []:
        d.rectangle(list(b[:4]), outline=(255, 80, 80))
    return np.asarray(im)


def street():
    ensure_data()
    return json.loads((CACHE / "street" / "street.json").read_text(encoding="utf-8"))


def street_image(fid):
    return np.asarray(Image.open(CACHE / "street" / "frames" / f"{fid}.png").convert("RGB"))


def camera_stream():
    ensure_data()
    return json.loads((CACHE / "stream" / "camera_stream.json").read_text(encoding="utf-8"))


def detail(img, box):
    """How much fine detail a region holds: the mean absolute Laplacian of its grey levels.
    A face or a plate that can still be read keeps most of its detail after 'blurring'."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    g = np.asarray(img, np.float32)[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1].mean(-1)
    if min(g.shape) < 3:
        return 0.0
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.abs(lap).mean())


# ---------------------------------------------------------------- statistics and timing

def box_iou(a, b):
    """IoU of every box of a (n x 4) with every box of b (m x 4): an n x m array."""
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    x0, y0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x1 - x0).clip(0) * (y1 - y0).clip(0)
    area = lambda q: (q[:, 2] - q[:, 0]) * (q[:, 3] - q[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def boot_ci(values, n_boot=1000, seed=0):
    """Mean and 95% percentile bootstrap interval of a list of per-item scores."""
    v = np.asarray(values, np.float64)
    r = np.random.default_rng(seed)
    means = v[r.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def latency_ms(fn, n=40, warmup=5):
    """Time fn() n times after a warm-up; returns the list of milliseconds."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return np.array(out)


def tp_fp(preds, gts, cls, iou_thr=0.5):
    """Match detections of one class to ground truth over a list of frames.
    preds[i]: rows of x0, y0, x1, y1, score, class. gts[i]: rows of x0, y0, x1, y1, class.
    Returns (scores, is_tp, n_gt) with detections sorted by falling score: each ground-truth
    box is given to the best-scoring detection that overlaps it by at least iou_thr."""
    rec, n_gt = [], 0
    for i, (p, g) in enumerate(zip(preds, gts)):
        g = g[g[:, 4] == cls]
        n_gt += len(g)
        rec += [(float(q[4]), i, q[:4], g) for q in p[p[:, 5] == cls]]
    rec.sort(key=lambda q: -q[0])
    used, scores, hits = {}, [], []
    for s, i, b, g in rec:
        ok = False
        if len(g):
            iou = box_iou(b[None], g[:, :4])[0]
            iou[list(used.get(i, []))] = -1
            j = int(iou.argmax())
            if iou[j] >= iou_thr:
                used.setdefault(i, set()).add(j)
                ok = True
        scores.append(s), hits.append(ok)
    return np.array(scores), np.array(hits, bool), n_gt


def ap_from(scores, is_tp, n_gt):
    """Average precision: the area under the precision-recall curve, precision made
    monotone from the right (the all-point interpolation used by COCO and VOC 2010+)."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    tp = np.cumsum(is_tp)
    rec = np.concatenate([[0], tp / n_gt])
    prec = np.concatenate([[1], tp / np.arange(1, len(tp) + 1)])
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def precision_recall(preds, gts, iou_thr=0.5):
    """Precision and recall of the detections as they stand (no sweep over scores), over all
    classes: returns (precision, recall, true positives, false positives, missed)."""
    tp = fp = n = 0
    for c in range(len(DET_CLASSES)):
        _, hit, n_gt = tp_fp(preds, gts, c, iou_thr)
        tp, fp, n = tp + int(hit.sum()), fp + int((~hit).sum()), n + n_gt
    return tp / max(tp + fp, 1), tp / max(n, 1), tp, fp, n - tp


def mean_ap(preds, gts, iou_thr=0.5):
    """Mean over the detector's classes of the average precision at one IoU threshold."""
    return float(np.mean([ap_from(*tp_fp(preds, gts, c, iou_thr)) for c in range(len(DET_CLASSES))]))


def boot_ap(preds, gts, iou_thr=0.5, n_boot=200, seed=0):
    """Mean AP with a 95% bootstrap interval over frames."""
    r = np.random.default_rng(seed)
    n = len(preds)
    vals = []
    for _ in range(n_boot):
        idx = r.integers(0, n, n)
        vals.append(mean_ap([preds[i] for i in idx], [gts[i] for i in idx], iou_thr))
    return mean_ap(preds, gts, iou_thr), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- plotting

def show(images, titles=None, cols=8, boxes=None, size=1.6):
    """A grid of images; boxes[i] is an optional list of x0, y0, x1, y1 (, label) per image."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size * 1.12))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i], cmap="gray")
        if titles is not None:
            ax.set_title(str(titles[i]), fontsize=7)
        for b in (boxes[i] if boxes is not None else []):
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                   color=["#ff5a4f", "#12b5a6", "#f5b700"][int(b[5]) % 3 if len(b) > 5 else 0], lw=1.2))
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- models (PyTorch)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(P.SEED)


class Norm(nn.Module):
    """ImageNet normalisation inside the model: the input is RGB in 0..1."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def _resnet18(pretrained):
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return torchvision.models.resnet18(weights=w)


class TileNet(nn.Module):
    """The crop classifier: a ResNet-18 trunk to stride 8, global average pooling, one linear layer."""

    def __init__(self, n_classes=5, pretrained=False):
        super().__init__()
        m = _resnet18(pretrained)
        self.norm = Norm()
        self.trunk = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2)
        self.head = nn.Linear(128, n_classes)

    def embed(self, x):
        return self.trunk(self.norm(x)).mean((2, 3))

    def forward(self, x):
        return self.head(self.embed(x))


class RoadDet(nn.Module):
    """The pothole and patch detector: anchor-free, one prediction per 8 x 8 pixel cell.
    Each cell gives a score per class and the distances from its centre to the four sides of
    a box (left, top, right, bottom, in cells). decode() turns the two maps into boxes."""

    stride = 8

    def __init__(self, n_classes=2, pretrained=False):
        super().__init__()
        m = _resnet18(pretrained)
        self.norm = Norm()
        self.trunk = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2)
        self.neck = nn.Sequential(
            nn.Conv2d(128, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.cls = nn.Conv2d(96, n_classes, 3, padding=1)
        self.box = nn.Conv2d(96, 4, 3, padding=1)
        nn.init.constant_(self.cls.bias, -4.6)
        nn.init.constant_(self.box.bias, 2.0)

    def forward(self, x):
        f = self.neck(self.trunk(self.norm(x)))
        return self.cls(f), F.relu(self.box(f))


class RoadSeg(nn.Module):
    """The crack segmenter: the same trunk as an encoder and a two-step decoder with skips.
    Returns one logit per pixel."""

    def __init__(self, pretrained=False):
        super().__init__()
        m = _resnet18(pretrained)
        self.norm = Norm()
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)
        self.pool, self.l1, self.l2 = m.maxpool, m.layer1, m.layer2
        block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.up2, self.d1, self.d0 = block(128, 64), block(128, 64), block(128, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s = self.stem(self.norm(x))
        a = self.l1(self.pool(s))
        b = self.l2(a)
        u = self.d1(torch.cat([F.interpolate(self.up2(b), size=a.shape[2:], mode="bilinear", align_corners=False), a], 1))
        u = self.d0(torch.cat([F.interpolate(u, size=s.shape[2:], mode="bilinear", align_corners=False), s], 1))
        return F.interpolate(self.out(u), size=x.shape[2:], mode="bilinear", align_corners=False)


BACKBONES = {"resnet18": "torchvision resnet18, ImageNet-1k weights",
             "vit_tiny": "timm vit_tiny_patch16_224.augreg_in21k_ft_in1k"}


def backbone(name, n_classes=5, pretrained=True):
    """One of the two pretrained backbones of Day 2 with a new n_classes head, input 128 x 128.
    The head is model[1].fc for the ResNet and model[1].head for the vision transformer."""
    if name == "resnet18":
        m = _resnet18(pretrained)
        m.fc = nn.Linear(512, n_classes)
    else:
        import timm
        m = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained,
                              num_classes=n_classes, img_size=128)
    return nn.Sequential(Norm(), m)


def save_ckpt(model, name):
    """Save weights at half precision, which keeps the supplied checkpoints small."""
    CKPT.mkdir(parents=True, exist_ok=True)
    sd = {k: (v.half() if v.is_floating_point() else v).cpu() for k, v in model.state_dict().items()}
    torch.save(sd, CKPT / f"{name}.pt")


def load_ckpt(model, name):
    sd = torch.load(CKPT / f"{name}.pt", map_location="cpu")
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
    return model.to(DEV).eval()


def to_tensor(images, size=None):
    """uint8 N x H x W x 3 to float N x 3 x H x W in 0..1, resized when size is given."""
    x = torch.from_numpy(np.ascontiguousarray(images)).to(DEV).permute(0, 3, 1, 2).float() / 255
    if size is not None and x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return x


@torch.no_grad()
def predict(model, images, size=None, bs=128, embed=False):
    """Logits (or embeddings) of a classifier for a uint8 image array, as NumPy."""
    model = model.to(DEV).eval()
    out = []
    for i in range(0, len(images), bs):
        x = to_tensor(images[i:i + bs], size)
        out.append((model.embed(x) if embed else model(x)).float().cpu().numpy())
    return np.concatenate(out)


def fit(model, images, y, epochs=1, lr=1e-3, bs=32, size=None, params=None, augment=None, seed=0, log=print):
    """A plain training loop for a classifier: AdamW, cross-entropy, a cosine schedule.
    params: the parameters to train (default: every parameter that requires grad).
    augment: an optional function applied to each float batch (N x 3 x H x W)."""
    model = model.to(DEV).train()
    params = [p for p in (params or model.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * ((len(images) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 2), pct_start=0.15)
    r = np.random.default_rng(seed)
    yt = torch.as_tensor(np.asarray(y)).long()
    for ep in range(epochs):
        order, total = r.permutation(len(images)), 0.0
        for i in range(0, len(order), bs):
            idx = np.sort(order[i:i + bs])
            x = to_tensor(images[idx], size)
            if augment is not None:
                x = augment(x)
            loss = F.cross_entropy(model(x), yt[idx].to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        log(f"  epoch {ep + 1}/{epochs}  loss {total / len(order):.3f}")
    return model.eval()


# ---------------------------------------------------------------- detection

def det_targets(gts, hw, n_classes=2, stride=8):
    """Training targets for RoadDet. A cell is positive for a box when its centre falls in
    the middle of the box; it then learns that box's class and its four distances."""
    H, W = hw[0] // stride, hw[1] // stride
    B = len(gts)
    cls_t = torch.zeros(B, n_classes, H, W)
    box_t = torch.zeros(B, 4, H, W)
    cy, cx = np.meshgrid((np.arange(H) + 0.5) * stride, (np.arange(W) + 0.5) * stride, indexing="ij")
    for b, g in enumerate(gts):
        for x0, y0, x1, y1, c in sorted(g.tolist(), key=lambda q: -(q[2] - q[0]) * (q[3] - q[1])):
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = max(stride * 0.75, 0.3 * (x1 - x0)), max(stride * 0.75, 0.3 * (y1 - y0))
            m = (np.abs(cx - mx) <= rx) & (np.abs(cy - my) <= ry) & (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
            if not m.any():
                j = np.argmin((cx - mx) ** 2 + (cy - my) ** 2)
                m.flat[j] = True
            mt = torch.from_numpy(m)
            cls_t[b, :, mt] = 0
            cls_t[b, int(c), mt] = 1
            ltrb = np.stack([cx - x0, cy - y0, x1 - cx, y1 - cy]) / stride
            box_t[b, :, mt] = torch.from_numpy(ltrb[:, m]).float().clamp(min=0.05)
    return cls_t, box_t


def _cells_to_boxes(ltrb, stride=8):
    """Distance maps (B x 4 x H x W, in cells) to boxes (B x H x W x 4, in pixels)."""
    B, _, H, W = ltrb.shape
    cy = ((torch.arange(H, device=ltrb.device) + 0.5) * stride).view(1, H, 1)
    cx = ((torch.arange(W, device=ltrb.device) + 0.5) * stride).view(1, 1, W)
    l, t, r, b = (ltrb * stride).unbind(1)
    return torch.stack([cx - l, cy - t, cx + r, cy + b], -1)


def det_loss(cls, box, cls_t, box_t):
    """Focal loss on the class map plus a generalised-IoU loss on the positive cells."""
    from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss
    pos = cls_t.sum(1) > 0
    n = pos.sum().clamp(min=1)
    l_cls = sigmoid_focal_loss(cls, cls_t, alpha=0.25, gamma=2.0, reduction="sum") / n
    pb, tb = _cells_to_boxes(box)[pos], _cells_to_boxes(box_t)[pos]
    l_box = generalized_box_iou_loss(pb, tb, reduction="sum") / n
    return l_cls + l_box, float(l_cls.detach()), float(l_box.detach())


def decode(cls, box, score_thr=0.05, nms_iou=0.5, topk=100, stride=8):
    """Class and distance maps to detections. Returns one array per image with rows of
    x0, y0, x1, y1, score, class. nms_iou=None skips non-maximum suppression."""
    from torchvision.ops import batched_nms
    cls, box = torch.as_tensor(cls).float(), torch.as_tensor(box).float()
    prob = cls.sigmoid()
    boxes = _cells_to_boxes(box, stride)
    out = []
    for b in range(prob.shape[0]):
        p = prob[b].permute(1, 2, 0)                      # H x W x C
        keep = p > score_thr
        idx = keep.nonzero()
        sc = p[keep]
        bx = boxes[b][idx[:, 0], idx[:, 1]]
        lb = idx[:, 2]
        if len(sc) > 4 * topk:
            top = sc.topk(4 * topk).indices
            sc, bx, lb = sc[top], bx[top], lb[top]
        if nms_iou is not None and len(sc):
            k = batched_nms(bx, sc, lb, nms_iou)[:topk]
            sc, bx, lb = sc[k], bx[k], lb[k]
        out.append(torch.cat([bx, sc[:, None], lb[:, None].float()], 1).cpu().numpy())
    return out


@torch.no_grad()
def detect(model, images, size=None, bs=32, **kw):
    """Run RoadDet on a uint8 image array, at another input size when asked; boxes are
    returned in the pixels of the original frame."""
    model = model.to(DEV).eval()
    out = []
    for i in range(0, len(images), bs):
        cls, box = model(to_tensor(images[i:i + bs], size))
        dets = decode(cls.cpu(), box.cpu(), **kw)
        if size is not None and size != images.shape[1]:
            for d in dets:
                d[:, :4] *= images.shape[1] / size
        out += dets
    return out


def fit_detector(model, images, gts, epochs=1, lr=1e-3, bs=16, seed=0, flip=True, log=print):
    """A short training schedule for RoadDet (AdamW, one cycle). Mirrors frames at random."""
    model = model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(images) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 2), pct_start=0.15)
    r = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        order, tot = r.permutation(len(images)), np.zeros(3)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            x = images[idx]
            g = [gts[j].copy() for j in idx]
            if flip and r.random() < 0.5:
                x = x[:, :, ::-1]
                for q in g:
                    q[:, [0, 2]] = x.shape[2] - q[:, [2, 0]]
            cls, box = model(to_tensor(x))
            ct, bt = det_targets(g, x.shape[1:3])
            loss, lc, lb = det_loss(cls, box, ct.to(DEV), bt.to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += np.array([float(loss.detach()), lc, lb]) * len(idx)
        tot /= len(order)
        hist.append(tot.tolist())
        log(f"  epoch {ep + 1}/{epochs}  loss {tot[0]:.3f} (class {tot[1]:.3f}, box {tot[2]:.3f})")
    model.eval()
    return hist


# ---------------------------------------------------------------- segmentation

@torch.no_grad()
def segment(model, images, bs=16):
    """Crack probability per pixel for a uint8 image array: float N x H x W."""
    model = model.to(DEV).eval()
    out = [model(to_tensor(images[i:i + bs])).sigmoid()[:, 0].cpu().numpy() for i in range(0, len(images), bs)]
    return np.concatenate(out)


def fit_segmenter(model, images, masks, epochs=1, lr=1e-3, bs=8, seed=0, log=print):
    """A short schedule for RoadSeg: binary cross-entropy plus a Dice term, random mirroring."""
    model = model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(images) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 2), pct_start=0.15)
    r = np.random.default_rng(seed)
    for ep in range(epochs):
        order, tot = r.permutation(len(images)), 0.0
        for i in range(0, len(order), bs):
            idx = np.sort(order[i:i + bs])
            x, m = images[idx], masks[idx]
            if r.random() < 0.5:
                x, m = x[:, :, ::-1], m[:, :, ::-1]
            logit = model(to_tensor(x))[:, 0]
            t = torch.from_numpy(np.ascontiguousarray(m)).float().to(DEV)
            p = logit.sigmoid()
            dice = 1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
            loss = F.binary_cross_entropy_with_logits(logit, t, pos_weight=torch.tensor(4.0, device=DEV)) + dice.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        log(f"  epoch {ep + 1}/{epochs}  loss {tot / len(order):.3f}")
    return model.eval()


# ---------------------------------------------------------------- error analysis (lab 3.2)

def render_frame(address, **how):
    """Draw a frame again from its address: (image, ground-truth boxes for the detector).
    how: the condition, severity, blur and view to draw it under (default: as it was filmed)."""
    img, _, objs, _ = P.render(*address, **how)
    return img, det_gt(objs)


def field_mix(split="val", n=None):
    """Frames of a split as the roadside prototype met them: each drawn again under a
    lighting condition chosen at random (day, glare, dust, rain, night, at high severity),
    one in five from a pole camera. Returns (images, objects per frame, how each was drawn)."""
    imgs, objs, hows = [], [], []
    for f in frames(split)[: n or len(frames(split))]:
        r = np.random.default_rng([P.SEED, 61, P.frame_number(*f["address"])])
        cond = CONDITIONS[int(r.integers(5))]
        how = {"condition": cond, "severity": 0.0 if cond == "day" else 0.8, "blur": 0,
               "view": "pole" if r.random() < 0.2 else "vehicle"}
        im, _, ob, _ = P.render(*f["address"], **how)
        imgs.append(im), objs.append(ob), hows.append(how)
    return np.stack(imgs), objs, hows


def oracle_fix(preds, gts, kind):
    """Detections as they would be if one error type were fixed by an oracle (after TIDE).
    kind: 'duplicate', 'class', 'localisation' or 'missed'. The other errors stay."""
    out = []
    for p, g in zip(preds, gts):
        p = p[np.argsort(-p[:, 4])].copy()
        keep, used = np.ones(len(p), bool), set()
        iou = box_iou(p[:, :4], g[:, :4]) if len(g) and len(p) else np.zeros((len(p), len(g)))
        for i in range(len(p)):
            if not len(g):
                break
            j = int(iou[i].argmax())
            same = g[j, 4] == p[i, 5]
            if iou[i, j] >= 0.5 and same:
                if j in used and kind == "duplicate":
                    keep[i] = False
                used.add(j)
            elif iou[i, j] >= 0.5 and kind == "class":
                p[i, 5] = g[j, 4]
            elif 0.1 <= iou[i, j] < 0.5 and same and kind == "localisation":
                p[i, :4] = g[j, :4]
        p = p[keep]
        if kind == "missed" and len(g):
            best = box_iou(g[:, :4], p[:, :4]).max(1) if len(p) else np.zeros(len(g))
            add = g[best < 0.1]
            p = np.concatenate([p, np.concatenate([add[:, :4], np.ones((len(add), 1), np.float32), add[:, 4:5]], 1)])
        out.append(p)
    return out


# ---------------------------------------------------------------- export and runtime (Day 4)

def export_onnx(model, path, size=S):
    """Export RoadDet to ONNX with a dynamic batch and input size. Returns the path."""
    model = model.cpu().eval()
    torch.onnx.export(model, torch.rand(1, 3, size, size), str(path), input_names=["image"],
                      output_names=["cls", "box"], opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                                    "cls": {0: "batch", 2: "h", 3: "w"}, "box": {0: "batch", 2: "h", 3: "w"}})
    return path


def onnx_session(path, threads=None):
    import onnxruntime as ort
    opt = ort.SessionOptions()
    if threads:
        opt.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])


def to_input(images, size=None):
    """uint8 N x H x W x 3 frames to the float32 N x 3 x size x size array the model expects."""
    return to_tensor(images, size).cpu().numpy()


def detect_onnx(sess, images, size=None, bs=16, **kw):
    """detect() for an ONNX Runtime session: boxes in the pixels of the original frame."""
    out = []
    for i in range(0, len(images), bs):
        cls, box = sess.run(None, {"image": to_input(images[i:i + bs], size)})
        dets = decode(cls, box, **kw)
        if size is not None and size != images.shape[1]:
            for d in dets:
                d[:, :4] *= images.shape[1] / size
        out += dets
    return out
