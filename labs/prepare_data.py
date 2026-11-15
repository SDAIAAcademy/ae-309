"""The Al-Manar road data: a deterministic generator for every lab of the course.

    uv run python prepare_data.py              # the full pack: 300 road segments, 6,000 frames
    uv run python prepare_data.py --segments 60   # the quick pack: 1,200 frames

Everything is synthetic and drawn with NumPy and Pillow from one seed, so every participant
gets the same pixels and the same planted problems. Nothing here is a real road, camera or
person. The pack is written to labs/cache/ (not committed): road frames with instance maps,
defect crops, an annotation batch for a new class, traffic clips with ground-truth tracks,
street frames with faces and plates, and a 30-day stream of camera statistics.

A frame is addressed by (segment, spot, frame): a road segment has 4 spots, a spot is filmed
5 times from nearly the same place. render() is a pure function of that address, so the quick
pack is exactly the first 60 segments of the full one, and a test frame can be rendered again
under rain, glare, night or from a moved camera.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SEED = 309
S = 192          # side of a road frame, pixels
C = 256          # side of the canvas a spot is drawn on
TILE = 96        # side of a defect crop
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
ROADS = CACHE / "roads"

DISTRICTS = ["Al-Qalaa", "Al-Bustan", "Al-Rabwa", "Al-Sinaiya", "Al-Corniche"]
CONDITIONS = ["day", "glare", "dust", "rain", "night"]
TILE_CLASSES = ["clear", "pothole", "crack_long", "crack_trans", "patch"]
DET_CLASSES = ["pothole", "patch"]
N_SEGMENTS, N_SPOTS, N_FRAMES = 300, 4, 5


def rng(*key):
    return np.random.default_rng([SEED, *key])


def lowfreq(r, n, size):
    """Smooth noise: an n by n grid of normals, enlarged to size by size."""
    g = (r.normal(size=(n, n)) * 40 + 128).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(g).resize((size, size), Image.BICUBIC)
    return (np.asarray(im, dtype=np.float32) - 128) / 40


# ---------------------------------------------------------------- segments and spots

def segment(seg: int) -> dict:
    r = rng(1, seg)
    d = int(r.choice(5, p=[0.24, 0.20, 0.20, 0.18, 0.18]))
    wear = [0.95, 0.50, 0.45, 0.75, 0.60][d] + float(r.uniform(-0.15, 0.15))
    m = seg % 20
    return {
        "seg": seg, "id": f"SEG-{seg:03d}", "d": d, "district": DISTRICTS[d],
        "base": float(r.uniform(80, 128)), "tint": r.normal(0, 4, 3).tolist(),
        "grain": float(r.uniform(5, 9)), "wear": wear,
        "lane_x": float(r.uniform(40, C - 40)) if r.random() < 0.7 else None,
        "lane_yellow": bool(r.random() < 0.3), "lane_slope": float(r.uniform(-0.12, 0.12)),
        "split": "test" if m < 3 else "val" if m < 6 else "train",
    }


def _line_mask(points, width):
    im = Image.new("L", (C, C), 0)
    ImageDraw.Draw(im).line([tuple(p) for p in points], fill=255, width=width, joint="curve")
    return np.asarray(im) > 0


def _crack_points(r, cx, cy, length, along_y, wobble=5.0):
    n = max(3, int(length / 10))
    t = np.linspace(-length / 2, length / 2, n)
    side = np.cumsum(r.normal(0, wobble, n))
    side -= side.mean()
    pts = np.stack([cx + side, cy + t], 1) if along_y else np.stack([cx + t, cy + side], 1)
    return pts.clip(2, C - 3)


def _crack_mask(r, cx, cy, length, along_y, width):
    m = _line_mask(_crack_points(r, cx, cy, length, along_y), width)
    if r.random() < 0.4:  # a short branch
        k = _crack_points(r, cx + r.uniform(-8, 8), cy + r.uniform(-8, 8), length * 0.25, not along_y, 3.0)
        m |= _line_mask(k, max(2, width - 1))
    return m


def _network_mask(r, cx, cy, radius, nx, ny):
    """A network of cracks on a jittered nx by ny grid: (nx-1)*(ny-1) closed cells."""
    xs = np.linspace(-radius, radius, nx) if nx > 1 else np.zeros(1)
    ys = np.linspace(-radius, radius, ny) if ny > 1 else np.zeros(1)
    jit = radius / max(nx, ny) * 0.45
    pts = np.array([[[cx + x + r.normal(0, jit), cy + y + r.normal(0, jit)] for x in xs] for y in ys])
    m = np.zeros((C, C), bool)
    for j in range(ny):
        for i in range(nx):
            for dj, di in ((0, 1), (1, 0)):
                if j + dj < ny and i + di < nx:
                    a, b = pts[j, i], pts[j + dj, i + di]
                    mid = (a + b) / 2 + r.normal(0, jit * 0.6, 2)
                    m |= _line_mask(np.array([a, mid, b]).clip(2, C - 3), 2)
    return m


def _blob(r, cx, cy, a, b):
    yy, xx = np.mgrid[0:C, 0:C].astype(np.float32)
    th = r.uniform(0, math.pi)
    xr = (xx - cx) * math.cos(th) + (yy - cy) * math.sin(th)
    yr = -(xx - cx) * math.sin(th) + (yy - cy) * math.cos(th)
    phi = np.arctan2(yr / b, xr / a)
    wob = 1 + 0.15 * np.sin(3 * phi + r.uniform(0, 6)) + 0.10 * np.sin(5 * phi + r.uniform(0, 6))
    return np.hypot(xr / a, yr / b) / wob, xr / a


def _box(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


@lru_cache(maxsize=4)
def spot_canvas(seg: int, spot: int):
    """The canvas of one spot: the image, the instance map and the objects drawn on it."""
    sg = segment(seg)
    r = rng(2, seg, spot)
    g = sg["base"] + 9 * lowfreq(r, 6, C) + 5 * lowfreq(r, 24, C) + r.normal(0, sg["grain"], (C, C))
    img = np.stack([g + t for t in sg["tint"]], -1).astype(np.float32)
    if sg["lane_x"] is not None:
        x0 = sg["lane_x"]
        m = _line_mask([(x0, 0), (x0 + sg["lane_slope"] * C, C)], 7)
        paint = np.array([208, 178, 70] if sg["lane_yellow"] else [212, 212, 208], np.float32)
        worn = (0.75 + 0.2 * lowfreq(r, 10, C)).clip(0.4, 1)[..., None]
        img = np.where(m[..., None], img * (1 - worn) + paint * worn, img)
    inst = np.zeros((C, C), np.uint8)
    objects, taken = [], []

    def place(radius):
        for _ in range(30):
            cx, cy = r.uniform(34, C - 34, 2)
            if all(math.hypot(cx - x, cy - y) > 0.8 * (radius + q) for x, y, q in taken):
                taken.append((cx, cy, radius))
                return float(cx), float(cy)
        return None

    n = int(np.clip(r.poisson(sg["wear"] * 1.7), 0, 4))
    kinds = list(r.choice(["pothole", "crack_long", "crack_trans", "patch", "alligator"], n,
                          p=[0.30, 0.22, 0.20, 0.21, 0.07]))
    if r.random() < 0.14:
        kinds.append("manhole")
    if r.random() < 0.3:
        kinds.append("shadow")
    for kind in kinds:
        iid = len(objects) + 1
        if kind == "pothole":
            rad = float(r.uniform(6, 24))
            c = place(rad)
            if c is None:
                continue
            d, lit = _blob(r, c[0], c[1], rad * r.uniform(0.85, 1.3), rad * r.uniform(0.7, 1.0))
            m = d < 1
            depth = float(r.uniform(0, 1))  # a shallow pothole is barely darker than the road
            shade = (0.36 + 0.3 * depth + 0.22 * d ** 2 + 0.12 * lit).clip(0.25, 0.92)
            img = np.where(m[..., None], img * shade[..., None], img)
            if depth < 0.7:
                img = np.where(((d >= 1) & (d < 1.2))[..., None], img * 1.13, img)
            if r.random() < 0.2:  # a second pothole right beside the first
                kinds.append(("pothole_near", c, rad))
        elif isinstance(kind, tuple):
            (cx0, cy0), rad0 = kind[1], kind[2]
            rad = float(r.uniform(5, max(6, rad0)))
            ang = r.uniform(0, 2 * math.pi)
            cx = float(np.clip(cx0 + (rad0 + rad) * 0.95 * math.cos(ang), 12, C - 12))
            cy = float(np.clip(cy0 + (rad0 + rad) * 0.95 * math.sin(ang), 12, C - 12))
            d, lit = _blob(r, cx, cy, rad, rad * r.uniform(0.7, 1.0))
            m = (d < 1) & (inst == 0)
            if m.sum() < 20:
                continue
            img = np.where(m[..., None], img * (0.4 + 0.25 * d ** 2 + 0.1 * lit).clip(0.25, 0.9)[..., None], img)
            kind = "pothole"
        elif kind in ("crack_long", "crack_trans"):
            length = float(r.uniform(70, 170))
            c = place(length * 0.3)
            if c is None:
                continue
            m = _crack_mask(r, c[0], c[1], length, kind == "crack_long", int(r.integers(2, 5)))
            img = np.where(m[..., None], img * r.uniform(0.42, 0.66), img)
        elif kind == "alligator":
            rad = float(r.uniform(26, 40))
            c = place(rad)
            if c is None:
                continue
            m = _network_mask(r, c[0], c[1], rad, int(r.integers(3, 5)), int(r.integers(3, 5)))
            img = np.where(m[..., None], img * 0.45, img)
        elif kind == "patch":
            w, h = r.uniform(30, 70, 2)
            round_patch = r.random() < 0.3  # a small round repair: easy to take for a pothole
            if round_patch:
                w, h = r.uniform(22, 44, 2)
            c = place(max(w, h) * 0.6)
            if c is None:
                continue
            if round_patch:
                d, _ = _blob(r, c[0], c[1], w / 2, h / 2)
                m = d < 1
                fill = sg["base"] * r.uniform(0.58, 0.74) + 3 * lowfreq(r, 20, C) + r.normal(0, 2.5, (C, C))
                img = np.where(m[..., None], fill[..., None] + np.array(sg["tint"], np.float32), img)
                img = np.where(((d >= 0.9) & (d < 1))[..., None], img * 0.7, img)
                inst[m] = iid
                objects.append({"iid": iid, "cls": kind, "area": int(m.sum()), "box": _box(m)})
                continue
            th = r.uniform(-0.5, 0.5)
            cs = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
            rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            poly = [tuple(p) for p in (cs @ rot.T + c)]
            pm = Image.new("L", (C, C), 0)
            ImageDraw.Draw(pm).polygon(poly, fill=255)
            m = np.asarray(pm) > 0
            sm = Image.new("L", (C, C), 0)
            ImageDraw.Draw(sm).line(poly + [poly[0]], fill=255, width=2)
            fill = sg["base"] * r.uniform(0.62, 0.8) + 3 * lowfreq(r, 20, C) + r.normal(0, 2.5, (C, C))
            img = np.where(m[..., None], fill[..., None] + np.array(sg["tint"], np.float32), img)
            img = np.where((np.asarray(sm) > 0)[..., None], img * 0.6, img)
        elif kind == "manhole":
            rad = float(r.uniform(11, 17))
            c = place(rad)
            if c is None:
                continue
            d, _ = _blob(rng(7), c[0], c[1], rad, rad)
            yy, xx = np.mgrid[0:C, 0:C]
            m = d < 1
            tex = 92 + 14 * (((xx // 4) + (yy // 4)) % 2) - 34 * ((d > 0.82) & (d < 1))
            img = np.where(m[..., None], tex[..., None].astype(np.float32), img)
        else:  # shadow: darkens, no edge, never labelled
            c = place(30)
            if c is None:
                continue
            d, _ = _blob(rng(7), c[0], c[1], r.uniform(14, 44), r.uniform(12, 30))
            img = img * (1 - r.uniform(0.35, 0.55) * np.exp(-d ** 2 * 1.4))[..., None]
            m = d < 0.8
        if kind != "shadow":
            inst[m & (inst == 0)] = iid
        objects.append({"iid": iid, "cls": kind, "area": int(m.sum()), "box": _box(m)})
    return img, inst, objects


# ---------------------------------------------------------------- conditions and views

def _gauss(cx, cy, sx, sy):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    return np.exp(-((xx - cx) ** 2 / (2 * sx ** 2) + (yy - cy) ** 2 / (2 * sy ** 2)))


def _pil_blur(f, radius):
    im = Image.fromarray(f.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(im, dtype=np.float32)


def apply_condition(f, condition, sev, r):
    if condition == "glare":
        hot = _gauss(r.uniform(30, S - 30), r.uniform(30, S - 30), 32 + 22 * sev, 32 + 22 * sev)
        f = f * (1 - 0.4 * sev) + 255 * 0.22 * sev + 255 * hot[..., None] * (0.35 + 0.35 * sev)
    elif condition == "dust":
        a = 0.25 + 0.33 * sev
        f = _pil_blur(f * (1 - a) + np.array([196, 168, 120], np.float32) * a, 0.5 + sev)
    elif condition == "rain":
        f = f * (0.8 - 0.25 * sev)
        im = Image.new("L", (S, S), 0)
        dr = ImageDraw.Draw(im)
        for _ in range(int(40 + 140 * sev)):
            x, y, ln = r.uniform(0, S), r.uniform(0, S), r.uniform(8, 16)
            dr.line([(x, y), (x + ln * 0.25, y + ln)], fill=255, width=1)
        streak = np.asarray(im.filter(ImageFilter.GaussianBlur(0.7)), dtype=np.float32)
        wet = lowfreq(r, 5, S).clip(0, None) * 38 * sev
        f = _pil_blur(f + (streak * 0.36 + wet)[..., None], 0.6)
    elif condition == "night":
        beam = _gauss(S / 2, S * 0.92, 70, 62)
        f = f * (0.22 - 0.1 * sev) + f * 0.5 * beam[..., None]
        f = f + r.normal(0, 6 + 8 * sev, f.shape)
        f = f * np.array([0.95, 1.0, 1.08], np.float32)
    return f


def _motion_blur(f, k):
    pad = np.pad(f, ((0, 0), (k // 2 + 1, k // 2), (0, 0)), mode="edge")
    cs = np.cumsum(pad, axis=1)
    return (cs[:, k:] - cs[:, :-k]) / k


def frame_meta(seg, spot, frame) -> dict:
    """What the camera did on this pass: offset, condition, severity, blur."""
    sg = segment(seg)
    r = rng(3, seg, spot, frame)
    if frame == 0:
        dx = dy = 32
    elif r.random() < 0.35:
        dx, dy = (32 + r.integers(-2, 3, 2)).tolist()
    else:
        dx, dy = (32 + r.integers(-14, 15, 2)).tolist()
    p = np.array([0.58, 0.11, 0.11, 0.10, 0.10])
    if sg["district"] == "Al-Sinaiya":
        p = np.array([0.46, 0.09, 0.27, 0.08, 0.10])
    if sg["district"] == "Al-Corniche":
        p = np.array([0.50, 0.14, 0.08, 0.18, 0.10])
    cond = CONDITIONS[int(r.choice(5, p=p))]
    return {"dx": int(dx), "dy": int(dy), "condition": cond,
            "severity": round(float(r.uniform(0.3, 0.7)), 2) if cond != "day" else 0.0,
            "blur": int(r.choice([5, 9])) if r.random() < 0.12 else 0, "view": "vehicle"}


def render(seg, spot, frame, condition=None, severity=None, view=None, blur=None):
    """One frame: (uint8 image, instance map, objects in the frame, meta)."""
    canvas, inst, objects = spot_canvas(seg, spot)
    meta = frame_meta(seg, spot, frame)
    if condition is not None:
        meta["condition"], meta["severity"] = condition, 0.9 if severity is None else severity
    if view is not None:
        meta["view"] = view
    if blur is not None:
        meta["blur"] = blur
    r = rng(4, seg, spot, frame)
    scale = 1.0
    if meta["view"] == "pole":  # a moved camera: rotated, further away
        th, scale = math.radians(r.uniform(14, 30) * r.choice([-1, 1])), float(r.uniform(0.62, 0.78))
        pad = 96
        big = np.pad(canvas, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        bigi = np.pad(inst, pad, mode="constant")
        c, s_ = math.cos(th) / scale, math.sin(th) / scale
        cx = cy = pad + C / 2
        coef = (c, -s_, cx - c * S / 2 + s_ * S / 2, s_, c, cy - s_ * S / 2 - c * S / 2)
        f = np.asarray(Image.fromarray(big.clip(0, 255).astype(np.uint8)).transform(
            (S, S), Image.AFFINE, coef, Image.BILINEAR), dtype=np.float32)
        fi = np.asarray(Image.fromarray(bigi).transform((S, S), Image.AFFINE, coef, Image.NEAREST))
    else:
        dx, dy = meta["dx"], meta["dy"]
        f, fi = canvas[dy:dy + S, dx:dx + S].copy(), inst[dy:dy + S, dx:dx + S]
    f = f + r.normal(0, 2.5, f.shape)
    f = apply_condition(f, meta["condition"], meta["severity"], r)
    if meta["blur"]:
        f = _motion_blur(f, meta["blur"])
    out = []
    for o in objects:
        if o["cls"] == "shadow":
            continue
        m = fi == o["iid"]
        if m.sum() < 0.4 * o["area"] * scale ** 2 or m.sum() < 12:
            continue
        box = _box(m)
        if min(box[2] - box[0], box[3] - box[1]) < 5:
            continue
        if o["cls"] in DET_CLASSES and seg % 3 == 0:
            # this segment's annotator leaves a margin around every box; the others draw tight
            box = [max(0, box[0] - 3), max(0, box[1] - 3), min(S, box[2] + 3), min(S, box[3] + 3)]
        out.append({"iid": o["iid"], "cls": o["cls"], "box": box})
    return f.clip(0, 255).astype(np.uint8), np.ascontiguousarray(fi), out, meta


def inherited_label(seg, spot, obj):
    """The label the department's old annotations carry. Alligator cracking was never a class:
    it was filed under one of the two crack classes. Manhole covers were never labelled."""
    if obj["cls"] == "alligator":
        return ["crack_long", "crack_trans"][int(rng(6, seg, spot, obj["iid"]).integers(2))]
    return obj["cls"] if obj["cls"] in TILE_CLASSES else None


# ---------------------------------------------------------------- the road pack

def frame_number(seg, spot, frame):
    """A frame id that does not reveal the segment."""
    perm = _perm()
    return int(perm[(seg * N_SPOTS + spot) * N_FRAMES + frame])


@lru_cache(maxsize=1)
def _perm():
    return rng(9).permutation(N_SEGMENTS * N_SPOTS * N_FRAMES)


def build_roads(n_segments: int, log=print):
    (ROADS / "frames").mkdir(parents=True, exist_ok=True)
    (ROADS / "inst").mkdir(parents=True, exist_ok=True)
    frames, tiles, truth = [], [], {}
    for seg in range(n_segments):
        sg = segment(seg)
        for spot in range(N_SPOTS):
            for fr in range(N_FRAMES):
                img, inst, objs, meta = render(seg, spot, fr)
                fid = f"F{frame_number(seg, spot, fr):05d}"
                Image.fromarray(img).save(ROADS / "frames" / f"{fid}.jpg", quality=90)
                Image.fromarray(inst).save(ROADS / "inst" / f"{fid}.png", optimize=True)
                r = rng(8, seg, spot, fr)
                u = r.random()
                for o in objs:
                    o["label"] = inherited_label(seg, spot, o)
                truth[fid] = sg["id"]
                frames.append({
                    "id": fid, "segment": None if r.random() < 0.15 else sg["id"],
                    "district": sg["district"], "split": sg["split"],
                    "split_random": "train" if u < 0.7 else "val" if u < 0.85 else "test",
                    "condition": meta["condition"], "severity": meta["severity"],
                    "blur": meta["blur"], "view": meta["view"],
                    "address": [seg, spot, fr], "objects": objs})
                for o in objs:
                    if o["label"] is None:
                        continue
                    x0, y0, x1, y1 = o["box"]
                    cx = int(np.clip((x0 + x1) // 2 + r.integers(-6, 7), TILE // 2, S - TILE // 2))
                    cy = int(np.clip((y0 + y1) // 2 + r.integers(-6, 7), TILE // 2, S - TILE // 2))
                    tiles.append({"id": f"T{len(tiles):05d}", "frame": fid, "x0": cx - TILE // 2,
                                  "y0": cy - TILE // 2, "label": o["label"], "cls": o["cls"], "iid": o["iid"]})
                if r.random() < 0.6:
                    labelled = np.isin(inst, [o["iid"] for o in objs if o["label"] is not None])
                    for _ in range(12):
                        x, y = r.integers(0, S - TILE + 1, 2)
                        if labelled[y:y + TILE, x:x + TILE].mean() < 0.01:
                            tiles.append({"id": f"T{len(tiles):05d}", "frame": fid, "x0": int(x), "y0": int(y),
                                          "label": "clear", "cls": "clear", "iid": 0})
                            break
        if (seg + 1) % 20 == 0:
            log(f"  road segments drawn: {seg + 1}/{n_segments}")
    (ROADS / "frames.json").write_text(json.dumps(frames), encoding="utf-8")
    (ROADS / "tiles.json").write_text(json.dumps(tiles), encoding="utf-8")
    (ROADS / "truth_segments.json").write_text(json.dumps(truth), encoding="utf-8")
    coco = {"images": [], "annotations": [], "categories": [
        {"id": i + 1, "name": n} for i, n in enumerate(TILE_CLASSES[1:])]}
    for i, f in enumerate(frames):
        coco["images"].append({"id": i, "file_name": f"{f['id']}.jpg", "width": S, "height": S})
        for o in f["objects"]:
            if o["label"]:
                x0, y0, x1, y1 = o["box"]
                coco["annotations"].append({"id": len(coco["annotations"]), "image_id": i,
                                            "category_id": TILE_CLASSES.index(o["label"]),
                                            "bbox": [x0, y0, x1 - x0, y1 - y0], "iscrowd": 0})
    (ROADS / "annotations_coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return len(frames), len(tiles)


# ---------------------------------------------------------------- lab 1.2: the annotation batch

def annotation_batch():
    """30 crops for the new class 'alligator cracking', with two annotators' two passes."""
    out = CACHE / "annot"
    out.mkdir(parents=True, exist_ok=True)
    kinds = ["alligator"] * 9 + ["borderline"] * 8 + ["crack"] * 5 + ["patch"] * 3 + ["clear"] * 3 + ["pothole"] * 2
    order = rng(20).permutation(30)
    items = []
    global C
    for n, k in enumerate(order):
        kind = kinds[k]
        r = rng(21, int(k))
        size = 128
        g = r.uniform(85, 125) + 8 * lowfreq(r, 6, size) + r.normal(0, 6, (size, size))
        img = np.stack([g, g, g], -1).astype(np.float32)
        keep, C = C, size   # the mask helpers draw on a canvas of side C
        try:
            cells, m = 0, None
            if kind == "alligator":
                nx, ny = int(r.integers(3, 5)), int(r.integers(3, 5))
                m = _network_mask(r, 64, 64, r.uniform(32, 42), nx, ny)
                cells = (nx - 1) * (ny - 1)
            elif kind == "borderline":
                if k % 2:   # two cracks that cross: no closed cell
                    m = _crack_mask(r, 64, 64, 100, True, 3) | _crack_mask(r, 64, 64, 100, False, 3)
                else:       # a sparse network: one or two cells
                    nx = 2 + int(k % 4 == 0)
                    m = _network_mask(r, 64, 64, 30, nx, 2)
                    cells = nx - 1
            elif kind == "crack":
                m = _crack_mask(r, 64, 64, 105, bool(k % 2), 3)
            elif kind == "pothole":
                d, _ = _blob(r, 64, 64, 22, 17)
                img = np.where((d < 1)[..., None], img * 0.4, img)
            elif kind == "patch":
                img[30:95, 25:100] = img[30:95, 25:100] * 0.72
                img[30:95, 25:27] *= 0.6
                img[30:32, 25:100] *= 0.6
        finally:
            C = keep
        if m is not None:
            img = np.where(m[..., None], img * 0.42, img)
        Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(out / f"A{n + 1:02d}.png")
        items.append({"id": f"A{n + 1:02d}", "kind": kind, "cells": cells,
                      "truth": int(cells >= 3), "box": _box(m) if (m is not None and cells >= 1) or kind == "borderline" else None})
    r = rng(22)
    lenient = {i["id"]: int(i["kind"] == "alligator" or (i["kind"] == "borderline" and r.random() < 0.8)
                             or (i["kind"] == "crack" and r.random() < 0.2)) for i in items}
    strict = {i["id"]: int((i["kind"] == "alligator" and r.random() < 0.9)
                           or (i["kind"] == "borderline" and r.random() < 0.12)) for i in items}
    second = {}
    for who, slip in (("A", 0), ("B", 1)):
        lab = {i["id"]: i["truth"] for i in items}
        border = [i["id"] for i in items if i["kind"] == "borderline"]
        lab[border[slip * 3]] = 1 - lab[border[slip * 3]]
        second[who] = lab

    def boxes(labels, grow):
        res = {}
        for i in items:
            if labels[i["id"]] and i["box"]:
                x0, y0, x1, y1 = i["box"]
                res[i["id"]] = [max(0, x0 - grow), max(0, y0 - grow), min(128, x1 + grow), min(128, y1 + grow)]
        return res

    pack = {"items": [{"id": i["id"]} for i in items],
            "pass1": {"A": lenient, "B": strict}, "pass2": second,
            "boxes_pass2": {"A": boxes(second["A"], 9), "B": boxes(second["B"], 1)}}
    (out / "batch.json").write_text(json.dumps(pack, indent=1), encoding="utf-8")
    (out / "truth.json").write_text(json.dumps(items, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- lab 3.4: traffic clips

def traffic(n_clips: int):
    W, H, T, line_y = 160, 120, 180, 84
    kinds = ["day", "queue", "gantry", "night"]
    clips = []
    for c in range(n_clips):
        r = rng(30, c)
        kind = kinds[c % 4]
        occ = [44, 64] if kind == "gantry" else None
        vehicles, tid = [], 0
        for lane_x in (40, 80, 120):
            t = float(r.uniform(-20, 10))
            while t < T:
                h = float(r.uniform(46, 58) if r.random() < 0.15 else r.uniform(24, 34))
                vehicles.append({"tid": tid, "x": lane_x + float(r.uniform(-3, 3)), "w": float(r.uniform(16, 22)),
                                 "h": h, "t0": t, "v": float(r.uniform(2.2, 4.2))})
                tid += 1
                t += (h + 30) / 2.2 + float(r.exponential(14))
        gt, dets, count = [], [[] for _ in range(T)], 0
        for v in vehicles:
            y, boxes, crossed = -v["h"], {}, False
            for t in range(T):
                if t < v["t0"]:
                    continue
                speed = v["v"] * (0.22 if kind == "queue" and 60 <= t < 110 else 1.0)
                prev_c = y + v["h"] / 2
                y += speed
                cy = y + v["h"] / 2
                if prev_c < line_y <= cy:
                    crossed = True
                if y > H or y + v["h"] < 0:
                    continue
                box = [v["x"] - v["w"] / 2, max(0.0, y), v["x"] + v["w"] / 2, min(float(H), y + v["h"])]
                if box[3] - box[1] < 6:
                    continue
                boxes[t] = [round(b, 1) for b in box]
                hidden = occ is not None and occ[0] <= cy <= occ[1]
                miss = 0.92 if hidden else 0.12 if kind == "night" else 0.04
                if r.random() >= miss:
                    j = r.normal(0, 2.0 if kind == "night" else 1.2, 4)
                    dets[t].append([round(float(b + e), 1) for b, e in zip(box, j)] + [round(float(r.uniform(0.6, 0.99)), 2)])
            if boxes:
                gt.append({"tid": v["tid"], "boxes": boxes})
                count += int(crossed)
        for t in range(T):
            if r.random() < 0.02:  # a stray false alarm, one frame long
                x, y = float(r.uniform(10, W - 30)), float(r.uniform(0, H - 20))
                dets[t].append([round(x, 1), round(y, 1), round(x + 14, 1), round(y + 12, 1), round(float(r.uniform(0.5, 0.7)), 2)])
            if kind == "night" and r.random() < 0.06:
                # a headlight reflection: slides down the wet road for two or three frames, across the line
                x, y, v = float(r.uniform(15, W - 35)), float(r.uniform(70, 80)), float(r.uniform(5, 7))
                w, h = float(r.uniform(14, 20)), float(r.uniform(12, 18))
                for k in range(int(r.integers(2, 4))):
                    if t + k < T:
                        yy = y + v * k
                        dets[t + k].append([round(x, 1), round(yy, 1), round(x + w, 1), round(yy + h, 1),
                                            round(float(r.uniform(0.5, 0.75)), 2)])
        clips.append({"id": f"CLIP-{c + 1:02d}", "kind": kind, "width": W, "height": H, "frames": T,
                      "line_y": line_y, "occluder": occ, "gt": gt, "detections": dets, "gt_count": count})
    out = CACHE / "traffic"
    out.mkdir(parents=True, exist_ok=True)
    (out / "clips.json").write_text(json.dumps(clips), encoding="utf-8")


# ---------------------------------------------------------------- lab 4.4: street frames and camera stream

def street(n_frames: int):
    W, H = 256, 160
    out = CACHE / "street"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    records = []
    for n in range(n_frames):
        r = rng(40, n)
        im = Image.new("RGB", (W, H), (150, 160, 172))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, 70], fill=tuple(int(v) for v in r.integers(150, 200, 3)))
        for x in range(10, W, 40):
            d.rectangle([x, 12, x + 22, 40], fill=(70, 90, 110))
        d.rectangle([0, 100, W, H], fill=(96, 96, 98))
        regions = []
        for _ in range(int(r.integers(1, 4))):  # pedestrians
            s = float(r.uniform(0.6, 1.3))
            x, y = float(r.uniform(15, W - 30)), float(r.uniform(52, 82))
            fw, fh = 13 * s, 16 * s
            d.rectangle([x - 9 * s, y + fh, x + 9 * s, y + fh + 44 * s], fill=tuple(int(v) for v in r.integers(30, 200, 3)))
            d.ellipse([x - fw / 2, y, x + fw / 2, y + fh], fill=(214, 172, 136))
            for ex in (-0.22, 0.22):
                d.ellipse([x + ex * fw - s, y + 0.36 * fh - s, x + ex * fw + s, y + 0.36 * fh + s], fill=(25, 20, 20))
            d.line([x - 0.2 * fw, y + 0.72 * fh, x + 0.2 * fw, y + 0.72 * fh], fill=(120, 40, 40), width=max(1, int(s)))
            regions.append({"kind": "face", "box": [x - fw / 2, y, x + fw / 2, y + fh]})
        for _ in range(int(r.integers(1, 3))):  # vehicles with plates
            x, y = float(r.uniform(20, W - 90)), float(r.uniform(96, 118))
            d.rectangle([x, y, x + 74, y + 34], fill=tuple(int(v) for v in r.integers(40, 220, 3)))
            px, py = x + 24, y + 20
            d.rectangle([px, py, px + 27, py + 9], fill=(238, 238, 232))
            for k in range(6):
                if r.random() < 0.85:
                    d.rectangle([px + 2 + k * 4, py + 2, px + 4 + k * 4, py + 7], fill=(20, 20, 24))
            regions.append({"kind": "plate", "box": [px, py, px + 27, py + 9]})
        arr = np.asarray(im, dtype=np.float32) + r.normal(0, 3, (H, W, 3))
        fid = f"ST{n + 1:02d}"
        Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(out / "frames" / f"{fid}.png")
        dets = []
        for g in regions:
            x0, y0, x1, y1 = g["box"]
            small = (x1 - x0) < 11
            u = r.random()
            score = r.uniform(0.5, 0.99) if u < (0.6 if small else 0.84) else r.uniform(0.2, 0.5) if u < 0.97 else None
            g["box"] = [round(float(v), 1) for v in g["box"]]
            if score is not None:
                sx, sy = 0.1 * (x1 - x0), 0.1 * (y1 - y0)  # the detector's boxes run tight
                j = r.normal(0, 0.8, 4)
                dets.append({"kind": g["kind"], "score": round(float(score), 2),
                             "box": [round(float(v), 1) for v in (x0 + sx + j[0], y0 + sy + j[1], x1 - sx + j[2], y1 - sy + j[3])]})
        if r.random() < 0.3:
            x, y = float(r.uniform(0, W - 20)), float(r.uniform(0, H - 15))
            dets.append({"kind": "face", "score": round(float(r.uniform(0.2, 0.5)), 2),
                         "box": [round(x, 1), round(y, 1), round(x + 14, 1), round(y + 12, 1)]})
        records.append({"id": fid, "truth": regions, "detections": dets})
    (out / "street.json").write_text(json.dumps(records), encoding="utf-8")


def camera_stream():
    """30 days of daily statistics from 12 pole cameras. Planted: CAM-07 is knocked on day 19,
    CAM-03 gathers dust from day 9, and it rains on days 12, 13 and 21 over the whole city."""
    r = rng(50)
    rain = {12, 13, 21}
    rows = []
    for cam in range(1, 13):
        b0, s0, n0 = r.uniform(105, 130), r.uniform(310, 380), r.uniform(2.2, 3.4)
        for day in range(1, 31):
            wet = day in rain
            sharp = s0 * (0.9 if wet else 1.0) * (1 + r.normal(0, 0.025))
            if cam == 3 and day >= 9:
                sharp *= 0.965 ** (day - 8)
            off = r.normal(0, 0.8) + (13.0 if cam == 7 and day >= 19 else 0.0)
            count = n0 * (0.88 if wet else 1.0) * (1 + r.normal(0, 0.05)) * (0.55 if cam == 7 and day >= 19 else 1.0)
            rows.append({"camera": f"CAM-{cam:02d}", "day": day,
                         "brightness": round(float(b0 - (18 if wet else 0) + r.normal(0, 3)), 1),
                         "sharpness": round(float(sharp), 1), "offset_px": round(float(off), 2),
                         "detections_per_frame": round(float(count), 2)})
    out = CACHE / "stream"
    out.mkdir(parents=True, exist_ok=True)
    (out / "camera_stream.json").write_text(json.dumps(rows), encoding="utf-8")


def build(n_segments: int = N_SEGMENTS, log=print):
    log(f"Drawing the Al-Manar pack: {n_segments} road segments, {n_segments * N_SPOTS * N_FRAMES} frames")
    n_frames, n_tiles = build_roads(n_segments, log)
    annotation_batch()
    traffic(12)
    street(40)
    camera_stream()
    (CACHE / "manifest.json").write_text(json.dumps(
        {"seed": SEED, "segments": n_segments, "frames": n_frames, "tiles": n_tiles}), encoding="utf-8")
    log(f"Done: {n_frames} frames, {n_tiles} defect crops, 30 annotation crops, 12 traffic clips, 40 street frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--segments", type=int, default=N_SEGMENTS)
    build(ap.parse_args().segments)
