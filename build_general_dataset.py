"""Build one YOLO dataset from FGVD (Supervisely .tar), UA-DETRAC (Roboflow YOLO .zip)
and your own labelled frames. Archives are read in place (nothing is fully extracted).

    python build_general_dataset.py ^
        --fgvd fgvd-DatasetNinja.tar ^
        --detrac UA-DETRAC-DATASET-10K.v1-v2.yolov11.zip ^
        --own dataset --out dataset_general

Final classes: 0 Car, 1 Van, 2 Bus, 3 Truck, 4 Motorcycle, 5 Autorickshaw
  FGVD:       car->Car, bus->Bus, mini-bus->Bus, truck->Truck,
              motorcycle->Motorcycle, scooter->Motorcycle, autorickshaw->Autorickshaw
  UA-DETRAC:  car->Car, van->Van, bus->Bus, truck->Truck
  Own frames: matched by class name using <own>/data.yaml

Splits (no leakage):
  * UA-DETRAC is split by video sequence: --detrac-holdout sequences are never used for
    training (half go to val, half to test).
  * FGVD keeps its own train/val/test split (sampled down).
  * Your own frames keep the split you made earlier; train frames are repeated --own-repeat times.
Every file name starts with fgvd_ / detrac_ / own_ so you can score each source separately.
"""
import argparse
import json
import math
import random
import re
import shutil
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import yaml

CLASSES = ["Car", "Van", "Bus", "Truck", "Motorcycle", "Autorickshaw"]
CID = {c.lower(): i for i, c in enumerate(CLASSES)}
FGVD_MAP = {"car": "car", "bus": "bus", "mini-bus": "bus", "truck": "truck",
            "motorcycle": "motorcycle", "scooter": "motorcycle", "autorickshaw": "autorickshaw"}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def save_image(data: bytes, dst: Path, max_side: int):
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return False
    h, w = im.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def write_label(out: Path, split: str, name: str, lines):
    p = out / "labels" / split
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def weighted_sample(items, weights, k, rng):
    """Sample k items without replacement, favouring higher weights (Efraimidis-Spirakis)."""
    keyed = [(rng.random() ** (1.0 / max(w, 1e-9)), it) for it, w in zip(items, weights)]
    keyed.sort(key=lambda t: t[0], reverse=True)
    return [it for _, it in keyed[:k]]


# ------------------------------------------------------------------ FGVD
def fgvd_boxes(ann):
    w, h = ann["size"]["width"], ann["size"]["height"]
    lines = []
    for o in ann.get("objects", []):
        t = FGVD_MAP.get(o.get("classTitle"))
        if t is None or o.get("geometryType") != "rectangle":
            continue
        (x1, y1), (x2, y2) = o["points"]["exterior"][:2]
        x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
        y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{CID[t]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def do_fgvd(path, out, n_train, n_val, n_test, max_side, rng, stats):
    print(f"\nFGVD: reading annotations from {path} ...")
    ann_re = re.compile(r"^(train|val|test)/ann/(.+)\.json$")
    per_split = defaultdict(dict)            # split -> image file name -> label lines
    with tarfile.open(path) as tf:
        for m in tf:
            mt = ann_re.match(m.name) if m.isfile() else None
            if not mt:
                continue
            lines = fgvd_boxes(json.loads(tf.extractfile(m).read()))
            if lines:
                per_split[mt.group(1)][mt.group(2)] = lines
    want = {}
    quota = {"train": n_train, "val": n_val, "test": n_test}
    freq = Counter(int(l.split()[0]) for d in per_split["train"].values() for l in d)
    for sp, items in per_split.items():
        names = sorted(items)
        if sp == "train":   # favour images that contain rarer classes
            wts = [max(1.0 / math.sqrt(freq[int(l.split()[0])]) for l in items[n]) for n in names]
            pick = weighted_sample(names, wts, quota[sp], rng)
        else:
            pick = rng.sample(names, min(quota[sp], len(names)))
        for n in pick:
            want[f"{sp}/img/{n}"] = (sp, n, items[n])
    print("FGVD selected:", {sp: sum(1 for v in want.values() if v[0] == sp) for sp in quota})
    done = 0
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile() or m.name not in want:
                continue
            sp, n, lines = want[m.name]
            stem = "fgvd_" + PurePosixPath(n).stem
            if save_image(tf.extractfile(m).read(), out / "images" / sp / f"{stem}.jpg", max_side):
                write_label(out, sp, stem, lines)
                for l in lines:
                    stats[(sp, "fgvd")][int(l.split()[0])] += 1
                done += 1
    print(f"FGVD written: {done} images")


# ------------------------------------------------------------------ UA-DETRAC
def do_detrac(path, out, n_train, holdout, max_per_holdout, max_side, rng, stats):
    print(f"\nUA-DETRAC: indexing {path} ...")
    with zipfile.ZipFile(path) as zf:
        names = None
        for n in zf.namelist():
            if n.lower().endswith("data.yaml"):
                names = yaml.safe_load(zf.read(n))["names"]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        remap = {i: CID[n.lower()] for i, n in enumerate(names)}
        print("  class remap:", {names[i]: CLASSES[j] for i, j in remap.items()})
        seqs = defaultdict(list)
        for n in zf.namelist():
            if PurePosixPath(n).suffix.lower() in IMG_EXT:
                m = re.search(r"(MVI_\d+)_img(\d+)", n)
                if m:
                    seqs[m.group(1)].append((int(m.group(2)), n))
        ids = sorted(seqs)
        rng.shuffle(ids)
        hold = ids[:holdout]
        val_seqs, test_seqs = hold[: holdout // 2], hold[holdout // 2:]
        train_seqs = ids[holdout:]
        print(f"  sequences: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test")

        def evenly(frames, k):
            frames = sorted(frames)
            if len(frames) <= k:
                return frames
            step = len(frames) / k
            return [frames[int(i * step)] for i in range(k)]

        plan = []
        per_seq = max(1, n_train // max(len(train_seqs), 1))
        for s in train_seqs:
            plan += [("train", n) for _, n in evenly(seqs[s], per_seq)]
        for s in val_seqs:
            plan += [("val", n) for _, n in evenly(seqs[s], max_per_holdout)]
        for s in test_seqs:
            plan += [("test", n) for _, n in evenly(seqs[s], max_per_holdout)]
        done = 0
        for sp, n in plan:
            lab = re.sub(r"/images/", "/labels/", n)
            lab = str(PurePosixPath(lab).with_suffix(".txt"))
            try:
                raw = zf.read(lab).decode()
            except KeyError:
                continue
            lines = []
            for l in raw.splitlines():
                p = l.split()
                if len(p) == 5 and int(p[0]) in remap:
                    lines.append(" ".join([str(remap[int(p[0])])] + p[1:]))
            m = re.search(r"(MVI_\d+_img\d+)", n)
            stem = "detrac_" + m.group(1)
            if save_image(zf.read(n), out / "images" / sp / f"{stem}.jpg", max_side):
                write_label(out, sp, stem, lines)
                for l in lines:
                    stats[(sp, "detrac")][int(l.split()[0])] += 1
                done += 1
    print(f"UA-DETRAC written: {done} images")


# ------------------------------------------------------------------ own frames
def do_own(src, out, repeat, stats):
    src = Path(src)
    y = src / "data.yaml"
    if not y.exists():
        print(f"\nOwn frames: {y} not found, skipping")
        return
    names = yaml.safe_load(y.read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    remap = {i: CID[n.lower().replace("-", "")]
             for i, n in enumerate(names)}
    print("\nOwn frames class remap:", {names[i]: CLASSES[j] for i, j in remap.items()})
    total = 0
    for sp in ("train", "val", "test"):
        imgs = sorted((src / "images" / sp).glob("*.*")) if (src / "images" / sp).exists() else []
        for im in imgs:
            lab = src / "labels" / sp / f"{im.stem}.txt"
            lines = []
            if lab.exists():
                for l in lab.read_text().splitlines():
                    p = l.split()
                    if len(p) == 5:
                        lines.append(" ".join([str(remap[int(p[0])])] + p[1:]))
            reps = repeat if sp == "train" else 1
            for r in range(reps):
                stem = f"own_{im.stem}" + (f"_r{r}" if reps > 1 else "")
                (out / "images" / sp).mkdir(parents=True, exist_ok=True)
                shutil.copy2(im, out / "images" / sp / f"{stem}{im.suffix}")
                write_label(out, sp, stem, lines)
                total += 1
            for l in lines:
                stats[(sp, "own")][int(l.split()[0])] += 1 * reps
    print(f"Own frames written: {total} files")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fgvd")
    p.add_argument("--detrac")
    p.add_argument("--own", default="dataset", help="your earlier split dataset (has data.yaml)")
    p.add_argument("--out", default="dataset_general")
    p.add_argument("--fgvd-train", type=int, default=2000)
    p.add_argument("--fgvd-val", type=int, default=250)
    p.add_argument("--fgvd-test", type=int, default=250)
    p.add_argument("--detrac-train", type=int, default=1500)
    p.add_argument("--detrac-holdout", type=int, default=10, help="sequences kept out of training")
    p.add_argument("--detrac-per-holdout", type=int, default=40)
    p.add_argument("--own-repeat", type=int, default=3)
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists and is not empty - delete it or pick another --out")
    rng = random.Random(a.seed)
    stats = defaultdict(Counter)
    if a.fgvd:
        do_fgvd(a.fgvd, out, a.fgvd_train, a.fgvd_val, a.fgvd_test, a.max_side, rng, stats)
    if a.detrac:
        do_detrac(a.detrac, out, a.detrac_train, a.detrac_holdout, a.detrac_per_holdout,
                  a.max_side, rng, stats)
    if a.own:
        do_own(a.own, out, a.own_repeat, stats)

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n")

    print("\nObjects per class (source, split):")
    print(f"{'':18}" + "".join(f"{c:>13}" for c in CLASSES))
    for (sp, src), c in sorted(stats.items()):
        print(f"{src + ' ' + sp:18}" + "".join(f"{c[i]:>13}" for i in range(len(CLASSES))))
    print("\nImages:", {sp: len(list((out / 'images' / sp).glob('*.*')))
                        for sp in ("train", "val", "test") if (out / 'images' / sp).exists()})
    print(f"Wrote {out / 'data.yaml'}  (edit 'path:' to the Colab location before training there)")


if __name__ == "__main__":
    main()
