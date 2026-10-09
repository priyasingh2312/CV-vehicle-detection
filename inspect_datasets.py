"""Read-only summary of the FGVD (Supervisely .tar) and UA-DETRAC (YOLO .zip) downloads.

Nothing is extracted or modified: archives are read in place.

    python inspect_datasets.py --fgvd fgvd-DatasetNinja.tar --detrac UA-DETRAC-DATASET-10K.v1-v2.yolov11.zip
"""
import argparse
import json
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

IMG_EXT = {".jpg", ".jpeg", ".png"}


def split_of(name):
    for part in PurePosixPath(name).parts:
        if part.lower() in {"train", "val", "valid", "test", "ds0", "ds1", "ds2"}:
            return part.lower()
    return "?"


def inspect_fgvd(path):
    print(f"\n=== FGVD: {path}")
    imgs, anns = Counter(), Counter()
    cls_objs = Counter()
    cls_by_split = defaultdict(Counter)
    tag_names = Counter()
    sizes = Counter()
    meta, sample, first_names = None, None, []
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            n = m.name
            if len(first_names) < 12:
                first_names.append(n)
            ext = PurePosixPath(n).suffix.lower()
            sp = split_of(n)
            if ext in IMG_EXT:
                imgs[sp] += 1
            elif ext == ".json":
                data = json.loads(tf.extractfile(m).read())
                if n.endswith("meta.json"):
                    meta = data
                    continue
                anns[sp] += 1
                if sample is None:
                    sample = data
                if "size" in data:
                    sizes[(data["size"].get("width"), data["size"].get("height"))] += 1
                for o in data.get("objects", []):
                    c = o.get("classTitle")
                    cls_objs[c] += 1
                    cls_by_split[sp][c] += 1
                    for t in o.get("tags", []):
                        tag_names[t.get("name")] += 1
    print("first entries:", *first_names, sep="\n  ")
    print("images per split:", dict(imgs))
    print("annotation files per split:", dict(anns))
    if meta:
        print("meta classes:", [c.get("title") for c in meta.get("classes", [])])
        print("meta tags:", [t.get("name") for t in meta.get("tags", [])])
    print("objects per class:", dict(cls_objs.most_common()))
    for sp, c in cls_by_split.items():
        print(f"  {sp}:", dict(c.most_common()))
    print("tag names on objects:", dict(tag_names.most_common(10)))
    print("image sizes (top 5):", sizes.most_common(5))
    if sample:
        print("sample annotation (trimmed):")
        print(json.dumps(sample, indent=1)[:900])


def inspect_detrac(path):
    print(f"\n=== UA-DETRAC: {path}")
    import yaml
    imgs, lbls = Counter(), Counter()
    cls_objs = Counter()
    cls_by_split = defaultdict(Counter)
    stems = defaultdict(set)
    first_names, yaml_txt = [], None
    with zipfile.ZipFile(path) as zf:
        for n in zf.namelist():
            if n.endswith("/"):
                continue
            if len(first_names) < 12:
                first_names.append(n)
            ext = PurePosixPath(n).suffix.lower()
            sp = split_of(n)
            if n.lower().endswith("data.yaml"):
                yaml_txt = zf.read(n).decode("utf8", "replace")
            elif ext in IMG_EXT:
                imgs[sp] += 1
                stems[sp].add(PurePosixPath(n).stem)
            elif ext == ".txt" and "label" in n.lower():
                lbls[sp] += 1
                for line in zf.read(n).decode().splitlines():
                    p = line.split()
                    if p:
                        cls_objs[p[0]] += 1
                        cls_by_split[sp][p[0]] += 1
        # image size of a few samples
        import cv2
        import numpy as np
        shown = 0
        for n in zf.namelist():
            if PurePosixPath(n).suffix.lower() in IMG_EXT and shown < 3:
                im = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR)
                print("sample image", n, "->", None if im is None else im.shape[:2][::-1])
                shown += 1
    print("first entries:", *first_names, sep="\n  ")
    print("images per split:", dict(imgs))
    print("label files per split:", dict(lbls))
    names = None
    if yaml_txt:
        d = yaml.safe_load(yaml_txt)
        names = d.get("names")
        print("data.yaml names:", names)
    def nm(k):
        if isinstance(names, list) and k.isdigit() and int(k) < len(names):
            return f"{k}={names[int(k)]}"
        if isinstance(names, dict) and k.isdigit():
            return f"{k}={names.get(int(k))}"
        return k
    print("objects per class:", {nm(k): v for k, v in sorted(cls_objs.items())})
    for sp, c in cls_by_split.items():
        print(f"  {sp}:", {nm(k): v for k, v in sorted(c.items())})
    # near-duplicate hint: Roboflow names look like MVI_20011_img00001_jpg.rf.<hash>
    seqs = Counter()
    for sp, ss in stems.items():
        for s in ss:
            m = re.match(r"(MVI_\d+)", s)
            if m:
                seqs[m.group(1)] += 1
    if seqs:
        print(f"distinct source sequences (MVI_*): {len(seqs)}; "
              f"largest: {seqs.most_common(3)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fgvd")
    p.add_argument("--detrac")
    a = p.parse_args()
    if a.fgvd:
        inspect_fgvd(a.fgvd)
    if a.detrac:
        inspect_detrac(a.detrac)


if __name__ == "__main__":
    main()
