"""Step 3 of retraining: split labelled data into train / val / test and write data.yaml.

Splits by VIDEO when you have 3+ videos (best: no near-duplicate frames across splits).
With fewer videos it splits each video into consecutive time blocks
(70% train / 15% val / 15% test) with a small gap between blocks, so neighbouring
frames don't end up in different splits.

    python split_dataset.py --src dataset_raw --dst dataset

Expects:  <src>/images/*.jpg  and  <src>/labels/*.txt  (YOLO format) + <src>/classes.txt
          Image names must look like <videoname>_<frame>.jpg (extract_frames.py does this).
Creates:  <dst>/images/{train,val,test}, <dst>/labels/{train,val,test}, <dst>/data.yaml
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

# Matches "<video>_<6-digit frame>" and also Roboflow-renamed files such as
# "video_000120_jpg.rf.3f9a...".
NAME_RE = re.compile(r"^(?P<vid>.+?)_(?P<frame>\d{6})(?:_.*)?$")


def parse_name(stem: str):
    m = NAME_RE.match(stem)
    if m:
        return m.group("vid"), int(m.group("frame"))
    return stem.rsplit("_", 1)[0], 0


def video_of(stem: str) -> str:
    return parse_name(stem)[0]


def load_names_from_yaml(path: Path):
    """Class names in the exact id order of an exported (e.g. Roboflow) data.yaml."""
    import yaml
    names = yaml.safe_load(path.read_text())["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="dataset_raw")
    p.add_argument("--dst", default="dataset")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--gap", type=int, default=2, help="frames skipped between time blocks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--names-from", default=None,
                   help="data.yaml exported by Roboflow/CVAT: use ITS class order for the "
                        "label ids (recommended after labelling in Roboflow)")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if args.names_from:
        classes = load_names_from_yaml(Path(args.names_from))
    else:
        classes = [l.strip() for l in (src / "classes.txt").read_text().splitlines() if l.strip()]
    print("Class ids:", dict(enumerate(classes)))
    images = sorted(p for p in (src / "images").iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"No images found in {src / 'images'}")

    by_video = defaultdict(list)
    for img in images:
        by_video[video_of(img.stem)].append(img)
    for v in by_video.values():
        v.sort(key=lambda im: parse_name(im.stem)[1])   # time order by frame number

    assign = {"train": [], "val": [], "test": []}
    videos = sorted(by_video)
    if len(videos) >= 3:
        random.Random(args.seed).shuffle(videos)
        n = len(videos)
        n_val = max(1, round(n * args.val))
        n_test = max(1, round(n * (1 - args.train - args.val)))
        for i, v in enumerate(videos):
            split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
            assign[split] += by_video[v]
        print(f"Split by video: {n} videos")
    else:
        for v in videos:
            frames = by_video[v]
            n = len(frames)
            a = int(n * args.train)
            b = int(n * (args.train + args.val))
            assign["train"] += frames[:max(a - args.gap, 0)]
            assign["val"] += frames[a:max(b - args.gap, a)]
            assign["test"] += frames[b:]
        print(f"Only {len(videos)} video(s): split into consecutive time blocks. "
              "Add more videos for a more trustworthy test score.")

    for split, items in assign.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in items:
            shutil.copy2(img, dst / "images" / split / img.name)
            lab = src / "labels" / f"{img.stem}.txt"
            if lab.exists():
                shutil.copy2(lab, dst / "labels" / split / lab.name)
        print(f"{split}: {len(items)} images")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    (dst / "data.yaml").write_text(
        f"path: {dst.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n"
    )
    print(f"\nWrote {dst / 'data.yaml'}")
    print("If you train on Colab, edit 'path:' in data.yaml to the dataset's location there.")


if __name__ == "__main__":
    main()
