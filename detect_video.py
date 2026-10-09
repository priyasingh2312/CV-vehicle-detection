"""Detect, track and count vehicles in a video with a trained YOLO model.

Usage (from this folder):
    python detect_video.py                         # uses video.mp4 + vehicle_model.pt
    python detect_video.py --source other.mp4 --conf 0.35
    python detect_video.py --source 0              # webcam
    python detect_video.py --show-conf --show-id   # add confidence / track id to the labels
    python detect_video.py --source video2.mp4 --line-y 0.58   # count vehicles crossing a line
    python detect_video.py --source video2.mp4 --line-y 0.58 --merge-large   # Bus+Truck+Van = Large vehicles

Output: annotated video (video.mp4) + vehicle_counts.csv under runs/detect/<name>/ and a
per-class count of UNIQUE vehicles (each tracked vehicle is counted once, not once per frame).
With --line-y, a horizontal line is drawn and every vehicle that crosses it is counted once per
class (live counter on the video, line_crossings.csv, and a summary at the end).
By default the boxes show only the class name (no track id, no confidence score).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# One BGR colour per class id (cycled if the model has more classes)
PALETTE = [
    (0, 0, 255),      # red
    (255, 200, 0),    # cyan-ish
    (255, 0, 0),      # blue
    (0, 165, 255),    # orange
    (0, 200, 0),      # green
    (200, 0, 200),    # magenta
    (0, 220, 220),    # yellow
    (150, 150, 150),  # grey
]


def draw_detections(img, boxes, names, show_conf=False, show_id=False):
    """Draw boxes with class-name labels on img (in place) and return it.

    boxes: iterable of (x1, y1, x2, y2, class_id, conf, track_id_or_None)
    """
    h, w = img.shape[:2]
    thickness = max(2, round(h / 450))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, h / 1600)
    text_thick = max(1, round(h / 700))

    for x1, y1, x2, y2, cid, conf, tid in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = PALETTE[int(cid) % len(PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = str(names[int(cid)])
        if show_id and tid is not None:
            label = f"#{int(tid)} {label}"
        if show_conf:
            label += f" {conf:.2f}"

        (tw, th), base = cv2.getTextSize(label, font, font_scale, text_thick)
        ty = y1 - 4 if y1 - th - base - 4 >= 0 else y1 + th + base + 4
        top = ty - th - base
        cv2.rectangle(img, (x1, top), (x1 + tw + 4, ty + base // 2), color, -1)
        cv2.putText(img, label, (x1 + 2, ty - 1), font, font_scale, (255, 255, 255),
                    text_thick, cv2.LINE_AA)
    return img


class LineCounter:
    """Counts each tracked vehicle once, when its reference point crosses a horizontal line.

    line_frac: line height as a fraction of the frame height (0 = top, 1 = bottom)
    direction: 'any', 'down' (towards the bottom of the frame) or 'up'
    ref:       'bottom' (bottom-centre of the box, ~ where the vehicle touches the road) or 'center'
    min_frames: a track must have been seen in at least this many frames to be counted
    """

    def __init__(self, line_frac, direction="any", ref="bottom", min_frames=5):
        self.frac, self.direction, self.ref, self.min_frames = line_frac, direction, ref, min_frames
        self.prev = {}      # track id -> side of the line at its previous sighting (+1 / -1)
        self.crossed = {}   # track id -> (frame number, 'down' | 'up')

    def update(self, tid, box, frame_h, frame_no, frames_seen):
        x1, y1, x2, y2 = box
        y = y2 if self.ref == "bottom" else (y1 + y2) / 2
        side = 1 if y >= self.frac * frame_h else -1
        before = self.prev.get(tid)
        self.prev[tid] = side
        if before is None or before == side or tid in self.crossed:
            return None
        d = "down" if side > 0 else "up"
        if self.direction != "any" and d != self.direction:
            return None
        if frames_seen < self.min_frames:
            return None
        self.crossed[tid] = (frame_no, d)
        return d


def majority(votes):
    return max(votes, key=votes.get)


def draw_line_overlay(img, line_frac, counts, total):
    """Draw the counting line and a small live-count panel (in place)."""
    h, w = img.shape[:2]
    y = int(h * line_frac)
    cv2.line(img, (0, y), (w, y), (0, 255, 255), max(2, round(h / 400)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = max(0.6, h / 1100), max(1, round(h / 600))
    rows = [f"{n}: {c}" for n, c in sorted(counts.items())] + [f"TOTAL: {total}"]
    line_h = int(34 * fs) + 10
    pw = int(max(cv2.getTextSize(r, font, fs, th)[0][0] for r in rows + ["COUNTED"]) + 24)
    ph = line_h * (len(rows) + 1) + 10
    roi = img[8:8 + ph, 8:8 + pw]
    img[8:8 + ph, 8:8 + pw] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    cv2.putText(img, "COUNTED", (18, 8 + line_h), font, fs, (0, 255, 255), th, cv2.LINE_AA)
    for i, r in enumerate(rows):
        cv2.putText(img, r, (18, 8 + line_h * (i + 2)), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    return img


def make_out_dir(name):
    base = Path("runs") / "detect"
    out = base / name
    i = 2
    while out.exists():
        out = base / f"{name}{i}"
        i += 1
    out.mkdir(parents=True)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Vehicle detection + tracking + counting")
    p.add_argument("--source", default="video.mp4", help="video path, or 0 for webcam")
    p.add_argument("--model", default="vehicle_model.pt", help="trained weights (.pt)")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference size (try 960 for small/far vehicles)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    p.add_argument("--stride", type=int, default=1,
                   help="process every Nth frame (>1 is faster on CPU but can hurt tracking)")
    p.add_argument("--min-frames", type=int, default=5,
                   help="a track must appear in at least this many processed frames to be counted "
                        "(filters out flickering false detections)")
    p.add_argument("--classes", nargs="+", default=None,
                   help="only detect/count these class NAMES, e.g. --classes car motorcycle bus truck "
                        "(useful with COCO models like yolo11n.pt to ignore 'person' etc.)")
    p.add_argument("--line-y", type=float, default=None,
                   help="count vehicles crossing a horizontal line at this fraction of the frame "
                        "height (0=top, 1=bottom), e.g. 0.58")
    p.add_argument("--line-dir", choices=["any", "down", "up"], default="any",
                   help="only count crossings towards the bottom ('down') or top ('up') of the frame")
    p.add_argument("--line-ref", choices=["bottom", "center"], default="bottom",
                   help="point of the box that must cross the line (bottom = where it touches the road)")
    p.add_argument("--merge-large", action="store_true",
                   help="treat Bus, Truck and Van as one class called 'Large vehicles' "
                        "(boxes, counts and the line counter all use the merged class)")
    p.add_argument("--merge", action="append", default=[], metavar="LABEL=A,B,C",
                   help="merge classes into one, e.g. --merge \"Large vehicles=Bus,Truck,Van\" "
                        "(can be repeated)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop after this many processed frames (quick test on the start of a long video)")
    p.add_argument("--show-conf", action="store_true", help="show confidence score in each label")
    p.add_argument("--show-id", action="store_true", help="show the tracker id in each label")
    p.add_argument("--device", default=None, help="'cpu', '0' for first GPU, etc. (default: auto)")
    p.add_argument("--name", default="vehicle_track", help="output run name")
    return p.parse_args()


def main():
    from ultralytics import YOLO  # imported here so draw_detections can be tested without it

    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    if isinstance(source, str) and not Path(source).exists():
        raise SystemExit(f"Video not found: {source}")
    # Official names like yolo11m.pt are downloaded automatically by Ultralytics;
    # a custom path that doesn't exist will raise its own error below.
    model = YOLO(args.model)
    print("Model classes:", model.names)

    class_ids = None
    if args.classes:
        name_to_id = {n.lower(): i for i, n in model.names.items()}
        missing = [c for c in args.classes if c.lower() not in name_to_id]
        if missing:
            raise SystemExit(f"Unknown class name(s) {missing}. Available: {list(model.names.values())}")
        class_ids = [name_to_id[c.lower()] for c in args.classes]

    # Frame rate of the output video (divided by stride, so playback speed stays natural)
    fps = 30.0
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    fps = fps / max(args.stride, 1)

    out_dir = make_out_dir(args.name)
    out_video = out_dir / "video.mp4"
    writer = None

    results = model.track(
        classes=class_ids,
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        tracker=args.tracker,
        persist=True,
        vid_stride=args.stride,
        device=args.device,
        save=False,           # we draw and save the video ourselves (clean labels)
        stream=True,          # frame-by-frame, keeps memory low
        verbose=False,
    )

    # Optional class merging: several model classes are reported as one
    merge_map = {}
    specs = list(args.merge)
    if args.merge_large:
        specs.append("Large vehicles=Bus,Truck,Van")
    lower_names = {n.lower() for n in model.names.values()}
    for spec in specs:
        label, _, members = spec.partition("=")
        members = [m.strip() for m in members.split(",") if m.strip()]
        if not label.strip() or not members:
            raise SystemExit(f"Bad --merge value {spec!r}; use LABEL=Class1,Class2")
        for m in members:
            if m.lower() in lower_names:
                merge_map[m.lower()] = label.strip()
            elif not args.merge_large:
                raise SystemExit(f"Unknown class {m!r} in --merge. Available: {list(model.names.values())}")

    def label_of(name):
        return merge_map.get(name.lower(), name)

    group_names = sorted({label_of(n) for n in model.names.values()})
    group_id = {g: i for i, g in enumerate(group_names)}
    draw_names = dict(enumerate(group_names))
    if merge_map:
        print("Merged classes:", {k: v for k, v in merge_map.items()})

    if args.line_y is not None and not 0 < args.line_y < 1:
        raise SystemExit("--line-y must be between 0 and 1 (fraction of the frame height)")
    counter = (LineCounter(args.line_y, args.line_dir, args.line_ref, args.min_frames)
               if args.line_y is not None else None)

    track_class = {}                 # track id -> class name (majority vote below)
    class_votes = defaultdict(lambda: defaultdict(int))
    frames_seen = defaultdict(int)   # track id -> number of frames it appeared in
    n_frames = 0

    try:
        for r in results:
            n_frames += 1
            frame = r.orig_img.copy()

            dets = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.tolist()
                cls = r.boxes.cls.int().tolist()
                conf = r.boxes.conf.tolist()
                ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(cls)
                for (x1, y1, x2, y2), c, cf, tid in zip(xyxy, cls, conf, ids):
                    cname = label_of(r.names[c])
                    dets.append((x1, y1, x2, y2, group_id[cname], cf, tid))
                    if tid is not None:
                        frames_seen[tid] += 1
                        class_votes[tid][cname] += 1
                        if counter is not None:
                            counter.update(tid, (x1, y1, x2, y2), frame.shape[0], n_frames,
                                           frames_seen[tid])

            draw_detections(frame, dets, draw_names, show_conf=args.show_conf, show_id=args.show_id)
            if counter is not None:
                live = defaultdict(int)
                for t in counter.crossed:
                    live[majority(class_votes[t])] += 1
                draw_line_overlay(frame, args.line_y, live, len(counter.crossed))

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)

            if n_frames % 50 == 0:
                print(f"  processed {n_frames} frames...")
            if args.max_frames and n_frames >= args.max_frames:
                break
    finally:
        if writer is not None:
            writer.release()

    # A track can flip class between frames; use its most frequent class.
    for tid, votes in class_votes.items():
        track_class[tid] = max(votes, key=votes.get)

    counted = {tid: c for tid, c in track_class.items() if frames_seen[tid] >= args.min_frames}
    counts = defaultdict(int)
    for c in counted.values():
        counts[c] += 1

    print(f"\nProcessed {n_frames} frames")
    print(f"Tracks ignored as too short (<{args.min_frames} frames): {len(track_class) - len(counted)}")
    print("Unique vehicles:")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")
    print(f"  TOTAL: {sum(counts.values())}")

    out_csv = out_dir / "vehicle_counts.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "class", "frames_seen"])
        for tid, c in sorted(counted.items()):
            w.writerow([tid, c, frames_seen[tid]])
    if counter is not None:
        by_class = defaultdict(lambda: defaultdict(int))
        rows = []
        for tid, (frame_no, d) in sorted(counter.crossed.items(), key=lambda kv: kv[1][0]):
            cname = track_class[tid]
            by_class[cname][d] += 1
            rows.append((tid, cname, frame_no, d))
        print(f"\nVehicles that crossed the line at {args.line_y:.0%} of the frame height "
              f"(direction: {args.line_dir}, each vehicle counted once):")
        for cname in sorted(by_class):
            dirs = by_class[cname]
            detail = ", ".join(f"{k} {v}" for k, v in sorted(dirs.items()))
            print(f"  {cname}: {sum(dirs.values())}   ({detail})")
        print(f"  TOTAL: {len(rows)}")
        with open(out_dir / "line_crossings.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "class", "processed_frame", "direction"])
            w.writerows(rows)
        print("Line-crossing list: line_crossings.csv")

    print(f"\nAnnotated video (video.mp4) + vehicle_counts.csv saved in: {out_dir}")


if __name__ == "__main__":
    main()
