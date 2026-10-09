"""Step 1 of retraining: extract frames from your videos and pre-label them.

Pre-labelling uses a COCO model (default yolo11m.pt) for the classes it already knows
and remaps them to YOUR class list. Auto-rickshaws are NOT pre-labelled (no pretrained
model has that class) - you add them by hand in the annotation tool.

    python extract_frames.py --video video.mp4 --every 10 --prelabel
    python extract_frames.py --video clip1.mp4 clip2.mp4 clip3.mp4 --every 15 --prelabel

Output (default folder: dataset_raw/):
    images/<video>_<frame>.jpg
    labels/<video>_<frame>.txt     YOLO format: class xc yc w h (normalized)
    classes.txt                    one class name per line (id = line number)

Final class list (ids):  0 Car, 1 Motorcycle, 2 Bus, 3 Truck, 4 Auto-rickshaw
Pre-labels are only a head start: open them in Roboflow / CVAT / Label Studio,
fix wrong boxes, add missed vehicles, and relabel auto-rickshaws as class 4
(the COCO model usually calls them car or bus).
"""
import argparse
from pathlib import Path

import cv2

# Your classes (id = index)
CLASS_NAMES = ["Car", "Motorcycle", "Bus", "Truck", "Auto-rickshaw"]

# COCO class name -> your class id
COCO_TO_MINE = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, nargs="+", help="one or more video files")
    p.add_argument("--out", default="dataset_raw")
    p.add_argument("--every", type=int, default=10, help="save every Nth frame (10-15 is a good start)")
    p.add_argument("--prelabel", action="store_true", help="auto-label frames with --model")
    p.add_argument("--model", default="yolo11m.pt", help="COCO model for pre-labels (bigger = better recall)")
    p.add_argument("--conf", type=float, default=0.3, help="min confidence for pre-labels")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n")

    model = None
    coco_ids = {}
    if args.prelabel:
        from ultralytics import YOLO
        model = YOLO(args.model)
        name_to_id = {n.lower(): i for i, n in model.names.items()}
        coco_ids = {name_to_id[n]: mine for n, mine in COCO_TO_MINE.items() if n in name_to_id}
        print("Pre-label mapping (COCO id -> your class):",
              {k: CLASS_NAMES[v] for k, v in coco_ids.items()})

    saved = 0
    for video in args.video:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"Could not open {video}, skipping")
            continue
        stem = Path(video).stem
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.every == 0:
                name = f"{stem}_{idx:06d}"
                cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame)
                lines = []
                if model is not None:
                    r = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                      classes=list(coco_ids), device=args.device, verbose=False)[0]
                    for c, (xc, yc, w, h) in zip(r.boxes.cls.int().tolist(), r.boxes.xywhn.tolist()):
                        lines.append(f"{coco_ids[c]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                (out / "labels" / f"{name}.txt").write_text("\n".join(lines))
                saved += 1
            idx += 1
        cap.release()
        print(f"{video}: {idx} frames read")

    print(f"\nSaved {saved} frames to {out.resolve()}")
    print("Next: annotate (fix boxes + add Auto-rickshaw = class 4), export in YOLO format,")
    print("then run split_dataset.py")


if __name__ == "__main__":
    main()
