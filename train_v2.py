"""Retrain / fine-tune the vehicle detector with stronger settings.

Run on a GPU machine (Colab / Kaggle / local GPU) - on CPU one epoch took ~25 min.

Examples:
    # fresh start from a bigger pretrained model
    python train_v2.py --data data.yaml --model yolo11s.pt

    # fine-tune your existing model on the extended dataset
    python train_v2.py --data data.yaml --model vehicle_model.pt --epochs 60

    # small / far vehicles
    python train_v2.py --data data.yaml --model yolo11s.pt --imgsz 960 --batch 8

data.yaml must point to YOUR dataset (train/val and, ideally, test) and list every
class. If you add classes (bus, truck, ...), start from a fresh pretrained model
(yolo11s.pt), not from vehicle_model.pt, because the class count changes.
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20, help="early-stop if no improvement")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="'cpu', '0', ... (default: auto)")
    p.add_argument("--name", default="vehicle_v2")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        cos_lr=True,          # smoother learning-rate decay
        close_mosaic=10,      # turn mosaic off for the last epochs
        # augmentation for varied footage
        degrees=5.0,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,            # lighting / time-of-day variation
        plots=True,           # confusion matrix, PR curves
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    best_model = YOLO(str(best))

    print("\n=== Validation split ===")
    best_model.val(data=args.data, split="val")

    # Held-out test split (only if data.yaml defines one)
    with open(args.data) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("test"):
        print("\n=== Test split (held out, unbiased estimate) ===")
        best_model.val(data=args.data, split="test")
    else:
        print("\nNo 'test:' entry in data.yaml - add a held-out test split for an honest score.")


if __name__ == "__main__":
    main()
