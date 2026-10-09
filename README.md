# Vehicle detection, tracking and line-crossing counting (YOLO11)

Detects vehicles in a video with Ultralytics YOLO11, tracks them with ByteTrack, and counts them by type.
It can count unique vehicles, or only the vehicles that cross a line you place on the road.

Classes of the general model (`vehicle_general_v1.pt`): 0 Car, 1 Van, 2 Bus, 3 Truck, 4 Motorcycle, 5 Autorickshaw.

## What the outputs mean

Each run writes to `runs/detect/<name>/`:

| File | Meaning |
|---|---|
| `video.mp4` | Input video with boxes labelled by class. Add `--show-conf` / `--show-id` for score / tracker id. With `--line-y` it also shows the line and a live COUNTED panel. |
| `vehicle_counts.csv` | Unique vehicles per class (a vehicle that stays in view is counted once). Can be inflated if the tracker loses and re-finds the same vehicle. |
| `line_crossings.csv` | One row per vehicle that crossed the line: track_id, class, frame, direction. Only written with `--line-y`. **Use this for traffic counts.** |

## Setup (Windows PowerShell)

```powershell
cd "D:\priya\vehicle detection"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Model weights, videos and datasets are **not** in the repository (see `.gitignore`). Put your `.pt` file and videos in the project folder.

## Run

Count vehicles crossing a line at 58% of the frame height, merging Van/Bus/Truck into "Large vehicles":

```powershell
python detect_video.py --source video2.mp4 --model vehicle_general_v1.pt --imgsz 960 --conf 0.3 --stride 3 --line-y 0.58 --merge-large --name video2_final
```

Quick test on the first 30 seconds of a 30 fps video (300 processed frames at stride 3 = 900 source frames, so use `--max-frames 300`):

```powershell
python detect_video.py --source video2.mp4 --model vehicle_general_v1.pt --imgsz 960 --stride 3 --line-y 0.58 --max-frames 300 --name quick_check
```

### Options of `detect_video.py`

| Option | Meaning |
|---|---|
| `--source` | Video path (or `0` for webcam) |
| `--model` | Trained weights `.pt` |
| `--conf` / `--iou` | Confidence and NMS thresholds (raise `--conf` to ~0.45 to remove weak false boxes) |
| `--imgsz` | Inference size; 960 helps small/far vehicles |
| `--stride` | Process every Nth frame (faster; 3 is fine at 30 fps) |
| `--min-frames` | Ignore tracks seen fewer than N frames |
| `--classes` | Keep only these classes |
| `--line-y` | Counting line as a fraction of frame height (0.58 = 58% down) |
| `--line-dir` | `any`, `down` or `up` |
| `--line-ref` | Use box `bottom` (default) or `center` as the reference point |
| `--merge-large` | Merge Van, Bus, Truck into "Large vehicles" |
| `--merge LABEL=A,B,C` | Custom merge, repeatable |
| `--max-frames` | Stop after N processed frames |
| `--show-conf`, `--show-id` | Show score / tracker id on boxes |
| `--device` | `cpu`, `0` (first GPU)... |
| `--name` | Output folder name under `runs/detect/` |

## How the counter works

1. Each frame is run through YOLO and ByteTrack, which gives every vehicle a track id.
2. For each track, the bottom-centre of its box is compared with the line; when it moves to the other side the vehicle is counted once.
3. A track must be seen for `--min-frames` frames before it can count.
4. A vehicle's class is the majority vote over all frames of its track, which reduces label flicker.

## Building and improving the model

| Script | Purpose |
|---|---|
| `extract_frames.py` | Extract frames from videos (optionally pre-label with COCO weights) for labelling |
| `split_dataset.py` | Split labelled frames into train/val/test by video (no leakage) and write `data.yaml` |
| `inspect_datasets.py` | Read-only summary of the FGVD `.tar` and UA-DETRAC `.zip` downloads |
| `build_general_dataset.py` | Merge FGVD + UA-DETRAC + own frames into the 6-class `dataset_general` |
| `train_v2.py` | Training script (GPU recommended; Colab T4 works) |

```powershell
python inspect_datasets.py --fgvd fgvd-DatasetNinja.tar --detrac UA-DETRAC-DATASET-10K.v1-v2.yolov11.zip
python build_general_dataset.py --fgvd fgvd-DatasetNinja.tar --detrac UA-DETRAC-DATASET-10K.v1-v2.yolov11.zip --own dataset --out dataset_general
```

`vehicle_general_v1.pt` was trained on Colab (yolo11m, imgsz 960) on `dataset_general`
(3,665 train / 464 val / 467 test images). Validation mAP50 0.857, mAP50-95 0.692.
FGVD test mAP50 0.885; UA-DETRAC Car/Van/Truck 0.958/0.977/0.858 on unseen sequences; own frames 0.632 (only 17 images, same video as training, so treat as rough).

Datasets: FGVD (datasetninja.com/fgvd) and UA-DETRAC (Roboflow export); check their licences before sharing trained weights.

## Known limitations

- The model flips among Van/Truck/Bus for the same large vehicle, so use `--merge-large` for reliable totals.
- Small vans can flip between Car and Large vehicles; merging does not fix this.
- Large false boxes can appear on empty road in a new camera view (domain gap). Raise `--conf`, or add empty-road frames and retrain.
- Unique-vehicle counts are inflated by ID fragmentation; line-crossing counts are more reliable but can be wrong if IDs swap exactly at the line.
- Always verify by counting a short stretch by eye.

## Troubleshooting

- `running scripts is disabled`: run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
- Slow on CPU: use `--stride 3` and `--max-frames` for tests.
- Windows zip warning about backslashes when unzipping on Colab is harmless.
