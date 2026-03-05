# train_and_eval.py
# Train YOLO11n on custom dataset and export metrics/bbox samples

import os
import sys
import shutil
import argparse
from pathlib import Path

BASE_DIR = Path(r"E:\dataset\UAS_Computer_Vision")
YOLO_DIR = BASE_DIR / "DATASET_YOLO"
DATA_YAML = YOLO_DIR / "data.yaml"
TRAINING_RESULTS = BASE_DIR / "TRAINING_RESULTS"
MODEL_DIR = BASE_DIR / "MODEL"
SAMPLE_BBOX_DIR = TRAINING_RESULTS / "sample_bbox_quality"

DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
MODEL_NAME = "yolo11n.pt"


def train_yolo(epochs, imgsz, batch):
    from ultralytics import YOLO

    print(f"\nTraining {MODEL_NAME}, epochs={epochs}, imgsz={imgsz}, batch={batch}")
    model = YOLO(MODEL_NAME)
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(TRAINING_RESULTS),
        name="yolo11n_custom",
        exist_ok=True,
        verbose=True,
    )
    return model, results


def extract_metrics(results):
    """Extract mAP, precision, recall from training results."""
    metrics_file = TRAINING_RESULTS / "metrics.txt"

    try:
        m = results.results_dict
        map50 = m.get("metrics/mAP50(B)", "N/A")
        map50_95 = m.get("metrics/mAP50-95(B)", "N/A")
        precision = m.get("metrics/precision(B)", "N/A")
        recall = m.get("metrics/recall(B)", "N/A")
    except Exception:
        # fallback: read from CSV
        map50, map50_95, precision, recall = parse_csv()

    text = f"""TRAINING METRICS - YOLO11n
{'='*40}
mAP@0.5      : {map50}
mAP@0.5:0.95 : {map50_95}
Precision    : {precision}
Recall       : {recall}
{'='*40}
Model: {MODEL_NAME}, imgsz={DEFAULT_IMGSZ}, epochs={DEFAULT_EPOCHS}, batch={DEFAULT_BATCH}
"""
    metrics_file.write_text(text, encoding="utf-8")

    print(f"\nmAP@0.5: {map50}")
    print(f"mAP@0.5:0.95: {map50_95}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Saved: {metrics_file}")


def parse_csv():
    import csv

    csv_path = TRAINING_RESULTS / "yolo11n_custom" / "results.csv"
    if not csv_path.exists():
        return "N/A", "N/A", "N/A", "N/A"
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "N/A", "N/A", "N/A", "N/A"
    last = {k.strip(): v.strip() for k, v in rows[-1].items()}
    return (
        last.get("metrics/mAP50(B)", "N/A"),
        last.get("metrics/mAP50-95(B)", "N/A"),
        last.get("metrics/precision(B)", "N/A"),
        last.get("metrics/recall(B)", "N/A"),
    )


def copy_artifacts():
    """Copy training artifacts (graphs, csv, etc.) to results folder."""
    run_dir = TRAINING_RESULTS / "yolo11n_custom"
    if not run_dir.exists():
        return

    files = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxF1_curve.png",
        "results.csv",
        "args.yaml",
        "labels.jpg",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
    ]
    for fname in files:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(TRAINING_RESULTS / fname))


def copy_weights():
    weights_dir = TRAINING_RESULTS / "yolo11n_custom" / "weights"
    if not weights_dir.exists():
        return
    for w in ["best.pt", "last.pt"]:
        src = weights_dir / w
        if src.exists():
            shutil.copy2(str(src), str(MODEL_DIR / w))
            print(f"  {w} -> {MODEL_DIR / w}")


def export_bbox_samples(max_samples=10):
    """Export val images with ground truth bbox overlay to prove annotation quality."""
    import cv2

    SAMPLE_BBOX_DIR.mkdir(parents=True, exist_ok=True)
    val_img_dir = YOLO_DIR / "images" / "val"
    val_lbl_dir = YOLO_DIR / "labels" / "val"

    img_files = sorted(val_img_dir.glob("*.jpg"))[:max_samples]
    if not img_files:
        return

    colors = {0: (0, 255, 0), 1: (255, 128, 0)}
    names = {0: "ps_controller", 1: "correction_tape"}

    for img_file in img_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_file = val_lbl_dir / f"{img_file.stem}.txt"

        if lbl_file.exists():
            for line in lbl_file.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                xc, yc, bw, bh = [float(x) for x in parts[1:]]

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                color = colors.get(cls_id, (0, 0, 255))
                label = names.get(cls_id, f"cls{cls_id}")

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    img,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

        cv2.imwrite(str(SAMPLE_BBOX_DIR / f"gt_{img_file.name}"), img)

    print(f"  {len(img_files)} samples -> {SAMPLE_BBOX_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument(
        "--bbox_only", action="store_true", help="Only export bbox samples"
    )
    args = parser.parse_args()

    if not DATA_YAML.exists():
        print(f"data.yaml not found, please run prepare_dataset.py first")
        sys.exit(1)

    if args.bbox_only:
        export_bbox_samples()
        return

    model, results = train_yolo(args.epochs, args.imgsz, args.batch)

    print("\nExtracting metrics...")
    extract_metrics(results)

    print("Copying artifacts...")
    copy_artifacts()

    print("Copying weights...")
    copy_weights()

    print("Exporting bbox samples...")
    export_bbox_samples()

    print(f"\nDone! Results: {TRAINING_RESULTS}")


if __name__ == "__main__":
    main()
