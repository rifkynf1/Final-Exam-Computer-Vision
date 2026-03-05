# prepare_dataset.py
# YOLO dataset preparation script
# Stratified 80/20 split, sync labels, validate format

import os
import sys
import glob
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(r"E:\dataset\UAS_Computer_Vision")
DATASET_RAW_DIR = Path(r"E:\dataset\dataset_raw")

CLASS_FOLDERS = {0: "ps_controller_myunit", 1: "correction_tape_myunit"}
CLASS_NAMES = {0: "ps_controller_myunit", 1: "correction_tape_myunit"}

SPLIT_RATIO = 0.8
RANDOM_SEED = 42

YOLO_DIR = BASE_DIR / "DATASET_YOLO"
IMAGES_TRAIN = YOLO_DIR / "images" / "train"
IMAGES_VAL = YOLO_DIR / "images" / "val"
LABELS_TRAIN = YOLO_DIR / "labels" / "train"
LABELS_VAL = YOLO_DIR / "labels" / "val"
TRAINING_RESULTS = BASE_DIR / "TRAINING_RESULTS"
MODEL_DIR = BASE_DIR / "MODEL"
DEMO_DIR = BASE_DIR / "DEMO_VIDEO"
LAPORAN_DIR = BASE_DIR / "LAPORAN"
CODE_DIR = BASE_DIR / "CODE"
README_FILE = BASE_DIR / "dataset_info.txt"
DATA_YAML = YOLO_DIR / "data.yaml"


def create_folder_structure():
    folders = [
        IMAGES_TRAIN,
        IMAGES_VAL,
        LABELS_TRAIN,
        LABELS_VAL,
        TRAINING_RESULTS,
        TRAINING_RESULTS / "sample_bbox_quality",
        MODEL_DIR,
        DEMO_DIR,
        DEMO_DIR / "pred_images",
        LAPORAN_DIR,
        CODE_DIR,
    ]
    for f in folders:
        f.mkdir(parents=True, exist_ok=True)


def collect_images():
    """Collect all jpg images from dataset_raw folder recursively."""
    images = []
    for class_id, folder_name in CLASS_FOLDERS.items():
        class_dir = DATASET_RAW_DIR / folder_name
        if not class_dir.exists():
            print(f"  Folder not found: {class_dir}")
            continue
        jpg_files = sorted(glob.glob(str(class_dir / "**" / "*.jpg"), recursive=True))
        for fpath in jpg_files:
            fpath = Path(fpath)
            images.append((fpath, class_id, fpath.name))
        print(f"  Class {class_id} ({folder_name}): {len(jpg_files)} images")
    return images


def stratified_split(images):
    """Split 80/20 per class to balance distribution."""
    random.seed(RANDOM_SEED)
    by_class = defaultdict(list)
    for item in images:
        by_class[item[1]].append(item)

    train_list, val_list = [], []
    for class_id in sorted(by_class.keys()):
        class_images = by_class[class_id]
        random.shuffle(class_images)
        n_train = int(len(class_images) * SPLIT_RATIO)
        train_list.extend(class_images[:n_train])
        val_list.extend(class_images[n_train:])
        print(f"  Class {class_id}: {n_train} train, {len(class_images) - n_train} val")
    return train_list, val_list


def copy_images(train_list, val_list):
    for src_path, _, fname in train_list:
        shutil.copy2(str(src_path), str(IMAGES_TRAIN / fname))
    for src_path, _, fname in val_list:
        shutil.copy2(str(src_path), str(IMAGES_VAL / fname))
    print(f"  Copied {len(train_list)} train + {len(val_list)} val")


def generate_data_yaml():
    yaml_content = f"""path: {str(YOLO_DIR).replace(chr(92), '/')}
train: images/train
val: images/val

names:
  0: ps_controller_myunit
  1: correction_tape_myunit
"""
    DATA_YAML.write_text(yaml_content, encoding="utf-8")
    print(f"  data.yaml saved")


def _roboflow_stem_to_original(stem):
    """Extract original name from roboflow exported file."""
    if "_jpg.rf." in stem:
        return stem.split("_jpg.rf.")[0]
    return stem


def _remap_label_line(line):
    """
    Remap class ID and convert format.
    Roboflow order: 0=correction_tape, 1=ps_controller
    We use: 0=ps_controller, 1=correction_tape
    So we swap 0<->1.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return " ".join(parts)

    cls_id = int(parts[0])
    remapped = 1 - cls_id  # swap 0<->1

    coords = [float(x) for x in parts[1:]]

    if len(parts) == 5:
        return f"{remapped} {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
    else:
        # polygon format -> convert to bbox
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        xc = max(0.0, min(1.0, (x_min + x_max) / 2))
        yc = max(0.0, min(1.0, (y_min + y_max) / 2))
        w = max(0.0, min(1.0, x_max - x_min))
        h = max(0.0, min(1.0, y_max - y_min))
        return f"{remapped} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def sync_labels(labels_raw_dir):
    """Sync exported roboflow .txt labels to labels/train and labels/val folders."""
    labels_raw = Path(labels_raw_dir)
    if not labels_raw.exists():
        print(f"  Labels dir not found: {labels_raw}")
        sys.exit(1)

    all_labels = {}
    for txt_file in glob.glob(str(labels_raw / "**" / "*.txt"), recursive=True):
        txt_path = Path(txt_file)
        if txt_path.name.lower().startswith("readme"):
            continue
        original_name = _roboflow_stem_to_original(txt_path.stem)
        all_labels[original_name] = txt_path

    print(f"  Found {len(all_labels)} label files")

    def write_remapped(src_path, dst_path):
        content = src_path.read_text(encoding="utf-8").strip()
        if not content:
            dst_path.write_text("", encoding="utf-8")
            return
        lines = [_remap_label_line(l) for l in content.split("\n") if l.strip()]
        dst_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    matched_train = matched_val = 0
    missing_train = missing_val = 0

    for img_file in sorted(IMAGES_TRAIN.glob("*.jpg")):
        if img_file.stem in all_labels:
            write_remapped(
                all_labels[img_file.stem], LABELS_TRAIN / f"{img_file.stem}.txt"
            )
            matched_train += 1
        else:
            missing_train += 1

    for img_file in sorted(IMAGES_VAL.glob("*.jpg")):
        if img_file.stem in all_labels:
            write_remapped(
                all_labels[img_file.stem], LABELS_VAL / f"{img_file.stem}.txt"
            )
            matched_val += 1
        else:
            missing_val += 1

    print(f"  Train: {matched_train} matched, {missing_train} missing")
    print(f"  Val  : {matched_val} matched, {missing_val} missing")
    return missing_train + missing_val


def validate_labels():
    """Check all labels: each image must have a .txt, class 0/1, YOLO format."""
    errors = []
    total_bboxes = 0

    for split_name, img_dir, lbl_dir in [
        ("train", IMAGES_TRAIN, LABELS_TRAIN),
        ("val", IMAGES_VAL, LABELS_VAL),
    ]:
        for img_file in sorted(img_dir.glob("*.jpg")):
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                errors.append(f"[{split_name}] No label: {img_file.name}")
                continue

            lines = lbl_file.read_text(encoding="utf-8").strip().split("\n")
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(
                        f"[{split_name}] {lbl_file.name} line {i}: got {len(parts)} values"
                    )
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = [float(x) for x in parts[1:]]
                except ValueError:
                    errors.append(
                        f"[{split_name}] {lbl_file.name} line {i}: format error"
                    )
                    continue
                if cls_id not in (0, 1):
                    errors.append(
                        f"[{split_name}] {lbl_file.name} line {i}: class={cls_id}"
                    )
                for name, val in [("xc", xc), ("yc", yc), ("w", w), ("h", h)]:
                    if not (0.0 <= val <= 1.0):
                        errors.append(
                            f"[{split_name}] {lbl_file.name} line {i}: {name}={val}"
                        )
                total_bboxes += 1

    print(f"  Total bboxes: {total_bboxes}")
    if errors:
        print(f"  {len(errors)} errors:")
        for e in errors[:10]:
            print(f"    - {e}")
    else:
        print(f"  All labels are valid")
    return errors


def count_classes(label_dir):
    counts = defaultdict(int)
    for txt_file in sorted(label_dir.glob("*.txt")):
        for line in txt_file.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                counts[int(line.split()[0])] += 1
    return dict(counts)


def write_readme(train_list, val_list):
    train_cls = count_classes(LABELS_TRAIN)
    val_cls = count_classes(LABELS_VAL)

    train_per_class = defaultdict(int)
    val_per_class = defaultdict(int)
    for _, cls_id, _ in train_list:
        train_per_class[cls_id] += 1
    for _, cls_id, _ in val_list:
        val_per_class[cls_id] += 1

    lines = [
        "UAS COMPUTER VISION 2025-2026",
        "Personal Environment Object Detector",
        "=" * 45,
        f"Total images: {len(train_list) + len(val_list)}",
        f"Train: {len(train_list)}, Val: {len(val_list)}",
        f"Split: {SPLIT_RATIO:.0%}/{1-SPLIT_RATIO:.0%}, seed={RANDOM_SEED}",
        "",
        "Classes:",
        "  0: ps_controller_myunit",
        "  1: correction_tape_myunit",
        "",
        "Distribution per class:",
    ]
    for cls_id, name in CLASS_NAMES.items():
        lines.append(
            f"  {name}: {train_per_class.get(cls_id,0)} train, {val_per_class.get(cls_id,0)} val"
        )

    if train_cls or val_cls:
        lines.append("")
        lines.append("Bbox per kelas:")
        for cls_id, name in CLASS_NAMES.items():
            lines.append(
                f"  {name}: {train_cls.get(cls_id,0)} train, {val_cls.get(cls_id,0)} val"
            )

    lines.extend(["", "Model: YOLO11n, imgsz=640", "Annotasi: Roboflow (YOLO format)"])
    README_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Persiapan dataset YOLO")
    parser.add_argument(
        "--labels_raw",
        type=str,
        default=None,
        help="Path to Roboflow YOLO labels folder",
    )
    args = parser.parse_args()

    print("=" * 45)
    print("PREPARE DATASET")
    print("=" * 45)

    print("\n[1] Creating folders...")
    create_folder_structure()

    print("[2] Collecting images...")
    images = collect_images()
    if not images:
        print("  No images found!")
        sys.exit(1)
    print(f"  Total: {len(images)}")

    print("[3] Split 80/20...")
    train_list, val_list = stratified_split(images)

    print("[4] Copying images...")
    copy_images(train_list, val_list)

    print("[5] Generate data.yaml...")
    generate_data_yaml()

    if args.labels_raw:
        print("[6] Sync labels...")
        sync_labels(args.labels_raw)
        print("  Validate...")
        validate_labels()
    else:
        print("[6] Labels not synced")
        print(f"  Run: python prepare_dataset.py --labels_raw <path>")

    print("\nWriting README...")
    write_readme(train_list, val_list)

    print(f"\nDone! Dataset: {YOLO_DIR}")


if __name__ == "__main__":
    main()
