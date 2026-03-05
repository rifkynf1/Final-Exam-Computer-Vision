# Computer Vision Final Exam

**Name:** Rifky Nurfaishal  
**ID:** 001202405012

**Project:** Personal Environment Object Detector

---

## Project Description

This project trains a custom object detection model (YOLO11n) to detect 2 specific objects in my personal environment:

1. **PS Controller**
2. **Correction Tape**

## Deliverables and File Locations (As per Requirements)

Here are the locations of the required files for grading. I have linked the directories so you can just click on them:

- **1. Custom Dataset (Images & Labels)**
  Located in the [`DATASET_YOLO/`](DATASET_YOLO/) folder. This contains 92 self-collected images with consistent bounding box annotations.
- **2. Training Results (mAP graphs, F1-Curve, Confusion Matrix)**
  Located in the [`TRAINING_RESULTS/`](TRAINING_RESULTS/) folder. The accuracy summary is also discussed in the report.
- **3. Best Model (Weights)**
  Located in the [`MODEL/`](MODEL/) folder (file `best.pt`).

- **4. Demo Video**
  Located in the [`DEMO_VIDEO/`](DEMO_VIDEO/) folder.

- **5. "Garbage In, Garbage Out" Analysis, Dataset Quality & mAP**
  Located in the [`LAPORAN/Draft_Laporan.md`](LAPORAN/Draft_Laporan.md) file. This document includes a complete analysis of why the PS Controller detection is less precise in a new environment like the webcam test.

## Dataset Statistics

- **Total images:** 92
- **Split Ratio:** 72 Train, 20 Val (80% / 20%, stratified, seed=42)

**Class Distribution:**

- `ps_controller_myunit`: 36 train, 10 val
- `correction_tape_myunit`: 36 train, 10 val

**Model Training:**

- Base Model: YOLO11n (imgsz=640)
- Epochs: 150
- Hardware: CPU (i5-12400F)

---

_Detailed reporting regarding accuracy metrics, dataset analysis, and bounding box quality can be found in [`LAPORAN/Draft_Laporan.md`](LAPORAN/Draft_Laporan.md)._
