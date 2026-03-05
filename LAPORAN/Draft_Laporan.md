# Computer Vision Final Exam Report

**Name:** Rifky Nurfaishal  
**ID:** 001202405012

## Custom Object Detection — YOLO11n

---

## 1. Dataset

I collected 92 photos of 2 objects commonly found around my desk: a **PS Controller** and a **Correction Tape**. There are 46 photos for each class. The images were taken across 6 different scenes/locations to provide a good variety of backgrounds, lighting environments, and angles.

| Detail          | Value                           |
| --------------- | ------------------------------- |
| Total           | 92 images                       |
| Class 0         | ps_controller_myunit            |
| Class 1         | correction_tape_myunit          |
| Distribution    | 46 / 46 (balanced)              |
| Split           | 80% train, 20% val (stratified) |
| Annotation Tool | Roboflow                        |

Split distribution:

| Class           | Train | Val |
| --------------- | ----- | --- |
| ps_controller   | 36    | 10  |
| correction_tape | 36    | 10  |

---

## 2. Annotation

Annotations were done using Roboflow. I initially used the "Smart Tools / Find Object" feature to create polygon bounding boxes, which were later exported and automatically converted to the YOLO bounding box format via script.

Example bounding boxes can be viewed in the [`../TRAINING_RESULTS/sample_bbox_quality/`](../TRAINING_RESULTS/sample_bbox_quality/) folder.

**Important Note:** I discovered that the Smart Tools in Roboflow often generated bounding boxes that were excessively large for the PS controller. This became a key finding in my error analysis (see Section 5).

---

## 3. Training

| Parameter  | Value                |
| ---------- | -------------------- |
| Model      | YOLO11n (yolo11n.pt) |
| Image size | 640                  |
| Epochs     | 150                  |
| Batch size | 16                   |
| Hardware   | CPU (i5-12400F)      |
| Time taken | ~24 minutes          |

---

## 4. Results

### Overall Metrics

| Metric       | Value |
| ------------ | ----- |
| mAP@0.5      | 0.796 |
| mAP@0.5:0.95 | 0.756 |
| Precision    | 0.967 |
| Recall       | 0.773 |

### Per-Class Metrics

| Class           | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --------------- | --------- | ------ | ------- | ------------ |
| ps_controller   | 0.934     | 1.000  | 0.995   | 0.995        |
| correction_tape | 1.000     | 0.547  | 0.597   | 0.518        |

The metrics above show the validation results after 150 epochs. While the PS controller scored nearly perfectly on this static validation set, the correction tape had a low recall of 0.547 (meaning almost half were missed). Interestingly, when I tested the model live with my webcam, the real-world performance did not match these validation numbers. Instead, testing revealed a clear trade-off that depended heavily on how long the model was trained (50 vs. 150 epochs). A detailed explanation of these contrasting real-world results is provided in Section 5.

---

## 5. Evaluation and Failure Analysis (50 vs. 150 Epochs)

To better understand the model's limitations and behavior, I evaluated the predictions across two different training durations: 50 epochs and 150 epochs. This comparison revealed an interesting trade-off over the course of the training.

### Result at 50 Epochs

- **PS Controller (Underfitting / High False Positives):** The model struggled significantly. It frequently misidentified generic dark objects and even my dark shirt as the PS controller. This showed that the model hadn't learned enough distinct features to separate the controller's dark matte texture from similarly colored distractor objects.
- **Correction Tape (Excellent Performance):** The detection for the correction tape was virtually perfect and highly accurate. The bounding boxes generated during live inference fit the tape tightly without randomly expanding.

### Result at 150 Epochs

To fix the poor detection of the PS Controller, I expanded the training up to 150 epochs. Surprisingly, the live webcam test results completely flipped:

- **PS Controller (Highly Accurate):** The extended training successfully mapped the PS controller's features. The model rarely misidentifies shirts or random background objects as the controller anymore. However, a minor visual quirk appeared: the bounding box sometimes widens outwards briefly before quickly snapping back to the correct size. This bounding box instability is a common side-effect when models attempt to aggregate surrounding pixels while struggling with dynamic lighting and motion blur in a live webcam feed.
- **Correction Tape (Over-sensitive / High False Positives):** Extending the training unexpectedly caused the correction tape detection to degrade. The model became overly eager and started misidentifying hands, bright shirts, or light-colored surfaces as the correction tape. This indicates early signs of overfitting—the model dropped its precision threshold and aggressively guessed that anything vaguely light-colored or shaped like a dispenser was the tape.

### Summary of the Trade-Off

| Training State | PS Controller                                                | Correction Tape                                      |
| -------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| **50 Epochs**  | Frequent false positives (confused by dark shirts).          | Perfect detection, highly accurate bounding boxes.   |
| **150 Epochs** | Highly accurate. Box occasionally widens but quickly resets. | Frequent false positives (confused by hands/shirts). |

**Conclusion:**
This behavior perfectly highlights the core limitation of using a very small baseline dataset (92 images). Pushing the model to train longer (from 50 to 150 epochs) managed to fix the underfitting issue for the dark, feature-sparse object (PS Controller) by forcing the model to learn its specific geometry. However, doing so simultaneously caused overfitting for the bright object (Correction Tape), removing its initial precision.

**Possible Improvements for Future Work:**

1. **Add Negative Examples:** Expand the dataset to include diverse images of human hands holding nothing, and photos of people wearing various shirts without the target objects. This teaches the model exactly what the background "looks like".
2. **Increase Background Diversity:** Add more cluttered scene diversity so the model does not hyper-fixate on specific background lighting or random textures in the room.

---

## 6. Submission Files Tree

```text
UAS_Computer_Vision/
├── README.md
├── LAPORAN/Draft_Laporan.md
├── DATASET_YOLO/
│   ├── data.yaml
│   ├── images/ (train + val)
│   └── labels/ (train + val)
├── TRAINING_RESULTS/
│   ├── metrics.txt
│   ├── results.png, confusion_matrix.png
│   ├── PR_curve.png, F1_curve.png
│   ├── val_batch0_pred.jpg
│   └── sample_bbox_quality/
├── MODEL/ (best.pt, last.pt)
├── DEMO_VIDEO/
│   ├── pred_images/
│   ├── Demo Video.mp4
└── CODE/
    ├── prepare_dataset.py
    ├── train_and_eval.py
    ├── infer_demo.py
    └── requirements.txt
```
