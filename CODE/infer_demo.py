# infer_demo.py
# Inference using best.pt model
# Predicts val images, webcam feed, or video file

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(r"E:\dataset\UAS_Computer_Vision")
MODEL_DIR = BASE_DIR / "MODEL"
BEST_PT = MODEL_DIR / "best.pt"
DEMO_DIR = BASE_DIR / "DEMO_VIDEO"
PRED_IMAGES_DIR = DEMO_DIR / "pred_images"
DEMO_VIDEO_OUT = DEMO_DIR / "demo_deteksi.mp4"
VAL_IMAGES = BASE_DIR / "DATASET_YOLO" / "images" / "val"

DEFAULT_CONF = 0.15
DEFAULT_IMGSZ = 640


def predict_val_images(model, conf):
    """Predict all validation images and save results."""
    print(f"\nPredicting val images (conf={conf})...")
    PRED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(VAL_IMAGES),
        conf=conf,
        imgsz=DEFAULT_IMGSZ,
        save=True,
        project=str(DEMO_DIR),
        name="pred_images",
        exist_ok=True,
    )

    for r in results:
        fname = Path(r.path).name
        n = len(r.boxes)
        print(f"  {fname}: {n} detections")
    print(f"Output: {PRED_IMAGES_DIR}")


def predict_webcam(model, conf):
    """Live detection using webcam. Press 'q' to exit."""
    import cv2

    print(f"\nWebcam mode (conf={conf})")
    print("Press 'q' in the window to exit")
    print("Tip: Win+Alt+R for screen recording\n")

    for r in model.predict(
        source=0,
        conf=conf,
        imgsz=DEFAULT_IMGSZ,
        show=True,
        stream=True,
    ):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def predict_video(model, video_path, conf):
    """Predict on video file, save output."""
    import shutil

    print(f"\nPredicting video: {video_path}")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    model.predict(
        source=str(video_path),
        conf=conf,
        imgsz=DEFAULT_IMGSZ,
        save=True,
        project=str(DEMO_DIR),
        name="video_output",
        exist_ok=True,
    )

    # copy video output
    out_dir = DEMO_DIR / "video_output"
    if out_dir.exists():
        vids = list(out_dir.glob("*.avi")) + list(out_dir.glob("*.mp4"))
        if vids:
            src = vids[0]
            if src.suffix == ".avi":
                # convert to mp4
                try:
                    import cv2

                    cap = cv2.VideoCapture(str(src))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = cv2.VideoWriter(
                        str(DEMO_VIDEO_OUT),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        writer.write(frame)
                    cap.release()
                    writer.release()
                except Exception:
                    shutil.copy2(str(src), str(DEMO_VIDEO_OUT.with_suffix(".avi")))
            else:
                shutil.copy2(str(src), str(DEMO_VIDEO_OUT))
            print(f"Output: {DEMO_VIDEO_OUT}")


def main():
    parser = argparse.ArgumentParser(description="Inference demo")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="0 for webcam, path for video, empty for val images",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    args = parser.parse_args()

    if not BEST_PT.exists():
        print(f"Model not found: {BEST_PT}")
        print("Run train_and_eval.py first")
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(str(BEST_PT))
    conf = args.conf

    if args.source is None:
        predict_val_images(model, conf)
    elif args.source == "0":
        predict_webcam(model, conf)
    else:
        video_path = Path(args.source)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            sys.exit(1)
        predict_video(model, args.source, conf)

    # screen recording instructions
    print(
        f"""
Screen Recording:
  1. Win+Alt+R to start recording
  2. Run: python infer_demo.py --source 0
  3. Point objects at the camera
  4. Win+Alt+R to stop
  Video is saved in Videos/Captures
"""
    )


if __name__ == "__main__":
    main()
