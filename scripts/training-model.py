# scripts/training-model.py
import os
import pickle
from ultralytics import YOLO

def main():
    # Avoid OpenMP duplicate runtime crash
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # CPU-specific performance tweaks
    os.environ["OMP_NUM_THREADS"] = "4"  # limit CPU threading
    os.environ["MKL_NUM_THREADS"] = "4"  # limit MKL (PyTorch backend)
    os.environ["ULTRALYTICS_CACHE"] = "ram"  # speed up by caching to RAM

    # Load lightweight YOLOv8 model
    model = YOLO("runs/detect/AI-In-Robotics-CPU-Exp502/weights/best.pt")

    # Train (GPU-optimized config)
    results = model.train(
        data="../dataset/data.yaml",
        epochs=50,
        imgsz=512,
        batch=8,
        name="AI-In-Robotics-CPU-Exp502",
        verbose=True,
        optimizer="AdamW",
        lr0=0.0005,
        pretrained=True,
        freeze=0,
        degrees=5.0,
        translate=0.1,
        scale=0.2,
        shear=2.0,
        mosaic=0.5,
        mixup=0.05,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.2,
    )

    # Save results
    results_dir = "runs/detect/AI-In-Robotics-CPU-Exp502"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n✅ Training complete! Results saved to: {results_path}")
    print(f"📂 Plots, labels, and checkpoints are in '{results_dir}'")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
