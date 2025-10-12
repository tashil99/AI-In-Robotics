# scripts/training-model.py
import os
import pickle
from ultralytics import YOLO

# Avoid OpenMP duplicate runtime crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CPU-specific performance tweaks
os.environ["OMP_NUM_THREADS"] = "4"  # limit CPU threading
os.environ["MKL_NUM_THREADS"] = "4"  # limit MKL (PyTorch backend)
os.environ["ULTRALYTICS_CACHE"] = "ram"  # speed up by caching to RAM (if enough memory)

# Load lightweight YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' = nano version, smallest available

# Train (CPU-optimized config)
results = model.train(
    data="../dataset/data.yaml",
    epochs=3,          # keep short for CPU runs
    imgsz=320,         # smaller input = faster (sacrifice some accuracy)
    batch=4,           # smaller batch fits CPU cache
    device="cpu",      # force CPU explicitly
    workers=2,         # limit dataloader threads to avoid overload
    name="AI-In-Robotics-CPU-Exp1",
    verbose=True,
    optimizer="SGD",   # lighter than Adam
    lr0=0.01,          # lower LR for small batch size
    patience=2,        # early stopping faster
)

# Save results (optional)
results_dir = "runs/detect/AI-In-Robotics-CPU-Exp1"
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "results.pkl")

with open(results_path, "wb") as f:
    pickle.dump(results, f)

print(f"\n✅ Training complete! Results saved to: {results_path}")
print(f"📂 Plots, labels, and checkpoints are in '{results_dir}'")
