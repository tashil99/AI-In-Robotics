# scripts/training-model.py

import os
import pickle
from ultralytics import YOLO

# Prevent OpenMP duplicate runtime crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Train
results = model.train(
    data="../dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="AI-In-Robotics-Exp1",
    verbose=True
)

# Save results object (pickle)
results_dir = "runs/detect/AI-In-Robotics-Exp1"
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "results.pkl")

with open(results_path, 'wb') as f:
    pickle.dump(results, f)

print(f"\nTraining complete! Results saved to: {results_path}")
print(f"Plots, labels, and checkpoints are in '{results_dir}'")
