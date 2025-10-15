import os
from ultralytics import YOLO
import shutil
import cv2
import glob

# Define paths
MODEL_PATH = "runs/detect/AI-In-Robotics-CPU-Exp502/weights/best.pt"
SOURCE_PATH = "../dataset/test/images"
SAVE_DIR = "runs/predict/AI-In-Robotics-CPU-Test"  # fixed output folder

# Clear old results and create folder
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

# Ensure paths exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
if not os.path.exists(SOURCE_PATH):
    raise FileNotFoundError(f"Source path not found: {SOURCE_PATH}")

# Load trained YOLO model
model = YOLO(MODEL_PATH)

# Run inference
results = model.predict(
    source=SOURCE_PATH,   # directory or single image
    imgsz=320,            # smaller = faster
    conf=0.25,            # confidence threshold
    save=True,            # save annotated images
    save_dir=SAVE_DIR,    # fixed output folder
    device="cpu"          # CPU mode
)

# Print summary
print("\n✅ Inference complete!")
print(f"Detected {len(results)} image(s)")
print(f"Results saved to: {SAVE_DIR}")
print(f"Model used: {MODEL_PATH}")

# Preview all predicted images one by one
predicted_images = sorted(glob.glob(os.path.join(SAVE_DIR, "*.jpg")))

if predicted_images:
    print("\n🖼 Press any key to move to the next image. Press 'Esc' to exit early.")
    for img_path in predicted_images:
        img = cv2.imread(img_path)
        cv2.imshow("Prediction", img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()
else:
    print("No predicted images found to preview.")
