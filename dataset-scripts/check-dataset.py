import os
import cv2

def check_images(image_dir):
    bad_images = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    bad_images.append(img_path)

    if bad_images:
        print(f"❌ Corrupted or unreadable images found ({len(bad_images)} total):")
        for b in bad_images:
            print(b)
    else:
        print("✅ All images are readable")

check_images("../merged-dataset/test/images")


# Function to check if all images have label files
# Check if image name matches that in the label file
def check_labels(image_dir, label_dir):
    missing_labels = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            label_file = os.path.splitext(file)[0] + ".txt"
            if not os.path.exists(os.path.join(label_dir, label_file)):
                missing_labels.append(file)

    if missing_labels:
        print(f"❌ Missing label files ({len(missing_labels)} total):")
        for m in missing_labels:
            print(m)
    else:
        print("✅ All images have label files")

check_labels("../merged-dataset/test/images", "../merged-dataset/test/labels")
