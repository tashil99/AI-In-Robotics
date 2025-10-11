import os
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
from typing import Dict, List, Tuple

#Function to check if all images are readable
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

# Function to check if all images have label files
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

#Function to find duplicate filenames in a given folder and its subfolders
def find_duplicate_filenames(root_folder):
    """
    Finds files with the same name in a given folder and its subfolders.
    Prints total files scanned and lists duplicate filenames (if any).
    """
    filenames = defaultdict(list)
    duplicates = []
    total_files = 0

    print(f"🔍 Scanning for duplicate filenames in '{root_folder}'...\n")

    # Walk through the directory tree
    for dirpath, _, file_list in os.walk(root_folder):
        for filename in file_list:
            total_files += 1
            filenames[filename].append(os.path.join(dirpath, filename))

    # Find duplicates
    for filename, paths in filenames.items():
        if len(paths) > 1:
            duplicates.append(paths)

    print(f"\n📂 Total files checked: {total_files}")

    if not duplicates:
        print("✅ No duplicate filenames found.")
    else:
        print(f"\n⚠️ Found {len(duplicates)} sets of duplicate filenames:")
        for i, path_list in enumerate(duplicates, 1):
            print(f"\nSet {i} (Filename: '{os.path.basename(path_list[0])}'):")
            for filepath in path_list:
                print(f"  - {filepath}")

# Function to validate YOLO annotation format
def validate_annotations(label_dir, num_classes=6):
    """
    Validates YOLO format annotations in label files
    Checks: format, normalization (0-1 range), bounds, and class IDs
    """
    errors = []
    total_files = 0
    total_annotations = 0

    for file in os.listdir(label_dir):
        if file.lower().endswith('.txt'):
            total_files += 1
            label_path = os.path.join(label_dir, file)

            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                if not lines or all(line.strip() == '' for line in lines):
                    continue

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()

                    # Check format (should have 5 values)
                    if len(parts) != 5:
                        errors.append(f"{label_path} Line {line_num}: Expected 5 values, got {len(parts)}")
                        continue

                    try:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        total_annotations += 1

                        # Check class ID
                        if class_id < 0 or class_id >= num_classes:
                            errors.append(
                                f"{label_path} Line {line_num}: Invalid class_id={class_id} (valid: 0-{num_classes - 1})")

                        # Check normalization (0-1 range, inclusive)
                        if center_x < 0 or center_x > 1:
                            errors.append(f"{label_path} Line {line_num}: center_x={center_x} out of range [0,1]")
                        if center_y < 0 or center_y > 1:
                            errors.append(f"{label_path} Line {line_num}: center_y={center_y} out of range [0,1]")
                        if width < 0 or width > 1:
                            errors.append(f"{label_path} Line {line_num}: width={width} out of range [0,1]")
                        if height < 0 or height > 1:
                            errors.append(f"{label_path} Line {line_num}: height={height} out of range [0,1]")

                        # Check dimensions are positive
                        if width <= 0 or height <= 0:
                            errors.append(f"{label_path} Line {line_num}: width and height must be > 0")

                        # Check if the bounding box is within image boundaries
                        x_min = center_x - width / 2
                        x_max = center_x + width / 2
                        y_min = center_y - height / 2
                        y_max = center_y + height / 2

                        if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
                            errors.append(f"{label_path} Line {line_num}: Bounding box extends beyond image boundaries")

                    except ValueError:
                        errors.append(f"{label_path} Line {line_num}: Cannot parse values")

            except Exception as e:
                errors.append(f"{label_path}: Error reading file - {e}")

    print(f"Checked {total_files} files with {total_annotations} annotations")

    if errors:
        print(f"❌ Found {len(errors)} annotation errors:")
        for error in errors:
            print(f"   {error}")
    else:
        print("✅ All annotations are valid")

#Function to delete very large files
def delete_large_images(image_dir, sizes, min_size=820):
    """
    Ask the user before deleting large and very large images and their corresponding labels
    """
    large_images = [s for s in sizes if s[0] > min_size or s[1] > min_size]
    very_large_images = [s for s in sizes if s[0] > 4000 or s[1] > 4000]

    if not large_images:
        print("\n✅ No large images to delete")
        return

    print(f"\n🗑️  Found {len(large_images)} large images (>{min_size}px)")
    if very_large_images:
        print(f"   Including {len(very_large_images)} very large images (>4000px)")

    # Determine label directory
    label_dir = image_dir.replace("/images", "/labels").replace("\\images", "\\labels")

    # Show the images that would be deleted
    print("\n📋 Images and labels that will be deleted:")
    for width, height, filename in large_images[:20]:
        category = "VERY LARGE" if (width > 4000 or height > 4000) else "LARGE"
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        label_exists = "✓" if os.path.exists(label_path) else "✗"
        print(f"  [{category}] {filename}: {width}x{height} [Label: {label_exists}]")

    if len(large_images) > 20:
        print(f"  ... and {len(large_images) - 20} more")

    # Ask for confirmation
    print(f"\n⚠️  Do you want to delete these {len(large_images)} large images and their labels?")
    response = input("Type 'yes' to confirm deletion, or anything else to cancel: ").strip().lower()

    if response == 'yes':
        deleted_images = 0
        deleted_labels = 0
        failed_images = 0
        failed_labels = 0

        for width, height, filename in large_images:
            image_path = os.path.join(image_dir, filename)
            label_name = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(label_dir, label_name)

            # Delete image
            try:
                os.remove(image_path)
                deleted_images += 1
                print(f"  ✓ Deleted image: {filename}")
            except Exception as e:
                failed_images += 1
                print(f"  ✗ Failed to delete image {filename}: {e}")

            # Delete the corresponding label if it exists
            if os.path.exists(label_path):
                try:
                    os.remove(label_path)
                    deleted_labels += 1
                    print(f"  ✓ Deleted label: {label_name}")
                except Exception as e:
                    failed_labels += 1
                    print(f"  ✗ Failed to delete label {label_name}: {e}")

        print(f"\n📊 Deletion Summary:")
        print(f"  ✅ Successfully deleted: {deleted_images} images, {deleted_labels} labels")
        if failed_images > 0 or failed_labels > 0:
            print(f"  ❌ Failed to delete: {failed_images} images, {failed_labels} labels")
        print(f"  📁 Remaining images: {len(sizes) - deleted_images}")
    else:
        print("\n❌ Deletion cancelled")

#Function to check size of images
def check_image_sizes(image_dir, target_width=640, target_height=640, ask_delete=False):
    """
    Check dimensions of all images in a directory
    """
    sizes = []

    for file in os.listdir(image_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_dir, file)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    sizes.append((width, height, file))
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if not sizes:
        print("No images found")
        return

    # Check for target size
    target_size_images = [s for s in sizes if s[0] == target_width and s[1] == target_height]
    non_target_size_images = [s for s in sizes if s[0] != target_width or s[1] != target_height]

    # Check for the perfect range (416x416 to 640x640)
    perfect_range_images = [s for s in sizes if 416 <= s[0] <= 640 and 416 <= s[1] <= 640]

    print(f"\n🎯 Target Size Check ({target_width}x{target_height}):")
    print(f"Total images: {len(sizes)}")
    print(f"Matching target size: {len(target_size_images)} ({len(target_size_images) / len(sizes) * 100:.1f}%)")
    print(f"Perfect range (416-640): {len(perfect_range_images)} ({len(perfect_range_images) / len(sizes) * 100:.1f}%)")
    print(f"Not matching: {len(non_target_size_images)} ({len(non_target_size_images) / len(sizes) * 100:.1f}%)")

    if len(target_size_images) == len(sizes):
        print(f"\n✅ All images are {target_width}x{target_height}!")
    elif len(target_size_images) > 0:
        print(f"\n⚠️  Some images match {target_width}x{target_height}")
    else:
        print(f"\n❌ No images are {target_width}x{target_height}")

    # Show non-matching images
    if non_target_size_images and len(non_target_size_images) <= 20:
        print(f"\n📋 Images with different sizes:")
        for width, height, filename in non_target_size_images:
            print(f"  {filename}: {width}x{height}")
    elif non_target_size_images:
        print(f"\n📋 First 20 images with different sizes:")
        for width, height, filename in non_target_size_images[:20]:
            print(f"  {filename}: {width}x{height}")

    # Show size categories
    small = sum(1 for w, h, _ in sizes if w < 416 or h < 416)
    medium = sum(1 for w, h, _ in sizes if 640 <= w <= 820 and 640 <= h <= 820)
    large = sum(1 for w, h, _ in sizes if w > 820 or h > 820)
    very_large = sum(1 for w, h, _ in sizes if w > 1920 or h > 1920)

    print(f"\n📦 Size Categories:")
    print(f"  Small (<416px): {small}")
    print(f"  Medium (640-1920px): {medium}")
    print(f"  Large (1920-4000px): {large}")
    print(f"  Very Large (>4000px): {very_large}")

    if very_large > len(sizes) * 0.5:
        print(f"\n⚠️  Warning: {very_large} very large images found - consider resizing")
    elif large > len(sizes) * 0.5:
        print(f"\n💡 Info: Many large images - resizing may improve training speed")
    else:
        print(f"\n✅ Image sizes look reasonable for YOLO training")

    # Ask to delete large images AFTER all checks are done
    if ask_delete and (large > 0 or very_large > 0):
        delete_large_images(image_dir, sizes)

#Functions to check pixel values in images and print report after check completed
def check_pixel_range(
        image_dir: str,
        valid_ext: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
) -> Dict[str, List[str]]:
    """
    Check if image pixel values are in the expected [0, 255] range (uint8).

    Returns:
        Dict with keys:
        - valid_range: images with pixels in [0, 255] (uint8)
        - invalid_range: images with unexpected pixel ranges
        - unreadable: could not load
    """
    results: Dict[str, List[str]] = {
        "valid_range": [],
        "invalid_range": [],
        "unreadable": [],
    }

    for root, _, files in os.walk(image_dir):
        for filename in files:
            if not filename.lower().endswith(valid_ext):
                continue
            path = os.path.join(root, filename)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                results["unreadable"].append(path)
                continue

            min_val = float(np.min(img))
            max_val = float(np.max(img))

            # Check if in the standard [0, 255] uint8 range
            if img.dtype == np.uint8 and 0 <= min_val <= 255 and 0 <= max_val <= 255:
                results["valid_range"].append(f"{path} (range=[{min_val:.0f}, {max_val:.0f}])")
            else:
                results["invalid_range"].append(
                    f"{path} (range=[{min_val:.4f}, {max_val:.4f}], dtype={img.dtype})"
                )

    return results

def print_pixel_range_report(results: Dict[str, List[str]]) -> None:
    total_checked = len(results["valid_range"]) + len(results["invalid_range"]) + len(results["unreadable"])
    print(f"📊 Total files checked: {total_checked}")
    print()

    if results["valid_range"]:
        print(f"✅ Valid [0, 255] range ({len(results['valid_range'])} images)")
    if results["invalid_range"]:
        print(f"❌ Invalid/Unexpected range ({len(results['invalid_range'])}):")
        for e in results["invalid_range"]:
            print(e)
    if results["unreadable"]:
        print(f"❌ Unreadable ({len(results['unreadable'])}):")
        for e in results["unreadable"]:
            print(e)
    if not results["invalid_range"] and not results["unreadable"]:
        print("✅ All images are in standard [0, 255] uint8 range")

#Functions to check for image quality and remove low-quality pictures
def check_image_quality(
    image_dir: str,
    min_size: Tuple[int, int] = (64, 64),
    max_aspect_ratio: float = 5.0,
    low_variance_thresh: float = 3.0,
    valid_ext: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
) -> Dict[str, List[str]]:
    """
    Scan images in a directory (recursively) and return potential quality issues.

    Returns a dict with keys:
      - unreadable: image could not be decoded
      - zero_size: decoded but width/height is 0
      - too_small: below min_size
      - extreme_aspect: aspect ratio beyond max_aspect_ratio
      - low_variance: near-blank (variance below the threshold)

    Args:
        image_dir: Root directory to scan.
        min_size: (min_width, min_height).
        max_aspect_ratio: max(w/h, h/w) allowed.
        low_variance_thresh: threshold on grayscale variance.
        valid_ext: considered file extensions.

    Returns:
        Dict[str, List[str]] mapping issue -> list of paths (with brief info).
    """
    issues: Dict[str, List[str]] = {
        "unreadable": [],
        "zero_size": [],
        "too_small": [],
        "extreme_aspect": [],
        "low_variance": [],
    }

    min_w, min_h = min_size

    for root, _, files in os.walk(image_dir):
        for file in files:
            if not file.lower().endswith(valid_ext):
                continue
            path = os.path.join(root, file)

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                issues["unreadable"].append(path)
                continue

            h, w = img.shape[:2]
            if w == 0 or h == 0:
                issues["zero_size"].append(path)
                continue

            if w < min_w or h < min_h:
                issues["too_small"].append(f"{path} ({w}x{h})")

            ar = max(w / h, h / w)
            if ar > max_aspect_ratio:
                issues["extreme_aspect"].append(f"{path} (AR={ar:.2f}, {w}x{h})")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            var = float(np.var(gray))
            if var < low_variance_thresh:
                issues["low_variance"].append(f"{path} (var={var:.2f})")

    return issues

def print_quality_report(issues: Dict[str, List[str]]) -> None:
    any_issues = any(issues.values())
    if not any_issues:
        print("✅ No obvious image quality issues found")
        return

    for key in ["unreadable", "zero_size", "too_small", "extreme_aspect", "low_variance"]:
        entries = issues.get(key, [])
        if entries:
            title = key.replace("_", " ").title()
            print(f"❌ {title} ({len(entries)}):")
            for e in entries:
                print(e)

def remove_low_quality_images(
    issues: Dict[str, List[str]],
    issue_types_to_remove: List[str] = None,
    label_dir: str = None,
) -> None:
    """
    Remove images with quality issues and their corresponding label files.
    Shows all files and asks for confirmation once before deleting.

    Args:
        issues: Dict returned by check_image_quality.
        issue_types_to_remove: List of issue types to remove (e.g., ['too_small', 'unreadable']).
                                If None, removes all issue types.
        label_dir: Directory containing label files. If None, assumes labels are in a sibling
                   'labels' directory (e.g., images in 'train/images', labels in 'train/labels').
    """
    if issue_types_to_remove is None:
        issue_types_to_remove = list(issues.keys())

    files_to_delete = []
    for issue_type in issue_types_to_remove:
        if issue_type in issues:
            for entry in issues[issue_type]:
                # Extract just the path (entries may have extra info like "(640x480)")
                path = entry.split(" (")[0]
                files_to_delete.append((path, issue_type))

    if not files_to_delete:
        print("No files to delete.")
        return

    print(f"\n{'='*70}")
    print(f"Found {len(files_to_delete)} images with quality issues")
    print(f"{'='*70}\n")

    # Prepare all file pairs
    all_pairs = []
    for img_path, issue_type in files_to_delete:
        # Find the corresponding label file
        if label_dir:
            img_basename = os.path.basename(img_path)
            label_name = os.path.splitext(img_basename)[0] + ".txt"
            label_path = os.path.join(label_dir, label_name)
        else:
            img_dir = os.path.dirname(img_path)
            label_dir_auto = img_dir.replace("/images", "/labels").replace("\\images", "\\labels")
            img_basename = os.path.basename(img_path)
            label_name = os.path.splitext(img_basename)[0] + ".txt"
            label_path = os.path.join(label_dir_auto, label_name)

        all_pairs.append((img_path, label_path, issue_type))

    # Display all files
    for idx, (img_path, label_path, issue_type) in enumerate(all_pairs, 1):
        print(f"{idx}. Issue: {issue_type}")
        print(f"   Image: {img_path}")
        if os.path.exists(label_path):
            print(f"   Label: {label_path}")
        else:
            print(f"   Label: {label_path} (not found)")
        print()

    # Ask for confirmation at once
    print(f"{'='*70}")
    response = input(f"Delete all {len(files_to_delete)} images and their labels? (y/yes/n/no): ").strip().lower()

    if response not in ['y', 'yes']:
        print("\n🛑 Deletion cancelled by user.")
        return

    # Delete all files
    deleted_images = 0
    deleted_labels = 0

    print(f"\n{'='*70}")
    print("Deleting files...")
    print(f"{'='*70}\n")

    for img_path, label_path, _ in all_pairs:
        # Delete image
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
                deleted_images += 1
                print(f"✓ Deleted: {img_path}")
            except Exception as e:
                print(f"✗ Error deleting {img_path}: {e}")

        # Delete label
        if os.path.exists(label_path):
            try:
                os.remove(label_path)
                deleted_labels += 1
                print(f"✓ Deleted: {label_path}")
            except Exception as e:
                print(f"✗ Error deleting {label_path}: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total images deleted: {deleted_images}")
    print(f"Total labels deleted: {deleted_labels}")
    print(f"{'='*70}\n")

#Check for duplicate filenames in the directory below
DATASET_DIRECTORY = "../dataset"
find_duplicate_filenames(DATASET_DIRECTORY)

#Check if all images in the following directory are readable
check_images("../dataset/train/images")

#Check if all images have their corresponding label
check_labels("../dataset/train/images", "../dataset/train/labels")

#Check if all labels in the following directory are in YOLO format
validate_annotations("../dataset/train/labels")

#Check if all images in the following directory are in the expected size
check_image_sizes("../dataset/train/images", ask_delete=True)

#Check if all images in the following directory are in the expected pixel range -- Normalization check
target_dir = "../dataset/train/images"
res = check_pixel_range(target_dir)
print_pixel_range_report(res)

#Check if all images in the following directory are of good quality and remove low-quality pictures
results = check_image_quality(
    target_dir,
    min_size=(400, 400),
    max_aspect_ratio=6.0,
    low_variance_thresh=2.0,
)
print_quality_report(results)
remove_low_quality_images(
    results,
    issue_types_to_remove=["too_small"]
)