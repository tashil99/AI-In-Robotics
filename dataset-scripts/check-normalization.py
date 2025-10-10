# Python
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple


def check_pixel_range(
        image_dir: str,
        valid_ext: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
) -> Dict[str, List[str]]:
    """
    Check if image pixel values are in expected [0, 255] range (uint8).

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
        for fname in files:
            if not fname.lower().endswith(valid_ext):
                continue
            path = os.path.join(root, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                results["unreadable"].append(path)
                continue

            min_val = float(np.min(img))
            max_val = float(np.max(img))

            # Check if in standard [0, 255] uint8 range
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

if __name__ == "__main__":
    target_dir = "../merged-dataset/valid/images"
    res = check_pixel_range(target_dir)
    print_pixel_range_report(res)

