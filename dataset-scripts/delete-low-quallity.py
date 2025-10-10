# Python
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple


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
      - low_variance: near-blank (variance below threshold)

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
        for fname in files:
            if not fname.lower().endswith(valid_ext):
                continue
            path = os.path.join(root, fname)

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


if __name__ == "__main__":
    # Path of directory to scan
    target_dir = "../dataset/laptop/train/images"
    results = check_image_quality(
        target_dir,
        min_size=(400, 400),
        max_aspect_ratio=6.0,
        low_variance_thresh=2.0,
    )
    print_quality_report(results)