import os

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

                        # Check if bounding box is within image boundaries
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


validate_annotations("../dataset/laptop/train/labels")