import os

def check_class_ids(expected_id):
    # The directories to check
    labels_dirs = [
        "dataset/Laptop/train/labels"
    ]

    mismatched_files = []

    print(f"Verifying that all class IDs equals to '{expected_id}'...")

    # Iterate through each directory
    for labels_dir in labels_dirs:
        if not os.path.isdir(labels_dir):
            print(f"Warning: Directory not found, skipping: {labels_dir}")
            continue

        # Iterate over each file in the labels directory
        for filename in os.listdir(labels_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(labels_dir, filename)
                with open(filepath, 'r') as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) > 0:
                            try:
                                class_id = int(parts[0])
                                if class_id != expected_id:
                                    mismatched_files.append((filepath, i + 1, class_id))
                            except ValueError:
                                print(f"Could not parse class ID in {filepath} on line {i+1}")

    if not mismatched_files:
        print(f"Verification successful! All files checked contain only class ID '{expected_id}'.")
    else:
        print("Verification failed. The following files contain incorrect class IDs:")
        for file, line_num, found_id in mismatched_files:
            print(f"  - File: {file}, Line: {line_num}, Found ID: {found_id} (Expected: {expected_id})")

ID_TO_VERIFY = 3
check_class_ids(ID_TO_VERIFY)