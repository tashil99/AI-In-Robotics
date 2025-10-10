import os
from collections import defaultdict

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

# The root directory to scan for duplicates.
DATASET_DIR = "../merged-dataset"

if not os.path.exists(DATASET_DIR):
    print("❌ INCORRECT PATH")
else:
    find_duplicate_filenames(DATASET_DIR)


