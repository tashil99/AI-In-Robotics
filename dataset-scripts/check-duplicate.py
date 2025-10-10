import os
from collections import defaultdict

def find_duplicate_filenames(root_folder):
    """
    Finds files with the same name in a given folder and its subfolders.

    Args:
        root_folder (str): The path to the root folder to start scanning.
    """
    filenames = defaultdict(list)
    duplicates = []

    print(f"Scanning for duplicate filenames in '{root_folder}'...")

    # Walk through the directory tree
    for dirpath, _, file_list in os.walk(root_folder):
        for filename in file_list:
            # Store the full path of each file, keyed by its name
            filenames[filename].append(os.path.join(dirpath, filename))

    # Find filenames that are associated with more than one path
    for filename, paths in filenames.items():
        if len(paths) > 1:
            duplicates.append(paths)

    if not duplicates:
        print("No duplicate filenames found.")
    else:
        print("Found the following sets of duplicate filenames:")
        for i, path_list in enumerate(duplicates, 1):
            print(f"\nSet {i} (Filename: '{os.path.basename(path_list[0])}'):")
            for filepath in path_list:
                print(f"  - {filepath}")

# The root directory to scan for duplicates.
DATASET_DIR = "../dataset"

find_duplicate_filenames(DATASET_DIR)

