import os
from collections import defaultdict

# def delete_files_with_class_id(class_id_to_find, labels_dir="../dataset/valid/labels", images_dir="../dataset/valid/images"):
#     """
#     Delete all label files that contain a specific class ID and their corresponding image files.
#
#     Args:
#         class_id_to_find (int): The class ID to search for.
#         labels_dir (str): Directory containing YOLO label files (.txt).
#         images_dir (str): Directory containing image files.
#
#     Returns:
#         None
#     """
#     if not os.path.isdir(labels_dir):
#         print(f"Error: Labels directory not found - {labels_dir}")
#         return
#
#     deleted_count = 0
#     not_found_images = []
#
#     for filename in os.listdir(labels_dir):
#         if not filename.endswith(".txt"):
#             continue
#
#         label_path = os.path.join(labels_dir, filename)
#         delete_this = False
#
#         # Check if the file contains the class ID
#         with open(label_path, "r") as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) > 0:
#                     try:
#                         class_id = int(parts[0])
#                         if class_id == class_id_to_find:
#                             delete_this = True
#                             break
#                     except ValueError:
#                         continue
#
#         if delete_this:
#             # Delete label file
#             os.remove(label_path)
#             deleted_count += 1
#
#             # Try to delete corresponding image
#             image_name = os.path.splitext(filename)[0]
#             image_deleted = False
#
#             for ext in [".jpg", ".jpeg", ".png"]:
#                 image_path = os.path.join(images_dir, image_name + ext)
#                 if os.path.exists(image_path):
#                     os.remove(image_path)
#                     image_deleted = True
#                     break
#
#             if not image_deleted:
#                 not_found_images.append(image_name)
#
#     print(f"✅ Deleted {deleted_count} label file(s) containing class ID '{class_id_to_find}'.")
#     if not_found_images:
#         print("\n⚠️ Some corresponding images were not found:")
#         for name in not_found_images:
#             print(f"  - {name}")
#
#
# # Example usage:
# if __name__ == "__main__":
#     CLASS_ID = 2
#     delete_files_with_class_id(CLASS_ID)


def count_labels_per_class(label_dir, num_classes=6):
    """
    Counts how many label files contain each YOLO class ID.

    Args:
        label_dir (str): Path to YOLO labels folder.
        num_classes (int): Total number of classes (default = 6).

    Returns:
        dict: {class_id: count_of_files_containing_it}
    """
    if not os.path.isdir(label_dir):
        print(f"❌ Directory not found: {label_dir}")
        return {}

    class_counts = defaultdict(int)
    total_files = 0

    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue

        total_files += 1
        filepath = os.path.join(label_dir, filename)

        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            # Collect all class IDs in this file
            file_class_ids = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    try:
                        cid = int(parts[0])
                        if 0 <= cid < num_classes:
                            file_class_ids.add(cid)
                    except ValueError:
                        continue

            # Increment count for each class found in this file
            for cid in file_class_ids:
                class_counts[cid] += 1

        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")

    # Print summary
    print(f"\n📊 Analyzed {total_files} label files in: {label_dir}\n")
    for cid in range(num_classes):
        count = class_counts.get(cid, 0)
        print(f"Class ID {cid}: {count} file(s) contain this class")

    return class_counts


# Example usage
if __name__ == "__main__":
    count_labels_per_class("../dataset/valid/labels", num_classes=6)
