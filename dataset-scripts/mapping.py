import os

class_mapping = {
    # old_id: new_id
    1 : 3
}

# Defining the path to the directory containing the label files
directories = [
    "dataset/Laptop/train/labels"
]

for labels_dir in directories:
    # Iterating over each file in the labels directory
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(labels_dir, filename)

            with open(filepath, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    if class_id in class_mapping:
                        parts[0] = str(class_mapping[class_id])
                        new_lines.append(" ".join(parts))
                    else:
                        new_lines.append(line.strip())
                else:
                    new_lines.append(line.strip())

            with open(filepath, 'w') as f:
                for line in new_lines:
                    f.write(line + '\n')

print("Class ID mapping completed.")


