import os
import random
import shutil

# Define the root directory of the dataset
dataset_root = "/data/hdd_1/DevDatasets/ROI/SipakMed/SipakMed"
# Define the list of subdirectories (classes)
classes = os.listdir(dataset_root)
# Define the split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Ensure that the sum of the split ratios is 1
assert train_ratio + val_ratio + test_ratio == 1

# Traverse each class directory
for class_name in classes:
    # Construct the directory path of the current class
    class_dir = os.path.join(dataset_root, class_name)
    # Get the paths of all files in the current class directory
    all_files = [os.path.join(class_dir, file) for file in os.listdir(class_dir)]
    # Shuffle the list of files randomly
    random.shuffle(all_files)
    num_files = len(all_files)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)

    # Split the file lists for training set, validation set and test set
    train_files = all_files[:num_train]
    val_files = all_files[num_train:num_train + num_val]
    test_files = all_files[num_train + num_val:]

    # Create the target folders if they don't exist
    train_dir = os.path.join(dataset_root, "train", class_name)
    val_dir = os.path.join(dataset_root, "val", class_name)
    test_dir = os.path.join(dataset_root, "test", class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move files to the corresponding folders
    for file in train_files:
        shutil.move(file, train_dir)
    for file in val_files:
        shutil.move(file, val_dir)
    for file in test_files:
        shutil.move(file, test_dir)