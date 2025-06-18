import os, sys
import shutil

# Check if the folder path argument is provided
if len(sys.argv) < 3:
    print("Please provide the folder path as following: Argument1 = path to openpcdet data kitti, Argument2 = path to pseudo ground truth labels.")
    sys.exit(1)

# Paths to folders
original_folder = sys.argv[1]
replacement_folder = sys.argv[2]
txt_file = original_folder + 'ImageSets/train.txt'

if os.path.exists(original_folder + "gt_database"):
    shutil.rmtree(original_folder + "gt_database")

files_to_remove = ['kitti_dbinfos_train.pkl', 'kitti_infos_test.pkl', 'kitti_infos_train.pkl', 'kitti_infos_trainval.pkl', 'kitti_infos_val.pkl']

for filename in files_to_remove:
    if os.path.exists(original_folder + filename):
        os.remove(original_folder + filename)

# Read the index file
with open(txt_file, 'r') as file:
    indexes = [line.strip() for line in file]

# Iterate over the files in the original folder
for filename in os.listdir(original_folder + '/training/label_2'):
    # Extract the index from the filename
    index = os.path.splitext(filename)[0]

    # Check if the index is in the list of indexes
    if index in indexes:
        # Construct the paths to the original and replacement files
        original_path = os.path.join(original_folder + '/training/label_2/', filename)
        replacement_path = os.path.join(replacement_folder, filename)

        # Replace the file
        shutil.copyfile(replacement_path, original_path)
        print(f"Replaced file: {original_path}")