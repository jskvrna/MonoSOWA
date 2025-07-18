import os
import glob
import shutil
from tqdm import tqdm

data_folder = "/mnt/personal/skvrnjan/KITTI/"
output_kitti_folder = "/mnt/personal/skvrnjan/test_kitti"

training_sequences = ["2013_05_28_drive_0000_sync",
                      "2013_05_28_drive_0002_sync",
                      "2013_05_28_drive_0004_sync",
                      "2013_05_28_drive_0005_sync",
                      "2013_05_28_drive_0006_sync",
                      "2013_05_28_drive_0009_sync"]

validation_sequences = ["2013_05_28_drive_0003_sync",
                        "2013_05_28_drive_0007_sync"]

testing_sequences = ["2013_05_28_drive_0010_sync"]

# Create directory structure
subdirs = {
    "training": ['calib', 'image_2', 'labels_gt', 'velodyne', 'labels_pseudo', 'velodyne_pseudo'],
    "testing": ['calib', 'image_2', 'labels_gt', 'velodyne', 'labels_pseudo'],
    "ImageSets": []
}

for top_dir, sub_list in subdirs.items():
    os.makedirs(os.path.join(output_kitti_folder, top_dir), exist_ok=True)
    for sub in sub_list:
        os.makedirs(os.path.join(output_kitti_folder, top_dir, sub), exist_ok=True)

# --- Collect all files to be processed ---
training_files = []
for folder in sorted(os.listdir(data_folder)):
    if folder in training_sequences:
        cur_folder = os.path.join(data_folder, folder)
        training_files.extend(sorted(glob.glob(os.path.join(cur_folder, "image_00/data_rect/", "*.png"))))

validation_files = []
for folder in sorted(os.listdir(data_folder)):
    if folder in validation_sequences:
        cur_folder = os.path.join(data_folder, folder)
        validation_files.extend(sorted(glob.glob(os.path.join(cur_folder, "image_00/data_rect/", "*.png"))))

testing_files = []
for folder in sorted(os.listdir(data_folder)):
    if folder in testing_sequences:
        cur_folder = os.path.join(data_folder, folder)
        testing_files.extend(sorted(glob.glob(os.path.join(cur_folder, "image_00/data_rect/", "*.png"))))

# --- Process Training Files ---
cur_img_index = 0
for image in tqdm(training_files, desc="Processing Training Set"):
    img_number = os.path.basename(image).split(".")[0]
    folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image))))
    cur_folder = os.path.join(data_folder, folder)

    cur_calib = os.path.join(cur_folder, "calib", str(img_number) + ".txt")
    cur_label = os.path.join(cur_folder, "label_00", str(img_number) + ".txt")
    cur_pseudo_label = os.path.join(data_folder, "label_pseudo", str(folder) + '_' + str(img_number) + ".txt")
    # cur_velo = os.path.join(cur_folder, "velodyne_points/data", str(img_number) + ".bin")
    # cur_velo_pseudo = os.path.join('/mnt/personal/skvrnjan/output/frames_k360_v2/lidar_raw', str(folder), "pcds", str(img_number) + ".npz")

    if not os.path.exists(cur_calib) or not os.path.exists(cur_label):
        continue

    shutil.copy(image, os.path.join(output_kitti_folder, "training", "image_2", str(cur_img_index).zfill(6) + ".png"))
    shutil.copy(cur_calib, os.path.join(output_kitti_folder, "training", "calib", str(cur_img_index).zfill(6) + ".txt"))
    shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "labels_gt", str(cur_img_index).zfill(6) + ".txt"))
    # shutil.copy(cur_velo, os.path.join(output_kitti_folder, "training", "velodyne", str(cur_img_index).zfill(6) + ".bin"))
    # shutil.copy(cur_velo_pseudo, os.path.join(output_kitti_folder, "training", "velodyne_pseudo", str(cur_img_index).zfill(6) + ".npz"))
    if os.path.exists(cur_pseudo_label):
        shutil.copy(cur_pseudo_label, os.path.join(output_kitti_folder, "training", "labels_pseudo", str(cur_img_index).zfill(6) + ".txt"))
    else:
        print(f"Pseudo label not found for {folder} {img_number}")
        with open(os.path.join(output_kitti_folder, "training", "labels_pseudo", str(cur_img_index).zfill(6) + ".txt"), 'w') as f:
            f.write("")
    cur_img_index += 1

num_of_training = cur_img_index
with open(os.path.join(output_kitti_folder, "ImageSets", "train.txt"), 'w') as f:
    for i in range(cur_img_index):
        f.write(str(i).zfill(6) + "\n")

# --- Process Validation Files ---
for image in tqdm(validation_files, desc="Processing Validation Set"):
    img_number = os.path.basename(image).split(".")[0]
    folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image))))
    cur_folder = os.path.join(data_folder, folder)

    cur_calib = os.path.join(cur_folder, "calib", str(img_number) + ".txt")
    cur_label = os.path.join(cur_folder, "label_00", str(img_number) + ".txt")
    # cur_velo = os.path.join(cur_folder, "velodyne_points/data", str(img_number) + ".bin")

    if not os.path.exists(cur_calib) or not os.path.exists(cur_label):
        continue

    shutil.copy(image, os.path.join(output_kitti_folder, "training", "image_2", str(cur_img_index).zfill(6) + ".png"))
    shutil.copy(cur_calib, os.path.join(output_kitti_folder, "training", "calib", str(cur_img_index).zfill(6) + ".txt"))
    shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "labels_gt", str(cur_img_index).zfill(6) + ".txt"))
    shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "labels_pseudo", str(cur_img_index).zfill(6) + ".txt"))
    # shutil.copy(cur_velo, os.path.join(output_kitti_folder, "training", "velodyne", str(cur_img_index).zfill(6) + ".bin"))
    cur_img_index += 1

with open(os.path.join(output_kitti_folder, "ImageSets", "val.txt"), 'w') as f:
    for i in range(num_of_training, cur_img_index):
        f.write(str(i).zfill(6) + "\n")

# --- Process Testing Files ---
cur_img_index = 0
for image in tqdm(testing_files, desc="Processing Testing Set"):
    img_number = os.path.basename(image).split(".")[0]
    folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image))))
    cur_folder = os.path.join(data_folder, folder)

    cur_calib = os.path.join(cur_folder, "calib", str(img_number) + ".txt")
    cur_label = os.path.join(cur_folder, "label_00", str(img_number) + ".txt")
    # cur_velo = os.path.join(cur_folder, "velodyne_points/data", str(img_number) + ".bin")

    if not os.path.exists(cur_calib) or not os.path.exists(cur_label):
        continue

    shutil.copy(image, os.path.join(output_kitti_folder, "testing", "image_2", str(cur_img_index).zfill(6) + ".png"))
    shutil.copy(cur_calib, os.path.join(output_kitti_folder, "testing", "calib", str(cur_img_index).zfill(6) + ".txt"))
    shutil.copy(cur_label, os.path.join(output_kitti_folder, "testing", "labels_gt", str(cur_img_index).zfill(6) + ".txt"))
    shutil.copy(cur_label, os.path.join(output_kitti_folder, "testing", "labels_pseudo", str(cur_img_index).zfill(6) + ".txt"))
    # shutil.copy(cur_velo, os.path.join(output_kitti_folder, "testing", "velodyne", str(cur_img_index).zfill(6) + ".bin"))
    cur_img_index += 1

with open(os.path.join(output_kitti_folder, "ImageSets", "test.txt"), 'w') as f:
    for i in range(cur_img_index):
        f.write(str(i).zfill(6) + "\n")
