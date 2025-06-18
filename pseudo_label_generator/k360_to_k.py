

import os
import glob
import shutil

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

if not os.path.exists(output_kitti_folder):
    os.makedirs(output_kitti_folder)
if not os.path.exists(os.path.join(output_kitti_folder, "training")):
    os.makedirs(os.path.join(output_kitti_folder, "training"))
if not os.path.exists(os.path.join(output_kitti_folder, "testing")):
    os.makedirs(os.path.join(output_kitti_folder, "testing"))
if not os.path.exists(os.path.join(output_kitti_folder, "ImageSets")):
    os.makedirs(os.path.join(output_kitti_folder, "ImageSets"))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'calib')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'calib'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'image_2')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'image_2'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'label_2')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_2'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'velodyne')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'velodyne'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'label_pseudo')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_pseudo'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'velodyne_pseudo')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'velodyne_pseudo'))
if not os.path.exists(os.path.join(output_kitti_folder, "testing", 'calib')):
    os.makedirs(os.path.join(output_kitti_folder, "testing", 'calib'))
if not os.path.exists(os.path.join(output_kitti_folder, "testing", 'image_2')):
    os.makedirs(os.path.join(output_kitti_folder, "testing", 'image_2'))
if not os.path.exists(os.path.join(output_kitti_folder, "testing", 'label_2')):
    os.makedirs(os.path.join(output_kitti_folder, "testing", 'label_2'))
if not os.path.exists(os.path.join(output_kitti_folder, "testing", 'velodyne')):
    os.makedirs(os.path.join(output_kitti_folder, "testing", 'velodyne'))
if not os.path.exists(os.path.join(output_kitti_folder, "testing", 'label_pseudo')):
    os.makedirs(os.path.join(output_kitti_folder, "testing", 'label_pseudo'))

cur_img_index = 0
for folder in sorted(os.listdir(data_folder)):
    if folder in training_sequences:
        cur_folder = os.path.join(data_folder, folder)
        for image in sorted(glob.glob(os.path.join(cur_folder, "image_00/data_rect/", "*.png"))):
            img_number = os.path.basename(image).split(".")[0]
            cur_calib = os.path.join(cur_folder, "calib", str(img_number) + ".txt")
            cur_label = os.path.join(cur_folder, "label_00", str(img_number) + ".txt")
            cur_pseudo_label = os.path.join(data_folder, "label_pseudo", str(folder) + '_' + str(img_number) + ".txt")
            cur_velo = os.path.join(cur_folder, "velodyne_points/data", str(img_number) + ".bin")
            cur_velo_pseudo = os.path.join('/mnt/personal/skvrnjan/output/frames_k360_v2/lidar_raw', str(folder), "pcds", str(img_number) + ".npz")

            if not os.path.exists(cur_calib):
                continue
            elif not os.path.exists(cur_label):
                continue

            shutil.copy(image, os.path.join(output_kitti_folder, "training", "image_2", str(cur_img_index).zfill(6) + ".png"))
            shutil.copy(cur_calib, os.path.join(output_kitti_folder, "training", "calib", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_2", str(cur_img_index).zfill(6) + ".txt"))
            #shutil.copy(cur_velo, os.path.join(output_kitti_folder, "training", "velodyne", str(cur_img_index).zfill(6) + ".bin"))
            #shutil.copy(cur_velo_pseudo, os.path.join(output_kitti_folder, "training", "velodyne_pseudo", str(cur_img_index).zfill(6) + ".npz"))
            if os.path.exists(cur_pseudo_label):
                shutil.copy(cur_pseudo_label, os.path.join(output_kitti_folder, "training", "label_pseudo", str(cur_img_index).zfill(6) + ".txt"))
            else:
                print("Pseudo label not found for", folder, img_number)
                with open(os.path.join(output_kitti_folder, "training", "label_pseudo", str(cur_img_index).zfill(6) + ".txt"), 'w') as f:
                    f.write("")
            cur_img_index += 1

num_of_training = cur_img_index
with open(os.path.join(output_kitti_folder, "ImageSets", "train.txt"), 'w') as f:
    for i in range(cur_img_index):
        f.write(str(i).zfill(6) + "\n")

for folder in sorted(os.listdir(data_folder)):
    if folder in validation_sequences:
        cur_folder = os.path.join(data_folder, folder)
        for image in sorted(glob.glob(os.path.join(cur_folder, "image_00/data_rect/", "*.png"))):
            img_number = os.path.basename(image).split(".")[0]
            cur_calib = os.path.join(cur_folder, "calib", str(img_number) + ".txt")
            cur_label = os.path.join(cur_folder, "label_00", str(img_number) + ".txt")
            #cur_pseudo_label = os.path.join(data_folder, "label_pseudo", str(folder) + '_' + str(img_number) + ".txt")
            cur_velo = os.path.join(cur_folder, "velodyne_points/data", str(img_number) + ".bin")

            if not os.path.exists(cur_calib):
                continue
            elif not os.path.exists(cur_label):
                continue

            shutil.copy(image, os.path.join(output_kitti_folder, "training", "image_2", str(cur_img_index).zfill(6) + ".png"))
            shutil.copy(cur_calib, os.path.join(output_kitti_folder, "training", "calib", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_2", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_pseudo", str(cur_img_index).zfill(6) + ".txt"))
            #shutil.copy(cur_velo, os.path.join(output_kitti_folder, "training", "velodyne", str(cur_img_index).zfill(6) + ".bin"))
            #if os.path.exists(cur_pseudo_label):
            #    shutil.copy(cur_pseudo_label, os.path.join(output_kitti_folder, "training", "label_pseudo", str(cur_img_index).zfill(6) + ".txt"))
            #else:
            #    print("Pseudo label not found for", folder, img_number)
            #    with open(os.path.join(output_kitti_folder, "training", "label_pseudo", str(cur_img_index).zfill(6) + ".txt"), 'w') as f:
            #        f.write("")
            cur_img_index += 1

num_of_validation = cur_img_index - num_of_training
with open(os.path.join(output_kitti_folder, "ImageSets", "val.txt"), 'w') as f:
    for i in range(num_of_training, cur_img_index):
        f.write(str(i).zfill(6) + "\n")

cur_img_index = 0
for folder in sorted(os.listdir(data_folder)):
    if folder in testing_sequences:
        cur_folder = os.path.join(data_folder, folder)
        for image in sorted(glob.glob(os.path.join(cur_folder, "image_00/data_rect/", "*.png"))):
            img_number = os.path.basename(image).split(".")[0]
            cur_calib = os.path.join(cur_folder, "calib", str(img_number) + ".txt")
            cur_label = os.path.join(cur_folder, "label_00", str(img_number) + ".txt")
            cur_pseudo_label = os.path.join(data_folder, "label_pseudo", str(folder) + '_' + str(img_number) + ".txt")
            cur_velo = os.path.join(cur_folder, "velodyne_points/data", str(img_number) + ".bin")

            if not os.path.exists(cur_calib):
                continue
            elif not os.path.exists(cur_label):
                continue

            shutil.copy(image, os.path.join(output_kitti_folder, "testing", "image_2", str(cur_img_index).zfill(6) + ".png"))
            shutil.copy(cur_calib, os.path.join(output_kitti_folder, "testing", "calib", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label, os.path.join(output_kitti_folder, "testing", "label_2", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label, os.path.join(output_kitti_folder, "testing", "label_pseudo", str(cur_img_index).zfill(6) + ".txt"))
            #shutil.copy(cur_velo, os.path.join(output_kitti_folder, "testing", "velodyne", str(cur_img_index).zfill(6) + ".bin"))
            #if os.path.exists(cur_pseudo_label):
                #shutil.copy(cur_pseudo_label, os.path.join(output_kitti_folder, "testing", "label_pseudo", str(cur_img_index).zfill(6) + ".txt"))
            #else:
            #    print("Pseudo label not found for", folder, img_number)
            #    with open(os.path.join(output_kitti_folder, "testing", "label_pseudo", str(cur_img_index).zfill(6) + ".txt"), 'w') as f:
            #        f.write("")
            cur_img_index += 1

with open(os.path.join(output_kitti_folder, "ImageSets", "test.txt"), 'w') as f:
    for i in range(cur_img_index):
        f.write(str(i).zfill(6) + "\n")




