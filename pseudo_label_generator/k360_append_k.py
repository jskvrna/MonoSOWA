import os
import glob
import shutil

data_folder = "/mnt/personal/skvrnjan/KITTI/"

output_kitti_folder = "/mnt/personal/skvrnjan/kk360/"

training_sequences = ["2013_05_28_drive_0000_sync",
                      "2013_05_28_drive_0002_sync",
                      "2013_05_28_drive_0004_sync",
                      "2013_05_28_drive_0005_sync",
                      "2013_05_28_drive_0006_sync",
                      "2013_05_28_drive_0009_sync"]

validation_sequences = ["2013_05_28_drive_0003_sync",
                        "2013_05_28_drive_0007_sync"]

testing_sequences = ["2013_05_28_drive_0010_sync"]

cur_img_index = 10000
img_start = cur_img_index
for folder in sorted(os.listdir(data_folder)):
    if folder in training_sequences:
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

            shutil.copy(image, os.path.join(output_kitti_folder, "training", "image_2", str(cur_img_index).zfill(6) + ".png"))
            shutil.copy(cur_calib, os.path.join(output_kitti_folder, "training", "calib", str(cur_img_index).zfill(6) + ".txt"))
            #shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_2", str(cur_img_index).zfill(6) + ".txt"))
            if os.path.exists(cur_pseudo_label):
                shutil.copy(cur_pseudo_label, os.path.join(output_kitti_folder, "training", "label_2", str(cur_img_index).zfill(6) + ".txt"))
            else:
                print("Pseudo label not found for", folder, img_number)
                with open(os.path.join(output_kitti_folder, "training", "label_2", str(cur_img_index).zfill(6) + ".txt"), 'w') as f:
                    f.write("")
            shutil.copy(cur_velo, os.path.join(output_kitti_folder, "training", "velodyne", str(cur_img_index).zfill(6) + ".bin"))
            cur_img_index += 1

num_of_training = cur_img_index
with open(os.path.join(output_kitti_folder, "ImageSets", "train.txt"), 'a') as f:
    for i in range(img_start, cur_img_index):
        f.write(str(i).zfill(6) + "\n")




