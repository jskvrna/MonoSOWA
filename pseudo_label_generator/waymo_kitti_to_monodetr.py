import os
import glob
import shutil

data_folder = "/mnt/personal/skvrnjan/waymo_to_kitti/"

output_kitti_folder = "/mnt/personal/skvrnjan/waymo_kitti_monodetr/"

if not os.path.exists(output_kitti_folder):
    os.makedirs(output_kitti_folder)
if not os.path.exists(os.path.join(output_kitti_folder, "training")):
    os.makedirs(os.path.join(output_kitti_folder, "training"))
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
if not os.path.exists(os.path.join(output_kitti_folder, "testing", 'calib')):
    os.makedirs(os.path.join(output_kitti_folder, "testing", 'calib'))
if not os.path.exists(os.path.join(output_kitti_folder, "testing", 'image_2')):
    os.makedirs(os.path.join(output_kitti_folder, "testing", 'image_2'))
if not os.path.exists(os.path.join(output_kitti_folder, "testing", 'label_2')):
    os.makedirs(os.path.join(output_kitti_folder, "testing", 'label_2'))
if not os.path.exists(os.path.join(output_kitti_folder, "testing", 'velodyne')):
    os.makedirs(os.path.join(output_kitti_folder, "testing", 'velodyne'))

#First lets start with the training samples
with open(os.path.join(output_kitti_folder, "ImageSets", "mapping.txt"), 'w') as f:
    cur_img_index = 0
    for folder in sorted(os.listdir(os.path.join(data_folder, 'training'))):
        print(folder)
        cur_folder = os.path.join(data_folder, 'training', folder)
        for image in sorted(glob.glob(os.path.join(cur_folder, "image_2/", "*.png"))):
            img_number = os.path.basename(image).split(".")[0]
            cur_calib = os.path.join(cur_folder, "calib", str(img_number) + ".txt")
            cur_label = os.path.join(cur_folder, "label_2", str(img_number) + ".txt")

            if not os.path.exists(cur_calib):
                continue
            elif not os.path.exists(cur_label):
                continue

            shutil.copy(image, os.path.join(output_kitti_folder, "training", "image_2", str(cur_img_index).zfill(6) + ".png"))
            shutil.copy(cur_calib, os.path.join(output_kitti_folder, "training", "calib", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_2", str(cur_img_index).zfill(6) + ".txt"))
            f.write(str(cur_img_index).zfill(6) + " " + image + "\n")
            cur_img_index += 1

num_of_training = cur_img_index
with open(os.path.join(output_kitti_folder, "ImageSets", "train.txt"), 'w') as f:
    for i in range(cur_img_index):
        f.write(str(i).zfill(6) + "\n")

#Now lets do the validation samples
for folder in sorted(os.listdir(os.path.join(data_folder, 'validation'))):
    print(folder)
    cur_folder = os.path.join(data_folder, 'validation', folder)
    for image in sorted(glob.glob(os.path.join(cur_folder, "image_2", "*.png"))):
        img_number = os.path.basename(image).split(".")[0]
        cur_calib = os.path.join(cur_folder, "calib", str(img_number) + ".txt")
        cur_label = os.path.join(cur_folder, "label_2", str(img_number) + ".txt")

        if not os.path.exists(cur_calib):
            continue
        elif not os.path.exists(cur_label):
            continue

        shutil.copy(image, os.path.join(output_kitti_folder, "training", "image_2", str(cur_img_index).zfill(6) + ".png"))
        shutil.copy(cur_calib, os.path.join(output_kitti_folder, "training", "calib", str(cur_img_index).zfill(6) + ".txt"))
        shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_2", str(cur_img_index).zfill(6) + ".txt"))
        cur_img_index += 1

num_of_validation = cur_img_index - num_of_training
with open(os.path.join(output_kitti_folder, "ImageSets", "val.txt"), 'w') as f:
    for i in range(num_of_training, cur_img_index):
        f.write(str(i).zfill(6) + "\n")



