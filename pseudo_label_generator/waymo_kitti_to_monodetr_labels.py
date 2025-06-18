import os
import glob
import shutil

data_folder = "/mnt/personal/skvrnjan/waymo_to_kitti/"

output_kitti_folder = "/mnt/personal/skvrnjan/waymo_kitti_monodetr_label/"

if not os.path.exists(output_kitti_folder):
    os.makedirs(output_kitti_folder)
if not os.path.exists(os.path.join(output_kitti_folder, "training")):
    os.makedirs(os.path.join(output_kitti_folder, "training"))
if not os.path.exists(os.path.join(output_kitti_folder, "ImageSets")):
    os.makedirs(os.path.join(output_kitti_folder, "ImageSets"))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'label_2')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_2'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'label_l1')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_l1'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'label_l1_030')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_l1_030'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'label_l1_3050')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_l1_3050'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'label_l1_50xx')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_l1_50xx'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'label_l2_030')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_l2_030'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'label_l2_3050')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_l2_3050'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'label_l2_50xx')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_l2_50xx'))
if not os.path.exists(os.path.join(output_kitti_folder, "training", 'label_pseudo')):
    os.makedirs(os.path.join(output_kitti_folder, "training", 'label_pseudo'))


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
            cur_pseudo_label = os.path.join(data_folder, "training", "label_pseudo", str(folder) + '_' + str(img_number) + ".txt")

            if not os.path.exists(cur_calib):
                continue
            elif not os.path.exists(cur_label):
                continue

            shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_2", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label,os.path.join(output_kitti_folder, "training", "label_l1", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label,os.path.join(output_kitti_folder, "training", "label_l1_030", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label,os.path.join(output_kitti_folder, "training", "label_l1_3050", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label,os.path.join(output_kitti_folder, "training", "label_l1_50xx", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label,os.path.join(output_kitti_folder, "training", "label_l2_030", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label,os.path.join(output_kitti_folder, "training", "label_l2_3050", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_label,os.path.join(output_kitti_folder, "training", "label_l2_50xx", str(cur_img_index).zfill(6) + ".txt"))
            shutil.copy(cur_pseudo_label,os.path.join(output_kitti_folder, "training", "label_pseudo", str(cur_img_index).zfill(6) + ".txt"))

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
        cur_label_l1 = os.path.join(cur_folder, "label_l1", str(img_number) + ".txt")
        cur_label_l1_030 = os.path.join(cur_folder, "label_l1_030", str(img_number) + ".txt")
        cur_label_l1_3050 = os.path.join(cur_folder, "label_l1_3050", str(img_number) + ".txt")
        cur_label_l1_50xx = os.path.join(cur_folder, "label_l1_50xx", str(img_number) + ".txt")
        cur_label_l2_030 = os.path.join(cur_folder, "label_l2_030", str(img_number) + ".txt")
        cur_label_l2_3050 = os.path.join(cur_folder, "label_l2_3050", str(img_number) + ".txt")
        cur_label_l2_50xx = os.path.join(cur_folder, "label_l2_50xx", str(img_number) + ".txt")

        if not os.path.exists(cur_calib):
            continue
        elif not os.path.exists(cur_label):
            continue

        shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_2", str(cur_img_index).zfill(6) + ".txt"))
        shutil.copy(cur_label_l1, os.path.join(output_kitti_folder, "training", "label_l1", str(cur_img_index).zfill(6) + ".txt"))
        shutil.copy(cur_label_l1_030, os.path.join(output_kitti_folder, "training", "label_l1_030", str(cur_img_index).zfill(6) + ".txt"))
        shutil.copy(cur_label_l1_3050, os.path.join(output_kitti_folder, "training", "label_l1_3050", str(cur_img_index).zfill(6) + ".txt"))
        shutil.copy(cur_label_l1_50xx, os.path.join(output_kitti_folder, "training", "label_l1_50xx", str(cur_img_index).zfill(6) + ".txt"))
        shutil.copy(cur_label_l2_030, os.path.join(output_kitti_folder, "training", "label_l2_030", str(cur_img_index).zfill(6) + ".txt"))
        shutil.copy(cur_label_l2_3050, os.path.join(output_kitti_folder, "training", "label_l2_3050", str(cur_img_index).zfill(6) + ".txt"))
        shutil.copy(cur_label_l2_50xx, os.path.join(output_kitti_folder, "training", "label_l2_50xx", str(cur_img_index).zfill(6) + ".txt"))
        shutil.copy(cur_label, os.path.join(output_kitti_folder, "training", "label_pseudo", str(cur_img_index).zfill(6) + ".txt"))

        cur_img_index += 1

num_of_validation = cur_img_index - num_of_training
with open(os.path.join(output_kitti_folder, "ImageSets", "val.txt"), 'w') as f:
    for i in range(num_of_training, cur_img_index):
        f.write(str(i).zfill(6) + "\n")



