import shutil, os

random_indexes = []

#Read the file which contains the ordering of the scenes
with open('<KITTI_DATASET_PATH>/object_detection/devkit_object/mapping/train_rand.txt', 'r') as f:
    line = f.readline().strip()

    random_indexes = line.split(',')

mapping_data = []

with open('<KITTI_DATASET_PATH>/object_detection/devkit_object/mapping/train_mapping.txt', 'r') as f:
    for line in f:
        mapping_data.append(line.strip().split(' '))

index = 0
for rnd_idx in random_indexes:
    map_data_cur = mapping_data[int(rnd_idx)]
    path_to_folder = '<KITTI_COMPLETE_SEQUENCES_PATH>/' + map_data_cur[0] + '/' + map_data_cur[1] + '/'

    if not os.path.exists('<CUSTOM_KITTI_OUTPUT_PATH>/image_2_add/' + f'{index:0>6}'):
        f_timestamp = open(path_to_folder + 'oxts/' + 'timestamps.txt', 'r')
        timestamps = []
        for line in f_timestamp:
            timestamps.append(line.strip().split(' '))

        f_time_out = open('<CUSTOM_KITTI_OUTPUT_PATH>/timestamps/' + str(index) + '.txt', 'w')

        os.mkdir('<CUSTOM_KITTI_OUTPUT_PATH>/image_2_add/' + f'{index:0>6}')
        os.mkdir('<CUSTOM_KITTI_OUTPUT_PATH>/velodyne_add/' + f'{index:0>6}')
        os.mkdir('<CUSTOM_KITTI_OUTPUT_PATH>/odx_add/' + f'{index:0>6}')

        file_number = int(map_data_cur[2])
        for i in range(-30,30):
            #First copy the img
            path_to_img = path_to_folder + 'image_02/data/' + f'{file_number+i:0>10}' + '.png'
            path_to_velo = path_to_folder + 'velodyne_points/data/' + f'{file_number+i:0>10}' + '.bin'
            path_to_odo = path_to_folder + 'oxts/data/' + f'{file_number+i:0>10}' + '.txt'

            path_save_img = '<CUSTOM_KITTI_OUTPUT_PATH>/image_2_add/' + f'{index:0>6}' + '/' + str(i) + '.png'
            path_save_velo = '<CUSTOM_KITTI_OUTPUT_PATH>/velodyne_add/' + f'{index:0>6}' + '/' + str(i) + '.bin'
            path_save_odo = '<CUSTOM_KITTI_OUTPUT_PATH>/odx_add/' + f'{index:0>6}' + '/' + str(i) + '.txt'

            if os.path.isfile(path_to_img) and os.path.isfile(path_to_velo) and os.path.isfile(path_to_odo):
                shutil.copy(path_to_img, path_save_img)
                shutil.copy(path_to_velo, path_save_velo)
                shutil.copy(path_to_odo, path_save_odo)

                f_time_out.write(str(i) + " " + str(timestamps[file_number+i][1]) + "\n")

        f_timestamp.close()
        f_time_out.close()
    index += 1
