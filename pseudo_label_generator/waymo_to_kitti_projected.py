import sys

import tensorflow.compat.v1 as tf
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from tqdm import tqdm
import os
import cv2
import numpy as np
import argparse

dataset_folder = '<path_to_waymo_dataset>'
#dataset_folder = '<path_to_waymo_dataset>'
training_folder = os.path.join(dataset_folder, 'training')
validation_folder = os.path.join(dataset_folder, 'validation')

#output_dir = '<path_to_output_dir>'
output_dir = '<path_to_output_dir>'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(os.path.join(output_dir, 'training')):
    os.makedirs(os.path.join(output_dir, 'training'))
if not os.path.exists(os.path.join(output_dir, 'validation')):
    os.makedirs(os.path.join(output_dir, 'validation'))

class Label_3D:
    def __init__(self, label):
        self.x = label.box.center_x
        self.y = label.box.center_y
        self.z = label.box.center_z

        self.width = label.box.width
        self.length = label.box.length
        self.height = label.box.height

        self.heading = label.box.heading

        self.id = label.id
        self.num_of_points = label.num_lidar_points_in_box

        self.left_u = -1
        self.left_v = -1
        self.right_u = -1
        self.right_v = -1

class Label_2D:
    def __init__(self, label):
        self.left_u = label.box.center_x - label.box.length / 2.
        self.left_v = label.box.center_y - label.box.width / 2.

        self.right_u = label.box.center_x + label.box.length / 2.
        self.right_v = label.box.center_y + label.box.width / 2.

        self.id = label.id

        self.x = None
        self.y = None
        self.z = None

        self.width = None
        self.length = None
        self.height = None

        self.heading = None
        self.level = None
        self.dist = None

        self.correspondence = None

def visu_labels(image, labels):
    for label in labels:
        if label.left_u == -1 or label.correspondence is None:
            continue
        cv2.rectangle(image, (int(label.left_u), int(label.left_v)), (int(label.right_u), int(label.right_v)), (0, 255, 0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Main script for waymo to kitti convertor')
    parser.add_argument('--seq_start', type=int, default=-1, help='Sequence start index default: -1')
    parser.add_argument('--seq_end', type=int, default=-1, help='Sequence end index default: -1')

    args = parser.parse_args(argv)

    return args

def write_label(f, label):
    f.write("Car -1 -1 -10 ")
    f.write(f"{float(label.left_u.item() if hasattr(label.left_u, 'item') else label.left_u):.2f} ")
    f.write(f"{float(label.left_v.item() if hasattr(label.left_v, 'item') else label.left_v):.2f} ")
    f.write(
        f"{float(label.right_u.item() if hasattr(label.right_u, 'item') else label.right_u):.2f} ")
    f.write(
        f"{float(label.right_v.item() if hasattr(label.right_v, 'item') else label.right_v):.2f} ")
    f.write(f"{float(label.width.item() if hasattr(label.width, 'item') else label.width):.2f} ")
    f.write(f"{float(label.height.item() if hasattr(label.height, 'item') else label.height):.2f} ")
    f.write(f"{float(label.length.item() if hasattr(label.length, 'item') else label.length):.2f} ")
    f.write(f"{float(label.x.item() if hasattr(label.x, 'item') else label.x):.2f} ")
    f.write(
        f"{float(label.y.item() + label.height.item() / 2. if (hasattr(label.y, 'item') and hasattr(label.height, 'item')) else label.y + label.height / 2.):.2f} ")
    f.write(f"{float(label.z.item() if hasattr(label.z, 'item') else label.z):.2f} ")
    f.write(
        f"{float(label.heading.item() if hasattr(label.heading, 'item') else label.heading):.2f}\n")

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    all_files_in_folder = os.listdir(training_folder)
    all_files_in_folder = sorted(all_files_in_folder)

    all_files_in_folder = all_files_in_folder[args.seq_start:args.seq_end]

    loop = tqdm(all_files_in_folder)
    for file_name in loop:
        loop.set_description(file_name)
        dataset = tf.data.TFRecordDataset(os.path.join(training_folder, file_name), compression_type='')
        if not os.path.exists(os.path.join(output_dir, 'training', file_name)):
            os.makedirs(os.path.join(output_dir, 'training', file_name))
            os.makedirs(os.path.join(output_dir, 'training', file_name, 'label_2'))
            os.makedirs(os.path.join(output_dir, 'training', file_name, 'image_2'))
            os.makedirs(os.path.join(output_dir, 'training', file_name, 'velodyne'))
            os.makedirs(os.path.join(output_dir, 'training', file_name, 'calib'))

        for i, data in enumerate(dataset):
            # if i != 31:
            #    continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            image_size = None
            # First, lets do the image
            for index, raw_image in enumerate(frame.images):
                if index == 0:
                    decoded_image = tf.image.decode_jpeg(raw_image.image)
                    image_size = decoded_image.shape[:2]
                    path_to_save = os.path.join(output_dir, 'training', file_name, 'image_2', str(i).zfill(10) + '.png')
                    cv2.imwrite(path_to_save, cv2.cvtColor(decoded_image.numpy(), cv2.COLOR_RGB2BGR))

            # Now also decode the lidar
            (range_images, camera_projections, _,
             range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections,
                                                                               range_image_top_pose)

            points_all = np.concatenate(points, axis=0)
            cp_points_all = np.concatenate(cp_points, axis=0)

            cp_points_all_concat = np.concatenate([points_all, cp_points_all[..., 0:3]], axis=-1)
            path_to_save_lidar = os.path.join(output_dir, 'training', file_name, 'velodyne', str(i).zfill(10) + '.npz')
            np.savez_compressed(path_to_save_lidar, np.float32(cp_points_all_concat))

            # Now lets do the calib
            calibration = [cc for cc in frame.context.camera_calibrations][0]
            extrinsic = list(calibration.extrinsic.transform)
            intrinsic = list(calibration.intrinsic)

            calib_matrix = np.eye(4)
            calib_matrix[0, 0] = intrinsic[0]
            calib_matrix[1, 1] = intrinsic[1]
            calib_matrix[0, 2] = intrinsic[2]
            calib_matrix[1, 2] = intrinsic[3]
            calib = calib_matrix.copy()
            calib_matrix = calib_matrix[:3, :4]
            calib_matrix = calib_matrix.flatten().tolist()
            P2 = [str(c) for c in calib_matrix]

            R0_rect = np.eye(3).flatten().tolist()

            R = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]
            ])

            Tr_velo_to_cam = np.zeros((4, 4))
            Tr_velo_to_cam[0, 1] = -1.
            Tr_velo_to_cam[1, 2] = -1.
            Tr_velo_to_cam[2, 0] = 1.
            Tr_velo_to_cam = Tr_velo_to_cam @ np.linalg.inv(np.array(extrinsic).reshape(4, 4))
            Tr_velo_to_cam = Tr_velo_to_cam[:3, :4].flatten().tolist()

            Tr_imu_to_velo = np.eye(4)[:3, :4].flatten().tolist()

            P2 = [str(x) for x in P2]
            R0_rect = [str(x) for x in R0_rect]
            Tr_velo_to_cam = [str(x) for x in Tr_velo_to_cam]
            Tr_imu_to_velo = [str(x) for x in Tr_imu_to_velo]

            #Now look also on trasformations
            T_w_imu_cur = np.array(frame.pose.transform).reshape((4, 4))
            T_w_imu_cur = T_w_imu_cur.flatten().tolist()
            cur_pose = [str(x) for x in T_w_imu_cur]

            with open(os.path.join(output_dir, 'training', file_name, 'calib', str(i).zfill(10) + '.txt'), 'w') as f:
                f.write('P0: ' + ' '.join(P2) + '\n')
                f.write('P1: ' + ' '.join(P2) + '\n')
                f.write('P2: ' + ' '.join(P2) + '\n')
                f.write('P3: ' + ' '.join(P2) + '\n')
                f.write('R0_rect: ' + ' '.join(R0_rect) + '\n')
                f.write('Tr_velo_to_cam: ' + ' '.join(Tr_velo_to_cam) + '\n')
                f.write('Tr_imu_to_velo: ' + ' '.join(Tr_imu_to_velo) + '\n')
                f.write('Cur_pose: ' + ' '.join(cur_pose) + '\n')

            # Now lets do labels
            all_3d_labels = []
            for label_3d in frame.laser_labels:
                # Only if it is a vehicle
                if label_3d.type == 1 and label_3d.num_lidar_points_in_box > 0:
                    cur_label = Label_3D(label_3d)
                    # print(cur_label.left_u, cur_label.left_v, cur_label.right_u, cur_label.right_v)
                    all_3d_labels.append(cur_label)

            # Now lets find 2D labels
            # print("2D labels_now")
            all_2d_labels = []
            for index, image_labels in enumerate(frame.projected_lidar_labels):
                if index == 0:
                    for image_label in image_labels.labels:
                        if image_label.type == 1:
                            cur_label = Label_2D(image_label)
                            # print(cur_label.left_u, cur_label.left_v, cur_label.right_u, cur_label.right_v)
                            all_2d_labels.append(cur_label)

            # Now lets create correspondence between 2D and 3D labels
            for index, label_2d in enumerate(all_2d_labels):
                cur_id = label_2d.id[:-6]
                for label_3d in all_3d_labels:
                    if label_3d.id == cur_id:
                        label_2d.x = label_3d.x
                        label_2d.y = label_3d.y
                        label_2d.z = label_3d.z
                        label_2d.width = label_3d.width
                        label_2d.length = label_3d.length
                        label_2d.height = label_3d.height
                        label_2d.heading = - label_3d.heading - np.pi / 2. #This is to fulfill the kitti format
                        label_2d.correspondence = label_3d.id

            for index_2d, label_2d in enumerate(all_2d_labels):
                if label_2d.correspondence is not None:
                    center = np.array([[label_2d.x, label_2d.y, label_2d.z, 1.]]).transpose()
                    center = np.linalg.inv(np.array(extrinsic).reshape(4, 4)) @ center
                    R = np.array([
                        [0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]
                    ])
                    center = R @ center[:3, :]
                    label_2d.x = center[0]
                    label_2d.y = center[1]
                    label_2d.z = center[2]

            # visu_labels(cv2.imread(path_to_save), all_2d_labels)

            with open(os.path.join(output_dir, 'training', file_name, 'label_2', str(i).zfill(10) + '.txt'), 'w') as f:
                for label in all_2d_labels:
                    if label.correspondence is not None:
                        write_label(f, label)

    # Second lets do the validation
    all_files_in_folder = os.listdir(validation_folder)
    all_files_in_folder = sorted(all_files_in_folder)

    all_files_in_folder = all_files_in_folder[args.seq_start:args.seq_end]

    loop = tqdm(all_files_in_folder)
    for file_name in loop:
        loop.set_description(file_name)
        dataset = tf.data.TFRecordDataset(os.path.join(validation_folder, file_name), compression_type='')
        if not os.path.exists(os.path.join(output_dir, 'validation', file_name)):
            os.makedirs(os.path.join(output_dir, 'validation', file_name))
            os.makedirs(os.path.join(output_dir, 'validation', file_name, 'label_2'))
            os.makedirs(os.path.join(output_dir, 'validation', file_name, 'image_2'))
            os.makedirs(os.path.join(output_dir, 'validation', file_name, 'velodyne'))
            os.makedirs(os.path.join(output_dir, 'validation', file_name, 'calib'))

            os.makedirs(os.path.join(output_dir, 'validation', file_name, 'label_l1'))

            os.makedirs(os.path.join(output_dir, 'validation', file_name, 'label_l1_030'))
            os.makedirs(os.path.join(output_dir, 'validation', file_name, 'label_l1_3050'))
            os.makedirs(os.path.join(output_dir, 'validation', file_name, 'label_l1_50xx'))
            os.makedirs(os.path.join(output_dir, 'validation', file_name, 'label_l2_030'))
            os.makedirs(os.path.join(output_dir, 'validation', file_name, 'label_l2_3050'))
            os.makedirs(os.path.join(output_dir, 'validation', file_name, 'label_l2_50xx'))

        for i, data in enumerate(dataset):
            # if i != 31:
            #    continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            image_size = None
            # First, lets do the image
            for index, raw_image in enumerate(frame.images):
                if index == 0:
                    decoded_image = tf.image.decode_jpeg(raw_image.image)
                    image_size = decoded_image.shape[:2]
                    path_to_save = os.path.join(output_dir, 'validation', file_name, 'image_2',
                                                str(i).zfill(10) + '.png')
                    cv2.imwrite(path_to_save, cv2.cvtColor(decoded_image.numpy(), cv2.COLOR_RGB2BGR))

            # Now lets do the calib
            calibration = [cc for cc in frame.context.camera_calibrations][0]
            extrinsic = list(calibration.extrinsic.transform)
            intrinsic = list(calibration.intrinsic)

            calib_matrix = np.eye(4)
            calib_matrix[0, 0] = intrinsic[0]
            calib_matrix[1, 1] = intrinsic[1]
            calib_matrix[0, 2] = intrinsic[2]
            calib_matrix[1, 2] = intrinsic[3]
            calib = calib_matrix.copy()
            calib_matrix = calib_matrix[:3, :4]
            calib_matrix = calib_matrix.flatten().tolist()
            P2 = [str(c) for c in calib_matrix]

            R0_rect = np.eye(3).flatten().tolist()

            R = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]
            ])

            Tr_velo_to_cam = np.zeros((4, 4))
            Tr_velo_to_cam[0, 1] = -1.
            Tr_velo_to_cam[1, 2] = -1.
            Tr_velo_to_cam[2, 0] = 1.
            Tr_velo_to_cam = Tr_velo_to_cam @ np.linalg.inv(np.array(extrinsic).reshape(4, 4))
            Tr_velo_to_cam = Tr_velo_to_cam[:3, :4].flatten().tolist()

            Tr_imu_to_velo = np.eye(4)[:3, :4].flatten().tolist()

            P2 = [str(x) for x in P2]
            R0_rect = [str(x) for x in R0_rect]
            Tr_velo_to_cam = [str(x) for x in Tr_velo_to_cam]
            Tr_imu_to_velo = [str(x) for x in Tr_imu_to_velo]

            with open(os.path.join(output_dir, 'validation', file_name, 'calib', str(i).zfill(10) + '.txt'), 'w') as f:
                f.write('P0: ' + ' '.join(P2) + '\n')
                f.write('P1: ' + ' '.join(P2) + '\n')
                f.write('P2: ' + ' '.join(P2) + '\n')
                f.write('P3: ' + ' '.join(P2) + '\n')
                f.write('R0_rect: ' + ' '.join(R0_rect) + '\n')
                f.write('Tr_velo_to_cam: ' + ' '.join(Tr_velo_to_cam) + '\n')
                f.write('Tr_imu_to_velo: ' + ' '.join(Tr_imu_to_velo) + '\n')

            # Now lets do labels
            all_3d_labels = []
            for label_3d in frame.laser_labels:
                # Only if it is a vehicle
                if label_3d.type == 1 and label_3d.num_lidar_points_in_box > 0:
                    cur_label = Label_3D(label_3d)
                    # print(cur_label.left_u, cur_label.left_v, cur_label.right_u, cur_label.right_v)
                    all_3d_labels.append(cur_label)

            # Now lets find 2D labels
            # print("2D labels_now")
            all_2d_labels = []
            for index, image_labels in enumerate(frame.projected_lidar_labels):
                if index == 0:
                    for image_label in image_labels.labels:
                        if image_label.type == 1:
                            cur_label = Label_2D(image_label)
                            # print(cur_label.left_u, cur_label.left_v, cur_label.right_u, cur_label.right_v)
                            all_2d_labels.append(cur_label)

            # Now lets create correspondence between 2D and 3D labels
            for index, label_2d in enumerate(all_2d_labels):
                cur_id = label_2d.id.split('_')[0]
                for label_3d in all_3d_labels:
                    if label_3d.id == cur_id:
                        label_2d.x = label_3d.x
                        label_2d.y = label_3d.y
                        label_2d.z = label_3d.z
                        label_2d.width = label_3d.width
                        label_2d.length = label_3d.length
                        label_2d.height = label_3d.height
                        label_2d.heading = - label_3d.heading - np.pi / 2. #This is to fulfill the kitti format
                        label_2d.correspondence = label_3d.id
                        if label_3d.num_of_points < 5:
                            label_2d.level = 2
                        else:
                            label_2d.level = 1

            for index_2d, label_2d in enumerate(all_2d_labels):
                if label_2d.correspondence is not None:
                    center = np.array([[label_2d.x, label_2d.y, label_2d.z, 1.]]).transpose()
                    center = np.linalg.inv(np.array(extrinsic).reshape(4, 4)) @ center
                    R = np.array([
                        [0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]
                    ])
                    center = R @ center[:3, :]
                    label_2d.x = center[0]
                    label_2d.y = center[1]
                    label_2d.z = center[2]
                    label_2d.dist = np.sqrt(label_2d.x ** 2 + label_2d.y ** 2 + label_2d.z ** 2)

            # visu_labels(cv2.imread(path_to_save), all_2d_labels)

            with open(os.path.join(output_dir, 'validation', file_name, 'label_2', str(i).zfill(10) + '.txt'),
                      'w') as f:
                for label in all_2d_labels:
                    if label.correspondence is not None:
                        write_label(f, label)

            with open(os.path.join(output_dir, 'validation', file_name, 'label_l1', str(i).zfill(10) + '.txt'),
                      'w') as f:
                for label in all_2d_labels:
                    if label.correspondence is not None and label.level == 1:
                        write_label(f, label)

            with open(os.path.join(output_dir, 'validation', file_name, 'label_l1_030', str(i).zfill(10) + '.txt'),
                      'w') as f:
                for label in all_2d_labels:
                    if label.correspondence is not None and label.level == 1 and label.dist < 30.:
                        write_label(f, label)

            with open(os.path.join(output_dir, 'validation', file_name, 'label_l1_3050', str(i).zfill(10) + '.txt'),
                      'w') as f:
                for label in all_2d_labels:
                    if label.correspondence is not None and label.level == 1 and 30. <= label.dist < 50.:
                        write_label(f, label)

            with open(os.path.join(output_dir, 'validation', file_name, 'label_l1_50xx', str(i).zfill(10) + '.txt'),
                      'w') as f:
                for label in all_2d_labels:
                    if label.correspondence is not None and label.level == 1 and 50. <= label.dist:
                        write_label(f, label)

            with open(os.path.join(output_dir, 'validation', file_name, 'label_l2_030', str(i).zfill(10) + '.txt'),
                      'w') as f:
                for label in all_2d_labels:
                    if label.correspondence is not None and label.dist < 30.:
                        write_label(f, label)

            with open(os.path.join(output_dir, 'validation', file_name, 'label_l2_3050', str(i).zfill(10) + '.txt'),
                      'w') as f:
                for label in all_2d_labels:
                    if label.correspondence is not None and 30. <= label.dist < 50.:
                        write_label(f, label)

            with open(os.path.join(output_dir, 'validation', file_name, 'label_l2_50xx', str(i).zfill(10) + '.txt'),
                      'w') as f:
                for label in all_2d_labels:
                    if label.correspondence is not None and 50. <= label.dist:
                        write_label(f, label)


