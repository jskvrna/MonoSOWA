import time

import zstd

from anno_V3 import AutoLabel3D

import torch
import glob
import os
from tqdm import tqdm
import cv2
import numpy as np
import dill
import open3d as o3d
import random

class Metric3D(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)
        self.metric3d_model = None
        if self.generate_raw_lidar:
            # NOTE: It is assumed that the config 'args' contains a 'path_templates' attribute
            # with placeholders for dataset-specific path components.
            self.metric3d_model = torch.hub.load('yvanyin/metric3d', self.cfg.metric3d.model, pretrain=True)

            if self.cfg.general.device == 'gpu':
                self.metric3d_model.eval().cuda()
            else:
                self.metric3d_model.eval()

    def shuffle_with_seed(self, folders):
        # Generate a unique seed using the current time and process ID
        unique_seed = int(time.time() * 1000) + os.getpid()
        random.seed(unique_seed)
        random.shuffle(folders)
        print(f"Using seed: {unique_seed}")
        return folders

    def generate_pseudo_lidar(self, dataset='kitti'):
        # Input: img: [H, W, 3] (RGB image), intrinsic: [4] (fx, fy, cx, cy)
        # Output: pseudo_lidar: [H * W, 3] (XYZ)

        for folder in self.shuffle_with_seed(os.listdir(self.cfg.paths.all_dataset_path)):
            if not os.path.isdir(os.path.join(self.cfg.paths.all_dataset_path, folder)):
                continue
            if not self.cfg.general.supress_debug_prints:
                print("Current folder for pseudo_lidar: ", folder)
            output = os.path.join(self.cfg.paths.merged_frames_path, "lidar_raw/")
            if not os.path.exists(os.path.join(output, folder)):
                os.makedirs(os.path.join(output, folder))

            # load calibration
            if dataset == 'kitti':
                calibration_file_path = os.path.join(self.cfg.paths.all_dataset_path, folder, 'calib_cam_to_cam.txt')
                P2 = self.load_calibration(calibration_file_path)
            else:
                calibration_file_path = os.path.join(self.cfg.paths.all_dataset_path, 'calibration', 'perspective.txt')
                P2 = self.load_calibration(calibration_file_path)

            for subfolder in self.shuffle_with_seed(os.listdir(os.path.join(self.cfg.paths.all_dataset_path, folder))):
                if not os.path.isdir(os.path.join(self.cfg.paths.all_dataset_path, folder, subfolder)):
                    continue
                if not self.cfg.general.supress_debug_prints:
                    print("Current subfolder for pseudo_lidar: ", subfolder)
                tmp_folder_path = os.path.join(self.cfg.paths.all_dataset_path, folder, subfolder)
                path_to_imgs = os.path.join(tmp_folder_path, 'image_02/data/')
                image_paths = self.shuffle_with_seed(sorted(glob.glob(os.path.join(path_to_imgs, '*.png'))))

                output_path = os.path.join(output, folder, subfolder)

                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                if self.cfg.metric3d.save_also_imgs:
                    if not os.path.exists(os.path.join(output_path, 'pngs')):
                        os.makedirs(os.path.join(output_path, 'pngs'))
                if not os.path.exists(os.path.join(output_path, 'pcds')):
                    os.makedirs(os.path.join(output_path, 'pcds'))

                for image_path in tqdm(image_paths, desc="Processing Images"):
                    file_name = os.path.basename(image_path)
                    file_name = file_name.split('.')[0]
                    if os.path.exists(os.path.join(output_path, 'pcds', file_name + '.npz')):
                        continue

                    img = cv2.imread(image_path)[:, :, ::-1]
                    intrinsic = [P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]]

                    pseudo_lidar, rgbd = self.compute_pseudo_lidar(img, intrinsic)
                    if self.cfg.metric3d.save_also_imgs:
                        #rgbd = rgbd * 500
                        #rgbd = np.clip(rgbd, 0, 65535)
                        #rgbd = rgbd.astype(np.uint16)
                        #cv2.imwrite(os.path.join(output_path, 'pngs', file_name + '.png'), rgbd)
                        np.savez_compressed(os.path.join(output_path, 'pngs', file_name + '.npz'), array1=rgbd)


                    np.savez_compressed(os.path.join(output_path, 'pcds', file_name + '.npz'), array1=pseudo_lidar)

        return None

    def generate_pseudo_lidar_all(self, dataset='kitti'):
        # Input: img: [H, W, 3] (RGB image), intrinsic: [4] (fx, fy, cx, cy)
        # Output: pseudo_lidar: [H * W, 3] (XYZ)

        all_times = []

        for folder in self.shuffle_with_seed(os.listdir(self.cfg.paths.all_dataset_path)):
            if not os.path.isdir(os.path.join(self.cfg.paths.all_dataset_path, folder)):
                continue
            if not self.cfg.general.supress_debug_prints:
                print("Current folder for pseudo_lidar: ", folder)
            output = os.path.join(self.cfg.paths.merged_frames_path, "lidar_raw/")
            if not os.path.exists(os.path.join(output, folder)):
                os.makedirs(os.path.join(output, folder))

            # load calibration
            if dataset == 'kitti':
                calibration_file_path = os.path.join(self.cfg.paths.all_dataset_path, folder, 'calib_cam_to_cam.txt')
                P2 = self.load_calibration(calibration_file_path)
            else:
                calibration_file_path = os.path.join(self.cfg.paths.all_dataset_path, 'calibration', 'perspective.txt')
                P2 = self.load_calibration_all(calibration_file_path)

            if not self.cfg.general.supress_debug_prints:
                print("Current subfolder for pseudo_lidar: ", folder)
            tmp_folder_path = os.path.join(self.cfg.paths.all_dataset_path, folder)
            path_to_imgs = os.path.join(tmp_folder_path, 'image_00/data_rect/')
            image_paths = self.shuffle_with_seed(sorted(glob.glob(os.path.join(path_to_imgs, '*.png'))))

            output_path = os.path.join(output, folder)

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if self.cfg.metric3d.save_also_imgs:
                if not os.path.exists(os.path.join(output_path, 'pngs')):
                    os.makedirs(os.path.join(output_path, 'pngs'))
            if not os.path.exists(os.path.join(output_path, 'pcds')):
                os.makedirs(os.path.join(output_path, 'pcds'))

            for image_path in tqdm(image_paths, desc="Processing Images"):
                start = time.time_ns()
                file_name = os.path.basename(image_path)
                file_name = file_name.split('.')[0]
                if os.path.exists(os.path.join(output_path, 'pcds', file_name + '.npz')):
                    continue

                img = cv2.imread(image_path)[:, :, ::-1]
                intrinsic = [P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]]

                pseudo_lidar, rgbd = self.compute_pseudo_lidar(img, intrinsic)

                if not self.cfg.general.supress_debug_prints:
                    time_taken = (time.time_ns() - start) / 1e9
                    all_times.append(time_taken)
                    print('Mean: ', np.mean(all_times), 'Var: ', np.var(all_times), 'Total_frames: ', len(all_times))

                if self.cfg.metric3d.save_also_imgs:
                    cv2.imwrite(os.path.join(output_path, 'pngs', file_name + '.png'), rgbd)
                #o3d_pcd = o3d.geometry.PointCloud()
                #o3d_pcd.points = o3d.utility.Vector3dVector(pseudo_lidar)
                #o3d.io.write_point_cloud(os.path.join(output_path, 'pcds', file_name + '.pcd'), o3d_pcd)
                np.savez_compressed(os.path.join(output_path, 'pcds', file_name + '.npz'), array1=pseudo_lidar)
                #show the point cloud
                #o3d.visualization.draw_geometries([o3d_pcd])

        return None

    def generate_pseudo_lidar_waymoc(self, seq_start=-1, seq_end=-1):
        # Input: img: [H, W, 3] (RGB image), intrinsic: [4] (fx, fy, cx, cy)
        # Output: pseudo_lidar: [H * W, 3] (XYZ)
        max_points = 500000
        training_path = os.path.join(self.cfg.paths.all_dataset_path, "training")
        all_folders = os.listdir(training_path)
        all_folders = sorted(all_folders)
        all_folders = all_folders[seq_start:seq_end]

        for folder in all_folders:
            if not os.path.isdir(os.path.join(training_path, folder)):
                continue
            if not self.cfg.general.supress_debug_prints:
                print("Current folder for pseudo_lidar: ", folder)
            output = os.path.join(self.cfg.paths.merged_frames_path, "lidar_raw/")
            if not os.path.exists(os.path.join(output, folder)):
                os.makedirs(os.path.join(output, folder))

            if not self.cfg.general.supress_debug_prints:
                print("Current subfolder for pseudo_lidar: ", folder)
            tmp_folder_path = os.path.join(training_path, folder)
            path_to_imgs = os.path.join(tmp_folder_path, 'image_2/')
            image_paths = sorted(glob.glob(os.path.join(path_to_imgs, '*.png')))

            output_path = os.path.join(output, folder)

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if self.cfg.metric3d.save_also_imgs:
                if not os.path.exists(os.path.join(output_path, 'pngs')):
                    os.makedirs(os.path.join(output_path, 'pngs'))
            if not os.path.exists(os.path.join(output_path, 'pcds')):
                os.makedirs(os.path.join(output_path, 'pcds'))

            for image_path in tqdm(image_paths, desc="Processing Images"):
                start = time.time_ns()
                file_name = os.path.basename(image_path)
                file_name = file_name.split('.')[0]
                if os.path.exists(os.path.join(output_path, 'pcds', file_name + '.npz')):
                    continue

                calib_path = os.path.join(tmp_folder_path, 'calib', file_name + '.txt')
                P2 = self.load_calibration(calib_path)

                img = cv2.imread(image_path)[:, :, ::-1]
                intrinsic = [P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]]

                pseudo_lidar, rgbd = self.compute_pseudo_lidar(img, intrinsic)
                pseudo_lidar = np.float32(pseudo_lidar)
                print('Time for pseudolidar_nosave: ', (time.time_ns() - start) / 1e9)
                if self.cfg.metric3d.save_also_imgs:
                    cv2.imwrite(os.path.join(output_path, 'pngs', file_name + '.png'), rgbd)

                #o3d_pcd = o3d.geometry.PointCloud()
                #o3d_pcd.points = o3d.utility.Vector3dVector(pseudo_lidar)
                #o3d.io.write_point_cloud(os.path.join(output_path, 'pcds', file_name + '.pcd'), o3d_pcd)
                N = pseudo_lidar.shape[0]
                if N > max_points:
                    indices = np.random.choice(N, max_points, replace=False)
                    pseudo_lidar = pseudo_lidar[indices]

                np.savez_compressed(os.path.join(output_path, 'pcds', file_name + '.npz'), array1=np.float32(pseudo_lidar))
                #show the point cloud
                #o3d.visualization.draw_geometries([o3d_pcd])
                print("Time for pseudo-lidar: ", (time.time_ns() - start) / 1e9)

        return None

    def generate_pseudo_lidar_kittibasic(self, dataset='kitti'):
        # Input: img: [H, W, 3] (RGB image), intrinsic: [4] (fx, fy, cx, cy)
        # Output: pseudo_lidar: [H * W, 3] (XYZ)
        data_path = os.path.join(self.cfg.paths.kitti_path, "object_detection", "training")
        output = os.path.join(self.cfg.paths.merged_frames_path, "depth/")
        if not os.path.exists(output):
            os.makedirs(output)

        path_to_imgs = os.path.join(data_path, 'image_2/')
        image_paths = sorted(glob.glob(os.path.join(path_to_imgs, '*.png')))
        print(output, image_paths)
        output_path = output

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for image_path in tqdm(image_paths, desc="Processing Images"):
            file_name = os.path.basename(image_path)
            file_name = file_name.split('.')[0]

            calibration_file_path = os.path.join(data_path, 'calib', file_name + '.txt')
            P2 = self.load_calibration(calibration_file_path)

            img = cv2.imread(image_path)[:, :, ::-1]
            intrinsic = [P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]]

            pseudo_lidar, rgbd = self.compute_pseudo_lidar(img, intrinsic)

            cv2.imwrite(os.path.join(output_path, file_name + '.png'), rgbd)
            #save rgbd with zstd
            #with open(os.path.join(output_path, file_name + '.zst'), 'wb') as f:
            #    # Serialize the array using dill
            #   serialized_data = dill.dumps(rgbd)
            #    # Compress the serialized data directly with zstd
            #    compressed = zstd.compress(serialized_data)
            #    f.write(compressed)

            #o3d_pcd = o3d.geometry.PointCloud()
            #o3d_pcd.points = o3d.utility.Vector3dVector(pseudo_lidar)
            #o3d.io.write_point_cloud(os.path.join(output_path, 'depth', file_name + '.pcd'), o3d_pcd)
            np.savez_compressed(os.path.join(output_path, file_name + '.npz'), array1=pseudo_lidar)
            #show the point cloud
            #o3d.visualization.draw_geometries([o3d_pcd])

    def compute_pseudo_lidar(self, img, intrinsic_input):
        img_encoded, padding, intrinsic_scaled = self.encode_img(img, intrinsic_input)

        with torch.no_grad():
            pred_depth, confidence, output_dict = self.metric3d_model.inference({'input': img_encoded})

        pcloud, rgbd = self.decode_img(pred_depth, padding, intrinsic_scaled, img, intrinsic_input)

        return pcloud, rgbd

    def encode_img(self, img, intrinsic_input):
        #### ajust input size to fit pretrained model
        # keep ratio resize
        input_size = (616, 1064)  # for vit model
        # input_size = (544, 1216) # for convnext model
        h, w = img.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        # remember to scale intrinsic, hold depth
        intrinsic = [intrinsic_input[0] * scale, intrinsic_input[1] * scale, intrinsic_input[2] * scale, intrinsic_input[3] * scale]
        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                 cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()

        return rgb, pad_info, intrinsic

    def decode_img(self, pred_depth, pad_info, intrinsic_input, img_orig, intrinsic_orig):
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1], pad_info[2]: pred_depth.shape[1] - pad_info[3]]

        # upsample to original size
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], img_orig.shape[:2],
                                                     mode='bilinear').squeeze()
        ###################### canonical camera space ######################

        #### de-canonical transform
        canonical_to_real_scale = intrinsic_input[0] / 1000.0  # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300).cpu()

        pred_depth_np = pred_depth.cpu().numpy()
        pred_depth_normalized = cv2.normalize(pred_depth_np, None, 0, 255, cv2.NORM_MINMAX)
        pred_depth_uint8 = pred_depth_normalized.astype(np.uint8)

        out_height, out_width = pred_depth.shape
        intrinsic = intrinsic_orig
        x, y = np.meshgrid(np.arange(out_width), np.arange(out_height))
        x = (x - intrinsic[2]) / intrinsic[0]
        y = (y - intrinsic[3]) / intrinsic[1]
        z = np.array(pred_depth)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)

        return points, pred_depth_np
    def load_calibration(self, calibration_file):
        with open(calibration_file, 'r') as f:
            lines = f.readlines()

        # Find the projection matrix
        # We look for "P2" and "P_rect_02" in the calibration file

        for line in lines:
            if 'P2' in line:
                P2 = np.array([float(value) for value in line.split()[1:]]).reshape(3, 4)
                return P2
            if 'P_rect_02' in line:
                P2 = np.array([float(value) for value in line.split()[1:]]).reshape(3, 4)
                return P2
        return None

    def load_calibration_all(self, calibration_file):
        with open(calibration_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'P_rect_00' in line:
                P2 = np.array([float(value) for value in line.split()[1:]]).reshape(3, 4)
                return P2
            if 'P_rect_02' in line:
                P2 = np.array([float(value) for value in line.split()[1:]]).reshape(3, 4)
                return P2
        return None

    def load_pseudo_lidar(self, img_path, intrinsic_path):
        img = cv2.imread(img_path)
        intrinsic = np.loadtxt(intrinsic_path)
        return img, intrinsic