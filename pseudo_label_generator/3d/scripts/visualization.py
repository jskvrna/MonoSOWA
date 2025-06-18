import os
import cv2
from anno_V3 import AutoLabel3D
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
from matplotlib import cm
import random
from scipy.spatial import ConvexHull
from numpy import *
import warnings
import open3d
from utils2 import load_velo_scan, get_perfect_scale, load_pseudo_lidar, load_waymoc_scan

if AutoLabel3D.use_pcdet:
    import matplotlib.pyplot as plt

    import glob
    from pathlib import Path

    import open3d_vis_utils as V

    OPEN3D_FLAG = True

    import torch

    try:
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
    except:
        warnings.warn("OpenPCDet not found. Only needed in specific cases for visualization")

class Visualization(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)

    def visualize_3D(self, cars, pedestrians=None):
        if not self.cfg.visualization.show_real_lidar:
            lidar = load_pseudo_lidar(self.pseudo_lidar_folder + str(self.file_number).zfill(10) + '.npz')
        else:
            lidar = np.array(load_velo_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + self.file_name + '.bin'))
        self.prepare_scan(self.file_name, self.img, lidar, save=True, crop=True, visu=True)
        self.bboxes = []
        self.color = np.ones((self.lidar.shape[1], 3))
        for car_idx in range(len(cars)):
            if cars[car_idx].optimized:
                cars[car_idx] = self.create_bboxes_from_opt_values_tracker([cars[car_idx]], False, True)[0]
            self.visu3D_car(cars[car_idx], car_idx)
        if pedestrians is not None:
            for ped_idx in range(len(pedestrians)):
                if pedestrians[ped_idx].optimized:
                    pedestrians[ped_idx] = self.create_bboxes_from_opt_values_tracker([pedestrians[ped_idx]], False, True)[0]
                self.visu3D_car(pedestrians[ped_idx], ped_idx)

        if self.args.dataset == 'waymo':
            self.visu_ground_truth_waymo()
        if self.cfg.visualization.visu_labels_gt != 'None':
            self.draw_boxes_from_labels(self.cfg.visualization.visu_labels_gt, [0, 1, 0]) #Green
        if self.cfg.visualization.visu_labels1 != 'None':
            self.draw_boxes_from_labels(self.cfg.visualization.visu_labels1, [0, 0, 1]) #Blue
        if self.cfg.visualization.visu_labels2 != 'None':
            self.draw_boxes_from_labels(self.cfg.visualization.visu_labels2, [1, 1, 0]) #Yellow
        if self.cfg.visualization.visu_output_labels:
            self.draw_boxes_from_labels(self.cfg.paths.labels_path, [1, 0, 0]) #Red

        if self.cfg.visualization.show_img:
            #show self.img
            immg = np.moveaxis(self.img, 0, -1)
            cv2.imwrite('img.png', immg)

        self.draw_pcloud()

    def visualize_3D_all(self, cars):
        if not self.cfg.visualization.show_real_lidar:
            lidar = np.array(load_pseudo_lidar(self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(self.number) + '.npz'))
        else:
            lidar = np.array(load_velo_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + self.file_name + '.bin'))
        self.prepare_scan_all(self.file_name, self.img, lidar, save=True, crop=True, visu=True)
        self.bboxes = []
        self.color = np.ones((self.lidar.shape[1], 3))

        if False: #Show pcloud only
            pcloud = open3d.geometry.PointCloud()
            pcloud.points = open3d.utility.Vector3dVector(self.lidar.T[:, 0:3])
            pcloud.colors = open3d.utility.Vector3dVector(self.color)
            pcloud = pcloud.voxel_down_sample(voxel_size=0.05)
            visualizer = open3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.add_geometry(pcloud)
            # visualizer.get_render_option().point_size = 5  # Adjust the point size if necessary
            visualizer.get_render_option().background_color = np.asarray([0, 0, 0])  # Set background to black
            visualizer.get_view_control().set_front([0, -0.3, -0.5])
            visualizer.get_view_control().set_lookat([0, 0, 1])
            visualizer.get_view_control().set_zoom(0.05)
            visualizer.get_view_control().set_up([0, -1, 0])
            visualizer.get_view_control().camera_local_translate(5., 0., 8.)
            visualizer.run()
            visualizer.destroy_window()

        for car_idx in range(len(cars)):
            if cars[car_idx].optimized:
                cars[car_idx] = self.create_bboxes_from_opt_values_tracker([cars[car_idx]], False, True)[0]
            self.visu3D_car(cars[car_idx], car_idx)

        if self.args.dataset == 'waymo':
            self.visu_ground_truth_waymo()
        if self.cfg.visualization.visu_labels_gt != 'None':
            self.draw_boxes_from_labels_all(self.cfg.visualization.visu_labels_gt, [0, 1, 0]) #Green
        if self.cfg.visualization.visu_labels1 != 'None':
            self.draw_boxes_from_labels(self.cfg.visualization.visu_labels1, [0, 0, 1]) #Blue
        if self.cfg.visualization.visu_labels2 != 'None':
            self.draw_boxes_from_labels(self.cfg.visualization.visu_labels2, [1, 1, 0]) #Yellow
        if self.cfg.visualization.visu_output_labels:
            self.draw_boxes_from_labels(self.cfg.paths.labels_path, [0, 0, 1]) #Red
        self.draw_pcloud()

    def visualize_3D_waymoc(self, cars):
        if not self.cfg.visualization.show_real_lidar:
            lidar = np.array(load_pseudo_lidar(self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(self.number) + '.npz'))
        else:
            lidar = np.array(load_waymoc_scan(os.path.join(self.cfg.paths.all_dataset_path, 'training', self.folder, 'velodyne/', self.number + '.npz')))
            lidar[:, 3] = 1
            velo_tc_cam = self.calib['Tr_velo_to_cam'].reshape(3, 4)
            velo_tc_cam = np.vstack((velo_tc_cam, np.array([0, 0, 0, 1])))
            lidar = np.matmul(velo_tc_cam, lidar.T).T
        self.prepare_scan_waymoc(self.file_name, self.img, lidar, save=True, crop=True, visu=True)
        self.bboxes = []
        self.color = np.ones((self.lidar.shape[1], 3))

        for car_idx in range(len(cars)):
            if cars[car_idx].optimized:
                cars[car_idx] = self.create_bboxes_from_opt_values_tracker([cars[car_idx]], False, True)[0]
            self.visu3D_car(cars[car_idx], car_idx)

        if self.args.dataset == 'waymo':
            self.visu_ground_truth_waymo()
        if self.cfg.visualization.visu_labels_gt != 'None':
            self.draw_boxes_from_labels_all(self.cfg.visualization.visu_labels_gt, [0, 1, 0]) #Green
        if self.cfg.visualization.visu_labels1 != 'None':
            self.draw_boxes_from_labels(self.cfg.visualization.visu_labels1, [0, 0, 1]) #Blue
        if self.cfg.visualization.visu_labels2 != 'None':
            self.draw_boxes_from_labels(self.cfg.visualization.visu_labels2, [1, 1, 0]) #Yellow
        if self.cfg.visualization.visu_output_labels:
            self.draw_boxes_from_labels(self.cfg.paths.labels_path, [0, 0, 1]) #Red

        if self.cfg.visualization.show_img:
            #show self.img
            immg = np.moveaxis(self.img, 0, -1)
            cv2.imwrite('img.png', immg)
        self.draw_pcloud()

    def visualize_3D_custom_dataset(self, cars, scale_cars):
        # TODO Recreate for scale detector also
        self.bboxes = []
        self.lidar = None
        self.color = None

        for car_idx in range(len(cars)):
            if cars[car_idx].optimized:
                cars[car_idx] = self.create_bboxes_from_opt_values_tracker([cars[car_idx]], from_scale_params=False, visu=True)[0]
            self.visu3D_car(cars[car_idx], car_idx, custom=True)

        if scale_cars is not None:
            for car_idx in range(len(scale_cars)):
                scale_cars[car_idx].x = scale_cars[car_idx].x_scale
                scale_cars[car_idx].y = scale_cars[car_idx].y_scale
                scale_cars[car_idx].z = scale_cars[car_idx].z_scale
                if scale_cars[car_idx].optimized:
                    scale_cars[car_idx] = self.create_bboxes_from_opt_values_tracker([scale_cars[car_idx]], from_scale_params=False, visu=True)[0]
                self.visu3D_car(scale_cars[car_idx], car_idx, custom=True)

        self.draw_pcloud()

    def visu3D_car(self, car, car_idx, custom=False):
        if car.lidar is None or len(car.lidar) == 0 or car.lidar.shape[1] == 0:
            return
        if car.optimized:
            if self.cfg.visualization.visu_predicted_bbox:
                car.bbox.color = np.array([1, 0, 0])
                self.bboxes.append(car.bbox)

            if self.cfg.visualization.visu_template:
                template = self.get_template(car.x, car.y, car.z, car.theta, car.model, car.length/self.cfg.templates.template_length, car.width/self.cfg.templates.template_width, car.height/self.cfg.templates.template_height).T
                padding = np.ones((3, template.shape[1]))
                template = np.concatenate((template, padding), axis=0)
                if self.lidar is None:
                    self.lidar = template
                else:
                    self.lidar = np.concatenate((self.lidar, template), axis=1)
                color_temp = np.zeros((template.shape[1], 3))
                color_temp[:, 0] = 1
                if self.color is None:
                    self.color = color_temp
                else:
                    self.color = np.concatenate((self.color, color_temp))

        if self.cfg.visualization.visu_aggregated_lidar:
            seg_lidar = car.lidar
            if self.lidar is None:
                self.lidar = seg_lidar
            else:
                self.lidar = np.concatenate((self.lidar, seg_lidar), axis=1)
            color_temp = np.zeros((seg_lidar.shape[1], 3))
            color_temp[:, 0] = self.random_colors[car_idx % 16, 0]
            color_temp[:, 1] = self.random_colors[car_idx % 16, 1]
            color_temp[:, 2] = self.random_colors[car_idx % 16, 2]
            if self.color is None:
                self.color = color_temp
            else:
                self.color = np.concatenate((self.color, color_temp))

        if self.cfg.visualization.visu_closest_lidar_for_moving and car.moving:
            if car.moving_scale_lidar[0] is not None:
                closest_lidar = np.concatenate((car.moving_scale_lidar[0], np.zeros((car.moving_scale_lidar[0].shape[0], 3))), axis=1)
                if closest_lidar is not None:
                    if self.lidar is None:
                        self.lidar = closest_lidar.T
                    else:
                        self.lidar = np.concatenate((self.lidar, closest_lidar.T), axis=1)
                    color_temp = np.zeros((closest_lidar.T.shape[1], 3))
                    color_temp[:, 0] = self.random_colors[car_idx % 16, 0]
                    color_temp[:, 1] = self.random_colors[car_idx % 16, 1]
                    color_temp[:, 2] = self.random_colors[car_idx % 16, 2]
                    if self.color is None:
                        self.color = color_temp
                    else:
                        self.color = np.concatenate((self.color, color_temp))

        if car.scale_lidar is not None and self.cfg.visualization.visu_scale_lidar:
            seg_lidar = car.scale_lidar
            padding = np.zeros((2, seg_lidar.shape[1]))
            # Concatenate the original array with the padding
            seg_lidar = np.vstack((seg_lidar, padding))
            if self.lidar is None:
                self.lidar = seg_lidar
            else:
                self.lidar = np.concatenate((self.lidar, seg_lidar), axis=1)
            color_temp = np.zeros((seg_lidar.shape[1], 3))
            color_temp[:, 0] = self.random_colors[car_idx % 16, 0]
            color_temp[:, 1] = self.random_colors[car_idx % 16, 1]
            color_temp[:, 2] = self.random_colors[car_idx % 16, 2]
            if self.color is None:
                self.color = color_temp
            else:
                self.color = np.concatenate((self.color, color_temp))

        if car.locations is not None and self.cfg.visualization.visu_locations and car.moving:
            for loc in car.locations:
                if loc is not None:
                    small_bbox = open3d.geometry.OrientedBoundingBox(loc[:3], np.eye(3), np.array([0.1, 0.1, 0.1]))
                    small_bbox.color = np.array([self.random_colors[car_idx % 16, 0], self.random_colors[car_idx % 16, 1], self.random_colors[car_idx % 16, 2]])
                    self.bboxes.append(small_bbox)

        if custom:
            center = np.array([car.gt_bbox[0], car.gt_bbox[1], car.gt_bbox[2]])
            size = np.array([car.gt_bbox[3], car.gt_bbox[4], car.gt_bbox[5]])
            yaw = car.gt_bbox[6]
            r = R.from_euler('zyx', [yaw, 0, 0], degrees=False)
            bbox = open3d.geometry.OrientedBoundingBox(center, r.as_matrix(), size)
            bbox.color = np.array([0, 1, 0])
            self.bboxes.append(bbox)

    def visu_ground_truth_waymo(self):
        frame = self.waymo_frame[self.pic_index]

        for idx, laser_label in enumerate(frame.laser_labels):
            # print(laser_label)
            if laser_label.type != label_pb2.Label.Type.TYPE_VEHICLE:
                continue
            size = np.array(
                [laser_label.box.length, laser_label.box.width, laser_label.box.height])  # Height, width, length
            center = np.array([laser_label.box.center_x, laser_label.box.center_y, laser_label.box.center_z])  # x,y,z
            yaw = laser_label.box.heading
            r = R.from_euler('zyx', [yaw, 0, 0], degrees=False)
            bbox = open3d.geometry.OrientedBoundingBox(center, r.as_matrix(), size)
            bbox.color = np.array([0., 1., 0.])
            self.bboxes.append(bbox)

    def show_image(self, img):
        if self.args.dataset == 'waymo':
            for i in range(len(img)):
                cv2.imshow("", img[i])
        else:
            cv2.imshow("", img)
            k = cv2.waitKey(0)

    def draw_GT_bboxes(self):
        # Draw the GT bboxes
        f = open(self.cfg.paths.kitti_path + 'object_detection/training/label_2/' + self.file_name + '.txt', 'r')
        # f = open('<path_to_your_data>/Data_object_det/training/label_2/' + file_name + '.txt', 'r')
        # Read all lines from the file
        lines = f.readlines()
        arr = [line.strip().split(" ") for line in lines]
        if not self.cfg.general.supress_debug_prints:
            print("GROUND TRUTH:")
            print(arr)
        for i in range(len(arr)):
            if arr[i][0] == 'Car' or arr[i][0] == 'car':
                size = np.array(
                    [(float(arr[i][9])), (float(arr[i][8])), (float(arr[i][10]))])  # Height, width, length
                center = np.array([(float(arr[i][11])), (float(arr[i][12]) - size[1]/2), (float(arr[i][13]))])  # x,y,z
                yaw = (float(arr[i][14])) - np.pi / 2.  # For unknow reasons, have to be shifted ...

                r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
                bbox = open3d.geometry.OrientedBoundingBox(center, r.as_matrix(), size)
                bbox.color = np.array([0, 1, 0])
                self.bboxes.append(bbox)
    def draw_boxes_from_labels(self, labels_path, color, color_ped=[0, 1, 1]):
        f = open(labels_path + self.file_name + '.txt', 'r')
        lines = f.readlines()
        arr = [line.strip().split(" ") for line in lines]
        if not self.cfg.general.supress_debug_prints:
            print(labels_path)
            print(arr)
        for i in range(len(arr)):
            if arr[i][0] == 'Car' or arr[i][0] == 'car':
                size = np.array(
                    [(float(arr[i][9])), (float(arr[i][8])), (float(arr[i][10]))])  # Height, width, length
                center = np.array(
                    [(float(arr[i][11])), (float(arr[i][12]) - size[1] / 2), (float(arr[i][13]))])  # x,y,z
                yaw = (float(arr[i][14])) - np.pi / 2.  # For unknow reasons, have to be shifted ...

                r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
                bbox = open3d.geometry.OrientedBoundingBox(center, r.as_matrix(), size)
                bbox.color = np.array(color)
                self.bboxes.append(bbox)
            elif arr[i][0] == 'Pedestrian' or arr[i][0] == 'pedestrian' or arr[i][0] == 'Cyclist' or arr[i][0] == 'cyclist':
                size = np.array(
                    [(float(arr[i][9])), (float(arr[i][8])), (float(arr[i][10]))])  # Height, width, length
                center = np.array(
                    [(float(arr[i][11])), (float(arr[i][12]) - size[1] / 2), (float(arr[i][13]))])  # x,y,z
                yaw = (float(arr[i][14])) - np.pi / 2.  # For unknow reasons, have to be shifted ...

                r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
                bbox = open3d.geometry.OrientedBoundingBox(center, r.as_matrix(), size)
                bbox.color = np.array(color_ped)
                self.bboxes.append(bbox)


    def draw_boxes_from_labels_all(self, labels_path, color):
        if not os.path.exists(labels_path + self.folder + "/labels_00/" + self.number + '.txt'):
            return
        f = open(labels_path + self.folder + "/labels_00/" + self.number + '.txt', 'r')
        lines = f.readlines()
        arr = [line.strip().split(" ") for line in lines]
        if not self.cfg.general.supress_debug_prints:
            print(labels_path)
            print(arr)
        for i in range(len(arr)):
            if arr[i][0] == 'Car' or arr[i][0] == 'car':
                size = np.array(
                    [(float(arr[i][9])), (float(arr[i][8])), (float(arr[i][10]))])  # Height, width, length
                center = np.array(
                    [(float(arr[i][11])), (float(arr[i][12]) - size[1] / 2), (float(arr[i][13]))])  # x,y,z
                yaw = (float(arr[i][14])) - np.pi / 2.  # For unknow reasons, have to be shifted ...

                r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
                bbox = open3d.geometry.OrientedBoundingBox(center, r.as_matrix(), size)
                bbox.color = np.array(color)
                self.bboxes.append(bbox)


    def draw_pcloud(self):
        point_cloud = open3d.geometry.PointCloud()
        coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)  # Only for visu purpose
        point_cloud.points = open3d.utility.Vector3dVector(np.flipud(self.lidar.T[:, 0:3]))
        point_cloud.colors = open3d.utility.Vector3dVector(np.flipud(self.color))
        point_cloud = point_cloud.voxel_down_sample(voxel_size=0.05)

        visu_things = [point_cloud]
        for k in range(len(self.bboxes)):
            visu_things.append(self.bboxes[k])
        visu_things.append(coord_frame)

        #open3d.visualization.draw_geometries(visu_things, zoom=0.1, lookat=[0, 0, 1], front=[0, -0.2, -0.5],
        #                                     up=[0, -1, 0])
        visualizer = open3d.visualization.Visualizer()
        visualizer.create_window()
        for k in range(len(visu_things)):
            visualizer.add_geometry(visu_things[k])
        #visualizer.get_render_option().point_size = 5  # Adjust the point size if necessary
        visualizer.get_render_option().background_color = np.asarray([0, 0, 0])  # Set background to black
        visualizer.get_view_control().set_front([0, -0.3, -0.5])
        visualizer.get_view_control().set_lookat([0, 0, 1])
        visualizer.get_view_control().set_zoom(0.05)
        visualizer.get_view_control().set_up([0, -1, 0])
        visualizer.get_view_control().camera_local_translate(5.,0.,8.)
        visualizer.run()
        visualizer.destroy_window()

    def polygon_clip(self, subjectPolygon, clipPolygon):
        """ Clip a polygon with another polygon.

        Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

        Args:
          subjectPolygon: a list of (x,y) 2d points, any polygon.
          clipPolygon: a list of (x,y) 2d points, has to be *convex*
        Note:
          **points have to be counter-clockwise ordered**

        Return:
          a list of (x,y) vertex point for the intersection polygon.
        """

        def inside(p):
            return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

        def computeIntersection():
            dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
            dp = [s[0] - e[0], s[1] - e[1]]
            n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
            n2 = s[0] * e[1] - s[1] * e[0]
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
            return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

        outputList = subjectPolygon
        cp1 = clipPolygon[-1]

        for clipVertex in clipPolygon:
            cp2 = clipVertex
            inputList = outputList
            outputList = []
            s = inputList[-1]

            for subjectVertex in inputList:
                e = subjectVertex
                if inside(e):
                    if not inside(s):
                        outputList.append(computeIntersection())
                    outputList.append(e)
                elif inside(s):
                    outputList.append(computeIntersection())
                s = e
            cp1 = cp2
            if len(outputList) == 0:
                return None
        return (outputList)

    def poly_area(self, x, y):
        """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def convex_hull_intersection(self, p1, p2):
        """ Compute area of two convex hull's intersection area.
            p1,p2 are a list of (x,y) tuples of hull vertices.
            return a list of (x,y) for the intersection and its volume
        """
        inter_p = self.polygon_clip(p1, p2)
        if inter_p is not None:
            hull_inter = ConvexHull(inter_p)
            return inter_p, hull_inter.volume
        else:
            return None, 0.0

    def box3d_vol(self, corners):
        ''' corners: (8,3) no assumption on axis direction '''
        a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
        b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
        c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
        return a * b * c

    def box3d_iou_waymo(self, corners1, corners2):
        ''' Compute 3D bounding box IoU. Taken somewhere from the internet

        Input:
            corners1: numpy array (8,3), assume up direction is negative Y
            corners2: numpy array (8,3), assume up direction is negative Y
        Output:
            iou: 3D bounding box IoU
            iou_2d: bird's eye view 2D bounding box IoU

        todo (kent): add more description on corner points' orders.
        '''
        # corner points are in counter clockwise order
        rect1 = [(corners1[i, 0], corners1[i, 1]) for i in range(3, -1, -1)]
        rect2 = [(corners2[i, 0], corners2[i, 1]) for i in range(3, -1, -1)]

        area1 = self.poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
        area2 = self.poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

        inter, inter_area = self.convex_hull_intersection(rect1, rect2)
        iou_2d = inter_area / (area1 + area2 - inter_area)
        zmax = min(corners1[0, 2], corners2[0, 2])
        zmin = max(corners1[4, 2], corners2[4, 2])

        inter_vol = inter_area * max(0.0, zmax - zmin)

        vol1 = self.box3d_vol(corners1)
        vol2 = self.box3d_vol(corners2)
        iou = inter_vol / (vol1 + vol2 - inter_vol)
        return iou, iou_2d

    def get_3d_box_waymo(self, box_size, heading_angle, center):
        ''' Calculate 3D bounding box corners from its parameterization.

        Input:
            box_size: tuple of (length,wide,height)
            heading_angle: rad scalar, clockwise from pos x axis
            center: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box cornders
        '''

        def roty(t):
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, s, 0],
                             [-s, c, 0],
                             [0, 0, 1]])

        R = roty(heading_angle)
        l, w, h = box_size
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
        y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
        z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + center[0];
        corners_3d[1, :] = corners_3d[1, :] + center[1];
        corners_3d[2, :] = corners_3d[2, :] + center[2];
        corners_3d = np.transpose(corners_3d)
        return corners_3d

    def show3d_pcdet_car_bbox(self):
        boxes, scores, labels = self.run_openpcdet(ours=False)
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        if not self.cfg.general.supress_debug_prints:
            print("PCDET Results orig kitti")

        for i in range(boxes.shape[0]):
            line_set, box3d = V.translate_boxes_to_open3d_instance(boxes[i])
            center = box3d.get_center()
            transformed_center = np.matmul(self.velo_to_cam, np.array([center[0], center[1], center[2], 1]))
            box3d.center = transformed_center[:3]
            box3d.R = np.matmul(self.velo_to_cam[:3, :3], box3d.R)
            box3d.color = np.array([0., 1. - scores[i], scores[i]])
            box3d.color = np.array([0., 0., 1.])
            # Red to yellow, where red is equal to score 1 and yellow to score 0
            self.bboxes.append(box3d)
            if not self.cfg.general.supress_debug_prints:
                print(scores[i], transformed_center)

        boxes, scores, labels = self.run_openpcdet(ours=True)
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        if not self.cfg.general.supress_debug_prints:
            print("PCDET Results ours")
            print(boxes)
        for i in range(boxes.shape[0]):
            line_set, box3d = V.translate_boxes_to_open3d_instance(boxes[i])
            center = box3d.get_center()
            transformed_center = np.matmul(self.velo_to_cam, np.array([center[0], center[1], center[2], 1]))
            box3d.center = transformed_center[:3]
            box3d.R = np.matmul(self.velo_to_cam[:3, :3], box3d.R)
            box3d.color = np.array([scores[i], 1 - scores[i], 0.])
            box3d.color = np.array([1., 0., 0.])
            # Blue to light blue, where blue is equal to score 1 and light blue to score 0

            if not self.cfg.general.supress_debug_prints:
                print(scores[i], transformed_center)

            # get_3d_box(box_size, heading_angle, center)
            corners_3d_predict = self.get_3d_box((box3d.extent[2], box3d.extent[1], box3d.extent[0]), boxes[i, 6],
                                                 (box3d.center[0], box3d.center[1], box3d.center[2]))

            for z in range(len(self.debug_GT)):
                corners_3d_ground = self.get_3d_box(
                    (self.debug_GT[z][0][0], self.debug_GT[z][0][1], self.debug_GT[z][0][2]), self.debug_GT[z][1],
                    (self.debug_GT[z][2][0], self.debug_GT[z][2][1], self.debug_GT[z][2][2]))
                (IOU_3d, IOU_2d) = self.box3d_iou(corners_3d_predict, corners_3d_ground)
                if IOU_3d < 0.7 and IOU_2d > 0.7:
                    # box3d.color = np.array([1., 0., 0.])
                    if not self.cfg.general.supress_debug_prints:
                        print("3D: ", IOU_3d, "2D: ", IOU_2d)
            self.bboxes.append(box3d)

    if AutoLabel3D.use_pcdet:
        class DemoDataset(DatasetTemplate):
            def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
                """
                Args:
                    root_path:
                    dataset_cfg:
                    class_names:
                    training:
                    logger:
                """
                super().__init__(
                    dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
                )
                self.root_path = root_path
                self.ext = ext
                data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

                data_file_list.sort()
                self.sample_file_list = data_file_list

            def __len__(self):
                return len(self.sample_file_list)

            def __getitem__(self, index):
                if self.ext == '.bin':
                    points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
                elif self.ext == '.npy':
                    points = np.load(self.sample_file_list[index])
                else:
                    raise NotImplementedError

                input_dict = {
                    'points': points,
                    'frame_id': index,
                }

                data_dict = self.prepare_data(data_dict=input_dict)
                return data_dict

        def run_openpcdet(self, ours=False):
            cfg_from_yaml_file(self.cfg_to_yaml, cfg)
            logger = common_utils.create_logger()
            demo_dataset = self.DemoDataset(
                dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
                root_path=Path(self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number :0>10}' + '.bin'), ext='.bin', logger=logger
            )

            model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
            if ours:
                model.load_params_from_file(filename=self.pcdet_checkpoint_ours, logger=logger, to_cpu=True)
            else:
                model.load_params_from_file(filename=self.pcdet_checkpoint_orig, logger=logger, to_cpu=True)
            if self.cfg.general.device == 'gpu':
                model.cuda()
            model.eval()

            with torch.no_grad():
                for idx, data_dict in enumerate(demo_dataset):

                    data_dict = demo_dataset.collate_batch([data_dict])
                    load_data_to_gpu(data_dict)
                    pred_dicts, _ = model.forward(data_dict)

                    return pred_dicts[0]['pred_boxes'], pred_dicts[0]['pred_scores'], pred_dicts[0]['pred_labels']




