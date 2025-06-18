from anno_V3 import AutoLabel3D
import os
import numpy as np
from utils2 import load_velo_scan, load_pseudo_lidar
from scipy.spatial.transform import Rotation as R
import open3d
import copy


class ScaleDetector(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)

    def extract_lidar_data_from_bbox_tracker(self, cars):
        cars = self.create_bboxes_from_opt_values_tracker(cars)

        if self.cfg.frames_creation.use_pseudo_lidar:
            for car in cars:
                if car.lidar is not None and car.optimized:
                    car.scale_lidar = car.lidar
            return cars

        if self.args.dataset == 'waymo':
            transformations = self.calculate_transformations_waymo(self.cfg.frames_creation.nscans_before_scale, self.cfg.frames_creation.nscans_after_scale)
        else:
            transformations = self.calculate_transformationsV2(self.cfg.frames_creation.nscans_before_scale, self.cfg.frames_creation.nscans_after_scale)

        min_one_standing = False
        for i in range(len(cars)):
            if not cars[i].moving and cars[i].optimized:
                min_one_standing = True
                break

        if min_one_standing:
            # Placeholder for the relative path to velodyne data.
            # This should ideally be loaded from a configuration file.
            velodyne_data_subpath = os.path.join('velodyne_points', 'data')

            for i in range(-self.cfg.frames_creation.nscans_before_scale, self.cfg.frames_creation.nscans_after_scale + 1):
                # Check if we have enough data to do the merging over this frame
                if self.args.dataset == 'kitti':
                    if self.cfg.frames_creation.use_pseudo_lidar:
                        path_to_cur_velo = os.path.join(self.pseudo_lidar_folder, f'{self.file_number + i :0>10}.npz')
                    else:
                        path_to_cur_velo = os.path.join(self.path_to_folder, velodyne_data_subpath, f'{self.file_number + i :0>10}.bin')
                    # Check that we have everything we need
                    if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(path_to_cur_velo):
                        continue
                else:
                    if self.pic_index + i < 0 or self.pic_index + i >= len(self.waymo_lidar):
                        continue

                T_cur_to_ref = transformations[i + self.cfg.frames_creation.nscans_before_scale, :, :]
                T_cur_to_ref_inv = np.linalg.inv(T_cur_to_ref)

                if self.args.dataset == 'kitti':
                    if self.cfg.frames_creation.use_pseudo_lidar:
                        lidar_cur = load_pseudo_lidar(path_to_cur_velo)
                    else:
                        lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
                        lidar_cur = self.transform_velo_to_cam(self.file_name, lidar_cur, filter_points=False)
                        lidar_cur = lidar_cur[:4, :].T
                        lidar_cur[:, 3] = 1
                else:
                    lidar_cur = self.waymo_lidar[self.pic_index + i][:, :4]
                    lidar_cur[:, 3] = 1

                transformed_lidar = open3d.utility.Vector3dVector(lidar_cur[:, :3])

                for z in range(len(cars)):
                    # We do not want scale optimization over moving cars
                    if not cars[z].moving and cars[z].optimized and cars[z].bbox is not None:
                        cur_bbox = copy.deepcopy(cars[z].bbox)
                        center = np.array(cur_bbox.center).reshape((3, 1))
                        center = np.pad(center, ((0, 1), (0, 0)))
                        center[3] = 1.
                        cur_bbox.center = np.matmul(T_cur_to_ref_inv, center)[:3]
                        r = cur_bbox.R
                        r = np.matmul(T_cur_to_ref_inv[:3,:3], r)
                        cur_bbox.R = r

                        inside_points = cur_bbox.get_point_indices_within_bounding_box(transformed_lidar)

                        #TODO Perform context growing
                        #inside_points = self.perform_growing_bbox(cur_bbox, lidar_cur)

                        if inside_points is None or len(inside_points) == 0:
                            continue
                        lidar_points = np.array(lidar_cur.T[:, inside_points])
                        transformed_points = np.matmul(T_cur_to_ref, lidar_points)

                        if cars[z].scale_lidar is None:
                            cars[z].scale_lidar = transformed_points
                        else:
                            cars[z].scale_lidar = np.concatenate((cars[z].scale_lidar, transformed_points), axis=1)

            for i in range(len(cars)):
                if cars[i].scale_lidar is not None and cars[i].scale_lidar.shape[1] > 0:
                    if not cars[i].moving:
                        #tmp1 = self.downsample(cars[i].lidar.T[:,:3]).T
                        #tmp3 = self.downsample_random(cars[i].lidar.T[:, :3], 1000).T
                        #tmp2 = self.downsample(cars[i].scale_lidar.T[:, :3]).T
                        cars[i].scale_lidar = cars[i].scale_lidar[:3, :]
                        #cars[i].scale_lidar = np.concatenate((tmp1, tmp2, tmp3), axis=1)
                        cars[i].scale_lidar = np.pad(cars[i].scale_lidar, ((0, 1), (0, 0)))
                        cars[i].scale_lidar[3, :] = 1

        return cars

    def create_bboxes_from_opt_values_tracker(self, cars, from_scale_params=False, visu=False):
        for i in range(len(cars)):
            if cars[i].optimized:
                if self.args.dataset == 'waymo':
                    size = np.array([cars[i].length, cars[i].width, cars[i].height])  # Height, width, length # Height, width, length
                else:
                    size = np.array([cars[i].width, cars[i].height, cars[i].length])

                if from_scale_params:
                    center = np.array([cars[i].x_scale, cars[i].y_scale, cars[i].z_scale])  # x,y,z
                    yaw = cars[i].theta_scale
                else:
                    center = np.array([cars[i].x, cars[i].y, cars[i].z]) # x,y,z
                    if not visu:
                        size = size * self.cfg.scale_detector.bbox_scale
                    yaw = cars[i].theta
                if self.args.dataset == 'waymo':
                    r = R.from_euler('zyx', [yaw, 0, 0], degrees=False)
                else:
                    r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
                bbox = open3d.geometry.OrientedBoundingBox(center, r.as_matrix(), size)

                cars[i].bbox = bbox

            else:
                cars[i].bbox = None

        return cars

    def bbox_reducer_tracked(self, cars):
        cars = self.create_bboxes_from_opt_values_tracker(cars, from_scale_params=True)

        for i in range(len(cars)):
            if not cars[i].moving and cars[i].scale_lidar is not None:
                # For each bbox find points which are inside of it
                cur_bbox = copy.deepcopy(cars[i].bbox)
                if self.args.dataset == 'waymo':
                    old_width = cur_bbox.extent[1]
                    old_height = cur_bbox.extent[2]
                    old_length = cur_bbox.extent[0]
                else:
                    old_width = cur_bbox.extent[0]
                    old_height = cur_bbox.extent[1]
                    old_length = cur_bbox.extent[2]
                center_old = cur_bbox.center
                # Make the bbox a little bit bigger than our initial guess and also shift it by half of the Y axis extent so we discard points on ground
                if self.args.dataset == 'waymo':
                    extent_new = np.array(cur_bbox.extent) + [abs(np.cos(cars[i].theta_scale)) * self.cfg.scale_detector.width_bloat, abs(np.sin(cars[i].theta_scale) * self.cfg.scale_detector.width_bloat), 0.4]
                    center_new = np.array(cur_bbox.center) + [0., 0., 0.4]
                else:
                    extent_new = np.array(cur_bbox.extent) + [abs(np.cos(cars[i].theta_scale)) * self.cfg.scale_detector.width_bloat, 0.4, abs(np.sin(cars[i].theta_scale) * self.cfg.scale_detector.width_bloat)]
                    center_new = np.array(cur_bbox.center) + [0., -0.4, 0.]
                cur_bbox = open3d.geometry.OrientedBoundingBox(center_new, cur_bbox.R, extent_new)
                # Now filter the data by the bbox

                tmp = cars[i].lidar[:4, :]
                tmp1 = open3d.utility.Vector3dVector(tmp[:3, :].T)
                inside_points = cur_bbox.get_point_indices_within_bounding_box(tmp1)
                lidar_points_inside = np.array(tmp[:4, inside_points])

                # Shift the points to the center
                center = np.array(cur_bbox.get_center()).reshape((3, 1))
                lidar_points_inside[:3, :] -= center
                # Now rotate them so they will be aligned with the axis
                tr_matrix = np.eye(4)
                tr_matrix[:3, :3] = cur_bbox.R
                lidar_points_inside = np.matmul(np.linalg.inv(tr_matrix), lidar_points_inside)
                # Now find the axis aligned bbox
                point_cloud = open3d.geometry.PointCloud()
                point_cloud.points = open3d.utility.Vector3dVector(lidar_points_inside.T[:, 0:3])
                obb = point_cloud.get_axis_aligned_bounding_box()
                # Rotate the new center back and merge it with the old one
                center_axisaligned = np.array(obb.get_center()).reshape((3, 1))
                center_axisaligned = np.pad(center_axisaligned, ((0, 1), (0, 0)))
                center_axisaligned[3] = 1.
                center_axisaligned[0] = 0.
                center_axisaligned = np.matmul(tr_matrix, center_axisaligned)[:3]
                new_center = center + center_axisaligned

                if self.args.dataset == 'waymo':
                    width = obb.get_extent()[1]
                    length = obb.get_extent()[0]
                else:
                    width = obb.get_extent()[0]
                    length = obb.get_extent()[2]

                #Now we need to find the height
                cur_bbox = copy.deepcopy(cars[i].bbox)
                # Make the bbox a little bit higher to capture the height of the car.
                if self.args.dataset == 'waymo':
                    extent_new = np.array(cur_bbox.extent) + [abs(np.cos(cars[i].theta_scale)) * self.cfg.scale_detector.width_bloat, abs(np.sin(cars[i].theta_scale) * self.cfg.scale_detector.width_bloat), 0.8]
                    center_new = np.array(cur_bbox.center) + [0., 0., 0.2]
                else:
                    extent_new = np.array(cur_bbox.extent) + [abs(np.cos(cars[i].theta_scale)) * self.cfg.scale_detector.width_bloat, 0.8, abs(np.sin(cars[i].theta_scale) * self.cfg.scale_detector.width_bloat)]
                    center_new = np.array(cur_bbox.center) + [0., -0.2, 0.]
                cur_bbox = open3d.geometry.OrientedBoundingBox(center_new, cur_bbox.R, extent_new)
                # Now filter the data by the bbox

                tmp = cars[i].lidar[:4, :]
                tmp1 = open3d.utility.Vector3dVector(tmp[:3, :].T)
                inside_points = cur_bbox.get_point_indices_within_bounding_box(tmp1)
                lidar_points_inside = np.array(tmp[:4, inside_points])

                # Now find the axis aligned bbox
                point_cloud = open3d.geometry.PointCloud()
                point_cloud.points = open3d.utility.Vector3dVector(lidar_points_inside.T[:, 0:3])
                obb = point_cloud.get_axis_aligned_bounding_box()
                # Rotate the new center back and merge it with the old one
                if self.args.dataset == 'waymo':
                    height = obb.get_extent()[2]
                else:
                    height = obb.get_extent()[1]

                if cars[i].moving:
                    height = np.clip(height, 1.2, 2.5)

                if length/old_length < self.cfg.scale_detector.max_length_diff_scale or length/old_length > 1.1:
                    if self.args.dataset == 'waymo':
                        size = np.array([self.cfg.templates.template_length, self.cfg.templates.template_width, self.cfg.templates.template_height])  # Height, width, length
                    else:
                        size = np.array([self.cfg.templates.template_width, self.cfg.templates.template_height, self.cfg.templates.template_length])  # Height, width, length

                    center = np.array([cars[i].x, cars[i].y, cars[i].z])  # x,y,z
                    yaw = cars[i].theta

                    if self.args.dataset == 'waymo':
                        r = R.from_euler('zyx', [yaw, 0, 0], degrees=False)
                    else:
                        r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
                    new_obb = open3d.geometry.OrientedBoundingBox(center, r.as_matrix(), size)
                    new_obb.color = [1., 0., 1.]
                else:
                    if self.args.dataset == 'waymo':
                        new_obb = open3d.geometry.OrientedBoundingBox(new_center, tr_matrix[:3, :3], [length + self.cfg.scale_detector.scale_offset_length, old_width, height])
                    else:
                        new_obb = open3d.geometry.OrientedBoundingBox(new_center, tr_matrix[:3, :3],[old_width, height,length + self.cfg.scale_detector.scale_offset_length])
                    new_obb.color=[1., 1., 0]

                if cars is not None:
                    cars[i].x = new_obb.center[0]
                    cars[i].y = new_obb.center[1]
                    cars[i].z = new_obb.center[2]
                    cars[i].theta = cars[i].theta_scale
                    # extent
                    if self.args.dataset == 'waymo':
                        cars[i].length = new_obb.extent[0]
                        cars[i].width = new_obb.extent[1]
                        cars[i].height = new_obb.extent[2]
                    else:
                        cars[i].width = new_obb.extent[0]
                        cars[i].height = new_obb.extent[1]
                        cars[i].length = new_obb.extent[2]

        return cars
