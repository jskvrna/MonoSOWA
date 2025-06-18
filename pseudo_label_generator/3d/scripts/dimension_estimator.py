import numpy as np
import open3d as o3d
import copy

from trimesh.triangles import closest_point

from anno_V3 import AutoLabel3D

class Dimension_estimator(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)

    def estimate_dimensions(self, car, est_theta=False):
        if car.moving_scale_lidar is None:
            return car

        if est_theta:
            car.moving_scale_lidar = [car.lidar.T[:, :3], car.lidar.T[:, :3]] + car.moving_scale_lidar

        dimensions_all = np.zeros((min(len(car.moving_scale_lidar), self.cfg.frames_creation.k_to_scale_estimation), 3))
        if est_theta:
            est_theta_arr  = np.zeros((min(len(car.moving_scale_lidar), self.cfg.frames_creation.k_to_scale_estimation)))

        for i in range(min(len(car.moving_scale_lidar), self.cfg.frames_creation.k_to_scale_estimation)):
            if car.moving_scale_lidar[i] is None:
                continue
            cur_lidar = copy.deepcopy(car.moving_scale_lidar[i])

            cur_lidar, center = self.move_pcloud_to_center_numpy(cur_lidar)

            points = cur_lidar

            best_params, best_theta, best_extent = self.estimate_best_params(points)
            #best_param2, best_theta2, best_extent2 = self.estimate_best_params_old(points)

            y_min, y_max = self.estimate_height(points)

            rectangle_2d = self.construct_rectangle(best_params)
            #rectangle_2d2 = self.construct_rectangle(best_param2)

            bottom_points = np.hstack((rectangle_2d[:, 0].reshape(-1, 1), np.ones((4, 1)) * y_min,
                                       rectangle_2d[:, 1].reshape(-1, 1)))  # Z = 0 for bottom
            top_points = np.hstack((rectangle_2d[:, 0].reshape(-1, 1), np.ones((4, 1)) * y_max,
                                    rectangle_2d[:, 1].reshape(-1, 1)))  # Z = height for top
            all_points = np.vstack((bottom_points, top_points))  # Combine into one array
            #bottom_points2 = np.hstack((rectangle_2d2[:, 0].reshape(-1, 1), np.ones((4, 1)) * y_min,
            #                           rectangle_2d2[:, 1].reshape(-1, 1)))  # Z = 0 for bottom
            #top_points2 = np.hstack((rectangle_2d2[:, 0].reshape(-1, 1), np.ones((4, 1)) * y_max,
            #                        rectangle_2d2[:, 1].reshape(-1, 1)))
            #all_points2 = np.vstack((bottom_points2, top_points2))

            our_bbox, angle, bbox_center, extent = self.get_3D_bbox(all_points)
            our_bbox.color = (0, 0, 1)  # Blue color for the OBB
            #old_bbox, angle2, bbox_center2, extent2 = self.get_3D_bbox(all_points2)
            #old_bbox.color = (1, 0, 0)  # Red color for the OBB

            # Create a point cloud from all points
            #pcd = o3d.geometry.PointCloud()
            #pcd.points = o3d.utility.Vector3dVector(all_points)
            #pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]] * len(all_points)))
            #pcd_input = o3d.geometry.PointCloud()
            #pcd_input.points = o3d.utility.Vector3dVector(points)
            #pcd_input.paint_uniform_color([0.8, 0.8, 0.8])

            # Compute the oriented bounding box
            #obb = pcd.get_oriented_bounding_box()

            # Visualize the OBB and point cloud
            #obb.color = (1, 0, 0)  # Red color for the OBB

            center = center + bbox_center

            #Now it is the time to perform the optimization as fine fitting
            #car, obb2 = self.fine_tune_with_optim(points, our_bbox, angle, car)

            angle_of_car = - angle - np.pi / 2.
            viewing_angle = np.arctan2(center[2], center[0]) - np.pi/2

            diff = angle_of_car - viewing_angle
            #print(np.rad2deg(diff), np.sin(diff), np.cos(diff))
            if np.abs(np.sin(diff)) < 0.09:
                if not self.cfg.general.supress_debug_prints:
                    print("Degenerate solution: length")
                #Degenerate solution, we can estimate only the width and height
                dimensions_all[i] = np.array([self.cfg.templates.template_length, extent[1], extent[2]])
            elif np.abs(np.cos(diff)) < 0.2:
                if our_bbox.extent[0] > 2.5:
                    if not self.cfg.general.supress_debug_prints:
                        print("Degenerate solution: width")
                    #Degenerate solution, we can estimate only the length and height
                    dimensions_all[i] = np.array([extent[0], extent[1], self.cfg.templates.template_width])
                else:
                    if not self.cfg.general.supress_debug_prints:
                        print("Weird degenerate solution: length. It think the width is the length")
                    dimensions_all[i] = np.array([self.cfg.templates.template_length, extent[1], extent[0]])
            else:
                if not self.cfg.general.supress_debug_prints:
                    print("Normal solution")
                dimensions_all[i] = extent

            if not self.cfg.general.supress_debug_prints:
                print(f"Car dimensions (L x H x W): {dimensions_all[i][0]:.2f} x {dimensions_all[i][1]:.2f} x {dimensions_all[i][2]:.2f}")

            if est_theta:
                est_theta_arr[i] = angle - np.pi / 2.

            #o3d.visualization.draw_geometries([pcd, pcd_input, our_bbox, old_bbox])
        if est_theta:
            car.theta = np.median(est_theta_arr)
        else:
            car.length = np.median(dimensions_all[:, 0])
            car.width = np.median(dimensions_all[:, 2])
            car.height = np.median(dimensions_all[:, 1])

            if car.length / car.width > 4. or car.length < 2. or car.width < 1. or car.height < 1. or car.width > 2.5 or car.length > 6. or np.isnan(car.length) or np.isnan(car.width) or np.isnan(car.height) or True:
                car.length = self.cfg.templates.template_length
                car.width = self.cfg.templates.template_width
                car.height = self.cfg.templates.template_height

        return car

    def get_3D_bbox(self, points):
        points_xz = points[:, [0, 2]]
        centroid = np.mean(points_xz, axis=0)
        centered_points = points_xz - centroid
        cov_matrix = np.cov(centered_points, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_indices]
        rotated_points = centered_points @ eigenvectors
        min_coords = np.min(rotated_points, axis=0)
        max_coords = np.max(rotated_points, axis=0)
        extents = max_coords - min_coords
        center_rotated = (min_coords + max_coords) / 2

        obb_center = eigenvectors @ center_rotated + centroid
        angle = -np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        height = y_max - y_min
        y_center = (y_max + y_min) / 2

        obb_center = np.array([obb_center[0], y_center, obb_center[1]])
        extents = np.array([extents[0], height, extents[1]])

        rot_matrix = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        oriented_bbox = o3d.geometry.OrientedBoundingBox(center=obb_center, R=rot_matrix, extent=extents)

        return oriented_bbox, angle, obb_center, extents

    def fine_tune_with_optim(self, points, obb, rotation_y, car):
        car.x = obb.center[0]
        car.y = obb.center[1]
        car.z = obb.center[2]
        car.theta = rotation_y
        car.length = obb.extent[0]
        car.width = obb.extent[2]
        car.height = obb.extent[1]
        self.filtered_lidar = points

        self.index = self.create_faiss_tree(self.filtered_lidar)
        car = self.optimize_fine_dimensions_estimation(car)
        print("Before: ", np.rad2deg(rotation_y), "After: ", np.rad2deg(car.theta))

        rotation_matrix = np.array(
            [[np.cos(car.theta), 0, np.sin(car.theta)], [0, 1, 0], [-np.sin(car.theta), 0, np.cos(car.theta)]])
        obb2 = o3d.geometry.OrientedBoundingBox(center=obb.center, R=rotation_matrix,
                                                extent=[car.length, car.width, car.height])
        obb2.color = (0, 1, 0)  # Green color for the OBB
        
        return car, obb2

    def estimate_location(self, car):
        if car.lidar is not None:
            cur_lidar = car.lidar.T[:, :3]

            location = np.median(cur_lidar, axis=0)
            car.x = location[0]
            car.y = location[1]
            car.z = location[2]
            car.theta = 0.
            return car
        else:
            car.x = 0
            car.y = 0
            car.z = 0
            car.theta = 0
            return car

    def estimate_best_params(self, points):
        best_theta = None
        max_criterion = -np.inf
        best_params = None
        best_extent = None

        points_xz = points[:, [0, 2]]

        for theta in range(0, 90, 1):
            theta_rad = np.deg2rad(theta)
            e1 = np.array([np.cos(theta_rad), np.sin(theta_rad)])
            e2 = np.array([-np.sin(theta_rad), np.cos(theta_rad)])

            c1 = np.matmul(points_xz, e1)
            c2 = np.matmul(points_xz, e2)

            percentile_90_c1 = np.percentile(c1, 90)
            percentile_10_c1 = np.percentile(c1, 10)
            percentile_90_c2 = np.percentile(c2, 90)
            percentile_10_c2 = np.percentile(c2, 10)

            # Calculate the distance to the nearest edge
            d1 = np.minimum(np.abs(c1 - percentile_10_c1), np.abs(percentile_90_c1 - c1))
            d2 = np.minimum(np.abs(c2 - percentile_10_c2), np.abs(percentile_90_c2 - c2))
            #d1 = np.minimum(np.abs(c1 - np.min(c1)), np.abs(np.max(c1) - c1))
            #d2 = np.minimum(np.abs(c2 - np.min(c2)), np.abs(np.max(c2) - c2))

            d1 = 1.0 / (1.0 + np.exp(-d1 * self.cfg.optimization.steepness))
            d2 = 1.0 / (1.0 + np.exp(-d2 * self.cfg.optimization.steepness))

            # Compute the closeness score
            #closeness_score = -np.sum(np.maximum(np.minimum(d1, d2), 1.))
            closeness_score = -np.sum(np.minimum(d1, d2))

            if closeness_score > max_criterion:
                max_criterion = closeness_score
                best_theta = theta_rad
                # Determine the edge parameters
                best_extent = np.array([c1.max() - c1.min(), c2.max() - c2.min()])
                best_params = {
                    'a1': np.cos(theta_rad), 'b1': np.sin(theta_rad), 'c1': c1.min(),
                    'a2': -np.sin(theta_rad), 'b2': np.cos(theta_rad), 'c2': c2.min(),
                    'a3': np.cos(theta_rad), 'b3': np.sin(theta_rad), 'c3': c1.max(),
                    'a4': -np.sin(theta_rad), 'b4': np.cos(theta_rad), 'c4': c2.max()
                }

        return best_params, best_theta, best_extent

    def estimate_best_params_old(self, points):
        best_theta = None
        max_criterion = -np.inf
        best_params = None
        best_extent = None

        points_xz = points[:, [0, 2]]

        for theta in range(0, 90, 1):
            theta_rad = np.deg2rad(theta)
            e1 = np.array([np.cos(theta_rad), np.sin(theta_rad)])
            e2 = np.array([-np.sin(theta_rad), np.cos(theta_rad)])

            c1 = np.matmul(points_xz, e1)
            c2 = np.matmul(points_xz, e2)

            #percentile_90_c1 = np.percentile(c1, 90)
            #percentile_10_c1 = np.percentile(c1, 10)
            #percentile_90_c2 = np.percentile(c2, 90)
            #percentile_10_c2 = np.percentile(c2, 10)

            # Calculate the distance to the nearest edge
            d1 = np.minimum(np.abs(c1 - np.min(c1)), np.abs(np.max(c1) - c1))
            d2 = np.minimum(np.abs(c2 - np.min(c2)), np.abs(np.max(c2) - c2))

            # Compute the closeness score
            closeness_score = -np.sum(np.maximum(np.minimum(d1, d2), 5.))
            #closeness_score = -np.sum(np.minimum(d1, d2))

            if closeness_score > max_criterion:
                max_criterion = closeness_score
                best_theta = theta_rad
                # Determine the edge parameters
                best_extent = np.array([c1.max() - c1.min(), c2.max() - c2.min()])
                best_params = {
                    'a1': np.cos(theta_rad), 'b1': np.sin(theta_rad), 'c1': c1.min(),
                    'a2': -np.sin(theta_rad), 'b2': np.cos(theta_rad), 'c2': c2.min(),
                    'a3': np.cos(theta_rad), 'b3': np.sin(theta_rad), 'c3': c1.max(),
                    'a4': -np.sin(theta_rad), 'b4': np.cos(theta_rad), 'c4': c2.max()
                }

        return best_params, best_theta, best_extent

    def estimate_height(self, points):
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])

        return y_min, y_max

    def move_pcloud_to_center_numpy(self, pcloud):
        centroid = np.mean(pcloud, axis=0)
        pcloud -= centroid
        return pcloud, centroid

    def construct_rectangle(self, params):
        """
        Constructs a rectangle from line parameters.
        Assumes params contains keys: 'a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'a3', 'b3', 'c3', 'a4', 'b4', 'c4'
        Each ai, bi, ci corresponds to the line equation ai*x + bi*y = ci
        """
        # Retrieve the parameters for the four edges
        a1, b1, c1 = params['a1'], params['b1'], params['c1']
        a2, b2, c2 = params['a2'], params['b2'], params['c2']
        a3, b3, c3 = params['a3'], params['b3'], params['c3']
        a4, b4, c4 = params['a4'], params['b4'], params['c4']

        # Calculate the intersection points
        point1 = self.line_intersection(a1, b1, c1, a2, b2, c2)  # Intersection of edge 1 and 2
        point2 = self.line_intersection(a1, b1, c1, a4, b4, c4)  # Intersection of edge 1 and 4
        point3 = self.line_intersection(a3, b3, c3, a2, b2, c2)  # Intersection of edge 3 and 2
        point4 = self.line_intersection(a3, b3, c3, a4, b4, c4)  # Intersection of edge 3 and 4

        return np.array([point1, point2, point3, point4])

    def line_intersection(self, a1, b1, c1, a2, b2, c2):
        """
        Finds the intersection of two lines given the coefficients of the lines.
        Line 1: a1*x + b1*y = c1
        Line 2: a2*x + b2*y = c2
        """
        A = np.array([[a1, b1], [a2, b2]])
        b = np.array([c1, c2])
        x, y = np.linalg.solve(A, b)
        return x, y