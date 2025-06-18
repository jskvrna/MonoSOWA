from anno_V3 import AutoLabel3D
import numpy as np
from scipy.spatial import distance_matrix
import zstd
import pickle
import os

class CustomDataset(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)
    def create_custom_dataset_from_cars(self):
        cars = self.find_GT_for_car(self.cars)
        self.save_cars(cars)
    def compute_distance_matrix(self, array1, array2):
        diff = array1[:, np.newaxis, :] - array2[np.newaxis, :, :]
        sq_diff = diff ** 2
        sum_sq_diff = np.sum(sq_diff, axis=2)
        distance_matrix = np.sqrt(sum_sq_diff)
        return distance_matrix

    def find_GT_for_car(self, cars):
        frame = self.waymo_frame[self.pic_index]

        gt_bboxes = []
        gt_centers = []
        for idx, laser_label in enumerate(frame.laser_labels):
            # print(laser_label)
            if laser_label.type != label_pb2.Label.Type.TYPE_VEHICLE:
                continue
            size = np.array([laser_label.box.length, laser_label.box.width, laser_label.box.height])  # Height, width, length
            center = np.array([laser_label.box.center_x, laser_label.box.center_y, laser_label.box.center_z])  # x,y,z
            yaw = laser_label.box.heading
            bbox = np.array([center[0], center[1], center[2], size[0], size[1], size[2], yaw])
            gt_bboxes.append(bbox)
            gt_centers.append(center)

        gt_centers = np.array(gt_centers)

        new_cars = []
        for car in cars:
            if car.optimized:
                new_cars.append(car)
        cars = new_cars

        pred_centers = []
        for car in cars:
            car.gt_bbox = None
            x_mean = np.median(car.lidar[0, :])
            y_mean = np.median(car.lidar[1, :])
            z_mean = np.median(car.lidar[2, :])
            pred_centers.append(np.array([x_mean, y_mean, z_mean]))

        pred_centers = np.array(pred_centers)

        dist_matrix = distance_matrix(pred_centers, gt_centers)
        for i in range(np.min([dist_matrix.shape[0], dist_matrix.shape[1]])):
            dist_matrix[i, i] = 1000.

        for i in range(len(cars)):
            min_dist_GT = np.argmin(dist_matrix[i, :])
            if dist_matrix[i, min_dist_GT] < 5. and np.argmin(dist_matrix[:, min_dist_GT]) == i:
                cars[i].gt_bbox = gt_bboxes[min_dist_GT]

        new_cars = []
        for car in cars:
            if car.gt_bbox is not None and car.lidar is not None and (car.lidar.shape[1] > 200 or car.moving):
                if car.moving:
                    estimated_angle = self.estimate_angle_from_movement_tracked(car)
                    car.estimated_angle = estimated_angle
                new_cars.append(car)
        return new_cars

    def save_cars(self, cars):
        for car in cars:
            car.bbox = None
            compressed_arr = zstd.compress(pickle.dumps(car, pickle.HIGHEST_PROTOCOL))

            with open(self.cfg.paths.custom_dataset_path + "/" + str(self.custom_dataset_counter) + ".zstd", 'wb') as f:
                f.write(compressed_arr)
            self.custom_dataset_counter += 1

    def load_custom_dataset(self):
        files = [f for f in os.listdir(self.cfg.paths.custom_dataset_path) if os.path.isfile(os.path.join(self.cfg.paths.custom_dataset_path, f))]

        #random_indexes = random.sample(range(len(files)), self.custom_dataset_size * 2)
        random_indexes = np.arange(len(files))

        cars = []
        for rnd_indx in random_indexes:
            with open(self.cfg.paths.custom_dataset_path + "/" + files[rnd_indx], 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            car = pickle.loads(decompressed_data)
            if car.lidar is not None:
                if car.lidar.shape[1] > self.cfg.filtering.lidar_threshold_during_optim:
                    cars.append(car)
            if len(cars) >= self.cfg.custom_dataset.custom_dataset_size_to_load:
                break

        for car_idx, car in enumerate(cars):
            car.optimized = False
            mean_x = np.median(car.lidar[0, :])
            mean_y = np.median(car.lidar[1, :])
            mean_z = np.median(car.lidar[2, :])

            car.lidar[0, :] -= mean_x
            car.lidar[1, :] -= mean_y
            car.lidar[2, :] -= mean_z

            car.scale_lidar[0, :] -= mean_x
            car.scale_lidar[1, :] -= mean_y
            car.scale_lidar[2, :] -= mean_z

            car.scale_bbox = car.gt_bbox.copy()

            car.gt_bbox[0] -= mean_x
            car.gt_bbox[1] -= mean_y
            car.gt_bbox[2] -= mean_z

            car.scale_bbox[0] -= mean_x
            car.scale_bbox[1] -= mean_y
            car.scale_bbox[2] -= mean_z

            to_shift = -(len(cars) * self.cfg.custom_dataset.distance_between_cars * 0.5) + car_idx * self.cfg.custom_dataset.distance_between_cars
            car.lidar[0, :] += to_shift
            car.scale_lidar[0, :] += to_shift
            car.scale_lidar[1, :] += 10.
            car.gt_bbox[0] += to_shift
            car.scale_bbox[0] += to_shift
            car.scale_bbox[1] += 10.

            #pad scale lidar by zeros to match the size of the lidar
            car.scale_lidar = np.pad(car.scale_lidar, ((0, 2), (0, 0)), 'constant', constant_values=0)

        self.cars = cars

    def convert_optim_to_scale(self, cars):
        for car in cars:
            car.y += 10
            car.lidar = car.scale_lidar
            car.gt_bbox = car.scale_bbox
            #car.height *= 1.5
        return cars

    def custom_compute_iou(self, cars):
        for car_idx, car in enumerate(cars):
            corners_3d_gt = self.get_3d_box_waymo((car.gt_bbox[3], car.gt_bbox[4], car.gt_bbox[5]), car.gt_bbox[6],(car.gt_bbox[0], car.gt_bbox[1], car.gt_bbox[2]))
            corners_3d_pred = self.get_3d_box_waymo((car.length, car.width, car.height), car.theta,(car.x, car.y, car.z))
            (IOU_3d, IOU_2d) = self.box3d_iou_waymo(corners_3d_pred, corners_3d_gt)
            print(IOU_3d, IOU_2d)


