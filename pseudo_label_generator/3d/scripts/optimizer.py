import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
from anno_V3 import AutoLabel3D
from pytorch3d.transforms import euler_angles_to_matrix
import open3d as o3d

class Optimizer(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)

    def optimize_car(self, car):
        car.length = self.cfg.templates.template_length
        car.width = self.cfg.templates.template_width
        car.height = self.cfg.templates.template_height
        car.model = 0
        if not car.moving:
            car = self.optimize_coarse(car)
            car = self.optimize_fine(car)
        else:
            car = self.optimize_moving(car)
        car.optimized = True
        if self.cfg.optimization.use_dimensions_estimation_during_optim:
            car = self.estimate_dimensions(car)
        return car

    def optimize_car_robust(self, car):
        car.length = self.cfg.templates.template_length
        car.width = self.cfg.templates.template_width
        car.height = self.cfg.templates.template_height
        car.model = 0
        if not car.moving:
            car = self.estimate_dimensions(car, est_theta=True)
            if np.isnan(car.theta) or np.isnan(car.length) or np.isnan(car.width) or np.isnan(car.height):
                car = self.optimize_coarse(car)
                car = self.optimize_fine(car)
            else:
                car = self.optimize_loc_only(car)
        else:
            car = self.optimize_moving(car)
        car.optimized = True
        if self.cfg.optimization.use_dimensions_estimation_during_optim:
            car = self.estimate_dimensions(car)
        return car

    def optimize_car_scale(self, car):
        if car.optimized and car.scale_lidar is not None and car.scale_lidar.shape[1] > 0:
            car = self.optimize_scale(car)
            return car
        else:
            return car
        
    def optimize_coarse(self, car=None):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_values = np.array([0., 0., 0.])

        if self.cfg.loss_functions.loss_function == 'diffbin':
            if self.cfg.general.device == 'gpu':
                self.filtered_lidar = torch.tensor(self.filtered_lidar).cuda()
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0]).cuda()
            else:
                self.filtered_lidar = torch.tensor(self.filtered_lidar)
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0])
        elif self.cfg.loss_functions.loss_function == 'binary1way' or self.cfg.loss_functions.loss_function == 'binary2way':
            self.index = self.create_faiss_tree(self.filtered_lidar)

        for opt_param1 in np.linspace(self.cfg.optimization.opt_param1_min, self.cfg.optimization.opt_param1_max, num=self.cfg.optimization.opt_param1_iters):
            for opt_param2 in np.linspace(self.cfg.optimization.opt_param2_min, self.cfg.optimization.opt_param2_max, num=self.cfg.optimization.opt_param2_iters):
                for opt_param3 in np.linspace(0, 2 * np.pi - (2 * np.pi/self.cfg.optimization.opt_param3_iters), num=self.cfg.optimization.opt_param3_iters):
                    if self.args.dataset == 'waymo':
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, opt_param2 + self.y_mean_lidar, self.z_mean_lidar, opt_param3)
                    else:
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, self.y_mean_lidar, opt_param2 + self.z_mean_lidar, opt_param3)

                    loss = self.compute_loss(template_it)

                    if loss < min_loss:
                        min_loss = loss
                        opt_values = np.array([opt_param1, opt_param2, opt_param3])

        car.x = opt_values[0] + self.x_mean_lidar
        if self.args.dataset == 'waymo':
            car.y = opt_values[1] + self.y_mean_lidar
            car.z = self.z_mean_lidar
        else:
            car.y = self.y_mean_lidar
            car.z = opt_values[1] + self.z_mean_lidar
        car.theta = opt_values[2]

        return car

    def optimize_fine(self, car):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_value = 0.

        for theta in np.linspace(0, 2 * np.pi - (2 * np.pi/360), num=360):
            template_it = self.get_template(car.x, car.y, car.z, theta)

            loss = self.compute_loss(template_it)

            if loss < min_loss:
                min_loss = loss
                opt_value = theta

        car.theta = opt_value
        return car

    def optimize_loc_only(self, car=None):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_values = np.array([0., 0., 0.])

        if self.cfg.loss_functions.loss_function == 'diffbin':
            if self.cfg.general.device == 'gpu':
                self.filtered_lidar = torch.tensor(self.filtered_lidar).cuda()
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0]).cuda()
            else:
                self.filtered_lidar = torch.tensor(self.filtered_lidar)
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0])
        elif self.cfg.loss_functions.loss_function == 'binary1way' or self.cfg.loss_functions.loss_function == 'binary2way':
            self.index = self.create_faiss_tree(self.filtered_lidar)

        for opt_param1 in np.linspace(self.cfg.optimization.opt_param1_min, self.cfg.optimization.opt_param1_max, num=self.cfg.optimization.opt_param1_iters):
            for opt_param2 in np.linspace(self.cfg.optimization.opt_param2_min, self.cfg.optimization.opt_param2_max, num=self.cfg.optimization.opt_param2_iters):
                for opt_param3 in [car.theta, car.theta + np.pi]:
                    if self.args.dataset == 'waymo':
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, opt_param2 + self.y_mean_lidar, self.z_mean_lidar, opt_param3)
                    else:
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, self.y_mean_lidar, opt_param2 + self.z_mean_lidar, opt_param3)

                    loss = self.compute_loss(template_it)

                    if loss < min_loss:
                        min_loss = loss
                        opt_values = np.array([opt_param1, opt_param2, opt_param3])

        car.x = opt_values[0] + self.x_mean_lidar
        if self.args.dataset == 'waymo':
            car.y = opt_values[1] + self.y_mean_lidar
            car.z = self.z_mean_lidar
            car.theta = opt_values[2]
        else:
            car.y = self.y_mean_lidar
            car.z = opt_values[1] + self.z_mean_lidar
            car.theta = opt_values[2]

        return car

    def optimize_fine_dimensions_estimation(self, car):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_value = 0.

        for theta in np.linspace(0, 2 * np.pi - (2 * np.pi/360), num=360):
            scale_length = car.length / self.cfg.templates.template_length
            scale_width = car.width / self.cfg.templates.template_width
            scale_height = car.height / self.cfg.templates.template_height
            template_it = self.get_template(car.x, car.y, car.z, theta, scale_length=scale_length, scale_width=scale_width, scale_height=scale_height)

            loss = self.compute_loss(template_it)

            if loss < min_loss:
                min_loss = loss
                opt_value = theta

        car.theta = opt_value - np.pi/2
        return car

    def optimize_scale(self, car=None):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_values = np.array([0., 0., 0., 0., 0., 0., 0.])

        if car.scale_lidar is not None:
            tmp_opt_values = [car.x, car.y, car.z, car.theta]
            cur_lidar = car.scale_lidar.T[:, :3]
        else:
            return car

        if self.args.dataset == 'waymo':
            height_of_car = np.amax(cur_lidar[:, 2]) - np.amin(cur_lidar[:, 2])
        else:
            height_of_car = np.amax(cur_lidar[:, 1]) - np.amin(cur_lidar[:, 1])
        perfect_height_scale = height_of_car / self.cfg.templates.template_height
        perfect_height_scale = np.clip(perfect_height_scale, 0.75, 1.25)

        # Now we want to compute how much we want to move in each direction. If the car is parallel to our car (0 degrees)
        # Then we want to move more in the z axis than the x axis, because the length is usually our biggest problem.
        opt_param1_max = np.abs(np.cos(tmp_opt_values[3]) + np.sin(tmp_opt_values[3]))
        opt_param2_max = np.abs(np.sin(tmp_opt_values[3]) + np.cos(tmp_opt_values[3]))
        opt_param1_range = np.linspace(-opt_param1_max, opt_param1_max, num=self.cfg.optimization.opt_param1_iters)
        opt_param2_range = np.linspace(-opt_param2_max, opt_param2_max, num=self.cfg.optimization.opt_param2_iters)
        length_range = np.linspace(self.cfg.scale_detector.scale_min, self.cfg.scale_detector.scale_max, num=self.cfg.optimization.scale_num_scale_iters)

        if self.cfg.scale_detector.use_independent_width_scaling:
            width_range = np.linspace(self.cfg.scale_detector.scale_min, self.cfg.scale_detector.scale_max, num=self.cfg.optimization.width_num_scale_iters)
        else:
            width_range = [1.]

        #We are currently scaling the moving car.
        angle_to_iterate = [tmp_opt_values[3]]

        cur_lidar = self.downsample_lidar(cur_lidar)[:, :3]

        self.index = self.create_faiss_tree(cur_lidar)

        for template_index in range(self.cfg.scale_detector.num_of_templates):
            for scale_length in length_range:
                for scale_width in width_range:
                    for opt_param1 in opt_param1_range:
                        for opt_param2 in opt_param2_range:
                            for theta in angle_to_iterate:
                                if self.args.dataset == 'waymo':
                                    if self.cfg.scale_detector.use_independent_width_scaling:
                                        template_it = self.get_template(opt_param1 + tmp_opt_values[0], opt_param2 + tmp_opt_values[1], tmp_opt_values[2], theta, template_index, scale_length, scale_width, perfect_height_scale, scale=True)
                                    else:
                                        template_it = self.get_template(opt_param1 + tmp_opt_values[0], opt_param2 + tmp_opt_values[1], tmp_opt_values[2], theta, template_index, scale_length, scale_length, perfect_height_scale, scale=True)
                                else:
                                    if self.cfg.scale_detector.use_independent_width_scaling:
                                        template_it = self.get_template(opt_param1 + tmp_opt_values[0], tmp_opt_values[1], opt_param2 + tmp_opt_values[2], theta, template_index, scale_length, scale_width, perfect_height_scale, scale=True)
                                    else:
                                        template_it = self.get_template(opt_param1 + tmp_opt_values[0], tmp_opt_values[1], opt_param2 + tmp_opt_values[2], theta, template_index, scale_length, scale_length, perfect_height_scale, scale=True)

                                loss = self.compute_loss(template_it)

                                if loss < min_loss:
                                    min_loss = loss
                                    # Compute the loss of the template, the inputs are switched so we actually get loss for template.
                                    if self.cfg.scale_detector.use_independent_width_scaling:
                                        opt_values = np.array([template_index, scale_length, scale_width, opt_param1, opt_param2, theta])
                                    else:
                                        opt_values = np.array([template_index, scale_length, scale_length, opt_param1, opt_param2, theta])

        car.template_index = opt_values[0]
        car.length = opt_values[1] * self.cfg.templates.template_length
        car.width = opt_values[2] * self.cfg.templates.template_width
        car.height = perfect_height_scale * self.cfg.templates.template_height
        car.x_scale = opt_values[3] + tmp_opt_values[0]
        if self.args.dataset == 'waymo':
            car.y_scale = opt_values[4] + tmp_opt_values[1]
            car.z_scale = tmp_opt_values[2]
        else:
            car.y_scale = tmp_opt_values[1]
            car.z_scale = opt_values[4] + tmp_opt_values[2]
        car.theta_scale = opt_values[5]

        #Now lets iterate over height and Z value
        opt_param1_range = np.linspace(-opt_param1_max, opt_param1_max, num=20)
        height_range = np.linspace(self.cfg.scale_detector.scale_min, self.cfg.scale_detector.scale_max, num=self.cfg.optimization.scale_num_scale_iters)
        min_loss = np.inf

        for scale_height in height_range:
            for opt_param1 in opt_param1_range:
                if self.args.dataset == 'waymo':
                    if self.cfg.scale_detector.use_independent_width_scaling:
                        template_it = self.get_template(car.x_scale, car.y_scale, opt_param1 + tmp_opt_values[2], car.theta, 1, car.length, car.width, scale_height, scale=True)
                    else:
                        template_it = self.get_template(car.x_scale, car.y_scale, opt_param1 + tmp_opt_values[2], car.theta, 1, car.length, car.width, scale_height, scale=True)
                else:
                    if self.cfg.scale_detector.use_independent_width_scaling:
                        template_it = self.get_template(car.x_scale, opt_param1 + tmp_opt_values[1], car.z_scale, car.theta, 1, car.length, car.width, scale_height, scale=True)
                    else:
                        template_it = self.get_template(car.x_scale, opt_param1 + tmp_opt_values[1], car.z_scale, car.theta, 1, car.length, car.width, scale_height, scale=True)

                loss = self.compute_loss(template_it)

                if loss < min_loss:
                    min_loss = loss
                    # Compute the loss of the template, the inputs are switched so we actually get loss for template.
                    opt_values = np.array([opt_param1, scale_height])

        car.height = opt_values[1] * self.cfg.templates.template_height
        if self.args.dataset == 'waymo':
            car.z_scale = opt_values[0] + tmp_opt_values[2]
        else:
            car.y_scale = opt_values[0] + tmp_opt_values[1]

        return car


    def optimize_moving(self, car=None):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_values = np.array([0., 0., 0.])

        if hasattr(car, 'estimated_angle'):
            estimated_angle = car.estimated_angle
        else:
            estimated_angle = self.estimate_angle_from_movement_tracked(car)

        opt_param1_range = np.linspace(self.cfg.optimization.opt_param1_min, self.cfg.optimization.opt_param1_max, num=self.cfg.optimization.opt_param1_iters)
        opt_param2_range = np.linspace(self.cfg.optimization.opt_param2_min + 1., self.cfg.optimization.opt_param2_max + 1., num=self.cfg.optimization.opt_param2_iters)
        if estimated_angle is not None:
            opt_param3_range = [estimated_angle]
        else:
            opt_param3_range = np.linspace(0, 2 * np.pi - (2 * np.pi/self.cfg.optimization.opt_param3_iters), num=self.cfg.optimization.opt_param3_iters)

        if self.cfg.loss_functions.loss_function == 'diffbin':
            if self.cfg.general.device == 'gpu':
                self.filtered_lidar = torch.tensor(self.filtered_lidar).cuda()
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0]).cuda()
            else:
                self.filtered_lidar = torch.tensor(self.filtered_lidar)
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0])
        else:
            self.index = self.create_faiss_tree(self.filtered_lidar)

        for opt_param1 in opt_param1_range:
            for opt_param2 in opt_param2_range:
                for opt_param3 in opt_param3_range:
                    if self.args.dataset == 'waymo':
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, opt_param2 + self.y_mean_lidar, self.z_mean_lidar, opt_param3)
                    else:
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, self.y_mean_lidar, opt_param2 + self.z_mean_lidar, opt_param3)

                    loss = self.compute_loss(template_it)

                    if loss < min_loss:
                        min_loss = loss
                        opt_values = np.array([opt_param1, opt_param2, opt_param3])

        car.x = opt_values[0] + self.x_mean_lidar
        if self.args.dataset == 'waymo':
            car.y = opt_values[1] + self.y_mean_lidar
            car.z = self.z_mean_lidar
        else:
            car.y = self.y_mean_lidar
            car.z = opt_values[1] + self.z_mean_lidar
        car.theta = opt_values[2]

        return car

    def estimate_angle_from_movement_tracked(self, car):
        if self.args.dataset == 'waymo':
            moving_car_info = car.info
        else:
            moving_car_info = []
        moving_car_locations = car.locations

        # If the car has been seen on only few frames, then we cannot estimate anything.
        if len(moving_car_info) < 3 and len(moving_car_locations) < 3:
            car.theta = None
            return None
        else:
            # First we need to find the frame, where it was in the reference frame
            ref_idx = None
            if self.args.dataset == 'waymo':
                for i in range(len(moving_car_info)):
                    if moving_car_info[i] is not None:
                        if moving_car_info[i][1] == self.pic_index:
                            ref_idx = i
                if ref_idx is None:
                    car.theta = None
                    return None
            else:
                for i in range(len(moving_car_locations)):
                        if moving_car_locations[i][3] == 0:
                            ref_idx = i
                if ref_idx is None:
                    car.theta = None
                    return None

            estimation_arr = []

            i = ref_idx - 1
            count = 0
            while i >= 0 and count < 5:
                if moving_car_locations[i] is not None:
                    if self.args.dataset == 'waymo':
                        dist = np.sqrt((np.power(moving_car_locations[ref_idx][0] - moving_car_locations[i][0], 2) + np.power(moving_car_locations[ref_idx][1] - moving_car_locations[i][1], 2)))
                    else:
                        dist = np.sqrt((np.power(moving_car_locations[ref_idx][0] - moving_car_locations[i][0],2) + np.power(moving_car_locations[ref_idx][2] - moving_car_locations[i][2], 2)))
                    if dist > self.cfg.optimization.moving_cars_min_dist_for_angle:
                        if self.args.dataset == 'waymo':
                            angle = np.arctan2(moving_car_locations[ref_idx][1] - moving_car_locations[i][1],moving_car_locations[ref_idx][0] - moving_car_locations[i][0])
                        else:
                            angle = np.arctan2(moving_car_locations[ref_idx][2] - moving_car_locations[i][2], moving_car_locations[ref_idx][0] - moving_car_locations[i][0])
                        estimation_arr.append(angle)
                        count += 1
                i -= 1

            i = ref_idx + 1
            count = 0
            while i < len(moving_car_locations) and count < 5:
                if moving_car_locations[i] is not None:
                    if self.args.dataset == 'waymo':
                        dist = np.sqrt((np.power(moving_car_locations[i][0] - moving_car_locations[ref_idx][0], 2) + np.power(moving_car_locations[i][1] - moving_car_locations[ref_idx][1], 2)))
                    else:
                        dist = np.sqrt((np.power(moving_car_locations[i][0] - moving_car_locations[ref_idx][0], 2) + np.power(moving_car_locations[i][2] - moving_car_locations[ref_idx][2], 2)))
                    if dist > self.cfg.optimization.moving_cars_min_dist_for_angle:
                        if self.args.dataset == 'waymo':
                            angle = np.arctan2(moving_car_locations[i][1] - moving_car_locations[ref_idx][1], moving_car_locations[i][0] - moving_car_locations[ref_idx][0])
                        else:
                            angle = np.arctan2(moving_car_locations[i][2] - moving_car_locations[ref_idx][2],moving_car_locations[i][0] - moving_car_locations[ref_idx][0])
                        estimation_arr.append(angle)
                        count += 1
                i += 1

        if len(estimation_arr) < 3:
            return None
        else:
            if len(estimation_arr) % 2 == 0:
                estimation_arr.append(estimation_arr[-1])
            estimation_arr = np.array(estimation_arr)
            predicted_angle = np.median(estimation_arr)
            if predicted_angle > np.pi:
                predicted_angle -= 2 * np.pi
            if self.args.dataset == 'kitti' or self.args.dataset == 'all' or self.args.dataset == "waymo_converted":
                predicted_angle = -predicted_angle + np.pi/2
            return predicted_angle

    def get_template(self, x, y, z, theta, template_index=0, scale_length=1.0, scale_width=1.0, scale_height=1.0, scale=False):
        if self.cfg.loss_functions.loss_function == 'diffbin':
            if scale:
                template_it = self.lidar_car_template_scale[template_index].clone()
            else:
                template_it = self.lidar_car_template_non_filt[template_index].clone()
        else:
            if scale:
                template_it = self.lidar_car_template_scale[template_index].copy()
            else:
                template_it = self.lidar_car_template_non_filt[template_index].copy()

        if scale_length != 1.0 or scale_width != 1.0 or scale_height != 1.0:
            if self.args.dataset == 'waymo':
                if template_index == 0:
                    template_it[:, 2] -= self.cfg.templates.offset_fiat
                elif template_index == 1:
                    template_it[:, 2] -= self.cfg.templates.offset_passat
            else:
                if template_index == 0:
                    template_it[:, 1] += self.cfg.templates.offset_fiat
                elif template_index == 1:
                    template_it[:, 1] += self.cfg.templates.offset_passat

            template_it[:, 0] *= scale_length
            template_it[:, 1] *= scale_width
            template_it[:, 2] *= scale_height

            if self.args.dataset == 'waymo':
                if template_index == 0:
                    template_it[:, 2] += self.cfg.templates.offset_fiat
                elif template_index == 1:
                    template_it[:, 2] += self.cfg.templates.offset_passat
            else:
                if template_index == 0:
                    template_it[:, 1] -= self.cfg.templates.offset_fiat
                elif template_index == 1:
                    template_it[:, 1] -= self.cfg.templates.offset_passat

        if self.cfg.loss_functions.loss_function == 'diffbin':
            if self.cfg.general.device == 'gpu':
                if self.args.dataset == 'waymo':
                    r = euler_angles_to_matrix(torch.tensor([theta, 0., 0.]), "ZYX").cuda()
                else:
                    r = euler_angles_to_matrix(torch.tensor([0., theta, 0.]), "ZYX").cuda()
            else:
                if self.args.dataset == 'waymo':
                    r = euler_angles_to_matrix(torch.tensor([theta, 0., 0.]), "ZYX")
                else:
                    r = euler_angles_to_matrix(torch.tensor([0., theta, 0.]), "ZYX")
            template_it = torch.matmul(r, template_it.T).T
        else:
            if self.args.dataset == 'waymo':
                r = R.from_euler('zyx', [theta, 0, 0], degrees=False)
            else:
                r = R.from_euler('zyx', [0, theta, 0], degrees=False)
            template_it = np.matmul(r.as_matrix(), template_it.T).T

        template_it[:, 0] += x
        template_it[:, 1] += y
        template_it[:, 2] += z

        return template_it
