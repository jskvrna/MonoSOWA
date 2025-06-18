from detectron2.utils.logger import setup_logger
import numpy as np
import glob, os
import torch
import yaml

setup_logger()

class DictAsObject:
    def __init__(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    self.__dict__[key] = DictAsObject(value)
                elif isinstance(value, list):
                    self.__dict__[key] = [DictAsObject(item) if isinstance(item, dict) else item for item in value]
                else:
                    self.__dict__[key] = value
        else:
            raise TypeError("Expected a dictionary")

    def __getattr__(self, item):
        return self.__dict__.get(item)

    def __repr__(self):
        return f"{self.__dict__}"

class AutoLabel3D:
    use_pcdet = False
    # Example usage
    def load_yaml_as_object(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return DictAsObject(data)

    def __init__(self, args):
        # Load a YAML file
        self.cfg = self.load_yaml_as_object(args.config)
        self.args = args

        self.generate_merged_frames_only = False  # If True, we do not optimize, but only generate merged lidar frames.
        self.generate_merged_occlusion_only = False  # If True, we do not optimize, but only generate merged occupancy maps.
        self.generate_transformations_only = False  # If True, then the transformations with ICP are generated and saved
        self.generate_raw_lidar = False  # If True, then the raw lidars for waymo are computed
        self.generate_raw_masks_or_tracking = False  # If True, then the raw masks from detectron are computed
        self.generate_homographies_only = False  # If True, then the homographies are generated and saved
        self.generate_candidates = False  # If True, then the candidates are generated
        self.load_merged_frames = False  # If True, we do not perform the merging, but we directly load it
        self.load_transformations = False  # If True, then the transformations from the files will be loaded
        self.use_precomputed_homography = True # If True, then the precomputed homographies are used
        self.do_optim = False  # If False, then the frame is not optimized over -> for debug purposes
        self.do_optim_scale = False  # If False, then the frame is not optimized over with the scale
        self.compute_dimensions = False # If True, then the dimensions are computed

        if args.action == 'lidar_scans':
            self.generate_raw_lidar = True
        elif args.action == 'transformations':
            self.generate_transformations_only = True
        elif args.action == 'homographies':
            self.generate_homographies_only = True
            self.load_transformations = True
        elif args.action == 'mask_tracking':
            self.generate_raw_masks_or_tracking = True
            self.load_transformations = True
        elif args.action == 'candidates':
            self.generate_candidates = True
            self.load_transformations = True
        elif args.action == 'frames_aggregation':
            self.generate_merged_frames_only = True
            self.load_transformations = True
        elif args.action == 'optimization':
            self.load_transformations = True
            self.load_merged_frames = False
            self.do_optim = True
            if self.cfg.scale_detector.use_scale_detector:
                self.do_optim_scale = True
            else:
                self.do_optim_scale = False
        elif args.action == 'dimensions_output':
            self.load_transformations = True
            self.load_merged_frames = True
            self.compute_dimensions = True
            self.do_optim = False
            if self.cfg.scale_detector.use_scale_detector:
                self.do_optim_scale = True
            else:
                self.do_optim_scale = False
        elif args.action == 'demo':
            self.load_transformations = True
            self.load_merged_frames = True
            self.compute_dimensions = False
            self.do_optim = True
            if self.cfg.scale_detector.use_scale_detector:
                self.do_optim_scale = False
            else:
                self.do_optim_scale = False
        else:
            raise ValueError(f"Unknown action: {args.action}")

        self.precomputed_homos_approx = torch.tensor([[[ 3.2196e+00,  1.1214e-01, -6.3368e+02], [ 3.0874e-01,  1.9095e+00, -4.7618e+02], [ 4.8014e-04,  2.9707e-05,  1.0000e+00]],
                                  [[3.3657e+00, -4.1591e-03, -7.3796e+02],[3.2979e-01, 1.9213e+00, -5.6230e+02],[5.0463e-04, 1.2217e-05, 1.0000e+00]],
                                  [[4.3385e-01, -1.6411e-02, 1.7020e+03], [-1.6672e-01, 9.5433e-01, 8.0385e+01], [-2.6954e-04, -1.4319e-05, 1.0000e+00]],
                                  [[ 4.6936e-01, -8.6252e-03,  1.7212e+03], [-1.7158e-01,  9.7596e-01,  7.4091e+01], [-2.5763e-04, -4.6421e-06,  1.0000e+00]]])

        self.rng = np.random.default_rng(self.cfg.filtering.rnd_seed)  # Seed for the take_random

        if self.args.dataset == 'kitti':
            # The base path (self.cfg.paths.kitti_path) should be configured in your YAML file.
            kitti_image_dir = os.path.join(self.cfg.paths.kitti_path, 'object_detection', 'training', 'image_2')
            self.pics = glob.glob(os.path.join(kitti_image_dir, '*.png'))
            self.pics = sorted(self.pics, key=os.path.basename)

        self.random_colors = np.array([[0.58547825, 0.10008773, 0.56141813], [0.46844978, 0.88682883, 0.33014268],
                            [0.68185582, 0.01300377, 0.44983724], [0.68680198, 0.85496657, 0.37151209],
                            [0.76696593, 0.55142042, 0.64986374], [0.16990206, 0.07051164, 0.62984601],
                            [0.48070201, 0.40074112, 0.53333161], [0.36184151, 0.09169345, 0.13765216],
                            [0.27619561, 0.01809711, 0.17609385], [0.57131526, 0.7730725, 0.29672884],
                            [0.83913347, 0.95407624, 0.22593619], [0.82562147, 0.25773012, 0.46877934],
                            [0.22910274, 0.29108235, 0.1827987], [0.24255047, 0.13190583, 0.76127495],
                            [0.56652911, 0.57862834, 0.14781213], [0.17925054, 0.93785803, 0.28711138]])

        self.unit_cube = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ])

        self.loss_matrix = None
        self.loss_matrix_fine = None
        self.loss_matrix_outliers = None
        self.loss_matrix_scale = None
        self.P2_rect = None
        self.lidar_car_template_non_filt = None
        self.img_dist = None
        self.histogram = None
        self.min_dist = None
        self.opt_values = None
        self.filter_idx = None
        self.filt_success = None
        self.z_mean_lidar = None
        self.y_mean_lidar = None
        self.x_mean_lidar = None
        self.bboxes = []
        self.lidar_visu = None
        self.color = None
        self.filtered_lidar = None
        self.mask = None
        self.lidar = None
        self.unmerged_lidar = None
        self.out_data = None
        self.pic_index = None
        self.img_orig = None
        self.img = None
        self.file_name = None
        self.pic = None
        self.GT_yaw = None
        self.GT_center = None
        self.lidar_car_template = None
        self.template_histograms = None
        self.model = None
        self.cKDTree = None
        self.moving_cars = None
        self.path_to_folder = None
        self.file_number = None
        self.kitti_data = None
        self.standing_cars_lidar = None
        self.standing_cars_info = None
        self.curr_lidar = None
        self.velo_to_cam = None
        self.occupancy_map = None
        self.occupancy_map2d = None
        self.occupancy_map_raw = None
        self.sam_predictor = None
        self.detected_cars_mask = None
        self.detected_cars_opt_values = None
        self.detected_cars_opt_values_first_optim = None
        self.detected_cars_template_loss = None
        self.detected_car_bboxes = None
        self.scale_car_lidar = None
        self.scale_car_mask = None
        self.scale_optim_values = None
        self.index = None
        self.moving_cars_lidar_merged = None
        self.moving_cars_info_merged = None
        self.filt_success_moving = None
        self.moving_cars_masks = None
        self.standing_cars_masks = None
        self.moving_cars_masks = None
        self.segment_index = None
        self.frame_data = None
        self.waymo_data = []
        self.waymo_frame = []
        self.waymo_lidar = []
        self.prec_detectron_output = None
        self.car_locations = None
        self.detectron_output_arr = None
        self.car_locations_lidar = None
        self.car_locations_info = None
        self.car_locations_masks = None
        self.better_segment = None
        self.stitched_imgs = None
        self.homos_all = None
        self.detectron_masks = None
        self.extracted_lidar_points = None
        self.extracted_lidar_positions = None
        self.extracted_info = None
        self.extracted_masks = None
        self.cars = None
        self.pedestrians = None
        self.do_single_segment = None
        self.cars3D_start = None
        self.compressed_detandtracked = None
        self.custom_dataset_counter = 0
        self.lidar_car_template_scale = None
        self.pseudo_lidar_folder = None
        self.map_data_cur = None
        self.mesh_templates = None
        self.mesh_templates_p3d = None
        self.index_of_all_imgs = None
        self.folder = None
        self.number = None
        self.calib = None
