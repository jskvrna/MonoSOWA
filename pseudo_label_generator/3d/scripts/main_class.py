import time

from loader import Loader
from filtering import Filtering
from visualization import Visualization
from optimizer import Optimizer
from loss import Losses
from output import Output
from scale_detector import ScaleDetector
from context_growing import CAARGrowing
from stitching import Stitching
from tracker_ODTrack import Tracker_ODTrack
from custom_dataset import CustomDataset
from metric3d import Metric3D
from deformable_mesh import Deformable_Mesh
from dimension_estimator import Dimension_estimator

import os
import zstd
import pickle
import pykitti
import copy
import numpy as np
import dill
import glob
from tqdm import tqdm

class DetAndTracking:
    def __init__(self, lidar_points, lidar_locations, lidar_info, masks):
        self.lidar_points = lidar_points
        self.lidar_locations = lidar_locations
        self.lidar_info = lidar_info
        self.masks = masks

class MainClass(Output, Losses, Optimizer, Visualization, Filtering, Loader, ScaleDetector, CAARGrowing, Stitching, Tracker_ODTrack, CustomDataset, Metric3D, Dimension_estimator, Deformable_Mesh):
    def __init__(self, args):
        super().__init__(args)

    def main_waymo(self, argv):
        if self.generate_raw_masks_or_tracking:
            self.load_det2_and_sam()
        seq_num_min, seq_num_max = self.limit_sequences()

        self.prepare_dirs()

        self.load_lidar_templatesv2()
        dict_idx_to_opt = self.load_idx_to_opt()

        for self.segment_index in range(seq_num_min, seq_num_max):
            if not self.cfg.general.supress_debug_prints:
                print("Segment Index: ", self.segment_index)
            if self.do_single_segment is not None and self.do_single_segment != self.segment_index:
                continue

            self.load_current_segment()

            if self.generate_raw_lidar:
                continue

            if not self.load_merged_frames and not self.generate_transformations_only:
                if self.cfg.frames_creation.tracker_for_merging == '2D' and self.args.dataset == 'waymo':
                    if self.check_for_merging_done(dict_idx_to_opt):
                        continue
                    if not self.perform_stitching_and_trackingV2():
                        continue
                else:
                    if not self.perform_mask_extraction():
                        continue

            for self.pic_index, data in enumerate(self.waymo_data):
                if self.cfg.dataset.dataset_stride > 1 and not self.generate_transformations_only:
                    segment_name = self.random_indexes[self.segment_index].split('.')[0]
                    tmp_idx_arr = dict_idx_to_opt[segment_name]
                    if self.pic_index not in tmp_idx_arr:
                        continue
                if self.cfg.custom_dataset.create_custom_dataset:
                    if self.pic_index <= 80 or self.pic_index >= 120:
                        continue

                if not self.cfg.general.supress_debug_prints:
                    print("Frame Index: ", self.pic_index)
                self.prepare_pic_waymo(self.waymo_frame[self.pic_index])

                if self.check_for_optim_done() and self.cfg.output.output_txt:
                    continue

                if self.generate_merged_frames_only:
                    if self.cfg.frames_creation.tracker_for_merging == '2D':
                        self.load_and_prepare_lidar_scan_from_multiple_waymo_tracker(save=True)
                    else:
                        self.load_and_prepare_lidar_scan_from_multiple_waymo(save=True)
                    continue
                elif self.generate_transformations_only:
                    self.calculate_transformations_waymo(self.cfg.frames_creation.nscans_transformation_range, self.cfg.frames_creation.nscans_transformation_range, save=True)
                    continue
                elif self.load_merged_frames:
                    if self.cfg.frames_creation.tracker_for_merging == '2D':
                        self.load_merged_frames_from_files_waymo_tracker(track2D=True, merge_two_trackers=self.cfg.optimization.merge_two_trackers)
                    else:
                        self.load_merged_frames_from_files_waymo_tracker(track2D=False, merge_two_trackers=self.cfg.optimization.merge_two_trackers)
                else:
                    if self.cfg.frames_creation.tracker_for_merging == '2D':
                        self.load_and_prepare_lidar_scan_from_multiple_waymo_tracker(save=False)
                    else:
                        self.load_and_prepare_lidar_scan_from_multiple_waymo(save=False)

                if self.do_optim:
                    if not self.cfg.general.supress_debug_prints:
                        print("Optimizing over: ", len(self.cars), " cars")
                    for car_idx in range(len(self.cars)):
                        if not self.est_location_and_downsample(self.cars[car_idx]):
                            continue

                        self.cars[car_idx] = self.optimize_car(self.cars[car_idx])

                if not self.cfg.general.supress_debug_prints:
                    print("Doing NMS")

                self.cars, to_be_reopt = self.non_maxima_surpression(self.cars)
                if self.cfg.optimization.nms_merge_and_reopt:
                    if not self.cfg.general.supress_debug_prints:
                        print("Doing NMS merge and reopt")
                        print("Optimizing over: ", len(to_be_reopt), " cars")
                    for index in to_be_reopt:
                        if not self.est_location_and_downsample(self.cars[index]):
                            continue
                        self.cars[index] = self.optimize_car(self.cars[index])

                if self.cfg.scale_detector.use_scale_detector and self.do_optim:
                    self.cars = self.extract_lidar_data_from_bbox_tracker(self.cars)
                    if self.cfg.custom_dataset.create_custom_dataset:
                        self.create_custom_dataset_from_cars()
                    if self.do_optim_scale:
                        for i in range(len(self.cars)):
                            self.cars[i] = self.optimize_car_scale(self.cars[i])

                        if self.cfg.scale_detector.use_bbox_reducer:
                            self.bbox_reducer_tracked(self.cars)

                if self.cfg.output.output_txt:
                    self.writetxt_cars(self.cars)

                if self.cfg.visualization.show_3D_scan:
                    self.visualize_3D(self.cars)
                if self.cfg.visualization.show_image:
                    self.show_image(self.img)

                if not self.cfg.general.supress_debug_prints:
                    print("File name: ", self.file_name)
                    print("Pics_done: ", self.pic_index)

    def main_kitti(self, argv):
        if self.generate_raw_masks_or_tracking:
            self.load_det2_and_sam()

        self.prepare_dirs()
        pic_num_min, pic_num_max = self.limit_sequences()

        self.load_lidar_templatesv2()

        if self.generate_raw_lidar:
            self.generate_pseudo_lidar()
            return

        for self.pic_index in range(pic_num_min, pic_num_max):
            if not self.cfg.general.supress_debug_prints:
                print("Pic Index: ", self.pic_index)
            self.prepare_pic()
            self.map_data_cur = self.prepare_frame()

            if self.generate_raw_masks_or_tracking:
                start = time.time_ns()
                self.precompute_detectron_kitti()
                print("Time for detection: ", (time.time_ns() - start) / 1e6)
                continue

            if self.cfg.output.output_txt:
                if not self.cfg.general.supress_debug_prints:
                    print("Checking for opened file")
                if os.path.isfile(self.cfg.paths.labels_path + self.file_name + '.txt'):
                    continue

            if self.generate_merged_frames_only:
                self.load_and_prepare_lidar_scan_from_multiple_pykittiV2(self.file_name, self.img, save=True)
                continue
            elif self.generate_transformations_only:
                if os.path.isfile(self.cfg.paths.merged_frames_path + "/transformations/" + self.file_name + '.npy'):
                    continue
                if self.cfg.frames_creation.use_icp:
                    self.calculate_transformationsV2(self.cfg.frames_creation.nscans_transformation_range,
                                                     self.cfg.frames_creation.nscans_transformation_range, save=True)
                else:

                    self.calculate_transformations(self.cfg.frames_creation.nscans_transformation_range,
                                                   self.cfg.frames_creation.nscans_transformation_range, save=True)
                continue
            elif self.load_merged_frames:
                self.load_merged_frames_from_files_KITTI()
            else:
                self.load_and_prepare_lidar_scan_from_multiple_pykittiV2(self.file_name, self.img, save=False)

            if self.compute_dimensions:
                for car_idx in range(len(self.cars)):
                    self.cars[car_idx] = self.estimate_dimensions(self.cars[car_idx])
                    self.cars[car_idx] = self.estimate_location(self.cars[car_idx])
                self.writetxt_dimensions_cars(self.cars)
                continue

            if self.do_optim:
                if self.cfg.optimization.optimize_cars:
                    if not self.cfg.general.supress_debug_prints:
                        print("Optimizing over: ", len(self.cars), " cars")
                    for car_idx in range(len(self.cars)):
                        if not self.est_location_and_downsample(self.cars[car_idx]):
                            continue
                        if self.cfg.optimization.robust_optimization:
                            self.cars[car_idx] = self.optimize_car_robust(self.cars[car_idx])
                        else:
                            self.cars[car_idx] = self.optimize_car(self.cars[car_idx])

                if self.cfg.frames_creation.extract_pedestrians and self.cfg.optimization.optimize_pedestrians:
                    for ped_idx in range(len(self.pedestrians)):
                        if not self.est_location_and_downsample(self.pedestrians[ped_idx], pedestrian=True):
                            continue
                        self.pedestrians[ped_idx] = self.optimize_pedestrian(self.pedestrians[ped_idx])

            if not self.cfg.general.supress_debug_prints:
                print("Doing NMS")

            self.cars, to_be_reopt = self.non_maxima_surpression(self.cars)
            if self.cfg.optimization.nms_merge_and_reopt:
                if not self.cfg.general.supress_debug_prints:
                    print("Doing NMS merge and reopt")
                    print("Optimizing over: ", len(to_be_reopt), " cars")
                for index in to_be_reopt:
                    if not self.est_location_and_downsample(self.cars[index]):
                        continue
                    self.cars[index] = self.optimize_car(self.cars[index])

            if self.cfg.deformable_mesh.use_deformable_mesh:
                #for car_idx in range(len(self.cars)):
                    #self.cars[car_idx] = self.deformable_mesh_fit_lim_dof(self.cars[car_idx])
                self.cars = self.deformable_mesh_fit_lim_dof_batch(self.cars)

            if self.cfg.scale_detector.use_scale_detector and self.do_optim:
                if not self.cfg.general.supress_debug_prints:
                    print("Aggregating points for scale detector")
                self.cars = self.extract_lidar_data_from_bbox_tracker(self.cars)
                if self.cfg.custom_dataset.create_custom_dataset:
                    self.create_custom_dataset_from_cars()
                else:
                    for i in range(len(self.cars)):
                        if not self.cfg.general.supress_debug_prints:
                            print("Optimizing scale: ", i, " from ", len(self.cars))
                        self.cars[i] = self.optimize_car_scale(self.cars[i])

                    if self.cfg.scale_detector.use_bbox_reducer:
                        self.bbox_reducer_tracked(self.cars)

            if self.cfg.output.output_txt:
                self.writetxt_cars(self.cars, self.pedestrians)

            if self.cfg.output.save_optimized_cars:
                self.save_optimized_cars(self.cars)

            if self.cfg.visualization.show_3D_scan:
                self.visualize_3D(self.cars, self.pedestrians)
            if self.cfg.visualization.show_image:
                self.show_image(self.img)

            if not self.cfg.general.supress_debug_prints:
                print("File name: ", self.file_name)
                print("Pics_done: ", self.pic_index)

    def main_kitti360(self, argv):
        if self.generate_raw_masks_or_tracking:
            self.load_det2_and_sam()

        self.prepare_dirs()

        self.load_lidar_templatesv2()

        if self.generate_raw_lidar:
            self.generate_pseudo_lidar_all(dataset='all')
            return

        self.index_of_all_imgs = self.build_index_of_imgs()

        #pic_num_min, pic_num_max = self.limit_sequences_all()
        indexes = self.limit_sequences_all_random()

        #for self.pic_index in range(pic_num_min, pic_num_max):
        all_time_candidates = []
        all_time_frames = []
        all_time_optim = []
        all_time_masks = []
        all_time_transfors = []
        whole_time = []
        for self.pic_index in indexes:
            #if self.pic_index % 100 > 0:
            #if self.pic_index != 403:
                #continue
            if not self.cfg.general.supress_debug_prints:
                print("Pic Index: ", self.pic_index)
            self.prepare_pic_all(self.pic_index)

            start_whole = time.time_ns()

            #print(self.file_name)

            if self.generate_raw_masks_or_tracking:
                start_mask = time.time_ns()
                self.precompute_detectron_all()
                if not self.cfg.general.supress_debug_prints:
                    time_for_detection = (time.time_ns() - start_mask) / 1e9
                    all_time_masks.append(time_for_detection)
                    print("Mean time for detection: ", np.mean(all_time_masks), "Variance: ", np.var(all_time_masks), "Num of samples: ", len(all_time_masks))
                continue

            if self.generate_candidates:
                start_candidates = time.time_ns()
                if self.cfg.frames_creation.use_gt_masks:
                    self.precompute_candidates_gt()
                else:
                    self.precompute_candidates()
                if not self.cfg.general.supress_debug_prints:
                    time_taken = (time.time_ns() - start_candidates) / 1e9
                    all_time_candidates.append(time_taken)
                    print("Mean time for candidates: ", np.mean(all_time_candidates), "Variance: ", np.var(all_time_candidates), "Num of samples: ", len(all_time_candidates))
                continue

            if self.cfg.output.output_txt:
                if not self.cfg.general.supress_debug_prints:
                    print("Checking for opened file")
                if os.path.isfile(self.cfg.paths.labels_path + self.file_name + '.txt'):
                    continue

            if self.generate_merged_frames_only:
                start_frames = time.time_ns()
                self.load_and_prepare_lidar_scan_from_multiple_all(self.file_name, self.img, save=True)
                time_needed = (time.time_ns() - start_frames) / 1e9
                if not self.cfg.general.supress_debug_prints:
                    print("Time for merged frames: ", time_needed)
                    all_time_frames.append(time_needed)
                    print("Average time: ", np.mean(all_time_frames), "Variance: ", np.var(all_time_frames), "Num of samples: ", len(all_time_frames))
                continue
            elif self.generate_transformations_only:
                if os.path.isfile(self.cfg.paths.merged_frames_path + "/transformations/" + self.file_name + '.npy'):
                    continue
                start_transf = time.time_ns()
                self.calculate_transformations_all(self.cfg.frames_creation.nscans_transformation_range,
                                               self.cfg.frames_creation.nscans_transformation_range, save=True)
                time_needed = (time.time_ns() - start_transf) / 1e9
                print("Time for transformations: ", time_needed)
                all_time_transfors.append(time_needed)
                print("Average time: ", np.mean(all_time_transfors), "Variance: ", np.var(all_time_transfors), "Num of samples: ",
                      len(all_time_transfors))
                continue
            elif self.load_merged_frames:
                success = self.load_merged_frames_from_files_KITTI_all()
                if not success:
                    continue
            else:
                start_frames = time.time_ns()
                self.load_and_prepare_lidar_scan_from_multiple_all(self.file_name, self.img, save=False)
                time_needed = (time.time_ns() - start_frames) / 1e9
                if not self.cfg.general.supress_debug_prints:
                    print("Time for merged frames: ", time_needed)
                    all_time_frames.append(time_needed)
                    print("Merged frames Average time: ", np.mean(all_time_frames), "Variance: ", np.var(all_time_frames),
                          "Num of samples: ", len(all_time_frames))
            start_optim = time.time_ns()
            if self.do_optim:
                if not self.cfg.general.supress_debug_prints:
                    print("Optimizing over: ", len(self.cars), " cars")
                for car_idx in range(len(self.cars)):
                    if not self.est_location_and_downsample(self.cars[car_idx]):
                        continue
                    if self.cfg.optimization.robust_optimization:
                        self.cars[car_idx] = self.optimize_car_robust(self.cars[car_idx])
                    else:
                        self.cars[car_idx] = self.optimize_car(self.cars[car_idx])

            if not self.cfg.general.supress_debug_prints:
                print("Doing NMS")

            self.cars, to_be_reopt = self.non_maxima_surpression(self.cars)
            if self.cfg.optimization.nms_merge_and_reopt:
                if not self.cfg.general.supress_debug_prints:
                    print("Doing NMS merge and reopt")
                    print("Optimizing over: ", len(to_be_reopt), " cars")
                for index in to_be_reopt:
                    if not self.est_location_and_downsample(self.cars[index]):
                        continue
                    self.cars[index] = self.optimize_car(self.cars[index])

            if self.cfg.scale_detector.use_scale_detector and self.do_optim:
                if not self.cfg.general.supress_debug_prints:
                    print("Aggregating points for scale detector")
                self.cars = self.extract_lidar_data_from_bbox_tracker(self.cars)
                if self.cfg.custom_dataset.create_custom_dataset:
                    self.create_custom_dataset_from_cars()
                else:
                    for i in range(len(self.cars)):
                        if not self.cfg.general.supress_debug_prints:
                            print("Optimizing scale: ", i, " from ", len(self.cars))
                        self.cars[i] = self.optimize_car_scale(self.cars[i])

                    if self.cfg.scale_detector.use_bbox_reducer:
                        self.bbox_reducer_tracked(self.cars)
            time_taken = (time.time_ns() - start_optim) / 1e9
            all_time_optim.append(time_taken)
            time_whole_taken = (time.time_ns() - start_whole) / 1e9
            whole_time.append(time_whole_taken)
            print("Mean time for optimization: ", np.mean(all_time_optim), "Variance: ", np.var(all_time_optim), "Num of samples: ", len(all_time_optim))
            print("Whole mean time for optimization: ", np.mean(whole_time), "Variance: ", np.var(whole_time), "Num of samples: ", len(whole_time))
            if self.cfg.output.output_txt:
                self.writetxt_cars(self.cars)
            if self.cfg.output.save_optimized_cars:
                self.save_optimized_cars_all(self.cars)

            if self.cfg.visualization.show_3D_scan:
                self.visualize_3D_all(self.cars)
            if self.cfg.visualization.show_image:
                self.show_image(self.img)

            if not self.cfg.general.supress_debug_prints:
                print("File name: ", self.file_name)
                print("Pics_done: ", self.pic_index)

    def main_waymo_converted(self, argv):
        if self.generate_raw_masks_or_tracking:
            self.load_det2_and_sam()

        self.prepare_dirs()

        self.load_lidar_templatesv2()

        if self.generate_raw_lidar:
            self.generate_pseudo_lidar_waymoc(seq_start=self.args.seq_start, seq_end=self.args.seq_end)
            return

        self.index_of_all_imgs = self.build_index_of_imgs_waymoc()

        #pic_num_min, pic_num_max = self.limit_sequences_all()
        indexes = self.limit_sequences_all_random()

        #for self.pic_index in range(pic_num_min, pic_num_max):
        for self.pic_index in indexes:
            #if self.pic_index % 100 > 0:
            #if self.pic_index != 403:
                #continue
            if not self.cfg.general.supress_debug_prints:
                print("Pic Index: ", self.pic_index)
            self.prepare_pic_waymoc(self.pic_index)

            #print(self.file_name)

            if self.generate_raw_masks_or_tracking:
                start = time.time_ns()
                self.precompute_detectron_all()
                print("Time for detection: ", (time.time_ns() - start) / 1e9)
                continue

            if self.generate_candidates:
                start = time.time_ns()
                if self.cfg.frames_creation.use_gt_masks:
                    self.precompute_candidates_gt()
                else:
                    self.precompute_candidates_waymoc()
                print("Time for candidates: ", (time.time_ns() - start) / 1e9)
                continue

            if self.cfg.output.output_txt:
                if not self.cfg.general.supress_debug_prints:
                    print("Checking for opened file")
                if os.path.isfile(self.cfg.paths.labels_path + self.file_name + '.txt'):
                    continue

            if self.generate_merged_frames_only:
                start = time.time_ns()
                self.load_and_prepare_lidar_scan_from_multiple_waymoc(self.file_name, self.img, save=True)
                print("Time for merged frames: ", (time.time_ns() - start) / 1e9)
                continue
            elif self.generate_transformations_only:
                if os.path.isfile(self.cfg.paths.merged_frames_path + "/transformations/" + self.file_name + '.npy'):
                    continue
                start = time.time_ns()
                self.calculate_transformations_waymoc(self.cfg.frames_creation.nscans_transformation_range,
                                               self.cfg.frames_creation.nscans_transformation_range, save=True)
                print("Time for transformations: ", (time.time_ns() - start) / 1e9)
                continue
            elif self.load_merged_frames:
                success = self.load_merged_frames_from_files_KITTI_waymoc()
                if not success:
                    continue
            else:
                self.load_and_prepare_lidar_scan_from_multiple_waymoc(self.file_name, self.img, save=False)
            start = time.time_ns()
            if self.do_optim:
                if not self.cfg.general.supress_debug_prints:
                    print("Optimizing over: ", len(self.cars), " cars")
                for car_idx in range(len(self.cars)):
                    if not self.est_location_and_downsample(self.cars[car_idx]):
                        continue
                    if self.cfg.optimization.robust_optimization:
                        self.cars[car_idx] = self.optimize_car_robust(self.cars[car_idx])
                    else:
                        self.cars[car_idx] = self.optimize_car(self.cars[car_idx])

            if not self.cfg.general.supress_debug_prints:
                print("Doing NMS")

            self.cars, to_be_reopt = self.non_maxima_surpression(self.cars)
            if self.cfg.optimization.nms_merge_and_reopt:
                if not self.cfg.general.supress_debug_prints:
                    print("Doing NMS merge and reopt")
                    print("Optimizing over: ", len(to_be_reopt), " cars")
                for index in to_be_reopt:
                    if not self.est_location_and_downsample(self.cars[index]):
                        continue
                    self.cars[index] = self.optimize_car(self.cars[index])

            if self.cfg.scale_detector.use_scale_detector and self.do_optim:
                if not self.cfg.general.supress_debug_prints:
                    print("Aggregating points for scale detector")
                self.cars = self.extract_lidar_data_from_bbox_tracker(self.cars)
                if self.cfg.custom_dataset.create_custom_dataset:
                    self.create_custom_dataset_from_cars()
                else:
                    for i in range(len(self.cars)):
                        if not self.cfg.general.supress_debug_prints:
                            print("Optimizing scale: ", i, " from ", len(self.cars))
                        self.cars[i] = self.optimize_car_scale(self.cars[i])

                    if self.cfg.scale_detector.use_bbox_reducer:
                        self.bbox_reducer_tracked(self.cars)
            print("Time for optimization: ", (time.time_ns() - start) / 1e9)
            if self.cfg.output.output_txt:
                self.writetxt_cars(self.cars)
            if self.cfg.output.save_optimized_cars:
                self.save_optimized_cars_all(self.cars)

            if self.cfg.visualization.show_3D_scan:
                self.visualize_3D_waymoc(self.cars)
            if self.cfg.visualization.show_image:
                self.show_image(self.img)

            if not self.cfg.general.supress_debug_prints:
                print("File name: ", self.file_name)
                print("Pics_done: ", self.pic_index)


    def main_custom(self, argv):
        self.args = argv

        self.load_lidar_templatesv2()

        while True:
            self.load_custom_dataset()

            if self.do_optim:
                for car_idx in range(len(self.cars)):
                    if not self.est_location_and_downsample(self.cars[car_idx]):
                        continue

                    self.cars[car_idx] = self.optimize_car(self.cars[car_idx])

            if self.do_optim_scale:
                scale_cars = copy.deepcopy(self.cars)
                scale_cars = self.convert_optim_to_scale(scale_cars)
            else:
                scale_cars = None

            if self.do_optim_scale:
                for car_idx in range(len(scale_cars)):
                    if not self.est_location_and_downsample(scale_cars[car_idx]):
                        continue
                    scale_cars[car_idx] = self.optimize_car_scale(scale_cars[car_idx])

            if not self.cfg.general.supress_debug_prints:
                print("Doing NMS")

            self.custom_compute_iou(self.cars)
            self.custom_compute_iou(scale_cars)

            if self.cfg.visualization.show_3D_scan:
                self.visualize_3D_custom_dataset(self.cars, scale_cars)

            if not self.cfg.general.supress_debug_prints:
                print("File name: ", self.file_name)
                print("Pics_done: ", self.pic_index)

    def prepare_frame(self):
        map_data_cur = self.mapping_data[int(self.random_indexes[int(self.file_name)]) - 1]
        self.kitti_data = pykitti.raw(self.cfg.paths.kitti_sequences_path, map_data_cur[0], map_data_cur[1].split("_")[-2])
        self.path_to_folder = os.path.join(self.cfg.paths.kitti_sequences_path, map_data_cur[0], map_data_cur[1]) + '/'
        if self.cfg.frames_creation.use_pseudo_lidar:
            self.pseudo_lidar_folder = os.path.join(self.cfg.paths.pseudo_lidar_base_path, map_data_cur[0], map_data_cur[1], 'pcds') + '/'
        return map_data_cur

    def load_det2_and_sam(self):
        if not self.load_merged_frames:
            self.load_and_init_detectron_lazy()
            if self.cfg.frames_creation.use_SAM:
                self.load_and_init_SAM() #TODO repair this

    def limit_sequences(self):
        seg_num_max = len(self.random_indexes)
        if self.args.seq_start != -1:
            seg_num_min = self.args.seq_start
        else:
            seg_num_min = self.cfg.general.seq_start
        if self.args.seq_end != -1:
            if self.args.seq_end < len(self.random_indexes):
                seg_num_max = self.args.seq_end
        else:
            if self.cfg.general.seq_end < len(self.random_indexes):
                seg_num_max = self.cfg.general.seq_end

        return seg_num_min, seg_num_max

    def limit_sequences_all(self):
        seg_num_max = len(self.index_of_all_imgs)
        if self.args.seq_start != -1:
            seg_num_min = self.args.seq_start
        else:
            seg_num_min = self.cfg.general.seq_start
        if self.args.seq_end != -1:
            if self.args.seq_end < len(self.index_of_all_imgs):
                seg_num_max = self.args.seq_end
        else:
            if self.cfg.general.seq_end < len(self.index_of_all_imgs):
                seg_num_max = self.cfg.general.seq_end

        return seg_num_min, seg_num_max

    def limit_sequences_all_random(self):
        seg_num_max = len(self.index_of_all_imgs)

        # Set the fixed seed
        np.random.seed(360)

        indexes = np.arange(0, seg_num_max)
        # Shuffle the array
        np.random.shuffle(indexes)

        return indexes[self.args.seq_start:self.args.seq_end]

    def perform_stitching_and_trackingV2(self):
        self.stitched_imgs = []
        self.stitched_imgs, self.homos_all = self.perform_img_stitching()

        if self.generate_homographies_only:
            return False

        if not self.cfg.general.supress_debug_prints:
            print("Images stitched")
        if self.generate_raw_masks_or_tracking:
            pred_masks, _, mask_idxs = self.perform_tracking(self.stitched_imgs, self.homos_all)
            if not self.cfg.general.supress_debug_prints:
                print("Car tracking done")

            self.save_det_and_trackingV2(pred_masks, mask_idxs)
            return False
        else:
            self.load_det_and_trackingV2()
            return True

    def perform_mask_extraction(self):
        self.prec_detectron_output = self.precompute_detectron_waymo()
        if self.generate_raw_masks_or_tracking:
            return False
        self.precompute_standing_car_candidates_waymo()
        return True

    def check_for_optim_done(self):
        if not self.cfg.general.supress_debug_prints:
            print("Checking for opened file")
        if not os.path.isdir(self.cfg.paths.labels_path):
            os.mkdir(self.cfg.paths.labels_path)
        if not os.path.isdir(self.cfg.paths.labels_path + self.file_name):
            os.mkdir(self.cfg.paths.labels_path + self.file_name)
        if os.path.isfile(self.cfg.paths.labels_path + self.file_name + "/" + str(self.pic_index) + '.txt'):
            return True
        else:
            return False

    def load_idx_to_opt(self):
        if self.cfg.dataset.dataset_stride > 1:
            idx_dict = {}
            #Open a txt file, read all lines, separate by ;
            with open(self.cfg.paths.waymo_infos_path, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    elements = line.strip().split(';')
                    segment_name = elements.pop(0)  # remove the first element
                    int_elements = [int(e) for e in elements if e]  # convert the rest to integers
                    idx_dict[str(segment_name)] = int_elements

            return idx_dict
        else:
            return None

    def save_det_and_trackingV2(self, pred_masks, mask_idxs):
        det_and_tracked_dir = os.path.join(self.cfg.paths.det_and_tracked_path, self.file_name)
        if not os.path.isdir(det_and_tracked_dir):
            os.makedirs(det_and_tracked_dir)

        for idx, mask in enumerate(pred_masks):
            file_path = os.path.join(det_and_tracked_dir, f"mask_{idx}.zstd")
            compressed_arr = zstd.compress(pickle.dumps(mask, pickle.HIGHEST_PROTOCOL))
            with open(file_path, 'wb') as f:
                f.write(compressed_arr)

        for idx, info in enumerate(mask_idxs):
            file_path = os.path.join(det_and_tracked_dir, f"info_{idx}.zstd")
            compressed_arr = zstd.compress(pickle.dumps(info, pickle.HIGHEST_PROTOCOL))
            with open(file_path, 'wb') as f:
                f.write(compressed_arr)

    def load_det_and_trackingV2(self):
        pred_masks = []
        extracted_info = []

        det_and_tracked_dir = os.path.join(self.cfg.paths.det_and_tracked_path, self.file_name)
        idx = 0
        while True:
            mask_path = os.path.join(det_and_tracked_dir, f"mask_{idx}.zstd")
            if os.path.exists(mask_path):
                with open(mask_path, 'rb') as f:
                    decompressed = zstd.decompress(f.read())
                    mask = pickle.loads(decompressed)
                    pred_masks.append(mask)
                idx += 1
            else:
                break

        idx = 0
        while True:
            info_path = os.path.join(det_and_tracked_dir, f"info_{idx}.zstd")
            if os.path.exists(info_path):
                with open(info_path, 'rb') as f:
                    decompressed = zstd.decompress(f.read())
                    info = pickle.loads(decompressed)
                    extracted_info.append(info)
                idx += 1
            else:
                break

        extracted_lidar_points, extracted_lidar_positions, extracted_masks = self.extract_lidar_features(pred_masks, extracted_info)

        self.compressed_detandtracked = []
        for lidar, loc, info, mask in zip(extracted_lidar_points, extracted_lidar_positions, extracted_info, extracted_masks):
            to_save = DetAndTracking(lidar, loc, info, mask)
            compressed_arr = zstd.compress(pickle.dumps(to_save, pickle.HIGHEST_PROTOCOL))
            self.compressed_detandtracked.append(compressed_arr)

    def check_for_merging_done(self, dict_idx_to_opt):
        missing = False
        for pic_index, data in enumerate(self.waymo_data):
            if self.cfg.dataset.dataset_stride > 1:
                segment_name = self.random_indexes[self.segment_index].split('.')[0]
                tmp_idx_arr = dict_idx_to_opt[segment_name]
                if pic_index not in tmp_idx_arr:
                    continue

            if self.cfg.frames_creation.use_growing_for_point_extraction:
                file_path = os.path.join(self.cfg.paths.cars_2d_track_growing_path, self.file_name, f"{pic_index}.zstd")
                if not os.path.isfile(file_path):
                    missing = True
            else:
                file_path = os.path.join(self.cfg.paths.cars_2d_track_path, self.file_name, f"{pic_index}.zstd")
                if not os.path.isfile(file_path):
                    missing = True

        return not missing

    def save_optimized_cars(self, cars):
        new_cars = []
        for car in cars:
            car.bbox = None
            if car.lidar is not None:
                new_cars.append(car)
        cars = new_cars
        
        os.makedirs(self.cfg.paths.optimized_cars_path, exist_ok=True)
        
        car_path = os.path.join(self.cfg.paths.optimized_cars_path, f"{self.file_name}.zstd")
        compressed_arr = zstd.compress(dill.dumps(cars, dill.HIGHEST_PROTOCOL))
        with open(car_path, 'wb') as f:
            f.write(compressed_arr)

        # Save also calib info
        calib = [self.kitti_data.calib.T_cam2_velo, self.kitti_data.calib.P_rect_00]
        compressed_arr = zstd.compress(dill.dumps(calib, dill.HIGHEST_PROTOCOL))
        calib_path = os.path.join(self.cfg.paths.optimized_cars_path, f"{self.file_name}_calib.zstd")
        with open(calib_path, 'wb') as f:
            f.write(compressed_arr)

    def save_optimized_cars_all(self, cars):
        new_cars = []
        for car in cars:
            car.bbox = None
            if car.lidar is not None:
                new_cars.append(car)
        cars = new_cars
        
        os.makedirs(self.cfg.paths.optimized_cars_path, exist_ok=True)
        
        car_path = os.path.join(self.cfg.paths.optimized_cars_path, f"{self.file_name}.zstd")
        compressed_arr = zstd.compress(dill.dumps(cars, dill.HIGHEST_PROTOCOL))
        with open(car_path, 'wb') as f:
            f.write(compressed_arr)

        # Save also calib info
        calib = [np.eye(4), self.P2_rect]
        compressed_arr = zstd.compress(dill.dumps(calib, dill.HIGHEST_PROTOCOL))
        calib_path = os.path.join(self.cfg.paths.optimized_cars_path, f"{self.file_name}_calib.zstd")
        with open(calib_path, 'wb') as f:
            f.write(compressed_arr)

    def build_index_of_imgs(self):
        all_imgs_index = []
        for folder in sorted(os.listdir(self.cfg.paths.all_dataset_path)):
            if not os.path.isdir(os.path.join(self.cfg.paths.all_dataset_path, folder)):
                continue

            tmp_folder_path = os.path.join(self.cfg.paths.all_dataset_path, folder)
            path_to_imgs = os.path.join(tmp_folder_path, 'image_00/data_rect/')
            image_paths = sorted(glob.glob(os.path.join(path_to_imgs, '*.png')))

            if len(image_paths) == 0:
                continue
            for image_path in tqdm(image_paths, desc="Processing Images"):
                file_name = os.path.basename(image_path)
                file_name = file_name.split('.')[0]
                all_imgs_index.append((folder, file_name))

        return all_imgs_index

    def build_index_of_imgs_waymoc(self):
        all_imgs_index = []
        training_path = os.path.join(self.cfg.paths.all_dataset_path, self.cfg.paths.waymoc_training_subfolder)
        for folder in sorted(os.listdir(training_path)):
            folder_path = os.path.join(training_path, folder)
            if not os.path.isdir(folder_path):
                continue

            path_to_imgs = os.path.join(folder_path, 'image_2/')
            image_paths = sorted(glob.glob(os.path.join(path_to_imgs, '*.png')))

            if len(image_paths) == 0:
                continue
            for image_path in tqdm(image_paths, desc="Processing Images"):
                file_name = os.path.basename(image_path)
                file_name = file_name.split('.')[0]
                all_imgs_index.append((folder, file_name))

        return all_imgs_index
