import numpy as np
from anno_V3 import AutoLabel3D
import torch
import copy
import point_cloud_utils as pcu

from kornia.geometry.transform import warp_perspective

class Filtering(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)

    def est_location_and_downsample(self, car):
        self.curr_lidar = car.lidar

        if self.curr_lidar is None:
            return False
        else:
            filtered_lidar = self.curr_lidar.T

            if filtered_lidar.shape[0] < self.cfg.filtering.lidar_threshold_during_optim and not car.moving:
                return False

            x_mean = np.median(filtered_lidar[:, 0])
            y_mean = np.median(filtered_lidar[:, 1])
            z_mean = np.median(filtered_lidar[:, 2])

            if not self.cfg.general.supress_debug_prints:
                print(filtered_lidar.shape)
            if filtered_lidar.shape[0] > self.cfg.filtering.lidar_threshold_downsample:
                if self.cfg.downsampling.type == 'voxel':
                    filtered_lidar = self.downsample(filtered_lidar[:, :3])

                    padding = np.ones((filtered_lidar.shape[0], 3))

                    filtered_lidar = np.concatenate((filtered_lidar, padding), axis=1)

                elif self.cfg.downsampling.type == 'random':
                    filtered_lidar = self.downsample_random(filtered_lidar[:, :3])

                    padding = np.ones((filtered_lidar.shape[0], 3))

                    filtered_lidar = np.concatenate((filtered_lidar, padding), axis=1)

                elif self.cfg.downsampling.type == 'both':
                    tmp1 = self.downsample_random(filtered_lidar[:, :3])
                    tmp2 = self.downsample(filtered_lidar[:, :3])

                    filtered_lidar = np.concatenate((tmp1, tmp2), axis=0)

                    padding = np.ones((filtered_lidar.shape[0], 3))

                    filtered_lidar = np.concatenate((filtered_lidar, padding), axis=1)

            if not self.cfg.general.supress_debug_prints:
                print(filtered_lidar.shape)
            self.x_mean_lidar = x_mean
            self.y_mean_lidar = y_mean
            self.z_mean_lidar = z_mean
            self.filtered_lidar = filtered_lidar[:, 0:3]

        if car.moving and len(car.locations) < 3:
            return False
        else:
            return True

    def downsample_lidar(self, lidar):
        if self.cfg.downsampling.type == 'voxel':
            filtered_lidar = self.downsample(lidar[:, :3])

            padding = np.ones((filtered_lidar.shape[0], 3))

            filtered_lidar = np.concatenate((filtered_lidar, padding), axis=1)

        elif self.cfg.downsampling.type == 'random':
            filtered_lidar = self.downsample_random(lidar[:, :3])

            padding = np.ones((filtered_lidar.shape[0], 3))

            filtered_lidar = np.concatenate((filtered_lidar, padding), axis=1)

        elif self.cfg.downsampling.type == 'both':
            tmp1 = self.downsample_random(lidar[:, :3])
            tmp2 = self.downsample(lidar[:, :3])

            filtered_lidar = np.concatenate((tmp1, tmp2), axis=0)

            padding = np.ones((filtered_lidar.shape[0], 3))

            filtered_lidar = np.concatenate((filtered_lidar, padding), axis=1)

        else:
            raise ValueError('Unknown downsampling type')

        return filtered_lidar

    def run_detectron_batch(self, img=None, save=True):
        # Generate the ouptut, can be modified for multiple imgs
        with torch.inference_mode():
            tmp_arr = []
            for i in range(len(img)):
                tmp_arr.append({'image': torch.from_numpy(img[i])})
            outputs = self.model(tmp_arr)
        out_data_arr = []
        for i in range(len(outputs)):
            out_data = outputs[i]["instances"]
            out_data = out_data.to("cpu")
            out_data_arr.append(out_data[:])
        # We do not care about detections with low score -> probably occluded
        return out_data_arr

    def run_SAM_batch(self, img=None, save=True):
        # Generate the ouptut, can be modified for multiple imgs
        with torch.inference_mode():
            tmp_arr = []
            for i in range(len(img)):
                tmp_arr.append({'image': torch.from_numpy(img[i])})
            outputs = self.model(tmp_arr)
        out_data_arr = []
        ret_data_arr = []
        for i in range(len(outputs)):
            out_data = outputs[i]["instances"].to("cpu")
            out_data_arr.append(out_data[:])
        # We do not care about detections with low score -> probably occluded or actually not cars
        for i in range(len(out_data_arr)):
            out_det = copy.deepcopy(out_data_arr[i])
            #Move the axis of the img back as it should be loaded to the SAM
            cur_img = np.moveaxis(img[i], 0, -1)
            #Compute the embedding in the picture
            self.sam_predictor.set_image(cur_img)

            for z in range(len(out_det)):
                #It has score big enough and also it is a car then we want to adjust the mask According to SAM
                if out_det.scores[z] > self.cfg.filtering.score_detectron_thresh and out_det.pred_classes[z] == 2:
                    box_instance = out_det.pred_boxes[z].tensor[0].cpu()
                    box = np.array([box_instance[0], box_instance[1], box_instance[2], box_instance[3]]).astype(int)
                    masks, scores, logits = self.sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box[None, :],
                        multimask_output=False,
                    )

                    out_det.pred_masks[z] = torch.BoolTensor(masks[0])

            ret_data_arr.append(out_det)

        return ret_data_arr


    def downsample(self, filtered_lidar):
        filtered_lidar = pcu.downsample_point_cloud_on_voxel_grid([self.cfg.downsampling.voxel_size, self.cfg.downsampling.voxel_size, self.cfg.downsampling.voxel_size], filtered_lidar)
        return filtered_lidar

    def downsample_random(self, filtered_lidar, number=1000):
        size = filtered_lidar.shape[0]
        if size > number:
            idxs = np.random.choice(np.arange(size), number, replace=False)
            downsampled = filtered_lidar[idxs, :]
            return downsampled
        else:
            return filtered_lidar

    def extract_lidar_features(self, masks, car_info):
        extracted_lidar = []
        extracted_lidar_locations = []
        extracted_masks = []
        for car_idx, car in enumerate(car_info):
            if not self.cfg.general.supress_debug_prints:
                print("Extracting car ", car_idx, " from ", len(car_info))
            extracted_lidar_car = []
            extracted_lidar_location_car = []
            extracted_masks_car = []
            for frame_idx, frame in enumerate(car):
                if frame is None:
                    extracted_lidar_car.append(None)
                    extracted_lidar_location_car.append(None)
                    extracted_masks_car.append(None)
                    continue
                #Detectron2 mask
                elif frame[0] == 0:
                    mask = masks[frame[1]][frame[2]][frame[3]]
                #SAM mask
                elif frame[0] == 1:
                    #TODO if needed
                    pass

                cur_lidar = self.waymo_lidar[frame[1]]

                #Inverse of the mask
                if frame[2] == 0:
                    if self.cfg.general.device == 'cpu':
                        img0, img1 = self.inverse_of_mask_img01(mask.to(torch.float32), self.homos_all[frame[2]])
                    else:
                        img0, img1 = self.inverse_of_mask_img01(mask.cuda().to(torch.float32), self.homos_all[frame[2]])
                    img0 = img0.numpy()[-886:, :]
                    img1 = img1.numpy()

                    cur_lidar_img0 = cur_lidar[cur_lidar[:, 3] == 4].T
                    cur_lidar_img1 = cur_lidar[cur_lidar[:, 3] == 2].T

                    # Now, get indexes of the points which project into the mask
                    img0_idx = np.argwhere(img0.T[cur_lidar_img0[4, :].astype(int), cur_lidar_img0[5, :].astype(int)])
                    img1_idx = np.argwhere(img1.T[cur_lidar_img1[4, :].astype(int), cur_lidar_img1[5, :].astype(int)])

                    if img0_idx.shape[0] > 0 and img1_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar_img0 = self.perform_growing(img0.T, 4, cur_lidar)
                            filtered_lidar_img1 = self.perform_growing(img1.T, 2, cur_lidar)

                            if filtered_lidar_img0 is None or len(filtered_lidar_img0) == 0:
                                filtered_lidar_img0 = np.array([cur_lidar_img0[0, img0_idx], cur_lidar_img0[1, img0_idx], cur_lidar_img0[2, img0_idx]]).transpose()[0]
                            if filtered_lidar_img1 is None or len(filtered_lidar_img1) == 0:
                                filtered_lidar_img1 = np.array([cur_lidar_img1[0, img1_idx], cur_lidar_img1[1, img1_idx], cur_lidar_img1[2, img1_idx]]).transpose()[0]
                        else:
                            filtered_lidar_img0 = np.array([cur_lidar_img0[0, img0_idx], cur_lidar_img0[1, img0_idx], cur_lidar_img0[2, img0_idx]]).transpose()[0]
                            filtered_lidar_img1 = np.array([cur_lidar_img1[0, img1_idx], cur_lidar_img1[1, img1_idx], cur_lidar_img1[2, img1_idx]]).transpose()[0]

                        filtered_lidar = np.concatenate((filtered_lidar_img0, filtered_lidar_img1), axis=0)
                    elif img0_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar = self.perform_growing(img0.T, 4, cur_lidar)
                            if filtered_lidar is None or len(filtered_lidar) == 0:
                                filtered_lidar = np.array([cur_lidar_img0[0, img0_idx], cur_lidar_img0[1, img0_idx], cur_lidar_img0[2, img0_idx]]).transpose()[0]
                        else:
                            filtered_lidar = np.array([cur_lidar_img0[0, img0_idx], cur_lidar_img0[1, img0_idx], cur_lidar_img0[2, img0_idx]]).transpose()[0]
                    elif img1_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar = self.perform_growing(img1.T, 2, cur_lidar)
                            if filtered_lidar is None or len(filtered_lidar) == 0:
                                filtered_lidar = np.array([cur_lidar_img1[0, img1_idx], cur_lidar_img1[1, img1_idx], cur_lidar_img1[2, img1_idx]]).transpose()[0]
                        else:
                            filtered_lidar = np.array([cur_lidar_img1[0, img1_idx], cur_lidar_img1[1, img1_idx], cur_lidar_img1[2, img1_idx]]).transpose()[0]
                    else:
                        filtered_lidar = None

                elif frame[2] == 1:
                    if self.cfg.general.device == 'cpu':
                        img1, img2 = self.inverse_of_mask_img01(mask.to(torch.float32), self.homos_all[frame[2]])
                    else:
                        img1, img2 = self.inverse_of_mask_img01(mask.cuda().to(torch.float32), self.homos_all[frame[2]])
                    img1 = img1.numpy()
                    img2 = img2.numpy()

                    cur_lidar_img1 = cur_lidar[cur_lidar[:, 3] == 2].T
                    cur_lidar_img2 = cur_lidar[cur_lidar[:, 3] == 1].T

                    # Now, get indexes of the points which project into the mask
                    img1_idx = np.argwhere(img1.T[cur_lidar_img1[4, :].astype(int), cur_lidar_img1[5, :].astype(int)])
                    img2_idx = np.argwhere(img2.T[cur_lidar_img2[4, :].astype(int), cur_lidar_img2[5, :].astype(int)])

                    if img1_idx.shape[0] > 0 and img2_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar_img1 = self.perform_growing(img1.T, 2, cur_lidar)
                            filtered_lidar_img2 = self.perform_growing(img2.T, 1, cur_lidar)

                            if filtered_lidar_img1 is None or len(filtered_lidar_img1) == 0:
                                filtered_lidar_img1 = np.array([cur_lidar_img1[0, img1_idx], cur_lidar_img1[1, img1_idx], cur_lidar_img1[2, img1_idx]]).transpose()[0]
                            if filtered_lidar_img2 is None or len(filtered_lidar_img2) == 0:
                                filtered_lidar_img2 = np.array([cur_lidar_img2[0, img2_idx], cur_lidar_img2[1, img2_idx], cur_lidar_img2[2, img2_idx]]).transpose()[0]
                        else:
                            filtered_lidar_img1 = np.array([cur_lidar_img1[0, img1_idx], cur_lidar_img1[1, img1_idx], cur_lidar_img1[2, img1_idx]]).transpose()[0]
                            filtered_lidar_img2 = np.array([cur_lidar_img2[0, img2_idx], cur_lidar_img2[1, img2_idx], cur_lidar_img2[2, img2_idx]]).transpose()[0]

                        filtered_lidar = np.concatenate((filtered_lidar_img1, filtered_lidar_img2), axis=0)
                    elif img1_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar = self.perform_growing(img1.T, 2, cur_lidar)
                            if filtered_lidar is None or len(filtered_lidar) == 0:
                                filtered_lidar = np.array([cur_lidar_img1[0, img1_idx], cur_lidar_img1[1, img1_idx], cur_lidar_img1[2, img1_idx]]).transpose()[0]
                        else:
                            filtered_lidar = np.array([cur_lidar_img1[0, img1_idx], cur_lidar_img1[1, img1_idx], cur_lidar_img1[2, img1_idx]]).transpose()[0]
                    elif img2_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar = self.perform_growing(img2.T, 1, cur_lidar)
                            if filtered_lidar is None or len(filtered_lidar) == 0:
                                filtered_lidar = np.array([cur_lidar_img2[0, img2_idx], cur_lidar_img2[1, img2_idx],cur_lidar_img2[2, img2_idx]]).transpose()[0]
                        else:
                            filtered_lidar = np.array([cur_lidar_img2[0, img2_idx], cur_lidar_img2[1, img2_idx], cur_lidar_img2[2, img2_idx]]).transpose()[0]
                    else:
                        filtered_lidar = None

                elif frame[2] == 2:
                    if self.cfg.general.device == 'cpu':
                        img3, img2 = self.inverse_of_mask_img23(mask.to(torch.float32), self.homos_all[frame[2]])
                    else:
                        img3, img2 = self.inverse_of_mask_img23(mask.cuda().to(torch.float32), self.homos_all[frame[2]])
                    img2 = img2.numpy()
                    img3 = img3.numpy()

                    cur_lidar_img2 = cur_lidar[cur_lidar[:, 3] == 1].T
                    cur_lidar_img3 = cur_lidar[cur_lidar[:, 3] == 3].T

                    # Now, get indexes of the points which project into the mask
                    img2_idx = np.argwhere(img2.T[cur_lidar_img2[4, :].astype(int), cur_lidar_img2[5, :].astype(int)])
                    img3_idx = np.argwhere(img3.T[cur_lidar_img3[4, :].astype(int), cur_lidar_img3[5, :].astype(int)])

                    if img2_idx.shape[0] > 0 and img3_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar_img2 = self.perform_growing(img2.T, 1, cur_lidar)
                            filtered_lidar_img3 = self.perform_growing(img3.T, 3, cur_lidar)

                            if filtered_lidar_img2 is None or len(filtered_lidar_img2) == 0:
                                filtered_lidar_img2 = np.array([cur_lidar_img2[0, img2_idx], cur_lidar_img2[1, img2_idx], cur_lidar_img2[2, img2_idx]]).transpose()[0]
                            if filtered_lidar_img3 is None or len(filtered_lidar_img3) == 0:
                                filtered_lidar_img3 = np.array([cur_lidar_img3[0, img3_idx], cur_lidar_img3[1, img3_idx], cur_lidar_img3[2, img3_idx]]).transpose()[0]
                        else:
                            filtered_lidar_img2 = np.array([cur_lidar_img2[0, img2_idx], cur_lidar_img2[1, img2_idx], cur_lidar_img2[2, img2_idx]]).transpose()[0]
                            filtered_lidar_img3 = np.array([cur_lidar_img3[0, img3_idx], cur_lidar_img3[1, img3_idx], cur_lidar_img3[2, img3_idx]]).transpose()[0]

                        filtered_lidar = np.concatenate((filtered_lidar_img2, filtered_lidar_img3), axis=0)
                    elif img2_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar = self.perform_growing(img2.T, 1, cur_lidar)
                            if filtered_lidar is None or len(filtered_lidar) == 0:
                                filtered_lidar = np.array([cur_lidar_img2[0, img2_idx], cur_lidar_img2[1, img2_idx], cur_lidar_img2[2, img2_idx]]).transpose()[0]
                        else:
                            filtered_lidar = np.array([cur_lidar_img2[0, img2_idx], cur_lidar_img2[1, img2_idx], cur_lidar_img2[2, img2_idx]]).transpose()[0]
                    elif img3_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar = self.perform_growing(img3.T, 3, cur_lidar)
                            if filtered_lidar is None or len(filtered_lidar) == 0:
                                filtered_lidar = np.array([cur_lidar_img3[0, img3_idx], cur_lidar_img3[1, img3_idx], cur_lidar_img3[2, img3_idx]]).transpose()[0]
                        else:
                            filtered_lidar = np.array([cur_lidar_img3[0, img3_idx], cur_lidar_img3[1, img3_idx], cur_lidar_img3[2, img3_idx]]).transpose()[0]
                    else:
                        filtered_lidar = None

                else:
                    if self.cfg.general.device == 'cpu':
                        img4, img3 = self.inverse_of_mask_img23(mask.to(torch.float32), self.homos_all[frame[2]])
                    else:
                        img4, img3 = self.inverse_of_mask_img23(mask.cuda().to(torch.float32), self.homos_all[frame[2]])
                    img3 = img3.numpy()
                    img4 = img4.numpy()[-886:, :]

                    cur_lidar_img3 = cur_lidar[cur_lidar[:, 3] == 3].T
                    cur_lidar_img4 = cur_lidar[cur_lidar[:, 3] == 5].T

                    # Now, get indexes of the points which project into the mask
                    img3_idx = np.argwhere(img3.T[cur_lidar_img3[4, :].astype(int), cur_lidar_img3[5, :].astype(int)])
                    img4_idx = np.argwhere(img4.T[cur_lidar_img4[4, :].astype(int), cur_lidar_img4[5, :].astype(int)])

                    if img3_idx.shape[0] > 0 and img4_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar_img3 = self.perform_growing(img3.T, 3, cur_lidar)
                            filtered_lidar_img4 = self.perform_growing(img4.T, 5, cur_lidar)

                            if filtered_lidar_img3 is None or len(filtered_lidar_img3) == 0:
                                filtered_lidar_img3 = np.array([cur_lidar_img3[0, img3_idx], cur_lidar_img3[1, img3_idx],cur_lidar_img3[2, img3_idx]]).transpose()[0]
                            if filtered_lidar_img4 is None or len(filtered_lidar_img4) == 0:
                                filtered_lidar_img4 = np.array([cur_lidar_img4[0, img4_idx], cur_lidar_img4[1, img4_idx],cur_lidar_img4[2, img4_idx]]).transpose()[0]
                        else:
                            filtered_lidar_img3 = np.array([cur_lidar_img3[0, img3_idx], cur_lidar_img3[1, img3_idx], cur_lidar_img3[2, img3_idx]]).transpose()[0]
                            filtered_lidar_img4 = np.array([cur_lidar_img4[0, img4_idx], cur_lidar_img4[1, img4_idx], cur_lidar_img4[2, img4_idx]]).transpose()[0]

                        filtered_lidar = np.concatenate((filtered_lidar_img3, filtered_lidar_img4), axis=0)
                    elif img3_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar = self.perform_growing(img3.T, 3, cur_lidar)
                            if filtered_lidar is None or len(filtered_lidar) == 0:
                                filtered_lidar = np.array([cur_lidar_img3[0, img3_idx], cur_lidar_img3[1, img3_idx],cur_lidar_img3[2, img3_idx]]).transpose()[0]
                        else:
                            filtered_lidar = np.array([cur_lidar_img3[0, img3_idx], cur_lidar_img3[1, img3_idx], cur_lidar_img3[2, img3_idx]]).transpose()[0]
                    elif img4_idx.shape[0] > 0:
                        if self.cfg.frames_creation.use_growing_for_point_extraction:
                            filtered_lidar = self.perform_growing(img4.T, 5, cur_lidar)
                            if filtered_lidar is None or len(filtered_lidar) == 0:
                                filtered_lidar = np.array([cur_lidar_img4[0, img4_idx], cur_lidar_img4[1, img4_idx],cur_lidar_img4[2, img4_idx]]).transpose()[0]
                        else:
                            filtered_lidar = np.array([cur_lidar_img4[0, img4_idx], cur_lidar_img4[1, img4_idx], cur_lidar_img4[2, img4_idx]]).transpose()[0]
                    else:
                        filtered_lidar = None

                if filtered_lidar is not None:
                    extracted_masks_car.append(mask)

                    x_mean = np.median(filtered_lidar[:, 0])
                    y_mean = np.median(filtered_lidar[:, 1])
                    z_mean = np.median(filtered_lidar[:, 2])

                    extracted_lidar_location_car.append(np.array([x_mean, y_mean, z_mean]))

                    # Filter by circle
                    dist_from_mean = np.sqrt(
                        (x_mean - filtered_lidar[:, 0]) ** 2 + (y_mean - filtered_lidar[:, 1]) ** 2)

                    indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

                    filtered_lidar = np.array([filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

                    extracted_lidar_car.append(filtered_lidar)
                    #TODO Return also masks and corresponding img indexes.
                else:
                    extracted_lidar_car.append(None)
                    extracted_lidar_location_car.append(None)
                    extracted_masks_car.append(None)

            extracted_lidar.append(extracted_lidar_car)
            extracted_lidar_locations.append(extracted_lidar_location_car)
            extracted_masks.append(extracted_masks_car)

        return extracted_lidar, extracted_lidar_locations, extracted_masks

    def inverse_of_mask_img01(self, mask, homo):
        mask_non_deformed = mask[(self.cfg.image_stitching.height_pxl_pad//2):1280+(self.cfg.image_stitching.height_pxl_pad//2), -1920:]
        mask_deformed = mask.clone()
        #Just disable the non deformed mask
        #mask_deformed[:, -1920:] = 0
        #Now lets do reverse homography.
        mask_deformed = warp_perspective(mask_deformed.unsqueeze(0).unsqueeze(0), torch.linalg.inv(homo), (mask_deformed.shape[0], mask_deformed.shape[1]))
        mask_deformed = mask_deformed[0][0][:1280, 0:1920]

        return mask_deformed.to(torch.bool).cpu(), mask_non_deformed.to(torch.bool).cpu()

    def inverse_of_mask_img23(self, mask, homo):
        mask_non_deformed = mask[(self.cfg.image_stitching.height_pxl_pad//2):1280 + (self.cfg.image_stitching.height_pxl_pad//2), :1920]
        mask_deformed = mask.clone()
        #Just disable the non deformad mask
        #mask_deformed[:, :1920] = 0
        #Now lets do reverse homography.
        mask_deformed = warp_perspective(mask_deformed.unsqueeze(0).unsqueeze(0), torch.linalg.inv(homo), (mask_deformed.shape[0], mask_deformed.shape[1]))
        mask_deformed = mask_deformed[0][0][:1280, 0:1920]

        return mask_deformed.to(torch.bool).cpu(), mask_non_deformed.to(torch.bool).cpu()




