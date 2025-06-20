import importlib
import sys
import torch
import numpy as np
from kornia.geometry.transform import warp_perspective
import cv2

from anno_V3 import AutoLabel3D

class Tracker_ODTrack(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)

        if self.generate_raw_masks_or_tracking and self.cfg.frames_creation.tracker_for_merging == '2D':
            odtrack_path = self.cfg.paths.odtrack_path
            sam_path = None  # TODO
            sys.path.append(odtrack_path)

            module = importlib.import_module('lib.test.evaluation')
            Tracker = getattr(module, 'Tracker')

            self.tracker_class = Tracker(self.cfg.tracker_2D.tracker_name, self.cfg.tracker_2D.tracker_model, "video")

            params = self.tracker_class.get_parameters()

            debug_ = getattr(params, 'debug', 0)
            params.debug = debug_

            params.tracker_name = self.tracker_class.name
            params.param_name = self.tracker_class.parameter_name

            self.tracker = self.tracker_class.create_tracker(params)

            #sam = sam_model_registry["vit_h"](checkpoint=sam_path)
            #sam.to(device='cuda')
            #self.predictor = SamPredictor(sam)

    def _build_init_info(self, box):
        return {'init_bbox': box}

    def compute_bounding_box(self, mask):
        # Get the indices of the True values
        indices = torch.nonzero(mask)

        # If there are no True values in the mask, return an empty bounding box
        if indices.numel() == 0:
            return torch.tensor([0, 0, 0, 0])

        # Get the minimum and maximum y and x coordinates
        min_y, min_x = torch.min(indices, dim=0)[0]
        max_y, max_x = torch.max(indices, dim=0)[0]

        # Compute the width and height of the bounding box
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # Return the bounding box [x, y, width, height]
        return torch.tensor([min_x, min_y, width, height])

    def perform_tracking(self, imgs, homos):
        #First, use the detectron2 to get all the masks for all imgs
        if not self.cfg.general.supress_debug_prints:
            print("Computing all masks!")
        torch.cuda.empty_cache()
        pred_masks = self.get_all_masks(imgs, homos)

        torch.cuda.empty_cache()
        #Second, generate arrays for matching the masks to tracked cars and for that purpose generate bboxes from masks
        pred_masks_matching = []
        pred_masks_bboxes = []
        for i in range(len(pred_masks)):
            pred_masks_matching.append([])
            pred_masks_bboxes.append([])
            for z in range(4):
                pred_masks_matching[i].append([])
                pred_masks_bboxes[i].append([])
                for k in range(len(pred_masks[i][z])):
                    pred_masks_matching[i][z].append([])
                    min_bbox = self.compute_bounding_box(pred_masks[i][z][k])
                    pred_masks_bboxes[i][z].append(min_bbox)


        masks_for_car_ids = []
        sam_masks = []
        sam_mask_idx = 0
        #Third, for each mask which has not been matched -> track it and perform matching.
        if not self.cfg.general.supress_debug_prints:
            print("Performing tracking")
        for frame_idx, frame_imgs in enumerate(imgs):
            if not self.cfg.general.supress_debug_prints:
                print("Tracking frame: ", frame_idx, " from ", len(imgs))
            for img_idx, cur_img in enumerate(frame_imgs):
                cur_masks = pred_masks[frame_idx][img_idx]

                for mask_idx, mask in enumerate(cur_masks):
                    #The mask exists and also it has not been matched
                    if len(mask) > 0 and len(pred_masks_matching[frame_idx][img_idx][mask_idx]) == 0:
                        tracked_bboxes, tracked_imgs_idxs = self.perform_tracking_of_single(mask, imgs, frame_idx, img_idx)
                        pred_masks_matching[frame_idx][img_idx][mask_idx].append([frame_idx, img_idx, mask_idx])
                        masks_for_tracked_car = [[0, frame_idx, img_idx, mask_idx]]

                        #Skip the first one because it is the same as the input
                        for track_bbox_idx, track_bbox in enumerate(tracked_bboxes[1:], start=1):
                            candidates = pred_masks_bboxes[frame_idx + track_bbox_idx][tracked_imgs_idxs[track_bbox_idx]]
                            best_iou = -1
                            best_idx = None
                            for candidate_idx, candidate in enumerate(candidates):
                                iou = self.compute_iou(track_bbox, candidate)
                                if iou > best_iou and len(pred_masks_matching[frame_idx + track_bbox_idx][tracked_imgs_idxs[track_bbox_idx]][candidate_idx]) == 0:
                                    best_iou = iou
                                    best_idx = candidate_idx
                            #If we have found a match with a good IOU we add it to the matching
                            if best_iou > 0.25:
                                pred_masks_matching[frame_idx + track_bbox_idx][tracked_imgs_idxs[track_bbox_idx]][best_idx].append([frame_idx, img_idx, mask_idx])
                                masks_for_tracked_car.append([0, frame_idx + track_bbox_idx, tracked_imgs_idxs[track_bbox_idx], best_idx])
                            #If we have not found a match we add the bbox to the list of tracked bboxes so we can use SAM to refine it
                            else:
                                #sam_mask = self.use_sam_for_refine(imgs[frame_idx + track_bbox_idx][tracked_imgs_idxs[track_bbox_idx]], track_bbox)
                                #sam_bbox = self.compute_bounding_box_numpy(sam_mask)
                                matched = True
                                #for candidate_idx, candidate in enumerate(candidates):
                                #    #iou = self.compute_iou(sam_bbox, candidate)
                                #    intersection = self.compute_intersection(sam_bbox, candidate)
                                #    if intersection > 0.25:
                                #        matched = True
                                #        break
                                if not matched:
                                    sam_masks.append(sam_mask)
                                    masks_for_tracked_car.append([1, frame_idx + track_bbox_idx, tracked_imgs_idxs[track_bbox_idx], sam_mask_idx])
                                    sam_mask_idx += 1
                                else:
                                    masks_for_tracked_car.append(None)

                        masks_for_car_ids.append(masks_for_tracked_car)

        if not self.cfg.general.supress_debug_prints:
            print("Saving Visu results")
        '''
        for frame_idx, frame_imgs in enumerate(imgs):
            for img_idx, cur_img in enumerate(frame_imgs):
                if img_idx != 2:
                    continue
                cur_mask = pred_masks[frame_idx][img_idx]
                cur_matching = pred_masks_matching[frame_idx][img_idx]
                cur_img = imgs[frame_idx][img_idx]

                self.visu_masks_with_ids(cur_mask, cur_img, "masks_" + str(frame_idx) + "_" + str(img_idx), cur_matching)
        
        idx_visu = 0
        for masks_in_sequence in masks_for_car_ids:
            self.visu_to_video(pred_masks, sam_masks, masks_in_sequence, str(idx_visu), imgs)
            idx_visu += 1
        '''

        return pred_masks, sam_masks, masks_for_car_ids

    def get_all_masks(self, imgs, homos):
        all_pred_masks = []
        torch.cuda.empty_cache()

        for frame_idx, frame_imgs in enumerate(imgs):
            if not self.cfg.general.supress_debug_prints:
                print("Mask frame: ", frame_idx, " from ", len(imgs))
            tmp_pred_masks = []
            for img_idx, cur_img in enumerate(frame_imgs):
                with torch.inference_mode():
                    outputs = self.model([{'image': cur_img}])

                out_data = outputs[0]["instances"]

                out_data = out_data[out_data.scores > self.cfg.filtering.score_detectron_thresh]

                pred_masks = []

                for i in range(len(out_data)):
                    if out_data.pred_classes[i] == 2 or out_data.pred_classes[i] == 7:  # It is a Car
                        pred_masks.append(out_data.pred_masks[i])

                if img_idx == 0:
                    cur_homo = homos[1]
                elif img_idx == 3:
                    cur_homo = homos[2]
                else:
                    cur_homo = None

                # First, we need to check if the detected cars are not in the overlap
                filtered_masks = self.filter_overlapping_detections(pred_masks, cur_img, cur_homo, img_idx)

                if len(filtered_masks) > 0:
                    tmp_pred_masks.append(torch.stack(filtered_masks).cpu())
                else:
                    tmp_pred_masks.append(torch.tensor([]))

                #self.visu_dete_masks(filtered_masks, cur_img, str(frame_idx) + "_" + str(img_idx))
                #self.visu_dete(out_data, cur_img, str(frame_idx) + "_" + str(img_idx))

            all_pred_masks.append(tmp_pred_masks)

        return all_pred_masks

    def perform_tracking_of_single(self, mask, imgs, frame_start, img_start):
        #First refine all masks with HQSam
        img_ref = imgs[frame_start][img_start]
        img_ref = img_ref.numpy().transpose(1, 2, 0)
        img_ref = np.uint8(img_ref)
        #Empty the cuda because of the detector
        #torch.cuda.empty_cache()

        #print("Get bbox")
        min_bbox = self.compute_bounding_box(mask).cpu().numpy()

        self.tracker.initialize(img_ref, self._build_init_info(min_bbox))
        #print('ODTrack running ...')
        #Now lets track the car

        frame_idx_run = frame_start + 1
        img_idx_run = img_start
        rdy_to_switch = 0
        old_bbox = min_bbox
        old_center = [old_bbox[0] + old_bbox[2] / 2, old_bbox[1] + old_bbox[3] / 2]
        old_center_diff = [0., 0.]
        switched = 0

        tracked_car_bboxes = [list(min_bbox)]
        tracked_car_img_idxs = [img_start]

        while frame_idx_run < len(imgs):
            if switched > 0:
                switched -= 1
            img_cur = imgs[frame_idx_run][img_idx_run]
            img_cur = img_cur.numpy().transpose(1, 2, 0)
            img_cur = np.uint8(img_cur)

            out = self.tracker.track(img_cur)
            pred_bbox = [int(s) for s in out['target_bbox']]

            #self.vizu_sam_bboxes(img_cur, pred_bbox, "TRACK_" + str(frame_idx_run) + "_" + str(img_idx_run) + "_" + str(idx))

            out_shape = (img_cur.shape[-3], img_cur.shape[-2])

            if img_idx_run == 0:
                cur_homo = self.homos_all[1]
            elif img_idx_run == 3:
                cur_homo = self.homos_all[2]
            else:
                cur_homo = None

            img_idx_run, rdy_to_switch, switched = self.check_for_image_switch(pred_bbox, img_idx_run, rdy_to_switch, switched, cur_homo, out_shape)

            if switched == 0:
                iou = self.compute_iou(old_bbox, pred_bbox)
                old_bbox = pred_bbox
                #print("IOU: ", iou)

                new_center = [pred_bbox[0] + pred_bbox[2] / 2, pred_bbox[1] + pred_bbox[3] / 2]
                new_center_diff = [new_center[0] - old_center[0], new_center[1] - old_center[1]]
                if old_center_diff[0] != 0. or old_center_diff[1] != 0.:
                    diff = np.linalg.norm(np.array(new_center_diff) - np.array(old_center_diff))
                    #print(diff)
                else:
                    diff = 0.

                old_center = new_center
                old_center_diff = new_center_diff
                if iou < 0.5 and diff > 100:
                    break
            else:
                old_bbox = pred_bbox
                new_center = [pred_bbox[0] + pred_bbox[2] / 2, pred_bbox[1] + pred_bbox[3] / 2]
                new_center_diff = [new_center[0] - old_center[0], new_center[1] - old_center[1]]
                old_center = new_center
                old_center_diff = new_center_diff

            frame_idx_run += 1
            #print(frame_idx_run, img_idx_run)

            tracked_car_bboxes.append(pred_bbox)
            tracked_car_img_idxs.append(img_idx_run)

        return tracked_car_bboxes, tracked_car_img_idxs

    def compute_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate the (x, y)-coordinates of the intersection rectangle
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        box1Area = w1 * h1
        box2Area = w2 * h2

        # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
        iou = interArea / float(box1Area + box2Area - interArea)

        return iou

    def check_for_image_switch(self, pred_bbox, img_idx_run, rdy_to_switch, switched, cur_homo, out_shape):
        torch_track_mask = torch.zeros(out_shape, dtype=torch.float32, device='cuda')
        torch_track_mask[pred_bbox[1]:pred_bbox[1] + pred_bbox[3], pred_bbox[0]:pred_bbox[0] + pred_bbox[2]] = 1.

        if img_idx_run == 0:
            src_img = warp_perspective(
                torch_track_mask[50:1280 + 50, -1920:].unsqueeze(0).unsqueeze(0), cur_homo,
                out_shape)
            # If not overlapping
            if torch.sum(src_img[0, 0, :, -1920:]) > 50:
                rdy_to_switch += 1
                if rdy_to_switch >= 2:
                    img_idx_run = 1
                    rdy_to_switch = 0
                    switched = 3
            else:
                rdy_to_switch = 0
        elif img_idx_run == 1:
            # The detection is at least partly in non-warped img
            if torch.sum(torch_track_mask[:, -1920:]) > 50:
                indices = torch.nonzero(torch_track_mask[:, -1920:])
                center = indices.float().mean(dim=0)
                # The detection is not present in the right half of the non-warped image
                if center[1] > 960:
                    rdy_to_switch += 1
                    if rdy_to_switch >= 2:
                        img_idx_run = 2
                        rdy_to_switch = 0
                        switched = 3
                else:
                    rdy_to_switch = 0
            else:
                rdy_to_switch += 1
                if rdy_to_switch >= 2:
                    img_idx_run = 0
                    rdy_to_switch = 0
                    switched = 3
        elif img_idx_run == 2:
            # The detection is at least partly in non-warped img
            if torch.sum(torch_track_mask[:, :1920]) > 50:
                indices = torch.nonzero(torch_track_mask[:, :1920])
                center = indices.float().mean(dim=0)
                # The detection is not present in the left half of the non-warped image
                if center[1] <= 960:
                    rdy_to_switch += 1
                    if rdy_to_switch >= 2:
                        img_idx_run = 1
                        rdy_to_switch = 0
                        switched = 3
                else:
                    rdy_to_switch = 0
            else:
                rdy_to_switch += 1
                if rdy_to_switch >= 2:
                    img_idx_run = 3
                    rdy_to_switch = 0
                    switched = 3
        elif img_idx_run == 3:
            src_img = warp_perspective(
                torch_track_mask[50:1280 + 50, :1920].unsqueeze(0).unsqueeze(0).to(torch.float32), cur_homo,
                out_shape)
            # If not overlapping
            if torch.sum(src_img[0, 0, :, :1920]) > 50:
                rdy_to_switch += 1
                if rdy_to_switch >= 2:
                    img_idx_run = 2
                    rdy_to_switch = 0
                    switched = 3
            else:
                rdy_to_switch = 0

        return img_idx_run, rdy_to_switch, switched

    def visu_masks_with_ids(self, masks, img, file_name, ids):
        img_vizu = cv2.cvtColor(img.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        for idx, mask in enumerate(masks):
            min_bbox = self.compute_bounding_box(mask).cpu().numpy()
            mask_3channel = np.dstack([mask.cpu().numpy().astype(np.uint8)] * 3) * 255
            img_vizu = cv2.addWeighted(img_vizu, 1, mask_3channel, 0.5, 0)
            string_idx = '_'.join(str(x) for x in ids[idx])
            cv2.putText(img_vizu, string_idx, (min_bbox[0], min_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imwrite(file_name + ".png", img_vizu)

    def visu_to_video(self, masks, sam_masks, tracked_idxs, video_name, imgs):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_filename = video_name + '.mp4'
        video = cv2.VideoWriter(video_filename, fourcc, 10, (masks[0][0].shape[2], masks[0][0].shape[1]))

        for frame_idx, frame in enumerate(tracked_idxs):
            #Detectron2 mask
            if frame is None:
                continue
            elif frame[0] == 0:
                mask = masks[frame[1]][frame[2]][frame[3]]
                img_vizu = cv2.cvtColor(imgs[frame[1]][frame[2]].numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                mask_3channel = np.dstack([mask.cpu().numpy().astype(np.uint8)] * 3) * 255
                img_vizu = cv2.addWeighted(img_vizu, 1, mask_3channel, 0.5, 0)
            #SAM mask
            else:
                mask = sam_masks[frame[3]]
                img_vizu = cv2.cvtColor(imgs[frame[1]][frame[2]].numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                mask_3channel = np.dstack([mask.astype(np.uint8)] * 3) * 255
                img_vizu = cv2.addWeighted(img_vizu, 1, mask_3channel, 0.5, 0)

            video.write(img_vizu)

        # Release the VideoWriter
        video.release()

    def filter_overlapping_detections(self, detections, img, homo, img_idx):
        out_shape = (img.shape[-2] + self.cfg.image_stitching.height_pxl_pad, 2 * img.shape[-1] + self.cfg.image_stitching.width_pxl_pad)
        out_detections = []

        if img_idx == 0:
            for det_mask in detections:
                #The detection is in the warped side -> correct
                if torch.sum(det_mask[:, -1920:]) == 0:
                    out_detections.append(det_mask)
                #The detection is in the non-warped side -> need to check if it is not overlapping to next image
                else:
                    src_img = warp_perspective(det_mask[50:1280 + 50, -1920:].unsqueeze(0).unsqueeze(0).to(torch.float32), homo, out_shape)
                    #If not overlapping
                    if torch.sum(src_img[0, 0, :, -1920:]) == 0:
                        out_detections.append(det_mask)

        elif img_idx == 1:
            for det_mask in detections:
                #The detection is at least partly in non-warped img
                if torch.sum(det_mask[:, -1920:]) > 0:
                    indices = torch.nonzero(det_mask[:, -1920:])
                    center = indices.float().mean(dim=0)
                    #The detection is not present in the right half of the non-warped image
                    if center[1] <= 960:
                        out_detections.append(det_mask)

        elif img_idx == 2:
            for det_mask in detections:
                #The detection is at least partly in non-warped img
                if torch.sum(det_mask[:, :1920]) > 0:
                    indices = torch.nonzero(det_mask[:, :1920])
                    center = indices.float().mean(dim=0)
                    # The detection is not present in the left half of the non-warped image
                    if center[1] > 960:
                        out_detections.append(det_mask)

        elif img_idx == 3:
            for det_mask in detections:
                #The detection is in the warped side -> correct
                if torch.sum(det_mask[:, :1920]) == 0:
                    out_detections.append(det_mask)
                #The detection is in the non-warped side -> need to check if it is not overlapping to next image
                else:
                    src_img = warp_perspective(det_mask[50:1280 + 50, :1920].unsqueeze(0).unsqueeze(0).to(torch.float32), homo, out_shape)
                    #If not overlapping
                    if torch.sum(src_img[0, 0, :, :1920]) == 0:
                        out_detections.append(det_mask)

        return out_detections