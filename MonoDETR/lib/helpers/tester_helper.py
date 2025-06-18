import os
import tqdm
import shutil
import open3d as o3d
import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
import time
import numpy as np
from scipy.spatial.transform import Rotation as R


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, train_cfg=None, model_name='monodetr'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            assert os.path.exists(checkpoint_path)
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            self.evaluate()

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])

            if os.path.exists(self.output_dir + "/checkpoint_best.pth"):
                checkpoint = self.output_dir + "/checkpoint_best.pth"
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()

            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth"):
                        if len(f.split("_")) > 2:
                            checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list = sorted(checkpoints_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)
            img_sizes[:, 1] = img_sizes[:, 1] / info['height_crop'].to(self.device)

            start_time = time.time()
            ###dn
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)
            ###
            end_time = time.time()
            model_infer_time += end_time - start_time

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get('threshold', 0.2))
            '''
            for key in dets.keys():
                new_preds = []
                for det_idx in range(len(dets[key])):
                    bbox2d_height = dets[key][det_idx][5] - dets[key][det_idx][3]
                    bbox2d = dets[key][det_idx][2:6]

                    if bbox2d_height < 25:
                        continue
                    elif (bbox2d[0] < 50 or bbox2d[2] > 1200) and np.sqrt((dets[key][det_idx][9] ** 2) + (dets[key][det_idx][11] ** 2)) < 10.:
                        continue
                    else:
                        new_preds.append(dets[key][det_idx])
                #print('old_preds: ', dets[key])
                dets[key] = new_preds
                #print('new_preds: ', dets[key])
            '''
            for key in dets.keys():
                new_preds = []
                for det_idx in range(len(dets[key])):
                    dist = np.sqrt((dets[key][det_idx][9] ** 2) + (dets[key][det_idx][10] ** 2) + (dets[key][det_idx][11] ** 2))
                    #dets[key][det_idx][12] = dets[key][det_idx][12] - np.pi / 2.
                    #dets[key][det_idx][6] = 1.526
                    #dets[key][det_idx][7] = 1.63
                    #dets[key][det_idx][8] = 3.88
                    if True:
                    #if 0. < dist < 30.:
                    #if 30. <= dist < 50.:
                    #if 50. <= dist:
                        new_preds.append(dets[key][det_idx])
                    else:
                        continue

                #print('old_preds: ', dets[key])
                dets[key] = new_preds
                #print('new_preds: ', dets[key])

            results.update(dets)
            progress_bar.update()

            if self.cfg['visu_predictions'] or self.cfg['visu_ground_truth']:\
                self.visu_preds(predictions=dets, ground_truth=targets, infos=info, calib=calibs)


        print("inference on {} images by {}/per image".format(
            len(self.dataloader), model_infer_time / len(self.dataloader)))

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)

    def save_results(self, results):
        output_dir = os.path.join(self.output_dir, 'outputs', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

    def evaluate(self):
        results_dir = os.path.join(self.output_dir, 'outputs', 'data')
        assert os.path.exists(results_dir)
        result = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        return result

    def visu_preds(self, predictions, ground_truth, infos, calib):
        gt_objects = ground_truth['objects']

        for num_idx, img_idx in enumerate(infos["img_id"]):
            lidar_scan = self.dataloader.dataset.get_velodyne(img_idx)
            lidar_scan[:, 3] = 1.
            velo_to_cam = calib[num_idx].V2C
            lidar_scan = np.matmul(velo_to_cam, lidar_scan.T).T

            current_preds = predictions[img_idx]
            pred_boxes = []

            for pred in current_preds:
                center3D = np.array([pred[9], pred[10] - pred[6] / 2., pred[11]])
                yaw = pred[12] + np.pi / 2.
                dimensions = np.array([pred[7], pred[6], pred[8]])

                #create open3d bounding box from these parameters
                r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
                bbox = o3d.geometry.OrientedBoundingBox(center3D, r.as_matrix(), dimensions)
                bbox.color = np.array([1, 0, 0])
                pred_boxes.append(bbox)

            cur_gt = gt_objects[num_idx, :, :]
            gt_boxes = []
            for gt in cur_gt:
                if gt[0] == 0. or gt[1] == 0. or gt[2] == 0.:
                    continue

                center3D = np.array([gt[3], gt[4] - gt[0] / 2., gt[5]])
                yaw = gt[6] + np.pi / 2.
                dimensions = np.array([gt[1], gt[0], gt[2]])

                #create open3d bounding box from these parameters
                r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
                bbox = o3d.geometry.OrientedBoundingBox(center3D, r.as_matrix(), dimensions)
                bbox.color = np.array([0, 1, 0])
                gt_boxes.append(bbox)

            #Show lidar scan in open3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lidar_scan)

            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.add_geometry(pcd)
            for k in range(len(pred_boxes)):
                visualizer.add_geometry(pred_boxes[k])
            for k in range(len(gt_boxes)):
                visualizer.add_geometry(gt_boxes[k])
            # visualizer.get_render_option().point_size = 5  # Adjust the point size if necessary
            visualizer.get_render_option().background_color = np.asarray([0, 0, 0])  # Set background to black
            visualizer.get_view_control().set_front([0, -0.3, -0.5])
            visualizer.get_view_control().set_lookat([0, 0, 1])
            visualizer.get_view_control().set_zoom(0.05)
            visualizer.get_view_control().set_up([0, -1, 0])
            visualizer.get_view_control().camera_local_translate(5., 0., 8.)
            visualizer.run()
            visualizer.destroy_window()




