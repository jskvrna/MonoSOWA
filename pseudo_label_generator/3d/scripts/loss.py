import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from anno_V3 import AutoLabel3D
import torch
import faiss


class Losses(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)

    def avg_med_distance(self, meas_pcloud, template):
        # compute the Euclidean distance between the two point clouds
        distances = cdist(meas_pcloud, template, 'sqeuclidean')
        # distances = eucl_opt(meas_pcloud,template)
        # assign each point in cloud1 to its closest point in cloud2
        closest_dist_scan_to_temp = np.min(distances, axis=1)
        closest_dist_temp_to_scan = np.min(distances, axis=0)
        loss = np.median(closest_dist_scan_to_temp)/meas_pcloud.shape[0] + np.median(closest_dist_temp_to_scan)/template.shape[0]

        return np.sqrt(loss)

    def avg_chamfer_distance(self, meas_pcloud, template):
        # compute the Euclidean distance between the two point clouds
        distances = cdist(meas_pcloud, template, 'sqeuclidean')
        # distances = eucl_opt(meas_pcloud,template)
        # assign each point in cloud1 to its closest point in cloud2
        closest_dist_scan_to_temp = np.min(distances, axis=1)
        closest_dist_temp_to_scan = np.min(distances, axis=0)
        loss = np.sum(closest_dist_scan_to_temp)/meas_pcloud.shape[0] + np.sum(closest_dist_temp_to_scan)/template.shape[0]

        return np.sqrt(loss)

    def avg_med_distance_only_temp_to_scan(self, meas_pcloud, template):
        # compute the Euclidean distance between the two point clouds
        distances = cdist(meas_pcloud, template, 'sqeuclidean')
        # assign each point in cloud1 to its closest point in cloud2
        closest_dist_temp_to_scan = np.min(distances, axis=0)
        loss = np.median(closest_dist_temp_to_scan)/template.shape[0]

        return loss

    def avg_trim_distance(self, meas_pcloud, template, trim_per):
        # compute the Euclidean distance between the two point clouds
        distances = cdist(meas_pcloud, template, 'euclidean')
        # distances = eucl_opt(meas_pcloud,template)
        # assign each point in cloud1 to its closest point in cloud2
        closest_dist_scan_to_temp = np.min(distances, axis=1)
        closest_dist_temp_to_scan = np.min(distances, axis=0)
        loss = self.custom_trim_mean(closest_dist_scan_to_temp, trim_per) + self.custom_trim_mean(
            closest_dist_temp_to_scan, trim_per)
        return loss

    def custom_trim_mean(self, input, proportion_to_cut_high):
        data_sorted = np.sort(input)
        n_to_use = int((1. - proportion_to_cut_high) * len(data_sorted))
        mean = np.mean(data_sorted[:n_to_use])
        return mean

    def binary_loss(self, meas_pcloud, template):
        # compute the Euclidean distance between the two point clouds
        distances = cdist(meas_pcloud, template, 'sqeuclidean')
        # distances = eucl_opt(meas_pcloud,template)
        # assign each point in cloud1 to its closest point in cloud2
        closest_dist_temp_to_scan = np.min(distances, axis=0)
        loss = np.sum(closest_dist_temp_to_scan < self.cfg.loss_functions.binary_loss_threshold**2)
        return -loss/template.shape[0]

    def binary_loss_bothway_faiss(self, meas_pcloud, template):
        # compute the Euclidean distance between the two point clouds
        # Assuming your data is stored in a NumPy array called 'data'
        idx, distances, indexes = self.index.range_search(np.ascontiguousarray(template).astype('float32'), self.cfg.loss_functions.binary_loss_threshold**2)
        loss = float(len(np.unique(idx)) - 1)/template.shape[0] + float(len(np.unique(indexes)))/meas_pcloud.shape[0]
        return -loss

    def binary_diff_loss_bothway(self, meas_pcloud, template):
        # compute the Euclidean distance between the two point clouds
        # Assuming your data is stored in a NumPy array called 'data'
        distances = torch.cdist(meas_pcloud, template)
        closest_dist_temp_to_scan, _ = torch.min(distances, dim=0)
        closest_dist_scan_to_temp, _ = torch.min(distances, dim=1)
        closest_dist_temp_to_scan = torch.sigmoid(self.cfg.loss_functions.sigmoid_steepness * closest_dist_temp_to_scan) - 0.5
        closest_dist_scan_to_temp = torch.sigmoid(self.cfg.loss_functions.sigmoid_steepness * closest_dist_scan_to_temp) - 0.5
        loss = closest_dist_temp_to_scan.sum() / template.shape[0]
        loss += closest_dist_scan_to_temp.sum() / meas_pcloud.shape[0]
        return loss

    def create_faiss_tree(self, filtered_lidar):
        filtered_lidar = np.ascontiguousarray(filtered_lidar).astype('float32')
        quantizer = faiss.IndexFlatL2(filtered_lidar.shape[1])
        index = faiss.IndexIVFFlat(quantizer, filtered_lidar.shape[1], int(np.floor(np.sqrt(filtered_lidar.shape[0]))))
        index.train(filtered_lidar)
        index.add(filtered_lidar)
        index.nprobe = self.cfg.optimization.index_nprobe  # 1 is default, bigger number gives more accuracy at the tradeoff of linearly increasing time.
        return index

    def compute_loss(self, template_it):
        if self.cfg.loss_functions.loss_function == 'trimmed':
            loss = self.avg_trim_distance(self.filtered_lidar, template_it, self.cfg.loss_functions.trim_treshold)
        elif self.cfg.loss_functions.loss_function == 'med1way':
            loss = self.avg_med_distance_only_temp_to_scan(self.filtered_lidar, template_it)
        elif self.cfg.loss_functions.loss_function == 'medboth':
            loss = self.avg_med_distance(self.filtered_lidar, template_it)
        elif self.cfg.loss_functions.loss_function == 'chamfer':
            loss = self.avg_chamfer_distance(self.filtered_lidar, template_it)
        elif self.cfg.loss_functions.loss_function == 'binary1way':
            loss = self.binary_loss(self.filtered_lidar, template_it)
        elif self.cfg.loss_functions.loss_function == 'binary2way':
            loss = self.binary_loss_bothway_faiss(self.filtered_lidar, template_it)
        elif self.cfg.loss_functions.loss_function == 'diffbin':
            loss = self.binary_diff_loss_bothway(self.filtered_lidar, template_it)
        else:
            raise Exception("No loss function selected")

        return loss
