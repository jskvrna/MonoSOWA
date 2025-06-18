import numpy as np
import open3d
import point_cloud_utils as pcu
import faiss

from anno_V3 import AutoLabel3D

class CAARGrowing(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)

    def segment_plane(self, lidar_to_segment, return_abcd=False):
        """
        Segment the plane from the point cloud.
        return: Points outside the plane
        """
        if len(lidar_to_segment) == 0:
            return None, None
        # Create a point cloud from the points
        pcd = open3d.geometry.PointCloud()
        cur_lidar = lidar_to_segment
        lidar_downsampled = pcu.downsample_point_cloud_on_voxel_grid(0.2, cur_lidar[:, :3])
        if lidar_downsampled.shape[0] < 3 or lidar_downsampled is None or lidar_downsampled.ndim != 2:
            return None, None
        pcd.points = open3d.utility.Vector3dVector(lidar_downsampled[:, :3])

        #open3d.visualization.draw_geometries([pcd])

        # Segment the plane
        abcd, _ = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=10000)

        if return_abcd:
            return abcd

        # extract the points OUTSIDE the plane given the abcd parameters of plane but from the cur_lidar
        a, b, c, d = abcd
        distances = np.abs(a * cur_lidar[:, 0] + b * cur_lidar[:, 1] + c * cur_lidar[:, 2] + d) / np.sqrt(
            a ** 2 + b ** 2 + c ** 2)
        outlier_mask = distances > 0.2
        outlier_points = cur_lidar[outlier_mask]

        outlier_indices = np.argwhere(outlier_mask)

        #Visualize points and green should be the points that are not in the plane and red are the points that are in the plane
        #pcd = open3d.geometry.PointCloud()
        #pcd.points = open3d.utility.Vector3dVector(outlier_points)
        #pcd.paint_uniform_color([1, 0, 0])
        #pcd2 = open3d.geometry.PointCloud()
        #pcd2.points = open3d.utility.Vector3dVector(cur_lidar[~outlier_mask])
        #pcd2.paint_uniform_color([0, 1, 0])
        #open3d.visualization.draw_geometries([pcd, pcd2])

        return outlier_points, outlier_indices

    def filter_points_by_plane(self, lidar_to_segment, plane_abcd):
        # extract the points OUTSIDE the plane given the abcd parameters of plane but from the cur_lidar
        a, b, c, d = plane_abcd
        distances = np.abs(a * lidar_to_segment[:, 0] + b * lidar_to_segment[:, 1] + c * lidar_to_segment[:, 2] + d) / np.sqrt(
            a ** 2 + b ** 2 + c ** 2)
        outlier_mask = distances > 0.2
        outlier_points = lidar_to_segment[outlier_mask]

        outlier_indices = np.argwhere(outlier_mask)

        return outlier_points, outlier_indices

    def growing_algorithm(self, pcloud, car_points_indices):
        #First build an faiss index of pcloud
        pcloud = pcloud[:, :3]
        car_points_indices = car_points_indices.flatten()
        pcloud = np.ascontiguousarray(pcloud).astype('float32')
        quantizer = faiss.IndexFlatL2(pcloud.shape[1])
        index = faiss.IndexIVFFlat(quantizer, pcloud.shape[1],
                                   int(np.floor(np.sqrt(pcloud.shape[0]))))
        index.train(pcloud)
        index.add(pcloud)
        index.nprobe = 10 # 1 is default, bigger number gives more accuracy at the tradeoff of linearly increasing time.

        all_regions = []

        for i in range(len(self.cfg.context_aware_growing.growing_thresholds)):
            indices_mask = np.zeros(pcloud.shape[0], dtype=bool)

            regions = []
            # While all indices mask aren't true
            while not np.all(indices_mask[car_points_indices]):
                growing = np.array([], dtype=np.uint32)

                idx = np.random.choice(np.argwhere(indices_mask[car_points_indices] == False).flatten())
                idx_pcloud = car_points_indices[idx]

                # add idx to growing and set indices_mask[idx] to True
                growing = np.append(growing, idx_pcloud)
                to_grow = np.copy(growing)
                indices_mask[idx_pcloud] = True

                while True:
                    template = pcloud[to_grow]
                    _, distances, indexes = index.range_search(np.ascontiguousarray(template).astype('float32'), self.cfg.context_aware_growing.growing_thresholds[i] ** 2)

                    new_found_idxs = np.unique(indexes)
                    to_grow = np.setdiff1d(new_found_idxs, growing, assume_unique=True)

                    growing = np.append(growing, to_grow)
                    indices_mask[new_found_idxs] = True

                    foreground = len(np.intersect1d(growing, car_points_indices, assume_unique=True))
                    overlap = foreground / len(growing)

                    if overlap < 0.95:
                        break
                    else:
                        if len(to_grow) == 0:
                            regions.append(growing)
                            break

            all_regions.append(regions)

        return all_regions

    def perform_growing(self, mask, camera_idx, scan):
        cur_lidar = scan
        cur_lidar = cur_lidar[cur_lidar[:, 3] == camera_idx]
        cur_lidar = cur_lidar.T

        # Now, get indexes of the points which project into the mask
        tmp1 = np.argwhere(mask[cur_lidar[4, :].astype(int), cur_lidar[5, :].astype(int)])

        if len(tmp1) == 0:
            return None

        # Now, filter the points based on the indexes
        filtered_lidar = np.array([cur_lidar[0, tmp1], cur_lidar[1, tmp1], cur_lidar[2, tmp1]]).transpose()[0]
        cur_lidar = scan

        #Now we need to find a region, where we want to perform the plane removal

        x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

        dist_from_mean = np.sqrt(
            (x_mean - cur_lidar[:, 0]) ** 2 + (y_mean - cur_lidar[:, 1]) ** 2)

        indexes = np.argwhere(dist_from_mean < 10)

        cur_lidar_filtered = cur_lidar[indexes[:, 0]]

        if cur_lidar_filtered.shape[0] < 3:
            return None

        #Now lets remove the ground plane
        segmented_lidar, segmented_indices = self.segment_plane(cur_lidar_filtered)

        if segmented_lidar is None:
            return None

        # Now, get indexes of the points which project into the mask
        scan_in_camera = np.argwhere(segmented_lidar[:, 3] == camera_idx).flatten()
        tmp1 = np.argwhere(mask[segmented_lidar[:, 4][scan_in_camera].astype(int), segmented_lidar[:, 5][scan_in_camera].astype(int)]).flatten()
        #tmp1 = scan_in_camera[tmp1]

        regions_candidates = self.growing_algorithm(segmented_lidar[scan_in_camera], tmp1)

        #Now for all the region candidates compute IOU with the tmp1
        top_candidates = None
        max_points = -1

        for i in range(len(regions_candidates)):
            for z in range(len(regions_candidates[i])):
                if len(regions_candidates[i][z]) > max_points:
                    max_points = len(regions_candidates[i][z])
                    top_candidates = regions_candidates[i][z]

        candidates = top_candidates

        if candidates is None:
            return None

        #Do next round, in a smaller region
        x_mean, y_mean, z_mean = self.compute_mean(segmented_lidar[scan_in_camera][candidates, :3])

        dist_from_mean = np.sqrt(
            (x_mean - cur_lidar[:, 0]) ** 2 + (y_mean - cur_lidar[:, 1]) ** 2)

        # TODO Play with the constant
        indexes = np.argwhere(dist_from_mean < 5.)

        cur_lidar_filtered = cur_lidar[indexes[:, 0]]

        if cur_lidar_filtered.shape[0] < 3:
            return None

        # Now lets remove the ground plane
        segmented_lidar, segmented_indices = self.segment_plane(cur_lidar_filtered)

        # Now, get indexes of the points which project into the mask
        scan_in_camera = np.argwhere(segmented_lidar[:, 3] == camera_idx).flatten()
        tmp1 = np.argwhere(mask[segmented_lidar[:, 4][scan_in_camera].astype(int), segmented_lidar[:, 5][
            scan_in_camera].astype(int)]).flatten()
        #tmp1 = scan_in_camera[tmp1]

        regions_candidates = self.growing_algorithm(segmented_lidar[scan_in_camera], tmp1)

        # Now for all the region candidates compute IOU with the tmp1
        top_candidates = None
        max_points = -1

        for i in range(len(regions_candidates)):
            for z in range(len(regions_candidates[i])):
                if len(regions_candidates[i][z]) > max_points:
                    max_points = len(regions_candidates[i][z])
                    top_candidates = regions_candidates[i][z]

        candidates = top_candidates

        if candidates is None:
            return None

        '''
        #visualize the segmented point cloud and for each region candidate visualize the points with diff color for each region
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cur_lidar[:, :3])
        pcd.paint_uniform_color([1, 0, 0])

        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(np.array([segmented_lidar[tmp1, 0], segmented_lidar[tmp1, 1], segmented_lidar[tmp1, 2]]).transpose()[0])
        pcd2.paint_uniform_color([0, 1, 0])

        pcd3 = open3d.geometry.PointCloud()
        pcd3.points = open3d.utility.Vector3dVector(segmented_lidar[candidates, :3])
        pcd3.paint_uniform_color([0, 0, 1])
        open3d.visualization.draw_geometries([pcd, pcd2, pcd3])
        '''

        return segmented_lidar[scan_in_camera][candidates, :3]

    def perform_growing_bbox(self, bbox, scan):
        cur_lidar = scan[:, :3]

        # Now, get indexes of the points which project into the mask
        inside_points = np.array(bbox.get_point_indices_within_bounding_box(open3d.utility.Vector3dVector(cur_lidar[:, :3])))

        dists = np.linalg.norm(cur_lidar[:, :3] - bbox.center, axis=1)
        mask = dists < 7.5
        lidar_5mrange = cur_lidar[mask]

        if len(inside_points) == 0:
            return None

        inside_points_lidar = np.array(cur_lidar[inside_points, :])

        mask = np.ones(len(cur_lidar), dtype=bool)
        mask[inside_points] = False
        cur_lidar = cur_lidar[mask]

        #segmented_lidar, segmented_indices = self.segment_plane(inside_points_lidar)
        abcd = self.segment_plane(lidar_5mrange, return_abcd=True)
        if abcd[0] is None or abcd[1] is None:
            return None

        segmented_lidar, segmented_indices = self.filter_points_by_plane(inside_points_lidar, abcd)

        if segmented_lidar is None:
            return None

        #segmented_indices = inside_points[segmented_indices.squeeze(1)]

        dists = np.linalg.norm(cur_lidar[:, :3] - bbox.center, axis=1)
        mask = dists < 10
        cur_lidar = cur_lidar[mask]

        idx_to_grow = np.arange(len(cur_lidar), len(cur_lidar) + len(segmented_lidar))
        cur_lidar = np.concatenate((cur_lidar, segmented_lidar), axis=0)

        # Now, get indexes of the points which project into the mask

        regions_candidates = self.growing_algorithm(cur_lidar, idx_to_grow)
        #to_dbscan = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(segmented_lidar[:, :3]))
        #labels = np.array(to_dbscan.cluster_dbscan(eps=0.3, min_points=10, print_progress=True))

        #largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))
        #car_points = to_dbscan.select_by_index(np.where(labels == largest_cluster_idx)[0])

        #clusterer = hdbscan.HDBSCAN()
        #clusterer.fit(segmented_lidar[:, :3])

        #max_index = clusterer.labels_.max()
        #car_points = segmented_lidar[clusterer.labels_ == max_index]


        #Now for all the region candidates compute IOU with the tmp1
        top_candidates = None
        max_points = -1
        top_thresh = -1

        for i in range(len(regions_candidates)):
            for z in range(len(regions_candidates[i])):
                if len(regions_candidates[i][z]) > max_points:
                    max_points = len(regions_candidates[i][z])
                    top_candidates = regions_candidates[i][z]
                    top_thresh = i
        
        if not self.cfg.general.supress_debug_prints:
            print(top_thresh)
        candidates = top_candidates

        if candidates is None:
            return None

        '''
        #visualize the segmented point cloud and for each region candidate visualize the points with diff color for each region
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cur_lidar[:, :3])
        pcd.paint_uniform_color([1, 0, 0])

        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(inside_points_lidar)
        pcd2.paint_uniform_color([0, 1, 0])

        pcd3 = open3d.geometry.PointCloud()
        pcd3.points = open3d.utility.Vector3dVector(cur_lidar[candidates, :3])
        pcd3.paint_uniform_color([0, 0, 1])
        open3d.visualization.draw_geometries([pcd, pcd2, pcd3])
        '''
        out_lidar = cur_lidar[candidates, :3]
        #out_lidar = np.asarray(car_points.points)
        out_lidar = np.pad(out_lidar, ((0, 0), (0, 1)), 'constant', constant_values=1)
        return out_lidar.T

