import torch
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.ops.knn import knn_gather, knn_points
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
import pyvista as pv
from scipy.optimize import linear_sum_assignment
import time
from anno_V3 import AutoLabel3D
from pytorch3d.utils import ico_sphere

from sklearn.cluster import DBSCAN


class Deformable_Mesh(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)
        self.last_yaw = None
        #pyvista.start_xvfb()

    def deformable_mesh_fit(self, car):
        if car.lidar is None or car.moving:
            return car

        cur_lidar = car.lidar.T[:, :3]
        cur_lidar = torch.from_numpy(cur_lidar)
        cur_lidar = cur_lidar.float()

        device = torch.device('cuda' if self.cfg.general.device == 'gpu' else 'cpu')
        cur_lidar = cur_lidar.to(device)
        cur_lidar = cur_lidar.unsqueeze(0)

        src_mesh = self.mesh_templates_p3d[0].to(device)
        #src_mesh = ico_sphere(level=3, device=device)

        pitch = torch.tensor(0., requires_grad=False, device=device, dtype=torch.float32)
        roll = torch.tensor(0., requires_grad=False, device=device, dtype=torch.float32)
        yaw = torch.tensor(car.theta, requires_grad=False, device=device, dtype=torch.float32)
        angles = torch.stack([pitch, yaw, roll], dim=-1)
        rot_matrix = euler_angles_to_matrix(angles, convention="XYZ")
        verts = src_mesh.verts_packed()

        mesh_width = verts[:, 0].max() - verts[:, 0].min()
        mesh_height = verts[:, 1].max() - verts[:, 1].min()
        mesh_length = verts[:, 2].max() - verts[:, 2].min()

        scale = torch.tensor([car.width / mesh_width , car.height / mesh_height, car.length / mesh_length], device=device, dtype=torch.float32)
        verts = torch.matmul(verts * scale, rot_matrix)

        # Create mesh from the verts
        src_mesh = Meshes(verts=[verts], faces=[src_mesh.faces_packed()])

        offset = torch.tensor([car.x, car.y, car.z], device=device, dtype=torch.float32)
        src_mesh = src_mesh.offset_verts(offset)

        verts = src_mesh.verts_packed()

        deform_verts = torch.full(verts.shape, 0.0, device=device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.SGD([deform_verts], lr=0.9, momentum=0.9)

        # Number of optimization steps
        Niter = 5000
        # Weight for the chamfer loss
        w_chamfer = 2.0
        # Weight for mesh edge loss
        w_edge = 20.
        # Weight for mesh normal consistency
        w_normal = 0
        # Weight for mesh laplacian smoothing
        w_laplacian = 0
        # Plot period for the losses
        plot_period = 4999
        loop = tqdm(range(Niter), desc='Optimization Progress', leave=True)

        if self.cfg.deformable_mesh.visu_deformable_mesh:
            # Initialize the pv plotter
            pv.global_theme.background = 'white'
            pv.global_theme.show_edges = True

            # Initialize the plotter
            verts_pv = src_mesh.verts_packed().cpu().numpy()
            faces_pv = src_mesh.faces_packed().cpu().numpy()
            # PyVista expects faces in a specific format
            faces_pv = np.hstack((np.full((faces_pv.shape[0], 1), 3), faces_pv)).flatten()
            plotter = pv.Plotter(window_size=[1920, 1080])
            mesh_pv = pv.PolyData(verts_pv, faces_pv)
            mesh_actor = plotter.add_mesh(mesh_pv, color='lightblue', show_edges=True)
            cur_lidar_np = cur_lidar.squeeze(0).cpu().numpy()
            point_cloud = pv.PolyData(cur_lidar_np)
            plotter.add_points(point_cloud, color='red', point_size=5)
            plotter.camera_position = [(0, -10., -10.), (0, 0, 0), (0, 0, 1)]  # Set pre
            plotter.camera.elevation += 30
            plotter.show(auto_close=False, interactive_update=True)  # Keep the plotter op

        for i in loop:
            # Initialize optimizer
            optimizer.zero_grad()
            if not self.cfg.general.supress_debug_prints:
                if i % 100 == 0:
                    print(i)

            new_src_mesh = src_mesh.offset_verts(deform_verts)

            verts_updated = new_src_mesh.verts_packed().detach().cpu().numpy()
            mesh_pv.points = verts_updated
            plotter.update()

            sample_src = sample_points_from_meshes(new_src_mesh, 10000)
            loss_chamfer = self.tfl_knn(cur_lidar, sample_src)

            # and (b) the edge length of the predicted mesh
            loss_edge = self.mesh_edge_loss_v2(new_src_mesh, target_length=0.05)

            # mesh normal consistency
            loss_normal = mesh_normal_consistency(new_src_mesh)

            # mesh laplacian smoothing
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

            # Weighted sum of the losses
            loss = (loss_chamfer * w_chamfer + loss_edge * w_edge +
                    loss_normal * w_normal + loss_laplacian * w_laplacian)

            # Print the losses
            loop.set_description('total_loss = %.6f' % loss)

            # Plot mesh
            if i % plot_period == 0:
                #self.visu_in_open3d(new_src_mesh, cur_lidar)
                print(i)

            # Optimization step
            loss.backward()
            optimizer.step()

        plotter.close()

        return car

    def deformable_mesh_fit_lim_dof(self, car):
        if car.lidar is None or car.moving or not car.optimized:
            return car

        cur_lidar = car.lidar.T[:, :3]
        cur_lidar = torch.from_numpy(cur_lidar)
        cur_lidar = cur_lidar.float()

        device = torch.device('cuda' if self.cfg.general.device == 'gpu' else 'cpu')
        cur_lidar = cur_lidar.to(device)
        cur_lidar = cur_lidar.unsqueeze(0)

        src_mesh = self.mesh_templates_p3d[0].to(device)
        #src_mesh = ico_sphere(level=3, device=device)
        #src_mesh = load_objs_as_meshes(["../data/sedan_test.obj"], device=device)

        verts = src_mesh.verts_packed()
        num_verts = verts.shape[0]
        # Precompute mirrored positions
        mirrored_verts = verts.clone()
        mirrored_verts[:, 0] *= -1
        # Create a cost matrix for the Hungarian algorithm
        cost_matrix = torch.cdist(verts, mirrored_verts).cpu().numpy()
        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mirror_index = torch.full((num_verts,), -1, dtype=torch.long, device=device)
        mirror_index[row_ind] = torch.tensor(col_ind, dtype=torch.long, device=device)

        mesh_width = verts[:, 0].max() - verts[:, 0].min()
        mesh_height = verts[:, 1].max() - verts[:, 1].min()
        mesh_length = verts[:, 2].max() - verts[:, 2].min()

        scale = torch.tensor([car.width / mesh_width, car.height / mesh_height, car.length / mesh_length], device=device, dtype=torch.float32)
        verts = verts * scale

        # Create mesh from the verts
        src_mesh = Meshes(verts=[verts], faces=[src_mesh.faces_packed()])
        #visu the mesh with the corrdinate frame
        #self.visu_in_open3d(src_mesh, cur_lidar)

        offset = torch.tensor([car.x, car.y, car.z], device=device, dtype=torch.float32)

        verts = src_mesh.verts_packed()

        deform_verts = torch.zeros(verts.shape[0], device=device, dtype=torch.float32, requires_grad=True)
        deform_verts2 = torch.tensor([0., 0., 0., 1., 1., 1., 0.], device=device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.AdamW([deform_verts, deform_verts2], lr=0.025)
        padding = torch.zeros_like(deform_verts, requires_grad=False)

        # Number of optimization steps
        Niter = 1000
        # Weight for the chamfer loss
        w_chamfer = 5.0
        # Weight for mesh edge loss
        w_edge = 1.
        # Weight for mesh normal consistency
        w_normal = 0.
        # Weight for mesh laplacian smoothing
        w_laplacian = 10.
        # Weight for mesh symmetry
        w_symmetry = 1.
        # Plot period for the losses
        plot_period = 4999
        loop = tqdm(range(Niter), desc='Optimization Progress', leave=True)

        num_verts = verts.shape[0]

        if self.cfg.deformable_mesh.visu_deformable_mesh:
            # Initialize the pv plotter
            pv.global_theme.background = 'white'
            pv.global_theme.show_edges = True

            # Initialize the plotter
            verts_pv = src_mesh.verts_packed().cpu().numpy()
            faces_pv = src_mesh.faces_packed().cpu().numpy()
            # PyVista expects faces in a specific format
            faces_pv = np.hstack((np.full((faces_pv.shape[0], 1), 3), faces_pv)).flatten()
            plotter = pv.Plotter(window_size=[1920, 1080])
            mesh_pv = pv.PolyData(verts_pv, faces_pv)
            mesh_actor = plotter.add_mesh(mesh_pv, color='lightblue', show_edges=True)
            cur_lidar_np = cur_lidar.squeeze(0).cpu().numpy()
            point_cloud = pv.PolyData(cur_lidar_np)
            plotter.add_points(point_cloud, color='red', point_size=5)
            plotter.camera_position = [(0, -10., -10.), (0, 0, 0), (0, 0, 1)]  # Set pre
            plotter.camera.elevation += 30
            plotter.show(auto_close=False, interactive_update=True)  # Keep the plotter op

        for i in loop:
            start = time.time_ns()
            # Initialize optimizer
            optimizer.zero_grad()
            if not self.cfg.general.supress_debug_prints:
                if i % 100 == 0:
                    print(i)

            pitch = torch.tensor(0., requires_grad=False, device=device, dtype=torch.float32)
            roll = torch.tensor(0., requires_grad=False, device=device, dtype=torch.float32)
            angles = torch.stack([pitch, deform_verts2[6] + car.theta, roll], dim=-1)
            rot_matrix = euler_angles_to_matrix(angles, convention="XYZ")
            new_src_mesh = src_mesh.clone()
            verts = new_src_mesh.verts_packed()
            if i > 100:
                verts = torch.matmul(verts * deform_verts2[3:6], rot_matrix)
            else:
                verts = torch.matmul(verts, rot_matrix)
            new_src_mesh = Meshes(verts=[verts], faces=[new_src_mesh.faces_packed()])
            # visu the mesh with the corrdinate frame
            # self.visu_in_open3d(src_mesh, cur_lidar)
            new_src_mesh = new_src_mesh.offset_verts(deform_verts2[:3] + offset)

            if i >= 250:
                off = torch.stack((padding, deform_verts, padding), dim=1)
                new_src_mesh = new_src_mesh.offset_verts(off * 10)

            if self.cfg.deformable_mesh.visu_deformable_mesh:
                verts_updated = new_src_mesh.verts_packed().detach().cpu().numpy()
                mesh_pv.points = verts_updated
                plotter.update()

            # We sample 5k points from the surface of each mesh
            sample_src = sample_points_from_meshes(new_src_mesh, 10000)
            loss_chamfer = self.tfl_knn(cur_lidar, sample_src)
            # and (b) the edge length of the predicted mesh
            loss_edge = self.mesh_edge_loss_v2(new_src_mesh, target_length=0.00)
            loss_symmetry = self.symmetry_loss(new_src_mesh.verts_packed(), mirror_index, device)

            # mesh normal consistency
            #loss_normal = mesh_normal_consistency(new_src_mesh)
            # mesh laplacian smoothing
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

            # Weighted sum of the losses
            loss = (loss_chamfer * w_chamfer + loss_edge * w_edge +
                    loss_laplacian * w_laplacian + loss_symmetry * w_symmetry)

            # Print the losses
            loop.set_description('total_loss = %.6f' % loss)

            # Optimization step
            loss.backward()
            optimizer.step()
            loop.update(i)
            print(f"Time taken for iteration {i}: {(time.time_ns() - start) / 1e6} ms")

            #frames.append(plotter.screenshot())
        if self.cfg.deformable_mesh.visu_deformable_mesh:
            plotter.iren.interactor.TerminateApp()  # Ensure the interactor is terminated
            del plotter  # Clean up the object

        pitch = torch.tensor(0., requires_grad=False, device=device, dtype=torch.float32)
        roll = torch.tensor(0., requires_grad=False, device=device, dtype=torch.float32)
        angles = torch.stack([pitch, deform_verts2[6] + car.theta, roll], dim=-1)
        rot_matrix = euler_angles_to_matrix(angles, convention="XYZ")
        new_src_mesh = src_mesh.clone()
        verts = new_src_mesh.verts_packed()
        verts = torch.matmul(verts * deform_verts2[3:6], rot_matrix)
        new_src_mesh = Meshes(verts=[verts], faces=[new_src_mesh.faces_packed()])
        new_src_mesh = new_src_mesh.offset_verts(deform_verts2[:3] + offset)

        off = torch.stack((padding, deform_verts, padding), dim=1)
        new_src_mesh = new_src_mesh.offset_verts(off * 10)
        verts = new_src_mesh.verts_packed()
        height = verts[:, 1].max() - verts[:, 1].min()

        car.x = deform_verts2[0].cpu().item() + car.x
        car.y = deform_verts2[1].cpu().item() + car.y
        car.z = deform_verts2[2].cpu().item() + car.z
        car.theta = -deform_verts2[6].cpu().item() - car.theta

        car.length *= deform_verts2[5].cpu().item()
        car.height = height.cpu().item()
        car.width *= deform_verts2[3].cpu().item()

        car.length = np.clip(car.length, 0.75 * self.cfg.templates.template_length, 1.25 * self.cfg.templates.template_length)
        car.height = np.clip(car.height, 0.75 * self.cfg.templates.template_height, 1.25 * self.cfg.templates.template_height)
        car.width = np.clip(car.width, 0.75 * self.cfg.templates.template_width, 1.25 * self.cfg.templates.template_width)

        return car

    def deformable_mesh_fit_lim_dof_batch(self, cars):
        valid_cars = []
        valid_indices = []
        for i, car in enumerate(cars):
            if car.lidar is not None and not car.moving and car.optimized:
                valid_cars.append(car)
                valid_indices.append(i)

        if not valid_cars:
            return cars

        N_lidar_points = 10000  # Number of points to sample from each lidar
        device = torch.device('cuda' if self.cfg.general.device == 'gpu' else 'cpu')
        cur_lidars = []
        cur_weights = []
        for car in valid_cars:
            cur_lidar = car.lidar.T[:, :3]
            cur_lidar = self.ensamble_clustering(cur_lidar)
            cur_lidar = torch.from_numpy(cur_lidar).float()

            if cur_lidar.shape[0] >= N_lidar_points:
                indices = torch.randperm(cur_lidar.shape[0])[:N_lidar_points]
                cur_lidar = cur_lidar[indices]
                cur_weights.append(torch.ones(N_lidar_points, device=device, dtype=torch.float32))
            else:
                padding = torch.zeros(N_lidar_points - cur_lidar.shape[0], 3)
                cur_lidar = torch.cat([cur_lidar, padding], dim=0)
                cur_weights.append(torch.cat([torch.ones(cur_lidar.shape[0] - padding.shape[0], device=device, dtype=torch.float32),
                                              torch.zeros(padding.shape[0], device=device, dtype=torch.float32)], dim=0))
            cur_lidars.append(cur_lidar.to(device, non_blocking=True))

        cur_lidars = torch.stack(cur_lidars, dim=0)
        cur_weights = torch.stack(cur_weights, dim=0)

        src_mesh_template = self.mesh_templates_p3d[0].to(device)
        verts_template = src_mesh_template.verts_packed()  # Shape: [N_verts, 3]
        faces_template = src_mesh_template.faces_packed()  # Shape: [N_faces, 3]

        mesh_width = verts_template[:, 0].max() - verts_template[:, 0].min()
        mesh_height = verts_template[:, 1].max() - verts_template[:, 1].min()
        mesh_length = verts_template[:, 2].max() - verts_template[:, 2].min()

        scales = []
        offsets = []
        thetas = []
        for car in valid_cars:
            scale = torch.tensor([
                car.width / mesh_width,
                car.height / mesh_height,
                car.length / mesh_length
            ], device=device, dtype=torch.float32)
            scales.append(scale)
            offset = torch.tensor([car.x, car.y, car.z], device=device, dtype=torch.float32)
            offsets.append(offset)
            thetas.append(car.theta)

        scales = torch.stack(scales, dim=0)  # Shape: [batch_size, 3]
        offsets = torch.stack(offsets, dim=0)  # Shape: [batch_size, 3]
        thetas = torch.tensor(thetas, device=device, dtype=torch.float32)  # Shape: [batch_size]

        # Expand verts and apply scales
        verts_expanded = verts_template.unsqueeze(0).expand(len(valid_cars), -1, -1)  # Shape: [batch_size, N_verts, 3]
        verts_scaled = verts_expanded * scales.unsqueeze(1)

        faces_list = [faces_template for _ in range(len(valid_cars))]
        verts_scaled = list(verts_scaled)
        src_mesh = Meshes(verts=verts_scaled, faces=faces_list)

        mirrored_verts = verts_template.clone()
        mirrored_verts[:, 0] *= -1
        cost_matrix = torch.cdist(verts_template, mirrored_verts).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mirror_index = torch.full((verts_template.shape[0],), -1, dtype=torch.long, device=device)
        mirror_index[row_ind] = torch.tensor(col_ind, dtype=torch.long, device=device)

        # Initialize deformation variables
        deform_verts = torch.zeros(len(valid_cars), verts_template.shape[0], device=device, requires_grad=True, dtype=torch.float32)
        deform_verts2 = torch.tensor([[0., 0., 0., 1., 1., 1., 0.]] * len(valid_cars), device=device, requires_grad=True, dtype=torch.float32)

        optimizer = torch.optim.AdamW([deform_verts, deform_verts2], lr=0.025)
        padding = torch.zeros_like(deform_verts, requires_grad=False)

        Niter = 1000
        w_chamfer = 5.0
        w_edge = 1.
        w_laplacian = 10.
        w_symmetry = 1.
        w_tfl_new = 2.5

        loop = tqdm(range(Niter), desc='Optimization Progress', leave=True)

        if self.cfg.deformable_mesh.visu_deformable_mesh:
            # Initialize the pv plotter
            pv.global_theme.background = 'white'
            pv.global_theme.show_edges = True

            # Initialize the plotter
            verts_pv = src_mesh.verts_packed().cpu().numpy()
            faces_pv = src_mesh.faces_packed().cpu().numpy()
            # PyVista expects faces in a specific format
            faces_pv = np.hstack((np.full((faces_pv.shape[0], 1), 3), faces_pv)).flatten()
            plotter = pv.Plotter(window_size=[1920, 1080])
            mesh_pv = pv.PolyData(verts_pv, faces_pv)
            mesh_actor = plotter.add_mesh(mesh_pv, color='lightblue', show_edges=True)
            for cur_lidar in cur_lidars:
                cur_lidar_np = cur_lidar.squeeze(0).cpu().numpy()
                point_cloud = pv.PolyData(cur_lidar_np)
                plotter.add_points(point_cloud, color='red', point_size=5)
            plotter.camera_position = [(0, -10., -10.), (0, 0, 0), (0, 0, 1)]  # Set pre
            plotter.camera.elevation += 30
            plotter.show(auto_close=False, interactive_update=True)  # Keep the plotter op

        for i in loop:
            start = time.time_ns()
            optimizer.zero_grad()

            pitch = torch.zeros(len(valid_cars), device=device)
            roll = torch.zeros(len(valid_cars), device=device)
            angles = torch.stack([pitch, -deform_verts2[:, 6] - thetas, roll], dim=1)
            rot_matrices = euler_angles_to_matrix(angles, convention="XYZ")  # Shape: [batch_size, 3, 3]

            verts = src_mesh.clone().verts_packed()
            verts = verts.reshape(len(valid_cars), -1, 3)

            if i > 100:
                verts = torch.bmm(verts * deform_verts2[:, 3:6].unsqueeze(1), rot_matrices)
            else:
                verts = torch.bmm(verts, rot_matrices)

            verts_shifted = verts + (deform_verts2[:, :3].unsqueeze(1) + offsets.unsqueeze(1))

            if i >= 250:
                off = torch.stack((padding, deform_verts, padding), dim=2)
                verts_final = verts_shifted + (off * 10)
            else:
                verts_final = verts_shifted

            new_src_mesh = Meshes(verts=list(verts_final), faces=faces_list)

            if self.cfg.deformable_mesh.visu_deformable_mesh:
                verts_updated = new_src_mesh.verts_packed().detach().cpu().numpy()
                mesh_pv.points = verts_updated
                plotter.update()

            sample_src = sample_points_from_meshes(new_src_mesh, 10000)

            loss_chamfer = self.tfl_knn_batch(cur_lidars, sample_src, cur_weights)
            loss_tfl_new = self.tfl_new_batch(cur_lidars, sample_src, -deform_verts2[:, 6] - thetas)
            loss_edge = self.mesh_edge_loss_v2_batch(new_src_mesh, target_length=0.00)
            loss_symmetry = self.symmetry_loss(verts_final, mirror_index, device)
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

            loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_laplacian * w_laplacian + loss_symmetry * w_symmetry + w_tfl_new * loss_tfl_new
            loop.set_description('total_loss = %.6f' % loss)

            loss.backward()
            optimizer.step()
            loop.update(i)

            print(f"Time taken for iteration {i}: {(time.time_ns() - start) / 1e6} ms")

        pitch = torch.zeros(len(valid_cars), device=device)
        roll = torch.zeros(len(valid_cars), device=device)
        angles = torch.stack([pitch, deform_verts2[:, 6] + thetas, roll], dim=1)
        rot_matrices = euler_angles_to_matrix(angles, convention="XYZ")  # Shape: [batch_size, 3, 3]

        verts = src_mesh.clone().verts_packed()
        verts = verts.reshape(len(valid_cars), -1, 3)
        verts = torch.bmm(verts * deform_verts2[:, 3:6].unsqueeze(1), rot_matrices)
        verts = verts + (deform_verts2[:, :3].unsqueeze(1) + offsets.unsqueeze(1))
        verts = verts + (torch.stack((padding, deform_verts, padding), dim=2) * 10)

        heights = verts[:, :, 1].max(dim=1)[0] - verts[:, :, 1].min(dim=1)[0]
        for i in range(len(valid_indices)):
            cars[valid_indices[i]].x = deform_verts2[i, 0].cpu().item() + cars[valid_indices[i]].x
            cars[valid_indices[i]].y = deform_verts2[i, 1].cpu().item() + cars[valid_indices[i]].y
            cars[valid_indices[i]].z = deform_verts2[i, 2].cpu().item() + cars[valid_indices[i]].z
            cars[valid_indices[i]].theta = deform_verts2[i, 6].cpu().item() + cars[valid_indices[i]].theta

            cars[valid_indices[i]].length *= deform_verts2[i, 5].cpu().item()
            cars[valid_indices[i]].height = heights[i].cpu().item()
            cars[valid_indices[i]].width *= deform_verts2[i, 3].cpu().item()

            cars[valid_indices[i]].length = np.clip(cars[valid_indices[i]].length, 0.6 * self.cfg.templates.template_length, 1.25 * self.cfg.templates.template_length)
            cars[valid_indices[i]].height = np.clip(cars[valid_indices[i]].height, 0.6 * self.cfg.templates.template_height, 1.25 * self.cfg.templates.template_height)
            cars[valid_indices[i]].width = np.clip(cars[valid_indices[i]].width, 0.6 * self.cfg.templates.template_width, 1.25 * self.cfg.templates.template_width)

        return cars

    def mesh_edge_loss_v2(self, meshes, target_length: float = 0.0):
        """
        Computes mesh edge length regularization loss averaged across all meshes
        in a batch. Each mesh contributes equally to the final loss, regardless of
        the number of edges per mesh in the batch by weighting each mesh with the
        inverse number of edges. For example, if mesh 3 (out of N) has only E=4
        edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
        contribute to the final loss.

        Args:
            meshes: Meshes object with a batch of meshes.
            target_length: Resting value for the edge length.

        Returns:
            loss: Average loss across the batch. Returns 0 if meshes contains
            no meshes or all empty meshes.
        """
        if meshes.isempty():
            return torch.tensor(
                [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
            )

        N = len(meshes)
        edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
        verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
        edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
        num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

        # Determine the weight for each edge based on the number of edges in the
        # mesh it corresponds to.
        # TODO (nikhilar) Find a faster way of computing the weights for each edge
        # as this is currently a bottleneck for meshes with a large number of faces.
        weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
        weights = 1.0 / weights.float()

        verts_edges = verts_packed[edges_packed]
        v0, v1 = verts_edges.unbind(1)
        diffs = v0 - v1
        diffs_squared = diffs.abs()
        shifted_diffs = diffs_squared - torch.mean(diffs_squared, dim=0)
        loss = shifted_diffs.norm(dim=1, p=2)

        loss = loss + (diffs.norm(dim=1, p=2) - torch.mean(diffs.norm(dim=1, p=2), dim=0)).abs()
        loss = loss * weights

        return loss.sum() / N

    def mesh_edge_loss_v2_batch(self, meshes, target_length: float = 0.0):
        """
        Computes mesh edge length regularization loss averaged across all meshes
        in a batch. Each mesh contributes equally to the final loss, regardless of
        the number of edges per mesh in the batch by weighting each mesh with the
        inverse number of edges. For example, if mesh 3 (out of N) has only E=4
        edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
        contribute to the final loss.

        Args:
            meshes: Meshes object with a batch of meshes.
            target_length: Resting value for the edge length.

        Returns:
            loss: Average loss across the batch. Returns 0 if meshes contains
            no meshes or all empty meshes.
        """
        if meshes.isempty():
            return torch.tensor(
                [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
            )

        N = len(meshes)
        edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
        verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
        edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
        num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

        # Determine the weight for each edge based on the number of edges in the
        # mesh it corresponds to.
        # TODO (nikhilar) Find a faster way of computing the weights for each edge
        # as this is currently a bottleneck for meshes with a large number of faces.
        weights = torch.zeros((N, num_edges_per_mesh[0]), device=meshes.device, dtype=torch.float32)
        weights = weights + (1.0 / num_edges_per_mesh[0])

        verts_edges = verts_packed[edges_packed].reshape(N, -1, 2, 3)
        v0, v1 = verts_edges.unbind(2)
        diffs = v0 - v1
        diffs_squared = diffs.abs()
        shifted_diffs = diffs_squared - torch.mean(diffs_squared, dim=1).unsqueeze(1)
        loss = shifted_diffs.norm(dim=2, p=2)

        loss = loss + (diffs.norm(dim=2, p=2) - torch.mean(diffs.norm(dim=2, p=2), dim=1).unsqueeze(1)).abs()
        loss = loss * weights

        return loss.sum() / N

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

    def plot_pointcloud(self, pcloud, title=""):
        # Sample points uniformly from the surface of the mesh.
        x, y, z = pcloud.clone().detach().cpu().squeeze().unbind(1)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(x, z, -y)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_title(title)
        ax.view_init(190, 30)
        plt.show()

    def plot_mesh(self, mesh, title=""):
        # Sample points uniformly from the surface of the mesh.
        points = sample_points_from_meshes(mesh, 5000)
        x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(x, z, -y)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_title(title)
        ax.view_init(190, 30)
        plt.show()

    def move_pcloud_to_center(self, pcloud):
        pcloud = pcloud.squeeze(0)
        max_x, max_y, max_z = pcloud.max(0)[0]
        min_x, min_y, min_z = pcloud.min(0)[0]
        centroid = pcloud.mean(0)
        if pcloud.is_cuda:
            center = torch.tensor([(max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2], device='cuda')
        else:
            center = torch.tensor([(max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2], device='cpu')
        pcloud -= centroid
        pcloud = pcloud.unsqueeze(0)
        return pcloud, centroid

    def move_pcloud_to_center_numpy(self, pcloud):
        max_x, max_y, max_z = np.max(pcloud, axis=0)
        min_x, min_y, min_z = np.min(pcloud, axis=0)
        centroid = np.mean(pcloud, axis=0)
        pcloud -= centroid
        return pcloud, centroid

    def visu_in_open3d(self, src_mesh, target_pcloud):
        points = sample_points_from_meshes(src_mesh, 5000)
        points = points.clone().detach().cpu().squeeze().numpy()

        #visualize the src_mesh as mesh in open3d
        vertices = src_mesh.verts_list()
        faces = src_mesh.faces_list()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices[0].detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(faces[0].detach().cpu().numpy())
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        color = np.zeros_like(points)
        color[:, 0] = 1
        pcd.colors = o3d.utility.Vector3dVector(color)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(target_pcloud.clone().detach().cpu().squeeze().numpy())
        color = np.zeros_like(target_pcloud.clone().detach().cpu().squeeze().numpy())
        color[:, 1] = 1
        pcd2.colors = o3d.utility.Vector3dVector(color)

        #Show center and the axis of the world
        lines = [[0, 1], [0, 2], [0, 3]]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd, pcd2, line_set])

    def tfl(self, cur_lidar, src_pcloud):
        #First mirror the pcloud
        centroid = src_pcloud.squeeze(0).mean(dim=0)
        mirrored_pcloud = src_pcloud.clone().squeeze(0) - centroid
        mirrored_pcloud[:, 0] *= -1
        mirrored_pcloud = mirrored_pcloud + centroid
        mirrored_pcloud = mirrored_pcloud.unsqueeze(0)

        distances = torch.cdist(cur_lidar, src_pcloud).squeeze(0)
        closest_dist, _ = torch.min(distances, dim=0)
        closest_dist2, _ = torch.min(distances, dim=1)
        closest_dist = torch.sigmoid(10. * closest_dist) - 0.5
        closest_dist2 = torch.sigmoid(10. * closest_dist2) - 0.5

        #distances_mirrored = torch.cdist(cur_lidar, mirrored_pcloud).squeeze(0)
        #closest_dist_mirrored, _ = torch.min(distances_mirrored, dim=0)
        #closest_dist_mirrored2, _ = torch.min(distances_mirrored, dim=1)
        #closest_dist_mirrored = torch.sigmoid(1. * closest_dist_mirrored) - 0.5
        #closest_dist_mirrored2 = torch.sigmoid(1. * closest_dist_mirrored2) - 0.5

        loss = closest_dist.mean() + closest_dist2.mean()  #+ closest_dist_mirrored.mean() + closest_dist_mirrored2.mean()
        return loss

    def tfl_knn(self, cur_lidar, src_pcloud):
        # First mirror the pcloud

        # Use knn_points to find the nearest neighbors
        knn_output = knn_points(cur_lidar, src_pcloud, K=10)
        distances = knn_output.dists.squeeze(0)
        closest_dist = torch.sigmoid(10 * distances) - 0.5
        #knn_output2 = knn_points(src_pcloud, cur_lidar, K=50)
        #distances2 = knn_output2.dists.squeeze(0)
        #closest_dist2 = torch.sigmoid(10 * distances2) - 0.5

        #knn_output_mirrored = knn_points(cur_lidar, mirrored_pcloud, K=10)
        #distances_mirrored = knn_output_mirrored.dists.squeeze(0)
        #knn_output_mirrored2 = knn_points(mirrored_pcloud, cur_lidar, K=10)
        #distances_mirrored2 = knn_output_mirrored2.dists.squeeze(0)
        #closest_dist_mirrored = torch.sigmoid(distances_mirrored) - 0.5
        #closest_dist_mirrored2 = torch.sigmoid(distances_mirrored2) - 0.5

        loss = closest_dist.mean() #+ closest_dist2.mean()  #+ closest_dist_mirrored.mean() + closest_dist_mirrored2.mean()
        #loss = closest_dist.mean() + closest_dist_mirrored.mean()
        return loss

    def tfl_knn_batch(self, cur_lidar, src_pcloud, cur_weights):
        # First mirror the pcloud

        # Use knn_points to find the nearest neighbors
        knn_output = knn_points(cur_lidar, src_pcloud, K=10)
        distances = knn_output.dists
        closest_dist = torch.sigmoid(10 * distances) - 0.5

        loss_per_batch = closest_dist.mean(dim=2)
        loss_per_batch = loss_per_batch * cur_weights
        loss_per_batch = loss_per_batch.mean(dim=1)
        loss = loss_per_batch.mean()
        return loss

    def tfl_new_batch(self, cur_lidar, src_pcloud, thetas):
        #First create the two axis
        e1 = torch.stack((torch.cos(thetas), torch.zeros_like(thetas), torch.sin(thetas)), dim=1)
        e2 = torch.stack((-torch.sin(thetas), torch.zeros_like(thetas), torch.cos(thetas)), dim=1)

        cur_lidar_e1 = torch.bmm(cur_lidar, e1.unsqueeze(-1)).squeeze(2)
        cur_lidar_e1 = torch.stack((cur_lidar_e1, cur_lidar[:, :, 1]), dim=2)
        cur_lidar_e2 = torch.bmm(cur_lidar, e2.unsqueeze(-1)).squeeze(2)
        cur_lidar_e2 = torch.stack((cur_lidar_e2, cur_lidar[:, :, 1]), dim=2)

        src_pcloud_e1 = torch.bmm(src_pcloud, e1.unsqueeze(-1)).squeeze(2)
        src_pcloud_e1 = torch.stack((src_pcloud_e1, src_pcloud[:, :, 1]), dim=2)
        src_pcloud_e2 = torch.bmm(src_pcloud, e2.unsqueeze(-1)).squeeze(2)
        src_pcloud_e2 = torch.stack((src_pcloud_e2, src_pcloud[:, :, 1]), dim=2)

        # Use knn_points to find the nearest neighbors
        knn_output_e1 = knn_points(src_pcloud_e1, cur_lidar_e1, K=10)
        distances_e1 = knn_output_e1.dists
        #closest_dist_e1 = torch.sigmoid(10 * distances_e1) - 0.5

        knn_output_e2 = knn_points(src_pcloud_e2,cur_lidar_e2, K=10)
        distances_e2 = knn_output_e2.dists
        #closest_dist_e2 = torch.sigmoid(10 * distances_e2) - 0.5

        #loss_per_batch = torch.maximum(closest_dist_e1.mean(dim=2), closest_dist_e2.mean(dim=2))
        loss_per_batch = distances_e1.mean(dim=2) + distances_e2.mean(dim=2)
        loss_per_batch = loss_per_batch.mean(dim=1)
        loss = loss_per_batch.mean()
        return loss


    def symmetry_loss(self, verts, mirror_index, device):
        mirrored = verts[:, mirror_index, :]
        loss = (verts[:, :, 1] - mirrored[:, :, 1]).abs().mean()
        return loss
