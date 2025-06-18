"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
import time

import numpy
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy
import numpy as np

from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                            accuracy, get_world_size, interpolate,
                            is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .depthaware_transformer import build_depthaware_transformer
from .depth_predictor import DepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from lib.losses.focal_loss import sigmoid_focal_loss
from .dn_components import prepare_for_dn, dn_post_process, compute_dn_loss
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from PIL import Image, ImageFile

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MonoDETR(nn.Module):
    """ This is the MonoDETR module that performs monocualr 3D object detection """
    def __init__(self, backbone, depthaware_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, init_box=False, use_dab=False, group_num=11, two_stage_dino=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage MonoDETR
        """
        super().__init__()
 
        self.num_queries = num_queries
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.two_stage_dino = two_stage_dino
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation
        self.use_dab = use_dab

        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if not two_stage:
            if two_stage_dino:
                self.query_embed = None
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries * group_num, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries * group_num, 6)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.num_classes = num_classes

        if self.two_stage_dino:        
            _class_embed = nn.Linear(hidden_dim, num_classes)
            _bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
            # init the two embed layers
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(num_classes) * bias_value
            nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)   
            self.depthaware_transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.depthaware_transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (depthaware_transformer.decoder.num_layers + 1) if two_stage else depthaware_transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d  
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.depthaware_transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """

        features, pos = self.backbone(images) #This is last three layers of the ResNet Backbone, adding to this the 256 dims positional encoding is added
        #Masks looks always False in this stage.
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            # This projects all features (512, 1024, 2048) to hidden_dim which is 256
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        #If you dont have four feature levels (you dont, you have 3) just replicate the last feature level...
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.training:
                tgt_all_embed=tgt_embed = self.tgt_embed.weight           # nq, 256
                refanchor = self.refpoint_embed.weight      # nq, 4
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1) 
                
            else:
                tgt_all_embed=tgt_embed = self.tgt_embed.weight[:self.num_queries]         
                refanchor = self.refpoint_embed.weight[:self.num_queries]  
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1) 
        elif self.two_stage_dino:
            query_embeds = None
        else:
            if self.training:
                query_embeds = self.query_embed.weight
            else:
                # only use one group in inference
                query_embeds = self.query_embed.weight[:self.num_queries]

        pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = self.depth_predictor(srcs, masks[1], pos[1])
        
        hs, init_reference, inter_references, inter_references_dim, enc_outputs_class, enc_outputs_coord_unact = self.depthaware_transformer(
            srcs, masks, pos, query_embeds, depth_pos_embed, depth_pos_embed_ip)#, attn_mask)

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

          
            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)

            # depth_geo
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

            # depth_reg
            depth_reg = self.depth_embed[lvl](hs[lvl])

            # depth_map
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_center3d,
                mode='bilinear',
                align_corners=True).squeeze(1)

            # depth average + sigma
            depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1) + depth_map) / 3,
                                    depth_reg[:, :, 1: 2]], -1)
            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)
  
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out #, mask_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, group_num=11, cfg=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map
        self.group_num = group_num

        self.template_width = 1.63
        self.template_height = 1.526
        self.template_length = 3.88

        self.silhouette_renderer = None
        #self.rendering_templates = self.load_rendering_templates()

        if cfg is not None:
            self.use_tfl = cfg['use_tfl']
            self.use_mask_loss = cfg['use_mask_loss']
            self.mask_loss = cfg['mask_loss']
        else:
            self.use_tfl = False
            self.use_mask_loss = False
            self.mask_loss = 'DICE'

    def load_rendering_templates(self):
        fiat = load_objs_as_meshes(["../pseudo_label_generator/3d/data/fiat3_voxel.obj"], device='cuda')
        passat = load_objs_as_meshes(["../pseudo_label_generator/3d/data/passat_voxel.obj"], device='cuda')
        mpv = load_objs_as_meshes(["../pseudo_label_generator/3d/data/minivan_smooth.obj"], device='cuda')
        suv = load_objs_as_meshes(["../pseudo_label_generator/3d/data/suv_smooth.obj"], device='cuda')

        meshes = [fiat, passat, suv, mpv]
        mesh_rots = [[180, 0, -90], [180, -90, 0], [90, 180, 0], [-90, 90, 180]]
        out_meshes = []
        for mesh_idx, mesh in enumerate(meshes):
            verts = mesh.verts_packed()  # (V, 3) tensor of vertices
            faces = mesh.faces_packed()  # (F, 3) tensor of faces

            center_x = (verts[:, 0].min() + verts[:, 0].max()) / 2
            center_y = (verts[:, 1].min() + verts[:, 1].max()) / 2
            center_z = (verts[:, 2].min() + verts[:, 2].max()) / 2

            transform = Transform3d(device="cuda")
            transform = transform.translate(*[-center_x, -center_y, -center_z])
            verts = transform.transform_points(verts)

            transform = Transform3d(device="cuda")
            transform = transform.rotate_axis_angle(mesh_rots[mesh_idx][0], axis='X', degrees=True)
            verts = transform.transform_points(verts)
            transform = Transform3d(device="cuda")
            transform = transform.rotate_axis_angle(mesh_rots[mesh_idx][1], axis='Y', degrees=True)
            verts = transform.transform_points(verts)
            transform = Transform3d(device="cuda")
            transform = transform.rotate_axis_angle(mesh_rots[mesh_idx][2], axis='Z', degrees=True)
            verts = transform.transform_points(verts)

            width = verts[:, 0].max() - verts[:, 0].min()
            height = verts[:, 1].max() - verts[:, 1].min()
            length = verts[:, 2].max() - verts[:, 2].min()

            verts[:, 0] *= self.template_width / width
            verts[:, 1] *= self.template_height / height
            verts[:, 2] *= self.template_length / length

            # visualize the mesh
            transformed_mesh = Meshes(verts=[verts], faces=[faces])
            out_meshes.append(transformed_mesh)

            # verts_np = verts.cpu().numpy()
            # faces_np = faces.cpu().numpy()
            # mesh_o3d = o3d.geometry.TriangleMesh()
            # mesh_o3d.vertices = o3d.utility.Vector3dVector(verts_np)
            # mesh_o3d.triangles = o3d.utility.Vector3iVector(faces_np)
            # mesh_o3d.compute_vertex_normals()
            # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0.0, 0.0, 0.0])
            # o3d.visualization.draw_geometries([mesh_o3d, coordinate_frame])

        return out_meshes

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, info=None):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, info=None):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes, info=None):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, info=None):
        
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes, info=None):

        idx = self._get_src_permutation_idx(indices)
   
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance  
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes 
        return losses  
    
    def loss_dims(self, outputs, targets, indices, num_boxes, info=None):

        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes, info=None):

        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes, info=None):
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([80, 24, 80, 24], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)
        
        losses = dict()

        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def loss_tfl(self, outputs, targets, indices, num_boxes, info=None):
        if self.use_tfl or self.use_mask_loss:
            idx = self._get_src_permutation_idx(indices)
            losses = {}
            #start = time.time_ns()
            preds = self.decode_preds(outputs, info, idx, indices)
            #end = time.time_ns()
            #print("Decoding preds took: ", (end - start) / 1e6, "ms")
            #start = time.time_ns()
            matching, processed_cars, moving, angle_if_moving, masks_out = self.perform_matching(preds, info)
            #end = time.time_ns()
            #print("Matching took: ", (end - start) / 1e6, "ms")
            #start = time.time_ns()
            templates = self.get_templates(preds, info, matching, moving)
            #end = time.time_ns()
            #print("Getting templates took: ", (end - start) / 1e6, "ms")
            #start = time.time_ns()
            templates_mask = self.get_templates_mask(preds, info)

            if self.use_tfl:
                loss = self.compute_tfl_loss(templates, matching, processed_cars)
                losses['loss_tfl'] = loss
            else:
                losses['loss_tfl'] = torch.tensor(0., device='cuda', dtype=torch.float32, requires_grad=True)

            if self.use_mask_loss:
                loss_masks = self.compute_mask_loss(info, templates_mask, masks_out, processed_cars, preds, matching)
                losses['loss_mask'] = loss_masks
            else:
                losses['loss_mask'] = torch.tensor(0., device='cuda', dtype=torch.float32, requires_grad=True)

            if len(preds) == 1 and info['lidar_bool'][0, 0]:
                self.visu_preds(preds, info, templates)

            return losses
        else:
            return {'loss_tfl': torch.tensor(0., device='cuda', dtype=torch.float32, requires_grad=True),
                    'loss_mask': torch.tensor(0., device='cuda', dtype=torch.float32, requires_grad=True)}

    def compute_mask_loss(self, info, templates_mask, masks, processed_cars, preds, matching):
        final_loss = []
        max_number_of_renders = 10
        renders_per_batch = max_number_of_renders // len(preds)
        for batch_idx in range(len(preds)):
            self.init_rendering(info, batch_idx)

            cur_masks = masks[batch_idx]
            cur_templates = templates_mask[batch_idx]
            cur_matching = matching[batch_idx]
            cur_processed_cars = processed_cars[batch_idx]

            cur_masks = self.decode_masks(cur_masks, info, batch_idx)
            batch_loss = []

            if len(cur_masks) == 0:
                continue

            renders_per_mask = torch.zeros(len(cur_masks), dtype=torch.int8, device='cuda')
            equal_distribution = renders_per_batch // len(cur_masks)
            renders_per_mask[:] += equal_distribution
            remaining_renders = renders_per_batch - equal_distribution * len(cur_masks)
            for i in range(remaining_renders):
                idx = torch.randint(0, len(cur_masks), (1,))
                renders_per_mask[idx] += 1

            for mask_idx in range(len(cur_masks)):
                idxs_to_compute = torch.argwhere(cur_matching == mask_idx).squeeze()
                cur_template_fiat = cur_templates[0][idxs_to_compute]
                cur_template_sedan = cur_templates[1][idxs_to_compute]
                cur_template_mpv = cur_templates[2][idxs_to_compute]
                cur_template_suv = cur_templates[3][idxs_to_compute]
                if idxs_to_compute.dim() == 0:
                    cur_template_fiat = cur_template_fiat.unsqueeze(0)
                    cur_template_sedan = cur_template_sedan.unsqueeze(0)
                    cur_template_mpv = cur_template_mpv.unsqueeze(0)
                    cur_template_suv = cur_template_suv.unsqueeze(0)
                    idxs_to_compute = idxs_to_compute.unsqueeze(0)
                elif len(idxs_to_compute) == 0:
                    continue

                to_render = renders_per_mask[mask_idx]
                if to_render > 0:
                    perm = torch.randperm(idxs_to_compute.size(0))
                    idx = perm[:to_render]
                    cur_template_fiat = cur_template_fiat[idx]
                    cur_template_sedan = cur_template_sedan[idx]
                    cur_template_mpv = cur_template_mpv[idx]
                    cur_template_suv = cur_template_suv[idx]
                else:
                    continue

                target_mask, target_weight = self.create_target_mask(cur_masks, cur_processed_cars, mask_idx)

                rendered_masks = self.render_templates([cur_template_fiat, cur_template_sedan, cur_template_mpv, cur_template_suv], info, batch_idx)

                #for i in range(len(rendered_masks)):
                #    for z in range(len(rendered_masks[i])):
                #        self.vizu_masks(target_mask, rendered_masks[i][z], 'target_mask_' + str(info['img_id'][batch_idx]) + '_' + str(i) + '_' + str(z))

                #compute loss
                tmp_loss = []
                for i in range(len(rendered_masks)):
                    if self.mask_loss == 'BCE':
                        target_mask_expanded = target_mask.unsqueeze(0).expand(rendered_masks[i].shape[0], -1, -1)
                        loss_tmp = F.binary_cross_entropy(rendered_masks[i], target_mask_expanded, weight=target_weight, reduction='none')
                        loss_tmp = loss_tmp.sum(dim=-1)
                        loss_tmp = loss_tmp.sum(dim=-1)
                    elif self.mask_loss == 'DICE':
                        target_mask_expanded = target_mask.unsqueeze(0).expand(rendered_masks[i].shape[0], -1, -1)
                        target_weight_expanded = target_weight.unsqueeze(0).expand(rendered_masks[i].shape[0], -1, -1)
                        loss_tmp = self.dice_loss(rendered_masks[i], target_mask_expanded, target_weight_expanded)
                    else:
                        loss_tmp = torch.tensor(0., device='cuda', dtype=torch.float32, requires_grad=True)
                    tmp_loss.append(loss_tmp)

                tmp_loss = torch.stack(tmp_loss, dim=0) / torch.clip(torch.sum(target_mask), min=1)
                min_loss = torch.min(tmp_loss, dim=0).values
                #print('min_loss ', min_loss.shape)
                batch_loss.append(min_loss)
            if len(batch_loss) == 0:
                continue
            batch_loss = torch.cat(batch_loss)
            #print("batch_loss ", batch_loss.shape)
            final_loss.append(batch_loss)
        if len(final_loss) == 0:
            return torch.tensor(0., device='cuda', dtype=torch.float32, requires_grad=True)
        final_loss = torch.cat(final_loss)
        #print("final_loss ", final_loss.shape)
        #print(final_loss)
        return final_loss.mean()

    def dice_loss(self, pred, target, target_weight):
        #print(pred.shape, target.shape, target_weight.shape)
        pred = pred * target_weight
        target = target * target_weight

        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice_score = (2. * intersection + 1.0) / (union + 1.0)
        dice_loss = 1 - dice_score

        return dice_loss


    def vizu_masks(self, mask1, mask2, filename):
        rgba_image = torch.zeros((mask1.shape[0], mask1.shape[1], 4), dtype=torch.uint8)
        print(mask1.shape, mask2.shape)
        # Define colors with alpha (transparency)
        mask1 = mask1 > 0.5
        mask2 = mask2 > 0.5
        color1 = torch.tensor([255, 0, 0, 128], dtype=torch.uint8)  # Red with 50% opacity
        color2 = torch.tensor([0, 0, 255, 128], dtype=torch.uint8)

        rgba_image[mask1 == 1] = color1
        rgba_image[mask2 == 1] = color2

        Image.fromarray(rgba_image.cpu().numpy(), mode='RGBA').save(filename + '.png')

    def render_templates(self, templates, info, batch_idx):
        output = []
        for temp_idx in range(len(templates)):
            faces = self.rendering_templates[temp_idx].faces_packed()
            faces_expanded = faces.expand(templates[temp_idx].shape[0], -1, -1)
            tmp_mesh = Meshes(
                verts=templates[temp_idx],
                faces=faces_expanded,
            )

            #vertices = tmp_mesh.verts_packed().cpu().detach().numpy()  # Convert vertices to a numpy array
            #faces = tmp_mesh.faces_packed().cpu().detach().numpy()  # Convert faces to a numpy array

            # Create an Open3D mesh
            #o3d_mesh = o3d.geometry.TriangleMesh()
            #o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            #o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

            # Optional: Compute vertex normals for better visualization
            #o3d_mesh.compute_vertex_normals()

            # Add a coordinate frame for reference
            #coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

            # Visualize the mesh and coordinate frame
            #o3d.visualization.draw_geometries([o3d_mesh, coordinate_frame], window_name="Open3D Mesh Visualization")

            image = self.silhouette_renderer(tmp_mesh)[..., 3]
            image[image < 0.5] = image[image < 0.5] + 1e-3
            output.append(image)
        return output

    def decode_masks(self, masks, info, batch_idx):
        out_masks = []

        for mask in masks:
            mask = Image.fromarray(mask.cpu().numpy().transpose().astype('uint8') * 255)
            img_size = info['img_size'][batch_idx]
            trans = info['affine'][batch_idx]
            mask_transformed = mask.transform(
                size=tuple(img_size.tolist()),
                method=Image.AFFINE,
                data=tuple(trans.reshape(-1).tolist()),
                resample=Image.NEAREST
            )
            mask_transformed = (np.array(mask_transformed) > 127)
            out_masks.append(torch.tensor(mask_transformed, dtype=torch.bool, device='cuda'))

        return out_masks

    def create_target_mask(self, masks, cars, idx):
        target_mask = torch.zeros_like(masks[idx], dtype=torch.float32)
        target_weight = torch.ones_like(masks[idx], dtype=torch.float32)

        centers = []
        for car in cars:
            center = torch.median(car, dim=0).values
            centers.append(center)
        centers = torch.stack(centers)
        dists = torch.linalg.norm(centers, dim=1)

        target_mask[masks[idx] == True] = 1.

        for i in range(len(masks)):
            if i != idx:
                if dists[i] > dists[idx]:
                    target_weight[masks[i] == True] = 0.0
            else:
                continue

        return target_mask, target_weight

    def init_rendering(self, info, batch_idx):
        tmp_K = torch.eye(4)
        tmp_K[:3, :4] = info['calib_P2'][batch_idx]

        T_final = torch.eye(4, device='cuda', dtype=torch.float32)

        rot_W_to_cam = T_final[:3, :3]
        T_W_to_cam = T_final[:3, 3]

        w, h = info['img_size'][batch_idx]
        h = h.item()
        w = w.item()

        cameras = PerspectiveCameras(device='cuda', R=rot_W_to_cam.unsqueeze(0), T=T_W_to_cam.unsqueeze(0),
                                     focal_length=((tmp_K[0, 0], tmp_K[1, 1]),),
                                     principal_point=((tmp_K[0, 2], tmp_K[1, 2]),),
                                     image_size=((h, w),), in_ndc=False)

        blend_params = BlendParams(sigma=1e-5)
        raster_settings = RasterizationSettings(
            image_size=(h, w),
            faces_per_pixel=20,
            max_faces_per_bin=10000,  # TODO Play with this parameter.
            blur_radius=2e-4
        )
        # We can add a point light in front of the object.
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

    def compute_tfl_loss(self, templates, matching, processed_cars):
        loss = torch.tensor(0., dtype=torch.float32, device='cuda', requires_grad=True)
        for batch_idx in range(len(templates)):
            cur_templates = templates[batch_idx]
            cur_matching = matching[batch_idx]
            cur_processed_cars = processed_cars[batch_idx]

            cur_loss = self.tfl_knn(cur_templates, cur_matching, cur_processed_cars)
            loss = loss + cur_loss
        return loss

    def tfl_knn(self, templates, matching, processed_cars):
        loss = torch.tensor(0., dtype=torch.float32, device='cuda', requires_grad=True)
        for idx in range(len(processed_cars)):
            idxs_to_compute = torch.argwhere(matching == idx).squeeze()
            cur_templates = templates[idxs_to_compute]
            if idxs_to_compute.dim() == 0: cur_templates = cur_templates.unsqueeze(0)
            elif len(idxs_to_compute) == 0: continue

            cur_template = cur_templates.view(-1, cur_templates.shape[2], 3)
            cur_car = processed_cars[idx]
            cur_car = cur_car.unsqueeze(0).expand(cur_template.shape[0], -1, -1)

            knn_output = knn_points(cur_template, cur_car, K=1)
            knn_output2 = knn_points(cur_car, cur_template, K=1)
            distances = knn_output.dists.squeeze(-1)
            distances2 = knn_output2.dists.squeeze(-1)
            distances = distances.view(cur_templates.shape[0], cur_templates.shape[1], cur_templates.shape[2])
            distances2 = distances2.view(cur_templates.shape[0], cur_templates.shape[1], cur_car.shape[1])

            distances = torch.sigmoid(10. * distances) - 0.5
            distances2 = torch.sigmoid(10. * distances2) - 0.5

            temp_to_scan = torch.sum(distances, dim=2) / cur_template.shape[1]
            scan_to_temp = torch.sum(distances2, dim=2) / cur_car.shape[1]

            cur_pred_loss = torch.min(temp_to_scan + scan_to_temp, dim=1).values
            loss = loss + torch.sum(cur_pred_loss) / cur_pred_loss.shape[0]

        return loss

    def precise_tfl(self, templates, matching, processed_cars):
        loss = torch.tensor([0.], dtype=torch.float32, device='cuda', requires_grad=True)
        for idx in range(len(processed_cars)):
            idxs_to_compute = torch.argwhere(matching == idx).squeeze()
            cur_templates = templates[idxs_to_compute]

            for templ_idx in range(len(cur_templates)):
                cur_template = cur_templates[templ_idx]
                cur_car = processed_cars[idx]

                dists = torch.cdist(cur_template.view(-1, 3), cur_car)
                dists = dists.view(cur_templates.shape[0], cur_template.shape[1], cur_car.shape[0])

                closest_dist_temp_to_scan1, _ = torch.min(dists, dim=1)
                closest_dist_scan_to_temp1, _ = torch.min(dists, dim=2)

                closest_dist_temp_to_scan1 = torch.sigmoid(10. * closest_dist_temp_to_scan1) - 0.5
                closest_dist_scan_to_temp1 = torch.sigmoid(10. * closest_dist_scan_to_temp1) - 0.5

                temp_to_scan = torch.sum(closest_dist_temp_to_scan1, dim=1) / cur_template.shape[0]
                scan_to_temp = torch.sum(closest_dist_scan_to_temp1, dim=1) / cur_car.shape[0]

                cur_pred_loss = torch.min(temp_to_scan + scan_to_temp)
                loss = loss + cur_pred_loss

        return loss / len(templates)

    def precise_tfl_batch(self, templates, matching, processed_cars):
        #TODO if needed finish this function
        for idx in range(len(processed_cars)):
            idxs_to_compute = torch.argwhere(matching == idx).squeeze()
            cur_templates = templates[idxs_to_compute]
            templates_flattened = cur_templates.view(-1, 1000, 3).view(-1, 3)
            dists = torch.cdist(templates_flattened, processed_cars[idx])
            pairwise_distances = dists.view(cur_templates.shape[0], 4, 1000, processed_cars[idx].shape[0])

        return torch.tensor([0.]).cuda()

    def get_templates_mask(self, preds, info):
        templates_out = []

        templates = self.rendering_templates
        verts_fiat = templates[0].verts_packed()  # (V, 3) tensor of vertices
        verts_passat = templates[1].verts_packed()  # (V, 3) tensor of vertices
        verts_suv = templates[2].verts_packed()  # (V, 3) tensor of vertices
        verts_mpv = templates[3].verts_packed()  # (V, 3) tensor of vertices

        for batch_idx in range(len(preds)):
            cur_preds = preds[batch_idx]
            cur_templates_fiat = torch.zeros((cur_preds.shape[0], len(verts_fiat), 3), dtype=torch.float32,device='cuda')
            cur_templates_passat = torch.zeros((cur_preds.shape[0], len(verts_passat), 3), dtype=torch.float32,device='cuda')
            cur_templates_suv = torch.zeros((cur_preds.shape[0], len(verts_suv), 3), dtype=torch.float32, device='cuda')
            cur_templates_mpv = torch.zeros((cur_preds.shape[0], len(verts_mpv), 3), dtype=torch.float32, device='cuda')
            cur_templates_fiat[:] = verts_fiat
            cur_templates_passat[:] = verts_passat
            cur_templates_suv[:] = verts_suv
            cur_templates_mpv[:] = verts_mpv

            # first scale them
            scale_height = cur_preds[:, 3] / info["templates_dimensions"][batch_idx][0]
            scale_width = cur_preds[:, 4].detach() / info["templates_dimensions"][batch_idx][1]
            scale_length = cur_preds[:, 5].detach() / info["templates_dimensions"][batch_idx][2]

            cur_templates_fiat[:, :, 1] *= scale_height.unsqueeze(1)
            cur_templates_fiat[:, :, 0] *= scale_width.unsqueeze(1)
            cur_templates_fiat[:, :, 2] *= scale_length.unsqueeze(1)
            cur_templates_passat[:, :, 1] *= scale_height.unsqueeze(1)
            cur_templates_passat[:, :, 0] *= scale_width.unsqueeze(1)
            cur_templates_passat[:, :, 2] *= scale_length.unsqueeze(1)
            cur_templates_suv[:, :, 1] *= scale_height.unsqueeze(1)
            cur_templates_suv[:, :, 0] *= scale_width.unsqueeze(1)
            cur_templates_suv[:, :, 2] *= scale_length.unsqueeze(1)
            cur_templates_mpv[:, :, 1] *= scale_height.unsqueeze(1)
            cur_templates_mpv[:, :, 0] *= scale_width.unsqueeze(1)
            cur_templates_mpv[:, :, 2] *= scale_length.unsqueeze(1)

            # then rotate them
            r = torch.zeros((cur_preds.shape[0], 3, 3), dtype=torch.float32, device='cuda')
            r[:, 0, 0] = torch.cos(cur_preds[:, 6].detach() + torch.pi / 2.)
            r[:, 0, 2] = torch.sin(cur_preds[:, 6].detach() + torch.pi / 2.)
            r[:, 2, 0] = -torch.sin(cur_preds[:, 6].detach() + torch.pi / 2.)
            r[:, 2, 2] = torch.cos(cur_preds[:, 6].detach() + torch.pi / 2.)
            r[:, 1, 1] = 1.

            cur_templates_fiat = torch.matmul(cur_templates_fiat, r.transpose(1, 2))
            cur_templates_passat = torch.matmul(cur_templates_passat, r.transpose(1, 2))
            cur_templates_suv = torch.matmul(cur_templates_suv, r.transpose(1, 2))
            cur_templates_mpv = torch.matmul(cur_templates_mpv, r.transpose(1, 2))

            # then translate them
            cur_templates_fiat[:, :, 0] += cur_preds[:, 0].unsqueeze(1)
            cur_templates_fiat[:, :, 1] += cur_preds[:, 1].unsqueeze(1)
            cur_templates_fiat[:, :, 2] += cur_preds[:, 2].unsqueeze(1)
            cur_templates_passat[:, :, 0] += cur_preds[:, 0].unsqueeze(1)
            cur_templates_passat[:, :, 1] += cur_preds[:, 1].unsqueeze(1)
            cur_templates_passat[:, :, 2] += cur_preds[:, 2].unsqueeze(1)
            cur_templates_suv[:, :, 0] += cur_preds[:, 0].unsqueeze(1)
            cur_templates_suv[:, :, 1] += cur_preds[:, 1].unsqueeze(1)
            cur_templates_suv[:, :, 2] += cur_preds[:, 2].unsqueeze(1)
            cur_templates_mpv[:, :, 0] += cur_preds[:, 0].unsqueeze(1)
            cur_templates_mpv[:, :, 1] += cur_preds[:, 1].unsqueeze(1)
            cur_templates_mpv[:, :, 2] += cur_preds[:, 2].unsqueeze(1)

            # then rotate them again because of the rendering
            r = torch.zeros((cur_preds.shape[0], 3, 3), dtype=torch.float32, device='cuda')
            r[:, 0, 0] = torch.cos(torch.tensor(torch.pi))
            r[:, 0, 1] = -torch.sin(torch.tensor(torch.pi))
            r[:, 1, 0] = torch.sin(torch.tensor(torch.pi))
            r[:, 1, 1] = torch.cos(torch.tensor(torch.pi))
            r[:, 2, 2] = 1.

            cur_templates_fiat = torch.matmul(cur_templates_fiat, r.transpose(1, 2))
            cur_templates_passat = torch.matmul(cur_templates_passat, r.transpose(1, 2))
            cur_templates_suv = torch.matmul(cur_templates_suv, r.transpose(1, 2))
            cur_templates_mpv = torch.matmul(cur_templates_mpv, r.transpose(1, 2))

            templates_out.append([cur_templates_fiat, cur_templates_passat, cur_templates_suv, cur_templates_mpv])

        return templates_out

    def get_templates(self, preds, info, matching, moving):
        templates_out = []

        templates = info['lidar_templates'][0].cuda()
        for batch_idx in range(len(preds)):
            cur_preds = preds[batch_idx]
            cur_templates = torch.zeros((cur_preds.shape[0], 4, 1000, 3), dtype=torch.float32, device='cuda')

            cur_templates[:] = templates

            #first scale them
            scale_height = cur_preds[:, 3].detach() / info["templates_dimensions"][batch_idx][0]
            scale_width = cur_preds[:, 4].detach() / info["templates_dimensions"][batch_idx][1]
            scale_length = cur_preds[:, 5].detach() / info["templates_dimensions"][batch_idx][2]

            cur_templates[:, :, :, 1] *= scale_height.unsqueeze(1).unsqueeze(2)
            cur_templates[:, :, :, 0] *= scale_width.unsqueeze(1).unsqueeze(2)
            cur_templates[:, :, :, 2] *= scale_length.unsqueeze(1).unsqueeze(2)

            #then rotate them
            r = torch.zeros((cur_preds.shape[0], 3, 3), dtype=torch.float32, device='cuda')
            r[:, 0, 0] = torch.cos(cur_preds[:, 6] + torch.pi / 2.)
            r[:, 0, 2] = torch.sin(cur_preds[:, 6] + torch.pi / 2.)
            r[:, 2, 0] = -torch.sin(cur_preds[:, 6] + torch.pi / 2.)
            r[:, 2, 2] = torch.cos(cur_preds[:, 6] + torch.pi / 2.)
            r[:, 1, 1] = 1.

            # If moving then dont compute rotational gradient
            for idx in range(len(moving[batch_idx])):
                if moving[batch_idx][idx]:
                    idxs_to_detach = torch.argwhere(matching[batch_idx] == idx).squeeze()
                    r[idxs_to_detach] = r[idxs_to_detach].detach()

            templates_tensor_reshaped = cur_templates.view(cur_templates.shape[0], cur_templates.shape[1] * cur_templates.shape[2], 3)
            rotated_templates = torch.matmul(templates_tensor_reshaped, r.transpose(1, 2))
            cur_templates = rotated_templates.view(cur_templates.shape[0], cur_templates.shape[1], cur_templates.shape[2], 3)

            #then translate them
            cur_templates[:, :, :, 0] += cur_preds[:, 0].unsqueeze(1).unsqueeze(2)
            cur_templates[:, :, :, 1] += cur_preds[:, 1].unsqueeze(1).unsqueeze(2)
            cur_templates[:, :, :, 2] += cur_preds[:, 2].unsqueeze(1).unsqueeze(2)

            templates_out.append(cur_templates)

        return templates_out

    def perform_matching(self, preds, info):
        matching_out = []
        processed_cars_out = []
        moving_out = []
        angle_if_moving_out = []
        masks_out = []
        for batch_idx in range(len(preds)):
            #start = time.time_ns()
            cur_lidar = info['lidar'][batch_idx].to('cuda', non_blocking=True)
            moving = info['moving'][batch_idx].to('cuda', non_blocking=True)
            angle_if_moving = info['angle_if_moving'][batch_idx].to('cuda', non_blocking=True)
            masks = info['masks'][batch_idx].to('cuda', non_blocking=True)

            zero_cars_mask = (cur_lidar == 0).all(dim=2).all(dim=1)  # Shape: (50,)
            non_zero_cars_mask = ~zero_cars_mask  # Invert the mask
            cur_lidar = cur_lidar[non_zero_cars_mask]
            moving = moving[non_zero_cars_mask]
            angle_if_moving = angle_if_moving[non_zero_cars_mask]
            masks = masks[non_zero_cars_mask]

            processed_cars = []
            #print("Processing cars whole took: ", (time.time_ns() - start) / 1e6, "ms")
            #start = time.time_ns()
            for idx, car_data in enumerate(cur_lidar):
                # car_data shape: (10000, 3)
                # Create mask for non-zero points
                non_zero_points_mask = ~(car_data == 0).all(dim=1)  # Shape: (10000,)
                # Select non-zero points
                car_data = car_data[non_zero_points_mask]  # Shape: (M_i, 3), M_i ≤ 10000
                processed_cars.append(car_data)
            #print("Processing cars each took: ", (time.time_ns() - start) / 1e6, "ms")
            #start = time.time_ns()
            if len(processed_cars) == 0:
                matching_out.append(torch.zeros((0,), dtype=torch.uint8, device='cuda'))
                processed_cars_out.append([])
                moving_out.append([])
                angle_if_moving_out.append([])
                masks_out.append([])
                continue

            centers = torch.stack([car_data.median(dim=0).values for car_data in processed_cars]).to('cuda')  # Shape: (50, 3)

            cur_preds = preds[batch_idx]

            dists = torch.cdist(cur_preds[:, :3], centers)

            min_dist, matching = torch.min(dists, dim=1)

            idxs_to_delete = torch.argwhere(min_dist > 5.0)
            matching[idxs_to_delete] = -1
            matching_out.append(matching.to(torch.int8))
            #for i in range(len(processed_cars)):
            #    processed_cars[i] = processed_cars[i].to('cuda', non_blocking=True)
            processed_cars_out.append(processed_cars)
            moving_out.append(moving)
            angle_if_moving_out.append(angle_if_moving)
            masks_out.append(masks)
            #print("Matching took x: ", (time.time_ns() - start) / 1e6, "ms")

        return matching_out, processed_cars_out, moving_out, angle_if_moving_out, masks_out

    def decode_preds(self, outputs, info, idx, indices):
        out = []
        for batch_idx in range(len(indices)):
            # Projected 3D center
            src_3dcenter = outputs['pred_boxes'][batch_idx, :, 0: 2][indices[batch_idx][0]]
            src_3dcenter[:, 0] = src_3dcenter[:, 0] * info['resolution'][batch_idx, 0]
            src_3dcenter[:, 1] = src_3dcenter[:, 1] * info['resolution'][batch_idx, 1]

            # pad src_3dcenter to n x 3
            src_3dcenter = torch.cat([src_3dcenter, torch.ones(src_3dcenter.shape[0], 1).cuda()], dim=1)
            affine_transform = info['affine_inv'][batch_idx].to(torch.float32).cuda()
            src_3dcenter = torch.mm(affine_transform, src_3dcenter.T).T

            if info['flip'][batch_idx]:
                src_3dcenter[:, 0] = info['img_size'][batch_idx, 0] - src_3dcenter[:, 0]

            # Depth of 3D center
            src_depths = outputs['pred_depth'][batch_idx, indices[batch_idx][0]]
            depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1]
            depth_input = depth_input / info['scale_depth'][batch_idx]

            x = ((src_3dcenter[:, 0] - info['calib_P2'][batch_idx, 0, 2]) * depth_input) / info['calib_P2'][batch_idx, 0, 0] + (
                    info['calib_P2'][batch_idx, 0, 3] / -info['calib_P2'][batch_idx, 0, 0])
            y = ((src_3dcenter[:, 1] - info['calib_P2'][batch_idx, 1, 2]) * depth_input) / info['calib_P2'][batch_idx, 1, 1] + (
                    info['calib_P2'][batch_idx, 1, 3] / -info['calib_P2'][batch_idx, 1, 1])

            center = torch.cat((x.unsqueeze(1), y.unsqueeze(1), depth_input.unsqueeze(1)), dim=1)

            # 3D spatial dimensions
            dims = outputs['pred_3d_dim'][batch_idx, indices[batch_idx][0]]  # height, width, length
            # src_3dcenter[:, 1] = src_3dcenter[:, 1] + src_dims[:, 0] / 2.

            # Angle
            heading_input = outputs['pred_angle'][batch_idx, indices[batch_idx][0]]
            heading_input = heading_input.view(-1, 24)

            # Classification
            heading_input_cls = heading_input[:, 0:12]
            heading_input_cls = heading_input_cls.sigmoid()
            highest_probable_bin = torch.argmax(heading_input_cls, dim=1)
            # Regression
            heading_input_res = heading_input[:, 12:24]

            heading = heading_input_res[torch.arange(
                heading_input_res.shape[0]), highest_probable_bin] + highest_probable_bin.float().flatten() * (
                                  2 * torch.pi / 12)

            corner_2d_norm = torch.zeros((src_3dcenter.shape[0], 4)).cuda().to(torch.float32)
            corner_2d_norm[:, 0] = outputs['pred_boxes'][batch_idx, :, 0][indices[batch_idx][0]] - outputs['pred_boxes'][batch_idx, :, 2][indices[batch_idx][0]]
            corner_2d_norm[:, 1] = outputs['pred_boxes'][batch_idx, :, 1][indices[batch_idx][0]] - outputs['pred_boxes'][batch_idx, :, 4][indices[batch_idx][0]]
            corner_2d_norm[:, 2] = outputs['pred_boxes'][batch_idx, :, 0][indices[batch_idx][0]] + outputs['pred_boxes'][batch_idx, :, 3][indices[batch_idx][0]]
            corner_2d_norm[:, 3] = outputs['pred_boxes'][batch_idx, :, 1][indices[batch_idx][0]] + outputs['pred_boxes'][batch_idx, :, 5][indices[batch_idx][0]]

            corner_2d_norm[:, 0] = corner_2d_norm[:, 0] * info['resolution'][batch_idx, 0]
            corner_2d_norm[:, 1] = corner_2d_norm[:, 1] * info['resolution'][batch_idx, 1]
            corner_2d_norm[:, 2] = corner_2d_norm[:, 2] * info['resolution'][batch_idx, 0]
            corner_2d_norm[:, 3] = corner_2d_norm[:, 3] * info['resolution'][batch_idx, 1]

            corner_2d = corner_2d_norm
            tmp_corner = torch.cat([corner_2d[:, :2], torch.ones(src_3dcenter.shape[0], 1).cuda()], dim=1)
            affine_transform = info['affine_inv'][batch_idx].to(torch.float32).cuda()
            corner_2d[:, :2] = torch.mm(affine_transform, tmp_corner.T).T
            tmp_corner = torch.cat([corner_2d[:, 2:4], torch.ones(src_3dcenter.shape[0], 1).cuda()], dim=1)
            affine_transform = info['affine_inv'][batch_idx].to(torch.float32).cuda()
            corner_2d[:, 2:4] = torch.mm(affine_transform, tmp_corner.T).T

            if info['flip'][batch_idx]:
                corner_2d_temp = corner_2d.clone()
                corner_2d[:, 0] = info['img_size'][batch_idx, 0] - corner_2d_temp[:, 2]
                corner_2d[:, 2] = info['img_size'][batch_idx, 0] - corner_2d_temp[:, 0]

            u = (corner_2d[:, 0] + corner_2d[:, 2]) / 2
            if info['flip'][batch_idx]:
                heading = -heading + torch.pi
            heading = heading + torch.arctan2(u - info['calib_P2'][batch_idx, 0, 2], info['calib_P2'][batch_idx, 0, 0])


            preds = torch.zeros((src_3dcenter.shape[0], 7)).cuda()
            preds[:, 0:3] = center
            preds[:, 3:6] = dims
            preds[:, 6] = heading
            out.append(preds)

        return out

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
            'tfl': self.loss_tfl
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask_dict=None, info=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_num=group_num)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses

        losses = {}
        for loss in self.losses:
            #ipdb.set_trace()
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, info=info))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_num=group_num)
                for loss in self.losses:
                    if loss == 'depth_map':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, info=info)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def visu_preds(self, preds, infos, templates):
        for num_idx, img_idx in enumerate(infos["img_id"]):
            lidar_scan = infos['lidar_whole'][num_idx]#.cuda()
            #lidar_scan[:, 3] = 1.
            #velo_to_cam = infos['calib_V2C'][num_idx].cuda()
            #lidar_scan = torch.mm(velo_to_cam, lidar_scan[num_idx].T).T
            #lidar_scan = torch.mm(infos['calib_R0'][num_idx].cuda(), lidar_scan.T).T

            current_preds = preds[num_idx].cpu().detach().numpy()
            pred_boxes = []

            for pred in current_preds:
                center3D = np.array([pred[0], pred[1], pred[2]])
                yaw = pred[6] + np.pi / 2.
                dimensions = np.array([pred[4], pred[3], pred[5]])

                #create open3d bounding box from these parameters
                r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
                bbox = o3d.geometry.OrientedBoundingBox(center3D, r.as_matrix(), dimensions)
                bbox.color = np.array([1, 0, 0])
                pred_boxes.append(bbox)

            #current_templates = templates[num_idx]
            #for templ in current_templates:
                #lidar_scan = torch.cat((lidar_scan, templ[0]), 0)

            #Show lidar scan in open3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lidar_scan.cpu().detach().numpy())

            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.add_geometry(pcd)
            for k in range(len(pred_boxes)):
                visualizer.add_geometry(pred_boxes[k])
            # visualizer.get_render_option().point_size = 5  # Adjust the point size if necessary
            visualizer.get_render_option().background_color = np.asarray([0, 0, 0])  # Set background to black
            visualizer.get_view_control().set_front([0, -0.3, -0.5])
            visualizer.get_view_control().set_lookat([0, 0, 1])
            visualizer.get_view_control().set_zoom(0.05)
            visualizer.get_view_control().set_up([0, -1, 0])
            visualizer.get_view_control().camera_local_translate(5., 0., 8.)
            visualizer.run()
            visualizer.destroy_window()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    depthaware_transformer = build_depthaware_transformer(cfg)

    # depth prediction module
    depth_predictor = DepthPredictor(cfg)

    model = MonoDETR(
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        two_stage=cfg['two_stage'],
        init_box=cfg['init_box'],
        use_dab = cfg['use_dab'],
        two_stage_dino=cfg['two_stage_dino'])

    # matcher
    matcher = build_matcher(cfg)

    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']
    weight_dict['loss_tfl'] = cfg['tfl_loss_coef']
    weight_dict['loss_mask'] = cfg['mask_loss_coef']
    
    # dn loss
    if cfg['use_dn']:
        weight_dict['tgt_loss_ce']= cfg['cls_loss_coef']
        weight_dict['tgt_loss_bbox'] = cfg['bbox_loss_coef']
        weight_dict['tgt_loss_giou'] = cfg['giou_loss_coef']
        weight_dict['tgt_loss_angle'] = cfg['angle_loss_coef']
        weight_dict['tgt_loss_center'] = cfg['3dcenter_loss_coef']

    # TODO this is a hack
    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map', 'tfl']
    
    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses,
        cfg=cfg)

    device = torch.device(cfg['device'])
    criterion.to(device)
    
    return model, criterion
