import os
import kornia.feature as KF
import kornia as K
import numpy as np
import torch
import cv2
import pickle
from kornia.geometry.ransac import RANSAC
from kornia.geometry.transform import warp_perspective
from kornia.core.check import KORNIA_CHECK_SHAPE, KORNIA_CHECK_IS_TENSOR
from kornia.core import Tensor, where, pad

from anno_V3 import AutoLabel3D

class Stitching(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)

        if self.generate_homographies_only:
            if self.cfg.general.device == 'gpu':
                self.matcher = KF.LoFTR(pretrained="outdoor").cuda()
            else:
                self.matcher = KF.LoFTR(pretrained="outdoor")
        else:
            self.matcher = None

        self.ransac = RANSAC("homography", inl_th=2, confidence=0.99)

    def perform_img_stitching(self):
        stitched_imgs = []
        if not self.generate_homographies_only:
            if self.cfg.general.device == 'cpu':
                with open(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + '_cpu.pkl', 'rb') as f:
                    homos_all = pickle.load(f)
                    if not self.cfg.general.supress_debug_prints:
                        print("Loading all homographies to .pkl")
                with open(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + "_mkpts_cpu" + '.pkl', 'rb') as f:
                    mkpts_all = pickle.load(f)
                    if not self.cfg.general.supress_debug_prints:
                        print("Loading final homographies to .pkl")
                with open(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + "_optim_cpu" + '.pkl', 'rb') as f:
                    homos_optim = pickle.load(f)
                    if not self.cfg.general.supress_debug_prints:
                        print("Loading optim homographies to .pkl")
            else:
                with open(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + '.pkl', 'rb') as f:
                    homos_all = pickle.load(f)
                    if not self.cfg.general.supress_debug_prints:
                        print("Loading all homographies to .pkl")
                with open(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + "_mkpts" + '.pkl',
                          'rb') as f:
                    mkpts_all = pickle.load(f)
                    if not self.cfg.general.supress_debug_prints:
                        print("Loading final homographies to .pkl")
                with open(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + "_optim" + '.pkl',
                          'rb') as f:
                    homos_optim = pickle.load(f)
                    if not self.cfg.general.supress_debug_prints:
                        print("Loading optim homographies to .pkl")

        else:
            if (os.path.isfile(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + '.pkl') and
                    os.path.isfile(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + "_mkpts" + '.pkl') and
                    os.path.isfile(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + "_optim" + '.pkl')):
                return None, None
            homos_all = []
            mkpts_all = []

        for i in range(0, len(self.waymo_frame)):
            if not self.cfg.general.supress_debug_prints:
                print("Stitching frame: ", i)
            imgs = self.get_imgs(i)

            stitched_tmp = []
            homos = []
            mkpts = []

            #Lets merge them together
            for z in range(4):
                if z == 0:
                    #Left front to front
                    imgs_tmp = [imgs[0], imgs[1]]
                    left_to_right = True
                elif z == 1:
                    #Left to left front
                    imgs_tmp = [imgs[1], imgs[2]]
                    left_to_right = True
                elif z == 2:
                    #Right front to front
                    imgs_tmp = [imgs[2], imgs[3]]
                    left_to_right = False
                elif z == 3:
                    #Right to right front
                    imgs_tmp = [imgs[3], imgs[4]]
                    left_to_right = False
                else:
                    raise Exception("Image index out of the range")

                out_shape = (imgs_tmp[0].shape[-2] + self.cfg.image_stitching.height_pxl_pad, imgs_tmp[0].shape[-1] + imgs_tmp[1].shape[-1] + self.cfg.image_stitching.width_pxl_pad)

                if not self.generate_homographies_only:
                    homo = homos_optim[z]
                    if self.cfg.general.device == "gpu":
                        homo = homo.cuda()
                else:
                    homo, mkpts0, mkpts1 = self.get_homography_all(imgs_tmp, left_to_right, z)
                    homos.append(homo)
                    mkpts.append([mkpts0, mkpts1])
                    continue

                if left_to_right:
                    src_img = warp_perspective(imgs_tmp[0], homo, out_shape)

                    final_img = torch.zeros((3, out_shape[0], out_shape[1]))
                    final_img[:,:,:] = src_img[:,:,:]
                    final_img[:, (self.cfg.image_stitching.height_pxl_pad // 2):1280 + (self.cfg.image_stitching.height_pxl_pad // 2), -1920:] = imgs_tmp[1]

                    stitched_tmp.append(final_img)

                else:
                    src_img = warp_perspective(imgs_tmp[1], homo, out_shape)

                    final_img = torch.zeros((3, out_shape[0], out_shape[1]))
                    final_img[:, :, :] = src_img[:, :, :]
                    final_img[:, (self.cfg.image_stitching.height_pxl_pad // 2):1280 + (self.cfg.image_stitching.height_pxl_pad // 2), :1920] = imgs_tmp[0]

                    stitched_tmp.append(final_img)

            for k in range(len(stitched_tmp)):
                stitched_tmp[k] *= 255
                stitched_tmp[k] = stitched_tmp[k].to(torch.uint8)

            stitched_imgs.append(stitched_tmp)
            if self.generate_homographies_only:
                homos_all.append(homos)
                mkpts_all.append(mkpts)
        # save homos_all as .pkl
        if self.generate_homographies_only:
            with open(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + '.pkl', 'wb') as f:
                pickle.dump(homos_all, f)
                if not self.cfg.general.supress_debug_prints:
                    print("Dumping all homographies to .pkl")
            with open(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + "_mkpts" + '.pkl', 'wb') as f:
                pickle.dump(mkpts_all, f)
                if not self.cfg.general.supress_debug_prints:
                    print("Dumping feature points to .pkl")

        if self.generate_homographies_only:
            homos_optim = self.find_the_best_homo(homos_all, mkpts_all)

        return stitched_imgs, homos_optim

    def find_the_best_homo(self, homo, mkpts):
        final_loss = torch.zeros((len(self.waymo_frame), 4), dtype=torch.float64).cuda()

        for i in range(0, len(self.waymo_frame)):
            if not self.cfg.general.supress_debug_prints:
                print("Frame: ", i)

            for z in range(4):
                if z == 0:
                    # Left front to front
                    left_to_right = True
                elif z == 1:
                    # Left to left front
                    left_to_right = True
                elif z == 2:
                    # Right front to front
                    left_to_right = False
                else:
                    # Right to right front
                    left_to_right = False

                if left_to_right:
                    for idx_homo in range(0, len(self.waymo_frame)):
                        cur_homo = homo[idx_homo][z]
                        cur_mktps0 = mkpts[idx_homo][z][0]
                        cur_mktps1 = mkpts[idx_homo][z][1]

                        if len(cur_mktps0) < 4:
                            continue

                        best_model, inliers, score = self.verify(cur_mktps0, cur_mktps1, cur_homo, inl_th=2)

                        final_loss[idx_homo, z] += -score

                else:
                    for idx_homo in range(0, len(self.waymo_frame)):
                        cur_homo = homo[idx_homo][z]
                        cur_mktps0 = mkpts[idx_homo][z][0]
                        cur_mktps1 = mkpts[idx_homo][z][1]

                        if len(cur_mktps0) < 4:
                            continue

                        best_model, inliers, score = self.verify(cur_mktps1, cur_mktps0, cur_homo, inl_th=2)

                        final_loss[idx_homo, z] += -score

        best_idx0 = torch.argmin(final_loss[:, 0])
        best_idx1 = torch.argmin(final_loss[:, 1])
        best_idx2 = torch.argmin(final_loss[:, 2])
        best_idx3 = torch.argmin(final_loss[:, 3])

        if not self.cfg.general.supress_debug_prints:
            print("Best performance")
            print(best_idx0, best_idx1, best_idx2, best_idx3)

        homos_final = [homo[best_idx0][0], homo[best_idx1][1], homo[best_idx2][2], homo[best_idx3][3]]

        with open(self.cfg.paths.merged_frames_path + "homographies/" + self.file_name + "_optim" + '.pkl', 'wb') as f:
            pickle.dump(homos_final, f)
            if not self.cfg.general.supress_debug_prints:
                print("Dumping final homographies to .pkl")

        return homos_final

    def verify(self, kp1, kp2, models, inl_th):
        batch_size = models.shape[0]
        errors = self.oneway_transfer_error(kp1.expand(batch_size, -1, 2), kp2.expand(batch_size, -1, 2), models)
        inl = errors <= inl_th
        models_score = inl.to(kp1).sum(dim=1)
        best_model_idx = models_score.argmax()
        best_model_score = models_score[best_model_idx].item()
        model_best = models[best_model_idx].clone()
        inliers_best = inl[best_model_idx]
        return model_best, inliers_best, best_model_score

    def oneway_transfer_error(self, pts1, pts2, H, squared: bool = True, eps: float = 1e-8):
        #TAKEN FROM KORNIA
        KORNIA_CHECK_SHAPE(H, ["B", "3", "3"])

        if pts1.size(-1) == 3:
            pts1 = self.convert_points_from_homogeneous(pts1)

        if pts2.size(-1) == 3:
            pts2 = self.convert_points_from_homogeneous(pts2)

        # From Hartley and Zisserman, Error in one image (4.6)
        # dist = \sum_{i} ( d(x', Hx)**2)
        pts1_in_2: Tensor = self.transform_points(H, pts1)
        error_squared: Tensor = (pts1_in_2 - pts2).pow(2).sum(dim=-1)
        if squared:
            return error_squared
        return (error_squared + eps).sqrt()

    def get_homography_all(self, imgs, left_to_right, z):
        input_dict = {
            "image0": K.color.rgb_to_grayscale(imgs[0]),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(imgs[1]),
        }

        with torch.inference_mode():
            correspondences = self.matcher(input_dict)
            mkpts0 = correspondences["keypoints0"]
            mkpts1 = correspondences["keypoints1"]
            #conf = correspondences["confidence"]

            if mkpts0.shape[0] < 4:
                return self.precomputed_homos_approx[z].unsqueeze(0), [], []

            if left_to_right:
                mkpts1[:, 0] += 1920 + self.cfg.image_stitching.width_pxl_pad
                mkpts1[:, 1] += self.cfg.image_stitching.height_pxl_pad // 2
                homo, _ = self.ransac(mkpts0, mkpts1)

            else:
                mkpts0[:, 1] += self.cfg.image_stitching.height_pxl_pad // 2
                homo, _ = self.ransac(mkpts1, mkpts0)

            if torch.equal(homo, torch.zeros(3, 3, dtype=torch.float32, device=torch.device('cuda'))):
                return self.precomputed_homos_approx[z].unsqueeze(0), [], []

            homo = homo[None]

        return homo, mkpts0, mkpts1


    def get_imgs(self, i):
        images_sorted = sorted(self.waymo_frame[i].images, key=lambda z: z.name)
        # Pre-allocate memory for arr_temp
        if self.cfg.general.device == "gpu":
            images = torch.zeros((len(images_sorted), 1, 3, 1280, 1920), dtype=torch.float32, device='cuda')
        else:
            images = torch.zeros((len(images_sorted), 1, 3, 1280, 1920), dtype=torch.float32)

        for index, image in enumerate(images_sorted):
            decoded_image = tf.image.decode_jpeg(image.image).numpy()

            # Open the image, convert
            img = np.array(decoded_image, dtype=np.uint8)
            img = np.moveaxis(img, -1, 0)  # the model expects the image to be in channel first format

            if index >= 3:
                padding = 1280 - img.shape[1]
                img = np.pad(img, ((0, 0), (padding, 0), (0, 0)), 'constant')

            # Use in-place operation to fill arr_temp
            if index == 0:
                images[2, 0].copy_(torch.from_numpy(img), non_blocking=True)
            elif index == 1:
                images[1, 0].copy_(torch.from_numpy(img), non_blocking=True)
            elif index == 2:
                images[3, 0].copy_(torch.from_numpy(img), non_blocking=True)
            elif index == 3:
                images[0, 0].copy_(torch.from_numpy(img), non_blocking=True)
            else:
                images[4, 0].copy_(torch.from_numpy(img), non_blocking=True)

        # Normalize the image

        images = images / 255.0

        #for z in range(len(arr_temp)):

            #src_img = arr_temp[z][0].cpu().numpy().transpose(1, 2, 0)
            #src_img = np.uint8(src_img * 255)
            #cv2.imwrite('orig_image' + str(i) + "_" + str(z) + '.png', src_img)

        #self.interactive_align(images)
        return images

    def convert_points_from_homogeneous(self, points, eps: float = 1e-8):
        # TAKEN FROM KORNIA
        if not isinstance(points, Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(points)}")

        if len(points.shape) < 2:
            raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

        # we check for points at max_val
        z_vec: Tensor = points[..., -1:]

        # set the results of division by zeror/near-zero to 1.0
        # follow the convention of opencv:
        # https://github.com/opencv/opencv/pull/14411/files
        mask: Tensor = torch.abs(z_vec) > eps
        scale = where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

        return scale * points[..., :-1]

    def transform_points(self, trans_01, points_1):
        #TAKEN FROM KORNIA
        KORNIA_CHECK_IS_TENSOR(trans_01)
        KORNIA_CHECK_IS_TENSOR(points_1)
        if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
            raise ValueError(
                f"Input batch size must be the same for both tensors or 1. Got {trans_01.shape} and {points_1.shape}"
            )
        if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
            raise ValueError(f"Last input dimensions must differ by one unit Got{trans_01} and {points_1}")

        # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
        shape_inp = list(points_1.shape)
        points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
        trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
        # We expand trans_01 to match the dimensions needed for bmm
        trans_01 = torch.repeat_interleave(trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0)
        # to homogeneous
        points_1_h = self.convert_points_to_homogeneous(points_1)  # BxNxD+1
        # transform coordinates
        points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
        points_0_h = torch.squeeze(points_0_h, dim=-1)
        # to euclidean
        points_0 = self.convert_points_from_homogeneous(points_0_h)  # BxNxD
        # reshape to the input shape
        shape_inp[-2] = points_0.shape[-2]
        shape_inp[-1] = points_0.shape[-1]
        points_0 = points_0.reshape(shape_inp)
        return points_0

    def convert_points_to_homogeneous(self, points):
        #TAKEN FROM KORNIA
        if not isinstance(points, Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(points)}")
        if len(points.shape) < 2:
            raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

        return pad(points, [0, 1], "constant", 1.0)



