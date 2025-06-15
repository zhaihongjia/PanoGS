import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
import time
from argparse import ArgumentParser
from datetime import datetime
import random
import yaml
from munch import munchify
import cv2
import imgviz
import numpy as np
from tqdm import tqdm

import torch
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.config_utils import load_config
from datasets.load_func import load_dataset
from tools.vis_feature import calc_pca

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.camera_utils import Camera

import matplotlib.pyplot as plt
from decoders.decoder import FeatureDecoder

class GSRecon:
    def __init__(self, config, save_dir):
        self.config = config
        self.save_dir = save_dir
        self.device = "cuda"
        self.iteration_count = 0 # (if not fix_ply) only used for 3D Gaussian update
        self.dataset = load_dataset(config=config)

        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (model_params, opt_params, pipeline_params,)
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        # init params
        self.viewpoints = {}
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.scaling_slider = 1.0
        self.cameras_extent = 6.0

        self.fix_ply = self.config['Training']['fix_ply']
        if self.fix_ply:
            self.gaussians.init_from_ply(self.dataset.gt_ply)

        # debug and eval flags
        self.save_debug = config['Results']['save_debug']
        self.kf_inter = self.config["Training"]["kf_inter"]

        self.set_hyperparams()

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (self.cameras_extent * self.config["Training"]["init_gaussian_extent"])
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (self.cameras_extent * self.config["Training"]["gaussian_extent"])
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
        self.frames_to_optimize = self.config["Training"]["pose_window"]

    # @brief: rendering: rgb, depth, opacity, elipsoid
    def save_rendering(self, viewpoint):
        save_id = viewpoint.uid
        debug_save_path = os.path.join(self.save_dir, 'rendering')
        rgb_save_path = os.path.join(debug_save_path, 'rgb')
        depth_save_path = os.path.join(debug_save_path, 'depth')
        opacity_save_path = os.path.join(debug_save_path, 'opacity')

        os.makedirs(rgb_save_path, exist_ok=True)
        os.makedirs(depth_save_path, exist_ok=True)
        os.makedirs(opacity_save_path, exist_ok=True)

        rendering_data = render(
                viewpoint,
                self.gaussians,
                self.pipeline_params,
                self.background,
                self.scaling_slider,)

        # rgb
        rgb = ((torch.clamp(rendering_data["render"], min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        cv2.imwrite(os.path.join(rgb_save_path, 'rgb_{}.png'.format(save_id)), rgb[:,:,::-1])

        # depth
        depth = rendering_data["depth"]
        depth = depth[0, :, :].detach().cpu().numpy()
        max_depth = np.max(depth)
        depth = imgviz.depth2rgb(depth, min_value=0.1, max_value=max_depth, colormap="jet")
        depth = torch.from_numpy(depth)
        depth = torch.permute(depth, (2, 0, 1)).float()
        depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        cv2.imwrite(os.path.join(depth_save_path, 'depth_{}.png'.format(save_id)), depth[:,:,::-1])

        # opacity
        opacity = rendering_data["opacity"]
        opacity = opacity[0, :, :].detach().cpu().numpy()
        max_opacity = np.max(opacity)
        opacity = imgviz.depth2rgb(opacity, min_value=0.0, max_value=max_opacity, colormap="jet")
        opacity = torch.from_numpy(opacity)
        opacity = torch.permute(opacity, (2, 0, 1)).float()
        opacity = (opacity).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        cv2.imwrite(os.path.join(opacity_save_path, 'opacity_{}.png'.format(save_id)), opacity[:,:,::-1])

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        '''
        add keyframe and init 3D Gaussians
        '''
        self.gaussians.extend_from_pcd_seq(viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map)

    def reconstruction_loss(self, config, image, depth, viewpoint, opacity):
        gt_image = viewpoint.original_image.cuda()
        # print('viewpoint.depth shape: ', viewpoint.depth.shape) # [h, w]

        gt_depth = torch.from_numpy(viewpoint.depth).to(dtype=torch.float32, device=image.device)[None]
        # print('gt_depth shape: ', gt_depth.shape) # [1, h, w]
        # print('gt_depth: ', type(gt_depth))

        rgb_pixel_mask = (gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*depth.shape)
        depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

        l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
        l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

        # return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()
        return l1_rgb.mean() + l1_depth.mean()

    def mapping(self, iters=1):
        # all viewpoints        
        all_viewpoint_stack = []
        for cam_idx, viewpoint in self.viewpoints.items():
            all_viewpoint_stack.append(viewpoint)
        
        for _ in range(iters):
            self.iteration_count += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []

            for cam_idx in torch.randperm(len(all_viewpoint_stack))[:self.frames_to_optimize]:
                viewpoint = all_viewpoint_stack[cam_idx]
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )
                loss_mapping += self.reconstruction_loss(self.config, image, depth, viewpoint, opacity)

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            loss_mapping.backward()

            # deinsifying / pruning gaussians
            if not self.fix_ply:
                with torch.no_grad():
                    for idx in range(len(viewspace_point_tensor_acm)):
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]] = \
                            torch.max(self.gaussians.max_radii2D[visibility_filter_acm[idx]], radii_acm[idx][visibility_filter_acm[idx]],)
                        self.gaussians.add_densification_stats(viewspace_point_tensor_acm[idx], visibility_filter_acm[idx])

                    # update gaussian
                    update_gaussian = (self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset)
                    if update_gaussian:
                        self.gaussians.densify_and_prune(
                            self.opt_params.densify_grad_threshold,
                            self.gaussian_th,
                            self.gaussian_extent,
                            self.size_threshold,
                        )

                    # opacity reset
                    if (self.iteration_count % self.gaussian_reset) == 0 and (not update_gaussian):
                        print("Resetting the opacity of non-visible Gaussians")
                        self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)

    # @brief: refine 3DGS based on color loss
    def color_refinement(self):
        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(random.randint(0, len(viewpoint_idx_stack) - 1))
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipeline_params, self.background)
            
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (Ll1) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)

    def load_depth(self, cur_frame_idx):
        viewpoint = self.viewpoints[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > self.rgb_boundary_threshold)[None]
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def do_recon(self):
        projection_matrix = getProjectionMatrix2(
                    znear=0.01,
                    zfar=100.0,
                    fx=self.dataset.fx,
                    fy=self.dataset.fy,
                    cx=self.dataset.cx,
                    cy=self.dataset.cy,
                    W=self.dataset.width,
                    H=self.dataset.height,
                ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)

        for cur_frame_idx in range(0, len(self.dataset), self.kf_inter):
            viewpoint = Camera.init_from_dataset(self.dataset, cur_frame_idx, projection_matrix)
            self.viewpoints[cur_frame_idx] = viewpoint

            # add new kf
            if not self.fix_ply:
                depth_map = self.load_depth(cur_frame_idx)
                self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

            self.mapping(iters=self.mapping_itr_num)
            
        # final refinement
        self.color_refinement()

        if self.save_debug:
            print('Save rendering after reconstruction.')
            for test_id in range(0, len(self.dataset), self.kf_inter):
                viewpoint = Camera.init_from_dataset(self.dataset, test_id, projection_matrix)
                self.save_rendering(viewpoint)

        # save Gaussian ply
        self.gaussians.save_ply(os.path.join(self.save_dir, 'point_cloud', "point_cloud.ply"))

if __name__ == "__main__":
    parser = ArgumentParser(description="3DGS Reconstruction")
    parser.add_argument("--config", type=str)
    args = parser.parse_args(sys.argv[1:])
    config = load_config(args.config)

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])        
        path = config["Dataset"]["dataset_path"].split("/")
        scene_id = config["Dataset"]["scene_id"]

        # set save_dir
        if config['Dataset']['type'] == 'replica':
            save_dir = os.path.join(config["Results"]["save_dir"], path[-1], scene_id)
        elif config['Dataset']['type'] == 'scannet':
            save_dir = os.path.join(config["Results"]["save_dir"], path[-2], scene_id)
        else:
            print('Dataset type should be replica or scannet')
            exit()
        print('Save results to: ', save_dir)
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        
        # bak yml
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)

    recon = GSRecon(config, save_dir)
    recon.do_recon()

