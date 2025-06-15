# fuse feature
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
sys.path.append('/home/hongjia/Projects/lseg_feature_extraction')

import time
from argparse import ArgumentParser
import yaml
import open3d as o3d
import numpy as np
from tqdm import tqdm
import scipy
import json
from PIL import Image

import torch
import torchvision.transforms as transforms

from utils.config_utils import load_config
from datasets.load_func import load_dataset

from encoding.models.sseg import BaseNet
from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule

def extract_lseg_img_feature(img_dir, transform, evaluator, label='', resize_wh=None):
    # load RGB image
    image = Image.open(img_dir)
    if resize_wh is not None:
        image = image.resize(resize_wh) # (width, height)
    image = np.array(image)
    image = transform(image).unsqueeze(0)#.cuda()
    with torch.no_grad():
        outputs = evaluator.parallel_forward(image, label)
        feat_2d = outputs[0][0].half()

    return feat_2d

@torch.inference_mode()
def world2cam_pixel(points_w: np.array, intrinsic: np.array, poses: np.array):
    batch_size = 10000
    device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    intrinsic = torch.tensor(intrinsic, device=device, dtype=torch.float32)
    poses = torch.tensor(poses, device=device, dtype=torch.float32)
    poses_inv = torch.from_numpy(np.linalg.inv(poses.numpy()))   # (M, 4, 4)
    del poses

    N = points_w.shape[0]
    M = poses_inv.shape[0]
    final_points_c = np.zeros((N, M, 3), dtype=np.float32)
    uv_pixels = np.zeros((N, M, 2), dtype=int)

    for batch_start in tqdm(range(0, points_w.shape[0], batch_size)):
        points_world = torch.tensor(points_w[batch_start: batch_start+batch_size], device=device, dtype=torch.float32)
        points_world_homo = torch.cat((points_world, torch.ones((points_world.shape[0], 1), dtype=torch.float32, device=device)), 1)

        points_cam_homo = torch.matmul(poses_inv[None], points_world_homo[:, None, :, None])
        points_cam_homo = points_cam_homo[..., 0]        # (N, M, 4)
        points_cam = torch.div(points_cam_homo[..., :-1], points_cam_homo[..., [-1]])  # (N, M, 3)

        # (M, 3, 3) @ (N, M, 3, 1) = (N, M, 3, 1)
        points_pixel_homo = torch.matmul(intrinsic, points_cam[..., None])
        # (N, M, 3)
        points_pixel_homo = points_pixel_homo[..., 0]
        # (u, v) coordinate, (N, M, 2)
        points_pixel = torch.div(points_pixel_homo[..., :-1], torch.clip(points_pixel_homo[..., [-1]], min=1e-8)).round().to(torch.int32)
        final_points_c[batch_start: batch_start + batch_size] = points_cam.cpu().numpy()
        uv_pixels[batch_start: batch_start + batch_size] = points_pixel.cpu().numpy()

    torch.cuda.empty_cache()
    return final_points_c, uv_pixels

class FeatPlYFuser:
    def __init__(self, config):
        self.config = config
        self.save_dir = config["Results"]["save_dir"]
        self.depth_test = config["Training"]["depth_test"]
        self.thres_vis_dis = config['Training']['thres_vis_dis']
        self.kf_inter = self.config["Training"]["kf_inter"]

    def get_fused_pointfeat(self, points_c, uv_pixels):
        batch_size = 5000
        all_label = np.zeros([self.n_points, self.n_images], dtype=np.float32)
        all_seen_flag = np.zeros([self.n_points, self.n_images], dtype=bool)
        all_points_feat = np.zeros([self.n_points, self.n_feature_dim], dtype=np.float32)
        all_feats_var = np.zeros([self.n_points], dtype=np.float32)
        all_feats_weight = np.zeros([self.n_points], dtype=np.float32)

        # projected by batch
        for start_id in tqdm(range(0, self.n_points, batch_size)):
            p_cam0 = points_c[start_id: start_id + batch_size]
            pix0 = uv_pixels[start_id: start_id + batch_size]
            w0, h0 = np.split(pix0, 2, axis=-1)
            w0, h0 = w0[..., 0], h0[..., 0]  # (n_points_sub, n_images)
            bounded_flag = (0 <= w0) * (w0 <= self.width - 1) * (0 <= h0) * (h0 <= self.height - 1)  # (n_points_sub, n_images)

            # [n_points_sub, n_images, n_feat_dim]
            feats = np.zeros([w0.shape[0], self.n_images, self.n_feature_dim], dtype=np.float32)

            # TODO: tmp solution
            for frame_id in tqdm(range(self.n_images)):
                feats[:, frame_id, :] = self.features[frame_id][h0[:, frame_id].clip(0, self.height - 1), w0[:, frame_id].clip(0, self.width - 1)]
            
            # judge whether the point is visible
            real_depth0 = p_cam0[..., -1]  # (n_points_sub, n_images)
            # (n_points_sub, n_images), querying depths
            capture_depth = self.depths[np.arange(self.n_images), h0.clip(0, self.height - 1), w0.clip(0, self.width - 1)]
            visible_flag = np.isclose(real_depth0, capture_depth, rtol=self.thres_vis_dis)
            seen_flag = bounded_flag * visible_flag if self.depth_test else bounded_flag

            # [n_points, n_images]
            all_seen_flag[start_id: start_id + batch_size] = seen_flag

            # [n_points, n_images, 1] 
            expanded_mask = seen_flag[:, :, np.newaxis]
            # [n_points, n_images, 412]
            masked_features = feats * expanded_mask
            # [n_points, 512]
            sum_features = np.sum(masked_features, axis=1)  
            count = np.sum(expanded_mask, axis=1)  
            count = np.maximum(count, 1) 
            fused_mean = sum_features / count  

            centered_feat = (feats - fused_mean[:, np.newaxis, :]) * expanded_mask  # Shape will be [n_points, n_images, 512]
            var_features = np.sum(centered_feat ** 2, axis=1) / count  # Shape will be [n_points, 512] replace count-1, avoid divide 0

            all_points_feat[start_id: start_id + batch_size] = fused_mean
            all_feats_var[start_id: start_id + batch_size] = np.sum(var_features, axis=1) # shape [N]

        # compute weight
        all_seen_count = np.array(all_seen_flag).astype(float).sum(axis=1)
        all_seen_count = all_seen_count / (np.max(all_seen_count) - np.min(all_seen_count))
        all_feats_var = all_feats_var / (np.max(all_feats_var) - np.min(all_feats_var))
        all_feats_weight = 10 * (all_seen_count) / (all_feats_var + 1)

        return all_points_feat, all_feats_weight

    def get_feature_maps(self, ):
        img_long_side = self.dataset.width
        resize_wh = (self.dataset.width, self.dataset.height)
        transform, evaluator = load_lseg_extractor(img_long_side)

        print('resize_wh: ', resize_wh)
        feature_maps = []
        
        # sample frames
        for i in tqdm(range(0, len(self.dataset), self.kf_inter)):
            img_path = self.dataset.color_paths[i]
            # [512, H, W] -> [H, W, 512]
            feat = extract_lseg_img_feature(img_path, transform, evaluator, resize_wh=resize_wh) 
            feat = feat.detach().cpu().permute(1,2,0)
            feature_maps.append(feat)

        return feature_maps

    def fuse_feature(self):
        '''
        (1) Load dataset and reconstructed ply
        (2) Generate feature maps for keyframes
        (3) Projecet points_w into image pixels
        (3) Fusion
        '''    
        self.dataset = load_dataset(config=self.config)

        # TODO update load path
        ply_path = os.path.join(self.config['Results']['save_dir'], 'point_cloud', 'point_cloud.ply')
        ply = o3d.io.read_point_cloud(ply_path)

        self.points_w = np.array(ply.points)
        self.n_points = self.points_w.shape[0]
        self.width = self.dataset.width
        self.height = self.dataset.height

        # [N_images, 4, 4] # [N_images, H, W] # [N_images, 3, 3] # [N_images, H, W]
        self.poses, self.depths, self.intrinsics, self.masks = self.dataset.load_data_for_seg(self.kf_inter)
        self.features = self.get_feature_maps()
        self.n_feature_dim = self.features[0].shape[-1]
        self.n_images = self.poses.shape[0]

        # return: [N_points, N_images, 3] [N_points, N_images, 2]
        points_c, uv_pixels = world2cam_pixel(self.points_w, self.intrinsics, self.poses)

        # points_feat: [n_points, 512]
        # feats_weight: [n_points]
        points_feat, feats_weight = self.get_fused_pointfeat(points_c, uv_pixels)

        # save feat ply
        generated_floder = os.path.join(self.config['Dataset']['generated_floder'], config["Dataset"]["scene_id"], 'feats_weights')
        os.makedirs(generated_floder, exist_ok=True)

        feat_save_path = os.path.join(generated_floder, 'features.npy')
        weight_save_path = os.path.join(generated_floder, 'weights.npy')
        np.save(feat_save_path, points_feat)
        np.save(weight_save_path, feats_weight)

        print('Save fused feature to: ', feat_save_path)

def load_lseg_extractor(img_long_side):   
    seed = 1457
    torch.manual_seed(seed)

    module = LSegModule.load_from_checkpoint(
        checkpoint_path='/home/hongjia/Projects/lseg_feature_extraction/checkpoints/demo_e200.ckpt',
        data_path='/home/hongjia/Projects/lseg_feature_extraction/datasets',
        dataset='ade20k',
        backbone='clip_vitl16_384',
        aux=False,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=255,
        dropout=0.0,
        scale_inv=False,
        augment=False,
        no_batchnorm=False,
        widehead=True,
        widehead_hr=False,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )

    # model
    if isinstance(module.net, BaseNet):
        model = module.net
    else:
        model = module

    model = model.eval()
    model = model.cpu()
    scales = ([1])

    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]

    model.crop_size = 2*img_long_side
    model.base_size = 2*img_long_side

    evaluator = LSeg_MultiEvalModule(model, scales=scales, flip=True).cuda()
    evaluator.eval()

    transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
    return transform, evaluator

    # scannet: [512, 480, 640]
    # feat = extract_lseg_img_feature(file, transform, evaluator, resize_wh=process_size) # for scannet: use img resize

if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-view feature map fusion.")
    parser.add_argument("--config", type=str)
    args = parser.parse_args(sys.argv[1:])
    config = load_config(args.config)

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
    
    os.makedirs(save_dir, exist_ok=True)
    config['Results']['save_dir'] = save_dir

    fuser = FeatPlYFuser(config)
    fuser.fuse_feature()

