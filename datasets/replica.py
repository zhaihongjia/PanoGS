import csv
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from gaussian_splatting.utils.graphics_utils import focal2fov
from datasets import replica_class_utils 

class ReplicaDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = "cuda"
        self.dtype = torch.float32
        self.scene_id = config["Dataset"]["scene_id"]
        self.basedir = os.path.join(config["Dataset"]["dataset_path"], self.scene_id)
        self.generated_floder = config["Dataset"]["generated_floder"]

        self.color_paths = sorted(glob.glob(f"{self.basedir}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.basedir}/results/depth*.png"))
        self.num_frames = len(self.color_paths)

        self.class_names = replica_class_utils.MATTERPORT_LABELS_21 # reduced_to_scannet_names
        self.num_classes = len(self.class_names)
        self.class_colors = replica_class_utils.SCANNET_COLOR_MAP_200.values()
        self.class_text_feat = np.load(os.path.join('./datasets/replica', 'MATTERPORT_LABELS_21_dim512.npy'))

        # list of numpy array [4, 4]
        self.poses = self.load_poses(f"{self.basedir}/traj.txt")
        self.gt_semantics = np.load(f'./datasets/replica/3d_sem_ins/{self.scene_id}/semantic_labels_mp21.npy')
        self.gt_instances = np.load(f'./datasets/replica/3d_sem_ins/{self.scene_id}/instance_labels_mp21.npy')
        self.gt_ply = os.path.join(self.basedir, '..', '{}_trimesh.ply'.format(self.scene_id))

        self.mask_name = config["Dataset"]["mask_name"]
        self.mask_path = os.path.join(self.generated_floder, self.scene_id, self.mask_name, 'raw')

        # Camera prameters
        calibration = config["Dataset"]["Calibration"]
        self.resize_wh = calibration["resize_wh"]
        w_scale = float(self.resize_wh[0] / 1200.0)
        h_scale = float(self.resize_wh[1] / 680.0)
        self.depth_scale = calibration["depth_scale"]
        self.ignore_edge = calibration["ignore_edge"]
        self.crop = calibration["crop"]
        self.fx = calibration["fx"] * w_scale
        self.fy = calibration["fy"] * h_scale
        self.cx = calibration["cx"] * w_scale
        self.cy = calibration["cy"] * h_scale
        self.width = int(calibration["width"] * w_scale)
        self.height = int(calibration["height"] * h_scale)
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )

    def index_to_name(self, index):
        name = os.path.basename(self.color_paths[index])[:-4]    # frame000001.jpg
        return name

    def load_depth(self, index):
        depth = cv2.imread(self.depth_paths[index], cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, self.resize_wh, interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32) / self.depth_scale
        return depth

    def load_image(self, index):
        rgb = cv2.imread(self.color_paths[index], -1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.resize_wh)
        return rgb / 255.0

    def load_mask(self, index):
        name = self.index_to_name(index)
        mask_path = os.path.join(self.mask_path, '{}.png'.format(name))

        # zhj: test
        # self.mask_path = os.path.join('/mnt/nas_10/group/hongjia/datasets/Replica_generated/640x360/2D_masks', self.scene_id, self.mask_name, 'raw')
        # mask_path = os.path.join(self.mask_path, 'mask_{}.png'.format(name))
        
        mask = cv2.imread(mask_path, -1).astype(np.float32)
        return mask

    def load_data_for_seg(self, kf_inter):
        poses = []
        depths = []
        intrinsics = []
        masks = []

        for i in range(0, self.num_frames, kf_inter):
            poses.append(self.poses[i])
            depths.append(self.load_depth(i))
            intrinsics.append(self.K)
            masks.append(self.load_mask(i))
        
        poses = np.stack(poses, 0) # [N_images, 4, 4]
        depths = np.stack(depths, 0) # [N_images, H, W]
        intrinsics = np.stack(intrinsics, 0) # [N_images, 3, 3]
        masks = np.stack(masks, 0) # [N_images, H, W]

        print('poses: ', poses.shape)
        print('depths: ', depths.shape)
        print('intrinsics: ', intrinsics.shape)
        print('masks: ', masks.shape)

        return poses, depths, intrinsics, masks

    def get_frame(self, index):
        rgb = torch.from_numpy(self.load_image(index)).float()
        depth = torch.from_numpy(self.load_depth(index)).float()
        rgb = rgb
        depth = depth

        pose = self.poses[index]
        c2w = torch.from_numpy(pose).float()
        w2c = torch.from_numpy(np.linalg.inv(pose)).float()

        ret = {
            "K": self.K,    # [3, 3]
            "c2w": c2w,     # [4, 4]
            "w2c": w2c,     # [4, 4]
            "rgb": rgb,     # [H, W, 3]
            "depth": depth, # [H, W]
            "valid": True,  # bool: 
        }

        return ret

    def load_poses(self, path):
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        # twc ?
        for i in range(len(lines)):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            # pose = np.linalg.inv(pose)
            poses.append(pose)

        return poses

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        ret = self.get_frame(idx)

        return ret