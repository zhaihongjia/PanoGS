import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d
import matplotlib.pyplot as plt

class Decoder_dataset(Dataset):
    def __init__(self, config, ply_dir, feat_dir, weight_dir):
        self.config = config
        print('Load PLY from: ', ply_dir)
        print('Load Feature from: ', feat_dir)
        self.pcds = torch.from_numpy(np.asarray(o3d.io.read_point_cloud(ply_dir).points)) 
        self.features = torch.from_numpy(np.load(feat_dir)) # [N, 256]
        self.weights = torch.from_numpy(np.load(weight_dir)) # [N]

        # random sample
        # mask = np.random.rand(self.pcds.shape[0]) < 0.8
        # self.pcds = self.pcds[mask]
        # self.weights = self.weights[mask]
        # self.features = self.features[mask]

    def __getitem__(self, index):
        xyz = self.pcds[index]
        feat = self.features[index]
        weight = self.weights[index]
        return xyz, feat, weight

    def __len__(self):
        return self.features.shape[0] 
