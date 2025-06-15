import torch
from torch import nn
from gaussian_splatting.utils.graphics_utils import getWorld2View2

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        encode_feat=None,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        self.W2C = gt_T
        self.R = gt_T[:3, :3]
        self.T = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.encode_feat = encode_feat
        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        ret = dataset[idx]

        gt_color = ret['rgb'].permute(2, 0, 1).to(device=dataset.device, dtype=dataset.dtype)
        gt_depth = ret['depth'].numpy()
        gt_pose  = ret['w2c'].to(device=dataset.device)
        print('idx: ', idx, 'valid: ', ret['valid'])
        
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )
        
    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def clean(self):
        self.original_image = None
        self.depth = None
