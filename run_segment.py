import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
from argparse import ArgumentParser
import random
import yaml
from munch import munchify
import cv2
import open3d as o3d
import imgviz
import numpy as np
from tqdm import tqdm
import scipy
from plyfile import PlyData, PlyElement

import torch
import matplotlib.pyplot as plt
from collections import deque

from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.config_utils import load_config
from utils.seg_utils import num_to_natural, Universe, Edge, get_similar_confidence_matrix
from utils.eval_utils import calculate_iou_3d
from gaussian_splatting.utils.system_utils import mkdir_p
from datasets.load_func import load_dataset
from decoders.decoder import FeatureDecoder

class GSegmentation:
    def __init__(self, config):
        self.config = config
        self.save_dir = config["Results"]["save_dir"]

        self.seg_min_verts = config['segmentation']['seg_min_verts']
        self.k_thr = config['segmentation']['k_thresh']
        self.k_neigbor = config['segmentation']['k_neigbor']
        self.clustering_thres = config['segmentation']['thres_connect']
        self.thres_merge = config['segmentation']['thres_merge']

        self.discard_unseen = config["Training"]["discard_unseen"]
        self.thres_vis_dis = config['Training']['thres_vis_dis']
        self.kf_inter = self.config["Training"]["kf_inter"]

        self.n_workers = 20
        self.feat_decoder = None

    def load_decoder(self, ):
        ckpt_path = os.path.join(self.save_dir, 'decoder/ckpt.pth')
        print('Load feature decoder from: ', ckpt_path)
        self.feat_decoder = FeatureDecoder(self.config).cuda()
        self.feat_decoder.load_state_dict(torch.load(ckpt_path))
    
    # @breif: graph cuts with geo. and lang.
    def segment_graph(self, num_vertices, edges, threshold, feat):
        edges = sorted(edges, key=lambda e: e.w)
        u = Universe(num_vertices, feat) 
        normal_t = [threshold] * num_vertices  
        feat_thre = 0.99
        feat_t = [feat_thre] * num_vertices 

        for edge in edges:
            a = u.find(edge.a)
            b = u.find(edge.b)
            if a != b:  
                normal = edge.w <= normal_t[a] and edge.w <= normal_t[b]
                sim = u.get_feat_sim(a,b)
                feat = sim > 0.9
                # feat = ((sim >= feat_t[a]) and (sim >= feat_t[b]))
                
                if normal and feat:
                    u.union(a, b) 
                    new_root = u.find(a)
                    normal_t[new_root] = edge.w + threshold / u.component_size(new_root)
                    feat_t[new_root] = sim + feat_thre / u.component_size(new_root)

        return u

    # @breif: build super-gaussian or super-primitives
    def build_super_gaussians(self):
        # points normals neighbors
        ply = o3d.geometry.PointCloud()
        ply.points = o3d.utility.Vector3dVector(self.points_w)
        ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.array(ply.normals)

        points_kdtree = scipy.spatial.KDTree(self.points_w)
        # (n_points, k_neighbors)  the first one is itself
        self.points_neighbors = points_kdtree.query(self.points_w, self.k_neigbor, workers=self.n_workers)[1]
        # distances, indices = points_kdtree.query(self.points_w, k=k)

        num_points = self.points_w.shape[0] 
        num_neighbors = self.points_neighbors.shape[1]
        edges = []

        if self.feat_decoder is None:
            self.load_decoder()

        pc_feat = self.feat_decoder(torch.from_numpy(self.points_w)).cpu().detach() # [N, n_dim]
        pc_feat = torch.nn.functional.normalize(pc_feat, p=2, dim=1).numpy()

        # construct edges
        for i in range(num_points):
            for j in range(1, num_neighbors): # ignore itself
                a = i
                b = self.points_neighbors[i, j]
                norm_dist = 1.0 - np.dot(normals[a], normals[b])

                # for convex surface
                d_xyz = self.points_w[b] - self.points_w[a]
                d_xyz /= np.linalg.norm(d_xyz)
                dot2 = np.dot(normals[b], d_xyz)
                
                feat_dist = 1 - np.dot(pc_feat[a], pc_feat[b])

                if dot2 > 0:
                    norm_dist = norm_dist * norm_dist 
                    
                # W[i, j] = np.exp(-color_dist**2 / (2 * sigma_c**2)) * np.exp(-normal_dist**2 / (2 * sigma_n**2))
                if feat_dist > 0.1:
                    norm_dist = norm_dist * 20

                edge_w = norm_dist
                edges.append(Edge(a, b, edge_w, feat_dist))  

        u = self.segment_graph(num_points, edges, self.k_thr, pc_feat)

        # merge small segments
        for edge in edges:
            a = u.find(edge.a)
            b = u.find(edge.b)
            if (a != b) and (u.component_size(a) < self.seg_min_verts or u.component_size(b) < self.seg_min_verts):
                u.union(a, b)

        segments = [u.find(i) for i in range(num_points)]
        print('Graph Cuts.: ', len(np.unique(segments, return_counts=True)[1]), ' Super-Primitives.')
        return segments

    # @breif: update graph data before each iteration
    def build_graph(self, ins_label):
        ins_label = num_to_natural(ins_label)
        unique_gs_ids, ins_member_count = np.unique(ins_label, return_counts=True)  # from 0 to instance_num - 1
        ins_num = len(unique_gs_ids)

        # build dict: super-Gaussian ID -> 3D Gaussians index
        ins_members = {ins_id: np.where(ins_label == ins_id)[0] for ins_id in unique_gs_ids}

        ins_neighbors = np.zeros((ins_num, ins_num), dtype=bool)
        for ins_id, members in ins_members.items():
            # Gaussians with current 3d ins_label -> its neighbors -> its neighbors' 3d ins_label
            neighbor_ins_ids = ins_label[self.points_neighbors[members]].flatten()
            ins_neighbors[ins_id, neighbor_ins_ids] = True
        
        # neighboring matrix symmetric and exclude self
        ins_neighbors = np.maximum(ins_neighbors, ins_neighbors.T)
        np.fill_diagonal(ins_neighbors, 0)

        # indirect neighbor pool
        ins_neighbor_pool = np.zeros((ins_num, ins_num), dtype=bool)
        for ins_id in range(ins_num):
            neigbs = ins_neighbors[ins_id]
            neighbors_pool = ins_neighbors[neigbs].sum(0) > 0
            ins_neighbor_pool[ins_id] = neighbors_pool
        
        # exclude self
        np.fill_diagonal(ins_neighbor_pool, 0)
        ins_neighbor_pool[ins_neighbors] = 1
        ins_neighbor_pool[ins_neighbor_pool.T] = 1

        # update graph data
        # list of int
        self.ins_member_count = ins_member_count
        # binary matrix, (ins_num, ins_num)
        self.ins_neighbors = ins_neighbors
        # binary matrix, (ins_num, ins_num)
        self.ins_neighbor_pool = ins_neighbor_pool
        # (n_points)
        self.ins_label = ins_label
        # int
        self.ins_num = ins_num
        # dict ins_id : list of points id
        self.ins_members = ins_members

    # @breif: compute affinity between diff instance
    def compute_edge_affinity(self, points_mask_label, points_seen):
        # cal visible ratio of each sup-primitive/vertex in each view
        ins_vis_ratio = np.zeros([self.ins_num, self.n_images], dtype=np.float32)  # (ins_num, n_images)
        for ins_id, members in self.ins_members.items(): # dict {ins_id: point_array}
            ins_vis_ratio[ins_id] = ((points_mask_label[members] > 0).sum(axis=0)) / members.shape[0]  # (n_members, n_images) sum -> (n_images)

        # similar_mat [ins_num, ins_num]: weight sum of similar score in every view 
        # confidence_mat [ins_num, ins_num]: sum of confidence of how much we can trust the similar score in every view
        similar_mat, confidence_mat = get_similar_confidence_matrix(self.ins_neighbor_pool, self.ins_label, ins_vis_ratio, points_mask_label)

        # cal. adjacency_mat
        assert similar_mat.nonzero()[0].size > 0
        adjacency_mat = np.zeros([self.ins_num, self.ins_num])

        # https://blog.csdn.net/monchin/article/details/79750216 
        adjacency_mat[confidence_mat.nonzero()] = similar_mat[confidence_mat.nonzero()] / confidence_mat[confidence_mat.nonzero()]
        r, c = adjacency_mat.nonzero()
        adjacency_mat[r, c] = adjacency_mat[c, r] = np.maximum(adjacency_mat[r, c], adjacency_mat[c, r])

        return adjacency_mat

    # @breif: postprocess segmentation results by merging small regions into neighbor regions with high affinity 
    def merge_small_segs(self, ins_labels, merge_thres, adj):
        ins_member_count = self.ins_member_count
        unique_labels, ins_count = np.unique(ins_labels, return_counts=True)
        region_num = unique_labels.shape[0]

        merged_labels = ins_labels.copy()
        merge_count = 0
        # 0 means the superpoint is remain to merge
        merged_mask = np.ones_like(ins_labels)
        for i in range(region_num):
            if ins_count[i] > 2:
                continue
            label = unique_labels[i]
            seg_ids = (ins_labels == label).nonzero()[0]
            if ins_member_count[seg_ids].sum() < merge_thres:
                merged_mask[seg_ids] = 0

        finished = False
        while not finished:
            flag = False  # mark whether merging happened in this iteration
            for i in range(region_num):
                label = unique_labels[i]
                seg_ids = (ins_labels == label).nonzero()[0]
                if merged_mask[seg_ids[0]] > 0:
                    continue
                seg_sims = adj[seg_ids].sum(0)
                adj_sort = np.argsort(seg_sims)[::-1]

                for i in range(adj_sort.shape[0]):
                    target_seg_id = adj_sort[i]
                    if merged_mask[target_seg_id] == 0:
                        continue  # if the target region is also too samll and has not been merged, find next target
                    if seg_sims[target_seg_id] == 0:
                        break  # no more target region can be found
                    target_label = merged_labels[target_seg_id]
                    merged_labels[seg_ids] = target_label
                    merge_count += 1
                    merged_mask[seg_ids] = 1
                    flag = True
                    break
            if not flag:
                finished = True

        # for small regions that cannot be merged, set their labels to 0
        merged_labels[merged_mask == 0] = 0
        print('original region number:', ins_count.shape[0])
        print('mreging count:', merge_count)
        print("remove count:", (merged_mask == 0).sum())
        return merged_labels

    # @breif: graph clustering
    # @return affinity [n_ins, n_ins] :
    # @return ins_labels [n_ins] : 3D instance label of points
    def clustering(self, affinity, thres_connect):
        current_label = 1
        ins_labels = np.zeros(self.ins_num, dtype=np.float32)
        visited = np.zeros(self.ins_num, dtype=bool)

        for i in range(self.ins_num):
            if not visited[i]:
                queue = deque()
                queue.append(i)

                visited[i] = True
                ins_labels[i] = current_label

                while queue:
                    v = queue.popleft()
                    js = np.where(self.ins_neighbors[v])[0]

                    for j in js:
                        if visited[j]:
                            continue

                        # 
                        # direct neighbor
                        neighbor_ids = self.ins_neighbors[j]  # (n_ins.)
                        neighbor_ids = np.logical_and(neighbor_ids, ins_labels == current_label)  # (n_ins.)
                        neighbor_ids = neighbor_ids.nonzero()[0]  # (nei,)

                        # # (n_ins.)*(n_ins.) -> (n_ins.) -> (1.)
                        affinity_sum = (affinity[j, neighbor_ids] * self.ins_member_count[neighbor_ids]).sum(0)
                        weight_sum = (self.ins_member_count[neighbor_ids]).sum(0)  # (s.) -> (1.)

                        # indirect neighbor
                        neighbor_ids = self.ins_neighbor_pool[j]  # (n_ins.)
                        neighbor_ids = np.logical_and(neighbor_ids, np.logical_not(self.ins_neighbors[j]))
                        neighbor_ids = np.logical_and(neighbor_ids, ins_labels == current_label)  # (n_ins.)
                        neighbor_ids = neighbor_ids.nonzero()[0]  # (nei,)

                        # # (n_ins.)*(n_ins.) -> (n_ins.) -> (1.)
                        affinity_sum += (0.5 * affinity[j, neighbor_ids] * self.ins_member_count[neighbor_ids]).sum(0)
                        weight_sum += 0.5 * (self.ins_member_count[neighbor_ids]).sum(0)  # (s.) -> (1.)
                        
                        score = affinity_sum / weight_sum
                        connect = (score >= thres_connect)
                        # 
                        
                        if not connect:
                            continue

                        visited[j] = True
                        ins_labels[j] = current_label
                        queue.append(j)

                current_label += 1

        return ins_labels  # (n_ins, )

    # @breif: project N (1e6) points to M (1e3) images
    # @param: points_w: [n_points, 3], intrinsic: [n_images, 3, 3], poses: [n_images, 4, 4]
    # @return points_c: [n_points, n_images, 3], 3D coordinates of spatial points in different image-views (in camera coordinate sys.)
    # @return uv_pixels: [n_points, n_images, 2], pixel coordinates of spatial points in different image-views
    @torch.inference_mode()
    def get_camp_pixel(self):
        batch_size = 10000
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        intrinsics = torch.tensor(self.intrinsics, device=device, dtype=torch.float32)
        poses = torch.tensor(self.poses, device=device, dtype=torch.float32)
        poses_inv = torch.linalg.inv(poses)   # (n_images, 4, 4)
        del poses

        N = self.points_w.shape[0]
        M = poses_inv.shape[0]
        points_c = np.zeros((N, M, 3), dtype=np.float32)
        uv_pixels = np.zeros((N, M, 2), dtype=int)

        for batch_start in tqdm(range(0, N, batch_size)):
            points_world = torch.tensor(self.points_w[batch_start: batch_start+batch_size], device=device, dtype=torch.float32)
            points_world_homo = torch.cat((points_world, torch.ones((points_world.shape[0], 1), dtype=torch.float32, device=device)), 1)

            points_cam_homo = torch.matmul(poses_inv[None], points_world_homo[:, None, :, None])
            points_cam_homo = points_cam_homo[..., 0]        # (N, M, 4)
            points_cam = torch.div(points_cam_homo[..., :-1], points_cam_homo[..., [-1]])  # (N, M, 3)

            # (M, 3, 3) @ (N, M, 3, 1) = (N, M, 3, 1)
            points_pixel_homo = torch.matmul(intrinsics, points_cam[..., None])
            # (N, M, 3)
            points_pixel_homo = points_pixel_homo[..., 0]
            # (u, v) coordinate, (N, M, 2)
            points_pixel = torch.div(points_pixel_homo[..., :-1], torch.clip(points_pixel_homo[..., [-1]], min=1e-8)).round().to(torch.int32)
            points_c[batch_start: batch_start + batch_size] = points_cam.cpu().numpy()
            uv_pixels[batch_start: batch_start + batch_size] = points_pixel.cpu().numpy()

        torch.cuda.empty_cache()
        return points_c, uv_pixels

    # @breif: get 2D mask labels and vis flag of all spatial points in all views
    # @param points_c: [n_points, n_images, 3], 3D coordinates of spatial points in different image-views (in camera coordinate sys.)
    # @param uv_pixels: [n_points, n_images, 2], pixel coordinates of spatial points in different image-views
    # @return all_label: [n_points, n_images], 2D mask labels of all points in all views
    # @return all_seen_flag: [n_points, n_images], seen flag of all points in all views
    # @NOTE: 0 in all_label is invalid
    def get_points_label_seen(self, points_c, uv_pixels, discard_unseen, thres_vis_dis=0.15):
        # @TODO: hardcode
        batch_size = 50000
        all_label = np.zeros([self.n_points, self.n_images], dtype=np.float32)
        all_seen_flag = np.zeros([self.n_points, self.n_images], dtype=bool)

        # projected by batch
        for start_id in tqdm(range(0, self.n_points, batch_size)):
            p_cam0 = points_c[start_id: start_id + batch_size]
            pix0 = uv_pixels[start_id: start_id + batch_size]
            w0, h0 = np.split(pix0, 2, axis=-1)
            w0, h0 = w0[..., 0], h0[..., 0]  # (n_points_sub, n_images)
            bounded_flag = (0 <= w0)*(w0 <= self.width - 1)*(0 <= h0)*(h0 <= self.height - 1)  # (n_points_sub, n_images)

            # (n_points_sub, n_images), querying labels from masks (n_images, H, W) by h (n_points_sub, n_images) and w (n_points_sub, n_images)
            label_iter = self.masks[np.arange(self.n_images), h0.clip(0, self.height - 1), w0.clip(0, self.width - 1)]

            # visible check
            real_depth = p_cam0[..., -1]  # (n_points_sub, n_images)
            # (n_points_sub, n_images), querying depths
            capture_depth = self.depths[np.arange(self.n_images), h0.clip(0, self.height - 1), w0.clip(0, self.width - 1)]  
            visible_flag = np.isclose(real_depth, capture_depth, rtol=thres_vis_dis)
            seen_flag = bounded_flag * visible_flag

            if discard_unseen:
                label_iter = label_iter * seen_flag  # set label of invalid point to 0

            all_seen_flag[start_id: start_id + batch_size] = seen_flag
            all_label[start_id: start_id + batch_size] = label_iter

        return all_label, all_seen_flag

    # @breif: segmentation
    def do_segmentation(self):
        seg_save_path = os.path.join(self.save_dir, 'segmentation')
        os.makedirs(seg_save_path, exist_ok=True)
        
        # Graph Vertex Construction 
        print("====> Construct Vertex.")
        points_ins_label = self.build_super_gaussians()
        points_ins_label = np.array(points_ins_label)

        init_seg_path = os.path.join(seg_save_path, 'init_seg.npy')
        np.save(init_seg_path, points_ins_label)

        # Graph Clustering based Segmentation
        # return: [n_points, n_images, 3] [n_points, n_images, 2]
        points_c, uv_pixels = self.get_camp_pixel()
        # return: [n_points, n_images], [n_points, n_images]
        points_mask_label, points_seen = self.get_points_label_seen(points_c, uv_pixels, self.discard_unseen, self.thres_vis_dis)

        print("====> Perform region clustering.")
        steps = len(self.clustering_thres)
        for i in range(steps):
            # (a) Build Graph Edges
            self.build_graph(points_ins_label)
            
            # (b) Edge Affinity Computation
            edge_affinity = self.compute_edge_affinity(points_mask_label, points_seen)

            # (c) Clustering on instance-level
            ins_labels = self.clustering(edge_affinity, self.clustering_thres[i])

            # (last iter) Filter noise
            if i == (steps - 1) and self.thres_merge > 0:
                ins_labels = self.merge_small_segs(ins_labels, self.thres_merge, edge_affinity)

            # (d) Assign primitive labels to member points
            points_ins_label = np.zeros(self.n_points, dtype=int)
            for j in range(self.ins_num):
                label = ins_labels[j]
                points_ins_label[self.ins_members[j]] = label

        # save results
        final_seg_path = os.path.join(seg_save_path, 'final_seg.npy')
        np.save(final_seg_path, points_ins_label)

        # export results to .txt
        self.export_point_wise_segmentation(True)

    # @breif: load trained 3DGS model, xyz and feature decoder
    def load_data(self):
        self.dataset = load_dataset(config=config)

        # ply_path = os.path.join(self.save_dir, 'point_cloud/final/point_cloud.ply')
        ply_path = os.path.join(self.save_dir, 'point_cloud/point_cloud.ply')

        self.gaussians = GaussianModel(self.config["model_params"]['sh_degree'], config=self.config)
        self.gaussians.load_ply(ply_path)
        print('Load 3DGS model from: ', ply_path)

        self.points_w = (self.gaussians.get_xyz).cpu().detach().numpy()        
        self.n_points = self.points_w.shape[0]
        self.width = self.dataset.width
        self.height = self.dataset.height

        # [N_images, 4, 4] # [N_images, H, W] # [N_images, 3, 3] # [N_images, H, W]
        self.poses, self.depths, self.intrinsics, self.masks = self.dataset.load_data_for_seg(self.kf_inter)
        self.n_images = self.poses.shape[0]

    def export_point_wise_segmentation(self, smooth):
        if self.feat_decoder is None:
            self.load_decoder()
        pc_feat = self.feat_decoder(torch.from_numpy(self.points_w)).cpu().detach() # [N, n_dim]
        class_feat = torch.from_numpy(self.dataset.class_text_feat).float()

        pc_feat = torch.nn.functional.normalize(pc_feat, p=2, dim=1)
        class_feat = torch.nn.functional.normalize(class_feat, p=2, dim=1)
        similarity = torch.matmul(pc_feat, class_feat.t())     # [N1, N2]
        _, category = similarity.max(dim=1)

        # prediction vote
        if smooth:
            ind_mask_path = os.path.join(self.save_dir, 'segmentation', 'init_seg.npy')
            ind_mask = np.load(ind_mask_path)

            uni_ids = np.unique(ind_mask)
            for ind in uni_ids:
                mode_result = scipy.stats.mode(category[ind_mask==ind])
                mode_value = mode_result.mode[0]
                category[ind_mask==ind] = mode_value

        if self.config['Dataset']['type'] == 'scannet':
            seg_results = category.numpy() + 1 # 0 is invaliad
        else:
            seg_results = category.numpy()
        
        gt_semantics = self.dataset.gt_semantics
        n_classes = len(self.dataset.class_names)

        ious, accs, masks = calculate_iou_3d(seg_results, gt_semantics, n_classes)
        np.save(os.path.join(self.save_dir, 'pc_feat.npy'), pc_feat.numpy())
        np.save(os.path.join(self.save_dir, 'pre_semantic.npy'), seg_results)

        iou_file_path = os.path.join(self.save_dir, 'eval_pointwise_semantic.txt')
        self.record_results(ious, accs, iou_file_path, masks)

    def record_results(self, ious, accs, file_path, masks):
        # iou of each class
        text = f"{'Class':<20} | {'IoU':<6} | {'Acc':<6}\n" + "-" * 40 + '\n'
        for class_name, iou, acc in zip(self.dataset.class_names, ious, accs):
            text += f"{class_name:<20} | {iou:.4f} | {acc:.4f}\n" 
        text += "-" * 40 + '\n'

        # miou
        if self.config['Dataset']['type'] == 'scannet':
            mask19 = [False] + [True] * 19 + [False]
            # remove: picture, refrigerator, showercurtain, bathtub
            mask15 = [False] + [True] * 19 + [False]
            # remove: cabinet, counter, desk, curtain, sink
            mask10 = [False] + [True] * 19 + [False]
            
            miou19 = ious[mask19 & masks].mean()
            # miou19 = ious[mask19].mean()
            macc19 = accs[mask19 & masks].mean()

            text += f"{'ScanNet class: 19':<20} | {miou19:.4f} | {macc19:.4f}\n"
        elif self.config['Dataset']['type'] == 'replica':
            masks[19] = False 
            miou21 = ious[masks].mean()
            macc21 = accs[masks].mean()
            text += f"{'Replica class: 21':<20} | {miou21:.4f} | {macc21:.4f}\n"
        else:
            raise NotImplementedError
        
        with open(file_path, 'w') as file:
            file.write(text)

        print(text)


if __name__ == "__main__":
    parser = ArgumentParser(description="3D Gaussian Panoptic Segmentation.")
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

        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)

        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)

    seg = GSegmentation(config)

    seg.load_data()
    seg.do_segmentation()
