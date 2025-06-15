import numpy as np
from tqdm import tqdm
import torch

def num_to_natural(group_ids, void_number=-1):
    if (void_number == -1):
        # [-1,-1,0,3,4,0,6] -> [-1,-1,0,1,2,0,3]
        if np.all(group_ids == -1):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != -1])
        mapping = np.full(np.max(unique_values) + 2, -1)
        # map ith(start from 0) group_id to i
        mapping[unique_values + 1] = np.arange(len(unique_values))
        array = mapping[array + 1]

    elif (void_number == 0):
        # [0,3,4,0,6] -> [0,1,2,0,3]
        if np.all(group_ids == 0):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != 0])
        mapping = np.full(np.max(unique_values) + 2, 0)
        mapping[unique_values] = np.arange(len(unique_values)) + 1
        array = mapping[array]
    else:
        raise Exception("void_number must be -1 or 0")

    return array

class Edge:
    def __init__(self, a, b, w, sem):
        self.a = a
        self.b = b
        self.w = w
        self.sem = sem

# Disjoint-set (union-find) class
class Universe:
    def __init__(self, num_elements, feat):
        self.parent = list(range(num_elements))
        self.rank = [0] * num_elements
        self.size = [1] * num_elements
        self.num = num_elements
        self.feat = feat

    def find(self, u):
        if u != self.parent[u]:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def get_feat_sim(self, u, v):
        feat_u = self.feat[u]
        feat_v = self.feat[v]
        sim = np.dot(feat_u, feat_v) / (np.linalg.norm(feat_u) * np.linalg.norm(feat_v))
        return sim

    def union(self, u, v):
        u_root = self.find(u)
        v_root = self.find(v)

        if u_root == v_root:
            return

        # Union by rank
        if self.rank[u_root] > self.rank[v_root]:
            self.parent[v_root] = u_root
            self.size[u_root] += self.size[v_root]

            # update node feature
            # self.feat[u_root] = (self.size[v_root] * self.feat[v_root] + self.size[u_root] * self.feat[u_root]) / (self.size[v_root] + self.size[u_root])

        else:
            self.parent[u_root] = v_root
            self.size[v_root] += self.size[u_root]

            # update node feature
            # self.feat[v_root] = (self.size[v_root] * self.feat[v_root] + self.size[u_root] * self.feat[u_root]) / (self.size[v_root] + self.size[u_root])

            if self.rank[u_root] == self.rank[v_root]:
                self.rank[v_root] += 1

        self.num -= 1

    def component_size(self, u):
        return self.size[self.find(u)]

    def num_sets(self):
        return self.num

@torch.inference_mode()
def calcu_all_jsd_similar(p_dist, q_dist):
    # p_dist: [n1, d]
    # q_dist: [n2, d]
    assert p_dist.shape[1] == q_dist.shape[1], "dimension should be same."

    def kl_divergence(p, q):
        return torch.sum(p * torch.log(p / q), dim=-1)
    
    epsilon = 1e-10
    p_dist = p_dist + epsilon
    q_dist = q_dist + epsilon
    
    # p_dist -> [n1, 1, d], q_dist -> [1, n2, d]
    p_dist_expanded = p_dist.unsqueeze(1)  # [n1, 1, d]
    q_dist_expanded = q_dist.unsqueeze(0)  # [1, n2, d]
    
    # mean dist. M = 0.5 * (P + Q)
    m_dist = 0.5 * (p_dist_expanded + q_dist_expanded)  # [n1, n2, d]
    
    # cal JSD
    kl_p_m = kl_divergence(p_dist_expanded, m_dist)  # [n1, n2]
    kl_q_m = kl_divergence(q_dist_expanded, m_dist)  # [n1, n2]
    jsd_matrix = 1 - 0.5 * (kl_p_m + kl_q_m)  # [n1, n2]

    return jsd_matrix

# @breif: cal graph edge: similarity and confidence
# @param ins_neighbors [n_ins, n_ins]: 1 if two instance are neighbors
# @param ins_label [n_points]: 3D instance id of every point
# @param ins_vis_ratio [n_ins, n_imagse]: ratio of seen part of primitives in every view
# @param points_mask_label [n_points, n_images]: labels of all points in all views
# @return similar [n_ins, n_ins]: wighted sum of how much the two primitives are similar in every view
# @return confidence [n_ins, n_ins]: sum of wight of how much we can trust the similar score in every view
@torch.inference_mode()
def get_similar_confidence_matrix(ins_neighbors, ins_label, ins_vis_ratio, points_mask_label):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    view_num = ins_vis_ratio.shape[1]
    ins_num = ins_vis_ratio.shape[0]

    gpu_ins_label = torch.tensor(ins_label, device=device, dtype=torch.int32)  # (n,)
    gpu_points_mask_label = torch.tensor(points_mask_label, device='cpu', dtype=torch.float32)  # (n,m)
    gpu_ins_neighbors = torch.tensor(ins_neighbors, device=device, dtype=torch.bool)  # (s, s)
    gpu_ins_vis_ratio = torch.tensor(ins_vis_ratio, device=device, dtype=torch.float32)  # (s, m)

    similar_sum = torch.zeros([ins_num, ins_num], device=device, dtype=torch.float32)  # (s, s)
    confidence_sum = torch.zeros([ins_num, ins_num], device=device, dtype=torch.float32)
    one_view_similar = torch.zeros([ins_num, ins_num], device=device, dtype=torch.float32)
    one_view_confidence = torch.zeros([ins_num, ins_num], device=device, dtype=torch.float32)

    for m in tqdm(range(view_num)):
        plabels = gpu_points_mask_label[:, m].to(device)  # (n_points,)
        one_view_similar.fill_(0.)
        one_view_confidence.fill_(0.)

        label_range = int(torch.max(plabels) + 1) # exist 0
        if (label_range < 2):
            continue  

        # cal 3d_id & 2d_mask_id distribution
        seglabels = torch.zeros([ins_num, label_range], device=device, dtype=torch.float32)  # (s, lr)
        # concat: 2d mask label || 3d instance label
        p_maskIds_insIds = torch.stack([plabels, gpu_ins_label], dim=1) #  [n_points, 2]
        # unique_maskIds_insIds: [n_uniques, 2], unique_counts: [n_uniques]
        unique_maskIds_insIds, unique_counts = torch.unique(p_maskIds_insIds, return_counts=True, dim=0)
        unique_maskIds_insIds = unique_maskIds_insIds.type(torch.long)
        unique_counts = unique_counts.type(torch.float32)
        seglabels[unique_maskIds_insIds[:, 1], unique_maskIds_insIds[:, 0]] = unique_counts  # (n_ins, label_range)

        # 2D mask id 0 is invalid
        nonzero_seglabels = seglabels[:, 1:]  # (s,lr-1)
        nonzero_seglabels = torch.divide(nonzero_seglabels, torch.clamp(torch.norm(nonzero_seglabels, dim=-1), 1e-8)[:, None])

        del unique_counts, unique_maskIds_insIds
        del p_maskIds_insIds, plabels

        # in every iter, we process one batch primitives and its all neighbors
        batch_size = 200
        for start_id in range(0, ins_num, batch_size):
            if (ins_neighbors[start_id:start_id+batch_size].nonzero()[0].size == 0):
                continue
            
            # [batch_size, ins_num] --> sum --> [ins_num]
            all_neighbors_mask = torch.sum(gpu_ins_neighbors[start_id:start_id+batch_size], dim=0) > 0
            neighbors_labels = nonzero_seglabels[all_neighbors_mask] # [neibors, lr-1]
            
            one_view_similar[start_id:start_id+batch_size, all_neighbors_mask] = \
                calcu_all_jsd_similar(nonzero_seglabels[start_id:start_id+batch_size], neighbors_labels)
        
        # (s,1) @ (1,s) = (s,s)
        one_view_confidence = gpu_ins_vis_ratio[:,m][:, None] @ gpu_ins_vis_ratio[:, m][None, :]
        del seglabels, nonzero_seglabels

        confidence_sum += one_view_confidence
        similar_sum += (one_view_similar*one_view_confidence)
    
    # [ins_num, ins_num], [ins_num, ins_num]
    return [similar_sum.cpu().numpy(), confidence_sum.cpu().numpy()]