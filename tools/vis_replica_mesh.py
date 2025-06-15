import open3d as o3d
import numpy as np
import os
from os.path import join
import ArgumentParser

import sys 
sys.path.append('..')

from datasets import replica_class_utils

# instance label down from: https://github.com/Pointcept/OpenIns3D/blob/main/scripts/prepare_replica.sh
# semantic label down from: https://github.com/opennerf/opennerf/tree/main/datasets/replica_gt_semantics

def export_mesh(v, f, c=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    return mesh

def vis_replica_mesh(semantic_labels, instance_labels, ply_path):
    # vis semantic mesh
    points_num = semantic_labels.shape[0]
    colors = np.zeros((points_num, 3))

    for label in np.unique(semantic_labels):
        colors[semantic_labels == label] = np.array([i for i in replica_class_utils.MATTERPORT_COLOR_MAP_21.values()][label]) / 255.0
    
    mesh = o3d.io.read_triangle_mesh(ply_path)
    v = np.array(mesh.vertices)
    f = np.array(mesh.triangles)
    c_label = colors

    semantic_mesh = export_mesh(v, f, c_label)

    # vis instance mesh
    points_num = instance_labels.shape[0]
    colors = np.ones((points_num,3)) * 0.8
    for label in np.unique(instance_labels):
        colors[instance_labels == label] = np.random.rand(3) 
    
    # set background color to 1
    stuff_labels = [-100, 0, 1, 20]
    for stuff in stuff_labels:
        # use gt semantic here? align with GT?
        colors[semantic_labels == stuff] = np.ones(3) * 0.8
    
    mesh = o3d.io.read_triangle_mesh(ply_path)
    v = np.array(mesh.vertices)
    f = np.array(mesh.triangles)
    c_label = colors

    instance_mesh = export_mesh(v, f, c_label)

    return semantic_mesh, instance_mesh

def vis_replica(args):
    scene_ids = ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']
    
    save_dir = os.path.join(args.save_root, args.seg_name) 
    os.makedirs(save_dir, exist_ok=True)

    for scene_id in scene_ids:
        mesh_path = join(args.dataset_path, f'{scene_id}_trimesh.ply')
        if seg_name == 'gt':
            sem_label = np.array(np.load(join('{}/{}/semantic_labels_mp21.npy'.format(args.pred_path, scene_id)))).astype(int)
            ins_label = np.array(np.load(join('{}/{}/instance_labels_mp21.npy'.format(args.pred_path, scene_id)))).astype(int)
        elif seg_name == 'panogs':
            sem_label = np.array(np.load(join('{}/{}/pre_semantic.npy'.format(args.pred_path, scene_id)))).astype(int)
            ins_label = np.array(np.load(join('{}/{}/segmentation/final_seg.npy'.format(args.pred_path, scene_id)))).astype(int)
        
        # vis mesh
        sem_mesh, ins_mesh = vis_replica_mesh(sem_label, ins_label, mesh_path)

        sem_save_path = join(save_dir, f'semantic_{scene_id}.ply')
        ins_save_path = join(save_dir, f'instance_{scene_id}.ply')

        o3d.io.write_triangle_mesh(sem_save_path, sem_mesh)
        o3d.io.write_triangle_mesh(ins_save_path, ins_mesh)
        

if __name__ == '__main__':
    parser = ArgumentParser(description="Vis. mesh.")
    parser.add_argument("--seg_name", type=str, default='gt')
    parser.add_argument("--pred_path", type=str, default='/mnt/nas_10/group/hongjia/PanopticGS-test/Replica')
    parser.add_argument("--save_root", type=str, default='/mnt/nas_10/group/hongjia/PanopticGS-test/vis_mesh/Replica')
    parser.add_argument("--dataset_path", type=str, default='/mnt/nas_10/group/hongjia/datasets/Replica')
    args = parser.parse_args()

    vis_replica(args)
