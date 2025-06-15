import open3d as o3d
import numpy as np
import os
from os.path import join
import ArgumentParser

import sys 
sys.path.append('..')

from datasets import scannet_class_utils

def export_mesh(v, f, c=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    return mesh

def vis_scannet_mesh(semantic_labels, instance_labels, ply_path):
    # vis semantic mesh
    points_num = semantic_labels.shape[0]
    colors = np.zeros((points_num, 3))

    for label in np.unique(semantic_labels):
        if(label == 0):continue
        colors[semantic_labels == label] = np.array([i for i in scannet_class_utils.SCANNET_COLOR_MAP_20.values()][label]) / 255.0
    
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
    stuff_labels = [-100, 0, 1, 2]
    for stuff in stuff_labels:
        # use gt semantic here? align with GT?
        colors[semantic_labels == stuff] = np.ones(3) * 0.8
    
    mesh = o3d.io.read_triangle_mesh(ply_path)
    v = np.array(mesh.vertices)
    f = np.array(mesh.triangles)
    c_label = colors
    instance_mesh = export_mesh(v, f, c_label)

    return semantic_mesh, instance_mesh

def vis_scannet(seg_name):
    save_dir = join(args.save_root, args.seg_name) 
    os.makedirs(save_dir, exist_ok=True)

    scene_ids = ['scene0000_00', 'scene0062_00', 'scene0070_00', 'scene0097_00', 'scene0140_00', \
                 'scene0200_00', 'scene0347_00', 'scene0400_00', 'scene0590_00', 'scene0645_00']

    for scene_id in scene_ids:
        mesh_path = join(args.dataset_path, 'scans', scene_id, f'{scene_id}_vh_clean_2.ply')

        if args.seg_name == 'gt':
            semantic_labels = np.array(np.load(join(args.pred_path, scene_id, 'semantic_labels.npy'))).astype(int)
            instance_labels = np.array(np.load(join(args.pred_path, scene_id, 'instance_labels.npy'))).astype(int)
        elif args.seg_name == 'panogs':
            semantic_labels = np.array(np.load(join(args.pred_path, scene_id, 'pre_semantic.npy'))).astype(int)
            instance_labels = np.array(np.load(join(args.pred_path, scene_id, 'segmentation/final_seg.npy'))).astype(int)
        
        # vis mesh
        sem_mesh, ins_mesh = vis_scannet_mesh(semantic_labels, instance_labels, mesh_path)

        sem_save_path = join(save_dir, f'semantic_{scene_id}.ply')
        ins_save_path = join(save_dir, f'instance_{scene_id}.ply')

        o3d.io.write_triangle_mesh(sem_save_path, sem_mesh)
        o3d.io.write_triangle_mesh(ins_save_path, ins_mesh)

if __name__ == '__main__':
    parser = ArgumentParser(description="Vis. mesh.")
    parser.add_argument("--seg_name", type=str, default='gt')
    parser.add_argument("--pred_path", type=str, default='/mnt/nas_10/group/hongjia/PanopticGS-test/ScanNet')
    parser.add_argument("--save_root", type=str, default='/mnt/nas_10/group/hongjia/PanopticGS-test/vis_mesh/ScanNet')
    parser.add_argument("--dataset_path", type=str, default='/mnt/nas_10/group/hongjia/datasets/ScanNet')
    args = parser.parse_args()

    vis_scannet(args)
