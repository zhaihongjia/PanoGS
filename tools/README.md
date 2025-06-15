## Tool code

### Turn Quad Mesh into Tri Mesh  
The Replica dataset provided in NICE-SLAM is a Quad-mesh. 
For visualization in some software, such as MeshLab, we need to convert it into a triangular mesh.

You can use the following code:
```shell
python generate_replica_trimesh.py --replica_dir /mnt/nas_10/group/hongjia/datasets/Replica   
```
- `replica_dir`: Path to the Replica dataset, which will be used to read the mesh


### Visualization of semantics and instance meshes
When you want to visualize the semantics of GT and the mesh of instances, as well as the results of PanoGS predictions

You can use the following code:

```shell
# replica dataset
python vis_replica_mesh.py --seg_name gt \
                           --pred_path ../datasets/replica/3d_sem_ins \
                           --save_root /mnt/nas_10/group/hongjia/PanopticGS-test/vis_mesh/Replica \
                           --dataset_path /mnt/nas_10/group/hongjia/datasets/Replica

python vis_replica_mesh.py --seg_name panogs \
                           --pred_path /mnt/nas_10/group/hongjia/PanopticGS-test/Replica \
                           --save_root /mnt/nas_10/group/hongjia/PanopticGS-test/vis_mesh/Replica \
                           --dataset_path /mnt/nas_10/group/hongjia/datasets/Replica

# scannet dataset
python vis_scannet_mesh.py --seg_name gt \
                           --pred_path ../datasets/scannet/3d_sem_ins \
                           --save_root /mnt/nas_10/group/hongjia/PanopticGS-test/vis_mesh/ScanNet \
                           --dataset_path /mnt/nas_10/group/hongjia/datasets/ScanNet

python vis_scannet_mesh.py --seg_name panogs \
                           --pred_path /mnt/nas_10/group/hongjia/PanopticGS-test/ScanNet \
                           --save_root /mnt/nas_10/group/hongjia/PanopticGS-test/vis_mesh/ScanNet \
                           --dataset_path /mnt/nas_10/group/hongjia/datasets/ScanNet
```

- `seg_name`: The name of the visualization result, currently it can be `gt` or `panogs`. You can also customize other results in the code, such as LangSplat, OpenGaussian, OpenScene, SoftGroup, ...
- `pred_path`: Path of semantic and instance labels
- `save_root`: Path to store the visualization results
- `dataset_path`: Path to the dataset, which will be used to read the mesh


### Others


```shell
# Generate class color bar.
python vis_label.py

# Visualize the feature map or feature point cloud
python vis_feature.py
```
