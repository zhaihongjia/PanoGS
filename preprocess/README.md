# Dataset download and preprocessing instructions

In this file, we provide the instructions and codes for downloading and preprocessing the dataset (ScanNetV2 and Replica).

You can also use similar scripts to process your custom datasets.

## 1. Download the original dataset
- ScanNetV2: Download ScanNetV2 data from the official ScanNet [website](https://github.com/ScanNet/ScanNet).
- Replica: You can download through the [script](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) provided by NICE-SLAM.

## 2. Generate GT for evaluation
We temporarily provide our processed GT data in `datasets/replica` or `datasets/scannet`.
Therefore, you don't need to process the GT data yourself.

## 3. Fused feature ply and 2D instance mask generation

In order to perform open vocabulary query and 3D instance segmentation, we need to generate **fused feature point cloud (Sec. 3.2)** and **2D instance mask (Sec 3.3)**

Besides, in order to avoid contaminating the original dataset, we store the generated data in an additional directory.
You can set the storage path in the config file, `config["Dataset"]["generated_floder"]`

For example, for ScanNetV2 dataset, you will get generated data like the following, and the replica dataset is similar:
```
├── scene0000_00
│   ├── feats_weights
│   │   ├── features.npy # fused feature
│   │   └── weights.npy  # weights
│   └── sam              # mask_name in the config file
│   │   ├── raw
│   │   │   ├── name.png # 2D instance mask ID. `name` should be the same as the prefix of the original RGB image.
│   │   │   ├── ....
│   │   └── vis_color
│   │       ├── name.png # visualization of 2D instance mask
│   │       ├── ....
│   └── semantic-sam     # mask_name in the config file
│       ├── raw
│       │   ├── name.png # 2D instance mask ID
│       │   ├── ....
│       └── vis_color
│           ├── name.png # visualization of 2D instance mask
│           ├── ....
├── scene0062_00
```

### 1. Generate fused feature pointclouds (Sec. 3.2) 
We currently only support using LSeg to extract image features.

#### Use LSeg for feature extraction
Follow the setting in [LSeg](https://github.com/pengsongyou/lseg_feature_extraction) to install LSeg.

According to the [Installation](https://github.com/pengsongyou/lseg_feature_extraction?tab=readme-ov-file#installation) to set some pathes:
- `txt path` in [./modules/lseg_module.py Line#99](https://github.com/pengsongyou/lseg_feature_extraction/blob/60afac405fe146c1c9df01ec3ea1baee39d5ee3b/modules/lseg_module.py#L99)
- `checkpoint_path` and `data_path` in `fuse_feature.py`:
```shell
import sys
sys.path.append('/home/hongjia/Projects/lseg_feature_extraction')

module = LSegModule.load_from_checkpoint(
    checkpoint_path='/home/hongjia/Projects/lseg_feature_extraction/checkpoints/demo_e200.ckpt',
    data_path='/home/hongjia/Projects/lseg_feature_extraction/datasets',
    ...)
```

Then, run the following codes to extract fused feature pointcloud:
```shell
# e.g.
python scripts/fuse_feature.py --config configs/scannet/scene0000_00.yaml

# e.g.
python scripts/fuse_feature.py --config ./configs/replica/room0.yaml 
```

The `feats_weights` folder will be saved in `config["Dataset"]["generated_floder"]`.

#### TODO
@NOTE: to support OpenSeg, CLIP, SigLIP


### 2. Generate 2D instance mask

We use [SAM](https://github.com/facebookresearch/segment-anything) to process images to obtain its 2D instance mask.

You can also use other 2D instance segmentation model (e.g. [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM),
[Mask2Former](https://github.com/facebookresearch/Mask2Former), 
[CropFormer](https://github.com/qqlu/Entity/blob/main/Entityv2/CropFormer/INSTALL.md)) to generate 2D instance masks.

Some code for processing masks can be found in [sam_mask_utils.py](./sam_mask_utils.py)

MobileSAM and FastSAM are much **faster** than SAM. Mask2Former can provide **semantic information**, and CropFormer can provide **high-quality** entity level instance mask.

If you want to use other segmentation methods, just pay attention to the following points:
- Placed in the same directory level as sam and semantic-sam.
- Contains two subdirectories: `raw` and `vis_color`, corresponding to the instance mask id and the visualized mask image respectively.
- The mask id should start from 1, 0 means invalid mask.
- The prefix of the mask is consistent with the original RGB image


#### (a) Use SAM to extract 2D instance mask
1. Install [SAM](https://github.com/facebookresearch/segment-anything)
```shell
pip install git+https://github.com/facebookresearch/segment-anything.git
```

2. Download model checkpoints, you can refer to [Model Checkpoints](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)

```
# we use vit_h for example.
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

3. Run the following codes to get 2D instance mask:
```shell
# ScanNetV2
python sam_auto.py --checkpoint /mnt/nas_10/group/hongjia/sam_vit_h_4b8939.pth --model-type 'vit_h' \
                           --input_dir /mnt/nas_10/group/hongjia/datasets/ScanNet/scans/scene0000_00/color \
                           --output_dir /mnt/nas_10/group/hongjia/ScanNet_generated/scene0000_00 \
                           --dataset scannet

# Replica
python sam_auto.py --checkpoint /mnt/nas_10/group/hongjia/sam_vit_h_4b8939.pth --model-type 'vit_h' \
                           --input_dir /mnt/nas_10/group/hongjia/datasets/Replica/room0/results \
                           --output_dir /mnt/nas_10/group/hongjia/Replica_generated/room0 \
                           --dataset replica
```

The `sam` folder will be saved in `config["Dataset"]["generated_floder"]`.

#### (b) Use Semantic-SAM to extract 2D instance mask
1. Install [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM)
```shell
git clone https://github.com/UX-Decoder/Semantic-SAM.git Semantic-SAM --recursive
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
cd Semantic-SAM
python -m pip install -r requirements.txt
cd semantic_sam/body/encoder/ops
sh ./make.sh
```

2. Download model checkpoints

Change [the config here](https://github.com/UX-Decoder/Semantic-SAM/blob/e3b9/configs/semantic_sam_only_sa-1b_swinL.yaml#L42) to `false`.
```shell
cd Semantic-SAM && mkdir checkpoints && cd checkpoints
wget https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth
```

3. Similar to SAM, run the following codes to get 2D instance mask:
```shell
PROCESS_DIR="/home/hongjia/Projects/PanoGS/preprocess"  
export PYTHONPATH="${PROCESS_DIR}":$PYTHONPATH

cd Semantic_SAM

# e.g. ScanNetV2
python -m semantic_sam_auto --checkpoint $PROCESS_DIR/Semantic-SAM/checkpoints/swinl_only_sam_many2many.pth \
                           --input_dir /mnt/nas_10/group/hongjia/datasets/ScanNet/scans/scene0000_00/color \
                           --output_dir /mnt/nas_10/group/hongjia/ScanNet_generated/scene0000_00 \
                           --dataset scannet

# Replica
python -m semantic_sam_auto --checkpoint $PROCESS_DIR/Semantic-SAM/checkpoints/swinl_only_sam_many2many.pth \
                           --input_dir /mnt/nas_10/group/hongjia/datasets/Replica/room0/results \
                           --output_dir /mnt/nas_10/group/hongjia/Replica_generated/room0 \
                           --dataset replica
```

The `semantic-sam` folder will be saved in `config["Dataset"]["generated_floder"]`.

#### (c) Use MobileSAM to extract 2D instance mask
1. Install [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
```shell
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

2. Download model checkpoints
Please refer to [Getting Started (MobileSAMv2)](https://github.com/ChaoningZhang/MobileSAM?tab=readme-ov-file#getting-started-mobilesamv2) to download model weights.


3. Similar to SAM, run the following codes to get 2D instance mask:
```shell
# ScanNetV2
python mobilesam_auto.py --checkpoint /mnt/nas_10/group/hongjia/weight/mobile_sam.pt --model-type 'vit_t' \
                           --input_dir /mnt/nas_10/group/hongjia/datasets/ScanNet/scans/scene0000_00/color \
                           --output_dir /mnt/nas_10/group/hongjia/ScanNet_generated/scene0000_00 \
                           --dataset scannet

# Replica
python mobilesam_auto.py --checkpoint /mnt/nas_10/group/hongjia/weight/mobile_sam.pt --model-type 'vit_t' \
                           --input_dir /mnt/nas_10/group/hongjia/datasets/Replica/room0/results \
                           --output_dir /mnt/nas_10/group/hongjia/Replica_generated/room0 \
                           --dataset replica
```

The `mobilesam` folder will be saved in `config["Dataset"]["generated_floder"]`.

#### TODO
@NOTE: to support FastSAM, CropFormer, Mask2Former, ....

