import sys
sys.path.append('Semantic_SAM')

import os
import tqdm
import cv2
import glob
import argparse
from PIL import Image
import torch
import numpy as np

from sam_mask_utils import get_single_mask, viz_mask
from semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator

def segment_images(args):
    level = [3,]  # instance level
    sam_model = build_semantic_sam(model_type='L', ckpt=args.checkpoint)
    mask_generator = SemanticSamAutomaticMaskGenerator(sam_model, level=level)  # model_type: 'L' / 'T', depends on your checkpoint

    # search images
    if args.dataset == 'scannet':
        imgs = glob.glob(os.path.join(args.input_dir, '*.jpg'))
        img_wh = (640, 480)
    elif args.dataset == 'replica':
        imgs = glob.glob(os.path.join(args.input_dir, 'frame*.jpg'))
        img_wh = (640, 360) #(1200, 680)
    else:
        raise NotImplementedError

    print('Segment: ', len(imgs), ' images.')

    # save dir
    vis_dir = os.path.join(args.output_dir, 'semantic-sam', 'vis_color')
    raw_dir = os.path.join(args.output_dir, 'semantic-sam', 'raw')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    for img_i in tqdm.tqdm(imgs):
        # print('img_i: ', img_i)
        image = Image.open(img_i).convert('RGB')
        image = image.resize(img_wh) # [w, h]
        img = np.asarray(image)
        image_torch = torch.from_numpy(img.copy()).permute(2, 0, 1).cuda()
        # print('image_torch: ', image_torch.shape)

        anns = mask_generator.generate(image_torch)
        mask = get_single_mask(anns, metric='predicted_iou', descending=False)
        color_mask = viz_mask(mask)

        # save mask and color_mask
        name = img_i.split('/')[-1].split('.')[0]
        cv2.imwrite(os.path.join(vis_dir, f'{name}.png'), color_mask)
        cv2.imwrite(os.path.join(raw_dir, f'{name}.png'), mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the generated masks')
    parser.add_argument('--dataset', type=str, required=True, help='dataset type')
    parser.add_argument("--checkpoint", type=str, required=True, help="The path to the SAM checkpoint to use for mask generation.",)
    args = parser.parse_args()

    segment_images(args)
