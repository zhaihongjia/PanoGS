# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
from torch.serialization import save  # type: ignore
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import argparse
import json
import os
from typing import Any, Dict, List

import torch
import numpy as np
import glob
import tqdm
from sam_mask_utils import get_single_mask, viz_mask

parser = argparse.ArgumentParser(description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)
parser.add_argument("--input_dir", type=str, required=True, help="Path to either a single input image or folder of images.")
parser.add_argument("--output_dir", type=str, required=True, help=("Path to the directory where masks will be output. Output will be either a folder of PNGs per image or a single json with COCO-style masks."),)
parser.add_argument("--model-type", type=str, required=True, help="The type of model to load",)
parser.add_argument("--checkpoint", type=str, required=True, help="The path to the SAM checkpoint to use for mask generation.",)
parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
parser.add_argument("--convert-to-rle", action="store_true", help=("Save masks as COCO RLEs in a single json instead of as a folder of PNGs. Requires pycocotools."),)

amg_settings = parser.add_argument_group("AMG Settings")
amg_settings.add_argument("--points-per-side", type=int, default=None, help="Generate masks by sampling a grid over the image with this many points to a side.",)
amg_settings.add_argument("--points-per-batch", type=int, default=None, help="How many input points to process simultaneously in one batch.",)
amg_settings.add_argument("--pred-iou-thresh", type=float, default=None, help="Exclude masks with a predicted score from the model that is lower than this threshold.",)
amg_settings.add_argument("--stability-score-thresh", type=float, default=None, help="Exclude masks with a stability score lower than this threshold.",)
amg_settings.add_argument("--stability-score-offset", type=float, default=None, help="Larger values perturb the mask more when measuring stability score.",)
amg_settings.add_argument("--box-nms-thresh", type=float, default=None, help="The overlap threshold for excluding a duplicate mask.",)
amg_settings.add_argument("--crop-n-layers", type=int, default=None, help=("If >0, mask generation is run on smaller crops of the image to generate more masks. The value sets how many different scales to crop at."),)
amg_settings.add_argument("--crop-nms-thresh", type=float, default=None, help="The overlap threshold for excluding duplicate masks across different crops.",)
amg_settings.add_argument("--crop-overlap-ratio", type=int, default=None, help="Larger numbers mean image crops will overlap more.",)
amg_settings.add_argument("--crop-n-points-downscale-factor", type=int, default=None, help="The number of points-per-side in each layer of crop is reduced by this factor.")
amg_settings.add_argument("--min-mask-region-area", type=int, default=None, help=("Disconnected mask regions or holes with area smaller than this value in pixels are removed by postprocessing."),)

def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def main(args: argparse.Namespace):
    print("Loading model...")
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam_model.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam_model, **amg_kwargs)

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
    vis_dir = os.path.join(args.output_dir, 'mobilesam', 'vis_color')
    raw_dir = os.path.join(args.output_dir, 'mobilesam', 'raw')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    for img_i in tqdm.tqdm(imgs):
        image = cv2.imread(img_i)
        if image is None:
            print(f"Could not load '{img_i}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_wh, interpolation=cv2.INTER_LINEAR)
        name = img_i.split('/')[-1].split('.')[0]
        generator.predictor.set_image(image)

        anns = generator.generate(image)
        mask = get_single_mask(anns, metric='predicted_iou', descending=False)
        color_mask = viz_mask(mask)

        # save mask and color_mask
        cv2.imwrite(os.path.join(vis_dir, f'{name}.png'), color_mask)
        cv2.imwrite(os.path.join(raw_dir, f'{name}.png'), mask)

if __name__ == "__main__":
    parser.add_argument('--dataset', type=str, required=True, help='dataset type')
    args = parser.parse_args()
    main(args)
