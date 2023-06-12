from base_mask_generator import BaseMaskGenerator
from segment_anything import sam_model_registry
from utils import iou_single, iou_batch
import cv2
import os
import numpy as np
import torch
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Sam Robustness Evaluation on Commom Corruption")
parser.add_argument("--exp-name", type=str, help="Name of experiments. This will be also result file name.")
parser.add_argument("--save-path", type=str, help="Save path where the evaluation result is saved.")
parser.add_argument("--model-type", type=str, choices=['vit_h', 'vit_l', 'vit_b'])
parser.add_argument("--sam-checkpoint", type=str)
parser.add_argument("--path-img-list", type=str)
parser.add_argument("--path-clean-img-dir", type=str)
parser.add_argument("--path-perturbed-img-dir", type=str)
parser.add_argument("--points-per-side", type=int, default=32)
parser.add_argument("--multi-mask", type=bool, default=False)
parser.add_argument("--get-highest", type=bool, default=True)
args = parser.parse_args()


def main(args):

    # register SAM
    model_type = args.model_type
    checkpoint = args.sam_checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Register SAM-{model_type}..")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    mask_generator = BaseMaskGenerator(sam)
    
    # image list
    img_list = np.loadtxt(args.path_img_list, dtype=str)
    clean_img_dir = args.path_clean_img_dir
    perturbed_img_dir = args.path_perturbed_img_dir

    with torch.no_grad():
        score = []
        for img in tqdm(img_list):
            clean_img = cv2.imread(os.path.join(clean_img_dir, img))
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            clean_masks = mask_generator.get_masks(clean_img, args.points_per_side, args.multi_mask, args.get_highest)

            perturbed_img = cv2.imread(os.path.join(perturbed_img_dir, img))
            perturbed_img = cv2.cvtColor(perturbed_img, cv2.COLOR_BGR2RGB)
            perturbed_masks = mask_generator.get_masks(perturbed_img, args.points_per_side, args.multi_mask, args.get_highest)
            
            if len(clean_masks.shape) == 2:
                s = iou_single(clean_masks, perturbed_masks)
                score.append(s)
            else:
                s = iou_batch(clean_masks, perturbed_masks)
                score.append(s)
            
        print(np.mean(score))
        np.savetxt(os.path.join(args.save_path, args.exp_name+'.txt'), [np.mean(score)])

if __name__ == '__main__':
    main(args)
                




