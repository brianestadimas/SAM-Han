from predictor import Predictor
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

args = parser.parse_args()


def main(args):

    # register SAM
    model_type = args.model_type
    checkpoint = args.sam_checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Register SAM-{model_type}..")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    mask_generator = Predictor(sam)
    
    # TODO

if __name__ == '__main__':
    main(args)
                
