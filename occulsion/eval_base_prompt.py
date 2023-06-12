from segment_anything import sam_model_registry
from automatic_mask_generator import SamAutomaticMaskGenerator
import cv2
import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3) * 255, [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    return img
    
def generate_anns(anns):
    if len(anns) == 0:
        return
    anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))
    for ann in anns:
        m = ann['segmentation']
        color_mask = np.random.randint(low=0,high=255,size=3).tolist()
        for i in range(3):
            img[:,:,i][m] = color_mask[i]
    return img

def calculate_iou(gt, dt):
    """
    Compute mIoU between two masks in a batch
    :param gt (numpy array, uint8, B x H x W): binary masks
    :param dt (numpy array, uint8, B x H x W): binary masks
    :return: mean of IoU
    """
    intersection = ((gt * dt) > 0).sum(axis=(1,2))
    union = ((gt + dt) > 0).sum(axis=(1,2))
    boundary_iou = (intersection + 1) / (union + 1)
    return np.mean(boundary_iou)

def save_automatic_mask(args, image, masks, image_name, prefix):
    result = generate_anns(masks)
    #result = show_anns(masks)
    
    dir = os.path.join(args.save_path, 'drop_ratio_'+str(args.drop_ratio))
    if not os.path.exists(dir):
        os.makedirs(dir)

    mask_image_name = 'mask_' + prefix + image_name
    out = os.path.join(dir, mask_image_name)
    cv2.imwrite(out,result)
    print(out)

def eval_one_image(args, mask_generator, clean, perturbed, image_name):
    mask_c = mask_generator.generate(clean)
    clean_masks = mask_generator.masks
    clean_masks = torch.cat(clean_masks, dim=0)
    clean_masks = clean_masks.numpy().astype(np.uint8)
    mask_generator.reset()

    mask_p = mask_generator.generate(perturbed)
    perturbed_masks = mask_generator.masks
    perturbed_masks = torch.cat(perturbed_masks, dim=0)
    perturbed_masks = perturbed_masks.numpy().astype(np.uint8)
    mask_generator.reset()

    save_automatic_mask(args, clean, mask_p, image_name, 'perturbed_')
    
    save_automatic_mask(args, clean, mask_c, image_name, 'clean_')
    
    score = calculate_iou(clean_masks, perturbed_masks)
    
    return score


def main(args):

    # register SAM
    model_type = args.model_type
    checkpoint = args.sam_checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Register SAM-{model_type}..")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam, multi_mask=args.multi_mask)
    
    # image list
    img_list = np.loadtxt(args.path_img_list, dtype=str)
    clean_img_dir = args.path_clean_img_dir
    perturbed_img_dir = args.path_perturbed_img_dir

    with torch.no_grad():
        score = []
        for img in tqdm(img_list):
            clean_img = cv2.imread(os.path.join(clean_img_dir, img))
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            
            perturbed_img = cv2.imread(os.path.join(perturbed_img_dir, img))
            perturbed_img = cv2.cvtColor(perturbed_img, cv2.COLOR_BGR2RGB)

            s = eval_one_image(args, mask_generator, clean_img, perturbed_img, img)
            score.append(s)
            # break
        print(np.mean(score))
        np.savetxt(os.path.join(args.save_path, args.exp_name+'.txt'), [np.mean(score)])

parser = argparse.ArgumentParser(description="Sam Robustness Evaluation on Commom Corruption")
parser.add_argument("--exp-name", type=str, help="Name of experiments. This will be also result file name.")
parser.add_argument("--save-path", type=str, default = '/home/qiaoyu/SAM_Robustness/occulsion/results', help="Save path where the evaluation result is saved.")
parser.add_argument("--model-type", type=str, default = 'vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
parser.add_argument("--sam-checkpoint", type=str, default='/home/qiaoyu/SAM_Robustness/pretrain_model/sam_vit_h_4b8939.pth')
parser.add_argument("--path-img-list", type=str, default='/home/qiaoyu/SAM_Robustness/select_100_new.txt')
parser.add_argument("--path-clean-img-dir", type=str, default='/home/qiaoyu/SAM_Robustness/occulsion/sam_sampling_100/sam_original')
parser.add_argument("--path-perturbed-img-dir", type=str, default='/home/qiaoyu/SAM_Robustness/occulsion/sam_sampling_100/sam_occulsion/drop_ratio_0.1')
parser.add_argument("--multi-mask", type=bool, default=True, help="True: mask generator gives one highes score mask per prompt. False: mask generator gives one masks per prompt.")
parser.add_argument("--drop_ratio", type=float, help="The ratio of the image patch to be dropped", default=0.1)
args = parser.parse_args()

if __name__ == '__main__':
    main(args)

