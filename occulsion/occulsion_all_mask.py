import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from torchvision.utils import save_image
import sys, math, os
import argparse
from PIL import Image
import random
import shutil
from torchvision import transforms
from PIL import Image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
def divide_image_into_patches(image, patch_size):
    height, width, _ = image.shape
    patch_height = height // patch_size
    patch_width = width // patch_size
    patches = []

    for i in range(patch_height):
        for j in range(patch_width):
            patch = image[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
            patches.append(patch) 
    return patches

def random_drop_patches(patches, drop_ratio):
    num_patches = len(patches)
    # import pdb;pdb.set_trace()
    num_drop_patches = int(num_patches * drop_ratio)
    drop_indices = np.random.choice(num_patches, size=num_drop_patches, replace=False)

    for index in drop_indices:
        #patches[index] = np.zeros_like(patches[index])
        patch_shape = patches[index].shape
        noise = np.random.normal(loc=0, scale=1, size=patch_shape).astype(np.uint8) * 255
        patches[index] = np.clip(noise, 0, 255)

    return patches

def add_gaussian_noise(patch, mean, std_dev):
    noise = np.random.normal(loc=mean, scale=std_dev, size=patch.shape)
    noisy_patch = patch.astype(np.float32) + noise.astype(np.float32)
    noisy_patch = np.clip(noisy_patch, 0, 255).astype(np.uint8)  # Clip values to valid range
    return noisy_patch

def drop_patches_at_indices(patches, drop_indices):
    #patches = patches.astype(np.float32)  # Convert patches to float32 data type
    for index in drop_indices:
        # black
        #patches[index] = np.zeros_like(patches[index])
        # white
        #patches[index] = np.ones_like(patches[index]) * 255
        # add noise 1
        patch_shape = patches[index].shape
        noise = np.random.normal(loc=0, scale=1, size=patch_shape).astype(np.uint8) * 255
        patches[index] = np.clip(noise, 0, 255)
        # # add noise 2
        #patches[index] = add_gaussian_noise(patches[index], mean=0, std_dev=1)

    return patches

def find_patch_at_coordinates(image, patch_size, coordinates):
    patch_index = []
    for i in range(len(coordinates)):
        patch_row, patch_col = coordinates[i]
        num_cols = math.ceil(patch_row / patch_size)
        num_row = math.ceil(patch_col / patch_size)
        tmp = (num_row - 1) * (image.shape[0] // patch_size)  + num_cols - 1

        patch_index.append(tmp)
    return patch_index

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def get_iou_matrix(masks_gt, masks_pred):
    # area = []
    # shape = (len(masks_gt), 1)
    # iou_matrix = np.zeros(shape=shape)
    # for gt_idx in range(shape[0]):
    #     iou = calculate_iou(masks_gt[gt_idx, 0], masks_pred[gt_idx, 0])
    #     iou_matrix[gt_idx, 0] = iou
    #     # area.append(masks_gt[gt_idx]['area'])

    intersection = np.logical_and(masks_gt, masks_pred)
    union = np.logical_or(masks_gt, masks_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def add_occulsion(image, occulsion_type='random', drop_ratio=0.1):
    if image.shape[0] != 1024 or image.shape[1] != 1024:
        resized_image = cv2.resize(image, (1024, 1024))
    
    # define patch size
    patch_size = 16
    # divide image into patches
    patches = divide_image_into_patches(resized_image, patch_size)
    
    if occulsion_type == 'random':
        # set drop ratio
        dropped_patches = random_drop_patches(patches, drop_ratio)
        merged_image = np.vstack([np.hstack(row) for row in np.array_split(dropped_patches, resized_image.shape[0] // patch_size)])
    
    return merged_image

parser = argparse.ArgumentParser(description='SAM')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument("--occulsion_type", type=str, choices=['random', 'give_point'], default='random')
parser.add_argument("--drop_ratio", type=float, help="The ratio of the image patch to be dropped", default=0.1)
parser.add_argument("--optional", type=str)
parser.add_argument("--point_coords_x", type=int)
parser.add_argument("--point_coords_y", type=int)
args = parser.parse_args()


def save_resize_image():
    dir = os.path.join('/home/qiaoyu/SAM_Robustness/occulsion/sam_sampling_100', 'sam_'+ 'original')
    if not os.path.exists(dir):
        os.makedirs(dir)
    img_list = np.loadtxt('/home/qiaoyu/SAM_Robustness/select_100_new.txt', dtype=str)

    for file_name in img_list:
        image = cv2.imread(os.path.join('/home/qiaoyu/SAM_Robustness/sam_data100',file_name))
        if image.shape[0] != 1024 or image.shape[1] != 1024:
            resized_image = cv2.resize(image, (1024, 1024))
        name = os.path.join(dir, file_name)
        cv2.imwrite(name, resized_image)
        print(name)
    
    
drop_ratio = 0.6
if __name__ == '__main__':
    save_resize_image()
    
    dir = os.path.join('/home/qiaoyu/SAM_Robustness/occulsion/sam_sampling_100', 'sam_'+ 'occulsion', 'drop_ratio_'+str(drop_ratio))
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    img_list = np.loadtxt('/home/qiaoyu/SAM_Robustness/select_100_new.txt', dtype=str)

    for file_name in img_list:
        image = cv2.imread(os.path.join('/home/qiaoyu/SAM_Robustness/sam_data100',file_name))
        corrupted = add_occulsion(image, occulsion_type='random', drop_ratio=drop_ratio)
        name = os.path.join(dir, file_name)
        cv2.imwrite(name, corrupted)
        print(name)
