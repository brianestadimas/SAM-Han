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

    return patches, drop_indices

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
        # red
        patches[index] = np.ones_like(patches[index]) * [255, 0, 0]
        # add noise 1
        # patch_shape = patches[index].shape
        # noise = np.random.normal(loc=0, scale=1, size=patch_shape).astype(np.uint8) * 255
        # patches[index] = np.clip(noise, 0, 255)
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

def measure_star_positon(dropped_patches, resized_image, patch_size, selected_index, out_dir):
    dropped_patches = drop_patches_at_indices(dropped_patches, [selected_index])
    merged_image = np.vstack([np.hstack(row) for row in np.array_split(dropped_patches, resized_image.shape[0] // patch_size)])
    plt.figure(figsize=(10,10))
    plt.imshow(merged_image)
    #plt.axis('off')
    plt.savefig("{}.png".format(out_dir + '/new_1'), bbox_inches='tight', pad_inches = 0.0)
    plt.show()

parser = argparse.ArgumentParser(description='SAM')
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument("--occulsion_type", type=str, choices=['random', 'give_point'], default='random')
parser.add_argument("--drop_ratio", type=float, help="The ratio of the image patch to be dropped", default=0.1)
parser.add_argument("--optional", type=str)
parser.add_argument("--model_type", type=str)
parser.add_argument("--sam_checkpoint", type=str)
parser.add_argument("--point_coords_x", type=int)
parser.add_argument("--point_coords_y", type=int)
args = parser.parse_args()

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    sam_checkpoint = args.sam_checkpoint
    model_type = args.model_type
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # image_files = ['sa_227195', 'sa_228197', 'sa_232600', 'sa_233167', 'sa_234809']
    # args.optional = 'sa_227195.jpg'
    
    image_name = os.path.split(args.optional)[1].split('.')[0]
    #image_path = os.path.dirname(__file__) + '/datasets/sam_datasets/' + args.optional
    image_path = os.path.dirname(os.path.dirname(__file__)) + '/sam_data100/' + args.optional
    
    # import pdb;pdb.set_trace()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    resized_image = cv2.resize(image, (1024, 1024))
    
    out_dir = os.path.join(os.path.dirname(__file__) + "/results/", image_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    iou_dir = os.path.join(os.path.dirname(__file__) + "/results/", model_type + "_iou_ratio_" + str(args.drop_ratio))
    if not os.path.exists(iou_dir):
        os.makedirs(iou_dir)

    image_list = []
    image_list.append(resized_image)
    # show resized images
    # plt.figure(figsize=(10,10))
    # plt.imshow(resized_image)
    # #plt.axis('off')
    # plt.savefig("{}.png".format(out_dir + '/resized_image'), bbox_inches='tight', pad_inches = 0.0)
    # plt.show()
    
    # define patch size
    patch_size = 16
    
    # divide image into patches
    patches = divide_image_into_patches(resized_image, patch_size)
    
    if args.occulsion_type == 'random':
        # set drop ratio
        dropped_patches, drop_indices = random_drop_patches(patches, args.drop_ratio)
    
    # merge small pieces into an image
    merged_image = np.vstack([np.hstack(row) for row in np.array_split(dropped_patches, resized_image.shape[0] // patch_size)])
    image_list.append(merged_image)
    
    #### random point
    # Select a coordinate that is not in drop_indices
    valid_indices = set(range(len(patches))) - set(drop_indices)
    selected_index = np.random.choice(list(valid_indices))
    # debug
    # selected_index = 2048
    # measure_star_positon(dropped_patches, resized_image, patch_size, selected_index, out_dir)
    
    # Choose a coordinate that is not in drop_indices
    selected_row = selected_index // (1024 // patch_size)
    selected_col = selected_index % (1024 // patch_size)
    # Calculate the pixel position of the coordinate point
    x = selected_col * patch_size + patch_size // 2
    y = selected_row * patch_size + patch_size // 2
    #### random point
    
    input_point = np.array([[x, y]])    
    input_label = np.array([1])
        
    plt.figure(figsize=(10,10))
    plt.imshow(merged_image)
    # show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig("{}.png".format(out_dir + '/occulsion_image_' + model_type + '_ratio_'+str(args.drop_ratio)), bbox_inches='tight', pad_inches = 0.0)
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.imshow(resized_image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig("{}.png".format(out_dir + '/original_w_point_' + model_type + '_ratio_'+str(args.drop_ratio)), bbox_inches='tight', pad_inches = 0.0)
    plt.close() 
    
    plt.figure(figsize=(10,10))
    plt.imshow(merged_image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig("{}.png".format(out_dir + '/occulsion_w_point_' + model_type + '_drop_ratio_'+str(args.drop_ratio)), bbox_inches='tight', pad_inches = 0.0)
    plt.close()
    
    print("save occulsion images success")

    clean_mask = []
    occ_mask = []
    i = 0
    occulsion_names = ['original', 'occulsion']
    for image in image_list:
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        # point 
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        index = np.argmax(scores)
        mask_image = masks[index]
        
        if i == 0:
            clean_mask.append(mask_image)
        else:
            occ_mask.append(mask_image)
        
        blank_image = np.zeros_like(image)
        plt.figure()
        plt.imshow(blank_image)
        
        color = np.array([1.0, 1.0, 1.0])
        index = np.argmax(scores)
        mask_image = masks[index]
        result_image =mask_image[:,:,None]*color.reshape(1, 1, -1)
        
        plt.figure(figsize=(20,20))
        plt.imshow(result_image)
        plt.axis('off')
        plt.savefig("{}.png".format(out_dir + '/mask_' + model_type + '_' + str(occulsion_names[i]) +  '_drop_ratio_'+str(args.drop_ratio)), bbox_inches='tight', pad_inches = 0.0)
        # plt.show()
        
        print("save occlusion mask success")
        
        i += 1
        torch.cuda.empty_cache()
        import gc 
        gc.collect()
    iou = get_iou_matrix(clean_mask[0], occ_mask[0])
    # np.savetxt(os.path.dirname(__file__) + "/results/" + image_name + '_IoU_drop_ratio_'+str(args.drop_ratio) +'.txt', [np.mean(iou)])
    np.savetxt(iou_dir + "/"+ image_name + '_iou_ratio_'+str(args.drop_ratio) +'.txt', [np.mean(iou)])
    print(f"iou score: {iou}")
        
def cal_mIoU(args):
    folder_path = "./results/" + args.model_type + "_iou_ratio_" + str(args.drop_ratio)
    total_sum = 0
    file_count = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                content = file.read()
                value = float(content)
                total_sum += value
                file_count += 1

    if file_count > 0:
        average = total_sum / file_count
        print(f"file_count:{file_count}, iou score: {average}")
        np.savetxt("./results/" + "average_mIoU_" + args.model_type + "_" + str(args.drop_ratio) +'.txt', [np.mean(average)])
    else:
        print("nothing!")

if __name__ == "__main__":
    main(args)
    cal_mIoU(args)