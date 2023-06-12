import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry,sam_model_registry_train, SamPredictor
#from segment_anything import sam_model_registry, SamPredictor
from torchvision.utils import save_image
import sys, math, os
import argparse
from PIL import Image
import random
import shutil
from torchvision import transforms
from PIL import Image
from os.path import join
from torch.utils.data import DataLoader, Dataset
class MyDataset(Dataset):
    def __init__(self, datadir,fns=None):
        #self.data = data
        #self.labels = labels
        #sortkey = lambda key: os.path.split(key)[-1]
        self.datadir=datadir
        self.fns = fns or os.listdir(join(datadir, 'images'))
        #self.paths = sorted(make_dataset(datadir, fns), key=sortkey)
        #if size is not None:
        #    self.paths = self.paths[:size]
        pp=1
    def __len__(self):
        return len(self.fns)
    
    def __getitem__(self, index):
        fn = self.fns[index]
        feature_n = fn.replace(".jpg", ".npy")
        #m_img = Image.open(join(self.datadir, fn)).convert('RGB')
        image = cv2.imread(join(self.datadir,'images', fn))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        resized_image = cv2.resize(image, (1024, 1024))
        features = join(self.datadir,'features',feature_n)
        return {'resized_image':resized_image,'features_path':features}# y
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

parser = argparse.ArgumentParser(description='SAM')
parser.add_argument("--optional", type=int)
parser.add_argument("--point_coords_x", type=int)
parser.add_argument("--point_coords_y", type=int)
parser.add_argument("--model_type", type=str)
parser.add_argument("--sam_checkpoint", type=str)
parser.add_argument("--sam_checkpoint_change", type=str)
parser.add_argument("--Change", default=True, type=bool)
parser.add_argument("--Showfig", default=True, type=bool)
args = parser.parse_args()


'''
for the example images
'''
def main(args):
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    
    #sam_checkpoint = "/home/qiaoyu/SAM_Robustness/pretrain_model/sam_vit_h_4b8939.pth"
    sam_checkpoint = args.sam_checkpoint
    model_type = args.model_type
    sam_checkpoint_change = args.sam_checkpoint_change
    device = "cuda"
    sam_h = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    if(args.Change==True):
        sam_change = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam_b_save = torch.load(sam_checkpoint_change)
        sam_change.image_encoder = sam_b_save.module
    else:
        sam_change = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
    sam_h.to(device=device)
    sam_change.to(device=device)

    dataset = MyDataset('./data/val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    #print("save occulsion images success")
    iou_sum=0
    number=0
    for cb in dataloader:
        if(args.optional==number):
            break
        number=number+1
        featurename=cb['features_path'][0]
        image_name = os.path.basename(featurename)
        image_name=image_name[:-4]
        resized_image=cb['resized_image'][0]
        resized_image = resized_image.numpy()
            #handongshen
        # resized_image=torch.from_numpy(image).unsqueeze(0).cuda()
        # input_image_torch = resized_image.permute(0,3, 1, 2).contiguous()
        # input_image_torch = input_image_torch.to(torch.float32)
        # image=sam_h.preprocess(input_image_torch)
        # with torch.no_grad():
        #     encoder_h=sam_h.image_encoder(image)
        #     encoder_change=sam_change.image_encoder(image)
        #     encoder_change=encoder_change+ torch.from_numpy(np.load('./model_output/mean_100.npy')).cuda()
        #     l1=torch.cosine_similarity(encoder_h.flatten(),encoder_change.flatten(),dim=-1)
        out_dir = os.path.join(os.path.dirname(__file__) + "/test/", image_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if image_name == 'sa_227195':
            #input_point = np.array([[590, 400]]) # sa_227195
            input_point = np.array([[592, 396]]) # sa_227195
        elif image_name == 'sa_228197':
            input_point = np.array([[300, 600]]) # sa_228197
        elif image_name == 'sa_232600':
            input_point = np.array([[200, 720]]) # sa_232600
        elif image_name == 'sa_233167':
            input_point = np.array([[250, 600]]) # sa_233167
        elif image_name == 'sa_234809':
            input_point = np.array([[590, 400]]) # sa_234809
            #input_point = np.array([[595, 400]]) # sa_234809
        else:
            #input_point = np.array([[512, 512]]) # default
            print("Please specify the point coordinates!")
            input_point = np.array([[args.point_coords_x, args.point_coords_y]]) # prompt point coordinates
        input_point = np.array([[args.point_coords_x, args.point_coords_y]]) # prompt point coordinates
        input_label = np.array([1])
        predictor = SamPredictor(sam_h)
        predictor.set_image(resized_image)
        # point 
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
########################################################################################
        predictor2 = SamPredictor(sam_change)
        if(args.Change==True):
            predictor2.set_image_AddAverage(resized_image)
        else:  
            predictor2.set_image(resized_image)
        # point 
        index = np.argmax(scores)
        mask_image = masks[index]
        #######change
        masks_change, scores_change, logits_change = predictor2.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        index = np.argmax(scores_change)
        mask_image_change = masks_change[index]
        #occ_mask.append(mask_image_change)
        
        # blank_image = np.zeros_like(image)
        # plt.figure()
        # plt.imshow(blank_image)
        iou = get_iou_matrix(mask_image,mask_image_change)
        if(args.Showfig==True):
            color = np.array([1.0, 1.0, 1.0])
            index = np.argmax(scores)
            mask_image_show = masks[index]
            mask_image_show =mask_image_show[:,:,None]*color.reshape(1, 1, -1)
            #
            color = np.array([1.0, 1.0, 1.0])
            index_change = np.argmax(scores_change)
            masks_change_show = masks_change[index_change]
            masks_change_show =masks_change_show[:,:,None]*color.reshape(1, 1, -1)
            plt.figure(figsize=(20,20))
            plt.imshow(mask_image_show)
            plt.axis('off')
            plt.savefig("{}.png".format(out_dir + '/mask_' + str(iou)+  '_yuan'), bbox_inches='tight', pad_inches = 0.0)
            plt.close()
            # plt.show()
            plt.figure(figsize=(20,20))
            plt.imshow(masks_change_show)
            plt.axis('off')
            plt.savefig("{}.png".format(out_dir + '/mask_'  +str(iou)+  '_change'), bbox_inches='tight', pad_inches = 0.0)
            plt.close()
            # plt.show()
            print("save occlusion mask success")
            #############################  original
            plt.figure(figsize=(20,20))
            plt.imshow(resized_image)
            #plt.axis('off')
            plt.savefig("{}.png".format(out_dir + '/original'), bbox_inches='tight', pad_inches = 0.0)
            plt.close()
            #plt.show()
        #i += 1
        iou_sum=iou+iou_sum
        torch.cuda.empty_cache()
        import gc 
        gc.collect()
    iou_mean=iou_sum/number
    # np.savetxt(os.path.dirname(__file__) + "/results/" + image_name + '_IoU_drop_ratio_'+str(args.drop_ratio) +'.txt', [np.mean(iou)])
    #np.savetxt(iou_dir + "/"+ image_name + '_IoU_drop_ratio_'+str(args.drop_ratio) +'.txt', [np.mean(iou)])
    print(f"iou score: {iou_mean}")
        
def cal_mIoU(args):
    folder_path = "./test/" + "iou_drop_ratio_" + str(args.drop_ratio)
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
    else:
        print("nothing!")

if __name__ == "__main__":
    main(args)
    #cal_mIoU(args)