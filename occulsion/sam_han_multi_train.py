import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, sam_model_registry_train, SamPredictor
from torch.autograd.grad_mode import no_grad
from torchvision.utils import save_image
import sys, math, os
import argparse
from PIL import Image
import random
import shutil
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
#BaseDataset = torchdata.Dataset
batch_size = 1
shuffle = True
import os.path
from os.path import join
from datetime import datetime
from segment_anything.utils.transforms import ResizeLongestSide
gpu_list = '5'  # Example: GPUs 0 and 1
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
device = "cuda"
#device = torch.device("cuda:0,1")
device_ids = [5]  # 使用 GPU 0 和 1
    
# IMG_EXTENSIONS = [
#     '.jpg', #'.JPG', '.jpeg', '.JPEG',
#     # '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
# ]

# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# def make_dataset(dir, fns=None):
#     images = []
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir

#     if fns is None:
#         for root, _, fnames in sorted(os.walk(dir)):
#             for fname in fnames:
#                 if is_image_file(fname):                
#                     path = os.path.join(root, fname)
#                     images.append(path)
#     else:
#         for fname in fns:
#             if is_image_file(fname):
#                 path = os.path.join(dir, fname)
#                 images.append(path)

#     return images
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
        #pp=1
    def __len__(self):
        return len(self.fns)
    
    def __getitem__(self, index):
        fn = self.fns[index]
        feature_n = fn.replace(".jpg", ".npy")
        #m_img = Image.open(join(self.datadir, fn)).convert('RGB')
        image = cv2.imread(join(self.datadir,'images', fn))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        resized_image = cv2.resize(image, (1024, 1024))

        features = (np.load(join(self.datadir,'features',feature_n))[0])
        features=torch.from_numpy(features).to(device)
        resized_image=torch.from_numpy(resized_image).to(device)
        #y = self.labels[index]
        return {'resized_image':resized_image,'features':features,'fn':feature_n}# y


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
parser.add_argument("--Change", default=True, type=bool)
parser.add_argument("--Showfig", default=True, type=bool)
parser.add_argument("--optional", type=int)
parser.add_argument("--model_type", type=str)
parser.add_argument("--sam_checkpoint", type=str)
parser.add_argument("--point_coords_x", type=int)
parser.add_argument("--point_coords_y", type=int)
parser.add_argument('--local_rank', type=int, default=0, help='if true, augment input with vgg hypercolumn feature')

args = parser.parse_args()
# kkk=ResizeLongestSide(sam_model.image_encoder.img_size)
# def set_trainimage(image):
#     # Transform the image to the form expected by the model
    
#     input_image = kkk.apply_image(image)
#     input_image_torch = torch.as_tensor(input_image, device=self.device)
#     input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
def main(args):
    image_embedding_avg = np.load('./model_output/mean_100.npy')
    image_embedding_avg = torch.from_numpy(image_embedding_avg).to(device)

    sam_checkpoint = args.sam_checkpoint
    model_type = args.model_type

    sam = sam_model_registry_train[model_type](checkpoint=sam_checkpoint)

    if sam_checkpoint is not None:
        with open(sam_checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)

    model = sam.image_encoder
    model =  model.to(device)
    # if (1):
    # model = nn.DataParallel(model)
    if device == 'cuda':
        # model = torch.nn.DataParallel(model)
        #model = model.module
        model = torch.nn.DataParallel(model)
        # cudnn.benchmark = True
        

    #if isinstance(sam, nn.DataParallel):
    # sam = sam.module
    num_epochs = 40
    optimizer = torch.optim.Adam(model.parameters()) 
    #lambda1 = lambda epoch: math.pow(1 - epoch / args.max_epoch, args.poly_exp)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                       momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # optimizer = torch.optim.sgd(model.parameters(), lr=0.001,
    #                     momentum=0.9, weight_decay=5e-4)



    dataset = MyDataset('./data/train')
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    pascal_val_dataset = MyDataset('./data/val1')
    eval_dataloader_pascalvoc = DataLoader(pascal_val_dataset, batch_size=1, shuffle=False)
    sam_val_dataset = MyDataset('./data/val')
    eval_dataloader_sam = DataLoader(sam_val_dataset, batch_size=1, shuffle=False)
    #eval_dataloader_sam = DataLoader(val_dataset, batch_size=1, shuffle=False)

    #predictor = SamPredictor(sam)
    #mse_loss = nn.MSELoss()
    iout_sum_ture=0
    for epoch in range(num_epochs):
        iteration=0
        for cb in train_dataloader:
            optimizer.zero_grad()
            resized_image = cb['resized_image']#.numpy()
            gt_feature= cb['features']  
            input_image_torch = resized_image.permute(0,3, 1, 2).contiguous()
            input_image_torch = input_image_torch.to(torch.float32)
            input_image_torch=sam.preprocess_change(input_image_torch)
            input_image_torch = input_image_torch.to(device)
            gt_feature = gt_feature.to(device)
            model.train()
            # with no_grad():
            #     image_embedding = model(input_image_torch)
            image_embedding = model(input_image_torch)
            image_embedding = image_embedding +image_embedding_avg
            cosine_sim_avg = torch.cosine_similarity(image_embedding.view(batch_size,-1),gt_feature.view(batch_size,-1),dim=-1).mean()
            loss = 1 - cosine_sim_avg
            #optimizer.zero_grad()
            loss.backward()
            #break
            optimizer.step()
            iteration=iteration+1
            #print(f'EPOCH: {cosine}')
            print(f'Epcoch {epoch} Iteration{iteration} Cosine_Sim_Avg: {cosine_sim_avg}')
            # if iteration == 1:
            #     break
        scheduler.step()
        ######################################################################################test-feature-simirility              
        # cosine_sim_val=0
        # ii = 0
        # #epoch.tostring
        # for cb1 in eval_dataloader:
        #     resized_image = cb1['resized_image']#.numpy()
        #     gt_feature= cb1['features']  
        #     input_image_torch = resized_image.permute(0,3, 1, 2).contiguous()
        #     input_image_torch = input_image_torch.to(torch.float32)
        #     input_image_torch=sam.preprocess(input_image_torch)
        #     input_image_torch = input_image_torch.to(device)
        #     gt_feature = gt_feature.to(device)
        #     model.eval()
        #     with no_grad():
        #         image_embedding = model(input_image_torch)
        #         image_embedding=image_embedding+image_embedding_avg
        #     #ppp=image_embedding-gt_feature
        #     cosine_sim_avg = torch.cosine_similarity(image_embedding.view(batch_size,-1),gt_feature.view(batch_size,-1),dim=-1).mean()
        #     cosine_sim_val= cosine_sim_val + cosine_sim_avg
        # cosine_sim_val = cosine_sim_val/25
        # print(f'Cosine_sim_Val: {cosine_sim_val}')
        # if(error2<cosine_sim_val):
        #     error2=cosine_sim_val
        #     torch.save(model, './model_output/'+str(epoch)+'_'+str(cosine_sim_val.cpu().numpy())+'_han_b.pth')

        #########################################################################################################################test-image-iou
        current_time = datetime.now().strftime("%d%H%M%S")
        sam_h = sam_model_registry['vit_h'](checkpoint='sam_vit_h_4b8939.pth')
        sam_change = sam_model_registry['vit_h'](checkpoint='sam_vit_h_4b8939.pth')
        sam_change.image_encoder=model.module
        sam_change.eval()
        sam.eval()
        sam_h.to(device)
        sam_change.to(device)
        predictor2 = SamPredictor(sam_change)
        predictor = SamPredictor(sam_h)
        iou_mean=0
        iou_sum=0
        number=0
        consine_predictor2_sum=0

        #sam_change.image_encoder=model.module
        for cb1 in eval_dataloader_pascalvoc:
            if(args.optional==number):
                break
            number=number+1
            featurename=cb1['fn'][0]
            image_name = os.path.basename(featurename)
            image_name=image_name[:-4]
            resized_image=cb1['resized_image'][0]
            resized_image = resized_image.cpu().numpy()
            out_dir = os.path.join(os.path.dirname(__file__) + "/test/"+'pascal_voc', image_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            input_point = np.array([[args.point_coords_x, args.point_coords_y]]) # prompt point coordinates
            input_label = np.array([1])

            predictor.set_image(resized_image)
            # point 

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            index = np.argmax(scores)
            mask_image = masks[index]
    ########################################################################################
            predictor2.set_image_AddAverage(resized_image)
            masks_change, scores_change, logits_change = predictor2.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            consine_predictor2 = torch.cosine_similarity(predictor2.features.view(1,-1),cb1['features'].view(1,-1),dim=-1)#calculate cosine_sim
            index1 = np.argmax(scores_change)
            mask_image_change = masks_change[index1]
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
                cv2.imwrite(out_dir + '/mask_yuan.png', mask_image_show*255)
                cv2.imwrite(out_dir + '/mask_' + str(epoch) + '_' + str(iou) + '_change.png', masks_change_show*255)
                cv2.imwrite(out_dir + '/original.png', resized_image)#cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
            iou_sum=iou+iou_sum
            #consine_predictor1_sum=consine_predictor1+consine_predictor1_sum
            consine_predictor2_sum=consine_predictor2.cpu().numpy()+consine_predictor2_sum
        iou_mean=iou_sum/number
        consine_predictor2_sum=consine_predictor2_sum/number
        iou_mean = np.round(iou_mean, decimals=4)
        consine_predictor2_sum = np.round(consine_predictor2_sum[0], decimals=4)
        #with open('./model_output/eval_log.txt', 'w') as file:
        file = open('./model_output/eval_log'+current_time+'.txt', 'a')
        file.write("pascalvoc: Epoch {}, iou {}, cosin_sim {} \n".format(epoch, iou_mean,consine_predictor2_sum))
        file.flush()  # 强制刷新缓冲区，确保写入文件
        if(iout_sum_ture<iou_mean):
            iout_sum_ture=iou_mean
            torch.save(model, './model_output/'+str(epoch)+'_'+str(iout_sum_ture)+'_han_b.pth')
        print(f"iou score: {iou_mean}")
        ##########################################################################################################3#####-sam
        iou_mean=0
        iou_sum=0
        number=0
        consine_predictor2_sum=0
        #sam_change.image_encoder=model.module
        for cb1 in eval_dataloader_sam:
            if(args.optional==number):
                break
            number=number+1
            featurename=cb1['fn'][0]
            image_name = os.path.basename(featurename)
            image_name=image_name[:-4]
            resized_image=cb1['resized_image'][0]
            resized_image = resized_image.cpu().numpy()
            out_dir = os.path.join(os.path.dirname(__file__) + "/test/sam/", image_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            input_point = np.array([[args.point_coords_x, args.point_coords_y]]) # prompt point coordinates
            input_label = np.array([1])
            #predictor = SamPredictor(sam_h)
            predictor.set_image(resized_image)
            # point 

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            index = np.argmax(scores)
            mask_image = masks[index]
    ########################################################################################

            #predictor2 = SamPredictor(sam_change)
            #debug 
            #predictor2.feature_name=featurename
            predictor2.set_image_AddAverage(resized_image)
            
            #######change
            masks_change, scores_change, logits_change = predictor2.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            consine_predictor2 = torch.cosine_similarity(predictor2.features.view(1,-1),cb1['features'].view(1,-1),dim=-1)#calculate cosine_sim
            index1 = np.argmax(scores_change)
            mask_image_change = masks_change[index1]
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
                cv2.imwrite(out_dir + '/mask_yuan.png', mask_image_show*255)
                cv2.imwrite(out_dir + '/mask_' + str(epoch) + '_' + str(iou) + '_change.png', masks_change_show*255)
                cv2.imwrite(out_dir + '/original.png', resized_image)#cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
            iou_sum=iou+iou_sum
            #consine_predictor1_sum=consine_predictor1+consine_predictor1_sum
            consine_predictor2_sum=consine_predictor2.cpu().numpy()+consine_predictor2_sum
        iou_mean=iou_sum/number
        consine_predictor2_sum=consine_predictor2_sum/number
        iou_mean = np.round(iou_mean, decimals=4)
        consine_predictor2_sum = np.round(consine_predictor2_sum[0], decimals=4)

        #with open('./model_output/eval_log.txt', 'w') as file:
        #file = open('./model_output/eval_log.txt', 'a')
        file.write("sam: Epoch {}, iou {}, cosin_sim {} \n".format(epoch, iou_mean,consine_predictor2_sum))
        file.flush()  # 强制刷新缓冲区，确保写入文件
        file.close()
        # if(iout_sum_ture<iou_mean):
        #     iout_sum_ture=iou_mean
        #     torch.save(model, './model_output/'+str(epoch)+'_'+str(iout_sum_ture)+'_han_b.pth')
        print(f"iou score: {iou_mean}")

    # np.savetxt(os.path.dirname(__file__) + "/results/" + image_name + '_IoU_drop_ratio_'+str(args.drop_ratio) +'.txt', [np.mean(iou)])
    #np.savetxt(iou_dir + "/"+ image_name + '_IoU_drop_ratio_'+str(args.drop_ratio) +'.txt', [np.mean(iou)])

torch.cuda.empty_cache()
import gc 
gc.collect()
    #iou = get_iou_matrix(clean_mask[0], occ_mask[0])
    # np.savetxt(os.path.dirname(__file__) + "/results/" + image_name + '_IoU_drop_ratio_'+str(args.drop_ratio) +'.txt', [np.mean(iou)])
    #np.savetxt(iou_dir + "/"+ image_name + '_iou_ratio_'+str(args.drop_ratio) +'.txt', [np.mean(iou)])
    #print(f"iou score: {iou}")
        
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
    #cal_mIoU(args)