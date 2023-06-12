from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np



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


model_type = "vit_h"
device = "cuda"
checkpoint = ""


sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)
img_path = ""
image_perturb = cv2.imread(img_path)
image_perturb = cv2.cvtColor(image_perturb, cv2.COLOR_BGR2RGB)
masks_perturb= mask_generator.generate(image_perturb)
result = generate_anns(masks_perturb)
save_path = ""

cv2.imwrite(save_path,result)
