import os
import glob

def get_image_filenames(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # 可以根据需要添加其他图片格式的扩展名
    image_filenames = []

    # 使用glob模块匹配指定文件夹下的所有图片文件
    for extension in image_extensions:
        search_pattern = os.path.join(folder_path, '*' + extension)
        image_filenames.extend(glob.glob(search_pattern))

    # 仅保留文件名，去除路径信息
    image_filenames = [os.path.basename(filename) for filename in image_filenames]

    return image_filenames

# 指定文件夹路径
folder_path = '/home/qiaoyu/SAM_Robustness/sam_data100'

# 获取所有图片文件名
image_filenames = get_image_filenames(folder_path)

# 将图片文件名保存到文本文件
output_file = '/home/qiaoyu/SAM_Robustness/select_100_new.txt'  # 指定输出文本文件路径

with open(output_file, 'w') as f:
    for filename in image_filenames:
        f.write(filename + '\n')
