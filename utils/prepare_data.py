import os
import random
import tqdm
import shutil
import imghdr
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torchvision.utils import save_image

def split_data(data_dir, split=0.9):
    """
    给数据集图片的路径data_dir, 在该路径下创建两个文件夹train和test，按split比例随机划分
    """
    print('spliting dataset...')
    src_paths = []
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if imghdr.what(path) == None:
            continue
        src_paths.append(path)
    random.shuffle(src_paths)

    # separate the paths
    border = int(split * len(src_paths))
    train_paths = src_paths[:border]
    test_paths = src_paths[border:]
    print('train images: %d images.' % len(train_paths))
    print('test images: %d images.' % len(test_paths))

    # create dst directories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(train_dir) == False:
        os.makedirs(train_dir)
    if os.path.exists(test_dir) == False:
        os.makedirs(test_dir)

    # move the image files
    pbar = tqdm.tqdm(total=len(src_paths))
    for dset_paths, dset_dir in zip([train_paths, test_paths], [train_dir, test_dir]):
        for src_path in dset_paths:
            dst_path = os.path.join(dset_dir, os.path.basename(src_path))
            shutil.move(src_path, dst_path)
            pbar.update()
    pbar.close()


def check_image_file(filename):
    """
    用于判断filename是否为图片
    """
    return any([filename.endswith(extention) for extention in
                ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']])


def transform_data(data_dir, out_dir, out_size=256):
    """
    将data_dir下面的图片转换尺寸大小成out_size, 存入out_dir路径下面.
    data_dir下面可能还分不同类别的文件夹.转换存入out_dir后所有图片都只在out_dir一级目录下面.
    """
    os.makedirs(out_dir, exist_ok=True)
    image_files = [os.path.join(root, file) for root, dirs, files in os.walk(data_dir)
                        for file in files if check_image_file(file)]
    for image in tqdm.tqdm(image_files):
        image_ = Image.open(image)
        image_ = image_.resize((out_size, out_size))
        # image_ = image_.resize((446, 256))
        _, image_file_name = os.path.split(image)
        image_.save(os.path.join(out_dir, image_file_name))
        # break



if __name__ == '__main__':
    # split_data(data_dir="/home/data/wangzeyu/Image_Inpainting/ffhq/", split=0.9)
    # transform_data("/home/data/wangzeyu/CelebAMask-HQ/CelebA-HQ-img/train/", "/home/data/wangzeyu/Image_Inpainting/celebahq_256/train", 256)
    # transform_data("/home/data/wangzeyu/CelebAMask-HQ/CelebA-HQ-img/test/", "/home/data/wangzeyu/Image_Inpainting/celebahq_256/test", 256)
    # transform_data("/home/data/wangzeyu/ffhq/train/", "/home/data/wangzeyu/Image_Inpainting/ffhq_256/train", 256)
    # transform_data("/home/data/wangzeyu/ffhq/test/", "/home/data/wangzeyu/Image_Inpainting/ffhq_256/test", 256)
    # transform_data("/home/data/wangzeyu/Image_Inpainting/parisstreetview_256/paris_eval_gt_227/", "/home/data/wangzeyu/Image_Inpainting/parisstreetview_256/paris_eval_gt_256/", out_size=256)
    transform_data("/home/data/wangzeyu/Image_Inpainting/places2_my/train/", "/home/data/wangzeyu/Image_Inpainting/places2_my/train_all/", 256)
    # transform_data("/home/data/wangzeyu/Image_Inpainting/parisstreetview/paris_train_original/", "/home/data/wangzeyu/Image_Inpainting/parisstreetview/paris_train_256/", 256)