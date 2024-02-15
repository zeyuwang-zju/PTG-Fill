import os
import random
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def check_image_file(filename):
    """
    用于判断filename是否为图片
    """
    return any([filename.endswith(extention) for extention in
                ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']])


def random_regular_mask(image):
    """Generate a random regular mask，任意数量、大小的方块
    image: 输入图片（真实数据集中的图片）(C, H, W)
    return: mask (1, H, W) (元素都为0和1。 0代表被遮挡部分)
    """
    mask = torch.ones_like(image)[0:1, :, :]
    s = image.size()
    N_mask = random.randint(1, 5)
    lim_x = s[1] - s[1] / (N_mask + 1)
    lim_y = s[2] - s[2] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(lim_x))
        y = random.randint(0, int(lim_y))
        range_x = x + random.randint(int(s[1] / (N_mask + 7)), min(int(s[1] - x), int(s[1] / 2)))
        range_y = y + random.randint(int(s[2] / (N_mask + 7)), min(int(s[2] - y), int(s[2] / 2)))
        mask[:, int(x) : int(range_x), int(y) : int(range_y)] = 0
    return mask


def center_mask(image):
    """Generate a center hole with 1/4*W and 1/4*H （在中央的方块）
    :param img: original image size C*H*W
    :return: mask
    """
    mask = torch.ones_like(image)[0:1, :, :]
    s = image.size()
    mask[:, int(s[1]/4):int(s[1]*3/4), int(s[2]/4):int(s[2]*3/4)] = 0
    return mask


def random_irregular_mask(image):
    """Generate a random irregular mask with lines, circles and ellipses
    :param img: original image size C*H*W
    :return: mask
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mask = torch.ones_like(image)[0:1, :, :]
    s = mask.size()
    image = np.zeros((s[1], s[2], 1), np.uint8)

    max_width = int(min(s[1]/10, s[2]/10))
    N_mask = random.randint(16, 64)
    for _ in range(N_mask):
        model = random.random()
        if model < 0.2: # Draw random lines
            x1, x2 = random.randint(1, s[1]), random.randint(1, s[1])
            y1, y2 = random.randint(1, s[2]), random.randint(1, s[2])
            thickness = random.randint(2, max_width)
            cv2.line(image, (x1, y1), (x2, y2), (1, 1, 1), thickness)
        elif (model > 0.2 and model < 0.5): # Draw random circles
            x1, y1 = random.randint(1, s[1]), random.randint(1, s[2])
            radius = random.randint(2, max_width)
            cv2.circle(image, (x1, y1), radius, (1, 1, 1), -1)
        else: # draw random ellipses
            x1, y1 = random.randint(1, s[1]), random.randint(1, s[2])
            s1, s2 = random.randint(1, s[1]), random.randint(1, s[2])
            a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
            thickness = random.randint(2, max_width)
            cv2.ellipse(image, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    image = image.reshape(s[2], s[1])
    image = Image.fromarray(image*255)

    image_mask = transform(image)
    for j in range(s[0]):
        mask[j, :, :] = image_mask

    return 1 - mask


def random_irregular_mask_with_ratio(image, ratio):
    """Generate a random irregular mask with lines, circles and ellipses
    :param img: original image size C*H*W
           ratio: the ratio ranges of holes of the whole images. range: [ratio, ratio+0.1]
    :return: mask
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mask = torch.ones_like(image)[0:1, :, :]
    s = mask.size()
    image = np.zeros((s[1], s[2], 1), np.uint8)

    max_width = int(min(s[1]/10, s[2]/10))
    while True:
        model = random.random()
        if model < 0.2: # Draw random lines
            x1, x2 = random.randint(1, s[1]), random.randint(1, s[1])
            y1, y2 = random.randint(1, s[2]), random.randint(1, s[2])
            thickness = random.randint(2, max_width)
            cv2.line(image, (x1, y1), (x2, y2), (1, 1, 1), thickness)
        elif (model > 0.2 and model < 0.5): # Draw random circles
            x1, y1 = random.randint(1, s[1]), random.randint(1, s[2])
            radius = random.randint(2, max_width)
            cv2.circle(image, (x1, y1), radius, (1, 1, 1), -1)
        else: # draw random ellipses
            x1, y1 = random.randint(1, s[1]), random.randint(1, s[2])
            s1, s2 = random.randint(1, s[1]), random.randint(1, s[2])
            a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
            thickness = random.randint(2, max_width)
            cv2.ellipse(image, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)
        
        # 判断mask_ratio是否在 [ratio, ratio+0.1]区间内
        if np.sum(image)/(s[1]*s[2]) > ratio and np.sum(image)/(s[1]*s[2]) < ratio + 0.1:
            break
        elif np.sum(image)/(s[1]*s[2]) > ratio:
            image.fill(0)

    image = image.reshape(s[2], s[1])
    image = Image.fromarray(image*255)

    image_mask = transform(image)
    for j in range(s[0]):
        mask[j, :, :] = image_mask

    return 1 - mask


class ImageDataset(Dataset):
    def __init__(self, image_root, load_size, crop_size, mask_type='random_irregular', mask_ratio=0.1, mask_root=None, train=True, return_image_root=False):
        super(ImageDataset, self).__init__()
        """
        image_root: 存放数据集图片的地址
        load_size: 读取图片后resize成的大小
        crop_size: 图片经resize之后随机裁剪出的大小
        mask_type: mask的类型. choices=[random_regular, random_irregular, center, self_load]. 一般就用random_irregular.
        mask_ratio: 仅当random_irregular时有效。例如设置为0.1时，mask比例在[0.1, 0.2]. 设置为1时，mask比例为任意
        mask_root: 当mask_type为self_load时，从mask_root中获取mask;其它类型的mask时改参数无作用
        train: [True, False] True时为训练集，False为测试集，区别为训练集会有水平方向任意反转。测试集不需要。
        return_image_root: 如果为True, 会返回改图片的地址. 一般只有inference时候会用.
        """
        self.image_files = [os.path.join(root, file) for root, dirs, files in os.walk(image_root)
                            for file in files if check_image_file(file)]
        self.number_image = len(self.image_files)
        self.load_size = load_size
        self.crop_size = crop_size
        self.return_image_root = return_image_root
        if train == True: # 训练集
            self.image_files_transforms = transforms.Compose([
                        transforms.Resize(size=load_size, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.RandomCrop(size=(crop_size, crop_size)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor()
                    ])
        else: # 测试集
            self.image_files_transforms = transforms.Compose([
                        transforms.Resize(size=(crop_size, crop_size), interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor()
                    ])

        if mask_type in ['random_regular', 'random_irregular', 'center']:
            self.generate_mask = eval(f'{mask_type}_mask')
            if mask_type == 'random_irregular' and mask_ratio != 1:
                self.generate_mask = random_irregular_mask_with_ratio
                self.mask_ratio = mask_ratio
        elif mask_type == 'self_load':
            self.generate_mask = None
            self.mask_files = [os.path.join(root, file) for root, dirs, files in os.walk(mask_root)
                            for file in files if check_image_file(file)]
            self.number_mask = len(self.mask_files)
            if train==True:
                self.mask_files_transforms = transforms.Compose([
                    transforms.Resize(size=load_size, interpolation=transforms.InterpolationMode.NEAREST), # 因为mask只有0和1两种数值，因此用nearestr
                    transforms.RandomCrop(size=(crop_size, crop_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor()
                ])
            else:
                self.mask_files_transforms = transforms.Compose([
                    transforms.Resize(size=(crop_size, crop_size), interpolation=transforms.InterpolationMode.NEAREST), # 因为mask只有0和1两种数值，因此用nearestr
                    transforms.ToTensor()
                ])

    def __getitem__(self, index):
        image = Image.open(self.image_files[index % self.number_image])
        ground_truth = self.image_files_transforms(image.convert('RGB'))

        if self.generate_mask != None:
            if self.generate_mask == random_irregular_mask_with_ratio:
                mask = self.generate_mask(ground_truth, self.mask_ratio).expand(3, self.crop_size, self.crop_size)
            else:
                mask = self.generate_mask(ground_truth).expand(3, self.crop_size, self.crop_size)
        else:
            mask = Image.open(self.mask_files[random.randint(0, self.number_mask - 1)])
            mask = self.mask_files_transforms(mask.convert('RGB'))
            threshold = 0.5
            ones = mask >= threshold
            zeros = mask < threshold
            mask.masked_fill_(ones, 1.0)
            mask.masked_fill_(zeros, 0.0)
            # ---------------------------------------------------
            # white values(ones) denotes the area to be inpainted
            # dark values(zeros) is the values remained
            # ---------------------------------------------------
            mask = 1 - mask

        input_image = ground_truth * mask

        if not self.return_image_root:
            return input_image, ground_truth, mask
        else:
            return input_image, ground_truth, mask, self.image_files[index % self.number_image]


    def __len__(self):
        return self.number_image


if __name__ == '__main__':
    # torch.manual_seed(1)
    # random.seed(1)
    image_root = "/home/data/wangzeyu/Image_Inpainting/CelebAMask-HQ/CelebA-HQ-img/"
    mask_root = "/home/data/wangzeyu/Image_Inpainting/CelebAMask-HQ/CelebAMask-HQ-mask-anno/"
    load_size = 256
    crop_size = 256
    mask_type = 'random_irregular'
    mask_ratio = [0.1, 0.2]
    # mask_type = 'self_load'

    dataset = ImageDataset(image_root, load_size, crop_size, mask_type, mask_ratio=[0.1, 0.2], mask_root=None, train=True)
    print(len(dataset))
    data = dataset[2]
    input_image, ground_truth, mask = data
    # print(max(ground_truth))

    # image = torch.randn(size=(3, 256, 256))
    # ratio = [0.1, 0.2]
    # mask = random_irregular_mask_with_ratio(image, ratio)
