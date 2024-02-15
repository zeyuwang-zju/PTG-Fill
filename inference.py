"""
Using the pretrained model to inference on test dataset.
environment: py3.7-torch1.8
author: wangzeyu, Zhejiang University
date: 2022/02/22
"""
import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import ImageDataset
from models.generator import Generator
from utils.utils import seed_torch


def load_pretrained_generator(generator, args):
    pretrained_model_path = os.path.join(f'./checkpoints/', args.pretrained_model)
    if os.path.exists(pretrained_model_path):
        generator.load_state_dict(torch.load(pretrained_model_path)["generator"])
        # discriminator.load_state_dict(torch.load(pretrained_model_path)["discriminator"])
        start_epoch = torch.load(pretrained_model_path)["epoch"]
        print(f'loading pretrained model from {pretrained_model_path}. start from epoch {start_epoch}')
    else:
        print(f'not found pretrained model from {pretrained_model_path}. Exit')
        exit()
    return generator, start_epoch


def inference(args):
    """
    在测试集上，用预训练好的generator做推理.同时将得到的结果（补全的图像）存到相应的位置。位置文件名与原测试集文件名相同。
    """
    # seed_torch(args.seed)

    generator = Generator(base_channels=64, base_patch=16, kernel_size=3, activ='swish', norm_type='instance', init_type='xavier')
    generator, epoch = load_pretrained_generator(generator, args)
    generator = generator.to(args.device)

    save_path = os.path.join('./inference_results/', f'{args.dataset}_epoch{epoch}_mask_ratio{args.mask_ratio_test}')
    os.makedirs(save_path, exist_ok=True)
    seed_torch(args.seed) # 随机种子放在这个位置可以保证生成出的mask每次都一样

    testset = ImageDataset(args.test_image_root, args.load_size, args.crop_size, args.mask_type, args.mask_ratio_test, args.mask_root, train=False, return_image_root=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f'Inference, saving images to {save_path}')
    for input_image, ground_truth, mask, image_root in tqdm(testloader):
        input_image, ground_truth, mask = input_image.to(args.device), ground_truth.to(args.device), mask.to(args.device)
        refined, _ = generator(input_image, mask)
        comp = mask * ground_truth + (1 - mask) * refined
        _, image_name = os.path.split(image_root[0])
        comp = comp[0]
        save_image(comp, fp=os.path.join(save_path, image_name))
        if args.save_masked:
            damaged_images = ground_truth * mask + (1 - mask)
            save_image(damaged_images, fp=os.path.join(save_path, 'masked_'+image_name)) # 被损坏的图片也可以存下来，如果不存的话注释掉这行 # 计算FID的时候，文件夹下不能有其它的文件
    print('Inference done!')



def main(args):
    inference(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='celeba', type=str)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--batch_size", default=1, type=int) # 推理的时候bs设为1比较方便
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--load_size", default=512, type=int)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--mask_type", default='self_load', type=str, choices=['random_regular', 'random_irregular', 'center', 'self_load'])
    parser.add_argument("--mask_ratio_test", default=0.2, type=float, help='the mask ratio range. if None: random mask ratio. 若设为0.2，则mask比例为[0.2, 0.3]')
    parser.add_argument("--test_image_root", default='/home/data/wangzeyu/Image_Inpainting/celebahq_512/test', type=str)
    parser.add_argument("--mask_root", default="/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.2-0.3/", type=str, help='only useful when mask_type is self_load.')
    parser.add_argument("--pretrained_model", default='model_latest_celeba.pth')
    parser.add_argument("--save_masked", default=False, type=bool, help='whether to save the masked input. warning: do not use it for calculating FID')
    parser.add_argument("--seed", default=0, type=int, help='manual seed')
    args = parser.parse_args()


    main(args)

"""
88服务器：
imagenet路径：/home/data/wangzeyu/Imagenet/ILSVRC2012/train /home/data/wangzeyu/Imagenet/ILSVRC2012/val (1280000+)
celebA-HQ路径：/home/data/wangzeyu/Image_Inpainting/celebahq_512/train  /home/data/wangzeyu/Image_Inpainting/celebahq_512/test（30000）
ffhq路径：/home/data/wangzeyu/Image_Inpainting/ffhq_512/train  /home/data/wangzeyu/Image_Inpainting/ffhq_512/test（70000）
palces2路径：/home/data/wangzeyu/Image_Inpainting/Places2_512/train /home/data/wangzeyu/Image_Inpainting/Places2_512/val /home/data/wangzeyu/Image_Inpainting/Places2_512/test
partial_conv mask路径："/home/data/wangzeyu/Image_Inpainting/mask_pconv/irregular_mask/disocclusion_img_mask/" "/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/testing_mask_dataset/" 一般用后面的test

"/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.0-0.1/"
"/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.1-0.2/"
"/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.2-0.3/"
"/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.3-0.4/"
"/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.4-0.5/"
"/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.5-0.6/"
"""
