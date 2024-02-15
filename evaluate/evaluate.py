"""
Computing psnr, ssim between result images and ground truth images. The result_path and gt_path contain them respectively, and the file names should be the same.
environment: py3.7-torch1.8
author: wangzeyu, Zhejiang University
date: 2022/02/22
"""
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

# from skimage import transform
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.color import rgb2ycbcr
import skimage.io as io

import lpips
from scipy import linalg

from cleanfid import fid

# from dataset import ImageDataset


def check_image_file(filename):
    """
    用于判断filename是否为图片
    """
    return any([filename.endswith(extention) for extention in
                ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']])


def compute_psnr_ssim(result_path, gt_path):
    """
    result_path : Path to output data
    gt_path : Path to ground truth data (test path)
    """
    from skimage import transform
    print('computing psnr and ssim')
    image_files = [file for root, dirs, files in os.walk(gt_path)
                        for file in files if check_image_file(file)]
    psnr_list, ssim_list = [], []
    for image in tqdm(image_files):
        # print(image)
        result_image = os.path.join(result_path, image)
        gt_image = os.path.join(gt_path, image)
        image1 = io.imread(result_image)
        image2 = io.imread(gt_image)
        # print(image1.shape, image2.shape)
        if len(image2.shape) == 2:
            # image2 = np.unsqueeze(image2, axis=2)
            # image2 = np.concatenate((image2, image2, image2), axis=2)
            image2 = np.stack((image2, image2, image2), axis=2)
        # exit()
        # image2 = transform.resize(image2, image1.shape[:2]) # 原数据集中的图片大小可能和推理得到的不一样，转换一下。同时这步操作会除以255

        # rgb2ycbcr的输入需要归一化到0-1.0的float
        # 这个在上一篇blog中讲过了rgb2ycbcr输出为浮点型且范围是0-255.0 所以需要再次归一化0-1
        image1 = image1/255.0
        image2 = image2/255.0   # 算的时候检查一下，如果已经是0-1的，那么不需要除以255
        image1 = rgb2ycbcr(image1)[:, :, 0:1]
        image2 = rgb2ycbcr(image2)[:, :, 0:1] 
        image1 = image1/255.0
        image2 = image2/255.0
        # print(image1.shape)
        # exit()

        psnr_val = peak_signal_noise_ratio(image1, image2)
        ssim_val = structural_similarity(image1, image2, win_size=5, gaussian_weights=True, channel_axis=2, data_range=1.0, K1=0.01, K2=0.03, sigma=1.5)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
    
    psnr = np.mean(psnr_list)
    ssim = np.mean(ssim_list)

    return psnr, ssim


def compute_lpips(result_path, gt_path):
    from torchvision import transforms
    print('computing lpips')
    loss_fn = lpips.LPIPS(net='squeeze', spatial=True) # Can also set net = 'squeeze' or 'vgg'
    image_files = [file for root, dirs, files in os.walk(gt_path)
                    for file in files if check_image_file(file)] 
    dist_list = []
    for image in tqdm(image_files):   
        result_image = os.path.join(result_path, image)
        gt_image = os.path.join(gt_path, image)
        result_image = lpips.im2tensor(lpips.load_image(result_image))
        gt_image = lpips.im2tensor(lpips.load_image(gt_image)) 
        gt_image = transforms.Resize(size=result_image.shape[2:])(gt_image)
        dist = loss_fn.forward(result_image, gt_image)
        dist_list.append(dist.mean().item())
    
    return np.mean(dist_list)


# def frechet_distance(mu, cov, mu2, cov2):
#     """
#     https://blog.csdn.net/qq_37758122/article/details/115537703
#     for computing FID
#     """
#     cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
#     dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
#     return np.real(dist)

        

def main(args):
    print(f'result images path: {args.result_path}')
    print(f'ground_truth images path (test set): {args.gt_path}')
    psnr, ssim = compute_psnr_ssim(args.result_path, args.gt_path)
    print('psnr:', psnr)
    print('ssim:', ssim)
    lpips = compute_lpips(args.result_path, args.gt_path)
    print('lpips:', lpips)
    # compute_fid(args.result_path, args.gt_path, img_size=512)
    fid_ = fid.compute_fid(args.result_path, args.gt_path)
    print('fid:', fid_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default='./inference_results/celeba_epoch60_mask_ratio0.2', type=str, help='inference result images root')
    parser.add_argument("--gt_path", default='/home/data/wangzeyu/Image_Inpainting/celebahq_512/test', type=str, help='ground truth images root (test root)')
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
