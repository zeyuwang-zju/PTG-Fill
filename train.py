"""
environment: py3.7-torch1.8
author: wangzeyu, Zhejiang University
date: 2022/02/22
"""
import os
import argparse
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import ImageDataset
from models.generator import Generator
from models.discriminator import SN_PatchGAN
from loss_functions.loss_mymodel import VGG16FeatureExtractor, loss_refined, loss_coarse
from utils.utils import print_1, seed_torch



def load_pretrained_model(generator, discriminator, args):
    generator = generator.to(args.local_rank)
    discriminator = discriminator.to(args.local_rank)
    pretrained_model_path = os.path.join(f'./checkpoints/', args.pretrained_model)
    if dist.get_rank() == 0 and os.path.exists(pretrained_model_path):
        generator.load_state_dict(torch.load(pretrained_model_path)["generator"])
        discriminator.load_state_dict(torch.load(pretrained_model_path)["discriminator"])
        start_epoch = torch.load(pretrained_model_path)["epoch"]
        print_1(dist, f'loading pretrained model from {pretrained_model_path}. start from epoch {start_epoch+1}')
    else:
        start_epoch = 0
        print_1(dist, f'not found pretrained model from {pretrained_model_path}. Initialize model from epoch 1')
    generator = DDP(generator, device_ids=[args.local_rank], output_device=[args.local_rank], find_unused_parameters=False)
    discriminator = DDP(discriminator, device_ids=[args.local_rank], output_device=[args.local_rank], find_unused_parameters=False)
    return generator, discriminator, start_epoch


def prepare_dataloader(args):
    trainset = ImageDataset(args.train_image_root, args.load_size, args.crop_size, args.mask_type, args.mask_ratio_train, args.mask_root, train=True)
    print_1(dist, f'training set root: {args.train_image_root}.')
    print_1(dist, f'mask type: {args.mask_type}')
    if args.mask_type == 'self_load':
        print_1(dist, f'mask root: {args.mask_root}.')
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler)
    return trainloader


def train_generator(generator, optimizer_g, extractor, trainloader, epoch, scaler, args):
    start_time = time.time()
    trainloader.sampler.set_epoch(epoch)
    # # training generator
    generator.train()
    for batch_idx, (input_image, ground_truth, mask) in enumerate(trainloader):
        optimizer_g.zero_grad()
        input_image, ground_truth, mask = input_image.to(args.local_rank), ground_truth.to(args.local_rank), mask.to(args.local_rank)
        if args.use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                refined, coarse = generator(input_image, mask) # refined, coarse have not been completed
                loss_refined_ = loss_refined(refined, ground_truth, mask, extractor)
                loss_coarse_ = loss_coarse(coarse, ground_truth, mask)
                loss_g = loss_refined_ + loss_coarse_
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

        else:
            refined, coarse = generator(input_image, mask)
            loss_refined_ = loss_refined(refined, ground_truth, mask, extractor)
            loss_coarse_ = loss_coarse(coarse, ground_truth, mask)
            loss_g = loss_refined_ + loss_coarse_
            loss_g.backward()
            optimizer_g.step()

        print_1(dist, f'Training | generator | epoch: {epoch+1} | size: {args.crop_size} | batch: {batch_idx+1} / {len(trainloader)} | lr: {args.lr_g:.1e} | L_g: {loss_g:.3f} | L_refined: {loss_refined_:.3f} | L_coarse: {loss_coarse_:.3f} | time: {time.time()-start_time:.2f}s')
    print_1(dist, f'time of training generator for one epoch: {time.time()-start_time:.2f}s')


def train_discriminator(generator, discriminator, optimizer_d, trainloader, epoch, scaler, args):
    start_time = time.time()
    trainloader.sampler.set_epoch(epoch)
    generator.train()
    discriminator.train()
    for batch_idx, (input_image, ground_truth, mask) in enumerate(trainloader):
        optimizer_d.zero_grad()
        input_image, ground_truth, mask = input_image.to(args.local_rank), ground_truth.to(args.local_rank), mask.to(args.local_rank)
        if args.use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                refined, _ = generator(input_image, mask)
                comp_refined = mask * ground_truth + (1 - mask) * refined
                if args.loss_d == 'hinge':
                    loss_d = F.relu(1.0 - discriminator(ground_truth, mask)).mean() + F.relu(1.0 + discriminator(comp_refined, mask)).mean()
                elif args.loss_d == 'wasserstein':
                    loss_d = -discriminator(ground_truth, mask).mean() + discriminator(comp_refined, mask).mean()
                else:
                    raise ValueError('The discriminator loss must be hinge or wasserstein')
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)
                scaler.update()

        else:
            refined, _ = generator(input_image, mask)
            comp_refined = mask * ground_truth + (1 - mask) * refined
            if args.loss_d == 'hinge':
                loss_d = F.relu(1.0 - discriminator(ground_truth, mask)).mean() + F.relu(1.0 + discriminator(comp_refined, mask)).mean()
            elif args.loss_d == 'wasserstein':
                loss_d = -discriminator(ground_truth, mask).mean() + discriminator(comp_refined, mask).mean()
            else:
                raise ValueError('The discriminator loss must be hinge or wasserstein')
            loss_d.backward()
            optimizer_d.step()

        print_1(dist, f'Training | discriminator | epoch: {epoch+1} | size: {args.crop_size} | batch: {batch_idx+1} / {len(trainloader)} | lr: {args.lr_d:.1e} | L_d: {loss_d:.3f} | time: {time.time()-start_time:.2f}s ')
    print_1(dist, f'time of training discriminator for one epoch: {time.time()-start_time:.2f}s')


def train_generator_discriminator(generator, discriminator, optimizer_g, optimizer_d, extractor, trainloader, epoch, scaler, args):
    start_time = time.time()
    trainloader.sampler.set_epoch(epoch)
    # # training generator
    generator.train()
    discriminator.train()
    for batch_idx, (input_image, ground_truth, mask) in enumerate(trainloader):
        optimizer_d.zero_grad()
        optimizer_g.zero_grad()
        input_image, ground_truth, mask = input_image.to(args.local_rank), ground_truth.to(args.local_rank), mask.to(args.local_rank)
        if args.use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                refined, coarse = generator(input_image, mask)
                comp_refined = mask * ground_truth + (1 - mask) * refined 
                if args.loss_d == 'hinge':
                    loss_d = F.relu(1.0 - discriminator(ground_truth, mask)).mean() + F.relu(1.0 + discriminator(comp_refined, mask)).mean()
                elif args.loss_d == 'wasserstein':
                    loss_d = -discriminator(ground_truth, mask).mean() + discriminator(comp_refined, mask).mean()
                else:
                    raise ValueError('The discriminator loss must be hinge or wasserstein')
                loss_refined_ = loss_refined(refined, ground_truth, mask, extractor)
                loss_coarse_ = loss_coarse(coarse, ground_truth, mask)
                loss_adversarial = -discriminator(comp_refined, mask).mean()
                if args.finetune:
                    loss_g = loss_refined_ + args.weight_adv * loss_adversarial
                else:
                    loss_g = loss_refined_ + loss_coarse_ + args.weight_adv * loss_adversarial

                scaler.scale(loss_d).backward(retain_graph=True)
                scaler.scale(loss_g).backward()
                scaler.step(optimizer_d)
                scaler.step(optimizer_g)
                scaler.update()
        else:
            refined, coarse = generator(input_image, mask)
            comp_refined = mask * ground_truth + (1 - mask) * refined 
            if args.loss_d == 'hinge':
                loss_d = F.relu(1.0 - discriminator(ground_truth, mask)).mean() + F.relu(1.0 + discriminator(comp_refined, mask)).mean()
            elif args.loss_d == 'wasserstein':
                loss_d = -discriminator(ground_truth, mask).mean() + discriminator(comp_refined, mask).mean()
            else:
                raise ValueError('The discriminator loss must be hinge or wasserstein')
            loss_refined_ = loss_refined(refined, ground_truth, mask, extractor)
            loss_coarse_ = loss_coarse(coarse, ground_truth, mask)
            loss_adversarial = -discriminator(comp_refined, mask).mean()
            if args.finetune:
                loss_g = loss_refined_ + args.weight_adv * loss_adversarial
            else:
                loss_g = loss_refined_ + loss_coarse_ + args.weight_adv * loss_adversarial
            loss_d.backward(retain_graph=True)
            loss_g.backward()
            optimizer_d.step()
            optimizer_g.step()

        print_1(dist, f'Training | epoch:{epoch+1} | size:{args.crop_size} | batch:{batch_idx+1}/{len(trainloader)} | lr(g,d):{args.lr_g:.1e},{args.lr_d:.1e} | L_g:{loss_g:.3f} | L_refined:{loss_refined_:.3f} | L_coarse:{loss_coarse_:.3f} | L_adv_g:{loss_adversarial:.3f} | L_adv_d:{loss_d:.3f} | time:{time.time()-start_time:.2f}s')
    print_1(dist, f'time of training generator+discriminator for one epoch: {time.time()-start_time:.2f}s')


def save_images(args, generator, trainloader, epoch):
    save_root = f'./temp_results/{args.dataset}'
    os.makedirs(save_root, exist_ok=True)
    generator.train()
    input_image, ground_truth, mask = next(iter(trainloader))
    input_image, ground_truth, mask = input_image.to(args.local_rank), ground_truth.to(args.local_rank), mask.to(args.local_rank)
    refined, coarse = generator(input_image, mask)
    coarse_comp = mask * ground_truth + (1 - mask) * coarse
    refined_comp = mask * ground_truth + (1 - mask) * refined
    save_image_list = []
    for i in range(4):
        save_image_list.extend([ground_truth[i], 1-mask[i], input_image[i] + (1-mask[i]), coarse[i], coarse_comp[i], refined[i], refined_comp[i]])
    if dist.get_rank() == 0:
        save_image(save_image_list, fp=os.path.join(save_root, f'fig_epoch{epoch+1}.pdf'), nrow=7)


def save_model(args, generator, discriminator, epoch):
    os.makedirs(f'./checkpoints', exist_ok=True)
    if dist.get_rank() == 0:
        torch.save({"generator":generator.module.state_dict(), "discriminator":discriminator.module.state_dict(), "epoch":epoch+1}, f"./checkpoints/model_latest_{args.dataset}.pth")
        # torch.save({"generator":generator.module.state_dict(), "discriminator":discriminator.module.state_dict(), "epoch":epoch+1}, f"./checkpoints/model_epoch{epoch+1}_{args.dataset}.pth")
            

def main(args):
    seed_torch(args.seed)
    print_1(dist, args)

    trainloader = prepare_dataloader(args)
    num_epochs = args.num_epochs

    generator = Generator(base_channels=64, base_patch=16, kernel_size=3, activ='swish', norm_type='instance', init_type='xavier')
    discriminator = SN_PatchGAN(in_channels=4, out_channels=[64, 128, 256, 256, 256, 256], activation='leakyrelu', norm_type='instance', init_type='xavier')
    extractor = VGG16FeatureExtractor().cuda().eval()  # for computing style loss # 一定要在外面定义extractor，如果每次算loss都加载一遍则慢很多
    print_1(dist, generator)
    generator, discriminator, start_epoch = load_pretrained_model(generator, discriminator, args)
    optimizer_g = torch.optim.AdamW(params=generator.parameters(), lr=args.lr_g)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=args.lr_d)
    scaler = torch.cuda.amp.GradScaler(enabled=True) if args.use_amp else None

    for epoch in range(start_epoch, num_epochs):
        print_1(dist, f'====================================== epoch {epoch+1} ======================================')
        if args.train_mode == 'g':
            train_generator(generator, optimizer_g, extractor, trainloader, epoch, scaler, args)
        elif args.train_mode == 'd':
            train_discriminator(generator, discriminator, optimizer_d, trainloader, epoch, scaler, args)
        elif args.train_mode == 'g,d':
            train_generator_discriminator(generator, discriminator, optimizer_g, optimizer_d, extractor, trainloader, epoch, scaler, args)
        save_images(args, generator, trainloader, epoch)
        save_model(args, generator, discriminator, epoch)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='celeba', type=str)
    parser.add_argument("--use_amp", default=False, type=bool)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--lr_g", default=2e-4, type=float)
    parser.add_argument("--lr_d", default=2e-4, type=float)
    parser.add_argument("--loss_d", default='hinge', type=str, choices=['hinge', 'wasserstein'])
    parser.add_argument("--weight_adv", default=1e-1, type=float, help='the weight of adversarial loss when training generator')
    parser.add_argument("--load_size", default=512, type=int)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--mask_type", default='self_load', type=str, choices=['random_regular', 'random_irregular', 'center', 'self_load'])
    parser.add_argument("--mask_ratio_train", default=1, type=float, help='the mask ratio range. if None: random mask ratio. 若设为0.2，则mask比例为[0.2, 0.3]. 特殊情况：设置为1时，则任意mask比例。一般训练集设置为1')
    parser.add_argument("--train_image_root", default='/home/data/wangzeyu/Image_Inpainting/celebahq_512/train', type=str)
    parser.add_argument("--mask_root", default="/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/testing_mask_dataset/", type=str)
    parser.add_argument("--pretrained_model", default='model_latest_celeba_.pth', help='under the directory ./checkpoints')
    parser.add_argument("--train_mode", default='g', type=str, choices=['g', 'd', 'g,d'], help='training generator/discriminator/both')
    parser.add_argument("--seed", default=0, type=int, help='manual seed')
    parser.add_argument("--finetune", default=0, type=int, help='if 0, add L_coarse; if 1, remove L_coarse')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')

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
