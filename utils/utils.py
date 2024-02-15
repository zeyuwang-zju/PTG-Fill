import os
import random
import imghdr
import tqdm
import shutil
import torch

def random_split_dataset(dataset, train_perc, manual_seed=0):
    """
    输入一个dataset, 返回训练集和测试集，源目录不改动
    """
    train_size = int(train_perc * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(manual_seed))
    return train_dataset, test_dataset


def print_1(dist, text):
    """
    在分布式训练dist下，只打印一次
    """
    if dist.get_rank() == 0:
        print(text)


def seed_torch(seed=0):
    import torch
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fill_mask_with_randn(mask, mean=0, std=0.1):
    """
    mask: (B, C, H, W)
    where 0 means hole and 1 means valid.
    fill the holes with normal distribution.
    """
    # noise = torch.randn_like(mask, dtype=torch.float)
    noise = torch.normal(mean=0, std=0.1, size=mask.shape)
    print(mask)
    mask = noise * (1 - mask) + mask
    print(mask)
    # print(noise)


if __name__ == '__main__':
    seed_torch(0)
    mask = torch.randint(0, 2, size=(1, 1, 8, 8))
    fill_mask_with_randn(mask)
