"""
loss functions for partial convolution
"Image Inpainting for Irregular Holes Using Partial Convolutions" (2018) (Guilin Liu et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ---------------------
# VGG16 feature extract
# ---------------------
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        """
        重构出来的图片，输入进vgg-16网络，提取feature_maps.
        提取pool1, pool2, pool3三个特征层的feature_maps.
        return: [feature_map1, feature_map2, feature_map3]
        """
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, images):
        results = [images]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)

    return gram


def loss_tv(comp): # 是否可以把斜对角也加上 + torch.mean(torch.abs(comp[:, :, :-1, :-1] - comp[:, :, 1:, 1:])) + torch.mean(torch.abs(comp[:, :, 1:, :-1] - comp[:, :, :-1, 1:]))
    # shift one pixel and get difference (for both x, y, and diagonal direction)
    loss = torch.mean(torch.abs(comp[:, :, :, :-1] - comp[:, :, :, 1:])) + \
        torch.mean(torch.abs(comp[:, :, :-1, :] - comp[:, :, 1:, :])) + \
        torch.mean(torch.abs(comp[:, :, :-1, :-1] - comp[:, :, 1:, 1:])) + \
        torch.mean(torch.abs(comp[:, :, 1:, :-1] - comp[:, :, :-1, 1:]))
    return loss


def loss_l1(comp, ground_truths):
    l1 = nn.L1Loss()
    loss = l1(comp, ground_truths)
    return loss


def loss_perceptual(feat_comp, feat_out, feat_gt):
    l1 = nn.L1Loss()
    loss = 0.0
    for i in range(3):
        loss += (l1(feat_out[i], feat_gt[i]) + l1(feat_comp[i], feat_gt[i]))     
    return loss


def loss_hole(outputs, ground_truths, masks):
    """
    计算遮挡区域的L1-loss
    """
    l1 = nn.L1Loss()
    loss = l1((1 - masks) * outputs, (1 - masks) * ground_truths)
    return loss

def loss_valid(outputs, ground_truths, masks):
    """
    计算未遮挡区域的L1-loss
    """
    l1 = nn.L1Loss()
    loss = l1(masks * outputs, masks * ground_truths)
    return loss


def loss_style(feat_comp, feat_out, feat_gt):
    l1 = nn.L1Loss()
    loss = 0.0
    for i in range(3):
        loss += (l1(gram_matrix(feat_out[i]), gram_matrix(feat_gt[i])) + l1(gram_matrix(feat_comp[i]), gram_matrix(feat_gt[i])))
    return loss


def loss_refined(refined, ground_truths, masks, extractor):
    """
    outputs 为网络直接输出的结果。
    masks: 1为未遮挡区域，0为遮挡区域
    """
    comp = masks * ground_truths + (1 - masks) * refined # none-hole区域为原图片，hole区域为输出图片

    feat_comp = extractor(comp)
    feat_out = extractor(refined)
    feat_gt = extractor(ground_truths)

    loss_valid_ = loss_valid(refined, ground_truths, masks)
    loss_hole_ = loss_hole(refined, ground_truths, masks)
    loss_perceptual_ = loss_perceptual(feat_comp, feat_out, feat_gt)
    loss_style_ = loss_style(feat_comp, feat_out, feat_gt)
    loss_tv_ = loss_tv(comp)

    loss_total_ = loss_valid_ + 6 * loss_hole_ + 0.05 * loss_perceptual_ + 120 * loss_style_ + 0.05 * loss_tv_
    # print(f'loss_l1: {loss_l1_:.3f} | loss_perc: {loss_perceptual_:.3f} | loss_tv: {loss_tv_:.3f}')

    return loss_total_


def loss_coarse(coarse, ground_truths, masks):
    # comp = masks * ground_truths + (1 - masks) * coarse
    # loss_l1_ = loss_l1(comp, ground_truths)
    # loss_tv_ = loss_tv(comp)
    # loss_total_ = loss_l1_ + 0.05 * loss_tv_
    loss_valid_ = loss_valid(coarse, ground_truths, masks)
    loss_hole_ = loss_hole(coarse, ground_truths, masks)
    # loss_tv_ = loss_tv(comp)
    loss_total_ = loss_valid_ + 6 * loss_hole_

    return loss_total_

    


if __name__ == '__main__':
    outputs = torch.randn(size=(2, 3, 256, 256))
    ground_truths = torch.randn(size=(2, 3, 256, 256))

    # l = MAE_loss(outputs, ground_truths)
    # print(l)