import os, sys

from torchvision import transforms
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.weights_init import weights_init

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == None:
        nonlinearity_layer = nn.Sequential()
    elif activation_type.lower() == 'relu':
        nonlinearity_layer = nn.ReLU()
    elif activation_type.lower() == 'gelu':
        nonlinearity_layer = nn.GELU()
    elif activation_type.lower() == 'leakyrelu':
        nonlinearity_layer = nn.LeakyReLU(0.2)
    elif activation_type.lower() == 'prelu':
        nonlinearity_layer = nn.PReLU()
    elif activation_type.lower() == 'swish':
        nonlinearity_layer = Swish()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_norm_layer(channels, norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == None:
        norm_layer = nn.Sequential()
    elif norm_type.lower() == 'batch':
        norm_layer = nn.BatchNorm2d(channels, affine=True, track_running_stats=True)
    elif norm_type.lower() == 'instance':
        norm_layer = nn.InstanceNorm2d(channels, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# class PartialConv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         """
#         params: in_channels, out_channels, kernel_size, stride, padding, bias
#         """
#         super(PartialConv2d, self).__init__(*args, **kwargs)
#         self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
#         self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]
#         # self.update_mask = None
#         # self.mask_ratio = None

#     def forward(self, input, mask_in=None):
#         assert len(input.shape) == 4
#         with torch.no_grad():
#             if self.weight_maskUpdater.type() != input.type():
#                 self.weight_maskUpdater = self.weight_maskUpdater.to(input)
#             mask = mask_in
#             self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
#             self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-6)    # for mixed precision training, change 1e-8 to 1e-6
#             self.update_mask1 = torch.clamp(self.update_mask, 0, 1)
#             self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask1)
#         raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)
#         if self.bias is not None:
#             bias_view = self.bias.view(1, self.out_channels, 1, 1)
#             output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
#             output = torch.mul(output, self.update_mask1)
#         else:
#             output = torch.mul(raw_out, self.mask_ratio)
#         # print(self.update_mask / self.slide_winsize)
#         return output, self.update_mask / self.slide_winsize   # replace the valid value to confident score


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """
        params: in_channels, out_channels, kernel_size, stride, padding, bias
        """
        super(PartialConv2d, self).__init__(*args, **kwargs)
        self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]
        # self.update_mask = None
        # self.mask_ratio = None

    def forward(self, input, mask_in=None):
        mask_bool = (torch.mean(mask_in, dim=1) == 1).int().unsqueeze(1)
        # print(mask_in)
        # print(mask_bool)
        # print(mask_bool.shape)
        assert len(input.shape) == 4
        with torch.no_grad():
            if self.weight_maskUpdater.type() != input.type():
                self.weight_maskUpdater = self.weight_maskUpdater.to(input)
            mask = mask_in
            self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
            self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)    # for mixed precision training, change 1e-8 to 1e-6 # 可能1e-6更好一些
            self.update_mask1 = torch.clamp(self.update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask1)
        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask1)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        mask_bool = transforms.Resize(self.update_mask.shape[2:], interpolation=transforms.InterpolationMode.NEAREST)(mask_bool)
        mask_out = torch.clamp(mask_bool + (self.update_mask / self.slide_winsize), 0, 1)
        # mask_out = self.update_mask / self.slide_winsize
        return output, mask_out   # replace the valid value to confident score


class PConvNormActiv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, nb_conv, norm='instance', activ='relu', bias=False):
        """
        nb_Conv: number of convolutional layers;
        norm: ['batch', 'instance', None]
        activ: ['relu', 'gelu', 'leakyrelu', 'prelu']
        (partial_conv => norm => activ) × nb_conv  # norm and activ are only conducted on images, not on masks;
        stride is only set to the first block, while the stride in following blocks is 1.
        padding has been set to (kernel_size-1)//2 to fit the output sizes.
        """
        super(PConvNormActiv, self).__init__()
        self.nb_conv = nb_conv
        setattr(self, 'conv0', PartialConv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias))
        setattr(self, 'norm0', get_norm_layer(out_channels, norm))
        setattr(self, 'activ0', get_nonlinearity_layer(activ))
        for i in range(1, nb_conv, 1):
            setattr(self, f'conv{i}', PartialConv2d(out_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias))
            setattr(self, f'norm{i}', get_norm_layer(out_channels, norm))
            setattr(self, f'activ{i}', get_nonlinearity_layer(activ))

    def forward(self, images, masks):
        for i in range(self.nb_conv):
            images, masks = getattr(self, f'conv{i}')(images, masks)
            images = getattr(self, f'norm{i}')(images)
            images = getattr(self, f'activ{i}')(images)
        return images, masks


class ToRGB(nn.Module):
    def __init__(self, in_channels):
        super(ToRGB, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 3, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv1x1(x))


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # x: (B, C, H, W)
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return x



class PC2FTransformer(nn.Module):
    """
    Pyramid Coarse-to-Fine Transformer
    """
    def __init__(self, base_channels=64, base_patch=16, nhead=8, dropout=0.1, num_layers=4):
        super(PC2FTransformer, self).__init__()
        """
        num_patches = (h // base_patch) * (w // base_patch)
        dim = base_patch * base_patch * 3 // down_scale
        """
        self.base_patch = base_patch
        dim = base_patch * base_patch * 3
        self.proj1 = nn.Conv2d(in_channels=base_channels*2, out_channels=dim, kernel_size=base_patch//2, stride=base_patch//2, padding=0)
        self.proj2 = nn.Conv2d(in_channels=base_channels*4, out_channels=dim, kernel_size=base_patch//4, stride=base_patch//4, padding=0)
        self.proj3 = nn.Conv2d(in_channels=base_channels*8, out_channels=dim, kernel_size=base_patch//8, stride=base_patch//8, padding=0)
        self.proj4 = nn.Conv2d(in_channels=base_channels*16, out_channels=dim, kernel_size=base_patch//16, stride=base_patch//16, padding=0)
        transformer_layer = nn.TransformerEncoderLayer(dim, nhead, dim*2, dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)

    def forward(self, D1, D2, D3, D4, coarse_comp):
        """
        D1: (B, base_channels*2, H//2, W//2)
        D2: (B, base_channels*4, H//4, W//4)
        D3: (B, base_channels*8, H//8, W//8)
        D4: (B, base_channels*16, H//16, W//16)
        coarse_comp: (B, 3, H, W) # has been completed
        """
        b, _, h, w = coarse_comp.shape
        o1 = self.proj1(D1).flatten(2).transpose(1, 2)                                                          # (B, num_patches, dim)
        o2 = self.proj2(D2).flatten(2).transpose(1, 2)                                                          # (B, num_patches, dim)
        o3 = self.proj3(D3).flatten(2).transpose(1, 2)                                                          # (B, num_patches, dim)
        o4 = self.proj4(D4).flatten(2).transpose(1, 2)                                                          # (B, num_patches, dim)
        o = torch.cat((o1, o2, o3, o4), dim=1)                                                                  # (B, num_patches*4, dim)   # 如果直接加起来，不收敛
        o = self.transformer(o)                                                                                 # (B, num_patches*4, dim) 
        o = o.view(b, 4, h // self.base_patch, w // self.base_patch, self.base_patch, self.base_patch, 3)       # (B, 4, H//base_patch, W//base_patch, base_patch, base_patch, 3)  
        o = o.permute(0, 1, 6, 2, 4, 3, 5).contiguous()                                                         # (B, 4, 3, H//base_patch, base_patch, W//base_patch, base_patch)
        o = o.view(b, -1, h, w)                                                                                 # (B, 12, H, W)
        o = torch.cat((o, coarse_comp), dim=1)                                                                  # (B, 15, H, W)
        return o


class revisedgatedconv2d(nn.Module):
    """
    revised gated convolution which is aware of the edges of masks
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                norm='instance', activ='relu'):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.hole_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.valid_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm_layer(out_channels, norm)
        self.activ = get_nonlinearity_layer(activ)
        self.sigmoid = nn.Sigmoid()
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Sequential()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, stride, stride, 0)

    def forward(self, input, mask):
        mask = transforms.Resize(size=input.shape[2:], interpolation=transforms.InterpolationMode.NEAREST)(mask)
        x = self.conv2d(input)
        hole = self.hole_conv2d(input*(1-mask))
        valid = self.valid_conv2d(input*mask)
        comp = self.sigmoid(hole + valid)
        x = self.activ(self.norm(x * comp))
        x = x + self.residual(input)
        return x


class GConvNormActiv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, nb_conv, mode='down', norm='instance', activ='relu', bias=False):
        """
        nb_Conv: number of convolutional layers;
        norm: ['batch', 'instance', None]
        activ: ['relu', 'gelu', 'leakyrelu', 'prelu']
        (gated_conv => norm => activ) × nb_conv  # norm and activ are only conducted on images, not on masks;
        mode: ['down', 'up', None]
        stride is only set to the first block, while the stride in following blocks is 1. the stride is achieved by maxpool or interpolate
        padding has been set to (kernel_size-1)//2 to fit the output sizes.
        """
        super(GConvNormActiv, self).__init__()
        self.nb_conv = nb_conv
        self.mode = mode
        self.stride = stride
        setattr(self, 'conv0', revisedgatedconv2d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias, norm=norm, activ=activ))
        for i in range(1, nb_conv, 1):
            setattr(self, f'conv{i}', revisedgatedconv2d(out_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias, norm=norm, activ=activ))

    def forward(self, x, mask):
        for i in range(self.nb_conv):
        # for i in range(1):
            x = getattr(self, f'conv{i}')(x, mask)
            if i == 1:
                if self.mode =='down':
                    x = F.max_pool2d(x, self.stride)
                elif self.mode == 'up':
                    x = F.interpolate(x, scale_factor = self.stride)
        return x


class Generator(nn.Module):
    def __init__(self, base_channels=64, base_patch=16, kernel_size_p=5, kernel_size_g=3, activ='swish', norm_type='instance', init_type='xavier'):
        super(Generator, self).__init__()
        self.encoder0 = PConvNormActiv(in_channels=3, out_channels=base_channels, kernel_size=kernel_size_p, stride=1, nb_conv=1, norm=norm_type, activ=activ)
        self.encoder1 = PConvNormActiv(in_channels=base_channels, out_channels=base_channels*2, kernel_size=kernel_size_p, stride=2, nb_conv=2, norm=norm_type, activ=activ)
        self.encoder2 = PConvNormActiv(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=kernel_size_p, stride=2, nb_conv=2, norm=norm_type, activ=activ)
        self.encoder3 = PConvNormActiv(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=kernel_size_p, stride=2, nb_conv=2, norm=norm_type, activ=activ)
        self.encoder4 = PConvNormActiv(in_channels=base_channels*8, out_channels=base_channels*16, kernel_size=kernel_size_p, stride=2, nb_conv=2, norm=norm_type, activ=activ)
        self.encoder5 = PConvNormActiv(in_channels=base_channels*16, out_channels=base_channels*16, kernel_size=kernel_size_p, stride=2, nb_conv=5, norm=norm_type, activ=activ)

        self.decoder5 = PConvNormActiv(in_channels=base_channels*(16+16), out_channels=base_channels*16, kernel_size=kernel_size_p, stride=1, nb_conv=2, norm=norm_type, activ=activ)
        self.decoder4 = PConvNormActiv(in_channels=base_channels*(16+8), out_channels=base_channels*8, kernel_size=kernel_size_p, stride=1, nb_conv=2, norm=norm_type, activ=activ)
        self.decoder3 = PConvNormActiv(in_channels=base_channels*(8+4), out_channels=base_channels*4, kernel_size=kernel_size_p, stride=1, nb_conv=2, norm=norm_type, activ=activ)
        self.decoder2 = PConvNormActiv(in_channels=base_channels*(4+2), out_channels=base_channels*2, kernel_size=kernel_size_p, stride=1, nb_conv=2, norm=norm_type, activ=activ)
        self.decoder1 = PConvNormActiv(in_channels=base_channels*(2+1), out_channels=base_channels, kernel_size=kernel_size_p, stride=1, nb_conv=2, norm=norm_type, activ=activ)
        self.to_rgb1 = ToRGB(in_channels=base_channels+3)
        self.pc2ftransformer = PC2FTransformer(base_channels=base_channels, base_patch=base_patch, nhead=8, dropout=0.1, num_layers=2)
        

        # self.gatedconv1 = revisedgatedconv2d(in_channels=15, out_channels=base_channels, kernel_size=kernel_size_g, stride=1, padding=(kernel_size_g-1)//2, norm=norm_type, activ=activ)
        # self.gatedconv2 = revisedgatedconv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=kernel_size_g, stride=1, padding=(kernel_size_g-1)//2, norm=norm_type, activ=activ)
        # self.gatedconv3 = revisedgatedconv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=kernel_size_g, stride=1, padding=(kernel_size_g-1)//2, norm=norm_type, activ=activ)
        # self.gatedconv4 = revisedgatedconv2d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=kernel_size_g, stride=1, padding=(kernel_size_g-1)//2, norm=norm_type, activ=activ)

        # self.gatedconv5 = revisedgatedconv2d(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=kernel_size_g, stride=1, padding=kernel_size_g-1, dilation=2, norm=norm_type, activ=activ)
        # self.gatedconv6 = revisedgatedconv2d(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=kernel_size_g, stride=1, padding=kernel_size_g-1, dilation=2, norm=norm_type, activ=activ)
        # self.gatedconv7 = revisedgatedconv2d(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=kernel_size_g, stride=1, padding=kernel_size_g-1, dilation=2, norm=norm_type, activ=activ)
        # self.gatedconv8 = revisedgatedconv2d(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=kernel_size_g, stride=1, padding=kernel_size_g-1, dilation=2, norm=norm_type, activ=activ)

        # self.gatedconv9 = revisedgatedconv2d(in_channels=base_channels*2, out_channels=base_channels, kernel_size=kernel_size_g, stride=1, padding=(kernel_size_g-1)//2, norm=norm_type, activ=activ)
        # self.gatedconv10 = revisedgatedconv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=kernel_size_g, stride=1, padding=(kernel_size_g-1)//2, norm=norm_type, activ=activ)
        # self.gatedconv11 = revisedgatedconv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=kernel_size_g, stride=1, padding=(kernel_size_g-1)//2, norm=norm_type, activ=activ)
        # self.gatedconv12 = revisedgatedconv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=kernel_size_g, stride=1, padding=(kernel_size_g-1)//2, norm=None, activ=None, bias=True)
        self.gatedconv1 = GConvNormActiv(in_channels=15, out_channels=base_channels, kernel_size=kernel_size_g, stride=1, nb_conv=1, mode=None, norm=norm_type, activ=activ)
        self.gatedconv2 = GConvNormActiv(in_channels=base_channels, out_channels=base_channels, kernel_size=kernel_size_g, stride=2, nb_conv=2, mode='down', norm=norm_type, activ=activ)
        self.gatedconv3 = GConvNormActiv(in_channels=base_channels, out_channels=base_channels*2, kernel_size=kernel_size_g, stride=2, nb_conv=5, mode='down', norm=norm_type, activ=activ)
        # self.gatedconv4 = GConvNormActiv(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=kernel_size_g, stride=2, nb_conv=2, mode='down', norm=norm_type, activ=activ)
        # self.gatedconv5 = GConvNormActiv(in_channels=base_channels*8, out_channels=base_channels*8, kernel_size=kernel_size_g, stride=2, nb_conv=5, mode='down', norm=norm_type, activ=activ)

        # self.gatedconv6 = GConvNormActiv(in_channels=base_channels*8, out_channels=base_channels*8, kernel_size=kernel_size_g, stride=2, nb_conv=2, mode='up', norm=norm_type, activ=activ)
        # self.gatedconv7 = GConvNormActiv(in_channels=base_channels*8, out_channels=base_channels*4, kernel_size=kernel_size_g, stride=2, nb_conv=2, mode='up', norm=norm_type, activ=activ)
        self.gatedconv8 = GConvNormActiv(in_channels=base_channels*2, out_channels=base_channels, kernel_size=kernel_size_g, stride=2, nb_conv=2, mode='up', norm=norm_type, activ=activ)
        self.gatedconv9 = GConvNormActiv(in_channels=base_channels, out_channels=base_channels, kernel_size=kernel_size_g, stride=2, nb_conv=2, mode='up', norm=norm_type, activ=activ)
        self.gatedconv10 = GConvNormActiv(in_channels=base_channels, out_channels=base_channels, kernel_size=kernel_size_g, stride=1, nb_conv=1, mode=None, norm=norm_type, activ=activ)
        self.to_rgb2 = ToRGB(in_channels=base_channels)
        self.apply(weights_init(init_type=init_type))


    def forward(self, images, masks):
        origin_images, origin_masks = images, masks[:, 0, :, :].unsqueeze(1)                                                                                            # (B, 3, H, W), (B, 1, H, W)

        # E0, masks_E0 = F.interpolate(images, scale_factor=0.5, recompute_scale_factor=True), F.interpolate(masks, scale_factor=0.5, recompute_scale_factor=True)        # (B, 3, H//2, W//2), (B, 3, H//2, W//2) # 如果不先下采样，会慢很多
        E0, masks_E0 = self.encoder0(images, masks)                                                                                                                     # (B, base_channels, H, W)
        # E1, masks_E1 = self.encoder1(F.max_pool2d(E0, 2), F.max_pool2d(masks_E0, 2))                                                                                    # (B, base_channels*2, H//2, W//2)
        # E2, masks_E2 = self.encoder2(F.max_pool2d(E1, 2), F.max_pool2d(masks_E1, 2))                                                                                    # (B, base_channels*4, H//4, W//4)
        # E3, masks_E3 = self.encoder3(F.max_pool2d(E2, 2), F.max_pool2d(masks_E2, 2))                                                                                    # (B, base_channels*8, H//8, W//8)
        # E4, masks_E4 = self.encoder4(F.max_pool2d(E3, 2), F.max_pool2d(masks_E3, 2))                                                                                    # (B, base_channels*16, H//16, W//16)
        # E5, masks_E5 = self.encoder5(F.max_pool2d(E4, 2), F.max_pool2d(masks_E4, 2))                                                                                    # (B, base_channels*16, H//32, W//32)
        E1, masks_E1 = self.encoder1(E0, masks_E0)
        E2, masks_E2 = self.encoder2(E1, masks_E1)
        E3, masks_E3 = self.encoder3(E2, masks_E2)
        E4, masks_E4 = self.encoder4(E3, masks_E3)
        E5, masks_E5 = self.encoder5(E4, masks_E4)

        D4, masks_D4 = self.decoder5(torch.cat((F.interpolate(E5, scale_factor=2), E4), dim=1), torch.cat((F.interpolate(masks_E5, scale_factor=2), masks_E4), dim=1))  # (B, base_channels*16, H//16, W//16)
        D3, masks_D3 = self.decoder4(torch.cat((F.interpolate(D4, scale_factor=2), E3), dim=1), torch.cat((F.interpolate(masks_D4, scale_factor=2), masks_E3), dim=1))  # (B, base_channels*8, H//8, W//8)
        D2, masks_D2 = self.decoder3(torch.cat((F.interpolate(D3, scale_factor=2), E2), dim=1), torch.cat((F.interpolate(masks_D3, scale_factor=2), masks_E2), dim=1))  # (B, base_channels*4, H//4, W//4)
        D1, masks_D1 = self.decoder2(torch.cat((F.interpolate(D2, scale_factor=2), E1), dim=1), torch.cat((F.interpolate(masks_D2, scale_factor=2), masks_E1), dim=1))  # (B, base_channels*2, H//2, W//2)
        D0, masks_D0 = self.decoder1(torch.cat((F.interpolate(D1, scale_factor=2), E0), dim=1), torch.cat((F.interpolate(masks_D1, scale_factor=2), masks_E0), dim=1))  # (B, base_channels, H, W)   
        coarse = self.to_rgb1(torch.cat((D0, origin_images), dim=1))                                                                                                                                       # (B, 3, H, W)
        
        coarse_comp = origin_images * origin_masks + coarse * (1 - origin_masks)                                                                                        # (B, 3, H, W)
        refined = self.pc2ftransformer(D1, D2, D3, D4, coarse_comp)                                                                                                     # (B, 15, H, W)
        # print(refined.shape)
        # print(origin_masks.shape)

        refined = self.gatedconv1(refined, origin_masks)
        refined = self.gatedconv2(refined, origin_masks)
        refined = self.gatedconv3(refined, origin_masks)
        # refined = self.gatedconv4(refined, origin_masks)
        # refined = self.gatedconv5(refined, origin_masks)

        # refined = self.gatedconv6(refined, origin_masks)
        # refined = self.gatedconv7(refined, origin_masks)
        refined = self.gatedconv8(refined, origin_masks)
        refined = self.gatedconv9(refined, origin_masks)
        refined = self.gatedconv10(refined, origin_masks)
        refined = self.to_rgb2(refined)                                                                                                                                  # (B, 3, H, W)

        return refined, coarse # refined, coarse



if __name__ == '__main__':
    size = 256
    model = Generator(base_channels=32, base_patch=16, activ='swish', norm_type='instance', init_type='xavier')
    images = torch.randn(2, 3, size, size)
    # masks = torch.randint_like(images, high=2)
    # print(images.shape)
    masks = torch.randint(low=0, high=2, size=(2, 1, 256, 256)).expand(size=(2, 3, 256, 256)).float()
    # print(images)
    # print(masks)
    refined, coarse = model(images, masks)
    # print(model)
    print(refined.shape)
    print(coarse.shape)
    # images = diffuse(images, masks, clip_num=10)
    # print(images.shape)