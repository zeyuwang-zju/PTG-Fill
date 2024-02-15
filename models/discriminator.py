import os, sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from models.weights_init import weights_init
# from models.weights_init import weights_init


# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator"""

#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='batch', init_type='xavier'):
#         """Construct a PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#         norm_layer = nn.BatchNorm2d if norm_type == 'batch' else nn.InstanceNorm2d
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func != nn.BatchNorm2d
#         else:
#             use_bias = norm_layer != nn.BatchNorm2d

#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence.extend([
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2)
#             ])

#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence.extend([
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2)
#         ])

#         sequence.extend([nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)])  # output 1 channel prediction map
#         self.model = nn.Sequential(*sequence)
#         # self.apply(weights_init(init_type=init_type))

#     def forward(self, input):
#         """Standard forward."""
#         # print(input.shape)
#         return torch.sigmoid(self.model(input))


# class Conv2dLayer(nn.Module):
#     """
#     Define a 2D Convolution Layer with possibility to add spectral Normalization.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
#                  padding_mode='zeros', activation='relu', batch_norm=True, sn=True, power_iter=1):
#         """
#         Build a Gated 2D convolutional layer as a pytorch nn.Module.
#         ----------
#         INPUT
#             |---- in_channels (int) the number of input channels of the convolutions.
#             |---- out_channels (int) the number of output channels of the convolutions.
#             |---- kernel_size (int) the kernel size of the convolutions.
#             |---- stride (int) the stride of the convolution kernel.
#             |---- padding (int) the padding of the input prior to the convolution.
#             |---- dilation (int) the dilation of the kernel.
#             |---- bias (bool) wether to use a bias term on the convolution kernel.
#             |---- padding_mode (str) how to pad the image (see nn.Conv2d doc for details).
#             |---- activation (str) the activation function to use. Supported: 'relu' -> ReLU, 'lrelu' -> LeakyReLU,
#             |               'prelu' -> PReLU, 'selu' -> SELU, 'tanh' -> Hyperbolic tangent, 'sigmoid' -> sigmoid,
#             |               'none' -> No activation used
#             |---- batch_norm (bool) whether to use a batch normalization layer between the convolution and the activation.
#             |---- sn (bool) whether to use Spetral Normalization on the convolutional weights.
#             |---- power_iter (int) the number of iteration for Spectral norm estimation.
#         OUTPUT
#             |---- Conv2dLayer (nn.Module) the convolution layer.
#         """
#         super(Conv2dLayer, self).__init__()
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'lrelu':
#             self.activation = nn.LeakyReLU(0.2)
#         elif activation == 'prelu':
#             self.activation = nn.PReLU()
#         elif activation == 'selu':
#             self.activation = nn.SELU()
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation == 'sigmoid':
#             self.activation = nn.Sigmoid()
#         elif activation == 'none':
#             self.activation = None
#         else:
#             assert 0, f"Unsupported activation: {activation}"
#         if sn:
#             self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
#                                                          dilation=dilation, bias=bias, padding_mode=padding_mode),
#                                                n_power_iterations=power_iter)
#         else:
#             self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
#                                   bias=bias, padding_mode=padding_mode)

#         self.norm = nn.BatchNorm2d(out_channels) if batch_norm else None

#     def forward(self, x):
#         """
#         Forward pass of the GatedConvolution Layer.
#         ----------
#         INPUT
#             |---- x (torch.tensor) input with dimension (Batch x in_Channel x H x W).
#         OUTPUT
#             |---- out (torch.tensor) the output with dimension (Batch x out_Channel x H' x W').
#         """
#         # Conv->(BN)->Activation
#         out = self.conv(x)
#         if self.norm:
#             out = self.norm(out)
#         if self.activation:
#             out = self.activation(out)
#         return out


# class PatchDiscriminator(nn.Module):
#     """
#     Define a Patch Discriminator proposed in Yu et al. (Free-Form Image Inpainting with Gated Convolution) for image
#     inpainting. with a possible Self-attention layer from Zhang et al 2018.
#     """
#     def __init__(self, in_channels=2, out_channels=[64, 128, 256, 256, 256, 256], kernel_size=5, stride=2, bias=True,
#                  activation='relu', norm=True, padding_mode='zeros', sn=True):
#         """
#         Build a PatchDiscriminator.
#         ----------
#         INPUT
#             |---- in_channels (int) the number of channels of input. (image channel + mask channel (=1))
#             |---- out_channels (list of int) the number of output channels of each convolutinal layer. The lenght of this
#             |               list defines the depth of the network.
#             |---- kernel_size (int or list of int) the kernel size to use in convolutions. If a int is provided, the same
#             |               kernel size will be used for all layers. If a list is passed, it must be of the same length
#             |               as out_channels.
#             |---- stride (int or list of int) the convolution stride. If a int is passed, the same stride is applied in
#             |               each layer excpet the first one with sride=1. If a list, it must be of same size as out_channels.
#             |---- bias (bool or list of bool) whether to include a bias term on convolution. If a bool is passed, bias is
#             |               included in each layer. If a list, it must be of same size as out_channels.
#             |---- activation (str or list of string) activation function to use. If a str is passed, all layer expect the
#             |               last one are activated with the same passed activation. If a list, it must be of same size
#             |               as out_channels. (possible values : 'relu', 'lrelu', 'prelu', 'selu', 'sigmoid', 'tanh', 'none').
#             |---- norm (bool or list of bool) whether to use a Batchnorm layer in convolution. If a bool is passed all
#             |               layers have the same norm. If a list, it must be of same size as out_channels.
#             |---- padding_mode (str or list of str) how to pad the features map. If a str is passed all layers have the
#             |               same padding mode. If a list, it must be of same size as out_channels. (see pytorch Conv2d
#             |               doc for list of possible padding modes.)
#             |---- sn (bool or list of bool) whether to apply Spectral Normalization to the convolution layers. If a bool
#             |               is passed all layers have/have not the Spectral Normalization. If a list, it must be of same
#             |               size as out_channels.
#             |---- self_attention (bool) whether to add a self-attention layer before the last convolution
#         OUTPUT
#             |---- PatchDiscriminator (nn.Module) the patch discriminator.
#         """
#         super(PatchDiscriminator, self).__init__()
#         n_layer = len(out_channels)
#         in_channels = [in_channels] + out_channels[:-1]
#         if isinstance(activation, list):
#             assert len(activation) == n_layer, f"Activation provided as list but does not match the number of layers. Given {len(activation)} ; required {n_layer}"
#         else:
#             activation = [activation] * (n_layer-1) + ['none']
#         if isinstance(kernel_size, list):
#             assert len(kernel_size) == n_layer, f"Kernel sizes provided as list but does not match the number of layers. Given {len(kernel_size)} ; required {n_layer}"
#         else:
#             kernel_size = [kernel_size] * n_layer
#         if isinstance(stride, list):
#             assert len(stride) == n_layer, f"Stride provided as list but does not match the number of layers. Given {len(stride)} ; required {n_layer}"
#         else:
#             stride = [1] + [stride] * (n_layer - 1)
#         if isinstance(bias, list):
#             assert len(bias) == n_layer, f"Bias provided as list but does not match the number of layers. Given {len(bias)} ; required {n_layer}"
#         else:
#             bias = [bias] * n_layer
#         if isinstance(padding_mode, list):
#             assert len(padding_mode) == n_layer, f"Padding Mode provided as list but does not match the number of layers. Given {len(padding_mode)} ; required {n_layer}"
#         else:
#             padding_mode = [padding_mode] * n_layer
#         if isinstance(sn, list):
#             assert len(sn) == n_layer, f"Spectral Normalization provided as list but does not match the number of layers. Given {len(sn)} ; required {n_layer}"
#         else:
#             sn = [sn] * n_layer
#         if isinstance(norm, list):
#             assert len(norm) == n_layer, f"BatchNormalization provided as list but does not match the number of layers. Given {len(norm)} ; required {n_layer}"
#         else:
#             norm = [norm] * n_layer
#         padding = [(ks - 1) // 2 for ks in kernel_size]
#         # build conv_layers
#         self.layer_list = nn.ModuleList()
#         for i in range(n_layer):
#             self.layer_list.append(Conv2dLayer(in_channels[i], out_channels[i], kernel_size[i], stride=stride[i],
#                                                padding=padding[i], bias=bias[i], padding_mode=padding_mode[i],
#                                                activation=activation[i], batch_norm=norm[i], sn=sn[i]))

#     def forward(self, img, mask):
#         """
#         Forward pass of the patch discriminator.
#         ----------
#         INPUT
#             |---- img (torch.tensor) the image inpainted or not with dimension (Batch x In-Channels-1 x H x W)
#             |---- mask (torch.tensor) the mask of region inpainted. Region inpainted must be sepcified by 1 and region
#             |               kept untouched must be 0. The mask should have dimension (Batch x 1 x H x W).
#         OUTPUT
#             |---- x (torch.tensor) the output feature map with dimension (Batch x Out-Channels x H' x W').
#         """
#         # concat img and mask
#         x = torch.cat([img, mask], dim=1)
#         # CNN
#         for layer in self.layer_list:
#             x = layer(x)
#         return x


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


class SN_PatchGAN(nn.Module):
    def __init__(self, in_channels=4, out_channels=[64, 128, 256, 256, 256, 256], activation='leakyrelu', norm_type='instance', init_type='xavier'):
        super(SN_PatchGAN, self).__init__() 
        n_layers = len(out_channels)
        layer_list = []
        for i in range(n_layers):
            stride = 1 if i == 0 else 2
            layer_list.append(nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels[i], 5, stride=stride, padding=2), n_power_iterations=1))
            layer_list.append(get_nonlinearity_layer(activation_type=activation))
            layer_list.append(get_norm_layer(channels=out_channels[i], norm_type=norm_type))
            in_channels = out_channels[i]
        layer_list.append(nn.utils.spectral_norm(nn.Conv2d(in_channels, 1, 1, stride=1, padding=0), n_power_iterations=1))
        layer_list.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layer_list)
        self.apply(weights_init(init_type=init_type))

    def forward(self, image, mask):
        if mask.shape[1] == 3:
            mask = mask[:, 0:1, :, :]
        x = torch.cat([image, mask], dim=1)
        x = self.layers(x)

        return x


if __name__ == '__main__':
    import torch
    input = torch.randn(size=(2, 3, 512, 512))
    mask =torch.randn(size=(2, 1, 512, 512))
    # discriminator = NLayerDiscriminator(input_nc=3)
    # print(discriminator)
    # output = discriminator(input)
    # print(output.size())
    # print(discriminator)
    # print(output)
    discriminator = SN_PatchGAN(in_channels=4)
    output = discriminator(input, mask)
    print(discriminator)
    print(output.size())
