# PTG-Fill
Source code for PTG-Fill (Bridging Partial-Gated Convolution with Transformer for Smooth-Variation Image Inpainting).

# Abstract
Deep learning has brought essential improvement to image inpainting technology. Conventional deep-learning methods primarily focus on creating visually appealing content in the missing parts of images. However, these methods usually generate edge variations and blurry structures in the filled images, which lead to imbalances in quantitative metrics PSNR/SSIM and LPIPS/FID. In this work, we introduce a pioneering model called PTG-Fill, which utilizes
a coarse-to-fine architecture to achieve smooth-variation image inpainting. Our approach adopts the novel Stable-Partial Convolution to construct the coarse network, which integrates a smooth mask-update process to ensure its long-term operation. Meanwhile, we propose the novel Distinctive-Gated Convolution to construct the refined network, which diminishes pixel-level variations by the distinctive attention. Additionally, we build up a novel Transformer bridger to preserve the in-depth features for image refinement and facilitate the operation of the two-stage network. Our extensive experiments demonstrate that PTG-Fill outperforms previous state-of-the-art methods both quantitatively and qualitatively under various mask ratios on four benchmark datasets: CelebA-HQ, FFHQ, Paris StreetView, and Places2.

![image](https://github.com/zeyuwang-zju/PTG-Fill/assets/112078495/70de9dfb-ddf4-407b-9484-946b5acaff48)

# Requirements
- torch=1.9.1 
- torchvision=0.9.1 
- cuda=11.1

# Pre-trained Weights
Coming soon.

# Citation
If you are interested this repo for your research, welcome to cite our paper:

```
@article{wang2024bridging,
  title={Bridging partial-gated convolution with transformer for smooth-variation image inpainting},
  author={Wang, Zeyu and Shen, Haibin and Huang, Kejie},
  journal={Multimedia Tools and Applications},
  pages={1--20},
  year={2024},
  publisher={Springer}
}
```
