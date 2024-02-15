CUDA_VISIBLE_DEVICES="0,1,2" \
nohup python -m torch.distributed.launch --nproc_per_node 3 --master_port 66667 train.py \
--dataset parisstreetview \
--batch_size 16 \
--num_epochs 1000 \
--num_workers 8 \
--lr_g 1e-5 \
--lr_d 1e-5 \
--load_size 256 \
--crop_size 256 \
--mask_type self_load \
--train_image_root /home/data/wangzeyu/Image_Inpainting/parisstreetview_256/paris_train_original \
--mask_root /home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/testing_mask_dataset/ \
--pretrained_model model_latest_parisstreetview.pth \
--train_mode g,d \
--weight_adv 1e-1 \
--seed 0 \
>> parisstreetview.out & 

# batch_size 16 is the best for size 256 (no_amp) 
# batch_size 4 is the best for size 512 (no_amp)
# batch_size 8 is the best for size 512 (with amp)
# 先用256预训练再在512上微调

# """
# 88服务器：
# imagenet路径：/home/data/wangzeyu/Imagenet/ILSVRC2012/train /home/data/wangzeyu/Imagenet/ILSVRC2012/val (1280000+)
# celebA-HQ路径：/home/data/wangzeyu/Image_Inpainting/celebahq_512/train  /home/data/wangzeyu/Image_Inpainting/celebahq_512/test（30000）
# ffhq路径：/home/data/wangzeyu/Image_Inpainting/ffhq_512/train  /home/data/wangzeyu/Image_Inpainting/ffhq_512/test（70000）
# palces2路径：/home/data/wangzeyu/Image_Inpainting/Places2_512/train /home/data/wangzeyu/Image_Inpainting/Places2_512/val /home/data/wangzeyu/Image_Inpainting/Places2_512/test
# parisstreetview路径(256)：/home/data/wangzeyu/Image_Inpainting/parisstreetview_256/paris_train_original "/home/data/wangzeyu/Image_Inpainting/parisstreetview_256/paris_eval_gt/"
# partial_conv mask路径："/home/data/wangzeyu/Image_Inpainting/mask_pconv/irregular_mask/disocclusion_img_mask/" "/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/testing_mask_dataset/" 一般用后面的test
# "/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.0-0.1/"
# "/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.1-0.2/"
# "/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.2-0.3/"
# "/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.3-0.4/"
# "/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.4-0.5/"
# "/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.5-0.6/"
# """

