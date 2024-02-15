python inference.py \
--device cuda:1 \
--dataset zhangshichao \
--load_size 256 \
--crop_size 256 \
--test_image_root /home/wangzeyu/Desktop/Image_Inpainting/My_frame_20220326/others/zhangshichao/ \
--mask_type self_load \
--mask_ratio_test 0.4 \
--mask_root /home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/0.4-0.5/ \
--pretrained_model model_latest_celebahq_k5.pth \
--save_masked True \

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