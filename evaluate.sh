python evaluate/evaluate.py \
--result_path ./inference_results/parisstreetview_epoch31_mask_ratio0.3 \
--gt_path /home/data/wangzeyu/Image_Inpainting/parisstreetview_256/paris_eval_gt/ \

python -m pytorch_fid \
./inference_results/parisstreetview_epoch31_mask_ratio0.3 \
/home/data/wangzeyu/Image_Inpainting/parisstreetview_256/paris_eval_gt/
