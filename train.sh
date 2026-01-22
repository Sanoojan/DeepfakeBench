

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29517 training/train.py \
#     --detector_path training/config/detector/clip.yaml \
#     --test_name Clip_benchmark \
#     --ddp > Terminal_outputs/my_output_clip.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29517 training/train.py \
# --detector_path training/config/detector/xception.yaml \
# --test_name Xception_benchmark_proper \
# --ddp > Terminal_outputs/my_output_xception.log 2>&1 &


# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29518 training/train.py \
#     --detector_path training/config/detector/clip.yaml \
#     --test_name Clip_benchmark_linear_classifier_only \
#     --ddp > Terminal_outputs/my_output_clip_linear_classifier_only.log 2>&1 &


CUDA_VISIBLE_DEVICES=7 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29507 training/train.py \
    --detector_path training/config/detector/clip.yaml \
    --test_name Clip_L_benchmark_PatchClS_3DConv_TopKFrame_0.1_0.1_Patch_frame_MIL_ranking_all_pairs_tmp_frame_only_var_margin_f16 \
    --ddp > Terminal_outputs/my_output_clip_L_PatchClS_3DConv_TopKFrame_0.1_0.1_Patch_frame_MIL_ranking_all_pairs_tmp_frame_only_var_margin_f16.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29496 training/train.py \
#     --detector_path training/config/detector/clip.yaml \
#     --test_name Debug \
#     --ddp 

# CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29301 training/train.py \
#     --detector_path training/config/detector/dino.yaml \
#     --test_name Dino_benchmark_linear \
#     --ddp > Terminal_outputs/my_output_dino_linear.log 2>&1 &