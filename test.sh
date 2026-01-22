CUDA_VISIBLE_DEVICES=0 python3 training/test.py \
--detector_path training/config/detector/clip_inference.yaml \
--test_dataset "FaceForensics++" "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" "DFDC" "UADFV" \
--weights_path logs/training/clip_2026-01-20-19-16-32/test/avg/ckpt_best.pth \
--test_name "Clip_L_inference_PatchClS_3DConv_TopKFrame_0.1_0.1_Patch_frame_MIL_ranking_all_pairs_tmp_frame_only_var_margin_f8_f32inf_ep9" > Terminal_outputs/inference/my_test_Clip_L_inference_PatchClS_3DConv_TopKFrame_0.1_0.1_Patch_frame_MIL_ranking_all_pairs_tmp_frame_only_var_margin_f8_f32inf_ep9.log 2>&1 &
# 

# python3 training/test.py \
# --detector_path training/config/detector/xclip.yaml \
# --test_dataset "DFDC" \
# --weights_path ./training/weights/ffpp_best.pth
# "FaceForensics++" "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" "DFDC" "UADFV"
# WANDB_DISABLED=true

# CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=true python3 training/test.py \
# --detector_path training/config/detector/clip_inference.yaml \
# --test_dataset "DFDCP" "Celeb-DF-v2" \
# --weights_path logs/training/clip_2026-01-18-01-29-44/test/avg/ckpt_best.pth \
# --test_name "PatchClS_3DConv_TopKFrame_0.1_Patch_frame_MIL_ranking_only_correct_10ep" \
# --save_patch_scores \
# --visualize_patchwise
# > Terminal_outputs/inference/my_test_output_clip_Original_out.log 2>&1 &



