CUDA_VISIBLE_DEVICES=2 python3 training/test.py \
--detector_path training/config/detector/clip_inference.yaml \
--test_dataset "FaceForensics++" "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" "DFDC" "WDF" "DeepFakeDetection" "UADFV" \
--weights_path logs/training/clip_2026-01-30-13-17-55/test/avg/ckpt_best.pth \
--test_name "Clip_L_inference2_PatchClS_3DConv_TopKFrame_0.1_All_layer_CE_Patch_frame_MIL_ranking_all_margin_0.4_f32inf_ep8" > Terminal_outputs/inference/my_test_Clip_L_inference2_PatchClS_3DConv_TopKFrame_0.1_All_layer_CE_Patch_frame_MIL_ranking_all_margin_0.4_f32inf_ep8.log 2>&1 &
# # # 


# python3 training/test.py \
# --detector_path training/config/detector/xclip.yaml \
# --test_dataset "DFDC" \
# --weights_path ./training/weights/ffpp_best.pth
# "FaceForensics++" "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" "DFDC" "UADFV"
# WANDB_DISABLED=true

# CUDA_VISIBLE_DEVICES=7 WANDB_DISABLED=true python3 training/test.py \
# --detector_path training/config/detector/clip_inference.yaml \
# --test_dataset "DFDC" \
# --weights_path logs/training/clip_2026-01-20-19-16-32/test/avg/ckpt_best.pth \
# --test_name "my_output_clip_L_PatchClS_3DConv_TopKFrame_0.1_0.1_Patch_frame_MIL_ranking_all_pairs_tmp_frame_only_var_margin_f8" \
# --save_patch_scores \
# --visualize_patchwise




