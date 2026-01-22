import os
import sys
import pickle
import torch.nn.functional as F

clip_b_out_path="visualizations/saved_patch_scores/Clip_inference_PatchClassifier3D2Conv_10ep_All_layer_CE_avg/Celeb-DF-v1.pkl"
# clip_L_out_path="visualizations/saved_patch_scores/Clip_L_inference_PatchClassifier3D2Conv_10ep_All_layer_CE_avg/Celeb-DF-v1.pkl"
clip_L_out_path="visualizations/saved_patch_scores/Clip_inference_PatchClassifier3D2Conv_10ep_All_layer_CE_avg_patch_consistency_loss_Celeb-DF-v1_CKPT/Celeb-DF-v1.pkl"

# 'patch_scores': patch_scores_all,
# 'labels': label_lists,
# 'predictions': prediction_lists

clip_b_outs = pickle.load(open(clip_b_out_path, 'rb'))
clip_L_outs = pickle.load(open(clip_L_out_path, 'rb'))

clip_b_patch_scores=clip_b_outs['patch_scores']
clip_L_patch_scores=clip_L_outs['patch_scores']

clip_b_labels=clip_b_outs['labels']
clip_L_labels=clip_L_outs['labels']

assert clip_b_labels==clip_L_labels

# breakpoint()

clip_b_preds=clip_b_outs['predictions']
clip_L_preds=clip_L_outs['predictions']

import torch
import matplotlib.pyplot as plt

# clip_base_scores, clip_large_scores: [87, 32, 196]

base = clip_b_patch_scores.detach().cpu().flatten()
large = clip_L_patch_scores.detach().cpu().flatten()

V, T, _ = clip_b_patch_scores.shape

# reshape to grids
base_map  = clip_b_patch_scores.view(V, T, 14, 14)

if clip_L_patch_scores.shape[2]==196:
    large_map = clip_L_patch_scores.view(V, T, 14, 14)
    large_down = large_map
else:
    large_map = clip_L_patch_scores.view(V, T, 16, 16)
    large_down = F.interpolate(
        large_map.reshape(V*T, 1, 16, 16),
        size=(14, 14),
        mode="bilinear",
        align_corners=False
    ).reshape(V, T, 14, 14)
# downsample large → 14x14



# flatten again
base_flat  = base_map.flatten()
large_flat = large_down.flatten()

# base_flat_mask=base_flat>2.5
# large_flat_mask=large_flat>5

# base_flat=base_flat[base_flat_mask & large_flat_mask]
# large_flat=large_flat[base_flat_mask & large_flat_mask]

# correlation
corr = torch.corrcoef(torch.stack([base_flat, large_flat]))[0, 1]
print(f"Patch-wise correlation (aligned): {corr.item():.4f}")

# plot
plt.figure()
plt.scatter(base_flat.cpu().numpy(),
            large_flat.cpu().numpy(),
            s=1, alpha=0.3)
plt.xlabel("CLIP-Base patch scores (14×14)")
plt.ylabel("CLIP-Large patch scores (downsampled)")
plt.title(f"Patch correlation after alignment (r={corr.item():.4f})")
plt.savefig("visualizations/clip_base_vs_large_patch_scores_scatter.png", dpi=300)

corrs = []
for v in range(V):
    for t in range(T):
        b = base_map[v, t].flatten()
        l = large_down[v, t].flatten()
        corrs.append(torch.corrcoef(torch.stack([b, l]))[0, 1])

corrs = torch.stack(corrs)

print("Mean:", corrs.mean().item())
print("Std :", corrs.std().item())