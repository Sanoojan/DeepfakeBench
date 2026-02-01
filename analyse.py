import os
import sys
import pickle
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_curve, auc

def plot_roc(probs_ce, probs_mil, labels,name="roc_curve.png"):
    """
    probs_ce : array-like, CE model fake probabilities
    probs_mil: array-like, MIL model fake probabilities
    labels   : array-like {0,1}
    """

    fpr_ce, tpr_ce, _ = roc_curve(labels, probs_ce)
    fpr_mil, tpr_mil, _ = roc_curve(labels, probs_mil)

    auc_ce = auc(fpr_ce, tpr_ce)
    auc_mil = auc(fpr_mil, tpr_mil)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_ce, tpr_ce, label=f"CE (AUC={auc_ce:.3f})")
    plt.plot(fpr_mil, tpr_mil, label=f"MIL (AUC={auc_mil:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Detection Rate")
    plt.title("ROC Curve (Cross-Dataset Evaluation)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(name+".png", dpi=300)
    plt.close()

def compute_brier_score(probs, labels):
    """
    probs: array-like, shape (N,) – predicted probability for class=1 (fake)
    labels: array-like, shape (N,) – ground truth {0,1}
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    return np.mean((probs - labels) ** 2)

def compute_tdr_at_fpr(probs, labels, target_fpr=0.1):
    """
    probs: predicted probabilities for fake
    labels: ground truth {0,1}
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)

    # find closest FPR >= target
    idx = np.where(fpr >= target_fpr)[0]
    if len(idx) == 0:
        return tpr[-1]  # fallback
    return tpr[idx[0]]

def compute_confusion_at_fpr(probs, labels, target_fpr=0.1):
    """
    probs  : predicted probabilities for fake (shape [N])
    labels : ground truth {0,1} (shape [N])
    
    Returns:
        threshold, TP, FP, TN, FN
    """

    probs = np.asarray(probs)
    labels = np.asarray(labels)

    # Separate real samples (label = 0)
    real_probs = probs[labels == 0]

    # Choose threshold so that FP / (#real) ≈ target_fpr
    # FP happens when real_probs >= threshold
    num_real = len(real_probs)
    k = int(np.ceil(target_fpr * num_real))

    if k == 0:
        threshold = 1.0
    else:
        # sort descending
        sorted_real = np.sort(real_probs)[::-1]
        threshold = sorted_real[k - 1]

    # Apply threshold
    preds = (probs >= threshold).astype(int)

    # Confusion matrix components
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    TN = np.sum((preds == 0) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    return threshold, TP, FP, TN, FN

def compute_tdr_from_confusion(TP, FN):
    return TP / (TP + FN + 1e-8)

def compute_ece(probs, labels, n_bins=15):
    """
    probs: array-like, shape (N,) – predicted probabilities for class=1 (fake)
    labels: array-like, shape (N,) – ground truth {0,1}
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(probs)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        mask = (probs > bin_lower) & (probs <= bin_upper)
        if mask.sum() == 0:
            continue

        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()

        ece += (mask.sum() / N) * np.abs(bin_acc - bin_conf)

    return ece

clip_b_out_path="visualizations/saved_patch_scores/my_output_clip_L_PatchClassifierCNN3D_use_mean_pred/DFDC.pkl"
# clip_L_out_path="visualizations/saved_patch_scores/Clip_L_inference_PatchClassifier3D2Conv_10ep_All_layer_CE_avg/Celeb-DF-v1.pkl"
clip_L_out_path="visualizations/saved_patch_scores/my_output_clip_L_PatchClS_3DConv_TopKFrame_0.1_0.1_Patch_frame_MIL_ranking_all_pairs_tmp_frame_only_var_margin_f8/DFDC.pkl"

# 'patch_scores': patch_scores_all,
# 'labels': label_lists,
# 'predictions': prediction_lists

clip_b_outs = pickle.load(open(clip_b_out_path, 'rb'))
clip_L_outs = pickle.load(open(clip_L_out_path, 'rb'))

clip_b_patch_scores=clip_b_outs['patch_scores']
clip_L_patch_scores=clip_L_outs['patch_scores']

clip_b_labels=clip_b_outs['labels']
clip_L_labels=clip_L_outs['labels']

# breakpoint()

clip_b_preds=clip_b_outs['predictions']
clip_L_preds=clip_L_outs['predictions']

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


if clip_L_patch_scores.shape[2]==196:
    base_map  = clip_b_patch_scores.view(V, T, 14, 14)
    large_map = clip_L_patch_scores.view(V, T, 14, 14)
    large_down = large_map
elif clip_L_patch_scores.shape[2]==256 and clip_b_patch_scores.shape[2]==256:
    large_map = clip_L_patch_scores.view(V, T, 16, 16)
    large_down=large_map
    base_map  = clip_b_patch_scores.view(V, T, 16, 16)
    
else:
    base_map  = clip_b_patch_scores.view(V, T, 14, 14)
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
plt.xlabel("CLIP-Base patch scores ")
plt.ylabel("CLIP-Large patch scores")
plt.title(f"Patch correlation after alignment (r={corr.item():.4f})")
plt.savefig("visualizations/CE_vs_MIL1.png", dpi=300)

corrs = []
for v in range(V):
    for t in range(T):
        b = base_map[v, t].flatten()
        l = large_down[v, t].flatten()
        corrs.append(torch.corrcoef(torch.stack([b, l]))[0, 1])

corrs = torch.stack(corrs)

print("Mean:", corrs.mean().item())
print("Std :", corrs.std().item())



import numpy as np
import matplotlib.pyplot as plt

# Convert to numpy
clip_b = np.asarray(clip_b_preds)
clip_L = np.asarray(clip_L_preds)
GT = np.asarray(clip_b_labels)

assert clip_b.shape == clip_L.shape == GT.shape

# Masks
real_mask = GT == 0
fake_mask = GT == 1

plt.figure(figsize=(6, 6))

# Real samples
plt.scatter(
    clip_b[real_mask],
    clip_L[real_mask],
    c="blue",
    alpha=0.6,
    label="Real",
    s=20
)

# Fake samples
plt.scatter(
    clip_b[fake_mask],
    clip_L[fake_mask],
    c="red",
    alpha=0.6,
    label="Fake",
    s=20
)

# Diagonal reference line
min_v = min(clip_b.min(), clip_L.min())
max_v = max(clip_b.max(), clip_L.max())
plt.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)

plt.xlabel("CE-based score")
plt.ylabel("MIL-based score")
plt.title("CE vs MIL predictions (DFDC1)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("visualizations/CE_vs_MIL_preds_DFDC1.png", dpi=300)

diff = clip_L - clip_b

print("Mean diff (real):", diff[real_mask].mean())
print("Mean diff (fake):", diff[fake_mask].mean())


ece_ce  = compute_ece(clip_b_preds, GT)
ece_mil = compute_ece(clip_L_preds, GT)

threshold, TP, FP, TN, FN = compute_confusion_at_fpr(
    probs=clip_L_preds,
    labels=GT,
    target_fpr=0.2
)

tdr = compute_tdr_from_confusion(TP, FN)

tdr_ce  = compute_tdr_from_confusion(*compute_confusion_at_fpr(
    probs=clip_b_preds,
    labels=GT,
    target_fpr=0.2
)[1:3])
tdr_mil = compute_tdr_from_confusion(TP, FN)

brier_ce  = compute_brier_score(clip_b_preds, GT)
brier_mil = compute_brier_score(clip_L_preds, GT)

print(f"Brier score (CE) : {brier_ce:.4f}")
print(f"Brier score (MIL): {brier_mil:.4f}")

print(f"ECE (CE) : {ece_ce:.4f}")
print(f"ECE (MIL): {ece_mil:.4f}")

print(f"TDR@0.2FPR (CE) : {tdr_ce:.4f}")
print(f"TDR@0.2FPR (MIL): {tdr:.4f}")

plot_roc(clip_b_preds, clip_L_preds, GT, name="visualizations/roc_curve_CE_vs_MIL_DFDC1")
