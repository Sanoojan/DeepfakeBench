from sklearn import metrics
import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
import cv2
import os


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred, y_true, img_names):
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            a = parts[-2]
            b = parts[-1]

            if a not in result_dict:
                result_dict[a] = []

            result_dict[a].append(item)
        image_arr = list(result_dict.values())

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return v_auc, v_eer


    y_pred = y_pred.squeeze()
    # For UCF, where labels for different manipulations are not consistent.
    y_true[y_true >= 1] = 1
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)
    if type(img_names[0]) is not list:
        # calculate video-level auc for the frame-level methods.
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    else:
        # video-level methods
        v_auc=auc

    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

def visualize_batch(images, save_path="grid.png"):
    """
    images: tensor (B, 3, H, W) normalized by CLIP mean/std
    """

    # Move mean & std to the same device
    mean = CLIP_MEAN.to(images.device)
    std = CLIP_STD.to(images.device)

    # --- De-normalize CLIP images ---
    images_denorm = images * std + mean

    # Clamp to valid range
    images_denorm = images_denorm.clamp(0, 1)

    # Create a grid (8x4 by default for 32)
    grid = make_grid(images_denorm, nrow=8, padding=2)

    # Convert to uint8 HWC
    ndarr = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    img = Image.fromarray(ndarr)

    img.save(save_path)
    print(f"[Saved] {save_path}")
    
    
def patch_score_visualize(
    X, 
    y, 
    scores,
    preds, 
    save_dir="vis_scores", 
    fps=25,
    apply_softmax=True
):
    """
    X: video frames, shape (B, T, C, H, W) or (B, C, H, W)
    y: labels
    scores: patch scores (B, T, N)
    preds: predicted score for naming files
    """
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    #  CLIP mean/std
    # -----------------------------
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)

    # -----------------------------
    # Normalize shapes
    # -----------------------------
    if X.dim() == 4:
        X = X.unsqueeze(1)              # (B,1,C,H,W)
        scores = scores.unsqueeze(1)    # (B,1,N)

    B, T, C, H, W = X.shape
    N = scores.shape[-1]
    S = int(N ** 0.5)

    # -----------------------------
    # Optional softmax across N patches
    # -----------------------------
    if apply_softmax:
        scores = torch.softmax(scores, dim=-1)

    # -----------------------------
    # Global normalization 
    # across ALL videos, ALL frames
    # -----------------------------
    global_min = scores.min().item()
    global_max = scores.max().item()
    global_range = global_max - global_min + 1e-6

    print(f"Global score min={global_min:.4f}, max={global_max:.4f}")

    # -----------------------------
    # Generate video per batch item
    # -----------------------------
    for b in range(B):

        out_path = os.path.join(save_dir, f"{b}_{int(y[b])}_{float(preds[b]):.4f}.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H)
        )

        for t in range(T):
            frame = X[b, t]  # (C,H,W)

            # ---------------------------------------------------
            # 1. Denormalize CLIP
            # ---------------------------------------------------
            f = frame.unsqueeze(0)
            f = f * clip_std + clip_mean
            f = f.clamp(0, 1)
            img = (f[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

            # ---------------------------------------------------
            # 2. Patch score → heatmap
            # ---------------------------------------------------
            patch = scores[b, t]  # (N,)

            # GLOBAL normalization
            heat = (patch - global_min) / global_range
            heat = heat.reshape(S, S).cpu().numpy()

            # resize to full frame
            heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)

            heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

            # ---------------------------------------------------
            # 3. Overlay
            # ---------------------------------------------------
            overlay = (0.45 * heat_color + 0.55 * img).astype(np.uint8)
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        writer.release()

    print(f"Saved to: {save_dir}")