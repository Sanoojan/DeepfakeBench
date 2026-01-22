'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the CLIPDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from transformers import AutoProcessor
from networks.models.clip.clip import CLIPVisionModel,CLIPModel
import loralib as lora
import copy
from networks.classifiers import *
import wandb
logger = logging.getLogger(__name__)

import math

from training.networks.loss_helpers import *

@DETECTOR.register_module(module_name='clip')
class CLIPDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        D= config.get('dim', 768)
        topk_percent=config.get('topk_percent',0.1)
        spatio_temporal_patches= config.get('spatio_temporal_patches', False)
        # breakpoint()
        self.backbone = self.build_backbone(config)
        if config['classifier'] == 'linear':
            self.head = nn.Linear(D, 2)
        elif config['classifier'] == 'PatchTemporalClassifier':
            self.head = PatchTemporalClassifier(D=D)
        elif config['classifier'] == 'PatchClassifier':
            self.head = PatchClassifier(D=D)
        elif config['classifier'] == 'ClsDiffClassifier':
            self.head = ClsDiffClassifier(D=D)
        elif config['classifier'] == 'PatchClassifierCNN3D':
            self.head = PatchClassifierCNN3D(D=D)
        elif config['classifier'] == 'PatchClassifierCNN3D_ATTN':
            self.head = PatchClassifierCNN3D_ATTN(D=D)
        elif config['classifier'] == 'PatchClassifier3DConv_TopK':
            self.head = PatchClassifier3DConv_TopK(D=D)
        elif config['classifier'] == 'PatchClS_3DConv_TopKFrame':
            self.head = PatchClS_3DConv_TopKFrame(D=D, topk_percent=topk_percent, spatio_temporal_patches=spatio_temporal_patches)
        elif config['classifier'] == 'PatchClassifier3D2Conv':
            self.head = PatchClassifier3D2Conv(D=D)
        elif config['classifier'] == 'PatchTemporalClassifierCNN3D':
            self.head = PatchTemporalClassifierCNN3D(D=D)
            
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # prepare the backbone
        clip_backbone= config.get('clip_backbone', 'openai/clip-vit-base-patch16')
        _, backbone = get_clip_visual(model_name=clip_backbone)
        return backbone

        
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        # feat = self.backbone(data_dict['image'])['pooler_output']
        feat = self.backbone(data_dict['image'],output_hidden_states=True)
        return feat

    def classifier(self, features: torch.tensor,return_patch_scores=False) -> torch.tensor:
        if self.config.get('additional_losses', None):

            T = self.config['clip_size']
            B= features.hidden_states[1].shape[0] // T
            apply_layernorm = self.config.get('apply_layernorm', False)
            hidden_states = features.hidden_states
            if self.config.get('apply_layernorm', False):
                
                hidden_states = [hidden_states[0]] + [self.backbone.post_layernorm(h)for h in hidden_states[1:]]
            return [self.head(hidden_states[i][:,1:,:].view(B, T, -1, features.last_hidden_state.shape[-1]), return_patch_scores=return_patch_scores) for i in range(1, len(features.hidden_states))]
        elif self.config['classifier'] == 'linear' and self.config.get('Selected_layer_CE', None):
                
                layer_idx = self.config['Selected_layer_CE']['layer']
                return self.head(self.backbone.post_layernorm(features.hidden_states[layer_idx][:,0,:]))
        elif self.config['classifier'] == 'linear' :
            return self.head(features.pooler_output)
        elif self.config['classifier'] == 'ClsDiffClassifier':
            T= self.config['clip_size']
            B= features.pooler_output.shape[0] // T
            cls_features = features.pooler_output.view(B, T, -1)  # get the CLS token
            return self.head(cls_features, return_patch_scores=return_patch_scores)
        else:
            T = self.config['clip_size']
            B= features.last_hidden_state.shape[0] // T
            patch_feratures = features.last_hidden_state[:, 1:, :]  # exclude the CLS token
            patch_feratures = patch_feratures.view(B, T, -1, features.last_hidden_state.shape[-1])
            return self.head(patch_feratures, return_patch_scores=return_patch_scores)
        return self.head(features.pooler_output)
    
    def get_losses(self, data_dict: dict, pred_dict: dict,epoch=None) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        pred_list= pred_dict.get('pred_list', None)
        prob_list= pred_dict.get('prob_list', None)
        other_loss=0
        total_epochs= self.config.get('nEpochs', 15)
        # assert self.config.get('additional_losses', None)
        # assert 'Patch_frame_MIL_ranking' in self.config['additional_losses']
        # print('epoch:', epoch)
        assert epoch is not None
        if epoch is not None and self.config.get('additional_losses', None):
            if 'Patch_frame_MIL_ranking' in self.config['additional_losses']:
                cfg = self.config['Patch_frame_MIL_ranking']
                if cfg.get('use_variable_margin', False):
                    max_margin= cfg.get('max_margin', 0.9)
                    min_margin= cfg.get('min_margin', 0.1)
                    Variable_margin= min(max_margin, min_margin + epoch / total_epochs)
                else:
                    Variable_margin= cfg.get('margin', 0.3)
            elif 'Patch_frame_MIL_ranking_all' in self.config['additional_losses']:
                cfg = self.config['Patch_frame_MIL_ranking_all']
                if cfg.get('use_variable_margin', False):
                    max_margin= cfg.get('max_margin', 0.9)
                    min_margin= cfg.get('min_margin', 0.1)
                    Variable_margin= min(max_margin, min_margin + epoch / total_epochs)
                else:
                    Variable_margin= cfg.get('margin', 0.3)
        if self.config.get('additional_losses', None):
            if 'KL_loss' in self.config['additional_losses']:
                KL_loss_alpha=self.config['KL_loss']['alpha']
                KL_loss_temperature=self.config['KL_loss']['temperature']
                # compute the KL loss between different predictions
                random_other_layer = np.random.randint(0, len(pred_list)-1)
                pred_random = pred_list[random_other_layer]
                pred_main = pred_list[-1]
                kl_loss_func = nn.KLDivLoss(reduction='batchmean')
                log_probs = F.log_softmax(pred_random / KL_loss_temperature, dim=1)
                targets = F.softmax(pred_main / KL_loss_temperature, dim=1)
                KL_loss = kl_loss_func(log_probs, targets) * (KL_loss_temperature ** 2)
                other_loss= KL_loss_alpha * KL_loss
                loss = self.loss_func(pred, label)
                loss = loss + other_loss
                loss_dict = {'overall': loss}
            elif 'All_layer_CE' in self.config['additional_losses']:
                if self.config['All_layer_CE'].get('use_ranking_loss', False):
                    mean_pred = torch.mean(torch.stack(pred_list), dim=0)
                    fake_scores= mean_pred[label==1,1]
                    real_scores= mean_pred[label==0,1]
                    if len(fake_scores) == 0 or len(real_scores) == 0:
                        ranking_loss= torch.tensor(0.0, device=mean_pred.device)
                        other_loss+= ranking_loss
                    else:
                        margin= self.config['All_layer_CE']['ranking_margin']
                        # ranking_loss = F.relu(margin - (fake_scores.mean()- real_scores.mean()))
                        std_fake = fake_scores.std()
                        std_real = real_scores.std()
                        # ranking_loss = std_fake**2 + std_real**2 + (F.relu(margin - (fake_scores.mean()- real_scores.mean())))**2
                        other_loss+= ranking_loss
                    
                    wandb.log({'All_layer_CE_ranking_loss': ranking_loss.item()})
                    CE_loss=self.loss_func(mean_pred, label)
                    other_loss+= CE_loss*self.config['All_layer_CE']['CE_alpha']
                    wandb.log({'All_layer_CE_loss': CE_loss.item()})
                    loss_dict = {'overall': other_loss}
                    
                elif self.config['All_layer_CE'].get('use_mean_pred', False):
                    CE_loss=0
                    mean_pred = torch.mean(torch.stack(pred_list), dim=0)
                    
                    CE_loss += self.loss_func(mean_pred, label)
                    
                    other_loss+= CE_loss*self.config['All_layer_CE']['CE_alpha']
                    wandb.log({'All_layer_CE_mean_prob_loss': CE_loss.item()})
                    loss_dict = {'overall': other_loss}
                   
                else:
                    # CE_loss_alpha= self.config['All_layer_CE']['alpha']
                    CE_loss=0
                    for pred_i in pred_list:
                        CE_loss += self.loss_func(pred_i, label)
                    CE_loss = CE_loss / len(pred_list)
                    other_loss+= CE_loss*self.config['All_layer_CE']['CE_alpha']
                    wandb.log({'All_layer_CE_loss': CE_loss.item()})
                    loss_dict = {'overall': other_loss}
            
            if 'Patch_frame_MIL_ranking' in self.config['additional_losses']:
                cfg = self.config['Patch_frame_MIL_ranking']
                Other_Patch_loss=Patch_frame_MIL_ranking_loss(pred_dict['patch_scores'], label, cfg=cfg,margin=Variable_margin)
                
                wandb.log({'Other_Patch_loss': Other_Patch_loss.item()})
                other_loss += Other_Patch_loss
                loss_dict = {'overall': other_loss}
            if 'Patch_frame_MIL_ranking_all' in self.config['additional_losses']:
                cfg = self.config['Patch_frame_MIL_ranking_all']
                Other_Patch_loss=Patch_frame_MIL_ranking_all_loss(pred_dict['patch_scores'], label, cfg=cfg,margin=Variable_margin)
                
                wandb.log({'Other_Patch_loss': Other_Patch_loss.item()})
                other_loss += Other_Patch_loss
                loss_dict = {'overall': other_loss}
            
            if 'Patch_frame_CE' in self.config['additional_losses']:
                cfg = self.config['Patch_frame_CE']
                Other_Patch_loss=Patch_frame_CE_loss(pred_dict['patch_scores'], label, cfg=cfg, loss_function= F.binary_cross_entropy_with_logits)
                
                wandb.log({'Other_Patch_loss': Other_Patch_loss.item()})
                other_loss += Other_Patch_loss
                loss_dict = {'overall': other_loss}
                
            if 'Patch_MIL' in self.config['additional_losses']:
                MIL_loss_alpha = self.config['Patch_MIL']['alpha']
                MIL_loss_type  = self.config['Patch_MIL'].get('mil_type', 'topk')
                MIL_loss_k     = self.config['Patch_MIL'].get('k', 1)

                patch_scores = pred_dict['patch_scores']   # [B, T, N], logits
                label = data_dict['label']                 # [B]

                B, T, N = patch_scores.shape
                patch_scores_flat = patch_scores.view(B, -1)  # [B, T*N]. 
                mil_loss = 0.0
                num_terms = 0

                # -------- FAKE videos (at least K fake patches) --------
                fake_mask = (label == 1)
                if fake_mask.any():
                    fake_scores = patch_scores[fake_mask]  # [Bf, T,N]
                    # patch_scores_flat will enforce k patches o be fake in the whole video # [Bf, T*N]

                    # Top-K most fake-looking patches
                    topk_vals, _ = torch.topk(
                        fake_scores, 
                        k=min(MIL_loss_k, fake_scores.shape[1]), 
                        dim=1
                    )

                    # Encourage top-K patches to be fake (label = 1)
                    fake_targets = torch.ones_like(topk_vals)
                    mil_loss_fake = F.binary_cross_entropy_with_logits(
                        topk_vals, fake_targets
                    )

                    mil_loss += mil_loss_fake
                    num_terms += 1

                # -------- REAL videos (all patches should be real) --------
                real_mask = (label == 0)
                if real_mask.any():
                    real_scores = patch_scores_flat[real_mask]  # [Br, T*N]

                    # Encourage all patches to be real (label = 0)
                    real_targets = torch.zeros_like(real_scores)
                    mil_loss_real = F.binary_cross_entropy_with_logits(
                        real_scores, real_targets
                    )

                    mil_loss += mil_loss_real
                    num_terms += 1

                if num_terms > 0:
                    mil_loss = mil_loss / num_terms
                    other_loss = other_loss + MIL_loss_alpha * mil_loss
                    wandb.log({'MIL_loss': MIL_loss_alpha * mil_loss.item()})
                    loss_dict = {'overall': other_loss}  
            if "Patch_Xray" in self.config["additional_losses"]:
                cfg = self.config["Patch_Xray"]

                Other_Patch_loss = patch_xray_loss(
                    patch_scores=pred_dict["patch_scores"],
                    label=data_dict["label"],
                    mask=data_dict["mask"],
                    cfg=cfg,
                    extract_boundary_fn=extract_boundary,
                    boundary_to_patch_weights_fn=boundary_to_patch_weights,
                )
                wandb.log({'Other_Patch_loss': Other_Patch_loss.item()})
                other_loss += Other_Patch_loss
                loss_dict = {'overall': other_loss}
            if 'Patch_Xray_CE' in self.config['additional_losses']:
                cfg = self.config['Patch_Xray_CE']
                alpha = cfg['alpha']
                xray_importance = cfg.get('xray_importance', 1.0)
                k = cfg.get('k', 5)

                patch_scores = pred_dict['patch_scores']   # [B,T,N] logits
                label = data_dict['label']                 # [B]
                mask = data_dict['mask']                   # [B,T,H,W] or [B,T,H,W,1]

                B, T, N = patch_scores.shape
                device = patch_scores.device

                # ---- Boundary extraction ----
                boundary = extract_boundary(mask)                  # [B,T,H,W]
                patch_weights = boundary_to_patch_weights(
                    boundary, N
                )                                                   # [B,T,N]

                # Normalize per-sample (important)
                patch_weights = patch_weights / (
                    patch_weights.amax(dim=(1,2), keepdim=True) + 1e-6
                )

                patch_scores_flat = patch_scores.view(B, -1)
                patch_weights_flat = patch_weights.view(B, -1)

                xray_loss = 0.0
                num_terms = 0

                # ================= FAKE VIDEOS =================
                fake_mask = (label == 1)
                if fake_mask.any():
                    fake_scores = patch_scores_flat[fake_mask]        # [Bf, T*N]
                    fake_weights = patch_weights_flat[fake_mask]      # [Bf, T*N]

                    # ---- Consider only boundary patches ----
                    boundary_mask = fake_weights > 0.1
                    masked_scores = fake_scores.clone()
                    masked_scores[~boundary_mask] = -1e6

                    # ---- Top-K boundary patches ----
                    k_eff = min(k, boundary_mask.sum(dim=1).min().item())
                    topk_vals, _ = torch.topk(masked_scores, k=k_eff, dim=1)

                    # ---- Encourage them to be fake (soft) ----
                    fake_targets = torch.full_like(topk_vals, 0.8)
                    fake_loss = F.binary_cross_entropy_with_logits(
                        topk_vals, fake_targets
                    )

                    xray_loss += fake_loss
                    num_terms += 1

                # ================= REAL VIDEOS =================
                real_mask = (label == 0)
                if real_mask.any():
                    real_scores = patch_scores_flat[real_mask]
                    real_weights = patch_weights_flat[real_mask]

                    # Penalize fake confidence on boundary (weak)
                    real_bce = F.binary_cross_entropy_with_logits(
                        real_scores,
                        torch.zeros_like(real_scores),
                        reduction='none'
                    )

                    real_loss = (real_bce * real_weights).mean()

                    xray_loss += 0.3 * real_loss
                    num_terms += 1

                if num_terms > 0:
                    xray_loss = xray_loss / num_terms
                    other_loss += alpha * xray_loss
                    loss_dict = {'overall': other_loss}
            if 'Patch_Temporal_Xray' in self.config['additional_losses']:
                cfg = self.config['Patch_Temporal_Xray']
                alpha = cfg['alpha']
                window = cfg.get('window', 3)
                k = cfg.get('k', 5)

                patch_scores = pred_dict['patch_scores']   # [B,T,N] logits
                label = data_dict['label']                 # [B]
                mask = data_dict['mask']                   # [B,T,H,W] or [B,T,H,W,1]

                B, T, N = patch_scores.shape
                device = patch_scores.device

                # ---- Boundary weights ----
                boundary = extract_boundary(mask)                  # [B,T,H,W]
                patch_weights = boundary_to_patch_weights(
                    boundary, N
                )                                                   # [B,T,N]

                patch_weights = patch_weights / (
                    patch_weights.amax(dim=(1,2), keepdim=True) + 1e-6
                )

                temporal_loss = 0.0
                num_terms = 0

                fake_mask = (label == 1)
                if fake_mask.any():
                    ps = patch_scores[fake_mask]      # [Bf,T,N]
                    pw = patch_weights[fake_mask]     # [Bf,T,N]

                    # ---- Sliding window temporal inconsistency ----
                    temporal_inconsistency = []

                    for t in range(T - window + 1):
                        window_scores = ps[:, t:t+window, :]   # [Bf,w,N]
                        # local variance (can also use mean abs diff)
                        var = window_scores.var(dim=1)          # [Bf,N]
                        temporal_inconsistency.append(var)

                    temporal_inconsistency = torch.stack(
                        temporal_inconsistency, dim=1
                    )                                           # [Bf, T-w+1, N]

                    # ---- Weight by boundary (mean over window) ----
                    pw_window = []
                    for t in range(T - window + 1):
                        pw_window.append(pw[:, t:t+window, :].mean(dim=1))
                    pw_window = torch.stack(pw_window, dim=1)   # [Bf,T-w+1,N]

                    weighted_instability = temporal_inconsistency * pw_window

                    # ---- MIL Top-K over patches & time ----
                    flat_instability = weighted_instability.view(
                        weighted_instability.size(0), -1
                    )

                    topk_vals, _ = torch.topk(
                        flat_instability,
                        k=min(k, flat_instability.shape[1]),
                        dim=1
                    )

                    temporal_loss = topk_vals.mean()
                    num_terms += 1

                if num_terms > 0:
                    other_loss += alpha * temporal_loss
                    loss_dict = {'overall': other_loss}    
            if 'Patch_consistency_loss' in self.config['additional_losses']:
                cfg = self.config['Patch_consistency_loss']
                alpha = cfg['alpha']
                window = cfg.get('window', 3)
                use_sigmoid_weight = cfg.get('use_sigmoid_weight', False)

                use_variance = cfg.get('use_variance', False)
                variance_weight = cfg.get('variance_weight', 1.0)
                
                use_difference = cfg.get('use_difference', True)
                use_corr = cfg.get('use_corr', False)
                corr_weight = cfg.get('corr_weight', 1.0)

                patch_scores = pred_dict['patch_scores']   # [B,T,N] logits
                B, T, N = patch_scores.shape
                device = patch_scores.device

                loss_accum = 0.0
                count = 0
                
                ##--------------------------------------------------
                # 1) mean-absolute-difference-based consistency
                # -------------------------------------------------
                
                total_pc_loss = 0.0

                # --------------------------------------------------
                # 1) Difference-based consistency
                # --------------------------------------------------
                if use_difference:
                    diff_accum = 0.0
                    diff_count = 0

                    for t in range(1, T):
                        if t >= window:
                            continue

                        diff = torch.abs(patch_scores[:, t] - patch_scores[:, t - 1])

                        if use_sigmoid_weight:
                            diff *= torch.sigmoid(patch_scores[:, t])

                        diff_accum += diff.mean()
                        diff_count += 1

                    if diff_count > 0:
                        total_pc_loss += diff_accum / diff_count


                # --------------------------------------------------
                # 2) Variance-based consistency
                # --------------------------------------------------
                if use_variance and T >= window:
                    var_accum = 0.0
                    var_count = 0

                    probs = torch.sigmoid(patch_scores)

                    for t in range(window - 1, T):
                        window_scores = probs[:, t - window + 1 : t + 1]

                        mean = window_scores.mean(dim=1)
                        var  = window_scores.var(dim=1, unbiased=False)
                        var  = var / (mean.detach() + 1e-4)

                        if use_sigmoid_weight:
                            var *= window_scores[:, -1]

                        var_accum += var.mean()
                        var_count += 1

                    if var_count > 0:
                        total_pc_loss += variance_weight * (var_accum / var_count)


                # --------------------------------------------------
                # 3) Correlation-based consistency
                # --------------------------------------------------
                if use_corr:
                    corr_accum = 0.0
                    corr_count = 0

                    probs = torch.sigmoid(patch_scores)   # [B,T,N]

                    fake_patch_only = cfg.get('fake_patch_only', False)
                    fake_topk = cfg.get('fake_topk', 5)   # number of fake patches
                    fake_thresh = cfg.get('fake_thresh', None)  # optional

                    for t in range(1, T):
                        if t >= window:
                            continue

                        p_t   = probs[:, t]       # [B,N]
                        p_tm1 = probs[:, t - 1]   # [B,N]

                        if fake_patch_only:
                            # -----------------------------
                            # Select fake patches
                            # -----------------------------
                            if fake_thresh is not None:
                                # soft mask
                                mask = (p_t > fake_thresh).float()   # [B,N]
                            else:
                                # top-K fake patches
                                k = min(fake_topk, N)
                                topk_idx = torch.topk(p_t, k=k, dim=1).indices  # [B,K]

                                mask = torch.zeros_like(p_t)
                                mask.scatter_(1, topk_idx, 1.0)       # [B,N]

                            # apply mask
                            p_t_masked   = p_t * mask
                            p_tm1_masked = p_tm1 * mask

                            # skip empty masks (important early training)
                            valid = mask.sum(dim=1) > 0
                            if valid.any():
                                corr = patch_corr(
                                    p_t_masked[valid],
                                    p_tm1_masked[valid]
                                )
                                corr_accum += (1.0 - corr.mean())
                                corr_count += 1
                        else:
                            # -----------------------------
                            # All patches
                            # -----------------------------
                            corr = patch_corr(p_t, p_tm1)
                            corr_accum += (1.0 - corr.mean())
                            corr_count += 1

                    if corr_count > 0:
                        total_pc_loss += corr_weight * (corr_accum / corr_count)


                # --------------------------------------------------
                # Final aggregation
                # --------------------------------------------------
                wandb.log({'Patch_consistency_loss': total_pc_loss.item()})
                other_loss += alpha * total_pc_loss
                loss_dict = {'overall': other_loss}
                
                # --------------------------------------------------
                # Final aggregation
                # --------------------------------------------------
                # if count > 0:
                #     other_loss += alpha * (loss_accum / count)
                # else:
                #     other_loss += torch.tensor(0.0, device=device)

                # loss_dict = {'overall': other_loss}

            if 'Patch_continuity_loss' in self.config['additional_losses']:
                cfg = self.config['Patch_continuity_loss']
                alpha = cfg['alpha']
                topk = cfg.get('topk', 5)
                spatial_radius = cfg.get('spatial_radius', 1)

                patch_scores = pred_dict['patch_scores']  # [B,T,N]

                cont_loss = patch_continuity_loss(
                    patch_scores,
                    window=1,
                    topk=topk,
                    spatial_radius=spatial_radius
                )

                other_loss += alpha * cont_loss
                loss_dict = {'overall': other_loss}    
            if 'Patch_continuity_flow_loss' in self.config['additional_losses']:
                cfg = self.config['Patch_continuity_flow_loss']
                alpha = cfg['alpha']
                conf_thresh = cfg.get('conf_thresh', 0.6)

                cont_loss = patch_continuity_flow_loss(
                    pred_dict['patch_scores'],
                    data_dict['optical_flow'],    
                    conf_thresh=conf_thresh
                )

                other_loss += alpha * cont_loss
        else:  
               
            loss = self.loss_func(pred, label)
            loss = loss + other_loss
            loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False,return_patch_scores=False) -> dict:
        
        # get the features by backbone
        return_patch_scores=True # always true for patch scores
        
        features = self.features(data_dict)
        patch_scores = None
        # get the prediction by classifier
        if return_patch_scores:
            if self.config.get('additional_losses', None):
                assert self.config['classifier'] != 'linear'
                outcomes = self.classifier(features, return_patch_scores=return_patch_scores)
                preds=[]
                patch_scores_list=[]
                for outcome in outcomes:
                    pred, patch_scores = outcome
                    preds.append(pred)
                    patch_scores_list.append(patch_scores)
                pred = preds
                patch_scores = patch_scores_list
                patch_scores = torch.mean(torch.stack(patch_scores), dim=0)
                    
            else:
                pred, patch_scores = self.classifier(features, return_patch_scores=return_patch_scores)
        else:
            pred = self.classifier(features)
        pred_list=None
        prob_list=None
        # breakpoint()
        if self.config.get('additional_losses', None) :
            assert self.config['classifier'] != 'linear'
            if 'KL_loss' in self.config['additional_losses']:
                # return a list of predictions from different hidden states
                pred_list = pred
                pred = pred_list[-1]  # use the last one as the main prediction
            elif 'All_layer_CE' in self.config['additional_losses']:
                pred_list = pred
                # average the predictions from different hidden states
                pred = torch.mean(torch.stack(pred), dim=0)
                # pred = torch.mean(torch.stack(pred_list[9:23]), dim=0)
                # pred = pred_list[16] # just to check 
                prob_list = [torch.softmax(p, dim=1)[:, 1] for p in pred_list]
            elif 'Selected_layer_CE' in self.config['additional_losses']:
                
                layer_idx = self.config['Selected_layer_CE']['layer']
                pred_list = pred
                pred = pred_list[layer_idx]
                
        
        features = features.pooler_output   # use the pooled output as the feature
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        
        
        if self.config.get('video_mode', False) and not(prob.shape [0] == data_dict['label'].shape[0]):
            # reshape the output to (B, T, C)
            T = self.config['clip_size']
            B = prob.shape[0] // T
            prob = prob.view(B, T)
            pred = pred.view(B, T, -1)
            features = features.view(B, T, -1)
            
            # average the output along the temporal dimension
            # change these later
            prob = torch.mean(prob, dim=1)
            pred = torch.mean(pred, dim=1)
            features = torch.mean(features, dim=1)
        
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'patch_scores': patch_scores,'pred_list': pred_list, 'prob_list': prob_list}
        return pred_dict


def get_clip_visual(model_name = "openai/clip-vit-base-patch16"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model
