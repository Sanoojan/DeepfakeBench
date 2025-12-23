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

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='clip')
class CLIPDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        if config['classifier'] == 'linear':
            self.head = nn.Linear(768, 2)
        elif config['classifier'] == 'PatchTemporalClassifier':
            self.head = PatchTemporalClassifier(D=768)
        elif config['classifier'] == 'PatchClassifier':
            self.head = PatchClassifier(D=768)
        elif config['classifier'] == 'ClsDiffClassifier':
            self.head = ClsDiffClassifier(D=768)
        elif config['classifier'] == 'PatchClassifierCNN3D':
            self.head = PatchClassifierCNN3D(D=768)
        elif config['classifier'] == 'PatchTemporalClassifierCNN3D':
            self.head = PatchTemporalClassifierCNN3D(D=768)
            
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # prepare the backbone
        _, backbone = get_clip_visual(model_name="openai/clip-vit-base-patch16")
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
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        pred_list= pred_dict.get('pred_list', None)
        prob_list= pred_dict.get('prob_list', None)
        other_loss=0
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
            elif 'All_layer_CE' in self.config['additional_losses']:
                # CE_loss_alpha= self.config['All_layer_CE']['alpha']
                CE_loss=0
                for pred_i in pred_list:
                    CE_loss += self.loss_func(pred_i, label)
                CE_loss = CE_loss / len(pred_list)
                other_loss= CE_loss
                loss_dict = {'overall': other_loss}
                return loss_dict
                # other_loss= CE_loss_alpha * CE_loss
                
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
        
        features = self.features(data_dict)
        patch_scores = None
        # get the prediction by classifier
        if return_patch_scores:
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
