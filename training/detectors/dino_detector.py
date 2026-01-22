import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from networks.classifiers import *
from networks.models.dino.eva import vit_base_patch16_dinov3 as DINOv3_base_patch16

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='dino')
class DINODetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = self.build_backbone(config)
        self.feat_dim = config.get('feat_dim', 768)

        if config['classifier'] == 'linear':
            self.head = nn.Linear(self.feat_dim, 2)
        elif config['classifier'] == 'PatchTemporalClassifier':
            self.head = PatchTemporalClassifier(D=self.feat_dim)
        elif config['classifier'] == 'PatchClassifier':
            self.head = PatchClassifier(D=self.feat_dim)
        elif config['classifier'] == 'ClsDiffClassifier':
            self.head = ClsDiffClassifier(D=self.feat_dim)
        elif config['classifier'] == 'PatchClassifierCNN3D':
            self.head = PatchClassifierCNN3D(D=self.feat_dim)
        elif config['classifier'] == 'PatchTemporalClassifierCNN3D':
            self.head = PatchTemporalClassifierCNN3D(D=self.feat_dim)

        self.loss_func = self.build_loss(config)

    # ------------------------------------------------
    # Backbone
    # ------------------------------------------------
    def build_backbone(self, config):
        
        # careful here I am not using from the config
        model=DINOv3_base_patch16(pretrained=True, num_classes=0, global_pool='')
        
        # model_name = config.get('dino_model', 'vit_base_patch16_dinov3')

        # model = timm.create_model(
        #     model_name,
        #     pretrained=True,
        #     num_classes=0,     # remove classifier
        #     global_pool=''     # keep CLS + patch tokens
        # )
        
        return model

    # ------------------------------------------------
    # Loss
    # ------------------------------------------------
    def build_loss(self, config):
        return LOSSFUNC[config['loss_func']]()

    # ------------------------------------------------
    # Feature extraction
    # ------------------------------------------------
    def features(self, data_dict):
        """
        Returns tokens:
        shape = [B*T, 1 + N, D]
        """
        x = data_dict['image']
        tokens = self.backbone.forward_features(x)
        return tokens

    # ------------------------------------------------
    # Classifier
    # ------------------------------------------------
    def classifier(self, features, return_patch_scores=False):

        # timm DINOv2 → features: [BT, 1+N, D]
        cls_token = features[:, 0, :]
        patch_tokens = features[:, 1+4:, :]

        if self.config['classifier'] == 'linear':
            return self.head(cls_token)

        elif self.config['classifier'] == 'ClsDiffClassifier':
            T = self.config['clip_size']
            B = cls_token.shape[0] // T
            cls_token = cls_token.view(B, T, -1)
            return self.head(cls_token, return_patch_scores)

        else:
            T = self.config['clip_size']
            B = patch_tokens.shape[0] // T
            patch_tokens = patch_tokens.view(
                B, T, patch_tokens.shape[1], patch_tokens.shape[2]
            )
            return self.head(patch_tokens, return_patch_scores)

    # ------------------------------------------------
    # Loss computation
    # ------------------------------------------------
    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------
    def get_train_metrics(self, data_dict, pred_dict):
        auc, eer, acc, ap = calculate_metrics_for_train(
            data_dict['label'].detach(),
            pred_dict['cls'].detach()
        )
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------
    def forward(self, data_dict, inference=False, return_patch_scores=False):

        features = self.features(data_dict)

        if return_patch_scores:
            pred, patch_scores = self.classifier(features, True)
        else:
            pred = self.classifier(features)
            patch_scores = None
        
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
                # pred = pred_list[3] # just to check 
                prob_list = [torch.softmax(p, dim=1)[:, 1] for p in pred_list]
            elif 'Selected_layer_CE' in self.config['additional_losses']:
                
                layer_idx = self.config['Selected_layer_CE']['layer']
                pred_list = pred
                pred = pred_list[layer_idx]

        prob = torch.softmax(pred, dim=1)[:, 1]
        cls_feat = features[:, 0, :]

        # Video-level aggregation
        if self.config.get('video_mode', False) and \
           prob.shape[0] != data_dict['label'].shape[0]:

            T = self.config['clip_size']
            B = prob.shape[0] // T

            prob = prob.view(B, T).mean(dim=1)
            pred = pred.view(B, T, -1).mean(dim=1)
            cls_feat = cls_feat.view(B, T, -1).mean(dim=1)

        return {
            'cls': pred,
            'prob': prob,
            'feat': cls_feat,
            'patch_scores': patch_scores,
            'pred_list': None,
            'prob_list': None
        }