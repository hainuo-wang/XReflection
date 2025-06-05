import lightning as L
import torch
import os
from os import path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

from xreflection.losses.vgg import Vgg19
from xreflection.utils.registry import MODEL_REGISTRY
from xreflection.models.base_model import BaseModel


@MODEL_REGISTRY.register()
class DSRNetModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # Losses (initialized in setup)
        self.cri_pix = None
        self.cri_rec = None
        self.cri_perceptual = None
        self.cri_exc = None
        self.vgg = Vgg19().eval()
        for param in self.vgg.parameters():
            param.requires_grad = False  # 冻结参数

    def setup_losses(self):
        """Setup loss functions"""
        from xreflection.losses import build_loss
        if not hasattr(self, 'cri_pix') or self.cri_pix is None:
            if self.opt['train'].get('pixel_opt'):
                self.cri_pix = build_loss(self.opt['train']['pixel_opt'])

        if not hasattr(self, 'cri_rec') or self.cri_rec is None:
            if self.opt['train'].get('recon_opt'):
                self.cri_rec = build_loss(self.opt['train']['recon_opt'])

        if not hasattr(self, 'cri_perceptual') or self.cri_perceptual is None:
            if self.opt['train'].get('perceptual_opt'):
                self.cri_perceptual = build_loss(self.opt['train']['perceptual_opt'])

        if not hasattr(self, 'cri_exc') or self.cri_exc is None:
            if self.opt['train'].get('exclusion_opt'):
                self.cri_exc = build_loss(self.opt['train']['exclusion_opt'])

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch (dict): Input batch containing 'input', 'target_t', 'target_r'.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Total loss.
        """
        # Get inputs
        inp = batch['input']
        target_t = batch['target_t']
        target_r = batch['target_r']

        # Forward pass
        output_t, output_r, output_rr = self.net_g(inp, self.vgg(inp))

        # Calculate losses
        loss_dict = OrderedDict()

        # Pixel loss
        l_g_pix_t = self.cri_pix(output_t, target_t)
        l_g_pix_r = self.cri_pix(output_r, target_r)

        # Reconstruction loss
        l_g_rec = self.cri_rec(output_t + output_r + output_rr, inp)

        # Perceptual loss
        l_g_percep_t, _ = self.cri_perceptual(output_t, target_t)

        # Exclusion loss
        l_g_exc = self.cri_exc(output_t, output_r)

        # Total loss
        loss_dict['l_g_pix_t'] = l_g_pix_t
        loss_dict['l_g_pix_r'] = l_g_pix_r
        loss_dict['l_g_rec'] = l_g_rec
        loss_dict['l_g_percep_t'] = l_g_percep_t
        loss_dict['l_g_exc'] = l_g_exc

        l_g_total = l_g_pix_t + l_g_pix_r + l_g_percep_t + l_g_rec + l_g_exc

        # Log losses
        for name, value in loss_dict.items():
            self.log(f'train/{name}', value, prog_bar=True, sync_dist=True)

        # Store outputs for visualization
        self.last_inp = inp
        self.last_output_clean = output_t
        self.last_output_reflection = output_r
        self.last_target_t = target_t

        return l_g_total

    def configure_optimizer_params(self):
        """Configure optimizer parameters.

        Returns:
            list: List of parameter groups.
        """
        train_opt = self.opt['train']

        # Setup different parameter groups with their learning rates
        params_lr = [
            {'params': self.net_g.get_baseball_params(), 'lr': train_opt['optim_g']['baseball_lr']},
            {'params': self.net_g.get_other_params(), 'lr': train_opt['optim_g']['other_lr']},
        ]

        # Get optimizer configuration without modifying original config
        optim_type = train_opt['optim_g']['type']
        optim_config = {k: v for k, v in train_opt['optim_g'].items()
                        if k not in ['type', 'baseball_lr', 'other_lr']}

        return {
            'optim_type': optim_type,
            'params': params_lr,
            **optim_config,
        }

    def calculate_weighted_loss(self, loss_list):
        """Calculate weighted loss.
        This file gives a default implementation of calculating multi-scale weighted loss.
        Users can implement their own weighted loss function in the model file.

        Args:
            loss_list (list): List of losses at different scales.
        """

        weights = [i / len(loss_list) for i in range(1, len(loss_list) + 1)]

        while len(weights) < len(loss_list):
            weights.append(1.0)
        weights = weights[:len(loss_list)]
        return sum(w * loss for w, loss in zip(weights, loss_list))

    def testing(self, inp):
        if self.use_ema:
            model = self.ema_model
        else:
            model = self.net_g
        with torch.no_grad():
            t, r, rr = model(inp, self.vgg(inp))
            self.output = [t, r]
