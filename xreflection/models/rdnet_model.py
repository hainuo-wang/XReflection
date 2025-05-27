import lightning as L
import torch
import os
from os import path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from xreflection.utils.registry import MODEL_REGISTRY
from xreflection.models.base_model import BaseModel

@MODEL_REGISTRY.register()
class RDNetModel(BaseModel):
    """
    This file defines the training process of RDNet and RDNet+.

    Please refer to the paper for more details:
        
        Reversible Decoupling Network for Single Image Reflection Removal (CVPR 2025).

        Reversible Adaptor for Single Image Reflection Removal (Preprint).
    
    """

    def __init__(self, opt):
        """Initialize the ClsModel.
        
        Args:
            opt (dict): Configuration options.
        """
        super().__init__(opt)

        # Losses (initialized in setup)
        self.cri_pix = None
        self.cri_perceptual = None
        self.cri_grad = None

    def setup_losses(self):
        """Setup loss functions"""
        from xreflection.losses import build_loss
        if not hasattr(self, 'cri_pix') or self.cri_pix is None:
            if self.opt['train'].get('pixel_opt'):
                self.cri_pix = build_loss(self.opt['train']['pixel_opt'])

        if not hasattr(self, 'cri_perceptual') or self.cri_perceptual is None:
            if self.opt['train'].get('perceptual_opt'):
                self.cri_perceptual = build_loss(self.opt['train']['perceptual_opt'])

        if not hasattr(self, 'cri_grad') or self.cri_grad is None:
            if self.opt['train'].get('grad_opt'):
                self.cri_grad = build_loss(self.opt['train']['grad_opt'])


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
        x_cls_out, x_img_out = self.net_g(inp)
        output_clean, output_reflection = x_img_out[-1][:, :3, ...], x_img_out[-1][:, 3:, ...]

        # Calculate losses
        loss_dict = OrderedDict()
        pix_t_loss_list = []
        pix_r_loss_list = []
        per_loss_list = []
        grad_loss_list = []

        for i, out_imgs in enumerate(x_img_out):
            out_t, out_r = out_imgs[:, :3, ...], out_imgs[:, 3:, ...]
            # Pixel loss
            l_g_pix_t = self.cri_pix(out_t, target_t)
            pix_t_loss_list.append(l_g_pix_t)
            l_g_pix_r = self.cri_pix(out_r, target_r)
            pix_r_loss_list.append(l_g_pix_r)

            # Perceptual loss
            l_g_percep_t, _ = self.cri_perceptual(out_t, target_t)
            if l_g_percep_t is not None:
                per_loss_list.append(l_g_percep_t)

            # Gradient loss
            l_g_grad = self.cri_grad(out_t, target_t)
            grad_loss_list.append(l_g_grad)

        # Apply weights to losses
        l_g_pix_t = self.calculate_weighted_loss(pix_t_loss_list)
        l_g_pix_r = self.calculate_weighted_loss(pix_r_loss_list)
        l_g_percep_t = self.calculate_weighted_loss(per_loss_list)
        l_g_grad = self.calculate_weighted_loss(grad_loss_list)

        # Total loss
        loss_dict['l_g_pix_t'] = l_g_pix_t
        loss_dict['l_g_pix_r'] = l_g_pix_r
        loss_dict['l_g_percep_t'] = l_g_percep_t
        loss_dict['l_g_grad'] = l_g_grad
        l_g_total = l_g_pix_t + l_g_pix_r + l_g_percep_t + l_g_grad

        # Log losses
        for name, value in loss_dict.items():
            self.log(f'train/{name}', value, prog_bar=True, sync_dist=True)

        # Store outputs for visualization
        self.last_inp = inp
        self.last_output_clean = output_clean
        self.last_output_reflection = output_reflection
        self.last_target_t = target_t

        return l_g_total
    
    def testing(self, inp):
        if self.use_ema:
            model = self.ema_model
        else:
            model = self.net_g
        with torch.no_grad():
            x_cls_out, x_img_out = model(inp)
            output_clean, output_reflection = x_img_out[-1][:, :3, ...], x_img_out[-1][:, 3:, ...]
            self.output = [output_clean, output_reflection]

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