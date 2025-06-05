import lightning as L
import torch
import os
from os import path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from xreflection.utils.registry import MODEL_REGISTRY
from xreflection.models.base_model import BaseModel


@MODEL_REGISTRY.register()
class DSITModel(BaseModel):
    """
    This file defines the training process of DSIT.

    Please refer to the paper for more details:
        
        DSIT: Single Image Reflection Separation via Interactive Dual-Stream Transformers (NeurIPS 2024).

    """

    def __init__(self, opt):
        """Initialize the DSITModel.
        
        Args:
            opt (dict): Configuration options.
        """
        super().__init__(opt)

        # Losses (initialized in setup)
        self.cri_pix = None
        self.cri_perceptual = None
        self.cri_exclu = None
        self.cri_recons = None

    def setup_losses(self):
        """Setup loss functions"""
        from xreflection.losses import build_loss

        # Pixel loss for transmission layer
        if not hasattr(self, 'cri_pix') or self.cri_pix is None:
            if self.opt['train'].get('pixel_opt'):
                self.cri_pix = build_loss(self.opt['train']['pixel_opt'])

        # Perceptual loss
        if not hasattr(self, 'cri_perceptual') or self.cri_perceptual is None:
            if self.opt['train'].get('perceptual_opt'):
                self.cri_perceptual = build_loss(self.opt['train']['perceptual_opt'])

        # Exclusion loss
        if not hasattr(self, 'cri_exclu') or self.cri_exclu is None:
            if self.opt['train'].get('exclu_opt'):
                self.cri_exclu = build_loss(self.opt['train']['exclu_opt'])

        # Reconstruction loss
        if not hasattr(self, 'cri_recons') or self.cri_recons is None:
            if self.opt['train'].get('recons_opt'):
                self.cri_recons = build_loss(self.opt['train']['recons_opt'])

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
        output_t, output_r, output_rr = self.net_g(inp)

        # Calculate losses
        loss_dict = OrderedDict()

        # Pixel losses
        l_g_pix_t = self.cri_pix(output_t, target_t)
        l_g_pix_r = self.cri_pix(output_r, target_r)

        # Perceptual loss
        l_g_percep_t = self.cri_perceptual(output_t, target_t)
        if l_g_percep_t is None:
            l_g_percep_t = torch.tensor(0.0, device=inp.device)

        # Exclusion loss
        l_g_exclu = self.cri_exclu(output_t, output_r)

        # Reconstruction loss
        l_g_recons = self.cri_recons(output_t, output_r, output_rr, inp)

        # Apply loss weights
        train_opt = self.opt['train']
        lambda_vgg = train_opt.get('lambda_vgg', 1.0)
        lambda_rec = train_opt.get('lambda_rec', 1.0)

        l_g_percep_t = l_g_percep_t * lambda_vgg
        l_g_recons = l_g_recons * lambda_rec

        # Total loss
        loss_dict['l_g_pix_t'] = l_g_pix_t
        loss_dict['l_g_pix_r'] = l_g_pix_r
        loss_dict['l_g_percep_t'] = l_g_percep_t
        loss_dict['l_g_exclu'] = l_g_exclu
        loss_dict['l_g_recons'] = l_g_recons

        l_g_total = l_g_pix_t + l_g_pix_r + l_g_percep_t + l_g_exclu + l_g_recons

        # Log losses
        for name, value in loss_dict.items():
            self.log(f'train/{name}', value, prog_bar=True, sync_dist=True)

        # Store outputs for visualization
        self.last_inp = inp
        self.last_output_clean = output_t
        self.last_output_reflection = output_r
        self.last_target_t = target_t

        return l_g_total

    def testing(self, inp):
        """Testing/inference method.
        
        Args:
            inp (torch.Tensor): Input tensor.
        """
        if self.use_ema:
            model = self.ema_model
        else:
            model = self.net_g

        with torch.no_grad():
            output_t, output_r, output_rr = model(inp)
            self.output = [output_t, output_r]

    def configure_optimizer_params(self):
        """Configure optimizer parameters.
        
        Returns:
            dict: Optimizer configuration.
        """
        train_opt = self.opt['train']

        # Get all network parameters
        params = list(self.net_g.parameters())

        # Get optimizer configuration
        optim_type = train_opt['optim_g']['type']
        optim_config = {k: v for k, v in train_opt['optim_g'].items() if k != 'type'}

        return {
            'optim_type': optim_type,
            'params': params,
            **optim_config,
        }
