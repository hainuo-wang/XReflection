import lightning as L
import torch
import os
import re
from os import path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from xreflection.utils.registry import MODEL_REGISTRY
from xreflection.metrics import calculate_metric
from xreflection.utils import imwrite, tensor2img
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info, rank_zero_warn
from torchmetrics import MetricCollection


@MODEL_REGISTRY.register()
class BaseModel(L.LightningModule):
    """Classification Module for reflection removal using PyTorch Lightning.
    
    This module implements a classification-based approach for single image reflection removal.
    It supports progressive multi-scale image processing, EMA model updates,
    and extensive validation metrics.
    """

    def __init__(self, opt):
        """Initialize the ClsModel.
        
        Args:
            opt (dict): Configuration options.
        """
        from xreflection.archs import build_network
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.is_train = opt['is_train']

        # Define network
        self.net_g = build_network(opt['network_g'])

        self.current_val_metrics = {}
        self.val_dataset_names = {}
        self.top_psnr_epochs = []
        self.top_ssim_epochs = []

        # Flag to indicate if using EMA - will be set by EMACallback
        self.use_ema = False

    def setup(self, stage: Optional[str] = None):
        """Setup module based on stage.
        
        Args:
            stage (str, optional): 'fit', 'validate', 'test', or 'predict'
        """
        net_params = sum(map(lambda x: x.numel(), self.net_g.parameters()))
        rank_zero_info(f'Network: {self.net_g.__class__.__name__}, with parameters: {net_params:,d}')

        # Load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_weights(load_path)

        if stage == 'fit' or stage is None:
            self.setup_losses()

    def load_weights(self, load_path):
        """Load pretrained weights.
        
        Args:
            load_path (str): Path to the checkpoint file.
        """
        param_key = self.opt['path'].get('param_key_g', 'params')
        strict_load = self.opt['path'].get('strict_load_g', True)

        if self.trainer is None or self.trainer.global_rank == 0:
            rank_zero_info(f'Loading weights from {load_path} with param key: [{param_key}]')

        # Load weights
        checkpoint = torch.load(load_path, map_location='cpu')

        # Check available keys in checkpoint for better debugging
        if self.trainer is None or self.trainer.global_rank == 0:
            if isinstance(checkpoint, dict):
                rank_zero_info(f"Available keys in checkpoint: {list(checkpoint.keys())}")

        # Try to load with specified param_key, then fallback to alternatives
        if param_key in checkpoint:
            weights = checkpoint[param_key]
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"Successfully loaded weights using key '{param_key}'")
        elif 'params_ema' in checkpoint and param_key != 'params_ema':
            weights = checkpoint['params_ema']
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"Key '{param_key}' not found, using 'params_ema' instead")
        elif 'params' in checkpoint and param_key != 'params':
            weights = checkpoint['params']
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"Key '{param_key}' not found, using 'params' instead")
        else:
            # If no recognized keys, use the entire checkpoint
            weights = checkpoint
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"No recognized parameter keys found, using entire checkpoint")

        # Remove the prefix for torch.compile
        for k, v in list(weights.items()):
            if k.startswith('_orig_mod.'):
                weights[k[10:]] = weights.pop(k)

        # Remove unnecessary 'module.' prefix
        for k, v in list(weights.items()):
            if k.startswith('module.'):
                weights[k[7:]] = weights.pop(k)

        # Load to model
        self._print_different_keys_loading(self.net_g, weights, strict_load)
        self.net_g.load_state_dict(weights, strict=strict_load)

    @rank_zero_only
    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print key differences when loading models.
        
        Args:
            crt_net (nn.Module): Current network.
            load_net (dict): Loaded network state dict.
            strict (bool): Whether to strictly enforce parameter shapes.
        """
        # Get network state dict
        crt_net_keys = set(crt_net.state_dict().keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            rank_zero_warn('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                rank_zero_warn(f'  {v}')
            rank_zero_warn('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                rank_zero_warn(f'  {v}')

        # Check sizes of the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net.state_dict()[k].size() != load_net[k].size():
                    rank_zero_warn(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net.state_dict()[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def setup_losses(self):
        pass

    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            tuple: Classification outputs and image outputs.
        """
        return self.net_g(x)

    def training_step(self, batch, batch_idx):
        pass
    
    def on_train_epoch_end(self):
        self.trainer.train_dataloader.reset()
        
    def testing(self, inp):
        if self.use_ema:
            model = self.ema_model
        else:
            model = self.net_g
        with torch.no_grad():
            x_cls_out, x_img_out = model(inp)
            output_clean, output_reflection = x_img_out[-1][:, :3, ...], x_img_out[-1][:, 3:, ...]
            self.output = [output_clean, output_reflection]
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """验证步骤。
        
        Args:
            batch (dict): 输入批次。
            batch_idx (int): 批次索引。
            dataloader_idx (int, optional): 数据加载器索引，用于多个验证集。
            
        Returns:
            dict: 包含清晰图像和反射图像的输出字典。
        """
        # 获取当前验证数据集的名称
        dataset_name = self.val_dataset_names[dataloader_idx]
        
        # 验证批次是否包含所需字段
        required_keys = ['input']
        for key in required_keys:
            if key not in batch:
                rank_zero_warn(f"Required key '{key}' missing from batch during validation")
                return {'error': f"Missing required key: {key}"}

        # 保存输入图像信息
        inp = batch['input']
        self.testing(inp)
        output_clean, output_reflection = self.output

        # 优雅地处理缺失的inp_path
        if 'inp_path' in batch and len(batch['inp_path']) > 0:
            img_name = osp.splitext(osp.basename(batch['inp_path'][0]))[0]
        else:
            # 如果缺少inp_path，生成一个后备名称
            img_name = f"sample_{batch_idx}"
            rank_zero_warn(f"'inp_path' key missing in batch, using fallback name: {img_name}")

        # 处理图像用于指标计算和可视化
        clean_img = tensor2img([output_clean])
        reflection_img = tensor2img([output_reflection])
        target_t_img = tensor2img([batch['target_t']])

        metric_data = {'img': clean_img, 'img2': target_t_img}


        # 保存验证图像
        if self.opt['val'].get('save_img', False):
            self._save_images(clean_img, reflection_img, img_name, dataset_name)

        # 计算指标
        if 'img2' in metric_data and self.opt['val'].get('metrics') is not None:
            for name, opt_ in self.opt['val']['metrics'].items():
                try:
                    metric_value = calculate_metric(metric_data, opt_)
                    # 存储以供后续聚合
                    if dataset_name not in self.current_val_metrics:
                        self.current_val_metrics[dataset_name] = {}
                    if name not in self.current_val_metrics[dataset_name]:
                        self.current_val_metrics[dataset_name][name] = []
                    self.current_val_metrics[dataset_name][name].append(metric_value)
                except Exception as e:
                    rank_zero_warn(f"Error calculating metric '{name}': {str(e)}") 

        return {
            'output_clean': output_clean,
            'output_reflection': output_reflection,
            'img_name': img_name,
            'dataset_name': dataset_name
        }

    def on_validation_epoch_start(self):
        """Setup metrics collection at the start of validation epoch."""
        self.current_val_metrics = {}
        
        # 获取验证数据集名称，用于后续记录和显示
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'val_dataloaders'):
            if isinstance(self.trainer.val_dataloaders, list):
                for idx, loader in enumerate(self.trainer.val_dataloaders):
                    if hasattr(loader.dataset, 'opt') and 'name' in loader.dataset.opt:
                        dataset_name = loader.dataset.opt['name']
                    else:
                        dataset_name = f"val_{idx}"
                    self.val_dataset_names[idx] = dataset_name
            else:
                # 单个验证集情况
                loader = self.trainer.val_dataloaders
                if hasattr(loader.dataset, 'opt') and 'name' in loader.dataset.opt:
                    dataset_name = loader.dataset.opt['name']
                else:
                    dataset_name = "val"
                self.val_dataset_names[0] = dataset_name

    def on_validation_epoch_end(self):
        """Operations at the end of validation epoch."""
        # Calculate and log average metrics across all validation samples
        if self.current_val_metrics:
            total_average_metrics = {}
            for dataset_name, metrics in self.current_val_metrics.items():
                log_str = f'\n Validation [{dataset_name}] Epoch {self.current_epoch}\n'

                for metric_name, values in metrics.items():
                    
                    avg_value = sum(values) / len(values)
                    log_str += f'\t # {metric_name}: {avg_value:.4f}'

                    # Log to tensorboard 
                    self.logger.experiment.add_scalar(
                        f'metrics/{dataset_name}/{metric_name}', avg_value, self.current_epoch
                    )
                    if metric_name not in total_average_metrics.keys():
                        total_average_metrics[metric_name] = {
                            'val': sum(values),
                            'counts' : len(values)
                        }
                    else:
                        total_average_metrics[metric_name]['val'] += sum(values)
                        total_average_metrics[metric_name]['counts'] += len(values)
                # Log to console
                rank_zero_info(log_str)
            total_average_metrics = {
                k: v['val'] / v['counts'] for k, v in total_average_metrics.items()
            }
            for metric_name, metric_value in total_average_metrics.items():
                self.logger.experiment.add_scalar(
                    f'metrics/average/{metric_name}', metric_value, self.current_epoch
                )
            
            log_str = f'\n Validation Epoch {self.current_epoch} Average Metrics:\n'
            for metric_name, metric_value in total_average_metrics.items():
                log_str += f'\t # {metric_name}: {metric_value:.4f}'
                self.log(f'metrics/average/{metric_name}', metric_value, sync_dist=True)
            
            rank_zero_info(log_str)
            
            self.top_psnr_epochs.append((total_average_metrics['psnr'], self.current_epoch))
            self.top_psnr_epochs.sort(key=lambda x: (x[0], x[1]), reverse=True)
            self.top_psnr_epochs = self.top_psnr_epochs[:self.opt['val'].get('save_img_top_n', 5)]
            rank_zero_info(f'\t # The Best Average PSNR: {self.top_psnr_epochs[0][0]:.4f} at Epoch {self.top_psnr_epochs[0][1]}')
            
            self.top_ssim_epochs.append((total_average_metrics['ssim'], self.current_epoch))
            self.top_ssim_epochs.sort(key=lambda x: (x[0], x[1]), reverse=True)
            self.top_ssim_epochs = self.top_ssim_epochs[:1]
            rank_zero_info(f'\t # The Best Average SSIM: {self.top_ssim_epochs[0][0]:.4f} at Epoch {self.top_ssim_epochs[0][1]}\n')
            
            self._delete_images_not_in_top_psnr()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step.
        
        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
            
        Returns:
            dict: Output dict with clean and reflection images.
        """
        return self.validation_step(batch, batch_idx, dataloader_idx)
    
    def on_test_epoch_start(self):
        """Operations at the start of test epoch."""
        """Setup metrics collection at the start of test epoch."""
        self.current_val_metrics = {}
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'test_dataloaders'):
            if isinstance(self.trainer.test_dataloaders, list):
                for idx, loader in enumerate(self.trainer.test_dataloaders):
                    if hasattr(loader.dataset, 'opt') and 'name' in loader.dataset.opt:
                        dataset_name = loader.dataset.opt['name']
                    else:
                        dataset_name = f"test_{idx}"
                    self.val_dataset_names[idx] = dataset_name
            else:
                # 单个验证集情况
                loader = self.trainer.test_dataloaders
                if hasattr(loader.dataset, 'opt') and 'name' in loader.dataset.opt:
                    dataset_name = loader.dataset.opt['name']
                else:
                    dataset_name = "test"
                self.val_dataset_names[0] = dataset_name
    
    def on_test_epoch_end(self):
        """Operations at the end of test epoch."""
        return self.on_validation_epoch_end()

    def configure_optimizer_params(self):
        """Configure optimizer parameters.
        
        Returns:
            list: List of parameter groups.
        """
        pass

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            dict: Optimizer and scheduler configuration.
        """
        train_opt = self.opt['train']
        optimizer = self.get_optimizer(**self.configure_optimizer_params())

        # Setup learning rate scheduler without modifying original config
        scheduler_type = train_opt['scheduler']['type']
        scheduler_config = {k: v for k, v in train_opt['scheduler'].items()
                            if k != 'type'}

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
        elif scheduler_type == 'CosineAnnealingRestartLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

        # Get the monitor metric from checkpoint config if available
        monitor_metric = self.opt.get('checkpoint', {}).get('monitor', 'val/psnr')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor_metric,  # Use the same metric as checkpoint monitor
                "interval": "epoch",
                "frequency": 1
            }
        }

    def get_optimizer(self, optim_type, params, **kwargs):
        """Get optimizer based on type.
        
        Args:
            optim_type (str): Optimizer type.
            params (list): Parameter groups.
            **kwargs: Additional optimizer arguments.
            
        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer
    
    def _save_images(self, clean_img, reflection_img, img_name, dataset_name):
        try:
            save_dir = osp.join(self.opt['path']['visualization'], dataset_name, img_name)
            os.makedirs(save_dir, exist_ok=True)
            if self.opt['val'].get('suffix'):
                save_clean_img_path = osp.join(save_dir, f'{img_name}_clean_{self.opt["val"]["suffix"]}_epoch_{self.current_epoch}.png')
                save_reflection_img_path = osp.join(save_dir, f'{img_name}_reflection_{self.opt["val"]["suffix"]}_epoch_{self.current_epoch}.png')
            else:
                save_clean_img_path = osp.join(save_dir, f'{img_name}_clean_{self.opt["name"]}_epoch_{self.current_epoch}.png')
                save_reflection_img_path = osp.join(save_dir, f'{img_name}_reflection_{self.opt["name"]}_epoch_{self.current_epoch}.png')
            # 保存图像
            imwrite(clean_img, save_clean_img_path)
            imwrite(reflection_img, save_reflection_img_path)
        except Exception as e:
            rank_zero_warn(f"Error saving validation images: {str(e)}")
    
    def _delete_images_not_in_top_psnr(self):
        top_epochs_to_keep = [e for _, e in self.top_psnr_epochs]
        visualization_root_path = self.opt['path']['visualization']

        if not osp.isdir(visualization_root_path):
            rank_zero_warn(f"Visualization directory not found: {visualization_root_path}")
            return

        for root, _, files in os.walk(visualization_root_path):
            for file_name in files:
                # Optional: filter for common image file extensions
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    continue

                file_path = osp.join(root, file_name)
                if not osp.isfile(file_path): # Ensure it's a file before attempting to process/delete
                    continue
                
                epoch_match = re.search(r'epoch_(\d+)', file_name)
                if epoch_match:
                    img_epoch = int(epoch_match.group(1))

                    # Protect images from the current epoch and those explicitly in top_epochs_to_keep
                    if img_epoch == self.current_epoch or img_epoch in top_epochs_to_keep:
                        continue
                    
                    # If the epoch is not the current one and not in top_epochs_to_keep, delete it
                    try:
                        os.remove(file_path)
                        # rank_zero_info(f"Deleted old image not in top PSNR epochs: {file_path}") # Optional: for verbose logging
                    except Exception as e:
                        rank_zero_warn(f"Error deleting image {file_path}: {str(e)}")
